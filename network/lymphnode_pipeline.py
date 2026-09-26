from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO


# ============================================================
# Project imports
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.bus_model import ConcatFusionModel, PairedUltrasoundTransform
from network.image_geometry import crop_effective_roi, detect_depth_scale
from network.classifiers import ModalityClassifier, VascularityClassifier


# ============================================================
# Frozen model/config paths
# ============================================================

SEG_CHECKPOINT = ROOT / "deploy_models" / "yolo11s_seg_best.pt"
PHENOTYPE_CHECKPOINT = ROOT / "deploy_models" / "bus_phenotype_best.pt"

SEG_IMGSZ = 640
SEG_CONF = 0.30
SEG_NMS_IOU = 0.70

SYM15_MARGIN = 0.15

EXPECTED_PHENOTYPE_EPOCH = 27
EXPECTED_PHENOTYPE_MACRO = 0.8138359800807995
MACRO_TOLERANCE = 1e-6


LABELS = {
    "posterior": [
        "non_enhancement",
        "enhancement",
    ],
    "liquefaction": [
        "absent",
        "present",
    ],
    "hilum": [
        "disappearance",
        "destruction",
    ],
    "margin": [
        "blurred",
        "partial",
        "clear",
    ],
    "shape": [
        "ellipse",
        "round",
        "fusion",
    ],
}


# ============================================================
# Measurement
# ============================================================

def measure_binary_mask(
    mask: np.ndarray,
    mm_per_pixel: float | None = None,
) -> dict[str, float | None]:
    """
    Compute Maximum Feret Diameter and its perpendicular mask span.
    """

    mask = np.asarray(
        mask,
        dtype=np.uint8,
    )

    if mask.ndim != 2:
        raise ValueError(
            f"Expected 2D mask, got {mask.shape}"
        )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    if not contours:
        raise ValueError(
            "No contour found in predicted mask."
        )

    contour = max(
        contours,
        key=cv2.contourArea,
    )

    area = float(
        cv2.contourArea(contour)
    )

    if area <= 0:
        raise ValueError(
            "Predicted mask contour area <= 0."
        )

    height, width = mask.shape

    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float64)
    deltas = hull[:, None, :] - hull[None, :, :]
    distances = np.sqrt(np.sum(deltas * deltas, axis=2))
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    p1 = hull[first]
    p2 = hull[second]
    direction = p2 - p1
    long_axis = float(distances[first, second])
    if long_axis <= 0:
        raise ValueError("Predicted mask Feret diameter <= 0.")

    perpendicular = np.array([-direction[1], direction[0]]) / long_axis
    projections = hull @ perpendicular
    short_axis = float(projections.max() - projections.min())
    if short_axis <= 0:
        raise ValueError("Predicted mask perpendicular short axis <= 0.")

    short_p1 = hull[int(np.argmin(projections))]
    short_p2 = hull[int(np.argmax(projections))]

    ls_ratio = float(
        long_axis / short_axis
    )

    image_diagonal = float(
        math.sqrt(
            width ** 2
            + height ** 2
        )
    )

    if image_diagonal <= 0:
        raise ValueError(
            "Invalid image diagonal."
        )

    long_mm = long_axis * mm_per_pixel if mm_per_pixel is not None else None
    short_mm = short_axis * mm_per_pixel if mm_per_pixel is not None else None

    return {
        "long_diameter_px": long_axis,
        "short_diameter_px": short_axis,
        "long_diameter_mm": long_mm,
        "short_diameter_mm": short_mm,
        "long_axis_px": long_axis,
        "short_axis_px": short_axis,

        "ls_ratio":
            ls_ratio,

        "long_axis_norm":
            float(
                long_axis
                / image_diagonal
            ),

        "short_axis_norm":
            float(
                short_axis
                / image_diagonal
            ),

        "feret_p1": [float(p1[0]), float(p1[1])],
        "feret_p2": [float(p2[0]), float(p2[1])],
        "short_feret_p1": [float(short_p1[0]), float(short_p1[1])],
        "short_feret_p2": [float(short_p2[0]), float(short_p2[1])],
    }


# ============================================================
# Sym15 construction
# ============================================================

def bbox_from_binary_mask(
    mask: np.ndarray,
) -> tuple[int, int, int, int]:

    ys, xs = np.where(
        mask > 0
    )

    if len(xs) == 0:
        raise ValueError(
            "Empty predicted mask."
        )

    x = int(xs.min())
    y = int(ys.min())

    w = int(
        xs.max()
        - xs.min()
        + 1
    )

    h = int(
        ys.max()
        - ys.min()
        + 1
    )

    return x, y, w, h


def get_sym15_crop_box(
    mask: np.ndarray,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:

    x, y, w, h = (
        bbox_from_binary_mask(
            mask
        )
    )

    left_extra = round(
        w * SYM15_MARGIN
    )

    right_extra = round(
        w * SYM15_MARGIN
    )

    top_extra = round(
        h * SYM15_MARGIN
    )

    bottom_extra = round(
        h * SYM15_MARGIN
    )

    x1 = max(
        0,
        x - left_extra,
    )

    y1 = max(
        0,
        y - top_extra,
    )

    x2 = min(
        image_width,
        x + w + right_extra,
    )

    y2 = min(
        image_height,
        y + h + bottom_extra,
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        raise ValueError(
            "Invalid Sym15 crop box."
        )

    return (
        int(x1),
        int(y1),
        int(x2),
        int(y2),
    )


def jpeg_roundtrip(
    image_bgr: np.ndarray,
) -> np.ndarray:
    """
    Reproduce:
        cv2.imwrite(...jpg)
        -> cv2.imread(...)

    without creating temporary files.
    """

    ok, encoded = cv2.imencode(
        ".jpg",
        image_bgr,
    )

    if not ok:
        raise RuntimeError(
            "Failed encoding Context JPG."
        )

    decoded = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )

    if decoded is None:
        raise RuntimeError(
            "Failed decoding Context JPG."
        )

    return decoded


def build_sym15_inputs(
    image_bgr: np.ndarray,
    full_mask: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, int, int, int],
]:

    height, width = (
        image_bgr.shape[:2]
    )

    x1, y1, x2, y2 = (
        get_sym15_crop_box(
            full_mask,
            image_width=width,
            image_height=height,
        )
    )

    context_crop = (
        image_bgr[
            y1:y2,
            x1:x2,
        ]
        .copy()
    )

    if context_crop.size == 0:
        raise RuntimeError(
            "Empty Context crop."
        )

    # Match training-time JPG behaviour.
    context_jpg = jpeg_roundtrip(
        context_crop
    )

    cropped_mask = (
        full_mask[
            y1:y2,
            x1:x2,
        ]
        .copy()
        .astype(np.uint8)
    )

    if (
        context_jpg.shape[:2]
        != cropped_mask.shape[:2]
    ):
        raise RuntimeError(
            "Context/mask shape mismatch."
        )

    lesion_bgr = (
        context_jpg
        * cropped_mask[:, :, None]
    ).astype(np.uint8)

    return (
        context_jpg,
        lesion_bgr,
        (x1, y1, x2, y2),
    )


# ============================================================
# Phenotype post-processing
# ============================================================

def sigmoid(
    x: float,
) -> float:

    return float(
        1.0
        / (
            1.0
            + np.exp(-x)
        )
    )


def stable_softmax(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = (
        values
        - np.max(values)
    )

    exp_values = np.exp(
        values
    )

    return (
        exp_values
        / exp_values.sum()
    )


# ============================================================
# Main pipeline
# ============================================================

class LymphNodePipeline:

    def __init__(
        self,
        device: str | None = None,
    ):
        """
        Unified inference pipeline:

        Original ultrasound
            -> YOLO11s-seg
            -> predicted lymph-node instances
            -> L/S measurements
            -> Sym15 Context + Masked Lesion
            -> Concat BUS phenotype model
        """

        if device is None:
            self.torch_device = torch.device(
                "cuda:0"
                if torch.cuda.is_available()
                else "cpu"
            )
        else:
            self.torch_device = torch.device(
                device
            )

        self.yolo_device = (
            0
            if self.torch_device.type == "cuda"
            else "cpu"
        )

        self.transform = (
            PairedUltrasoundTransform(
                output_size=224,
                train=False,
            )
        )

        self.seg_model = (
            self._load_segmentation_model()
        )

        self.phenotype_model = (
            self._load_phenotype_model()
        )

        self.modality_model = ModalityClassifier(self.torch_device)
        self.vascularity_model = VascularityClassifier(self.torch_device)


    def _load_segmentation_model(
        self,
    ) -> YOLO:

        if not SEG_CHECKPOINT.exists():
            raise FileNotFoundError(
                SEG_CHECKPOINT
            )

        model = YOLO(
            str(SEG_CHECKPOINT)
        )

        return model


    def _load_phenotype_model(
        self,
    ) -> ConcatFusionModel:

        if not PHENOTYPE_CHECKPOINT.exists():
            raise FileNotFoundError(
                PHENOTYPE_CHECKPOINT
            )

        ckpt = torch.load(
            PHENOTYPE_CHECKPOINT,
            map_location="cpu",
        )

        if "model_state_dict" not in ckpt:
            raise RuntimeError(
                "Phenotype checkpoint missing "
                "model_state_dict."
            )

        epoch = ckpt.get(
            "epoch"
        )

        if (
            epoch
            != EXPECTED_PHENOTYPE_EPOCH
        ):
            raise RuntimeError(
                f"Phenotype epoch mismatch: "
                f"{epoch}"
            )

        macro = float(
            ckpt.get(
                "primary_macro_auroc"
            )
        )

        if abs(
            macro
            - EXPECTED_PHENOTYPE_MACRO
        ) > MACRO_TOLERANCE:
            raise RuntimeError(
                "Phenotype checkpoint macro "
                f"mismatch: {macro}"
            )

        model = ConcatFusionModel(
            pretrained=False,
            dropout=0.2,
        )

        model.load_state_dict(
            ckpt["model_state_dict"],
            strict=True,
        )

        model.to(
            self.torch_device
        )

        model.eval()

        return model


    def _prepare_phenotype_input(
        self,
        context_bgr: np.ndarray,
        lesion_bgr: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        context_rgb = cv2.cvtColor(
            context_bgr,
            cv2.COLOR_BGR2RGB,
        )

        lesion_rgb = cv2.cvtColor(
            lesion_bgr,
            cv2.COLOR_BGR2RGB,
        )

        context_pil = Image.fromarray(
            context_rgb
        )

        lesion_pil = Image.fromarray(
            lesion_rgb
        )

        (
            context_tensor,
            lesion_tensor,
        ) = self.transform(
            context_pil,
            lesion_pil,
        )

        context_tensor = (
            context_tensor
            .unsqueeze(0)
            .to(
                self.torch_device,
                non_blocking=True,
            )
        )

        lesion_tensor = (
            lesion_tensor
            .unsqueeze(0)
            .to(
                self.torch_device,
                non_blocking=True,
            )
        )

        return (
            context_tensor,
            lesion_tensor,
        )


    def _predict_phenotypes(
        self,
        context_bgr: np.ndarray,
        lesion_bgr: np.ndarray,
    ) -> dict[str, Any]:

        (
            context_tensor,
            lesion_tensor,
        ) = (
            self._prepare_phenotype_input(
                context_bgr,
                lesion_bgr,
            )
        )

        with torch.inference_mode():

            logits = (
                self.phenotype_model(
                    context_tensor,
                    lesion_tensor,
                )
            )

        output = {}

        # --------------------------------------------
        # Binary heads
        # --------------------------------------------

        for head in [
            "posterior",
            "liquefaction",
            "hilum",
        ]:

            logit = float(
                logits[head][0, 0]
                .detach()
                .cpu()
                .item()
            )

            p1 = sigmoid(
                logit
            )

            p0 = float(
                1.0 - p1
            )

            pred_class = int(
                p1 >= 0.5
            )

            pred_prob = (
                p1
                if pred_class == 1
                else p0
            )

            output[head] = {
                "class_id":
                    pred_class,

                "label":
                    LABELS[head][
                        pred_class
                    ],

                "probability":
                    float(pred_prob),

                "probabilities": {
                    LABELS[head][0]:
                        float(p0),

                    LABELS[head][1]:
                        float(p1),
                },

                "logit":
                    logit,
            }

        # --------------------------------------------
        # Multiclass heads
        # --------------------------------------------

        for head in [
            "margin",
            "shape",
        ]:

            z = (
                logits[head][0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )

            probs = stable_softmax(
                z
            )

            pred_class = int(
                np.argmax(
                    probs
                )
            )

            output[head] = {
                "class_id":
                    pred_class,

                "label":
                    LABELS[head][
                        pred_class
                    ],

                "probability":
                    float(
                        probs[
                            pred_class
                        ]
                    ),

                "probabilities": {
                    label:
                        float(prob)

                    for label, prob
                    in zip(
                        LABELS[head],
                        probs,
                    )
                },

                "logits": [
                    float(v)
                    for v in z
                ],
            }

        return output


    def predict(
        self,
        image_path: str | Path,
        manual_override: str | None = None,
    ) -> dict[str, Any]:

        image_path = Path(
            image_path
        )

        if not image_path.exists():
            raise FileNotFoundError(
                image_path
            )

        raw_image_bgr = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if raw_image_bgr is None:
            raise RuntimeError(
                f"Cannot read image: "
                f"{image_path}"
            )

        modality_prediction = self.modality_model.predict_bgr(raw_image_bgr)
        ai_mode = modality_prediction["label"]
        final_mode = manual_override or ai_mode
        modality = {
            "ai_mode": ai_mode,
            "p_bmode": modality_prediction["probabilities"]["B-mode"],
            "p_cdfi": modality_prediction["probabilities"]["CDFI"],
            "confidence": modality_prediction["confidence"],
            "manual_override": manual_override,
            "final_mode": final_mode,
        }

        depth_info = detect_depth_scale(raw_image_bgr)
        effective_image_bgr, roi_info = crop_effective_roi(raw_image_bgr)
        image_height, image_width = raw_image_bgr.shape[:2]
        effective_height, effective_width = effective_image_bgr.shape[:2]
        image_info = {
            **depth_info,
            "effective_roi_width_px": roi_info["width"],
            "effective_roi_height_px": roi_info["height"],
            "effective_roi_box": roi_info,
            "source_image_width_px": image_width,
            "source_image_height_px": image_height,
        }

        results = self.seg_model.predict(
            source=str(image_path),
            imgsz=SEG_IMGSZ,
            conf=SEG_CONF,
            iou=SEG_NMS_IOU,
            device=self.yolo_device,
            verbose=False,
            retina_masks=True,
        )

        if not results:
            raise RuntimeError(
                "YOLO returned no result object."
            )

        result = results[0]

        instances = []

        # No detected lymph node.
        if (
            result.masks is None
            or result.boxes is None
            or len(result.boxes) == 0
        ):

            combined_mask = np.zeros(
                (
                    effective_height,
                    effective_width,
                ),
                dtype=np.uint8,
            )

            return {
                "image_path":
                    str(image_path),

                "image_width":
                    image_width,

                "image_height":
                    image_height,

                "image_info":
                    image_info,

                "effective_image":
                    effective_image_bgr,

                "num_instances":
                    0,

                # backward-compatible display mask
                "mask":
                    combined_mask,

                "instances":
                    [],
                "modality": modality,
            }

        mask_data = (
            result.masks.data
            .detach()
            .cpu()
            .numpy()
        )

        confidences = (
            result.boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        boxes = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        raw_instances = []

        for idx in range(
            len(mask_data)
        ):

            mask_raw = (
                mask_data[idx]
            )

            mask_resized = cv2.resize(
                mask_raw.astype(
                    np.float32
                ),
                (
                    image_width,
                    image_height,
                ),
                interpolation=
                    cv2.INTER_NEAREST,
            )

            binary_mask = (
                mask_resized > 0.5
            ).astype(
                np.uint8
            )

            if not np.any(
                binary_mask
            ):
                continue

            raw_instances.append(
                {
                    "mask":
                        binary_mask,

                    "segmentation_confidence":
                        float(
                            confidences[idx]
                        ),

                    "yolo_bbox":
                        [
                            float(v)
                            for v
                            in boxes[idx]
                        ],
                }
            )

        combined_mask = np.zeros(
            (effective_height, effective_width),
            dtype=np.uint8,
        )

        for index, item in enumerate(
            raw_instances,
            start=1,
        ):

            binary_mask = (
                item["mask"]
            )

            effective_mask = binary_mask[
                roi_info["y1"]:roi_info["y2"],
                roi_info["x1"]:roi_info["x2"],
            ].copy()

            if not np.any(effective_mask):
                continue

            combined_mask[
                effective_mask > 0
            ] = 1

            # --------------------------------------------
            # Morphometric measurements
            # --------------------------------------------

            measurements = (
                measure_binary_mask(
                    effective_mask,
                    mm_per_pixel=depth_info["mm_per_pixel"],
                )
            )

            # --------------------------------------------
            # Sym15 Context + Masked Lesion
            # --------------------------------------------

            (
                context_bgr,
                lesion_bgr,
                sym15_box,
            ) = build_sym15_inputs(
                raw_image_bgr,
                binary_mask,
            )

            # --------------------------------------------
            # BUS phenotype
            # --------------------------------------------

            phenotypes = (
                self._predict_phenotypes(
                    context_bgr,
                    lesion_bgr,
                )
            )

            vascularity = (
                self.vascularity_model.predict_bgr(
                    raw_image_bgr[
                        sym15_box[1]:sym15_box[3],
                        sym15_box[0]:sym15_box[2],
                    ]
                    * binary_mask[
                        sym15_box[1]:sym15_box[3],
                        sym15_box[0]:sym15_box[2],
                    ][:, :, None]
                )
                if final_mode == "CDFI"
                else None
            )

            bbox_x, bbox_y, bbox_w, bbox_h = (
                bbox_from_binary_mask(
                    binary_mask
                )
            )

            instance = {
                "instance_id":
                    f"LN_{index:02d}",

                "segmentation": {
                    "confidence":
                        item[
                            "segmentation_confidence"
                        ],

                    "bbox_xyxy":
                        [
                            int(bbox_x),
                            int(bbox_y),
                            int(
                                bbox_x
                                + bbox_w
                            ),
                            int(
                                bbox_y
                                + bbox_h
                            ),
                        ],

                    "yolo_bbox_xyxy":
                        item[
                            "yolo_bbox"
                        ],

                    "sym15_bbox_xyxy":
                        [
                            int(v)
                            for v
                            in sym15_box
                        ],

                    "mask":
                        effective_mask,
                },

                "measurements":
                    measurements,

                "phenotypes":
                    phenotypes,

                "vascularity": (
                    {
                        "available": True,
                        "class_id": vascularity["predicted_class"],
                        "label": vascularity["label"],
                        "probabilities": vascularity["probabilities"],
                    }
                    if vascularity is not None
                    else {
                        "available": False,
                        "reason": "Final modality is B-mode",
                    }
                ),
            }

            instances.append(
                instance
            )

        return {
            "image_path":
                str(image_path),

            "image_width":
                image_width,

            "image_height":
                image_height,

            "image_info":
                image_info,

            "effective_image":
                effective_image_bgr,

            "num_instances":
                len(instances),

            # For the old platform overlay.
            # This is the union of all instances.
            "mask":
                combined_mask,

            # New instance-level result.
            "instances":
                instances,
            "modality": modality,
        }


# ============================================================
# JSON-safe summary
# ============================================================

def make_json_safe(
    result: dict[str, Any],
) -> dict[str, Any]:

    safe = {
        "image_path":
            result["image_path"],

        "image_width":
            result["image_width"],

        "image_height":
            result["image_height"],

        "num_instances":
            result["num_instances"],

        "image_info":
            result.get("image_info", {}),

        "modality":
            result.get("modality", {}),

        "instances":
            [],
    }

    for instance in result[
        "instances"
    ]:

        safe_instance = {
            "instance_id":
                instance[
                    "instance_id"
                ],

            "segmentation": {
                key: value
                for key, value
                in instance[
                    "segmentation"
                ].items()
                if key != "mask"
            },

            "measurements":
                instance[
                    "measurements"
                ],

            "phenotypes":
                instance[
                    "phenotypes"
                ],

            "vascularity":
                instance.get(
                    "vascularity",
                    {},
                ),
        }

        safe[
            "instances"
        ].append(
            safe_instance
        )

    return safe


# ============================================================
# CLI smoke test
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--device",
        default=None,
        type=str,
    )

    args = parser.parse_args()

    pipeline = LymphNodePipeline(
        device=args.device
    )

    result = pipeline.predict(
        args.image
    )

    safe = make_json_safe(
        result
    )

    print(
        json.dumps(
            safe,
            ensure_ascii=False,
            indent=2,
        )
    )

    print()
    print(
        "LYMPHNODE_PIPELINE_COMPLETE = YES"
    )


if __name__ == "__main__":
    main()

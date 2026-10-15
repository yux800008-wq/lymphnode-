import cv2
import numpy as np
from pathlib import Path

from network.lymphnode_pipeline import LymphNodePipeline


class AIService:

    def __init__(self, model_config=None):
        self._lymphnode_pipeline = None
        self.model_config = model_config or {}


    def _get_lymphnode_pipeline(self):
        """
        Lazy-load pipeline.

        First AI analysis:
            load YOLO + BUS Concat models

        Later analyses:
            reuse the same loaded models
        """

        if self._lymphnode_pipeline is None:
            self._lymphnode_pipeline = LymphNodePipeline(
                seg_checkpoint=self.model_config.get("segmentation_model") or None,
                phenotype_checkpoint=self.model_config.get("phenotype_model") or None,
            )

        return self._lymphnode_pipeline


    def analyze_image(self, image_path, manual_override=None, calibration=None):
        """
        Analyze one B-mode ultrasound image.

        Returns both:
        1. backward-compatible combined mask
        2. new instance-level lymph-node results
        """

        image_path = Path(str(image_path))

        if not image_path.exists():
            raise ValueError(
                f"图像不存在：{image_path}"
            )

        pipeline = self._get_lymphnode_pipeline()

        result = pipeline.predict(
            image_path,
            manual_override=manual_override,
            calibration=calibration,
        )

        instances = result.get(
            "instances",
            [],
        )

        num_instances = len(
            instances
        )

        # -----------------------------------------
        # Temporary platform-level summary
        # -----------------------------------------
        #
        # These are NOT diagnosis-model outputs.
        # We only use them to keep the old UI fields
        # from breaking during the integration stage.
        # -----------------------------------------

        if num_instances == 0:
            ai_score = ""
            diagnosis = "未检测到淋巴结"
            size = "0个淋巴结"

        else:
            mean_seg_conf = float(
                np.mean(
                    [
                        item[
                            "segmentation"
                        ][
                            "confidence"
                        ]
                        for item
                        in instances
                    ]
                )
            )

            ai_score = (
                f"{mean_seg_conf:.2f}"
            )

            diagnosis = (
                f"检测到{num_instances}个淋巴结"
            )

            size = (
                f"{num_instances}个淋巴结"
            )

        return {
            # -------------------------------------
            # old UI compatibility
            # -------------------------------------
            "ai_score":
                ai_score,

            "diagnosis":
                diagnosis,

            "size":
                size,

            "mask":
                result.get(
                    "mask"
                ),

            # -------------------------------------
            # new instance-level results
            # -------------------------------------
            "instances":
                instances,

            "num_instances":
                num_instances,

            "modality": result.get("modality", {}),
            "image_info": result.get("image_info", {}),
        }

    def analyze_video(self, video_path, frame_step=5, max_thumbnails=8):
        raise RuntimeError("本地最小演示包仅包含静态超声图像分析功能，未包含旧视频模型。")

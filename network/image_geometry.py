from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def effective_roi_box(image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Return the stable effective ultrasound window for the 4:3 captures."""

    height, width = image_shape[:2]
    if width < 400 or height < 300:
        raise ValueError(f"Unsupported ultrasound image size: {width}x{height}")

    # Current Mindray captures use a stable 768x576 layout.
    x1 = round(width * 138 / 768)
    y1 = round(height * 111 / 576)
    x2 = round(width * 503 / 768)
    y2 = round(height * 494 / 576)
    return x1, y1, x2, y2


def crop_effective_roi(image_bgr: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    x1, y1, x2, y2 = effective_roi_box(image_bgr.shape)
    roi = image_bgr[y1:y2, x1:x2].copy()
    if roi.size == 0:
        raise ValueError("Effective ultrasound ROI is empty.")
    return roi, {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "width": x2 - x1,
        "height": y2 - y1,
    }


def detect_depth_scale(image_bgr: np.ndarray) -> dict[str, Any]:
    """Read the regular 1 cm tick sequence on the right-side depth scale."""

    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    x1 = round(width * 0.74)
    x2 = round(width * 0.84)
    y1 = round(height * 0.15)
    y2 = round(height * 0.92)
    strip = gray[y1:y2, x1:x2]
    score = (strip >= 100).sum(axis=1)
    threshold = max(4, round(strip.shape[1] * 0.06))
    rows = np.flatnonzero(score >= threshold) + y1

    groups: list[list[int]] = []
    for row in rows:
        if not groups or row > groups[-1][-1] + 1:
            groups.append([int(row)])
        else:
            groups[-1].append(int(row))
    centers = [int(round(sum(group) / len(group))) for group in groups]

    best: tuple[int, float, list[int]] | None = None
    for start_index, start in enumerate(centers):
        for next_center in centers[start_index + 1:]:
            step = next_center - start
            if not 0.12 * height <= step <= 0.22 * height:
                continue
            sequence = [start]
            expected = start + step
            while expected <= y2:
                matches = [center for center in centers if abs(center - expected) <= max(5, 0.025 * height)]
                if not matches:
                    break
                sequence.append(min(matches, key=lambda center: abs(center - expected)))
                expected += step
            residual = sum(abs(sequence[i] - (start + i * step)) for i in range(len(sequence)))
            candidate = (len(sequence), -float(residual), sequence)
            if best is None or candidate[:2] > best[:2]:
                best = candidate

    if best is None or len(best[2]) < 3:
        return {
            "detected_depth_cm": None,
            "zero_depth_y": None,
            "max_depth_y": None,
            "depth_pixel_height": None,
            "mm_per_pixel": None,
        }

    sequence = best[2]
    depth_cm = float(len(sequence) - 1)
    pixel_height = int(sequence[-1] - sequence[0])
    if depth_cm <= 0 or pixel_height <= 0:
        return {
            "detected_depth_cm": None,
            "zero_depth_y": None,
            "max_depth_y": None,
            "depth_pixel_height": None,
            "mm_per_pixel": None,
        }

    return {
        "detected_depth_cm": depth_cm,
        "zero_depth_y": int(sequence[0]),
        "max_depth_y": int(sequence[-1]),
        "depth_pixel_height": pixel_height,
        "mm_per_pixel": depth_cm * 10.0 / pixel_height,
    }

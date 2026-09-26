#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Installation verification script for the Windows minimal release.
Checks library versions, checkpoint files, and pipeline importability.
"""

import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent

    print("=" * 50)
    print("  安装验证 - 淋巴结超声AI平台")
    print("=" * 50)
    print()

    # ----------------------------------------------------------
    # 1. Library versions
    # ----------------------------------------------------------
    print("[1/4] 检查依赖库 ...")
    try:
        import torch
        print(f"  torch          {torch.__version__}")
    except ImportError as exc:
        print(f"  [FAIL] 无法导入 torch: {exc}")
        sys.exit(1)

    try:
        import torchvision
        print(f"  torchvision    {torchvision.__version__}")
    except ImportError as exc:
        print(f"  [FAIL] 无法导入 torchvision: {exc}")
        sys.exit(1)

    try:
        import cv2
        print(f"  opencv-python  {cv2.__version__}")
    except ImportError as exc:
        print(f"  [FAIL] 无法导入 cv2: {exc}")
        sys.exit(1)

    try:
        import numpy
        print(f"  numpy          {numpy.__version__}")
    except ImportError as exc:
        print(f"  [FAIL] 无法导入 numpy: {exc}")
        sys.exit(1)

    try:
        from PyQt5.QtCore import PYQT_VERSION_STR
        print(f"  PyQt5          {PYQT_VERSION_STR}")
    except ImportError as exc:
        print(f"  [FAIL] 无法导入 PyQt5: {exc}")
        sys.exit(1)

    try:
        import ultralytics
        print(f"  ultralytics    {ultralytics.__version__}")
    except ImportError as exc:
        print(f"  [FAIL] 无法导入 ultralytics: {exc}")
        sys.exit(1)

    print()

    # ----------------------------------------------------------
    # 2. CUDA status
    # ----------------------------------------------------------
    print("[2/4] 检查 CUDA ...")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")
    if cuda_available:
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  将使用 CPU 模式运行（较慢但可用）")
    print()

    # ----------------------------------------------------------
    # 3. Checkpoint files
    # ----------------------------------------------------------
    print("[3/4] 检查模型文件 ...")
    required_files = [
        "deploy_models/yolo11s_seg_best.pt",
        "deploy_models/bus_phenotype_best.pt",
        "deploy_models/cdfi_resnet18_best.pt",
        "deploy_models/vascularity_resnet18_best.pt",
        "demo_images/demo_multi_3nodes.jpg",
    ]
    for rel in required_files:
        p = root / rel
        if p.exists():
            print(f"  [OK]   {rel}")
        else:
            print(f"  [FAIL] 缺失: {rel}")
            sys.exit(1)
    print()

    # ----------------------------------------------------------
    # 4. Pipeline import
    # ----------------------------------------------------------
    print("[4/4] 检查推理管线导入 ...")
    try:
        from network.lymphnode_pipeline import LymphNodePipeline
        print("  [OK]   LymphNodePipeline 导入成功")
    except Exception as exc:
        print(f"  [FAIL] 导入失败: {exc}")
        sys.exit(1)
    print()

    print("=" * 50)
    print("  INSTALL_VERIFY_PASS")
    print("  安装验证通过！")
    print("=" * 50)


if __name__ == "__main__":
    main()

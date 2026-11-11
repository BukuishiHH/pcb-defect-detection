# -*- coding: utf-8 -*-
"""pytest 公共夹具：构造与 PKU-Market-PCB 结构一致的迷你 COCO 数据集。

测试不依赖真实的 915 MB 数据集，全部在 tmp_path 下用合成图像与标注完成，
以覆盖转换脚本的边界情况（越界框、无效框、crowd/ignore、未知类别、空标注图像）。
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# 故意使用不连续的 category id，用于验证脚本会重映射为 0..N-1
CATEGORIES = [
    {"id": 2, "name": "missing_hole", "supercategory": "none"},
    {"id": 5, "name": "mouse_bite", "supercategory": "none"},
    {"id": 9, "name": "short", "supercategory": "none"},
]


@dataclass
class FakeDataset:
    """合成数据集在磁盘上的路径集合。"""

    raw_dir: Path
    images_dir: Path
    anno_tar: Path
    anno_dir: Path
    output_dir: Path


def make_train_coco() -> dict:
    """train 划分：2 张图，3 个有效框，另有 3 个应被跳过的标注。"""
    return {
        "images": [
            {"id": 1, "file_name": "board_a_01.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "board_a_02.jpg", "width": 200, "height": 100},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [10, 10, 20, 20], "iscrowd": 0, "ignore": 0},
            {"id": 2, "image_id": 1, "category_id": 5, "bbox": [90, 70, 20, 20], "iscrowd": 0, "ignore": 0},
            {"id": 3, "image_id": 2, "category_id": 9, "bbox": [0, 0, 0, 10], "iscrowd": 0, "ignore": 0},
            {"id": 4, "image_id": 2, "category_id": 9, "bbox": [5, 5, 10, 10], "iscrowd": 1, "ignore": 0},
            {"id": 5, "image_id": 2, "category_id": 99, "bbox": [5, 5, 10, 10], "iscrowd": 0, "ignore": 0},
            {"id": 6, "image_id": 2, "category_id": 2, "bbox": [0, 0, 200, 100], "iscrowd": 0, "ignore": 0},
        ],
        "categories": CATEGORIES,
    }


def make_val_coco() -> dict:
    """val 划分：1 张图且无标注，用于验证空标签文件。"""
    return {
        "images": [{"id": 3, "file_name": "board_b_01.jpg", "width": 120, "height": 60}],
        "annotations": [],
        "categories": CATEGORIES,
    }


def write_image(path: Path, width: int, height: int) -> None:
    Image.new("RGB", (width, height), color=(30, 60, 90)).save(path, format="JPEG")


@pytest.fixture
def fake_dataset(tmp_path: Path) -> FakeDataset:
    raw_dir = tmp_path / "PKU-Market-PCB_coco"
    images_dir = raw_dir / "images"
    images_dir.mkdir(parents=True)
    write_image(images_dir / "board_a_01.jpg", 100, 80)
    write_image(images_dir / "board_a_02.jpg", 200, 100)
    write_image(images_dir / "board_b_01.jpg", 120, 60)

    splits = {"train": make_train_coco(), "val": make_val_coco()}
    anno_dir = raw_dir / "coco_json"
    anno_dir.mkdir()
    for split, payload in splits.items():
        (anno_dir / f"{split}.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    anno_tar = raw_dir / "pcb_cocoanno.tar"
    with tarfile.open(anno_tar, "w") as tar:
        for split, payload in splits.items():
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            info = tarfile.TarInfo(f"pcb_cocoanno/{split}.json")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    return FakeDataset(
        raw_dir=raw_dir,
        images_dir=images_dir,
        anno_tar=anno_tar,
        anno_dir=anno_dir,
        output_dir=tmp_path / "datasets" / "pcb_yolo",
    )

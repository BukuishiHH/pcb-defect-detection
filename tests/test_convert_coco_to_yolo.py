# -*- coding: utf-8 -*-
"""scripts/convert_coco_to_yolo.py 的测试。

覆盖范围：
- 坐标换算、类别映射、标注读取（tar / 解压目录）、图像索引等单元逻辑
- 输出目录保护、图像写入、单划分转换
- main() 端到端：目录结构、标签内容、data.yaml、统计文件、幂等保护、dry-run、异常输入
- verify_output() 对损坏标签的检测能力
- 真实数据集可用时的只读集成检查（不存在则自动跳过）
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import convert_coco_to_yolo as c2y

CLASS_NAMES = ["missing_hole", "mouse_bite", "short"]
EXPECTED_COUNTS = {"train": 2, "val": 1}
EXPECTED_SIZES = {
    "train": {"board_a_01.jpg": (100, 80), "board_a_02.jpg": (200, 100)},
    "val": {"board_b_01.jpg": (120, 60)},
}
# board_a_01.jpg 上两个框的期望标签（含越界裁剪）
EXPECTED_TRAIN_LABELS = {
    "board_a_01.txt": (
        "0 0.200000 0.250000 0.200000 0.250000\n"
        "1 0.950000 0.937500 0.100000 0.125000\n"
    ),
    "board_a_02.txt": "0 0.500000 0.500000 1.000000 1.000000\n",
}

REAL_RAW_DIR = Path(__file__).resolve().parents[1] / "PKU-Market-PCB" / "raw" / "PKU-Market-PCB_coco"


@pytest.fixture
def converted(fake_dataset):
    """执行一次完整转换，返回数据集路径对象。"""
    code = c2y.main(
        [
            "--raw-dir",
            str(fake_dataset.raw_dir),
            "--output-dir",
            str(fake_dataset.output_dir),
        ]
    )
    assert code == 0
    return fake_dataset


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------
def test_bbox_to_yolo_converts_and_clips():
    # 普通框：100x80 图像上的 [10, 10, 20, 20]
    assert c2y.bbox_to_yolo([10, 10, 20, 20], 100, 80) == pytest.approx((0.2, 0.25, 0.2, 0.25))
    # 越界框：[90, 70, 20, 20] 裁剪为 10x10
    assert c2y.bbox_to_yolo([90, 70, 20, 20], 100, 80) == pytest.approx((0.95, 0.9375, 0.1, 0.125))
    # 负坐标 + 超出右下边界
    assert c2y.bbox_to_yolo([-10, -10, 30, 30], 100, 80) == pytest.approx((0.1, 0.125, 0.2, 0.25))
    # 整图框
    assert c2y.bbox_to_yolo([0, 0, 200, 100], 200, 100) == pytest.approx((0.5, 0.5, 1.0, 1.0))


@pytest.mark.parametrize(
    "bbox",
    [
        [0, 0, 0, 10],  # 宽为 0
        [0, 0, 10, -5],  # 高为负
        [150, 150, 10, 10],  # 完全在图像外
        [0, 0, float("nan"), 10],  # NaN
        [0, 0, float("inf"), 10],  # inf
        [0, 0, 10],  # 字段数不足
        "not-a-bbox",  # 类型错误
        None,
    ],
)
def test_bbox_to_yolo_rejects_invalid(bbox):
    assert c2y.bbox_to_yolo(bbox, 100, 80) is None


def test_format_label_line_uses_six_decimals():
    line = c2y.format_label_line(2, (0.1, 0.2, 0.3, 1.0))
    assert line == "2 0.100000 0.200000 0.300000 1.000000"


def test_build_class_map_orders_by_category_id_and_remaps():
    categories = [
        {"id": 9, "name": "short"},
        {"id": 2, "name": "missing_hole"},
        {"id": 5, "name": "mouse_bite"},
    ]
    names, mapping = c2y.build_class_map(categories, "train")
    assert names == ["missing_hole", "mouse_bite", "short"]
    assert mapping == {2: 0, 5: 1, 9: 2}


def test_build_class_map_rejects_bad_categories():
    with pytest.raises(c2y.ConversionError):
        c2y.build_class_map([], "train")
    with pytest.raises(c2y.ConversionError):
        c2y.build_class_map([{"id": 1, "name": "a"}, {"id": 2, "name": "a"}], "train")
    with pytest.raises(c2y.ConversionError):
        c2y.build_class_map([{"id": 1, "name": "a"}, {"id": 1, "name": "b"}], "train")
    with pytest.raises(c2y.ConversionError):
        c2y.build_class_map([{"id": 1, "name": " "}], "train")


# ---------------------------------------------------------------------------
# 标注读取与图像索引
# ---------------------------------------------------------------------------
def test_load_coco_splits_from_tar_and_directory(fake_dataset):
    from_tar = c2y.load_coco_splits(fake_dataset.anno_tar, None, ("train", "val"))
    from_dir = c2y.load_coco_splits(None, fake_dataset.anno_dir, ("train", "val"))
    assert from_tar == from_dir
    assert len(from_tar["train"]["images"]) == 2
    assert len(from_tar["val"]["images"]) == 1


def test_load_coco_splits_reports_missing_sources(fake_dataset):
    with pytest.raises(c2y.ConversionError, match="缺少"):
        c2y.load_coco_splits(fake_dataset.anno_tar, None, ("train", "test"))
    with pytest.raises(c2y.ConversionError, match="找不到 COCO 标注包"):
        c2y.load_coco_splits(fake_dataset.raw_dir / "missing.tar", None, ("train",))
    with pytest.raises(c2y.ConversionError, match="找不到标注文件"):
        c2y.load_coco_splits(None, fake_dataset.anno_dir, ("test",))


def test_index_images_is_case_insensitive_and_filters_suffixes(tmp_path):
    (tmp_path / "A.JPG").write_bytes(b"x")
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("skip", encoding="utf-8")

    index = c2y.index_images(tmp_path)
    assert set(index) == {"a.jpg", "b.png"}

    with pytest.raises(c2y.ConversionError, match="图像目录不存在"):
        c2y.index_images(tmp_path / "missing")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(c2y.ConversionError, match="没有可用图像"):
        c2y.index_images(empty)


# ---------------------------------------------------------------------------
# 文件写入辅助函数
# ---------------------------------------------------------------------------
def test_transfer_image_copy_and_replace(tmp_path):
    src = tmp_path / "src.jpg"
    dst = tmp_path / "dst.jpg"
    src.write_bytes(b"payload")
    dst.write_bytes(b"old")

    assert c2y.transfer_image(src, dst, "copy") is False
    assert dst.read_bytes() == b"payload"


def test_transfer_image_hardlink_or_fallback(tmp_path):
    src = tmp_path / "src.jpg"
    dst = tmp_path / "dst.jpg"
    src.write_bytes(b"payload")

    fallback = c2y.transfer_image(src, dst, "hardlink")
    assert dst.read_bytes() == b"payload"
    if fallback:
        # 文件系统不支持硬链接时，脚本会自动回退为复制
        assert not dst.samefile(src)
    else:
        assert dst.samefile(src)


def test_prepare_output_dir_requires_overwrite(tmp_path):
    output_dir = tmp_path / "out"
    (output_dir / "images" / "train").mkdir(parents=True)
    (output_dir / "images" / "train" / "old.jpg").write_bytes(b"x")

    with pytest.raises(c2y.ConversionError, match="--overwrite"):
        c2y.prepare_output_dir(output_dir, ("train", "val"), overwrite=False)

    images_root, labels_root = c2y.prepare_output_dir(output_dir, ("train", "val"), overwrite=True)
    assert not (images_root / "train" / "old.jpg").exists()
    assert (labels_root / "val").is_dir()


# ---------------------------------------------------------------------------
# 单个划分的转换
# ---------------------------------------------------------------------------
def test_convert_split_writes_labels_and_reports_skips(fake_dataset):
    coco = c2y.load_coco_splits(fake_dataset.anno_tar, None, ("train", "val"))
    class_names, class_map = c2y.build_class_map(coco["train"]["categories"], "train")
    images_index = c2y.index_images(fake_dataset.images_dir)
    images_root, labels_root = c2y.prepare_output_dir(fake_dataset.output_dir, ("train",), False)

    stats, sizes = c2y.convert_split(
        "train",
        coco["train"],
        images_index,
        class_names,
        class_map,
        images_root,
        labels_root,
        mode="copy",
        dry_run=False,
    )

    assert stats.images == 2
    assert stats.labels == 2
    assert stats.boxes == 3
    assert stats.empty_labels == 0
    assert stats.skipped == {"invalid_bbox": 1, "crowd_or_ignore": 1, "unknown_category": 1}
    assert stats.per_class == {"missing_hole": 2, "mouse_bite": 1}
    assert sizes == {"board_a_01.jpg": (100, 80), "board_a_02.jpg": (200, 100)}

    for name, expected in EXPECTED_TRAIN_LABELS.items():
        assert (labels_root / "train" / name).read_text(encoding="utf-8") == expected
    assert (images_root / "train" / "board_a_01.jpg").is_file()


def test_convert_split_writes_empty_label_for_unannotated_image(fake_dataset):
    coco = c2y.load_coco_splits(fake_dataset.anno_tar, None, ("train", "val"))
    class_names, class_map = c2y.build_class_map(coco["train"]["categories"], "train")
    images_index = c2y.index_images(fake_dataset.images_dir)
    images_root, labels_root = c2y.prepare_output_dir(fake_dataset.output_dir, ("val",), False)

    stats, sizes = c2y.convert_split(
        "val",
        coco["val"],
        images_index,
        class_names,
        class_map,
        images_root,
        labels_root,
        mode="copy",
        dry_run=False,
    )

    assert stats.boxes == 0
    assert stats.empty_labels == 1
    assert stats.labels == 1
    assert sizes == {"board_b_01.jpg": (120, 60)}
    assert (labels_root / "val" / "board_b_01.txt").read_text(encoding="utf-8") == ""


def test_convert_split_dry_run_writes_nothing(fake_dataset):
    coco = c2y.load_coco_splits(fake_dataset.anno_tar, None, ("train", "val"))
    class_names, class_map = c2y.build_class_map(coco["train"]["categories"], "train")
    images_index = c2y.index_images(fake_dataset.images_dir)
    images_root = fake_dataset.output_dir / "images"
    labels_root = fake_dataset.output_dir / "labels"

    stats, _ = c2y.convert_split(
        "train",
        coco["train"],
        images_index,
        class_names,
        class_map,
        images_root,
        labels_root,
        mode="copy",
        dry_run=True,
    )

    assert stats.boxes == 3
    assert not fake_dataset.output_dir.exists()


# ---------------------------------------------------------------------------
# 端到端：main()
# ---------------------------------------------------------------------------
def test_main_end_to_end(converted):
    output_dir = converted.output_dir

    assert len(list((output_dir / "images" / "train").glob("*.jpg"))) == 2
    assert len(list((output_dir / "images" / "val").glob("*.jpg"))) == 1
    for name, expected in EXPECTED_TRAIN_LABELS.items():
        assert (output_dir / "labels" / "train" / name).read_text(encoding="utf-8") == expected
    assert (output_dir / "labels" / "val" / "board_b_01.txt").read_text(encoding="utf-8") == ""

    cfg = yaml.safe_load((output_dir / "data.yaml").read_text(encoding="utf-8"))
    assert cfg["train"] == "images/train"
    assert cfg["val"] == "images/val"
    assert cfg["nc"] == 3
    assert cfg["names"] == {0: "missing_hole", 1: "mouse_bite", 2: "short"}
    assert "path" not in cfg

    summary = json.loads((output_dir / "conversion_summary.json").read_text(encoding="utf-8"))
    assert summary["classes"] == CLASS_NAMES
    assert summary["total"] == {"images": 3, "boxes": 3}
    assert summary["splits"]["train"]["skipped_annotations"] == {
        "invalid_bbox": 1,
        "crowd_or_ignore": 1,
        "unknown_category": 1,
    }
    assert summary["splits"]["val"]["empty_labels"] == 1


def test_main_refuses_to_overwrite_without_flag(fake_dataset, capsys):
    args = [
        "--raw-dir",
        str(fake_dataset.raw_dir),
        "--output-dir",
        str(fake_dataset.output_dir),
    ]
    assert c2y.main(args) == 0

    assert c2y.main(args) == 1
    assert "--overwrite" in capsys.readouterr().err

    # 追加 --overwrite 后可重新生成，且不会残留旧标签
    stale = fake_dataset.output_dir / "labels" / "train" / "stale.txt"
    stale.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    assert c2y.main(args + ["--overwrite"]) == 0
    assert not stale.exists()


def test_main_dry_run_creates_no_files(fake_dataset, capsys):
    code = c2y.main(
        [
            "--raw-dir",
            str(fake_dataset.raw_dir),
            "--output-dir",
            str(fake_dataset.output_dir),
            "--dry-run",
        ]
    )
    assert code == 0
    assert not fake_dataset.output_dir.exists()
    assert "[dry-run]" in capsys.readouterr().out


def test_main_reports_missing_image(fake_dataset, capsys):
    (fake_dataset.images_dir / "board_a_02.jpg").unlink()

    code = c2y.main(
        [
            "--raw-dir",
            str(fake_dataset.raw_dir),
            "--output-dir",
            str(fake_dataset.output_dir),
        ]
    )

    assert code == 1
    assert "缺少图像文件" in capsys.readouterr().err


def test_main_reads_extracted_annotation_directory(fake_dataset):
    code = c2y.main(
        [
            "--raw-dir",
            str(fake_dataset.raw_dir),
            "--output-dir",
            str(fake_dataset.output_dir),
            "--anno-dir",
            str(fake_dataset.anno_dir),
            "--splits",
            "val",
        ]
    )

    assert code == 0
    assert (fake_dataset.output_dir / "images" / "val" / "board_b_01.jpg").is_file()
    assert (
        fake_dataset.output_dir / "labels" / "val" / "board_b_01.txt"
    ).read_text(encoding="utf-8") == ""

    cfg = yaml.safe_load((fake_dataset.output_dir / "data.yaml").read_text(encoding="utf-8"))
    assert cfg["val"] == "images/val"
    assert "train" not in cfg


def test_main_hardlink_mode_produces_readable_images(fake_dataset):
    code = c2y.main(
        [
            "--raw-dir",
            str(fake_dataset.raw_dir),
            "--output-dir",
            str(fake_dataset.output_dir),
            "--mode",
            "hardlink",
        ]
    )

    assert code == 0
    source = fake_dataset.images_dir / "board_a_01.jpg"
    target = fake_dataset.output_dir / "images" / "train" / "board_a_01.jpg"
    assert target.read_bytes() == source.read_bytes()


# ---------------------------------------------------------------------------
# 结果校验
# ---------------------------------------------------------------------------
def test_verify_output_accepts_generated_dataset(converted):
    errors = c2y.verify_output(
        converted.output_dir,
        ("train", "val"),
        CLASS_NAMES,
        EXPECTED_COUNTS,
        EXPECTED_SIZES,
    )
    assert errors == []


def test_verify_output_detects_corrupted_labels(converted):
    label = converted.output_dir / "labels" / "train" / "board_a_01.txt"
    label.write_text(
        "0 0.2 0.2 0.2\n"  # 4 字段
        "9 0.5 0.5 0.1 1.5\n"  # 类别越界 + 坐标越界
        "0 0.5 0.5 0.0 0.2\n",  # 宽为 0
        encoding="utf-8",
    )

    errors = c2y.verify_output(
        converted.output_dir,
        ("train", "val"),
        CLASS_NAMES,
        EXPECTED_COUNTS,
        EXPECTED_SIZES,
    )
    text = "\n".join(errors)
    assert "字段数" in text
    assert "类别索引越界" in text
    assert "坐标超出" in text
    assert "宽高非正" in text


def test_verify_output_detects_missing_label(converted):
    (converted.output_dir / "labels" / "val" / "board_b_01.txt").unlink()

    errors = c2y.verify_output(
        converted.output_dir,
        ("train", "val"),
        CLASS_NAMES,
        EXPECTED_COUNTS,
        EXPECTED_SIZES,
    )
    text = "\n".join(errors)
    assert "标签数量" in text
    assert "图像缺少标签" in text


# ---------------------------------------------------------------------------
# 真实数据集（存在时执行只读检查，不存在则跳过）
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (REAL_RAW_DIR / "pcb_cocoanno.tar").is_file(),
    reason="真实 PKU-Market-PCB 数据集不存在，跳过集成测试",
)
def test_real_dataset_dry_run_is_consistent(tmp_path):
    coco = c2y.load_coco_splits(REAL_RAW_DIR / "pcb_cocoanno.tar", None, ("train", "val"))
    class_names, _ = c2y.build_class_map(coco["train"]["categories"], "train")

    assert class_names == [
        "missing_hole",
        "mouse_bite",
        "open_circuit",
        "short",
        "spur",
        "spurious_copper",
    ]
    assert len(coco["train"]["images"]) == 555
    assert len(coco["train"]["annotations"]) == 2363
    assert len(coco["val"]["images"]) == 138
    assert len(coco["val"]["annotations"]) == 590

    code = c2y.main(
        [
            "--raw-dir",
            str(REAL_RAW_DIR),
            "--output-dir",
            str(tmp_path / "dry_run_output"),
            "--dry-run",
        ]
    )
    assert code == 0
    assert not (tmp_path / "dry_run_output").exists()

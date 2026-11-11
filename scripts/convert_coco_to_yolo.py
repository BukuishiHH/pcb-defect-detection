#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""将 PKU-Market-PCB 的 COCO 标注转换为 Ultralytics YOLOv8 检测格式。

原始数据（不纳入 git 跟踪）::

    PKU-Market-PCB/raw/PKU-Market-PCB_coco/
    ├── images/*.jpg                    693 张 PCB 图像
    └── pcb_cocoanno.tar                pcb_cocoanno/train.json、val.json（COCO 格式）

转换结果（不纳入 git 跟踪）::

    datasets/pcb_yolo/
    ├── images/train/*.jpg
    ├── images/val/*.jpg
    ├── labels/train/*.txt              YOLO 标签：cls cx cy w h（归一化 0~1）
    ├── labels/val/*.txt
    ├── data.yaml                       Ultralytics 训练配置
    └── conversion_summary.json         转换统计，便于复现

用法示例::

    conda activate cv_demo
    python scripts/convert_coco_to_yolo.py --overwrite
    # 与原始数据在同一磁盘时可用硬链接，避免重复占用约 915 MB
    python scripts/convert_coco_to_yolo.py --overwrite --mode hardlink
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = REPO_ROOT / "PKU-Market-PCB" / "raw" / "PKU-Market-PCB_coco"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "datasets" / "pcb_yolo"
DEFAULT_SPLITS = ("train", "val")
ANNO_MEMBER_TEMPLATE = "pcb_cocoanno/{split}.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
_LINK_WARNED: set = set()
YAML_HEADER = (
    "# PCB 板缺陷检测数据集配置（由 scripts/convert_coco_to_yolo.py 自动生成）\n"
    "# 未设置 path：Ultralytics 以本文件所在目录作为数据集根目录，\n"
    "# 因此可在任意工作目录下使用该配置。\n"
)


class ConversionError(RuntimeError):
    """数据转换过程中的可预期错误（输入缺失、格式异常等）。"""


@dataclass
class SplitStats:
    """单个划分的转换统计。"""

    split: str
    images: int = 0
    labels: int = 0
    boxes: int = 0
    empty_labels: int = 0
    fallbacks: int = 0
    skipped: Counter = field(default_factory=Counter)
    per_class: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict:
        return {
            "images": self.images,
            "labels": self.labels,
            "boxes": self.boxes,
            "empty_labels": self.empty_labels,
            "link_fallbacks": self.fallbacks,
            "skipped_annotations": dict(self.skipped),
            "boxes_per_class": dict(self.per_class),
        }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PKU-Market-PCB：COCO 标注 -> YOLOv8 检测格式",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="原始数据集目录（含 images/ 与标注包）")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="YOLO 数据集输出目录")
    parser.add_argument("--anno-tar", type=Path, default=None, help="COCO 标注 tar 包，默认 <raw-dir>/pcb_cocoanno.tar")
    parser.add_argument("--anno-dir", type=Path, default=None, help="已解压的 COCO json 目录，优先于 --anno-tar")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), help="需要转换的划分")
    parser.add_argument("--mode", choices=("copy", "hardlink", "symlink"), default="copy", help="图像写入方式")
    parser.add_argument("--overwrite", action="store_true", help="清空输出目录中已有的 images/、labels/ 后重新生成")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写任何文件")
    return parser.parse_args(argv)


def load_coco_splits(anno_tar: Optional[Path], anno_dir: Optional[Path], splits: Sequence[str]) -> Dict[str, dict]:
    """读取各划分的 COCO 标注，兼容 tar 包与已解压目录两种形式。"""
    result: Dict[str, dict] = {}
    if anno_dir is not None:
        for split in splits:
            path = anno_dir / f"{split}.json"
            if not path.is_file():
                raise ConversionError(f"找不到标注文件: {path}")
            with path.open("r", encoding="utf-8") as handle:
                result[split] = json.load(handle)
        return result

    if anno_tar is None or not anno_tar.is_file():
        raise ConversionError(f"找不到 COCO 标注包: {anno_tar}")
    try:
        with tarfile.open(anno_tar, "r:*") as tar:
            for split in splits:
                member = ANNO_MEMBER_TEMPLATE.format(split=split)
                handle = tar.extractfile(member)
                if handle is None:
                    raise ConversionError(f"标注包中缺少 {member}")
                result[split] = json.loads(handle.read().decode("utf-8"))
    except tarfile.TarError as exc:
        raise ConversionError(f"读取标注包失败: {anno_tar} ({exc})") from exc
    return result


def build_class_map(categories: Sequence[dict], split: str) -> Tuple[List[str], Dict[int, int]]:
    """按 COCO category id 排序生成连续的 YOLO 类别索引。"""
    if not categories:
        raise ConversionError(f"{split} 划分缺少 categories 字段")
    ordered = sorted(categories, key=lambda item: int(item["id"]))
    class_ids = [int(item["id"]) for item in ordered]
    names = [str(item.get("name", "")).strip() for item in ordered]
    if len(set(class_ids)) != len(class_ids):
        raise ConversionError(f"{split} 划分存在重复的 category id: {class_ids}")
    if any(not name for name in names):
        raise ConversionError(f"{split} 划分存在空类别名: {names}")
    if len(set(names)) != len(names):
        raise ConversionError(f"{split} 划分存在重复类别名: {names}")
    return names, {class_id: index for index, class_id in enumerate(class_ids)}


def index_images(images_dir: Path) -> Dict[str, Path]:
    """建立小写文件名 -> 图像路径的索引，避免大小写差异导致漏配。"""
    if not images_dir.is_dir():
        raise ConversionError(f"图像目录不存在: {images_dir}")
    index: Dict[str, Path] = {}
    for path in sorted(images_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.name.lower(), path)
    if not index:
        raise ConversionError(f"图像目录中没有可用图像: {images_dir}")
    return index


def bbox_to_yolo(
    bbox: Sequence[float], width: float, height: float
) -> Optional[Tuple[float, float, float, float]]:
    """COCO 的 [x_min, y_min, w, h]（像素）转为 YOLO 的 (cx, cy, w, h)（归一化）。

    越界部分会被裁剪到图像范围内；裁剪后面积为零的框视为无效并返回 None。
    """
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x, y, w, h = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, w, h)):
        return None
    if w <= 0 or h <= 0:
        return None

    x1, y1 = max(x, 0.0), max(y, 0.0)
    x2, y2 = min(x + w, width), min(y + h, height)
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 0 or box_h <= 0:
        return None
    return (
        (x1 + x2) / 2.0 / width,
        (y1 + y2) / 2.0 / height,
        box_w / width,
        box_h / height,
    )


def format_label_line(class_index: int, box: Tuple[float, float, float, float]) -> str:
    cx, cy, width, height = box
    return f"{class_index} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"


def transfer_image(src: Path, dst: Path, mode: str) -> bool:
    """按指定方式写入图像；链接失败时回退为复制。返回是否发生回退。"""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return False
    try:
        if mode == "hardlink":
            os.link(src, dst)
        else:
            os.symlink(src, dst)
        return False
    except OSError as exc:
        if mode not in _LINK_WARNED:
            print(f"  [WARN] {mode} 失败，后续图像回退为复制: {dst.name} ({exc})")
            _LINK_WARNED.add(mode)
        shutil.copy2(src, dst)
        return True


def has_files(path: Path) -> bool:
    return path.is_dir() and any(item.is_file() for item in path.rglob("*"))


def prepare_output_dir(output_dir: Path, splits: Sequence[str], overwrite: bool) -> Tuple[Path, Path]:
    images_root = output_dir / "images"
    labels_root = output_dir / "labels"
    if not overwrite and (has_files(images_root) or has_files(labels_root)):
        raise ConversionError(
            f"输出目录中已存在转换结果: {output_dir}\n"
            "如需重新生成，请追加 --overwrite（会清空 images/ 与 labels/）"
        )
    if overwrite:
        for path in (images_root, labels_root):
            if path.exists():
                shutil.rmtree(path)
    for split in splits:
        (images_root / split).mkdir(parents=True, exist_ok=True)
        (labels_root / split).mkdir(parents=True, exist_ok=True)
    return images_root, labels_root


def convert_split(
    split: str,
    coco: dict,
    images_index: Dict[str, Path],
    class_names: Sequence[str],
    class_id_to_index: Dict[int, int],
    images_root: Path,
    labels_root: Path,
    mode: str,
    dry_run: bool,
) -> Tuple[SplitStats, Dict[str, Tuple[int, int]]]:
    """转换单个划分，返回统计信息与 文件名->(宽, 高) 映射（用于校验）。"""
    stats = SplitStats(split=split)
    images = sorted(coco.get("images", []), key=lambda item: str(item.get("file_name", "")))
    if not images:
        raise ConversionError(f"{split} 划分没有图像记录")

    annotations_by_image: Dict[object, List[dict]] = defaultdict(list)
    known_image_ids = {info.get("id") for info in images}
    for ann in coco.get("annotations", []):
        if ann.get("image_id") in known_image_ids:
            annotations_by_image[ann["image_id"]].append(ann)

    sizes: Dict[str, Tuple[int, int]] = {}
    used_names = set()
    stats.images = len(images)

    for order, info in enumerate(images, 1):
        file_name = str(info.get("file_name", "")).strip()
        target_name = Path(file_name).name
        if not target_name:
            raise ConversionError(f"{split} 划分存在空文件名")
        key = target_name.lower()
        if key in used_names:
            raise ConversionError(f"{split} 划分存在重名图像: {target_name}")
        used_names.add(key)

        try:
            width = int(info["width"])
            height = int(info["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConversionError(
                f"图像尺寸无效: {file_name} (width={info.get('width')}, height={info.get('height')})"
            ) from exc
        if width <= 0 or height <= 0:
            raise ConversionError(f"图像尺寸无效: {file_name} (width={width}, height={height})")
        sizes[target_name] = (width, height)

        source = images_index.get(key)
        if source is None:
            raise ConversionError(f"缺少图像文件: {file_name}（在 images/ 中未找到）")

        lines: List[str] = []
        for ann in annotations_by_image.get(info.get("id"), []):
            if ann.get("iscrowd") or ann.get("ignore"):
                stats.skipped["crowd_or_ignore"] += 1
                continue
            class_index = class_id_to_index.get(ann.get("category_id"))
            if class_index is None:
                stats.skipped["unknown_category"] += 1
                continue
            box = bbox_to_yolo(ann.get("bbox"), float(width), float(height))
            if box is None:
                stats.skipped["invalid_bbox"] += 1
                continue
            lines.append(format_label_line(class_index, box))
            stats.per_class[class_names[class_index]] += 1
            stats.boxes += 1

        if not lines:
            stats.empty_labels += 1
        stats.labels += 1

        if dry_run:
            continue

        if transfer_image(source, images_root / split / target_name, mode):
            stats.fallbacks += 1
        label_path = labels_root / split / f"{Path(target_name).stem}.txt"
        label_text = "\n".join(lines)
        if label_text:
            label_text += "\n"
        with label_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(label_text)

        if order % 100 == 0 or order == stats.images:
            print(f"  {split}: {order}/{stats.images}")

    return stats, sizes


def write_data_yaml(output_dir: Path, splits: Sequence[str], class_names: Sequence[str]) -> Path:
    """生成 Ultralytics data.yaml；不写 path，让数据集根目录取本文件所在目录。"""
    lines = [YAML_HEADER.rstrip("\n"), ""]
    for split in splits:
        lines.append(f"{split}: images/{split}")
    lines.extend([f"nc: {len(class_names)}", "names:"])
    lines.extend(f"  {index}: {name}" for index, name in enumerate(class_names))

    path = output_dir / "data.yaml"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")
    return path


def write_summary(
    output_dir: Path,
    raw_dir: Path,
    anno_source: Path,
    class_names: Sequence[str],
    stats_list: Sequence[SplitStats],
) -> Path:
    payload = {
        "source": {"raw_dir": str(raw_dir), "annotations": str(anno_source)},
        "output_dir": str(output_dir),
        "classes": list(class_names),
        "splits": {stats.split: stats.as_dict() for stats in stats_list},
        "total": {
            "images": sum(stats.images for stats in stats_list),
            "boxes": sum(stats.boxes for stats in stats_list),
        },
    }
    path = output_dir / "conversion_summary.json"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def verify_output(
    output_dir: Path,
    splits: Sequence[str],
    class_names: Sequence[str],
    expected_counts: Dict[str, int],
    expected_sizes: Dict[str, Dict[str, Tuple[int, int]]],
) -> List[str]:
    """校验转换结果：文件配对、标签字段、类别范围、坐标归一化、图像尺寸。"""
    images_root = output_dir / "images"
    labels_root = output_dir / "labels"
    errors: List[str] = []
    try:
        from PIL import Image
    except ImportError:
        Image = None  # type: ignore[assignment]
        print("  [WARN] 未安装 Pillow，跳过图像尺寸校验")

    for split in splits:
        split_images = sorted(p for p in (images_root / split).iterdir() if p.is_file())
        split_labels = sorted((labels_root / split).glob("*.txt"))
        if len(split_images) != expected_counts[split]:
            errors.append(f"{split}: 图像数量 {len(split_images)} != 期望 {expected_counts[split]}")
        if len(split_labels) != expected_counts[split]:
            errors.append(f"{split}: 标签数量 {len(split_labels)} != 期望 {expected_counts[split]}")

        label_stems = {label.stem for label in split_labels}
        for image in split_images:
            if image.stat().st_size == 0:
                errors.append(f"{split}: 图像为空文件 {image.name}")
            if image.stem not in label_stems:
                errors.append(f"{split}: 图像缺少标签 {image.name}")
            expected_size = expected_sizes.get(split, {}).get(image.name)
            if Image is not None and expected_size is not None:
                try:
                    with Image.open(image) as im:
                        if im.size != expected_size:
                            errors.append(
                                f"{split}: 图像尺寸 {im.size} != 标注 {expected_size} ({image.name})"
                            )
                except OSError as exc:
                    errors.append(f"{split}: 图像无法读取 {image.name} ({exc})")

        for label in split_labels:
            with label.open("r", encoding="utf-8") as handle:
                for lineno, line in enumerate(handle, 1):
                    parts = line.split()
                    if not parts:
                        continue
                    location = f"{split}/{label.name}:{lineno}"
                    if len(parts) != 5:
                        errors.append(f"{location} 字段数 {len(parts)} != 5")
                        continue
                    try:
                        class_index = int(parts[0])
                        values = [float(value) for value in parts[1:]]
                    except ValueError:
                        errors.append(f"{location} 无法解析为数值")
                        continue
                    if not 0 <= class_index < len(class_names):
                        errors.append(f"{location} 类别索引越界 {class_index}")
                    if not all(math.isfinite(value) and -1e-6 <= value <= 1.0 + 1e-6 for value in values):
                        errors.append(f"{location} 坐标超出 0~1: {values}")
                    elif values[2] <= 0 or values[3] <= 0:
                        errors.append(f"{location} 宽高非正: {values}")

        if len(errors) > 20:
            errors.append("... 错误过多，已截断")
            return errors

    return errors


def print_summary(stats_list: Sequence[SplitStats], class_names: Sequence[str]) -> None:
    print("\n转换统计:")
    print(f"  {'split':<6}{'images':>8}{'labels':>8}{'boxes':>8}{'empty':>8}")
    for stats in stats_list:
        print(
            f"  {stats.split:<6}{stats.images:>8}{stats.labels:>8}{stats.boxes:>8}{stats.empty_labels:>8}"
        )

    print("\n各类别目标框数量:")
    for stats in stats_list:
        detail = ", ".join(f"{name}={stats.per_class.get(name, 0)}" for name in class_names)
        print(f"  {stats.split}: {detail}")

    skipped: Counter = Counter()
    for stats in stats_list:
        skipped.update(stats.skipped)
    if skipped:
        detail = ", ".join(f"{reason}={count}" for reason, count in skipped.items())
        print(f"\n跳过的标注: {detail}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    anno_dir = args.anno_dir.resolve() if args.anno_dir else None
    anno_tar = args.anno_tar.resolve() if args.anno_tar else raw_dir / "pcb_cocoanno.tar"
    splits = list(dict.fromkeys(args.splits))
    images_dir = raw_dir / "images"
    anno_source = anno_dir if anno_dir is not None else anno_tar

    print("=" * 62)
    print("PKU-Market-PCB -> YOLOv8 数据转换")
    print(f"  原始数据: {raw_dir}")
    print(f"  标注来源: {anno_source}")
    print(f"  输出目录: {output_dir}")
    mode_line = f"  划分: {', '.join(splits)} | 写入方式: {args.mode}"
    print(mode_line + (" | dry-run" if args.dry_run else ""))
    print("=" * 62)

    try:
        coco = load_coco_splits(anno_tar, anno_dir, splits)
        class_names, class_id_to_index = build_class_map(coco[splits[0]]["categories"], splits[0])
        for split in splits[1:]:
            other_names, other_map = build_class_map(coco[split]["categories"], split)
            if other_names != class_names or other_map != class_id_to_index:
                raise ConversionError(f"{split} 划分的类别定义与 {splits[0]} 不一致")
        images_index = index_images(images_dir)

        if args.dry_run:
            images_root = output_dir / "images"
            labels_root = output_dir / "labels"
        else:
            images_root, labels_root = prepare_output_dir(output_dir, splits, args.overwrite)

        print(
            f"\n[1/3] 解析标注: {len(splits)} 个划分, {len(images_index)} 张图像, "
            f"{len(class_names)} 个类别 -> {', '.join(class_names)}"
        )
        stats_list: List[SplitStats] = []
        expected_counts: Dict[str, int] = {}
        expected_sizes: Dict[str, Dict[str, Tuple[int, int]]] = {}
        for index, split in enumerate(splits, 1):
            print(f"\n[2/3] 转换 {split} ({index}/{len(splits)})")
            stats, sizes = convert_split(
                split=split,
                coco=coco[split],
                images_index=images_index,
                class_names=class_names,
                class_id_to_index=class_id_to_index,
                images_root=images_root,
                labels_root=labels_root,
                mode=args.mode,
                dry_run=args.dry_run,
            )
            stats_list.append(stats)
            expected_counts[split] = stats.images
            expected_sizes[split] = sizes
    except ConversionError as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1

    print_summary(stats_list, class_names)

    if args.dry_run:
        print("\n[dry-run] 未写入任何文件。")
        return 0

    data_yaml = write_data_yaml(output_dir, splits, class_names)
    summary_path = write_summary(output_dir, raw_dir, anno_source, class_names, stats_list)
    print(f"\n[3/3] 训练配置: {data_yaml}")
    print(f"      转换统计: {summary_path}")

    errors = verify_output(output_dir, splits, class_names, expected_counts, expected_sizes)
    if errors:
        print("\n[ERROR] 校验未通过:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("\n[OK] 校验通过: 图像与标签一一对应，标签均为 5 字段且坐标已归一化。")
    print("\n下一步训练示例:")
    print(f'  yolo detect train data="{data_yaml}" model=yolov8n.pt epochs=100 imgsz=640 batch=8 device=0')
    return 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""PCB 缺陷检测 —— YOLOv8 单图 / 批量推理脚本。

职责：
- 读取 ``configs/predict.yaml``（支持 ``--set`` 与常用命令行参数覆盖）；
- 加载训练输出的 ``best.pt``（验证集最优权重，也可指定其他 ``.pt``）；
- 支持单张图像、图像目录、图像列表 ``.txt`` 三种输入，自动解析为图像列表；
- 调用 Ultralytics ``model.predict()``，控制台打印每个目标的类别 / 置信度 / 坐标；
- 通过 ``save=True`` 保存带检测框的可视化图像；
- 在报告目录写入 ``predictions.json``（逐图检测结果与类别统计）。

用法::

    # 单图推理
    python scripts/predict_yolo.py --source datasets/pcb_yolo/images/val/01.jpg

    # 批量推理（目录），只抽检前 20 张
    python scripts/predict_yolo.py --source datasets/pcb_yolo/images/val --limit 20

    # 指定权重与阈值
    python scripts/predict_yolo.py \
        --weights runs/pcb_yolo/train-2/weights/best.pt \
        --source datasets/pcb_yolo/images/val --set predict.conf=0.3
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from pcb_config import (
    ConfigError,
    DEFAULT_PREDICT_CONFIG,
    IMAGE_SUFFIXES,
    PredictConfig,
    PredictSettings,
    normalize_class_names,
)

ModelLoader = Callable[[Path], Any]
Printer = Callable[[str], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number else default


def _to_list(value: Any) -> List[Any]:
    """兼容 torch.Tensor / numpy.ndarray / list：统一转成 Python 列表。"""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


# ---------------------------------------------------------------------------
# 推理结果数据结构
# ---------------------------------------------------------------------------
@dataclass
class Detection:
    """单个检测目标。"""

    class_id: int
    class_name: str
    confidence: float
    xyxy: List[float]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 6),
            "xyxy": [round(value, 2) for value in self.xyxy],
        }


@dataclass
class ImagePrediction:
    """单张图像的推理结果。"""

    image: Path
    detections: List[Detection] = field(default_factory=list)
    speed_ms: float = 0.0
    saved: Optional[Path] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "image": str(self.image),
            "objects": len(self.detections),
            "speed_ms": round(self.speed_ms, 4),
            "saved": str(self.saved) if self.saved else None,
            "detections": [item.as_dict() for item in self.detections],
        }


@dataclass
class PredictResult:
    """一次完整推理的结构化结果，供 CLI、报告与测试复用。"""

    weights: Path
    source: str
    output_dir: Optional[Path] = None
    images: List[ImagePrediction] = field(default_factory=list)
    report_paths: Dict[str, Path] = field(default_factory=dict)
    generated_at: str = ""

    @property
    def total_objects(self) -> int:
        return sum(len(image.detections) for image in self.images)

    def per_class(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for image in self.images:
            for detection in image.detections:
                counts[detection.class_name] = counts.get(detection.class_name, 0) + 1
        return dict(sorted(counts.items()))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at or _utc_now(),
            "weights": str(self.weights),
            "source": self.source,
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "total_images": len(self.images),
            "total_objects": self.total_objects,
            "per_class": self.per_class(),
            "images": [image.as_dict() for image in self.images],
        }
# ---------------------------------------------------------------------------
# 输入解析
# ---------------------------------------------------------------------------
def _read_image_list(list_path: Path) -> List[Path]:
    """读取图像列表 txt：每行一个路径，支持 # 注释，相对路径以 txt 所在目录为基准。"""
    list_path = Path(list_path)
    images: List[Path] = []
    for number, line in enumerate(list_path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        candidate = Path(text).expanduser()
        candidate = (
            candidate.resolve()
            if candidate.is_absolute()
            else (list_path.parent / candidate).resolve()
        )
        if not candidate.is_file():
            raise ConfigError(f"图像列表第 {number} 行指向的文件不存在: {candidate}")
        if candidate.suffix.lower() not in IMAGE_SUFFIXES:
            raise ConfigError(f"图像列表第 {number} 行不是支持的图像类型: {candidate}")
        images.append(candidate)
    if not images:
        raise ConfigError(f"图像列表为空: {list_path}")
    return images


def resolve_sources(source: Path, max_images: int = 0) -> List[Path]:
    """把推理输入解析为图像路径列表：单张图像 / 图像目录 / 图像列表 txt。"""
    source = Path(source)
    if source.is_dir():
        images = sorted(
            (
                item
                for item in source.iterdir()
                if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
            ),
            key=lambda item: item.name.lower(),
        )
        if not images:
            raise ConfigError(
                f"目录中没有可推理的图像（支持 {', '.join(sorted(IMAGE_SUFFIXES))}）: {source}"
            )
    elif source.is_file() and source.suffix.lower() == ".txt":
        images = _read_image_list(source)
    elif source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        images = [source]
    elif not source.exists():
        raise ConfigError(f"推理输入不存在: {source}")
    else:
        raise ConfigError(f"不支持的推理输入（需要图像 / 目录 / txt 列表）: {source}")

    if max_images and max_images > 0:
        images = images[:max_images]
    return images


# ---------------------------------------------------------------------------
# 结果提取
# ---------------------------------------------------------------------------
def _detections(boxes: Any, names: Mapping[int, str]) -> List[Detection]:
    """从 Ultralytics ``Boxes`` 中提取类别、置信度与坐标。"""
    xyxy = _to_list(getattr(boxes, "xyxy", None))
    confidence = _to_list(getattr(boxes, "conf", None))
    classes = _to_list(getattr(boxes, "cls", None))
    total = min(len(xyxy), len(confidence), len(classes))
    detections: List[Detection] = []
    for index in range(total):
        class_id = int(classes[index])
        detections.append(
            Detection(
                class_id=class_id,
                class_name=str(names.get(class_id, class_id)),
                confidence=_safe_float(confidence[index]),
                xyxy=[_safe_float(value) for value in _to_list(xyxy[index])],
            )
        )
    return detections


def _image_speed(result: Any) -> float:
    """单图耗时（毫秒）：累加 preprocess / inference / postprocess。"""
    speed = getattr(result, "speed", None)
    if not isinstance(speed, Mapping):
        return 0.0
    return sum(_safe_float(value) for value in speed.values())


def _locate_saved(output_dir: Optional[Path], image: Path) -> Optional[Path]:
    """在输出目录中定位该图像对应的带框图（Ultralytics 保持原文件名或另存为 jpg）。"""
    if output_dir is None:
        return None
    folder = Path(output_dir)
    if not folder.is_dir():
        return None
    direct = folder / image.name
    if direct.is_file():
        return direct
    for suffix in sorted(IMAGE_SUFFIXES):
        candidate = folder / f"{image.stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def collect_predictions(
    results: Sequence[Any],
    *,
    images: Sequence[Path],
    output_dir: Optional[Path] = None,
    default_names: Optional[Mapping[int, str]] = None,
) -> List[ImagePrediction]:
    """把 Ultralytics 的 Results 列表规整为 ``ImagePrediction`` 列表。"""
    fallback_names = normalize_class_names(default_names) if default_names else {}
    predictions: List[ImagePrediction] = []
    for index, result in enumerate(results):
        raw_path = getattr(result, "path", None)
        if raw_path:
            image = Path(str(raw_path))
        elif index < len(images):
            image = Path(images[index])
        else:
            image = Path(f"image{index}")
        names = normalize_class_names(getattr(result, "names", None)) or fallback_names
        predictions.append(
            ImagePrediction(
                image=image,
                detections=_detections(getattr(result, "boxes", None), names),
                speed_ms=_image_speed(result),
                saved=_locate_saved(output_dir, image),
            )
        )
    return predictions


# ---------------------------------------------------------------------------
# 结果输出
# ---------------------------------------------------------------------------
def write_predictions(result: PredictResult, report_dir: Path) -> Dict[str, Path]:
    """把推理结果写入 ``predictions.json``。"""
    folder = Path(report_dir)
    folder.mkdir(parents=True, exist_ok=True)
    result.generated_at = result.generated_at or _utc_now()
    path = folder / "predictions.json"
    path.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result.report_paths["predictions"] = path
    return result.report_paths


def _print_summary(result: PredictResult, printer: Printer) -> None:
    printer("")
    printer("========== 推理结果 ==========")
    printer(f"权重: {result.weights}")
    for image in result.images:
        saved = f"  ->  {image.saved}" if image.saved else ""
        printer(f"-- {image.image.name}: {len(image.detections)} 个目标{saved}")
        for detection in image.detections:
            printer(
                f"   {detection.class_name:<16} conf={detection.confidence:.4f} "
                f"xyxy={[round(value, 2) for value in detection.xyxy]}"
            )
    printer(f"总计: {len(result.images)} 张图像，{result.total_objects} 个目标")
    counts = result.per_class()
    if counts:
        printer("类别统计: " + ", ".join(f"{name}={count}" for name, count in counts.items()))
    if result.output_dir:
        printer(f"带框图目录: {result.output_dir}")
    if result.report_paths.get("predictions"):
        printer(f"结果文件: {result.report_paths['predictions']}")
    printer("==============================")


# ---------------------------------------------------------------------------
# 推理流程
# ---------------------------------------------------------------------------
def _load_yolo(weights: Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(weights))


def _resolve_output_dir(results: Sequence[Any], settings: PredictSettings) -> Optional[Path]:
    """优先使用 Ultralytics 实际保存目录；未开启保存时返回 None。"""
    if not settings.save:
        return None
    for result in results:
        save_dir = getattr(result, "save_dir", None)
        if save_dir:
            return Path(save_dir)
    return Path(settings.project) / settings.name


def predict(
    config: PredictConfig,
    *,
    model_loader: Optional[ModelLoader] = None,
    printer: Printer = print,
) -> PredictResult:
    """执行推理：校验输入 -> 加载权重 -> predict -> 写结果 -> 打印摘要。"""
    settings = config.settings
    weights = settings.weights
    if weights is None:
        raise ConfigError("推理配置未指定 weights，请通过 --weights 或 --set weights=... 指定")
    weights = Path(weights)
    if not weights.is_file():
        raise ConfigError(f"权重文件不存在: {weights}")
    if settings.source is None:
        raise ConfigError("推理配置未指定 source，请通过 --source 或 --set source=... 指定")

    source = Path(settings.source)
    images = resolve_sources(source, settings.max_images)

    printer(f"[推理] 权重: {weights}")
    printer(f"[推理] 输入: {source}（{len(images)} 张图像）")
    printer(
        f"[推理] 参数: imgsz={settings.imgsz} conf={settings.conf} "
        f"iou={settings.iou} device={settings.device} max_det={settings.max_det}"
    )

    loader = model_loader or _load_yolo
    model = loader(weights)
    raw_results = model.predict(
        source=[str(path) for path in images], **settings.to_predict_kwargs()
    )
    results = list(raw_results) if raw_results is not None else []
    if not results:
        raise RuntimeError("model.predict() 未返回任何结果")

    output_dir = _resolve_output_dir(results, settings)
    result = PredictResult(
        weights=weights,
        source=str(source),
        output_dir=output_dir,
        images=collect_predictions(
            results,
            images=images,
            output_dir=output_dir,
            default_names=getattr(model, "names", None),
        ),
        generated_at=_utc_now(),
    )
    write_predictions(result, settings.report_dir)
    _print_summary(result, printer)
    return result


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PCB 缺陷检测 —— YOLOv8 单图 / 批量推理")
    parser.add_argument("--config", type=Path, default=DEFAULT_PREDICT_CONFIG, help="推理配置文件路径")
    parser.add_argument("--weights", type=Path, default=None, help="待加载权重，推荐 best.pt")
    parser.add_argument(
        "--source",
        default=None,
        help="推理输入：单张图像 / 图像目录 / 图像列表 txt（相对当前工作目录）",
    )
    parser.add_argument("--conf", type=float, default=None, help="覆盖 predict.conf")
    parser.add_argument("--iou", type=float, default=None, help="覆盖 predict.iou")
    parser.add_argument("--imgsz", type=int, default=None, help="覆盖 predict.imgsz")
    parser.add_argument("--device", default=None, help="覆盖 predict.device，如 0 / cpu")
    parser.add_argument("--limit", type=int, default=None, help="覆盖 predict.max_images，0 表示不限制")
    parser.add_argument("--name", default=None, help="覆盖 predict.name")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖任意配置项，可重复使用，例如 --set predict.save=false",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅打印解析后的配置，不加载模型")
    return parser


def _cli_overrides(args: argparse.Namespace) -> List[str]:
    overrides = list(args.overrides)
    for key, value in (
        ("predict.conf", args.conf),
        ("predict.iou", args.iou),
        ("predict.imgsz", args.imgsz),
        ("predict.device", args.device),
        ("predict.max_images", args.limit),
        ("predict.name", args.name),
    ):
        if value is not None:
            overrides.append(f"{key}={value}")
    return overrides


def _print_dry_run(config: PredictConfig) -> None:
    settings = config.settings
    print("[dry-run] 权重:", settings.weights)
    print(f"[dry-run] 输入: {settings.source}")
    print(
        f"[dry-run] imgsz={settings.imgsz} conf={settings.conf} iou={settings.iou} "
        f"device={settings.device} save={settings.save} max_images={settings.max_images}"
    )
    print("[dry-run] predict 参数:", json.dumps(settings.to_predict_kwargs(), ensure_ascii=False))
    print(f"[dry-run] 带框图目录: {Path(settings.project) / settings.name}")
    print(f"[dry-run] 结果目录: {settings.report_dir}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = PredictConfig.load(
            args.config,
            weights=args.weights,
            source=Path(args.source) if args.source else None,
            overrides=_cli_overrides(args),
        )
        if args.dry_run:
            _print_dry_run(config)
            return 0
        result = predict(config)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - 依赖真实权重 / GPU 环境
        print(f"[ERROR] 推理失败: {exc}")
        traceback.print_exc()
        return 1

    print(f"[完成] 推理图像数: {len(result.images)}，目标数: {result.total_objects}")
    if result.output_dir:
        print(f"[完成] 带框图目录: {result.output_dir}")
    if result.report_paths:
        print(f"[完成] 结果文件: {result.report_paths.get('predictions')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


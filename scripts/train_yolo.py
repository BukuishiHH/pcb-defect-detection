# -*- coding: utf-8 -*-
"""PCB 缺陷检测 —— YOLOv8 训练脚本。

职责：
- 读取 ``configs/train.yaml``（支持 ``--set`` 与常用命令行参数覆盖）；
- 训练前校验数据集配置；
- 调用 Ultralytics 训练，训练结束后自动执行验证评估（``scripts/evaluate_yolo.py``）；
- 在训练输出目录写入 ``train_summary.json``，记录最终指标、权重路径与完整配置快照。

用法::

    python scripts/train_yolo.py
    python scripts/train_yolo.py --epochs 200 --batch 16
    python scripts/train_yolo.py --set train.lr0=0.005 --no-evaluate
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
    DEFAULT_TRAIN_CONFIG,
    TrainConfig,
    TrainSettings,
    validate_dataset_yaml,
)

ModelFactory = Callable[[str], Any]
Evaluator = Callable[..., Any]
Printer = Callable[[str], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number else default


# ---------------------------------------------------------------------------
# 训练结果
# ---------------------------------------------------------------------------
@dataclass
class TrainResult:
    """训练输出摘要，供 CLI 与测试使用。"""

    save_dir: Path
    best_weights: Optional[Path] = None
    last_weights: Optional[Path] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    summary_path: Optional[Path] = None
    evaluation: Any = None
    evaluation_passed: Optional[bool] = None


def extract_metrics(results: Any) -> Dict[str, float]:
    """把 Ultralytics 训练返回值中的 results_dict 规整为普通字典。"""
    if results is None:
        return {}
    mapping = getattr(results, "results_dict", None)
    if mapping is None and isinstance(results, Mapping):
        mapping = results
    if not isinstance(mapping, Mapping):
        return {}
    return {str(key): _safe_float(value) for key, value in mapping.items()}


def resolve_train_dir(settings: TrainSettings, results: Any) -> Path:
    """优先使用 Ultralytics 实际保存目录，便于识别自动递增的 run 名称。"""
    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        return Path(save_dir)
    return Path(settings.project) / settings.name


def find_weights(train_dir: Path) -> tuple:
    """返回 (best, last) 权重路径，缺失项为 None。"""
    weights_dir = Path(train_dir) / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"
    return (
        best if best.is_file() else None,
        last if last.is_file() else None,
    )


def write_train_summary(
    train_dir: Path,
    result: TrainResult,
    settings: TrainSettings,
    snapshot: Mapping[str, Any],
) -> Path:
    """在训练目录写入 train_summary.json（最终指标 + 配置快照）。"""
    folder = Path(train_dir)
    folder.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "generated_at": result.evaluation.generated_at
        if getattr(result.evaluation, "generated_at", "")
        else _utc_now(),
        "model": settings.model,
        "data": str(settings.data),
        "epochs": settings.epochs,
        "save_dir": str(folder),
        "best_weights": str(result.best_weights) if result.best_weights else None,
        "last_weights": str(result.last_weights) if result.last_weights else None,
        "final_metrics": result.metrics,
        "evaluation_passed": result.evaluation_passed,
        "config": snapshot,
    }
    path = folder / "train_summary.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return path


# ---------------------------------------------------------------------------
# 训练流程
# ---------------------------------------------------------------------------
def _load_yolo(weights: str) -> Any:
    from ultralytics import YOLO

    return YOLO(weights)


def _default_evaluator(
    config_path: Path,
    *,
    weights: Path,
    strict: bool,
    printer: Printer = print,
) -> Any:
    from evaluate_yolo import run_evaluation

    return run_evaluation(config_path, weights=weights, strict=strict, printer=printer)


def train(
    config: TrainConfig,
    *,
    model_factory: Optional[ModelFactory] = None,
    evaluator: Optional[Evaluator] = None,
    printer: Printer = print,
) -> TrainResult:
    """执行训练，并在完成后按配置自动评估。"""
    settings = config.settings
    validate_dataset_yaml(settings.data, splits=("train", "val"))

    printer(f"[训练] 模型权重: {settings.model}")
    printer(f"[训练] 数据集: {settings.data}")
    printer(
        f"[训练] epochs={settings.epochs} imgsz={settings.imgsz} batch={settings.batch} "
        f"device={settings.device} workers={settings.workers}"
    )
    printer(f"[训练] 输出目录: {Path(settings.project) / settings.name}")

    factory = model_factory or _load_yolo
    model = factory(settings.model)
    results = model.train(**settings.to_train_kwargs())

    train_dir = resolve_train_dir(settings, results)
    best_weights, last_weights = find_weights(train_dir)
    result = TrainResult(
        save_dir=train_dir,
        best_weights=best_weights,
        last_weights=last_weights,
        metrics=extract_metrics(results),
    )
    result.summary_path = write_train_summary(train_dir, result, settings, config.snapshot)
    printer(f"[训练] 训练完成，输出目录: {train_dir}")

    if config.evaluation_enabled:
        target = best_weights or last_weights
        if target is None:
            raise ConfigError(
                f"训练完成但未找到 best.pt / last.pt，无法评估: {train_dir / 'weights'}"
            )
        printer(f"[训练] 开始训练后验证评估，权重: {target}")
        evaluate_fn = evaluator or _default_evaluator
        result.evaluation = evaluate_fn(
            config.evaluation_config,
            weights=target,
            strict=config.evaluation_strict,
            printer=printer,
        )
        result.evaluation_passed = bool(getattr(result.evaluation, "passed", True))
        result.summary_path = write_train_summary(train_dir, result, settings, config.snapshot)
        printer(
            f"[训练] 评估完成，阈值判定: {'通过' if result.evaluation_passed else '未通过'}"
        )
    else:
        printer("[训练] 已按配置跳过训练后评估 (evaluation.enabled=false)")

    return result


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PCB 缺陷检测 —— YOLOv8 训练")
    parser.add_argument("--config", type=Path, default=DEFAULT_TRAIN_CONFIG, help="训练配置文件路径")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖任意配置项，可重复使用，例如 --set train.lr0=0.005",
    )
    parser.add_argument("--epochs", type=int, default=None, help="覆盖 train.epochs")
    parser.add_argument("--batch", type=int, default=None, help="覆盖 train.batch")
    parser.add_argument("--device", default=None, help="覆盖 train.device，如 0 / 0,1 / cpu")
    parser.add_argument("--name", default=None, help="覆盖 train.name")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--evaluate", dest="evaluate", action="store_true", default=None, help="训练后执行评估")
    group.add_argument("--no-evaluate", dest="evaluate", action="store_false", help="训练后跳过评估")
    parser.add_argument("--dry-run", action="store_true", help="仅打印解析后的配置，不加载模型")
    return parser


def _cli_overrides(args: argparse.Namespace) -> List[str]:
    overrides = list(args.overrides)
    for key, value in (
        ("train.epochs", args.epochs),
        ("train.batch", args.batch),
        ("train.device", args.device),
        ("train.name", args.name),
    ):
        if value is not None:
            overrides.append(f"{key}={value}")
    if args.evaluate is not None:
        overrides.append(f"evaluation.enabled={'true' if args.evaluate else 'false'}")
    return overrides


def _print_dry_run(config: TrainConfig) -> None:
    settings = config.settings
    print("[dry-run] 模型权重:", settings.model)
    print(f"[dry-run] 数据集: {settings.data}")
    print(
        f"[dry-run] epochs={settings.epochs} imgsz={settings.imgsz} "
        f"batch={settings.batch} device={settings.device}"
    )
    print(f"[dry-run] 输出目录: {Path(settings.project) / settings.name}")
    print("[dry-run] train 参数:", json.dumps(settings.to_train_kwargs(), ensure_ascii=False))
    print(
        f"[dry-run] 训练后评估: {config.evaluation_enabled} "
        f"(config={config.evaluation_config}, strict={config.evaluation_strict})"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = TrainConfig.load(args.config, overrides=_cli_overrides(args))
        if args.dry_run:
            _print_dry_run(config)
            return 0
        result = train(config)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - 依赖真实训练/GPU 环境
        print(f"[ERROR] 训练失败: {exc}")
        traceback.print_exc()
        return 1

    print(f"[完成] 训练输出目录: {result.save_dir}")
    if result.best_weights:
        print(f"[完成] 最佳权重: {result.best_weights}")
    if result.summary_path:
        print(f"[完成] 训练摘要: {result.summary_path}")
    if result.evaluation_passed is False and config.evaluation_strict:
        print("[结果] 训练完成，但评估指标未达到配置阈值")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())


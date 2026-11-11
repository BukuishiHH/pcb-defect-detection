# -*- coding: utf-8 -*-
"""PCB 缺陷检测 —— YOLOv8 验证评估脚本。

在验证集上计算 Precision / Recall / mAP，输出：
- ``metrics.json``：总体与逐类指标、混淆矩阵、速度、阈值判定、最差图像；
- ``per_class.csv``：逐类指标表格；
- ``report.md``：可直接阅读的评估报告。

用法::

    python scripts/evaluate_yolo.py
    python scripts/evaluate_yolo.py --weights runs/pcb_yolo/train/weights/best.pt
    python scripts/evaluate_yolo.py --set dataset.split=test --no-strict
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from pcb_config import (
    ConfigError,
    DEFAULT_EVAL_CONFIG,
    EvalConfig,
    EvalSettings,
    ThresholdSettings,
    load_dataset_yaml,
    normalize_class_names,
    validate_dataset_yaml,
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


def _lookup(mapping: Mapping[str, Any], keys: Sequence[str], fallback: float = 0.0) -> float:
    for key in keys:
        if key in mapping:
            return _safe_float(mapping[key], fallback)
    return _safe_float(fallback)


def _f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    return (2 * precision * recall / denominator) if denominator > 0 else 0.0


def _ratio(value: float) -> str:
    return f"{value * 100:.2f}%"


# ---------------------------------------------------------------------------
# 结果数据结构
# ---------------------------------------------------------------------------
@dataclass
class ClassMetrics:
    """单个类别的评估指标。"""

    index: int
    name: str
    instances: int
    images: int
    precision: float
    recall: float
    f1: float
    map50: float
    map75: float
    map50_95: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "instances": self.instances,
            "images": self.images,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "map50": round(self.map50, 6),
            "map75": round(self.map75, 6),
            "map50_95": round(self.map50_95, 6),
        }


@dataclass
class ThresholdCheck:
    """单项阈值检查结果。"""

    name: str
    actual: Optional[float]
    threshold: Optional[float]
    passed: bool
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "actual": None if self.actual is None else round(self.actual, 6),
            "threshold": None if self.threshold is None else round(self.threshold, 6),
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass
class EvalResult:
    """一次完整评估的结构化结果。"""

    weights: Path
    data: Path
    split: str
    overall: Dict[str, float] = field(default_factory=dict)
    per_class: List[ClassMetrics] = field(default_factory=list)
    missing_classes: List[str] = field(default_factory=list)
    images: int = 0
    speed: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: List[List[int]] = field(default_factory=list)
    confusion_labels: List[str] = field(default_factory=list)
    worst_images: List[Dict[str, Any]] = field(default_factory=list)
    checks: List[ThresholdCheck] = field(default_factory=list)
    passed: bool = True
    save_dir: Optional[Path] = None
    artifacts: List[str] = field(default_factory=list)
    report_paths: Dict[str, Path] = field(default_factory=dict)
    generated_at: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at or _utc_now(),
            "weights": str(self.weights),
            "data": str(self.data),
            "split": self.split,
            "images": self.images,
            "overall": {key: round(value, 6) for key, value in self.overall.items()},
            "per_class": [row.as_dict() for row in self.per_class],
            "missing_classes": list(self.missing_classes),
            "speed_ms_per_image": {key: round(value, 4) for key, value in self.speed.items()},
            "confusion_matrix": {
                "labels": list(self.confusion_labels),
                "matrix": [list(row) for row in self.confusion_matrix],
            },
            "worst_images": list(self.worst_images),
            "thresholds": {
                "passed": self.passed,
                "checks": [check.as_dict() for check in self.checks],
            },
            "artifacts": list(self.artifacts),
            "save_dir": str(self.save_dir) if self.save_dir else None,
            "reports": {key: str(value) for key, value in self.report_paths.items()},
        }


# ---------------------------------------------------------------------------
# 指标提取
# ---------------------------------------------------------------------------
def _worst_images(image_metrics: Mapping[str, Any], limit: int) -> List[Dict[str, Any]]:
    """按 F1 升序挑出错误最多的图片，供人工复查（TP=FP=FN=0 的图片跳过）。"""
    if limit <= 0 or not isinstance(image_metrics, Mapping):
        return []
    rows: List[Dict[str, Any]] = []
    for name, stats in image_metrics.items():
        if not isinstance(stats, Mapping):
            continue
        tp = int(stats.get("tp", 0) or 0)
        fp = int(stats.get("fp", 0) or 0)
        fn = int(stats.get("fn", 0) or 0)
        if tp + fp + fn == 0:
            continue
        rows.append(
            {
                "image": str(name),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(_safe_float(stats.get("precision")), 6),
                "recall": round(_safe_float(stats.get("recall")), 6),
                "f1": round(_safe_float(stats.get("f1")), 6),
            }
        )
    rows.sort(key=lambda row: (row["f1"], -(row["fn"] + row["fp"]), row["image"]))
    return rows[:limit]


def _as_list(value: Any) -> List[Any]:
    """兼容 numpy 数组/None：避免对数组做布尔判断（numpy 会抛 ambiguous 错误）。"""
    if value is None:
        return []
    return list(value)


def _confusion_matrix(metrics: Any, names: Mapping[int, str]) -> tuple:
    """把 Ultralytics ConfusionMatrix 转成可 JSON 序列化的矩阵与标签。"""
    matrix_obj = getattr(metrics, "confusion_matrix", None)
    matrix = getattr(matrix_obj, "matrix", None)
    if matrix is None:
        return [], []
    try:
        rows = matrix.tolist()
    except AttributeError:
        rows = matrix
    data = [[int(value) for value in row] for row in rows]
    labels = [names.get(index, str(index)) for index in sorted(names)]
    if data and len(data) > len(labels):
        labels.append("background")
    return data, labels


def collect_metrics(
    metrics: Any,
    *,
    weights: Path,
    data: Path,
    split: str,
    worst_limit: int = 10,
) -> EvalResult:
    """从 Ultralytics 的 DetMetrics 中提取评估结果（不依赖模型对象，便于测试）。"""
    box = getattr(metrics, "box", None)
    if box is None:
        raise RuntimeError("评估结果缺少 box 指标，无法生成报告")

    names = normalize_class_names(getattr(metrics, "names", {}))
    class_indices = [int(index) for index in _as_list(getattr(box, "ap_class_index", None))]
    all_ap = _as_list(getattr(box, "all_ap", None))
    instances = [int(value) for value in _as_list(getattr(metrics, "nt_per_class", None))]
    image_counts = [int(value) for value in _as_list(getattr(metrics, "nt_per_image", None))]

    per_class: List[ClassMetrics] = []
    for position, class_index in enumerate(class_indices):
        if hasattr(box, "class_result"):
            precision, recall, ap50, ap50_95 = box.class_result(position)
        else:  # pragma: no cover - 兼容旧版 ultralytics
            precision = box.p[position]
            recall = box.r[position]
            ap50 = box.ap50[position]
            ap50_95 = box.ap[position]
        precision = _safe_float(precision)
        recall = _safe_float(recall)
        map75 = _safe_float(all_ap[position][5]) if len(all_ap) > position else 0.0
        per_class.append(
            ClassMetrics(
                index=class_index,
                name=names.get(class_index, str(class_index)),
                instances=instances[class_index] if class_index < len(instances) else 0,
                images=image_counts[class_index] if class_index < len(image_counts) else 0,
                precision=precision,
                recall=recall,
                f1=_f1(precision, recall),
                map50=_safe_float(ap50),
                map75=map75,
                map50_95=_safe_float(ap50_95),
            )
        )

    evaluated = {row.index for row in per_class}
    missing_classes = [names[index] for index in sorted(names) if index not in evaluated]

    results = getattr(metrics, "results_dict", {}) or {}
    precision = _lookup(results, ("metrics/precision(B)", "metrics/precision"), getattr(box, "mp", 0.0))
    recall = _lookup(results, ("metrics/recall(B)", "metrics/recall"), getattr(box, "mr", 0.0))
    map50 = _lookup(results, ("metrics/mAP50(B)", "metrics/mAP50"), getattr(box, "map50", 0.0))
    map50_95 = _lookup(results, ("metrics/mAP50-95(B)", "metrics/mAP50-95"), getattr(box, "map", 0.0))
    fitness = _lookup(results, ("fitness",), getattr(metrics, "fitness", 0.0))
    map75 = _safe_float(getattr(box, "map75", 0.0))
    overall = {
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "map50": map50,
        "map75": map75,
        "map50_95": map50_95,
        "fitness": fitness,
    }

    speed = {
        str(key): _safe_float(value)
        for key, value in (getattr(metrics, "speed", {}) or {}).items()
    }
    image_metrics = getattr(metrics, "image_metrics", None)
    if not isinstance(image_metrics, Mapping) or not image_metrics:
        # Ultralytics >= 8.3 把逐图指标挂在 DetMetrics.box.image_metrics 上
        image_metrics = getattr(box, "image_metrics", None)
    if not isinstance(image_metrics, Mapping):
        image_metrics = {}
    confusion_matrix, confusion_labels = _confusion_matrix(metrics, names)
    save_dir = getattr(metrics, "save_dir", None)
    return EvalResult(
        weights=Path(weights),
        data=Path(data),
        split=split,
        overall=overall,
        per_class=per_class,
        missing_classes=missing_classes,
        images=len(image_metrics) if isinstance(image_metrics, Mapping) else 0,
        speed=speed,
        confusion_matrix=confusion_matrix,
        confusion_labels=confusion_labels,
        worst_images=_worst_images(image_metrics, worst_limit),
        save_dir=Path(save_dir) if save_dir else None,
        generated_at=_utc_now(),
    )


def check_thresholds(result: EvalResult, thresholds: ThresholdSettings) -> List[ThresholdCheck]:
    """按配置阈值逐项判定；阈值为 0 的检查项视为跳过。"""
    if not thresholds.enabled:
        return [
            ThresholdCheck(
                name="阈值检查",
                actual=None,
                threshold=None,
                passed=True,
                detail="已在配置中关闭（thresholds.enabled=false）",
            )
        ]

    checks: List[ThresholdCheck] = []

    def add(name: str, actual: float, threshold: float, detail: str = "") -> None:
        if threshold <= 0:
            return
        checks.append(
            ThresholdCheck(
                name=name,
                actual=float(actual),
                threshold=float(threshold),
                passed=float(actual) + 1e-9 >= float(threshold),
                detail=detail,
            )
        )

    add("mAP@0.5", result.overall["map50"], thresholds.map50)
    add("mAP@0.5:0.95", result.overall["map50_95"], thresholds.map50_95)
    add("Precision", result.overall["precision"], thresholds.precision)
    add("Recall", result.overall["recall"], thresholds.recall)

    if thresholds.per_class_map50 > 0:
        for row in result.per_class:
            if row.instances <= 0:
                continue
            add(f"{row.name} mAP@0.5", row.map50, thresholds.per_class_map50, f"instances={row.instances}")

    add(
        "已评估类别数",
        float(len(result.per_class)),
        float(thresholds.min_classes),
        f"缺失类别: {', '.join(result.missing_classes) if result.missing_classes else '无'}",
    )
    return checks


# ---------------------------------------------------------------------------
# 报告输出
# ---------------------------------------------------------------------------
def list_artifacts(save_dir: Optional[Path]) -> List[str]:
    """列出 Ultralytics 验证目录中的图表/结果文件，便于在报告中索引。"""
    if save_dir is None:
        return []
    folder = Path(save_dir)
    if not folder.is_dir():
        return []
    names = set()
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.json", "*.csv"):
        for path in folder.glob(pattern):
            if path.is_file():
                names.add(path.name)
    return sorted(names)


def _display_value(value: Optional[float]) -> str:
    """阈值表展示：0~1 视为比例显示百分比，计数类（如类别数）直接显示整数。"""
    if value is None:
        return "-"
    if -1.0 <= value <= 1.0:
        return _ratio(value)
    return f"{value:g}"


def render_markdown(result: EvalResult) -> str:
    """生成 Markdown 评估报告。"""
    status = "通过" if result.passed else "未通过"
    lines: List[str] = [
        "# PCB 缺陷检测评估报告",
        "",
        f"- 生成时间：{result.generated_at or _utc_now()}",
        f"- 权重：`{result.weights}`",
        f"- 数据集：`{result.data}`",
        f"- 划分：`{result.split}`（{result.images} 张图像）",
        f"- 阈值判定：**{status}**",
        "",
        "## 总体指标",
        "",
        "| 指标 | 数值 |",
        "| --- | --- |",
        f"| Precision | {_ratio(result.overall.get('precision', 0.0))} |",
        f"| Recall | {_ratio(result.overall.get('recall', 0.0))} |",
        f"| F1 | {_ratio(result.overall.get('f1', 0.0))} |",
        f"| mAP@0.5 | {_ratio(result.overall.get('map50', 0.0))} |",
        f"| mAP@0.75 | {_ratio(result.overall.get('map75', 0.0))} |",
        f"| mAP@0.5:0.95 | {_ratio(result.overall.get('map50_95', 0.0))} |",
        "",
        "## 逐类指标",
        "",
        "| 类别 | 实例数 | 图像数 | P | R | F1 | mAP@0.5 | mAP@0.5:0.95 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in result.per_class:
        lines.append(
            f"| {row.name} | {row.instances} | {row.images} | {_ratio(row.precision)} | "
            f"{_ratio(row.recall)} | {_ratio(row.f1)} | {_ratio(row.map50)} | {_ratio(row.map50_95)} |"
        )
    if result.missing_classes:
        lines.extend(["", f"验证集中未出现的类别：{', '.join(result.missing_classes)}"])

    lines.extend(["", "## 阈值检查", "", "| 检查项 | 实际值 | 阈值 | 结果 | 备注 |", "| --- | --- | --- | --- | --- |"])
    for check in result.checks:
        actual = _display_value(check.actual)
        threshold = _display_value(check.threshold)
        verdict = "通过" if check.passed else "未通过"
        lines.append(f"| {check.name} | {actual} | {threshold} | {verdict} | {check.detail} |")

    if result.speed:
        lines.extend(["", "## 推理速度（毫秒/张）", ""])
        lines.extend(f"- {key}: {value:.3f}" for key, value in result.speed.items())

    if result.worst_images:
        lines.extend(
            ["", "## 待复查图像（F1 最低）", "", "| 图像 | TP | FP | FN | F1 |", "| --- | --- | --- | --- | --- |"]
        )
        for row in result.worst_images:
            lines.append(
                f"| {row['image']} | {row['tp']} | {row['fp']} | {row['fn']} | {_ratio(row['f1'])} |"
            )

    if result.artifacts:
        lines.extend(["", "## 评估图表", ""])
        lines.extend(f"- {name}" for name in result.artifacts)

    if result.report_paths:
        lines.extend(["", "## 报告文件", ""])
        lines.extend(f"- {name}: `{path}`" for name, path in result.report_paths.items())

    lines.append("")
    return "\n".join(lines)


def write_reports(result: EvalResult, report_dir: Path) -> Dict[str, Path]:
    """把评估结果写入 metrics.json / per_class.csv / report.md。"""
    folder = Path(report_dir)
    folder.mkdir(parents=True, exist_ok=True)
    result.generated_at = result.generated_at or _utc_now()
    result.artifacts = list_artifacts(result.save_dir)

    csv_path = folder / "per_class.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "index", "name", "instances", "images", "precision", "recall",
                "f1", "map50", "map75", "map50_95",
            ]
        )
        for row in result.per_class:
            writer.writerow(
                [
                    row.index, row.name, row.instances, row.images,
                    round(row.precision, 6), round(row.recall, 6),
                    round(row.f1, 6), round(row.map50, 6),
                    round(row.map75, 6), round(row.map50_95, 6),
                ]
            )

    metrics_path = folder / "metrics.json"
    result.report_paths = {"metrics": metrics_path, "per_class": csv_path}
    report_path = folder / "report.md"
    report_path.write_text(render_markdown(result), encoding="utf-8")
    result.report_paths["report"] = report_path

    metrics_path.write_text(
        json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result.report_paths


# ---------------------------------------------------------------------------
# 评估流程
# ---------------------------------------------------------------------------
def _load_yolo(weights: Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(weights))


def _check_class_names(data_names: Mapping[int, str], model_names: Mapping[int, str]) -> None:
    if not model_names:
        raise ConfigError("权重文件缺少类别信息 names，无法确认类别定义")
    differences = []
    for index in sorted(set(data_names) | set(model_names)):
        if data_names.get(index) != model_names.get(index):
            differences.append(
                f"  [{index}] 数据集={data_names.get(index)!r} 模型={model_names.get(index)!r}"
            )
    if differences:
        raise ConfigError(
            "数据集与权重的类别定义不一致，请确认权重与 data.yaml 对应同一类别体系:\n"
            + "\n".join(differences)
        )


def _print_summary(result: EvalResult, printer: Printer) -> None:
    printer("")
    printer("========== 评估结果 ==========")
    printer(
        "总体: P={:.4f}  R={:.4f}  F1={:.4f}  mAP50={:.4f}  mAP75={:.4f}  mAP50-95={:.4f}".format(
            result.overall.get("precision", 0.0),
            result.overall.get("recall", 0.0),
            result.overall.get("f1", 0.0),
            result.overall.get("map50", 0.0),
            result.overall.get("map75", 0.0),
            result.overall.get("map50_95", 0.0),
        )
    )
    printer(f"验证图像数: {result.images}  已评估类别数: {len(result.per_class)}")
    printer("-- 逐类指标（mAP@0.5）--")
    for row in result.per_class:
        printer(f"  {row.name}: P={row.precision:.4f} R={row.recall:.4f} mAP50={row.map50:.4f} instances={row.instances}")
    if result.missing_classes:
        printer(f"  验证集未出现类别: {', '.join(result.missing_classes)}")
    printer("-- 阈值检查 --")
    for check in result.checks:
        if check.actual is None:
            printer(f"  [跳过] {check.name}: {check.detail}")
            continue
        printer(
            f"  [{'通过' if check.passed else '未通过'}] {check.name}: "
            f"{check.actual:.4f} / 阈值 {check.threshold:.4f}{('  ' + check.detail) if check.detail else ''}"
        )
    printer(f"阈值判定: {'通过' if result.passed else '未通过'}")
    if result.report_paths:
        printer(f"报告: {result.report_paths.get('report')}")
        printer(f"指标: {result.report_paths.get('metrics')}")
        printer(f"逐类: {result.report_paths.get('per_class')}")
    if result.save_dir:
        printer(f"图表目录: {result.save_dir}")
    printer("==============================")


def evaluate(
    config: EvalConfig,
    *,
    model_loader: Optional[ModelLoader] = None,
    printer: Printer = print,
) -> EvalResult:
    """执行一次完整评估：校验输入 -> 加载权重 -> val -> 阈值判定 -> 写报告。"""
    settings = config.settings
    weights = settings.weights
    if weights is None:
        raise ConfigError("评估配置未指定 weights，请通过 --weights 或 --set weights=... 指定")
    weights = Path(weights)
    if not weights.is_file():
        raise ConfigError(f"权重文件不存在: {weights}")
    if not settings.data.is_file():
        raise ConfigError(f"数据集配置不存在: {settings.data}")

    validate_dataset_yaml(settings.data, splits=(settings.split,))
    dataset = load_dataset_yaml(settings.data)
    data_names = normalize_class_names(dataset.get("names"))

    printer(f"[评估] 权重: {weights}")
    printer(f"[评估] 数据集: {settings.data} (split={settings.split})")
    printer(f"[评估] 设备: {settings.device}  imgsz={settings.imgsz}  batch={settings.batch}  conf={settings.conf}")

    loader = model_loader or _load_yolo
    model = loader(weights)
    _check_class_names(data_names, normalize_class_names(getattr(model, "names", {})))

    printer("[评估] 开始在验证集上计算指标 ...")
    metrics = model.val(**settings.to_val_kwargs())
    if metrics is None:
        raise RuntimeError("model.val() 未返回指标，无法生成报告")

    result = collect_metrics(
        metrics,
        weights=weights,
        data=settings.data,
        split=settings.split,
        worst_limit=settings.worst_images,
    )
    result.checks = check_thresholds(result, settings.thresholds)
    result.passed = all(check.passed for check in result.checks)
    write_reports(result, settings.report_dir)
    _print_summary(result, printer)
    return result


def run_evaluation(
    config_path: Path = DEFAULT_EVAL_CONFIG,
    *,
    weights: Optional[Path] = None,
    overrides: Sequence[str] = (),
    strict: Optional[bool] = None,
    model_loader: Optional[ModelLoader] = None,
    printer: Printer = print,
) -> EvalResult:
    """供训练脚本 / 测试复用的入口：加载配置后执行评估。"""
    config = EvalConfig.load(config_path, weights=weights, overrides=overrides)
    if strict is not None:
        config = replace(config, strict=bool(strict))
    return evaluate(config, model_loader=model_loader, printer=printer)


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PCB 缺陷检测 —— YOLOv8 验证评估")
    parser.add_argument("--config", type=Path, default=DEFAULT_EVAL_CONFIG, help="评估配置文件路径")
    parser.add_argument("--weights", type=Path, default=None, help="待评估权重（默认取配置中的 weights）")
    parser.add_argument("--split", default=None, help="覆盖 dataset.split，例如 test")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="覆盖任意配置项，可重复使用，例如 --set thresholds.map50=0.6",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--strict", dest="strict", action="store_true", default=None, help="指标未达阈值时返回非 0")
    group.add_argument("--no-strict", dest="strict", action="store_false", help="指标未达阈值时仍返回 0")
    parser.add_argument("--dry-run", action="store_true", help="仅打印解析后的配置，不加载模型")
    return parser


def _print_dry_run(config: EvalConfig) -> None:
    settings = config.settings
    print("[dry-run] 权重:", settings.weights)
    print(f"[dry-run] 数据集: {settings.data} (split={settings.split})")
    print("[dry-run] val 参数:", json.dumps(settings.to_val_kwargs(), ensure_ascii=False))
    print("[dry-run] 阈值:", settings.thresholds)
    print("[dry-run] 报告目录:", settings.report_dir)
    print("[dry-run] strict:", config.strict)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        overrides = list(args.overrides)
        if args.split:
            overrides.append(f"dataset.split={args.split}")
        config = EvalConfig.load(args.config, weights=args.weights, overrides=overrides)
        if args.strict is not None:
            config = replace(config, strict=args.strict)
        if args.dry_run:
            _print_dry_run(config)
            return 0
        result = evaluate(config)
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - 依赖真实训练/GPU 环境
        print(f"[ERROR] 评估过程失败: {exc}")
        traceback.print_exc()
        return 1

    if result.passed:
        return 0
    if config.strict:
        print("[结果] 指标未达到配置阈值，返回非 0 退出码")
        return 1
    print("[结果] 指标未达到配置阈值（strict=false，仅记录）")
    return 0


if __name__ == "__main__":
    sys.exit(main())







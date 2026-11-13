# -*- coding: utf-8 -*-
"""训练 / 评估脚本共用的配置加载、校验与数据集检查工具。

设计约定：
- 所有可调参数集中在 ``configs/*.yaml``，脚本只负责流程编排；
- 配置中的相对路径一律相对仓库根目录解析，保证在任意工作目录下可运行；
- 配置在进入业务流程前完成校验，非法取值统一抛出 ``ConfigError``；
- 命令行可通过 ``--set section.key=value`` 临时覆盖，便于实验而不改动文件。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CONFIG = REPO_ROOT / "configs" / "train.yaml"
DEFAULT_EVAL_CONFIG = REPO_ROOT / "configs" / "eval.yaml"
DEFAULT_PREDICT_CONFIG = REPO_ROOT / "configs" / "predict.yaml"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ConfigError(ValueError):
    """配置文件缺失、结构错误或取值非法。"""


# ---------------------------------------------------------------------------
# YAML 读取与覆盖
# ---------------------------------------------------------------------------
def load_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"配置文件不存在: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"配置文件 YAML 解析失败: {path} ({exc})") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"配置文件顶层必须是映射: {path}")
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """递归合并两个映射，override 优先；不修改入参。"""
    merged: Dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_override_value(text: str) -> Any:
    """把 --set 的字符串值解析为 YAML 标量（数字/布尔/null/列表等）。"""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(config: Mapping[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """应用 ``a.b=value`` 形式的点号路径覆盖，键必须已存在，避免拼写错误被静默接受。"""
    merged = copy.deepcopy(dict(config))
    for item in overrides:
        if not isinstance(item, str) or "=" not in item:
            raise ConfigError(f"--set 需要 KEY=VALUE 形式，实际为: {item!r}")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ConfigError(f"--set 的键不能为空: {item!r}")
        cursor: Any = merged
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(cursor, dict) or part not in cursor or not isinstance(cursor[part], dict):
                raise ConfigError(f"--set 路径不存在: {key}")
            cursor = cursor[part]
        leaf = parts[-1]
        if not isinstance(cursor, dict) or leaf not in cursor:
            raise ConfigError(f"--set 路径不存在: {key}")
        cursor[leaf] = parse_override_value(raw_value)
    return merged


# ---------------------------------------------------------------------------
# 基础校验
# ---------------------------------------------------------------------------
def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context} 必须是映射，实际为: {type(value).__name__}")
    return value


def _check_keys(mapping: Mapping[str, Any], allowed: Sequence[str], context: str) -> None:
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ConfigError(
            f"{context} 存在未知配置项: {', '.join(unknown)}；可用配置项: {', '.join(sorted(allowed))}"
        )


def _as_int(value: Any, context: str, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{context} 必须是整数，实际为: {value!r}")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{context} 必须 >= {minimum}，实际为: {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{context} 必须 <= {maximum}，实际为: {value}")
    return value


def _as_float(
    value: Any,
    context: str,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{context} 必须是数值，实际为: {value!r}")
    number = float(value)
    if number != number:
        raise ConfigError(f"{context} 不能为 NaN")
    if minimum is not None and number < minimum:
        raise ConfigError(f"{context} 必须 >= {minimum}，实际为: {number}")
    if maximum is not None and number > maximum:
        raise ConfigError(f"{context} 必须 <= {maximum}，实际为: {number}")
    return number


def _as_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{context} 必须是布尔值 true/false，实际为: {value!r}")
    return value


def _as_str(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context} 必须是非空字符串，实际为: {value!r}")
    return value.strip()


def _as_device(value: Any, context: str) -> str:
    if isinstance(value, bool):
        raise ConfigError(f"{context} 必须是 GPU 编号或 'cpu'，实际为: {value!r}")
    if isinstance(value, int):
        if value < 0:
            raise ConfigError(f"{context} 不能为负数，实际为: {value}")
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ConfigError(f"{context} 必须是 GPU 编号或 'cpu'，实际为: {value!r}")


def resolve_path(value: Any, context: str, base: Path = REPO_ROOT) -> Path:
    """把配置中的路径解析为绝对路径；相对路径以仓库根目录为基准。"""
    if not isinstance(value, (str, Path)) or (isinstance(value, str) and not value.strip()):
        raise ConfigError(f"{context} 必须是非空路径，实际为: {value!r}")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


@lru_cache(maxsize=1)
def _ultralytics_option_keys() -> set:
    """Ultralytics 允许的训练/验证参数集合，用于校验直通配置项。"""
    try:
        from ultralytics.cfg import get_cfg
    except ImportError:  # pragma: no cover - 依赖缺失时给出明确提示
        raise ConfigError("未安装 ultralytics，无法校验训练/评估参数") from None
    return set(vars(get_cfg()))


def _split_extra(mapping: Mapping[str, Any], known: Sequence[str], context: str) -> Dict[str, Any]:
    """保留已知字段之外、但属于 Ultralytics 参数的配置项，实现可扩展的直通参数。"""
    extra_keys = sorted(set(mapping) - set(known))
    if not extra_keys:
        return {}
    valid = _ultralytics_option_keys()
    unknown = [key for key in extra_keys if key not in valid]
    if unknown:
        raise ConfigError(
            f"{context} 存在未知配置项: {', '.join(unknown)}（既不是内置字段，也不是 Ultralytics 参数）"
        )
    return {key: mapping[key] for key in extra_keys}


# ---------------------------------------------------------------------------
# 数据集配置
# ---------------------------------------------------------------------------
def normalize_class_names(raw: Any) -> Dict[int, str]:
    """把 YAML / 模型中的 names（dict 或 list）统一为 ``{索引: 名称}``。"""
    if isinstance(raw, Mapping):
        names: Dict[int, str] = {}
        for key, value in raw.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                raise ConfigError(f"names 的类别索引必须是整数，实际为: {key!r}") from None
            names[index] = str(value)
        return dict(sorted(names.items()))
    if isinstance(raw, (list, tuple)):
        return {index: str(value) for index, value in enumerate(raw)}
    return {}


def load_dataset_yaml(path: Path) -> Dict[str, Any]:
    """读取并校验 data.yaml：必须包含 names，且 nc 与类别数一致。"""
    data = load_yaml(Path(path))
    names = normalize_class_names(data.get("names"))
    if not names:
        raise ConfigError(f"数据集配置缺少 names: {path}")
    if "nc" in data:
        nc = data["nc"]
        if isinstance(nc, bool) or not isinstance(nc, int):
            raise ConfigError(f"数据集配置 nc 必须是整数，实际为: {nc!r}")
        if nc != len(names):
            raise ConfigError(f"数据集配置 nc={nc} 与 names 数量 {len(names)} 不一致: {path}")
    return data


def dataset_base_dir(yaml_path: Path, data: Mapping[str, Any]) -> Path:
    """数据集根目录：优先 data.yaml 中的 path，否则使用 yaml 所在目录。"""
    yaml_path = Path(yaml_path).resolve()
    base = yaml_path.parent
    raw = data.get("path")
    if raw:
        path = Path(str(raw)).expanduser()
        return path.resolve() if path.is_absolute() else (base / path).resolve()
    return base


def resolve_dataset_split(yaml_path: Path, data: Mapping[str, Any], split: str) -> List[Path]:
    """解析某个划分的图像目录 / 图像列表文件，并校验其存在性。"""
    value = data.get(split)
    if not value:
        raise ConfigError(f"数据集配置缺少 {split} 划分: {yaml_path}")
    entries = value if isinstance(value, (list, tuple)) else [value]
    base = dataset_base_dir(yaml_path, data)
    paths: List[Path] = []
    for entry in entries:
        path = Path(str(entry)).expanduser()
        path = path.resolve() if path.is_absolute() else (base / path).resolve()
        if not path.exists():
            raise ConfigError(f"数据集 {split} 路径不存在: {path}")
        paths.append(path)
    return paths


def _split_has_images(path: Path) -> bool:
    if path.is_dir():
        return any(
            item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES for item in path.iterdir()
        )
    if path.is_file() and path.suffix.lower() == ".txt":
        with path.open("r", encoding="utf-8") as handle:
            return any(line.strip() for line in handle)
    return path.is_file()


def validate_dataset_yaml(path: Path, splits: Sequence[str] = ("train", "val")) -> Dict[str, Any]:
    """训练/评估前的数据集快速检查：配置可解析、划分路径存在且包含图像。"""
    data = load_dataset_yaml(path)
    for split in splits:
        paths = resolve_dataset_split(path, data, split)
        if not any(_split_has_images(item) for item in paths):
            raise ConfigError(f"数据集 {split} 划分未找到图像: {', '.join(str(p) for p in paths)}")
    return data



# ---------------------------------------------------------------------------
# 训练配置
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainSettings:
    """训练超参数，与 Ultralytics ``model.train()`` 的参数一一对应。"""

    data: Path
    model: str
    project: Path = REPO_ROOT / "runs" / "pcb_yolo"
    name: str = "train"
    epochs: int = 100
    imgsz: int = 640
    batch: int = 8
    device: str = "0"
    workers: int = 4
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    patience: int = 30
    seed: int = 42
    deterministic: bool = True
    amp: bool = True
    cache: bool = False
    pretrained: bool = True
    resume: bool = False
    val: bool = True
    plots: bool = True
    save: bool = True
    save_period: int = -1
    cos_lr: bool = False
    close_mosaic: int = 10
    exist_ok: bool = False
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_train_kwargs(self) -> Dict[str, Any]:
        """转换为 Ultralytics train() 关键字参数（model 由 YOLO 构造函数接收）。"""
        kwargs: Dict[str, Any] = {
            "data": str(self.data),
            "epochs": self.epochs,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "device": self.device,
            "workers": self.workers,
            "optimizer": self.optimizer,
            "lr0": self.lr0,
            "lrf": self.lrf,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "warmup_epochs": self.warmup_epochs,
            "patience": self.patience,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "amp": self.amp,
            "cache": self.cache,
            "pretrained": self.pretrained,
            "resume": self.resume,
            "val": self.val,
            "plots": self.plots,
            "save": self.save,
            "save_period": self.save_period,
            "cos_lr": self.cos_lr,
            "close_mosaic": self.close_mosaic,
            "project": str(self.project),
            "name": self.name,
            "exist_ok": self.exist_ok,
        }
        kwargs.update(dict(self.extra))
        return kwargs

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "TrainSettings":
        dataset = _require_mapping(config.get("dataset"), "dataset")
        _check_keys(dataset, ("data",), "dataset")
        model = _require_mapping(config.get("model"), "model")
        _check_keys(model, ("weights",), "model")
        train = _require_mapping(config.get("train"), "train")

        known = (
            "epochs", "imgsz", "batch", "device", "workers", "optimizer", "lr0", "lrf",
            "momentum", "weight_decay", "warmup_epochs", "patience", "seed", "deterministic",
            "amp", "cache", "pretrained", "resume", "val", "plots", "save", "save_period",
            "cos_lr", "close_mosaic", "project", "name", "exist_ok",
        )
        extra = _split_extra(train, known, "train")

        return cls(
            data=resolve_path(dataset.get("data"), "dataset.data"),
            model=_as_str(model.get("weights"), "model.weights"),
            project=resolve_path(train.get("project", "runs/pcb_yolo"), "train.project"),
            name=_as_str(train.get("name", "train"), "train.name"),
            epochs=_as_int(train.get("epochs", 100), "train.epochs", minimum=1),
            imgsz=_as_int(train.get("imgsz", 640), "train.imgsz", minimum=32),
            batch=_as_int(train.get("batch", 8), "train.batch", minimum=-1),
            device=_as_device(train.get("device", "0"), "train.device"),
            workers=_as_int(train.get("workers", 4), "train.workers", minimum=0),
            optimizer=_as_str(train.get("optimizer", "auto"), "train.optimizer"),
            lr0=_as_float(train.get("lr0", 0.01), "train.lr0", minimum=0.0, maximum=1.0),
            lrf=_as_float(train.get("lrf", 0.01), "train.lrf", minimum=0.0, maximum=1.0),
            momentum=_as_float(train.get("momentum", 0.937), "train.momentum", minimum=0.0, maximum=1.0),
            weight_decay=_as_float(train.get("weight_decay", 0.0005), "train.weight_decay", minimum=0.0),
            warmup_epochs=_as_float(train.get("warmup_epochs", 3.0), "train.warmup_epochs", minimum=0.0),
            patience=_as_int(train.get("patience", 30), "train.patience", minimum=0),
            seed=_as_int(train.get("seed", 42), "train.seed", minimum=0),
            deterministic=_as_bool(train.get("deterministic", True), "train.deterministic"),
            amp=_as_bool(train.get("amp", True), "train.amp"),
            cache=_as_bool(train.get("cache", False), "train.cache"),
            pretrained=_as_bool(train.get("pretrained", True), "train.pretrained"),
            resume=_as_bool(train.get("resume", False), "train.resume"),
            val=_as_bool(train.get("val", True), "train.val"),
            plots=_as_bool(train.get("plots", True), "train.plots"),
            save=_as_bool(train.get("save", True), "train.save"),
            save_period=_as_int(train.get("save_period", -1), "train.save_period", minimum=-1),
            cos_lr=_as_bool(train.get("cos_lr", False), "train.cos_lr"),
            close_mosaic=_as_int(train.get("close_mosaic", 10), "train.close_mosaic", minimum=0),
            exist_ok=_as_bool(train.get("exist_ok", False), "train.exist_ok"),
            extra=extra,
        )


@dataclass(frozen=True)
class TrainConfig:
    """完整训练配置：模型、超参数与训练后评估开关。"""

    source: Path
    settings: TrainSettings
    evaluation_enabled: bool = True
    evaluation_config: Path = DEFAULT_EVAL_CONFIG
    evaluation_strict: bool = False
    snapshot: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = DEFAULT_TRAIN_CONFIG, overrides: Sequence[str] = ()) -> "TrainConfig":
        raw = apply_overrides(load_yaml(Path(path)), overrides)
        _check_keys(raw, ("dataset", "model", "train", "evaluation"), "训练配置")
        settings = TrainSettings.from_mapping(raw)

        evaluation = _require_mapping(raw.get("evaluation", {}), "evaluation")
        _check_keys(evaluation, ("enabled", "config", "strict"), "evaluation")
        return cls(
            source=Path(path),
            settings=settings,
            evaluation_enabled=_as_bool(evaluation.get("enabled", True), "evaluation.enabled"),
            evaluation_config=resolve_path(
                evaluation.get("config", "configs/eval.yaml"), "evaluation.config"
            ),
            evaluation_strict=_as_bool(evaluation.get("strict", False), "evaluation.strict"),
            snapshot=copy.deepcopy(raw),
        )


# ---------------------------------------------------------------------------
# 评估配置
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ThresholdSettings:
    """指标达标阈值；检查项阈值为 0 时表示跳过该项。"""

    enabled: bool = True
    map50: float = 0.70
    map50_95: float = 0.45
    precision: float = 0.60
    recall: float = 0.60
    per_class_map50: float = 0.50
    min_classes: int = 6

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ThresholdSettings":
        mapping = _require_mapping(config, "thresholds")
        _check_keys(
            mapping,
            ("enabled", "map50", "map50_95", "precision", "recall", "per_class_map50", "min_classes"),
            "thresholds",
        )
        return cls(
            enabled=_as_bool(mapping.get("enabled", True), "thresholds.enabled"),
            map50=_as_float(mapping.get("map50", 0.70), "thresholds.map50", 0.0, 1.0),
            map50_95=_as_float(mapping.get("map50_95", 0.45), "thresholds.map50_95", 0.0, 1.0),
            precision=_as_float(mapping.get("precision", 0.60), "thresholds.precision", 0.0, 1.0),
            recall=_as_float(mapping.get("recall", 0.60), "thresholds.recall", 0.0, 1.0),
            per_class_map50=_as_float(
                mapping.get("per_class_map50", 0.50), "thresholds.per_class_map50", 0.0, 1.0
            ),
            min_classes=_as_int(mapping.get("min_classes", 6), "thresholds.min_classes", minimum=0),
        )


@dataclass(frozen=True)
class EvalSettings:
    """验证集评估参数，对应 Ultralytics ``model.val()``。"""

    data: Path
    weights: Optional[Path] = None
    split: str = "val"
    imgsz: int = 640
    batch: int = 8
    conf: float = 0.001
    iou: float = 0.6
    device: str = "0"
    workers: int = 4
    max_det: int = 300
    half: bool = False
    plots: bool = True
    save_json: bool = False
    project: Path = REPO_ROOT / "runs" / "pcb_yolo"
    name: str = "eval"
    exist_ok: bool = True
    thresholds: ThresholdSettings = field(default_factory=ThresholdSettings)
    report_dir: Path = REPO_ROOT / "runs" / "pcb_yolo" / "eval_report"
    worst_images: int = 10
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_val_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "data": str(self.data),
            "split": self.split,
            "imgsz": self.imgsz,
            "batch": self.batch,
            "conf": self.conf,
            "iou": self.iou,
            "device": self.device,
            "workers": self.workers,
            "max_det": self.max_det,
            "plots": self.plots,
            "save_json": self.save_json,
            "project": str(self.project),
            "name": self.name,
            "exist_ok": self.exist_ok,
        }
        if self.half:
            # ultralytics >= 8.4 已弃用 half（改用 quantize），仅在显式开启时传入
            kwargs["half"] = True
        kwargs.update(dict(self.extra))
        return kwargs

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
        weights_override: Optional[Path] = None,
    ) -> "EvalSettings":
        dataset = _require_mapping(config.get("dataset"), "dataset")
        _check_keys(dataset, ("data", "split"), "dataset")
        eval_map = _require_mapping(config.get("eval"), "eval")
        known = (
            "imgsz", "batch", "conf", "iou", "device", "workers", "max_det",
            "half", "plots", "save_json", "project", "name", "exist_ok",
        )
        extra = _split_extra(eval_map, known, "eval")
        report = _require_mapping(config.get("report", {}), "report")
        _check_keys(report, ("dir", "worst_images"), "report")

        if weights_override is not None:
            weights_path = Path(weights_override).expanduser()
            weights = (
                weights_path.resolve()
                if weights_path.is_absolute()
                else (Path.cwd() / weights_path).resolve()
            )
        elif config.get("weights") is not None:
            weights = resolve_path(config.get("weights"), "weights")
        else:
            weights = None

        return cls(
            data=resolve_path(dataset.get("data"), "dataset.data"),
            weights=weights,
            split=_as_str(dataset.get("split", "val"), "dataset.split"),
            imgsz=_as_int(eval_map.get("imgsz", 640), "eval.imgsz", minimum=32),
            batch=_as_int(eval_map.get("batch", 8), "eval.batch", minimum=1),
            conf=_as_float(eval_map.get("conf", 0.001), "eval.conf", 0.0, 1.0),
            iou=_as_float(eval_map.get("iou", 0.6), "eval.iou", 0.0, 1.0),
            device=_as_device(eval_map.get("device", "0"), "eval.device"),
            workers=_as_int(eval_map.get("workers", 4), "eval.workers", minimum=0),
            max_det=_as_int(eval_map.get("max_det", 300), "eval.max_det", minimum=1),
            half=_as_bool(eval_map.get("half", False), "eval.half"),
            plots=_as_bool(eval_map.get("plots", True), "eval.plots"),
            save_json=_as_bool(eval_map.get("save_json", False), "eval.save_json"),
            project=resolve_path(eval_map.get("project", "runs/pcb_yolo"), "eval.project"),
            name=_as_str(eval_map.get("name", "eval"), "eval.name"),
            exist_ok=_as_bool(eval_map.get("exist_ok", True), "eval.exist_ok"),
            thresholds=ThresholdSettings.from_mapping(config.get("thresholds", {})),
            report_dir=resolve_path(report.get("dir", "runs/pcb_yolo/eval_report"), "report.dir"),
            worst_images=_as_int(report.get("worst_images", 10), "report.worst_images", minimum=0),
            extra=extra,
        )


@dataclass(frozen=True)
class EvalConfig:
    """完整评估配置：权重、验证参数、达标阈值与报告输出。"""

    source: Path
    settings: EvalSettings
    strict: bool = True
    snapshot: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        path: Path = DEFAULT_EVAL_CONFIG,
        weights: Optional[Path] = None,
        overrides: Sequence[str] = (),
    ) -> "EvalConfig":
        source = Path(path)
        raw = apply_overrides(load_yaml(source), overrides)
        _check_keys(raw, ("dataset", "weights", "eval", "thresholds", "report", "strict"), "评估配置")
        settings = EvalSettings.from_mapping(raw, weights_override=weights)
        return cls(
            source=source,
            settings=settings,
            strict=_as_bool(raw.get("strict", True), "strict"),
            snapshot=copy.deepcopy(raw),
        )


# ---------------------------------------------------------------------------
# 推理配置
# ---------------------------------------------------------------------------
def _as_int_list(
    value: Any,
    context: str,
    minimum: Optional[int] = None,
) -> Optional[Tuple[int, ...]]:
    """校验可选的整数列表（如 ``classes``），None 表示不过滤类别。"""
    if value is None:
        return None
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, (list, tuple)):
        raise ConfigError(f"{context} 必须是整数列表或 null，实际为: {value!r}")
    items: List[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ConfigError(f"{context} 的元素必须是整数，实际为: {item!r}")
        if minimum is not None and item < minimum:
            raise ConfigError(f"{context} 的元素必须 >= {minimum}，实际为: {item}")
        items.append(item)
    return tuple(items)


@dataclass(frozen=True)
class PredictSettings:
    """单图 / 批量推理参数，对应 Ultralytics ``model.predict()``。"""

    weights: Optional[Path] = None
    source: Optional[Path] = None
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45
    device: str = "0"
    max_det: int = 300
    classes: Optional[Tuple[int, ...]] = None
    save: bool = True
    save_txt: bool = False
    save_conf: bool = False
    show_labels: bool = True
    show_conf: bool = True
    line_width: Optional[int] = None
    verbose: bool = False
    max_images: int = 0
    project: Path = REPO_ROOT / "runs" / "pcb_yolo"
    name: str = "predict"
    exist_ok: bool = True
    report_dir: Path = REPO_ROOT / "runs" / "pcb_yolo" / "predict_report"
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_predict_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "device": self.device,
            "max_det": self.max_det,
            "save": self.save,
            "save_txt": self.save_txt,
            "save_conf": self.save_conf,
            "show_labels": self.show_labels,
            "show_conf": self.show_conf,
            "verbose": self.verbose,
            "project": str(self.project),
            "name": self.name,
            "exist_ok": self.exist_ok,
        }
        if self.classes is not None:
            kwargs["classes"] = list(self.classes)
        if self.line_width is not None:
            kwargs["line_width"] = self.line_width
        kwargs.update(dict(self.extra))
        return kwargs

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
        weights_override: Optional[Path] = None,
        source_override: Optional[Path] = None,
    ) -> "PredictSettings":
        predict_map = _require_mapping(config.get("predict"), "predict")
        known = (
            "imgsz", "conf", "iou", "device", "max_det", "classes",
            "save", "save_txt", "save_conf", "show_labels", "show_conf",
            "line_width", "verbose", "max_images", "project", "name", "exist_ok",
        )
        extra = _split_extra(predict_map, known, "predict")
        report = _require_mapping(config.get("report", {}), "report")
        _check_keys(report, ("dir",), "report")

        if weights_override is not None:
            weights_path = Path(weights_override).expanduser()
            weights = (
                weights_path.resolve()
                if weights_path.is_absolute()
                else (Path.cwd() / weights_path).resolve()
            )
        elif config.get("weights") is not None:
            weights = resolve_path(config.get("weights"), "weights")
        else:
            weights = None

        if source_override is not None:
            source_path = Path(source_override).expanduser()
            source = (
                source_path.resolve()
                if source_path.is_absolute()
                else (Path.cwd() / source_path).resolve()
            )
        elif config.get("source") is not None:
            source = resolve_path(config.get("source"), "source")
        else:
            source = None

        line_width = predict_map.get("line_width")
        return cls(
            weights=weights,
            source=source,
            imgsz=_as_int(predict_map.get("imgsz", 640), "predict.imgsz", minimum=32),
            conf=_as_float(predict_map.get("conf", 0.25), "predict.conf", 0.0, 1.0),
            iou=_as_float(predict_map.get("iou", 0.45), "predict.iou", 0.0, 1.0),
            device=_as_device(predict_map.get("device", "0"), "predict.device"),
            max_det=_as_int(predict_map.get("max_det", 300), "predict.max_det", minimum=1),
            classes=_as_int_list(predict_map.get("classes"), "predict.classes", minimum=0),
            save=_as_bool(predict_map.get("save", True), "predict.save"),
            save_txt=_as_bool(predict_map.get("save_txt", False), "predict.save_txt"),
            save_conf=_as_bool(predict_map.get("save_conf", False), "predict.save_conf"),
            show_labels=_as_bool(predict_map.get("show_labels", True), "predict.show_labels"),
            show_conf=_as_bool(predict_map.get("show_conf", True), "predict.show_conf"),
            line_width=None
            if line_width is None
            else _as_int(line_width, "predict.line_width", minimum=0),
            verbose=_as_bool(predict_map.get("verbose", False), "predict.verbose"),
            max_images=_as_int(predict_map.get("max_images", 0), "predict.max_images", minimum=0),
            project=resolve_path(predict_map.get("project", "runs/pcb_yolo"), "predict.project"),
            name=_as_str(predict_map.get("name", "predict"), "predict.name"),
            exist_ok=_as_bool(predict_map.get("exist_ok", True), "predict.exist_ok"),
            report_dir=resolve_path(report.get("dir", "runs/pcb_yolo/predict_report"), "report.dir"),
            extra=extra,
        )


@dataclass(frozen=True)
class PredictConfig:
    """完整推理配置：权重、输入路径与推理参数。"""

    source: Path
    settings: PredictSettings
    snapshot: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        path: Path = DEFAULT_PREDICT_CONFIG,
        weights: Optional[Path] = None,
        source: Optional[Path] = None,
        overrides: Sequence[str] = (),
    ) -> "PredictConfig":
        config_path = Path(path)
        raw = apply_overrides(load_yaml(config_path), overrides)
        _check_keys(raw, ("weights", "source", "predict", "report"), "推理配置")
        settings = PredictSettings.from_mapping(
            raw, weights_override=weights, source_override=source
        )
        return cls(
            source=config_path,
            settings=settings,
            snapshot=copy.deepcopy(raw),
        )


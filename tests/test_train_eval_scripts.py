# -*- coding: utf-8 -*-
"""训练脚本、评估脚本与共享配置模块的测试。

覆盖范围：
- 配置加载、点号路径覆盖、非法取值校验、数据集配置检查；
- 从 Ultralytics 指标对象提取总体/逐类指标、混淆矩阵、最差图像；
- 阈值判定与 metrics.json / per_class.csv / report.md 报告输出；
- evaluate() 全流程（注入假模型，不加载真实权重、不训练）；
- train() 编排：训练参数透传、best.pt 选择、训练后评估、train_summary.json；
- 两个脚本的 --dry-run 命令行入口。
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import evaluate_yolo as ev
import pcb_config as cfg
import train_yolo as tr

NAMES = {0: "missing_hole", 1: "mouse_bite", 2: "short"}


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------
def write_dataset(tmp_path: Path, names=NAMES, splits=("train", "val")) -> Path:
    root = tmp_path / "dataset"
    for split in splits:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        (root / "images" / split / f"{split}_01.jpg").write_bytes(b"fake-image")
        (root / "labels" / split / f"{split}_01.txt").write_text("", encoding="utf-8")
    data = root / "data.yaml"
    data.write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "nc": len(names),
                "names": names,
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return data


def write_train_config(tmp_path: Path, data_path: Path, evaluate: bool = True) -> Path:
    payload = {
        "dataset": {"data": str(data_path)},
        "model": {"weights": "yolov8n.pt"},
        "train": {
            "epochs": 2,
            "imgsz": 320,
            "batch": 2,
            "workers": 0,
            "device": "cpu",
            "project": str(tmp_path / "runs"),
            "name": "unit",
            "exist_ok": True,
        },
        "evaluation": {
            "enabled": evaluate,
            "config": str(tmp_path / "eval.yaml"),
            "strict": False,
        },
    }
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


def write_eval_config(tmp_path: Path, data_path: Path, strict: bool = True) -> Path:
    payload = {
        "dataset": {"data": str(data_path), "split": "val"},
        "weights": str(tmp_path / "missing.pt"),
        "eval": {
            "imgsz": 320,
            "batch": 2,
            "device": "cpu",
            "workers": 0,
            "conf": 0.001,
            "project": str(tmp_path / "runs"),
            "name": "eval",
            "exist_ok": True,
        },
        "thresholds": {
            "enabled": True,
            "map50": 0.5,
            "map50_95": 0.2,
            "precision": 0.4,
            "recall": 0.4,
            "per_class_map50": 0.3,
            "min_classes": 2,
        },
        "report": {"dir": str(tmp_path / "report"), "worst_images": 2},
        "strict": strict,
    }
    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


class _FakeBox:
    def __init__(self) -> None:
        self.ap_class_index = [0, 1]
        self.all_ap = [
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.65, 0.4, 0.3, 0.2, 0.1],
            [0.8, 0.7, 0.6, 0.5, 0.4, 0.55, 0.35, 0.25, 0.15, 0.05],
        ]
        self.p = [0.9, 0.8]
        self.r = [0.85, 0.75]
        self.ap50 = [0.88, 0.78]
        self.ap = [0.6, 0.5]
        self.mp = 0.85
        self.mr = 0.80
        self.map50 = 0.83
        self.map = 0.55
        self.map75 = 0.60

    def class_result(self, index: int):
        return self.p[index], self.r[index], self.ap50[index], self.ap[index]


class _FakeMetrics:
    def __init__(self, save_dir: Path | None = None) -> None:
        self.box = _FakeBox()
        self.names = dict(NAMES)
        self.results_dict = {
            "metrics/precision(B)": 0.85,
            "metrics/recall(B)": 0.80,
            "metrics/mAP50(B)": 0.83,
            "metrics/mAP50-95(B)": 0.55,
            "fitness": 0.70,
        }
        self.speed = {"preprocess": 1.0, "inference": 2.0, "loss": 0.0, "postprocess": 0.5}
        self.image_metrics = {
            "a.jpg": {"tp": 2, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
            "b.jpg": {"tp": 0, "fp": 1, "fn": 1, "precision": 0.0, "recall": 0.0, "f1": 0.0},
            "c.jpg": {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
        }
        self.confusion_matrix = SimpleNamespace(
            matrix=[[3, 1, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]
        )
        self.nt_per_class = [5, 3, 0]
        self.nt_per_image = [3, 2, 0]
        self.save_dir = Path(save_dir) if save_dir else None
        self.fitness = 0.70


class _FakeYolo:
    def __init__(self, metrics: _FakeMetrics, names=None) -> None:
        self._metrics = metrics
        self.names = dict(names if names is not None else NAMES)
        self.val_kwargs = None

    def val(self, **kwargs):
        self.val_kwargs = kwargs
        return self._metrics


class _FakeTrainModel:
    def __init__(self, save_dir: Path, *, create_weights: bool = True) -> None:
        self.save_dir = Path(save_dir)
        self.create_weights = create_weights
        self.train_kwargs = None

    def train(self, **kwargs):
        self.train_kwargs = kwargs
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.create_weights:
            weights = self.save_dir / "weights"
            weights.mkdir(exist_ok=True)
            (weights / "best.pt").write_bytes(b"best")
            (weights / "last.pt").write_bytes(b"last")
        return SimpleNamespace(
            save_dir=self.save_dir, results_dict={"metrics/mAP50(B)": 0.77}
        )


# ---------------------------------------------------------------------------
# 配置层
# ---------------------------------------------------------------------------
def test_train_config_defaults():
    config = cfg.TrainConfig.load(cfg.DEFAULT_TRAIN_CONFIG)
    assert config.settings.epochs == 100
    assert config.settings.imgsz == 640
    assert config.settings.data == (cfg.REPO_ROOT / "datasets" / "pcb_yolo" / "data.yaml").resolve()
    assert config.evaluation_enabled is True
    assert config.evaluation_config == cfg.DEFAULT_EVAL_CONFIG
    assert config.settings.to_train_kwargs()["project"] == str(
        (cfg.REPO_ROOT / "runs" / "pcb_yolo").resolve()
    )


def test_train_config_applies_overrides():
    config = cfg.TrainConfig.load(
        cfg.DEFAULT_TRAIN_CONFIG,
        overrides=["train.epochs=3", "train.batch=4", "evaluation.enabled=false"],
    )
    assert config.settings.epochs == 3
    assert config.settings.batch == 4
    assert config.evaluation_enabled is False


@pytest.mark.parametrize(
    "overrides",
    [
        ["train.epochs=0"],
        ["train.batch=abc"],
        ["train.unknown_option=1"],
        ["unknown.section=1"],
        ["train.epochs"],
    ],
)
def test_train_config_rejects_invalid_values(overrides):
    with pytest.raises(cfg.ConfigError):
        cfg.TrainConfig.load(cfg.DEFAULT_TRAIN_CONFIG, overrides=overrides)


def test_eval_config_cli_weights_and_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("best.pt").write_bytes(b"fake")
    config = cfg.EvalConfig.load(
        cfg.DEFAULT_EVAL_CONFIG,
        weights=Path("best.pt"),
        overrides=["dataset.split=test", "thresholds.map50=0.55", "strict=false"],
    )
    assert config.settings.weights == (tmp_path / "best.pt").resolve()
    assert config.settings.split == "test"
    assert config.settings.thresholds.map50 == pytest.approx(0.55)
    assert config.strict is False


@pytest.mark.parametrize(
    "overrides",
    [
        ["thresholds.map50=1.5"],
        ["eval.conf=-0.1"],
        ["thresholds.min_classes=-1"],
        ["eval.unknown_option=1"],
        ["report.dir"],
    ],
)
def test_eval_config_rejects_invalid_values(overrides):
    with pytest.raises(cfg.ConfigError):
        cfg.EvalConfig.load(cfg.DEFAULT_EVAL_CONFIG, overrides=overrides)


def test_apply_overrides_parses_yaml_scalars():
    merged = cfg.apply_overrides(
        {"train": {"epochs": 1, "cache": False}}, ["train.epochs=5", "train.cache=true"]
    )
    assert merged == {"train": {"epochs": 5, "cache": True}}


def test_validate_dataset_yaml_ok(tmp_path):
    data_path = write_dataset(tmp_path)
    loaded = cfg.validate_dataset_yaml(data_path)
    assert cfg.normalize_class_names(loaded["names"]) == NAMES


def test_validate_dataset_yaml_missing_split(tmp_path):
    data_path = write_dataset(tmp_path, splits=("train",))
    with pytest.raises(cfg.ConfigError, match="val"):
        cfg.validate_dataset_yaml(data_path)


def test_validate_dataset_yaml_empty_split(tmp_path):
    data_path = write_dataset(tmp_path)
    for image in (tmp_path / "dataset" / "images" / "val").iterdir():
        image.unlink()
    with pytest.raises(cfg.ConfigError, match="未找到图像"):
        cfg.validate_dataset_yaml(data_path)


def test_validate_dataset_yaml_nc_mismatch(tmp_path):
    data_path = write_dataset(tmp_path)
    payload = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    payload["nc"] = 5
    data_path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    with pytest.raises(cfg.ConfigError, match="nc"):
        cfg.validate_dataset_yaml(data_path)


def test_normalize_class_names_variants():
    assert cfg.normalize_class_names({2: "c", 0: "a", 1: "b"}) == {0: "a", 1: "b", 2: "c"}
    assert cfg.normalize_class_names(["a", "b"]) == {0: "a", 1: "b"}
    assert cfg.normalize_class_names(None) == {}


# ---------------------------------------------------------------------------
# 评估层
# ---------------------------------------------------------------------------
def _collect_fake(tmp_path: Path) -> ev.EvalResult:
    save_dir = tmp_path / "val"
    save_dir.mkdir(exist_ok=True)
    (save_dir / "confusion_matrix.png").write_bytes(b"png")
    return ev.collect_metrics(
        _FakeMetrics(save_dir),
        weights=tmp_path / "best.pt",
        data=tmp_path / "data.yaml",
        split="val",
        worst_limit=2,
    )


def test_collect_metrics_extracts_overall_per_class_and_worst_images(tmp_path):
    result = _collect_fake(tmp_path)

    assert result.overall["map50"] == pytest.approx(0.83)
    assert result.overall["map75"] == pytest.approx(0.60)
    assert result.overall["map50_95"] == pytest.approx(0.55)
    assert result.overall["f1"] == pytest.approx(2 * 0.85 * 0.80 / 1.65)

    assert [row.name for row in result.per_class] == ["missing_hole", "mouse_bite"]
    first = result.per_class[0]
    assert first.index == 0
    assert first.instances == 5
    assert first.images == 3
    assert first.map50 == pytest.approx(0.88)
    assert first.map75 == pytest.approx(0.65)  # all_ap[0][5]
    assert first.map50_95 == pytest.approx(0.60)

    assert result.missing_classes == ["short"]
    assert result.images == 3
    assert result.worst_images[0]["image"] == "b.jpg"
    assert [row["image"] for row in result.worst_images] == ["b.jpg", "a.jpg"]
    assert all(row["image"] != "c.jpg" for row in result.worst_images)  # 无 TP/FP/FN 的图片被跳过
    assert result.confusion_labels == ["missing_hole", "mouse_bite", "short", "background"]
    assert result.save_dir == tmp_path / "val"
    assert result.generated_at


def test_collect_metrics_supports_numpy_like_arrays(tmp_path):
    class _NoBoolArray:
        """模拟 numpy 数组：布尔判断会抛 ValueError。"""

        def __init__(self, values):
            self._values = list(values)

        def __iter__(self):
            return iter(self._values)

        def __len__(self):
            return len(self._values)

        def __bool__(self):
            raise ValueError("The truth value of an array with more than one element is ambiguous")

    metrics = _FakeMetrics()
    metrics.nt_per_class = _NoBoolArray([4, 2, 0])
    metrics.nt_per_image = _NoBoolArray([2, 1, 0])

    result = ev.collect_metrics(
        metrics, weights=tmp_path / "w.pt", data=tmp_path / "d.yaml", split="val"
    )
    assert [row.instances for row in result.per_class] == [4, 2]
    assert [row.images for row in result.per_class] == [2, 1]


def test_collect_metrics_reads_image_metrics_from_box(tmp_path):
    metrics = _FakeMetrics()
    metrics.image_metrics = {}
    metrics.box.image_metrics = {
        "x.jpg": {"tp": 1, "fp": 2, "fn": 0, "precision": 0.33, "recall": 1.0, "f1": 0.5}
    }

    result = ev.collect_metrics(
        metrics, weights=tmp_path / "w.pt", data=tmp_path / "d.yaml", split="val"
    )
    assert result.images == 1
    assert result.worst_images[0]["image"] == "x.jpg"


def test_check_thresholds_pass_fail_and_disabled(tmp_path):
    result = _collect_fake(tmp_path)

    passing = ev.check_thresholds(
        result,
        ev.ThresholdSettings(map50=0.5, map50_95=0.2, precision=0.4, recall=0.4, per_class_map50=0.3, min_classes=2),
    )
    assert passing and all(check.passed for check in passing)
    assert any(check.name == "missing_hole mAP@0.5" for check in passing)

    failing = ev.check_thresholds(
        result,
        ev.ThresholdSettings(map50=0.9, map50_95=0.9, precision=0.9, recall=0.9, per_class_map50=0.0, min_classes=2),
    )
    assert any(not check.passed for check in failing)

    disabled = ev.check_thresholds(result, ev.ThresholdSettings(enabled=False, map50=0.99))
    assert len(disabled) == 1
    assert disabled[0].passed is True
    assert disabled[0].actual is None


def test_write_reports_creates_json_csv_and_markdown(tmp_path):
    result = _collect_fake(tmp_path)
    result.checks = ev.check_thresholds(
        result, ev.ThresholdSettings(map50=0.5, map50_95=0.2, precision=0.4, recall=0.4, per_class_map50=0.3, min_classes=2)
    )
    result.passed = all(check.passed for check in result.checks)

    paths = ev.write_reports(result, tmp_path / "report")
    assert set(paths) == {"metrics", "per_class", "report"}
    assert all(path.is_file() for path in paths.values())

    payload = json.loads(paths["metrics"].read_text(encoding="utf-8"))
    assert payload["overall"]["map50"] == pytest.approx(0.83)
    assert payload["thresholds"]["passed"] is True
    assert payload["missing_classes"] == ["short"]
    assert "confusion_matrix.png" in payload["artifacts"]
    assert payload["reports"]["report"].endswith("report.md")

    csv_text = paths["per_class"].read_text(encoding="utf-8-sig")
    assert csv_text.splitlines()[0].startswith("index,name,instances,images")
    assert "missing_hole" in csv_text

    report_text = paths["report"].read_text(encoding="utf-8")
    assert "# PCB 缺陷检测评估报告" in report_text
    assert "mAP@0.5" in report_text
    assert "missing_hole" in report_text
    assert "待复查图像" in report_text
    assert "| 已评估类别数 | 2 | 2 | 通过 |" in report_text


def test_evaluate_end_to_end_with_fake_model(tmp_path):
    data_path = write_dataset(tmp_path)
    eval_path = write_eval_config(tmp_path, data_path)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")

    save_dir = tmp_path / "val"
    save_dir.mkdir()
    (save_dir / "PR_curve.png").write_bytes(b"png")
    fake_model = _FakeYolo(_FakeMetrics(save_dir))

    config = cfg.EvalConfig.load(eval_path, weights=weights)
    result = ev.evaluate(config, model_loader=lambda _: fake_model, printer=lambda *_: None)

    assert fake_model.val_kwargs["data"] == str(data_path.resolve())
    assert fake_model.val_kwargs["split"] == "val"
    assert fake_model.val_kwargs["imgsz"] == 320
    assert fake_model.val_kwargs["conf"] == pytest.approx(0.001)
    assert "half" not in fake_model.val_kwargs
    assert result.passed is True
    assert result.report_paths["report"].is_file()
    assert result.report_paths["metrics"].is_file()
    assert "PR_curve.png" in result.artifacts


def test_evaluate_rejects_class_name_mismatch(tmp_path):
    data_path = write_dataset(tmp_path)
    eval_path = write_eval_config(tmp_path, data_path)
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")

    fake_model = _FakeYolo(_FakeMetrics(), names={0: "wrong_name", 1: "mouse_bite", 2: "short"})
    config = cfg.EvalConfig.load(eval_path, weights=weights)
    with pytest.raises(cfg.ConfigError, match="类别定义不一致"):
        ev.evaluate(config, model_loader=lambda _: fake_model, printer=lambda *_: None)


def test_evaluate_rejects_missing_weights(tmp_path):
    data_path = write_dataset(tmp_path)
    eval_path = write_eval_config(tmp_path, data_path)
    config = cfg.EvalConfig.load(eval_path)
    with pytest.raises(cfg.ConfigError, match="权重文件不存在"):
        ev.evaluate(config, model_loader=lambda _: _FakeYolo(_FakeMetrics()), printer=lambda *_: None)


# ---------------------------------------------------------------------------
# 训练层
# ---------------------------------------------------------------------------
def test_train_runs_training_then_evaluates_best_weights(tmp_path):
    data_path = write_dataset(tmp_path)
    train_path = write_train_config(tmp_path, data_path, evaluate=True)
    config = cfg.TrainConfig.load(train_path)

    model = _FakeTrainModel(tmp_path / "runs" / "unit")
    calls = []

    def fake_evaluator(config_path, *, weights, strict, printer=print):
        calls.append({"config": Path(config_path), "weights": Path(weights), "strict": strict})
        return SimpleNamespace(passed=True, generated_at="2025-01-01T00:00:00+00:00")

    result = tr.train(
        config, model_factory=lambda _: model, evaluator=fake_evaluator, printer=lambda *_: None
    )

    assert model.train_kwargs["epochs"] == 2
    assert model.train_kwargs["imgsz"] == 320
    assert model.train_kwargs["data"] == str(data_path.resolve())
    assert model.train_kwargs["project"] == str((tmp_path / "runs").resolve())
    assert model.train_kwargs["exist_ok"] is True

    assert len(calls) == 1
    assert calls[0]["weights"] == tmp_path / "runs" / "unit" / "weights" / "best.pt"
    assert calls[0]["config"] == tmp_path / "eval.yaml"
    assert calls[0]["strict"] is False
    assert result.best_weights == calls[0]["weights"]
    assert result.evaluation_passed is True

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["final_metrics"]["metrics/mAP50(B)"] == pytest.approx(0.77)
    assert summary["best_weights"].endswith("best.pt")
    assert summary["evaluation_passed"] is True
    assert summary["config"]["train"]["epochs"] == 2


def test_train_skips_evaluation_when_disabled(tmp_path):
    data_path = write_dataset(tmp_path)
    train_path = write_train_config(tmp_path, data_path, evaluate=False)
    config = cfg.TrainConfig.load(train_path)
    model = _FakeTrainModel(tmp_path / "runs" / "unit")

    def fail_if_called(**kwargs):  # pragma: no cover - 不应被调用
        raise AssertionError("evaluation.enabled=false 时不应执行评估")

    result = tr.train(
        config, model_factory=lambda _: model, evaluator=fail_if_called, printer=lambda *_: None
    )

    assert result.evaluation is None
    assert result.evaluation_passed is None
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["evaluation_passed"] is None


def test_train_requires_weights_after_training(tmp_path):
    data_path = write_dataset(tmp_path)
    train_path = write_train_config(tmp_path, data_path, evaluate=True)
    config = cfg.TrainConfig.load(train_path)
    model = _FakeTrainModel(tmp_path / "runs" / "unit", create_weights=False)

    with pytest.raises(cfg.ConfigError, match="best.pt"):
        tr.train(config, model_factory=lambda _: model, printer=lambda *_: None)


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------
def test_train_cli_dry_run(capsys):
    code = tr.main(["--dry-run", "--no-evaluate", "--epochs", "2", "--set", "train.batch=4"])
    out = capsys.readouterr().out
    assert code == 0
    assert "epochs=2" in out
    assert "batch=4" in out
    assert "训练后评估: False" in out


def test_eval_cli_dry_run(capsys):
    code = ev.main(["--dry-run", "--set", "thresholds.map50=0.66"])
    out = capsys.readouterr().out
    assert code == 0
    assert "map50=0.66" in out
    assert "strict: True" in out


def test_eval_cli_missing_weights_returns_error(tmp_path):
    data_path = write_dataset(tmp_path)
    eval_path = write_eval_config(tmp_path, data_path)
    code = ev.main(["--config", str(eval_path), "--no-strict"])
    assert code == 1




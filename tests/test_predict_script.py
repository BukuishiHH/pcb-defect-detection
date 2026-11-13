# -*- coding: utf-8 -*-
"""推理脚本与推理配置的测试。

覆盖范围：
- ``PredictConfig`` 默认值、``--set`` 覆盖、CLI 路径解析与非法取值校验；
- 推理输入解析：单张图像、图像目录（后缀过滤 + 排序 + 数量限制）、图像列表 txt；
- ``predict()`` 全流程（注入假模型，不加载真实权重、不启动 GPU）：参数透传、
  结果提取、带框图定位、``predictions.json`` 输出与摘要打印；
- 命令行 ``--dry-run`` 与权重缺失时的错误退出码。

测试不依赖 ultralytics 与真实图像内容，全部使用 tmp_path 下的占位文件。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import pcb_config as cfg
import predict_yolo as pr

NAMES = {0: "missing_hole", 1: "mouse_bite", 2: "short"}


# ---------------------------------------------------------------------------
# 假模型：模拟 ultralytics 的 Results / Boxes 结构
# ---------------------------------------------------------------------------
class _FakeTensor:
    """模拟 torch.Tensor 的 ``tolist()`` 接口。"""

    def __init__(self, values) -> None:
        self._values = values

    def tolist(self):
        return self._values


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls) -> None:
        self.xyxy = _FakeTensor(xyxy)
        self.conf = _FakeTensor(conf)
        self.cls = _FakeTensor(cls)


class _FakeResult:
    def __init__(self, path: Path, boxes, names, save_dir=None) -> None:
        self.path = str(path)
        self.boxes = boxes
        self.names = dict(names)
        self.save_dir = Path(save_dir) if save_dir else None
        self.speed = {"preprocess": 0.5, "inference": 1.5, "postprocess": 0.2}


class _FakePredictModel:
    """假 YOLO 模型：按输入图像数量返回结果，并可写出占位带框图。"""

    def __init__(self, names=None, save_dir=None, boxes=None) -> None:
        self.names = dict(names if names is not None else NAMES)
        self._save_dir = Path(save_dir) if save_dir else None
        self._boxes = boxes
        self.predict_kwargs = None

    def predict(self, **kwargs):
        self.predict_kwargs = kwargs
        sources = list(kwargs.get("source", []))
        results = []
        for path in sources:
            boxes = self._boxes or _FakeBoxes(
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
                [0.9, 0.8],
                [0, 1],
            )
            results.append(_FakeResult(Path(path), boxes, self.names, save_dir=self._save_dir))
        if kwargs.get("save") and self._save_dir:
            self._save_dir.mkdir(parents=True, exist_ok=True)
            for path in sources:
                (self._save_dir / Path(path).name).write_bytes(b"annotated")
        return results


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------
def write_images(folder: Path, names=("a.jpg", "b.jpg")) -> list:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        path = folder / name
        path.write_bytes(b"fake-image")
        paths.append(path)
    return paths


def write_predict_config(tmp_path: Path, *, source: Path, weights: Path) -> Path:
    payload = {
        "weights": str(weights),
        "source": str(source),
        "predict": {
            "imgsz": 320,
            "conf": 0.3,
            "iou": 0.5,
            "device": "cpu",
            "save": True,
            "max_images": 0,
            "project": str(tmp_path / "runs"),
            "name": "predict",
            "exist_ok": True,
        },
        "report": {"dir": str(tmp_path / "report")},
    }
    path = tmp_path / "predict.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 配置层
# ---------------------------------------------------------------------------
def test_predict_config_defaults():
    config = cfg.PredictConfig.load(cfg.DEFAULT_PREDICT_CONFIG)
    settings = config.settings

    assert settings.imgsz == 640
    assert settings.conf == pytest.approx(0.25)
    assert settings.iou == pytest.approx(0.45)
    assert settings.save is True
    assert settings.max_images == 0
    assert settings.source is None
    assert settings.weights == (
        cfg.REPO_ROOT / "runs" / "pcb_yolo" / "train" / "weights" / "best.pt"
    ).resolve()
    assert settings.report_dir == (
        cfg.REPO_ROOT / "runs" / "pcb_yolo" / "predict_report"
    ).resolve()

    kwargs = settings.to_predict_kwargs()
    assert kwargs["project"] == str((cfg.REPO_ROOT / "runs" / "pcb_yolo").resolve())
    assert kwargs["save"] is True
    assert "classes" not in kwargs
    assert "line_width" not in kwargs


def test_predict_config_applies_overrides():
    config = cfg.PredictConfig.load(
        cfg.DEFAULT_PREDICT_CONFIG,
        overrides=[
            "predict.conf=0.4",
            "predict.save=false",
            "predict.max_images=5",
            "predict.classes=[0, 2]",
            "predict.line_width=2",
        ],
    )
    settings = config.settings

    assert settings.conf == pytest.approx(0.4)
    assert settings.save is False
    assert settings.max_images == 5
    assert settings.classes == (0, 2)

    kwargs = settings.to_predict_kwargs()
    assert kwargs["save"] is False
    assert kwargs["classes"] == [0, 2]
    assert kwargs["line_width"] == 2


def test_predict_config_cli_paths_resolve_against_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("best.pt").write_bytes(b"fake")
    Path("img.jpg").write_bytes(b"fake")

    config = cfg.PredictConfig.load(
        cfg.DEFAULT_PREDICT_CONFIG,
        weights=Path("best.pt"),
        source=Path("img.jpg"),
    )

    assert config.settings.weights == (tmp_path / "best.pt").resolve()
    assert config.settings.source == (tmp_path / "img.jpg").resolve()


@pytest.mark.parametrize(
    "overrides",
    [
        ["predict.conf=1.5"],
        ["predict.iou=-0.1"],
        ["predict.classes=abc"],
        ["predict.classes=[0, -1]"],
        ["predict.max_images=-1"],
        ["predict.unknown_option=1"],
        ["unknown.section=1"],
        ["predict.imgsz"],
    ],
)
def test_predict_config_rejects_invalid_values(overrides):
    with pytest.raises(cfg.ConfigError):
        cfg.PredictConfig.load(cfg.DEFAULT_PREDICT_CONFIG, overrides=overrides)



# ---------------------------------------------------------------------------
# 推理输入解析
# ---------------------------------------------------------------------------
def test_resolve_sources_single_image(tmp_path):
    image = write_images(tmp_path, names=("a.jpg",))[0]
    assert pr.resolve_sources(image) == [image.resolve()]


def test_resolve_sources_directory_filters_suffix_and_sorts(tmp_path):
    image_dir = tmp_path / "images"
    write_images(image_dir, names=("b.jpg", "a.jpg", "c.txt", "d.png"))

    resolved = pr.resolve_sources(image_dir)

    assert [path.name for path in resolved] == ["a.jpg", "b.jpg", "d.png"]
    assert pr.resolve_sources(image_dir, max_images=2) == resolved[:2]


def test_resolve_sources_txt_list(tmp_path):
    images = write_images(tmp_path / "images", names=("a.jpg", "b.png"))
    list_path = tmp_path / "list.txt"
    list_path.write_text(
        "# 注释行会被忽略\nimages/a.jpg\n\n" + str(images[1]) + "\n", encoding="utf-8"
    )

    assert pr.resolve_sources(list_path) == [images[0].resolve(), images[1].resolve()]


def test_resolve_sources_rejects_invalid_input(tmp_path):
    with pytest.raises(cfg.ConfigError):
        pr.resolve_sources(tmp_path / "missing.jpg")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(cfg.ConfigError):
        pr.resolve_sources(empty_dir)

    bad_list = tmp_path / "bad.txt"
    bad_list.write_text("missing.jpg\n", encoding="utf-8")
    with pytest.raises(cfg.ConfigError):
        pr.resolve_sources(bad_list)

    empty_list = tmp_path / "empty.txt"
    empty_list.write_text("# 只有注释\n", encoding="utf-8")
    with pytest.raises(cfg.ConfigError):
        pr.resolve_sources(empty_list)

    other = tmp_path / "notes.md"
    other.write_text("hello", encoding="utf-8")
    with pytest.raises(cfg.ConfigError):
        pr.resolve_sources(other)


def test_read_image_list_rejects_non_image_suffix(tmp_path):
    label = tmp_path / "label.txt"
    label.write_text("x", encoding="utf-8")
    list_path = tmp_path / "list.txt"
    list_path.write_text("label.txt\n", encoding="utf-8")

    with pytest.raises(cfg.ConfigError, match="不是支持的图像类型"):
        pr.resolve_sources(list_path)


# ---------------------------------------------------------------------------
# 推理主流程（注入假模型）
# ---------------------------------------------------------------------------
def test_predict_single_image(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    image = write_images(tmp_path / "images", names=("single.jpg",))[0]
    save_dir = tmp_path / "runs" / "predict"
    model = _FakePredictModel(save_dir=save_dir)
    config = cfg.PredictConfig.load(
        write_predict_config(tmp_path, source=image, weights=weights)
    )

    printed = []
    result = pr.predict(config, model_loader=lambda _: model, printer=printed.append)

    assert model.predict_kwargs["source"] == [str(image.resolve())]
    assert model.predict_kwargs["save"] is True
    assert model.predict_kwargs["conf"] == pytest.approx(0.3)
    assert len(result.images) == 1
    assert result.total_objects == 2
    assert result.per_class() == {"missing_hole": 1, "mouse_bite": 1}
    assert result.output_dir == save_dir
    assert result.images[0].saved == save_dir / "single.jpg"
    assert any("推理结果" in line for line in printed)


def test_predict_batch_directory(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    image_dir = tmp_path / "images"
    images = write_images(image_dir, names=("a.jpg", "b.png", "c.jpg"))
    save_dir = tmp_path / "runs" / "predict"
    model = _FakePredictModel(save_dir=save_dir)
    config = cfg.PredictConfig.load(
        write_predict_config(tmp_path, source=image_dir, weights=weights)
    )

    printed = []
    result = pr.predict(config, model_loader=lambda _: model, printer=printed.append)

    assert model.predict_kwargs["source"] == [str(path.resolve()) for path in images]
    assert [item.image.name for item in result.images] == ["a.jpg", "b.png", "c.jpg"]
    assert result.total_objects == 6
    assert result.per_class() == {"missing_hole": 3, "mouse_bite": 3}
    assert all(item.saved is not None and item.saved.is_file() for item in result.images)
    assert any("总计: 3 张图像，6 个目标" in line for line in printed)

    payload = json.loads((tmp_path / "report" / "predictions.json").read_text(encoding="utf-8"))
    assert payload["total_images"] == 3
    assert payload["total_objects"] == 6
    assert payload["per_class"] == {"missing_hole": 3, "mouse_bite": 3}
    assert payload["weights"] == str(weights.resolve())
    assert payload["output_dir"] == str(save_dir)
    assert payload["images"][0]["detections"][0]["class_name"] == "missing_hole"
    assert payload["images"][0]["detections"][0]["confidence"] == pytest.approx(0.9)
    assert payload["images"][0]["detections"][0]["xyxy"] == [1.0, 2.0, 3.0, 4.0]
    assert payload["images"][0]["speed_ms"] == pytest.approx(2.2)



def test_predict_respects_max_images(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    image_dir = tmp_path / "images"
    write_images(image_dir, names=("a.jpg", "b.jpg", "c.jpg"))
    model = _FakePredictModel(save_dir=tmp_path / "runs" / "predict")
    config = cfg.PredictConfig.load(
        write_predict_config(tmp_path, source=image_dir, weights=weights),
        overrides=["predict.max_images=2"],
    )

    result = pr.predict(config, model_loader=lambda _: model, printer=lambda *_: None)

    assert len(model.predict_kwargs["source"]) == 2
    assert len(result.images) == 2


def test_predict_without_save_skips_output_dir(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    image = write_images(tmp_path / "images", names=("a.jpg",))[0]
    model = _FakePredictModel(save_dir=tmp_path / "runs" / "predict")
    config = cfg.PredictConfig.load(
        write_predict_config(tmp_path, source=image, weights=weights),
        overrides=["predict.save=false"],
    )

    result = pr.predict(config, model_loader=lambda _: model, printer=lambda *_: None)

    assert model.predict_kwargs["save"] is False
    assert result.output_dir is None
    assert result.images[0].saved is None
    payload = json.loads((tmp_path / "report" / "predictions.json").read_text(encoding="utf-8"))
    assert payload["output_dir"] is None
    assert payload["images"][0]["saved"] is None


def test_predict_handles_image_without_detections(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    image = write_images(tmp_path / "images", names=("empty.jpg",))[0]
    model = _FakePredictModel(boxes=_FakeBoxes([], [], []))
    config = cfg.PredictConfig.load(
        write_predict_config(tmp_path, source=image, weights=weights),
        overrides=["predict.save=false"],
    )

    result = pr.predict(config, model_loader=lambda _: model, printer=lambda *_: None)

    assert result.total_objects == 0
    assert result.per_class() == {}
    assert result.images[0].detections == []


def test_predict_requires_existing_weights(tmp_path):
    image = write_images(tmp_path / "images", names=("a.jpg",))[0]
    config = cfg.PredictConfig.load(
        write_predict_config(tmp_path, source=image, weights=tmp_path / "missing.pt")
    )

    with pytest.raises(cfg.ConfigError, match="权重文件不存在"):
        pr.predict(
            config,
            model_loader=lambda _: pytest.fail("权重缺失时不应加载模型"),
            printer=lambda *_: None,
        )


def test_predict_requires_source(tmp_path):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    config = cfg.PredictConfig.load(cfg.DEFAULT_PREDICT_CONFIG, weights=weights)

    with pytest.raises(cfg.ConfigError, match="未指定 source"):
        pr.predict(
            config,
            model_loader=lambda _: pytest.fail("缺少输入时不应加载模型"),
            printer=lambda *_: None,
        )


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def test_predict_cli_dry_run(capsys):
    code = pr.main(
        ["--dry-run", "--source", "datasets/pcb_yolo/images/val", "--set", "predict.conf=0.35"]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "[dry-run]" in output
    assert "conf=0.35" in output
    assert "imgsz=640" in output
    assert "predictions" not in output


def test_predict_cli_reports_missing_weights(tmp_path, capsys):
    image = write_images(tmp_path / "images", names=("a.jpg",))[0]
    config_path = write_predict_config(
        tmp_path, source=image, weights=tmp_path / "missing.pt"
    )

    code = pr.main(["--config", str(config_path)])
    output = capsys.readouterr().out

    assert code == 1
    assert "权重文件不存在" in output


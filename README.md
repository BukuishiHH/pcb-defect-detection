# PCB 板缺陷检测系统

基于 [Ultralytics YOLOv8](https://docs.ultralytics.com/) 的 PCB 板缺陷检测系统，数据集采用北京大学 **PKU-Market-PCB**。

## 数据集

- 来源：PKU-Market-PCB（合成 PCB 缺陷数据集，693 张图像）
- 缺陷类别（6 类）：`missing_hole`、`mouse_bite`、`open_circuit`、`short`、`spur`、`spurious_copper`
- 官方划分：`train` 555 张 / `val` 138 张，共 2953 个标注框
- 原始标注：COCO 格式，打包在 `PKU-Market-PCB/raw/PKU-Market-PCB_coco/pcb_cocoanno.tar`

原始数据（`PKU-Market-PCB/`）与转换后的数据（`datasets/`）均已写入 `.gitignore`，不纳入版本管理。

## 环境准备

```bash
conda activate cv_demo
pip install -r requirements.txt

# 需要 GPU 训练时，先按 CUDA 版本安装 PyTorch（示例为 CUDA 12.1）
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
```

## 数据转换

将 COCO 标注转换为 YOLOv8 检测格式（归一化 `cls cx cy w h` 标签）：

```bash
# 默认读取 PKU-Market-PCB/raw/PKU-Market-PCB_coco，输出到 datasets/pcb_yolo
python scripts/convert_coco_to_yolo.py

# 输出目录已存在转换结果时，需显式覆盖
python scripts/convert_coco_to_yolo.py --overwrite

# 与原始数据同盘符时使用硬链接，避免重复占用约 915 MB 磁盘
python scripts/convert_coco_to_yolo.py --overwrite --mode hardlink

# 只做统计、不写文件
python scripts/convert_coco_to_yolo.py --dry-run
```

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--raw-dir` | 原始数据集目录，默认 `PKU-Market-PCB/raw/PKU-Market-PCB_coco` |
| `--output-dir` | 转换结果目录，默认 `datasets/pcb_yolo` |
| `--anno-tar` / `--anno-dir` | COCO 标注 tar 包 / 已解压的 json 目录（后者优先） |
| `--splits` | 需要转换的划分，默认 `train val` |
| `--mode` | 图像写入方式：`copy`（默认）/ `hardlink` / `symlink` |
| `--overwrite` | 清空已有的 `images/`、`labels/` 后重新生成 |
| `--dry-run` | 只统计，不写文件 |

转换结果结构：

```text
datasets/pcb_yolo/
├── images/
│   ├── train/   # 555 张
│   └── val/     # 138 张
├── labels/
│   ├── train/   # 与图像同名的 .txt 标签
│   └── val/
├── data.yaml              # Ultralytics 数据集配置（6 类）
└── conversion_summary.json
```

脚本结束时会自动校验：图像与标签一一对应、标签为 5 字段、类别索引合法、坐标在 0~1 之间、图像尺寸与 COCO 记录一致。

## 测试

测试不依赖 GPU、真实权重或约 915 MB 的真实数据：数据转换使用临时目录中的迷你 COCO 数据集，训练/评估使用假模型与假指标对象，因此可在任意机器上快速运行：

```bash
conda activate cv_demo
python -m pytest            # 全部测试
python -m pytest -v         # 显示每条用例
```

覆盖内容：

- 坐标换算与裁剪、非法框过滤、标签小数位格式
- 类别按 `category id` 排序并重映射为 0..N-1，重复/空类别校验
- 标注读取（tar 包与解压目录两种来源）、图像索引的大小写兼容
- 图像写入（复制 / 硬链接及回退）、输出目录的 `--overwrite` 保护
- 单划分转换统计（越界框、`crowd`/`ignore`、未知类别、空标注图像）
- `main()` 端到端：目录结构、标签内容、`data.yaml`、`conversion_summary.json`、`--dry-run`、缺少图像时的退出码
- `verify_output()` 对损坏标签、缺失标签、数量不一致的检测
- 真实数据集存在时，额外执行一次只读的 `--dry-run` 集成检查（不存在时自动跳过）
- 配置层：默认值、`--set` 点号路径覆盖、类型/范围/未知键校验、数据集 `data.yaml` 校验
- 指标提取：总体与逐类指标、混淆矩阵、numpy 数组兼容、F1 最低图像排序
- 阈值判定（通过 / 未通过 / 关闭）、`metrics.json` / `per_class.csv` / `report.md` 输出内容
- `evaluate()` 全流程与类别不一致、权重缺失的报错；`train()` 的训练参数透传、`best.pt` 选择、训练后评估与 `train_summary.json`
- 两个脚本的 `--dry-run` 命令行入口与错误退出码

## 模型训练

训练参数集中在 `configs/train.yaml`（数据、模型、超参数、训练后评估），脚本只做流程编排：

```bash
conda activate cv_demo

# 按 configs/train.yaml 训练（默认 100 epochs，训练后自动评估 best.pt）
python scripts/train_yolo.py

# 常用参数可直接在命令行覆盖
python scripts/train_yolo.py --epochs 200 --batch 16 --device 0
python scripts/train_yolo.py --name exp2 --no-evaluate

# 任意配置项可用 --set 覆盖（键必须已存在）
python scripts/train_yolo.py --set train.lr0=0.005 --set evaluation.strict=true

# 只打印解析后的配置，不加载模型
python scripts/train_yolo.py --dry-run
```

`train` 段除常用字段外，还可直接填写 Ultralytics 支持的训练参数（如 `mosaic`、`hsv_h`、`mixup`），脚本会用 `get_cfg()` 校验后原样透传。

| 参数 | 说明 |
| --- | --- |
| `--config` | 训练配置文件，默认 `configs/train.yaml` |
| `--epochs` / `--batch` / `--device` / `--name` | 覆盖对应的 `train.*` 字段 |
| `--evaluate` / `--no-evaluate` | 训练后是否执行验证评估（覆盖 `evaluation.enabled`） |
| `--set KEY=VALUE` | 覆盖任意配置项，可重复使用；`train` 段的额外键需为 Ultralytics 合法参数 |
| `--dry-run` | 只解析并打印配置 |

训练产物：

```text
runs/pcb_yolo/train/
├── weights/
│   ├── best.pt
│   └── last.pt
├── train_summary.json    # 最终训练指标、权重路径、配置快照、评估是否通过
├── results.csv
└── ...                   # Ultralytics 生成的曲线与可视化
```

训练结束后会以 `best.pt` 自动调用评估流程（复用 `configs/eval.yaml` 的验证参数、阈值与报告目录）。当 `evaluation.strict=true` 且指标未达阈值时，训练脚本返回非 0；默认 `false` 仅记录结果。

## 验证评估

无需重新训练即可对任意权重做验证并生成报告：

```bash
conda activate cv_demo

# 使用 configs/eval.yaml 中的权重与验证参数
python scripts/evaluate_yolo.py

# 指定权重 / 划分
python scripts/evaluate_yolo.py --weights runs/pcb_yolo/train/weights/best.pt
python scripts/evaluate_yolo.py --weights runs/pcb_yolo/train/weights/best.pt --split test

# 指标未达阈值时只记录、不返回非 0
python scripts/evaluate_yolo.py --weights runs/pcb_yolo/train/weights/best.pt --no-strict

# 只解析并打印配置
python scripts/evaluate_yolo.py --dry-run
```

| 参数 | 说明 |
| --- | --- |
| `--config` | 评估配置文件，默认 `configs/eval.yaml` |
| `--weights` | 待评估权重，覆盖配置中的 `weights` |
| `--split` | 数据集划分，覆盖 `dataset.split` |
| `--set KEY=VALUE` | 覆盖任意配置项，如 `--set eval.imgsz=320` |
| `--strict` / `--no-strict` | 阈值未通过时是否返回非 0（覆盖配置中的 `strict`） |
| `--dry-run` | 只解析并打印配置 |

评估报告写入 `configs/eval.yaml` 的 `report.dir`（默认 `runs/pcb_yolo/eval_report`）：

```text
runs/pcb_yolo/eval_report/
├── metrics.json     # 总体/逐类指标、阈值检查、混淆矩阵、图表路径
├── per_class.csv    # 逐类指标表格（UTF-8 BOM，Excel 可直接打开）
└── report.md        # Markdown 报告：总体指标、逐类指标、阈值判定、推理速度、待复查图像
```

PR 曲线、F1 曲线、混淆矩阵等图表由 Ultralytics 保存在 `eval.project/eval.name`（默认 `runs/pcb_yolo/eval`），报告中会列出全部图表文件名。

阈值说明：

- `thresholds.enabled=false` 时跳过全部阈值检查；
- 单项阈值为 `0` 表示不检查该项；`per_class_map50` 只检查验证集中有实例的类别；
- `min_classes` 检查验证集覆盖的类别数，缺失类别会写入报告；
- 权重与 `data.yaml` 的类别定义不一致时直接报错，避免用错权重得到无意义指标。

`data.yaml` 未设置 `path`，Ultralytics 会以该文件所在目录作为数据集根目录，因此可在任意工作目录下直接引用。

## 目录结构

```text
pcb-defect-detection/
├── PKU-Market-PCB/                 # 原始数据集（不入库）
├── datasets/                       # 转换后的 YOLO 数据集（不入库）
├── configs/
│   ├── train.yaml                  # 训练配置
│   └── eval.yaml                   # 验证评估配置
├── scripts/
│   ├── convert_coco_to_yolo.py     # COCO -> YOLO 数据转换脚本
│   ├── pcb_config.py               # 配置加载/校验共享层
│   ├── train_yolo.py               # YOLOv8 训练脚本
│   └── evaluate_yolo.py            # 验证评估与报告脚本
├── tests/
│   ├── conftest.py                 # 迷你 COCO 数据集夹具
│   ├── test_convert_coco_to_yolo.py
│   └── test_train_eval_scripts.py  # 配置、训练/评估编排、报告与 CLI 测试
├── pytest.ini
├── requirements.txt
├── .gitignore
└── README.md
```


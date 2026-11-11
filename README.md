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

## 模型训练

```bash
yolo detect train data=datasets/pcb_yolo/data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=8 device=0
```

`data.yaml` 未设置 `path`，Ultralytics 会以该文件所在目录作为数据集根目录，因此可在任意工作目录下直接引用。

## 目录结构

```text
pcb-defect-detection/
├── PKU-Market-PCB/                 # 原始数据集（不入库）
├── datasets/                       # 转换后的 YOLO 数据集（不入库）
├── scripts/
│   └── convert_coco_to_yolo.py     # COCO -> YOLO 数据转换脚本
├── requirements.txt
├── .gitignore
└── README.md
```

# YOLOv11 木材缺陷检测训练程序

基于YOLOv11s的木材缺陷检测模型训练系统。支持自定义数据增强、TensorBoard可视化、CSV/Excel指标导出和高质量矢量图生成。

## 项目结构

```
yolo_training_programn/
├── train.py                    # 主训练脚本
├── config.py                   # 配置管理模块
├── callbacks.py                # 自定义回调函数
├── preprocessor.py             # 数据预处理脚本
├── requirements.txt            # 项目依赖
├── utils/
│   ├── __init__.py
│   └── visualizer.py          # 矢量图生成工具
├── wood_defect_yolo/          # YOLO格式数据集
│   ├── data.yaml              # 数据集配置
│   ├── images/                # 图像文件
│   │   ├── train/             # 16,220张训练图
│   │   ├── val/               # 2,027张验证图
│   │   └── test/              # 2,029张测试图
│   └── labels/                # 标签文件
│       ├── train/
│       ├── val/
│       └── test/
└── model/                     # 训练输出目录（自动创建）
```

## 数据集信息

本项目使用 **Wood Defect Detection** 开源数据集进行训练。

### 数据集来源
- **DatasetNinja**: https://datasetninja.com/wood-defect-detection
- **GitHub**: https://github.com/dataset-ninja/wood-defect-detection
- **HuggingFace**: https://huggingface.co/datasets/iluvvatar/wood_surface_defects

### 数据集统计
- **类别数**: 10种木材缺陷
- **类别列表**: Blue_stain, Crack, Death_know, Knot_missing, knot_with_crack, Live_knot, Marrow, overgrown, Quartzity, resin
- **图像尺寸**: 2800x1024 (宽屏格式)
- **总图像数**: 20,276张
- **数据划分**: 80% 训练集 / 10% 验证集 / 10% 测试集


## 快速开始

### 0. 数据预处理（首次使用）

如果你从 DatasetNinja/HuggingFace 下载的是原始格式数据集，需要先运行预处理脚本转换为 YOLO 格式。

#### 配置路径
在运行预处理脚本前，编辑 [preprocessor.py](preprocessor.py) 中的路径配置：

```python
# 第 322-323 行
SOURCE_DIR = "/path/to/your/wood-defect-detection-DatasetNinja"  # 修改为你下载的原始数据集路径
OUTPUT_DIR = "/path/to/your/wood_defect_yolo"                     # 修改为输出目录路径
```

#### 运行预处理
```bash
python3 preprocessor.py
```

该脚本将自动完成：
- 解析 DatasetNinja 格式的 annotations
- 转换为 YOLO 格式 (class x_center y_center width height)
- 按 8:1:1 划分训练集/验证集/测试集
- 生成 `data.yaml` 配置文件
- 创建完整的 `wood_defect_yolo/` 目录结构

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- ultralytics >= 8.3.0 (YOLOv11)
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- openpyxl >= 3.1.0
- tensorboard >= 2.13.0

### 2. 验证配置

```bash
python3 config.py
```

这将显示完整的训练配置摘要。

### 3. 开始训练

```bash
python3 train.py
```

训练将自动：
- 加载YOLOv11s预训练权重
- 验证数据集路径
- 开始训练并实时记录指标
- 保存最佳和最后权重
- 导出CSV/Excel训练数据
- 生成矢量图

### 4. 监控训练（可选）

在新终端窗口启动TensorBoard：

```bash
tensorboard --logdir=model/wood_defect_yolo11s
```

然后访问: http://localhost:6006

## 训练输出

训练完成后，`model/wood_defect_yolo11s/` 将包含：

### 权重文件
- `weights/best.pt` - 最佳mAP权重
- `weights/last.pt` - 最后epoch权重

### 训练指标
- `results.csv` - YOLO内置训练日志
- `training_metrics.csv` - 自定义详细指标
- `training_metrics.xlsx` - Excel格式（带摘要sheet）

### 可视化
- `training_curves.pdf` - 训练曲线矢量图（PDF格式）
- `confusion_matrix.png` - 混淆矩阵
- `F1_curve.png`, `PR_curve.png` 等（YOLO自动生成）

### 配置备份
- `args.yaml` - 完整训练参数记录

## 自定义配置

编辑 [config.py]修改训练参数：

## 常见问题

### Q: 为什么使用 rect=True？
A: 因为数据集图像是2800x1024宽屏格式（宽高比≈2.73），矩形训练可以减少padding浪费，提高GPU利用率。

### Q: 为什么关闭Mosaic增强？
A: 根据项目需求，可能是因为木材缺陷检测需要保持原始上下文信息，Mosaic会破坏图像完整性。

### Q: 如何继续训练？
A: 使用最后保存的权重继续训练：
```python
model = YOLO('model/wood_defect_yolo11s/weights/last.pt')
model.train(resume=True)
```


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

- **类别数**: 10种木材缺陷
- **类别列表**: Blue_stain, Crack, Death_know, Knot_missing, knot_with_crack, Live_knot, Marrow, overgrown, Quartzity, resin
- **图像尺寸**: 2800x1024 (宽屏格式)
- **总图像数**: 20,276张
- **数据划分**: 80% 训练集 / 10% 验证集 / 10% 测试集

## 训练配置

### 模型配置
- **模型**: YOLOv11s
- **预训练权重**: COCO数据集预训练
- **输入尺寸**: 1600 (长边)
- **矩形训练**: 启用 (rect=True)

### 训练参数
- **训练轮数**: 200 epochs
- **批次大小**: 16
- **优化器**: 自动选择 (SGD/AdamW)
- **初始学习率**: 0.01
- **混合精度**: 启用 (AMP)

### 数据增强
- **禁用**: Mosaic (mosaic=0.0)
- **几何增强**:
  - 水平翻转: 0.5
  - 垂直翻转: 0.5
  - 缩放: 0.5
  - 平移: 0.1
- **颜色增强 (HSV)**:
  - 色调: 0.015
  - 饱和度: 0.7
  - 亮度: 0.4

## 快速开始

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
- `training_curves.pdf` - 矢量图（PDF格式）
- `training_curves.svg` - 矢量图（SVG格式）
- `training_curves.png` - 高清PNG（300 DPI）
- `confusion_matrix.png` - 混淆矩阵
- `F1_curve.png`, `PR_curve.png` 等

### 配置备份
- `args.yaml` - 完整训练参数记录

## 自定义配置

编辑 [config.py](config.py:1) 修改训练参数：

```python
@dataclass
class TrainingConfig:
    epochs: int = 200          # 修改训练轮数
    batch: int = 16            # 修改批次大小
    imgsz: int = 1600          # 修改输入尺寸

    # 数据增强参数
    fliplr: float = 0.5        # 水平翻转概率
    scale: float = 0.5         # 缩放范围
    # ... 更多参数
```

## 训练时间估算

- **硬件**: RTX 5080 (16GB)
- **每epoch**: 约15-20分钟
- **200 epochs**: 约50-65小时

建议使用 `screen` 或 `tmux` 进行长时间训练。

## 显存优化

如果遇到CUDA OOM错误：

1. 减小批次大小：
   ```python
   batch: int = 8  # 或 4
   ```

2. 降低输入尺寸：
   ```python
   imgsz: int = 1280  # 或 1024
   ```

3. 禁用混合精度：
   ```python
   amp: bool = False
   ```

## 推理使用

训练完成后，使用最佳权重进行推理：

```python
from ultralytics import YOLO

# 加载最佳权重
model = YOLO('model/wood_defect_yolo11s/weights/best.pt')

# 在测试集上评估
results = model.val(data='wood_defect_yolo/data.yaml', split='test')

# 单张图像推理
results = model.predict('path/to/image.bmp', imgsz=1600)

# 批量推理
results = model.predict('wood_defect_yolo/images/test/', save=True)
```

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

## 项目特点

- **模块化设计**: 配置、训练、可视化职责分离
- **数据安全**: 实时CSV保存，防止训练中断数据丢失
- **多格式输出**: CSV、Excel、PDF、SVG满足不同需求
- **针对性优化**: rect=True专为2800x1024宽屏优化
- **完整监控**: TensorBoard + 自定义日志 + 矢量图
- **生产级质量**: 错误处理、日志记录、参数备份

## 参考资源

- [Ultralytics YOLOv11文档](https://docs.ultralytics.com/)
- [训练配置指南](https://docs.ultralytics.com/modes/train/)
- [数据增强文档](https://docs.ultralytics.com/guides/yolo-data-augmentation/)
- [TensorBoard集成](https://docs.ultralytics.com/integrations/tensorboard/)

## 许可证

本项目仅用于学术研究和教育目的。

## 作者

训练系统设计: Claude Code
日期: 2025

"""
YOLOv11s木材缺陷检测训练配置模块

此模块集中管理所有训练参数，包括：
- 路径配置
- 训练超参数
- 数据增强策略
- 优化器配置
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict


@dataclass
class TrainingConfig:
    """YOLOv11木材缺陷检测训练配置"""
  
    # ==================== 实验配置 ====================
    name: str = "wood_defect_yolo11s"  # 实验名称
    exist_ok: bool = False  # 是否覆盖已存在的实验

    # ==================== 路径配置 ====================
    project_root: Path = Path("/data/chx25/yolo_training_programn")
    data_yaml: Path = Path("/data/chx25/yolo_training_programn/wood_defect_yolo/data.yaml")
    output_dir: Path = Path("/data/chx25/yolo_training_programn/model")
    pretrained_model: str = "yolo11s.pt"  # COCO预训练权重

    # ==================== 基础训练参数 ====================
    epochs: int = 300  # 总训练轮数
    batch: float = 24 
    imgsz: int = 1600 # 适配2800x1024高分辨率图像
    device: int = 0  # 使用的GPU设备ID

    # ==================== 数据加载配置 ====================
    rect: bool = True  # 矩形训练 - 关键！适配2800x1024宽屏格式
    workers: int = 10  # 数据加载线程数
    cache: str = "disk"  # 缓存策略: "disk", "ram", 或 False

    # ==================== 数据增强参数 ====================
    # 禁用的增强
    mosaic: float = 0.0  # 关闭Mosaic增强

    # 启用的几何增强
    fliplr: float = 0.5  # 水平翻转概率
    flipud: float = 0.5  # 垂直翻转概率
    scale: float = 0.1   # 缩放增强范围（0.5表示0.5-1.5倍）
    translate: float = 0.1  # 平移增强
    degrees: float = 0  # 旋转增强角度（木材可能以不同角度拍摄）
    shear: float = 0.0  # 剪切增强（默认关闭，可按需开启）
    perspective: float = 0.0  # 透视增强（默认关闭）

    # 高级增强
    mixup: float = 0.0  # Mixup增强（默认关闭，可按需开启）

    # HSV颜色增强（默认参数全部开启）
    hsv_h: float = 0.015  # 色调增强范围
    hsv_s: float = 0.7    # 饱和度增强范围
    hsv_v: float = 0.4    # 亮度增强范围

    # ==================== 损失函数权重 ====================
    box: float = 7.5  # 边界框损失权重（默认值）
    cls: float = 0.5  # 分类损失权重（默认值）

    # ==================== 优化器配置 ====================
    # 注意：optimizer='auto'时，lr0/lrf/momentum会被忽略！
    optimizer: str = "auto"  # 优化器类型
    #这里预期会使用SGD优化器，所以下面显式设置了学习率等参数
    lr0: float = 0.01  # 初始学习率(1e-5, 1e-1)
    lrf: float = 0.01  # 最终学习率因子(0.01, 1.0)
    momentum: float = 0.937  # 动量(0.6, 0.98)	
    weight_decay: float = 0.0005  # L2正则化权重衰减(0.0, 0.001)
    warmup_epochs: float = 3.0  # 学习率预热轮数(0.0, 5.0)
    warmup_momentum: float = 0.8  # 预热期间的初始动量(0.0, 0.95)
    warmup_bias_lr: float = 0.1  # 预热期间的偏置学习率
    cos_lr: bool = True  # 使用余弦学习率调度器（平滑衰减，避免突然下降）

    # ==================== 训练策略 ====================
    amp: bool = True  # 混合精度训练（节省约40%显存）
    patience: int = 50  # Early stopping耐心值（0表示禁用）
  
    # ==================== 日志和可视化 ====================
    verbose: bool = True  # 详细输出
    plots: bool = True  # 生成训练曲线图
    save: bool = True  # 保存检查点
    save_period: int = -1  # 每N个epoch保存一次检查点（-1表示只保存最后）

    def to_dict(self) -> Dict:
        """
        转换为字典格式，用于传递给YOLO训练API

        Returns:
            Dict: 包含所有训练参数的字典
        """
        return {
            # 数据集配置
            'data': str(self.data_yaml),

            # 基础训练参数
            'epochs': self.epochs,
            'batch': self.batch,
            'imgsz': self.imgsz,
            'device': self.device,

            # 数据加载
            'rect': self.rect,
            'workers': self.workers,
            'cache': self.cache,

            # 数据增强
            'mosaic': self.mosaic,
            'fliplr': self.fliplr,
            'flipud': self.flipud,
            'scale': self.scale,
            'translate': self.translate,
            'degrees': self.degrees,
            'shear': self.shear,
            'perspective': self.perspective,
            'mixup': self.mixup,
            'hsv_h': self.hsv_h,
            'hsv_s': self.hsv_s,
            'hsv_v': self.hsv_v,

            # 损失函数权重
            'box': self.box,
            'cls': self.cls,

            # 优化器
            'optimizer': self.optimizer,
            'lr0': self.lr0,
            'lrf': self.lrf,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs,
            'warmup_momentum': self.warmup_momentum,
            'warmup_bias_lr': self.warmup_bias_lr,
            'cos_lr': self.cos_lr,

            # 训练策略
            'amp': self.amp,
            'patience': self.patience,

            # 日志和保存
            'verbose': self.verbose,
            'plots': self.plots,
            'save': self.save,
            'save_period': self.save_period,

            # 实验配置
            'project': str(self.output_dir),
            'name': self.name,
            'exist_ok': self.exist_ok,
        }


if __name__ == "__main__":
    # 测试配置
    config = TrainingConfig()
    print("\n转换为字典格式:")
    import pprint
    pprint.pprint(config.to_dict())

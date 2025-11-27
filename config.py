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
    epochs: int = 200
    batch: int = 16
    imgsz: int = 1600  # 适配2800x1024高分辨率图像
    device: int = 0  # RTX 5080 GPU

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
    scale: float = 0.5   # 缩放增强范围（0.5表示0.5-1.5倍）
    translate: float = 0.1  # 平移增强（±10%）

    # HSV颜色增强（默认参数全部开启）
    hsv_h: float = 0.015  # 色调增强范围
    hsv_s: float = 0.7    # 饱和度增强范围
    hsv_v: float = 0.4    # 亮度增强范围

    # ==================== 优化器配置 ====================
    optimizer: str = "auto"  # 优化器类型：auto, SGD, Adam, AdamW
    lr0: float = 0.01  # 初始学习率
    lrf: float = 0.01  # 最终学习率因子（final_lr = lr0 * lrf）
    momentum: float = 0.937  # SGD动量
    weight_decay: float = 0.0005  # L2正则化权重衰减
    warmup_epochs: float = 3.0  # 学习率预热轮数

    # ==================== 训练策略 ====================
    amp: bool = True  # 混合精度训练（节省约40%显存）
    patience: int = 100  # Early stopping耐心值（0表示禁用）

    # ==================== 日志和可视化 ====================
    verbose: bool = True  # 详细输出
    plots: bool = True  # 生成训练曲线图
    save: bool = True  # 保存检查点
    save_period: int = 1  # 每N个epoch保存一次检查点（-1表示只保存最后）

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
            'hsv_h': self.hsv_h,
            'hsv_s': self.hsv_s,
            'hsv_v': self.hsv_v,

            # 优化器
            'optimizer': self.optimizer,
            'lr0': self.lr0,
            'lrf': self.lrf,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs,

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

    def print_summary(self) -> str:
        """
        打印配置摘要

        Returns:
            str: 格式化的配置摘要
        """
        summary = f"""
{'='*80}
YOLOv11 木材缺陷检测训练配置摘要
{'='*80}

[数据集配置]
  数据集路径: {self.data_yaml}
  输出目录: {self.output_dir}
  预训练模型: {self.pretrained_model}

[训练参数]
  训练轮数: {self.epochs}
  批次大小: {self.batch}
  图像尺寸: {self.imgsz}
  设备: GPU {self.device}

[数据加载]
  矩形训练: {self.rect} {'← 适配2800x1024宽屏' if self.rect else ''}
  工作线程: {self.workers}
  缓存策略: {self.cache}

[数据增强]
  Mosaic: {self.mosaic} {'(已禁用)' if self.mosaic == 0.0 else ''}
  水平翻转: {self.fliplr}
  垂直翻转: {self.flipud}
  缩放: {self.scale}
  平移: {self.translate}
  HSV-H: {self.hsv_h}
  HSV-S: {self.hsv_s}
  HSV-V: {self.hsv_v}

[优化器]
  类型: {self.optimizer}
  初始学习率: {self.lr0}
  最终学习率因子: {self.lrf}
  动量: {self.momentum}
  权重衰减: {self.weight_decay}
  预热轮数: {self.warmup_epochs}

[训练策略]
  混合精度: {self.amp}
  Early Stopping: {'Disabled' if self.patience == 0 else f'{self.patience} epochs'}

[输出配置]
  实验名称: {self.name}
  保存周期: {'仅最后' if self.save_period == -1 else f'每{self.save_period}轮'}

{'='*80}
        """
        return summary


if __name__ == "__main__":
    # 测试配置
    config = TrainingConfig()
    print(config.print_summary())
    print("\n转换为字典格式:")
    import pprint
    pprint.pprint(config.to_dict())

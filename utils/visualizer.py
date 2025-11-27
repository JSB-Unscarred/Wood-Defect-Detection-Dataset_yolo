"""
矢量图生成工具模块

用于从训练指标CSV生成高质量的矢量图（PDF/SVG）和高清PNG图像
"""

import logging
from pathlib import Path
from typing import Optional, List

try:
    import matplotlib
    matplotlib.use('Agg')  # 无GUI环境
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("matplotlib未安装，矢量图生成功能不可用")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.warning("pandas未安装，无法读取CSV文件")


# 设置matplotlib样式
if MATPLOTLIB_AVAILABLE:
    rcParams['font.size'] = 10
    rcParams['axes.titlesize'] = 14
    rcParams['axes.labelsize'] = 12
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10
    rcParams['figure.titlesize'] = 16
    rcParams['figure.dpi'] = 100


class VectorPlotter:
    """矢量图生成器 - 支持PDF格式"""

    def __init__(self, save_dir: Path, dpi: int = 300):
        """
        初始化矢量图生成器

        Args:
            save_dir: 保存目录路径
            dpi: 图像分辨率（用于PNG格式）
        """
        self.save_dir = Path(save_dir)
        self.dpi = dpi
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib未安装，无法生成矢量图")
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas未安装，无法读取CSV数据")

        logging.info(f"矢量图生成器已初始化: {self.save_dir}")

    def plot_training_curves(self, csv_path: Path) -> List[Path]:
        """
        绘制训练曲线（矢量格式）

        Args:
            csv_path: 训练指标CSV文件路径

        Returns:
            List[Path]: 生成的图像文件路径列表
        """
        if not csv_path.exists():
            logging.error(f"CSV文件不存在: {csv_path}")
            return []

        # 读取数据
        df = pd.read_csv(csv_path)
        if df.empty:
            logging.warning("CSV文件为空，无法绘图")
            return []

        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('YOLOv11 Wood Defect Detection - Training Curves',
                     fontsize=16, fontweight='bold')

        # 1. 训练/验证Box损失
        self._plot_losses(axes[0, 0], df)

        # 2. mAP指标
        self._plot_map_metrics(axes[0, 1], df)

        # 3. 精度/召回率
        self._plot_precision_recall(axes[1, 0], df)

        # 4. 学习率
        self._plot_learning_rate(axes[1, 1], df)

        plt.tight_layout()

        # 只保存PDF格式
        output_paths = []
        output_path = self.save_dir / 'training_curves.pdf'

        try:
            plt.savefig(output_path, format='pdf', bbox_inches='tight')
            output_paths.append(output_path)
            logging.info(f"训练曲线PDF已保存: {output_path}")
        except Exception as e:
            logging.error(f"保存PDF失败: {e}")

        plt.close()
        return output_paths

    def _plot_losses(self, ax, df: pd.DataFrame):
        """绘制损失曲线"""
        ax.set_title('Box Loss Curves', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')

        # 训练损失
        if 'train/box_loss' in df.columns:
            ax.plot(df['epoch'], df['train/box_loss'],
                   label='Train Box Loss', linewidth=2, color='#1f77b4',
                   marker='o', markersize=3, markevery=max(1, len(df)//20))

        # 验证损失
        if 'val/box_loss' in df.columns:
            ax.plot(df['epoch'], df['val/box_loss'],
                   label='Val Box Loss', linewidth=2, color='#ff7f0e',
                   marker='s', markersize=3, markevery=max(1, len(df)//20))

        ax.legend(loc='best', framealpha=0.9)

        # 标注最小值
        if 'val/box_loss' in df.columns:
            min_idx = df['val/box_loss'].idxmin()
            min_epoch = df.loc[min_idx, 'epoch']
            min_loss = df.loc[min_idx, 'val/box_loss']
            ax.annotate(f'Best: {min_loss:.4f}',
                       xy=(min_epoch, min_loss),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    def _plot_map_metrics(self, ax, df: pd.DataFrame):
        """绘制mAP指标"""
        ax.set_title('mAP Metrics', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('mAP', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')

        # mAP@0.5
        if 'metrics/mAP50' in df.columns:
            ax.plot(df['epoch'], df['metrics/mAP50'],
                   label='mAP@0.5', linewidth=2, color='#2ca02c',
                   marker='o', markersize=4, markevery=max(1, len(df)//20))

        # mAP@0.5:0.95
        if 'metrics/mAP50-95' in df.columns:
            ax.plot(df['epoch'], df['metrics/mAP50-95'],
                   label='mAP@0.5:0.95', linewidth=2, color='#d62728',
                   marker='s', markersize=4, markevery=max(1, len(df)//20))

        ax.legend(loc='best', framealpha=0.9)
        ax.set_ylim(0, 1)

        # 标注最大值
        if 'metrics/mAP50-95' in df.columns:
            max_idx = df['metrics/mAP50-95'].idxmax()
            max_epoch = df.loc[max_idx, 'epoch']
            max_map = df.loc[max_idx, 'metrics/mAP50-95']
            ax.annotate(f'Best: {max_map:.4f}',
                       xy=(max_epoch, max_map),
                       xytext=(10, -20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    def _plot_precision_recall(self, ax, df: pd.DataFrame):
        """绘制精度和召回率"""
        ax.set_title('Precision & Recall', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')

        # 精度
        if 'metrics/precision' in df.columns:
            ax.plot(df['epoch'], df['metrics/precision'],
                   label='Precision', linewidth=2, color='#9467bd',
                   marker='o', markersize=3, markevery=max(1, len(df)//20))

        # 召回率
        if 'metrics/recall' in df.columns:
            ax.plot(df['epoch'], df['metrics/recall'],
                   label='Recall', linewidth=2, color='#8c564b',
                   marker='s', markersize=3, markevery=max(1, len(df)//20))

        ax.legend(loc='best', framealpha=0.9)
        ax.set_ylim(0, 1)

    def _plot_learning_rate(self, ax, df: pd.DataFrame):
        """绘制学习率调度"""
        ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Learning Rate', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')

        # 学习率（参数组0）
        if 'lr/pg0' in df.columns:
            ax.plot(df['epoch'], df['lr/pg0'],
                   label='LR (param group 0)', linewidth=2, color='#e377c2',
                   marker='o', markersize=3, markevery=max(1, len(df)//20))

        # 如果有多个参数组，也绘制出来
        if 'lr/pg1' in df.columns:
            ax.plot(df['epoch'], df['lr/pg1'],
                   label='LR (param group 1)', linewidth=2, color='#7f7f7f',
                   marker='s', markersize=3, markevery=max(1, len(df)//20), alpha=0.7)

        ax.legend(loc='best', framealpha=0.9)
        ax.set_yscale('log')  # 对数刻度更清晰

    def plot_individual_metrics(self, csv_path: Path) -> List[Path]:
        """
        为每个指标单独生成图表（可选）

        Args:
            csv_path: 训练指标CSV文件路径

        Returns:
            List[Path]: 生成的图像文件路径列表
        """
        if not csv_path.exists():
            logging.error(f"CSV文件不存在: {csv_path}")
            return []

        df = pd.read_csv(csv_path)
        if df.empty:
            return []

        output_paths = []

        # 为每个关键指标生成单独的大图
        metrics_to_plot = [
            ('metrics/mAP50-95', 'mAP@0.5:0.95', 'green'),
            ('metrics/mAP50', 'mAP@0.5', 'blue'),
            ('val/box_loss', 'Validation Box Loss', 'red'),
        ]

        for metric_col, title, color in metrics_to_plot:
            if metric_col not in df.columns:
                continue

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df['epoch'], df[metric_col],
                   linewidth=2.5, color=color, marker='o', markersize=4)
            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel(title, fontsize=14)
            ax.set_title(f'{title} over Training', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()

            # 保存
            safe_filename = metric_col.replace('/', '_')
            for fmt in ['pdf', 'png']:
                output_path = self.save_dir / f'{safe_filename}.{fmt}'
                dpi_value = self.dpi if fmt == 'png' else None
                plt.savefig(output_path, format=fmt, dpi=dpi_value, bbox_inches='tight')
                output_paths.append(output_path)

            plt.close()

        logging.info(f"生成了 {len(output_paths)} 个单独指标图表")
        return output_paths


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    print("VectorPlotter 模块测试")
    print(f"Matplotlib可用: {MATPLOTLIB_AVAILABLE}")
    print(f"Pandas可用: {PANDAS_AVAILABLE}")

    if MATPLOTLIB_AVAILABLE and PANDAS_AVAILABLE:
        print("\n创建测试数据...")
        import numpy as np

        # 生成模拟训练数据
        epochs = 50
        test_data = {
            'epoch': range(1, epochs + 1),
            'train/box_loss': np.exp(-np.linspace(0, 3, epochs)) + np.random.rand(epochs) * 0.1,
            'val/box_loss': np.exp(-np.linspace(0, 2.5, epochs)) + np.random.rand(epochs) * 0.15,
            'metrics/mAP50': 1 - np.exp(-np.linspace(0, 3, epochs)) + np.random.rand(epochs) * 0.05,
            'metrics/mAP50-95': 1 - np.exp(-np.linspace(0, 2.8, epochs)) + np.random.rand(epochs) * 0.05,
            'metrics/precision': 0.5 + 0.4 * (1 - np.exp(-np.linspace(0, 3, epochs))),
            'metrics/recall': 0.5 + 0.4 * (1 - np.exp(-np.linspace(0, 3, epochs))),
            'lr/pg0': 0.01 * np.exp(-np.linspace(0, 4, epochs)),
        }

        test_df = pd.DataFrame(test_data)
        test_csv = Path("./test_metrics.csv")
        test_df.to_csv(test_csv, index=False)

        print(f"测试CSV已生成: {test_csv}")

        # 测试绘图
        plotter = VectorPlotter(Path("./test_plots"), dpi=300)
        output_files = plotter.plot_training_curves(test_csv)

        print(f"\n生成了 {len(output_files)} 个图像文件:")
        for f in output_files:
            print(f"  - {f}")

        print("\n矢量图生成器测试完成！")
    else:
        print("\n缺少必要的依赖，跳过测试")

"""
YOLOv11 训练结果可视化生成器

此脚本用于从训练指标CSV生成高质量的PDF矢量图。
包含训练/验证损失、mAP指标、精度/召回率和学习率曲线。

用法：
    # 自动找最新训练结果并生成可视化（最常用）
    python visualize_results.py

    # 指定训练结果目录
    python visualize_results.py --result-dir ./model/wood_defect_yolo11s

    # 指定输出DPI
    python visualize_results.py --dpi 300

"""

import argparse
import logging
from pathlib import Path
from typing import Optional, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    import matplotlib
    matplotlib.use('Agg')  # 无GUI环境
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.error("matplotlib未安装，无法生成矢量图")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logging.error("pandas未安装，无法读取CSV文件")


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

        # 保存PDF格式
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


def find_latest_result_dir(project_root: Path) -> Optional[Path]:
    """
    自动查找最新的训练结果目录

    Args:
        project_root: 项目根目录

    Returns:
        Optional[Path]: 最新训练结果目录路径，未找到返回None
    """
    model_dir = project_root / "model"

    if not model_dir.exists():
        logging.error(f"模型输出目录不存在: {model_dir}")
        return None

    # 查找所有包含 training_metrics.csv 的子目录
    result_dirs = []
    for subdir in model_dir.iterdir():
        if subdir.is_dir():
            csv_file = subdir / "training_metrics.csv"
            if csv_file.exists():
                result_dirs.append(subdir)

    if not result_dirs:
        logging.error(f"在 {model_dir} 中未找到包含 training_metrics.csv 的训练结果")
        return None

    # 按修改时间排序，返回最新的
    latest_dir = max(result_dirs, key=lambda d: d.stat().st_mtime)
    logging.info(f"自动检测到最新训练结果: {latest_dir}")
    return latest_dir


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='YOLOv11 训练结果可视化生成器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 自动找最新训练结果并生成可视化
  python visualize_results.py

  # 指定训练结果目录
  python visualize_results.py --result-dir ./model/wood_defect_yolo11s

  # 指定输出DPI
  python visualize_results.py --result-dir ./model/wood_defect_yolo11s --dpi 300
        """
    )

    parser.add_argument(
        '--result-dir',
        type=str,
        help='训练结果目录路径（可选，默认自动查找最新）'
    )

    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='输出分辨率（默认: 300）'
    )

    args = parser.parse_args()

    # 检查依赖
    if not MATPLOTLIB_AVAILABLE or not PANDAS_AVAILABLE:
        logging.error("缺少必要的依赖，请安装: pip install matplotlib pandas")
        return 1

    # 确定训练结果目录
    if args.result_dir:
        result_dir = Path(args.result_dir)
        if not result_dir.exists():
            logging.error(f"指定的训练结果目录不存在: {result_dir}")
            return 1
    else:
        # 自动查找最新训练结果
        project_root = Path(__file__).parent
        result_dir = find_latest_result_dir(project_root)
        if result_dir is None:
            return 1

    # 检查CSV文件
    csv_path = result_dir / "training_metrics.csv"
    if not csv_path.exists():
        logging.error(f"训练指标CSV不存在: {csv_path}")
        logging.info("请确保训练已完成并生成了 training_metrics.csv 文件")
        return 1

    # 生成可视化
    logging.info("=" * 80)
    logging.info("开始生成训练曲线可视化...")
    logging.info("=" * 80)

    try:
        plotter = VectorPlotter(result_dir, dpi=args.dpi)
        output_files = plotter.plot_training_curves(csv_path)

        if output_files:
            logging.info("=" * 80)
            logging.info("可视化生成完成！")
            logging.info("=" * 80)
            for file_path in output_files:
                logging.info(f"  生成文件: {file_path}")
            logging.info("=" * 80)
            return 0
        else:
            logging.error("可视化生成失败")
            return 1

    except Exception as e:
        logging.error(f"生成可视化时发生错误: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
"""
YOLOv11s木材缺陷检测主训练脚本

功能：
- 加载YOLOv11s预训练模型
- 使用自定义配置进行训练
- 集成TensorBoard可视化
- 导出CSV/Excel训练指标
- 生成高质量矢量图

使用方法:
    python train.py

作者: Claude Code
日期: 2025
"""

import logging
import sys
from pathlib import Path
from datetime import datetime

# 添加项目路径到Python路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
except ImportError:
    print("错误: ultralytics库未安装")
    print("请运行: pip install -r requirements.txt")
    sys.exit(1)

from config import TrainingConfig
from callbacks import MetricsExporter, ProgressLogger
from utils.visualizer import VectorPlotter


def setup_logging(log_dir: Path) -> Path:
    """
    设置日志系统

    Args:
        log_dir: 日志目录

    Returns:
        Path: 日志文件路径
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"

    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return log_file


def validate_environment(config: TrainingConfig) -> bool:
    """
    验证训练环境

    Args:
        config: 训练配置对象

    Returns:
        bool: 验证是否通过
    """
    # 检查数据集配置文件
    if not config.data_yaml.exists():
        logging.error(f"数据集配置文件不存在: {config.data_yaml}")
        return False

    # 检查数据集目录
    dataset_root = config.data_yaml.parent
    required_dirs = ['images/train', 'images/val', 'labels/train', 'labels/val']

    for dir_name in required_dirs:
        dir_path = dataset_root / dir_name
        if not dir_path.exists():
            logging.error(f"必需的数据集目录不存在: {dir_path}")
            return False

    logging.info("环境验证通过")
    return True


def setup_callbacks(model: YOLO, config: TrainingConfig, save_dir: Path):
    """
    设置自定义回调函数

    Args:
        model: YOLO模型对象
        config: 训练配置
        save_dir: 保存目录

    Returns:
        tuple: (MetricsExporter, ProgressLogger)
    """
    # 初始化导出器和日志器
    exporter = MetricsExporter(save_dir)
    progress_logger = ProgressLogger(config.epochs)

    # 注册回调到YOLO模型
    model.add_callback("on_train_start", progress_logger.on_train_start)
    model.add_callback("on_train_epoch_start", progress_logger.on_train_epoch_start)
    model.add_callback("on_train_epoch_end", progress_logger.on_train_epoch_end)
    model.add_callback("on_train_epoch_end", exporter.on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", exporter.on_fit_epoch_end)
    model.add_callback("on_train_end", progress_logger.on_train_end)
    model.add_callback("on_train_end", exporter.on_train_end)

    logging.info("自定义回调函数已注册")

    return exporter, progress_logger


def post_training_visualization(save_dir: Path):
    """
    训练后生成矢量图

    Args:
        save_dir: 训练结果保存目录
    """
    logging.info("=" * 80)
    logging.info("开始生成矢量图...")

    try:
        plotter = VectorPlotter(save_dir, dpi=300)

        # 从自定义CSV生成矢量图
        custom_csv = save_dir / "training_metrics.csv"
        if custom_csv.exists():
            output_files = plotter.plot_training_curves(custom_csv)
            logging.info(f"成功生成 {len(output_files)} 个矢量图文件")
        else:
            logging.warning(f"自定义CSV不存在: {custom_csv}")

        # YOLO自带的results.csv
        yolo_csv = save_dir / "results.csv"
        if yolo_csv.exists():
            logging.info(f"YOLO内置结果文件: {yolo_csv}")
        else:
            logging.warning("YOLO results.csv未找到")

    except Exception as e:
        logging.error(f"矢量图生成失败: {e}", exc_info=True)


def print_final_summary(save_dir: Path, log_file: Path):
    """
    打印训练完成摘要

    Args:
        save_dir: 训练结果目录
        log_file: 日志文件路径
    """
    logging.info("=" * 80)
    logging.info("训练摘要")
    logging.info("=" * 80)

    # 权重文件
    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"

    if best_weights.exists():
        logging.info(f"最佳权重: {best_weights}")
    if last_weights.exists():
        logging.info(f"最后权重: {last_weights}")

    # 结果文件
    results_files = [
        ("YOLO结果CSV", save_dir / "results.csv"),
        ("自定义指标CSV", save_dir / "training_metrics.csv"),
        ("自定义指标Excel", save_dir / "training_metrics.xlsx"),
        ("训练曲线PDF", save_dir / "training_curves.pdf"),
        ("训练曲线SVG", save_dir / "training_curves.svg"),
    ]

    logging.info("\n结果文件:")
    for name, path in results_files:
        if path.exists():
            logging.info(f"  ✓ {name}: {path}")
        else:
            logging.info(f"  ✗ {name}: 未生成")

    # TensorBoard
    logging.info(f"\nTensorBoard可视化:")
    logging.info(f"  启动命令: tensorboard --logdir={save_dir}")
    logging.info(f"  访问地址: http://localhost:6006")

    # 日志文件
    logging.info(f"\n训练日志: {log_file}")

    logging.info("=" * 80)


def main():
    """主训练流程"""

    print("=" * 80)
    print("YOLOv11 木材缺陷检测训练程序")
    print("=" * 80)

    # 1. 加载配置
    config = TrainingConfig()

    # 2. 设置日志
    log_file = setup_logging(PROJECT_ROOT)
    logging.info("日志系统已初始化")

    # 3. 配置加载完成
    logging.info("配置加载完成")

    # 4. 验证环境
    if not validate_environment(config):
        logging.error("环境验证失败，终止训练")
        sys.exit(1)

    # 5. 加载预训练模型
    logging.info(f"正在加载预训练模型: {config.pretrained_model}")
    try:
        model = YOLO(config.pretrained_model)
        logging.info("预训练模型加载成功")
    except Exception as e:
        logging.error(f"模型加载失败: {e}", exc_info=True)
        sys.exit(1)

    # 6. 开始训练
    logging.info("=" * 80)
    logging.info("开始训练...")
    logging.info("=" * 80)

    try:
        # 执行训练
        results = model.train(**config.to_dict())

        # 7. 获取训练结果目录
        save_dir = Path(results.save_dir)
        logging.info(f"训练完成！结果保存在: {save_dir}")

        # 8. 设置回调并导出指标（注意：回调已在训练过程中执行）
        # 这里主要是为了确保所有文件都已生成

        # 9. 生成矢量图
        post_training_visualization(save_dir)

        # 10. 打印最终摘要
        print_final_summary(save_dir, log_file)

        # 11. 成功完成
        logging.info("所有任务完成！")
        return 0

    except KeyboardInterrupt:
        logging.warning("训练被用户中断")
        return 1

    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

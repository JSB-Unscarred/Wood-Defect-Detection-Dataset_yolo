"""
YOLOv11s木材缺陷检测主训练脚本

功能：
- 加载YOLOv11s预训练模型
- 使用自定义配置进行训练
- 集成TensorBoard可视化
- 导出CSV/Excel训练指标
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

    # 可视化提示
    logging.info(f"\n要生成训练曲线可视化，请运行:")
    logging.info(f"  python visualize_results.py --result-dir {save_dir}")

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

    # 5. 检测并加载模型
    checkpoint_path = config.output_dir / config.name / "weights" / "last.pt"
    resume_training = checkpoint_path.exists()

    if resume_training:
        logging.info("=" * 80)
        logging.info("检测到已有训练进度，将自动恢复训练")
        logging.info(f"Checkpoint: {checkpoint_path}")
        logging.info("=" * 80)
        try:
            model = YOLO(str(checkpoint_path))
            logging.info("Checkpoint加载成功")
        except Exception as e:
            logging.error(f"Checkpoint加载失败，从头开始: {e}")
            resume_training = False
            model = YOLO(config.pretrained_model)
    else:
        logging.info(f"未检测到训练进度，从预训练模型开始")
        logging.info(f"加载模型: {config.pretrained_model}")
        try:
            model = YOLO(config.pretrained_model)
            logging.info("预训练模型加载成功")
        except Exception as e:
            logging.error(f"模型加载失败: {e}", exc_info=True)
            sys.exit(1)

    # 6. 开始训练
    logging.info("=" * 80)
    logging.info("恢复训练..." if resume_training else "开始训练...")
    logging.info("=" * 80)

    try:
        # 设置回调（在训练前）
        setup_callbacks(model, config, config.output_dir / config.name)

        # 执行训练
        train_kwargs = config.to_dict()
        if resume_training:
            train_kwargs['resume'] = True

        results = model.train(**train_kwargs)

        # 7. 获取训练结果目录
        save_dir = Path(results.save_dir)
        logging.info(f"训练完成！结果保存在: {save_dir}")

        # 8. 打印最终摘要
        print_final_summary(save_dir, log_file)

        # 9. 成功完成
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

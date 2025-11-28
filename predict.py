"""
YOLOv11s木材缺陷检测推理脚本

功能：
- 加载训练好的YOLOv11s模型
- 对测试集进行推理和可视化
- 计算性能评估指标
- 导出JSON格式结果
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import yaml
import json

# 添加项目路径到Python路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
except ImportError:
    print("错误: ultralytics库未安装")
    print("请运行: pip install -r requirements.txt")
    sys.exit(1)


def load_config(yaml_path: Path = None) -> dict:
    """
    从YAML文件加载推理配置

    Args:
        yaml_path: YAML配置文件路径，默认为 PROJECT_ROOT/predict_args.yml

    Returns:
        dict: 推理配置字典
    """
    if yaml_path is None:
        yaml_path = PROJECT_ROOT / "predict_args.yml"

    if not yaml_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 转换路径字符串为 Path 对象
    path_keys = ['model_path', 'data_yaml', 'source', 'output_dir']
    for key in path_keys:
        if key in config and config[key]:
            config[key] = Path(config[key])

    logging.info(f"配置已从 {yaml_path} 加载")
    return config


def setup_logging(log_dir: Path) -> Path:
    """
    设置日志系统

    Args:
        log_dir: 日志目录

    Returns:
        Path: 日志文件路径
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"inference_{datetime.now():%Y%m%d_%H%M%S}.log"

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


def validate_environment(config: dict) -> bool:
    """
    验证推理环境

    Args:
        config: 推理配置字典

    Returns:
        bool: 验证是否通过
    """
    # 检查模型文件
    model_path = config['model_path']
    if not model_path.exists():
        logging.error(f"模型文件不存在: {model_path}")
        return False

    # 检查数据集配置文件
    data_yaml = config['data_yaml']
    if not data_yaml.exists():
        logging.error(f"数据集配置文件不存在: {data_yaml}")
        return False

    # 检查测试集目录
    test_dir = config['source']
    if not test_dir.exists():
        logging.error(f"测试集目录不存在: {test_dir}")
        return False

    logging.info("环境验证通过")
    return True


def run_prediction(model, config: dict, log_dir: Path) -> dict:
    """
    执行基础推理，生成预测结果和可视化图像

    Args:
        model: YOLO模型对象
        config: 推理配置字典
        log_dir: 输出目录

    Returns:
        dict: 推理统计信息
    """
    logging.info("开始对测试集进行推理...")

    results = model.predict(
        source=str(config['source']),
        conf=config['conf'],
        iou=config['iou'],
        imgsz=config['imgsz'],
        device=config['device'],
        batch=config['batch'],
        save=config['visualize'],
        save_txt=config['save_txt'],
        save_conf=config['save_conf'],
        project=str(log_dir),
        name='predictions',
        exist_ok=True
    )

    # 统计检测结果
    total_detections = 0
    class_counts = {}

    for result in results:
        total_detections += len(result.boxes)
        for cls in result.boxes.cls:
            class_name = model.names[int(cls)]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    prediction_stats = {
        'total_images': len(results),
        'total_detections': total_detections,
        'avg_detections_per_image': total_detections / len(results) if len(results) > 0 else 0,
        'class_distribution': class_counts
    }

    logging.info(f"推理完成！处理图像: {len(results)}, 总检测数: {total_detections}")

    return prediction_stats


def run_validation(model, config: dict, log_dir: Path) -> dict:
    """
    在测试集上执行完整性能评估

    Args:
        model: YOLO模型对象
        config: 推理配置字典
        log_dir: 输出目录

    Returns:
        dict: 验证指标
    """
    logging.info("开始性能评估...")

    # 使用 split='test' 指定测试集
    metrics = model.val(
        data=str(config['data_yaml']),
        split='test',
        conf=config['conf'],
        iou=config['iou'],
        imgsz=config['imgsz'],
        device=config['device'],
        batch=config['batch'],
        plots=True,
        project=str(log_dir),
        name='validation',
        exist_ok=True
    )

    # 提取关键指标
    mp = metrics.box.mp  # mean precision
    mr = metrics.box.mr  # mean recall
    f1 = 2 * (mp * mr) / (mp + mr + 1e-6) if (mp + mr) > 0 else 0

    validation_metrics = {
        'mAP@0.5': float(metrics.box.map50),
        'mAP@0.5:0.95': float(metrics.box.map),
        'precision': float(mp),
        'recall': float(mr),
        'f1_score': float(f1)
    }

    logging.info(f"评估完成！mAP@0.5: {validation_metrics['mAP@0.5']:.4f}")

    return validation_metrics


def save_results_json(prediction_stats: dict, validation_metrics: dict, config: dict, log_dir: Path):
    """
    保存JSON格式的推理结果

    Args:
        prediction_stats: 推理统计信息
        validation_metrics: 验证指标
        config: 推理配置
        log_dir: 输出目录
    """
    results = {
        'inference_info': {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(config['model_path']),
            'test_set': str(config['source'])
        },
        'prediction_statistics': prediction_stats,
        'validation_metrics': validation_metrics
    }

    json_path = log_dir / 'metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"结果已保存到: {json_path}")


def print_summary(prediction_stats: dict, validation_metrics: dict, log_dir: Path, log_file: Path):
    """
    打印推理完成摘要

    Args:
        prediction_stats: 推理统计信息
        validation_metrics: 验证指标
        log_dir: 输出目录
        log_file: 日志文件路径
    """
    logging.info("=" * 80)
    logging.info("推理摘要")
    logging.info("=" * 80)

    # 推理统计
    logging.info("\n推理统计:")
    logging.info(f"  总图像数: {prediction_stats['total_images']:,}")
    logging.info(f"  总检测数: {prediction_stats['total_detections']:,}")
    logging.info(f"  平均检测/图: {prediction_stats['avg_detections_per_image']:.2f}")

    # 类别分布
    logging.info("\n类别分布:")
    for cls, count in sorted(prediction_stats['class_distribution'].items()):
        logging.info(f"  {cls}: {count}")

    # 性能指标
    if validation_metrics:
        logging.info("\n性能指标:")
        logging.info(f"  mAP@0.5: {validation_metrics['mAP@0.5']:.4f}")
        logging.info(f"  mAP@0.5:0.95: {validation_metrics['mAP@0.5:0.95']:.4f}")
        logging.info(f"  Precision: {validation_metrics['precision']:.4f}")
        logging.info(f"  Recall: {validation_metrics['recall']:.4f}")
        logging.info(f"  F1-Score: {validation_metrics['f1_score']:.4f}")

    # 输出文件
    logging.info("\n输出文件:")
    logging.info(f"  ✓ 结果目录: {log_dir}")
    logging.info(f"  ✓ 可视化图像: {log_dir / 'predictions'}")
    logging.info(f"  ✓ 评估图表: {log_dir / 'validation'}")
    logging.info(f"  ✓ 性能指标: {log_dir / 'metrics.json'}")
    logging.info(f"  ✓ 推理日志: {log_file}")

    logging.info("=" * 80)


def main():
    """主推理流程"""

    print("=" * 80)
    print("YOLOv11 木材缺陷检测推理程序")
    print("=" * 80)

    # 1. 加载配置
    config = load_config()

    # 2. 创建输出目录（时间戳子目录）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = config['output_dir'] / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    # 3. 设置日志
    log_file = setup_logging(log_dir)
    logging.info("日志系统已初始化")

    # 4. 验证环境
    if not validate_environment(config):
        logging.error("环境验证失败，终止推理")
        sys.exit(1)

    # 5. 加载模型
    logging.info(f"加载模型: {config['model_path']}")
    try:
        model = YOLO(str(config['model_path']))
        logging.info("模型加载成功")
    except Exception as e:
        logging.error(f"模型加载失败: {e}", exc_info=True)
        sys.exit(1)

    # 6. 执行基础推理
    logging.info("=" * 80)
    logging.info("开始基础推理...")
    logging.info("=" * 80)
    prediction_stats = run_prediction(model, config, log_dir)

    # 7. 执行完整评估
    validation_metrics = None
    if config.get('run_val', False):
        logging.info("=" * 80)
        logging.info("开始性能评估...")
        logging.info("=" * 80)
        validation_metrics = run_validation(model, config, log_dir)

    # 8. 保存JSON结果
    save_results_json(prediction_stats, validation_metrics, config, log_dir)

    # 9. 打印摘要
    print_summary(prediction_stats, validation_metrics, log_dir, log_file)

    logging.info("推理任务全部完成！")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

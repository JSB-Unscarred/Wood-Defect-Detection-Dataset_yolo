"""
YOLOv11训练自定义回调函数模块

包含：
- MetricsExporter: 训练指标导出器（CSV/Excel）
- ProgressLogger: 进度日志记录器
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    pd = None


class MetricsExporter:
    """训练指标导出器 - 实时保存CSV和训练结束后导出Excel"""

    def __init__(self, save_dir: Path):
        """
        初始化指标导出器

        Args:
            save_dir: 保存目录路径
        """
        self.save_dir = Path(save_dir)
        self.csv_path = self.save_dir / "training_metrics.csv"
        self.excel_path = self.save_dir / "training_metrics.xlsx"
        self.metrics_history: List[Dict] = []
        self.csv_initialized = False

        logging.info(f"指标导出器已初始化: {self.save_dir}")

    def on_train_epoch_end(self, trainer):
        """
        训练epoch结束时的回调

        Args:
            trainer: YOLO trainer对象
        """
        try:
            # 提取训练损失
            epoch_metrics = {
                'epoch': trainer.epoch + 1,
                'timestamp': datetime.now().isoformat(),
            }

            # 训练损失（从loss_items获取）
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                loss_items = trainer.loss_items
                if len(loss_items) >= 3:
                    epoch_metrics['train/box_loss'] = float(loss_items[0])
                    epoch_metrics['train/cls_loss'] = float(loss_items[1])
                    epoch_metrics['train/dfl_loss'] = float(loss_items[2])

            # 学习率
            if hasattr(trainer, 'optimizer') and trainer.optimizer is not None:
                param_groups = trainer.optimizer.param_groups
                for i, pg in enumerate(param_groups):
                    epoch_metrics[f'lr/pg{i}'] = pg['lr']

            self.metrics_history.append(epoch_metrics)

            # 实时保存CSV
            self._save_to_csv()

        except Exception as e:
            logging.warning(f"训练epoch指标提取失败: {e}")

    def on_fit_epoch_end(self, trainer):
        """
        验证epoch结束时的回调（添加验证指标）

        Args:
            trainer: YOLO trainer对象
        """
        try:
            if len(self.metrics_history) == 0:
                return

            # 更新最后一条记录的验证指标
            if hasattr(trainer, 'metrics') and trainer.metrics is not None:
                metrics = trainer.metrics
                last_record = self.metrics_history[-1]

                # 验证损失
                if hasattr(metrics, 'box') and hasattr(metrics.box, 'loss'):
                    last_record['val/box_loss'] = float(metrics.box.loss)
                if hasattr(metrics, 'cls') and hasattr(metrics.cls, 'loss'):
                    last_record['val/cls_loss'] = float(metrics.cls.loss)
                if hasattr(metrics, 'dfl') and hasattr(metrics.dfl, 'loss'):
                    last_record['val/dfl_loss'] = float(metrics.dfl.loss)

                # mAP指标
                if hasattr(metrics, 'box'):
                    if hasattr(metrics.box, 'map50'):
                        last_record['metrics/mAP50'] = float(metrics.box.map50)
                    if hasattr(metrics.box, 'map'):
                        last_record['metrics/mAP50-95'] = float(metrics.box.map)
                    if hasattr(metrics.box, 'mp'):
                        last_record['metrics/precision'] = float(metrics.box.mp)
                    if hasattr(metrics.box, 'mr'):
                        last_record['metrics/recall'] = float(metrics.box.mr)

                # 适应度函数
                if hasattr(trainer, 'fitness') and trainer.fitness is not None:
                    last_record['fitness'] = float(trainer.fitness)

            # 更新CSV
            self._save_to_csv()

        except Exception as e:
            logging.warning(f"验证epoch指标提取失败: {e}")

    def _save_to_csv(self):
        """实时保存为CSV格式"""
        if not self.metrics_history:
            return

        try:
            # 确保目录存在
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)

            # 获取所有列名
            all_keys = set()
            for record in self.metrics_history:
                all_keys.update(record.keys())
            fieldnames = sorted(all_keys)

            # 写入CSV
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.metrics_history)

        except Exception as e:
            logging.error(f"CSV保存失败: {e}")

    def on_train_end(self, trainer):
        """
        训练结束时的回调（导出Excel）

        Args:
            trainer: YOLO trainer对象
        """
        if not self.metrics_history:
            logging.warning("没有训练指标可导出")
            return

        try:
            # 最终保存CSV
            self._save_to_csv()
            logging.info(f"训练指标CSV已保存: {self.csv_path}")

            # 导出Excel（如果pandas可用）
            if pd is not None:
                self._save_to_excel()
            else:
                logging.warning("pandas未安装，跳过Excel导出")

        except Exception as e:
            logging.error(f"训练结束指标导出失败: {e}")

    def _save_to_excel(self):
        """导出为Excel格式（带统计摘要）"""
        try:
            df = pd.DataFrame(self.metrics_history)

            with pd.ExcelWriter(self.excel_path, engine='openpyxl') as writer:
                # Sheet1: 完整训练指标
                df.to_excel(writer, sheet_name='训练指标', index=False)

                # Sheet2: 统计摘要
                summary_data = []

                # mAP统计
                if 'metrics/mAP50' in df.columns:
                    best_map50_idx = df['metrics/mAP50'].idxmax()
                    summary_data.append({
                        '指标': '最佳 mAP@0.5',
                        '数值': df.loc[best_map50_idx, 'metrics/mAP50'],
                        'Epoch': df.loc[best_map50_idx, 'epoch']
                    })

                if 'metrics/mAP50-95' in df.columns:
                    best_map_idx = df['metrics/mAP50-95'].idxmax()
                    summary_data.append({
                        '指标': '最佳 mAP@0.5:0.95',
                        '数值': df.loc[best_map_idx, 'metrics/mAP50-95'],
                        'Epoch': df.loc[best_map_idx, 'epoch']
                    })

                # 损失统计
                if 'val/box_loss' in df.columns:
                    best_loss_idx = df['val/box_loss'].idxmin()
                    summary_data.append({
                        '指标': '最低验证Box Loss',
                        '数值': df.loc[best_loss_idx, 'val/box_loss'],
                        'Epoch': df.loc[best_loss_idx, 'epoch']
                    })

                # 精度/召回率统计
                if 'metrics/precision' in df.columns:
                    summary_data.append({
                        '指标': '最高精度',
                        '数值': df['metrics/precision'].max(),
                        'Epoch': df.loc[df['metrics/precision'].idxmax(), 'epoch']
                    })

                if 'metrics/recall' in df.columns:
                    summary_data.append({
                        '指标': '最高召回率',
                        '数值': df['metrics/recall'].max(),
                        'Epoch': df.loc[df['metrics/recall'].idxmax(), 'epoch']
                    })

                # 训练总轮数
                summary_data.append({
                    '指标': '训练总轮数',
                    '数值': len(df),
                    'Epoch': '-'
                })

                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='训练摘要', index=False)

            logging.info(f"训练指标Excel已保存: {self.excel_path}")

        except Exception as e:
            logging.error(f"Excel保存失败: {e}")


class ProgressLogger:
    """进度日志记录器"""

    def __init__(self, total_epochs: int):
        """
        初始化进度日志器

        Args:
            total_epochs: 总训练轮数
        """
        self.total_epochs = total_epochs
        self.start_time = None
        self.epoch_start_time = None

    def on_train_start(self, trainer):
        """训练开始时的回调"""
        self.start_time = datetime.now()

        logging.info("=" * 80)
        logging.info("开始训练 YOLOv11 木材缺陷检测模型")
        logging.info("=" * 80)
        logging.info(f"总训练轮数: {self.total_epochs}")

        if hasattr(trainer, 'data') and trainer.data:
            logging.info(f"数据集路径: {trainer.data.get('path', 'N/A')}")
            logging.info(f"类别数: {trainer.data.get('nc', 'N/A')}")

        if hasattr(trainer, 'args'):
            logging.info(f"图像尺寸: {trainer.args.imgsz}")
            logging.info(f"批次大小: {trainer.args.batch}")
            logging.info(f"矩形训练: {trainer.args.rect}")

        logging.info(f"设备: {trainer.device}")
        logging.info(f"开始时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logging.info("=" * 80)

    def on_train_epoch_start(self, trainer):
        """训练epoch开始时的回调"""
        self.epoch_start_time = datetime.now()

    def on_train_epoch_end(self, trainer):
        """训练epoch结束时的回调"""
        if self.start_time is None:
            return

        epoch_num = trainer.epoch + 1
        elapsed_total = datetime.now() - self.start_time
        progress = epoch_num / self.total_epochs * 100

        # Epoch耗时
        epoch_time = ""
        if self.epoch_start_time:
            epoch_elapsed = datetime.now() - self.epoch_start_time
            epoch_time = f" | Epoch耗时: {epoch_elapsed}"

        # 训练损失
        loss_str = ""
        if hasattr(trainer, 'loss') and trainer.loss is not None:
            loss_str = f" | Loss: {trainer.loss:.4f}"

        logging.info(
            f"Epoch {epoch_num}/{self.total_epochs} ({progress:.1f}%) | "
            f"总耗时: {elapsed_total}{epoch_time}{loss_str}"
        )

    def on_train_end(self, trainer):
        """训练结束时的回调"""
        if self.start_time is None:
            return

        total_time = datetime.now() - self.start_time

        logging.info("=" * 80)
        logging.info("训练完成！")
        logging.info("=" * 80)
        logging.info(f"总耗时: {total_time}")

        if hasattr(trainer, 'best'):
            logging.info(f"最佳权重: {trainer.best}")
        if hasattr(trainer, 'last'):
            logging.info(f"最后权重: {trainer.last}")

        logging.info("=" * 80)


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)

    print("测试MetricsExporter...")
    exporter = MetricsExporter(Path("./test_output"))

    print("\n测试ProgressLogger...")
    logger = ProgressLogger(200)

    print("\n回调模块测试完成！")

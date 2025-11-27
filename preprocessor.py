#!/usr/bin/env python3
"""
木材缺陷数据集转换为 YOLO 格式
将 DatasetNinja 格式的标注转换为 YOLOv11 目标检测格式
"""

import os
import json
import shutil
import random
import logging
from pathlib import Path
import yaml


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('conversion.log'),
        logging.StreamHandler()
    ]
)


def load_class_mapping(meta_path):
    """
    从 meta.json 加载类别映射

    Args:
        meta_path: meta.json 文件路径

    Returns:
        class_mapping: {classId: yolo_class_id} 字典
        class_names: 类别名称列表
    """
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    with open(meta_path, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    class_mapping = {}
    class_names = []

    for idx, class_info in enumerate(meta_data['classes']):
        class_id = class_info['id']
        class_name = class_info['title']
        class_mapping[class_id] = idx
        class_names.append(class_name)

    return class_mapping, class_names


def convert_rectangle_to_yolo(x1, y1, x2, y2, img_width, img_height):
    """
    将矩形坐标转换为 YOLO 格式

    Args:
        x1, y1: 左上角坐标
        x2, y2: 右下角坐标
        img_width, img_height: 图像尺寸

    Returns:
        x_center_norm, y_center_norm, width_norm, height_norm: 归一化坐标
    """
    # 计算中心点和尺寸
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    # 归一化到 0-1
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    width_norm = width / img_width
    height_norm = height / img_height

    return x_center_norm, y_center_norm, width_norm, height_norm


def process_annotation_file(ann_path, class_mapping, img_width, img_height):
    """
    处理单个标注文件，提取 rectangle 类型标注并转换为 YOLO 格式

    Args:
        ann_path: 标注文件路径
        class_mapping: 类别映射字典
        img_width, img_height: 图像尺寸

    Returns:
        yolo_annotations: YOLO 格式标注行列表
        stats: 统计信息 {'rectangles': int, 'bitmaps': int}
    """
    if not os.path.exists(ann_path):
        logging.warning(f"Annotation file missing: {ann_path}")
        return [], {'rectangles': 0, 'bitmaps': 0}

    with open(ann_path, 'r', encoding='utf-8') as f:
        ann_data = json.load(f)

    yolo_annotations = []
    stats = {'rectangles': 0, 'bitmaps': 0}

    for obj in ann_data.get('objects', []):
        geometry_type = obj.get('geometryType')
        class_id = obj.get('classId')

        if geometry_type == 'bitmap':
            stats['bitmaps'] += 1
            continue

        if geometry_type != 'rectangle':
            continue

        if class_id not in class_mapping:
            logging.warning(f"Unknown classId {class_id} in {ann_path}")
            continue

        yolo_class_id = class_mapping[class_id]

        points = obj.get('points', {}).get('exterior', [])
        if len(points) != 2:
            logging.warning(f"Invalid rectangle points in {ann_path}")
            continue

        x1, y1 = points[0]
        x2, y2 = points[1]

        x_center, y_center, width, height = convert_rectangle_to_yolo(
            x1, y1, x2, y2, img_width, img_height
        )

        if width <= 0 or height <= 0:
            logging.warning(f"Invalid bbox dimensions in {ann_path}: w={width}, h={height}")
            continue

        if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                0 <= width <= 1 and 0 <= height <= 1):
            logging.warning(f"Bbox coordinates out of range in {ann_path}")
            continue

        yolo_line = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_annotations.append(yolo_line)
        stats['rectangles'] += 1

    return yolo_annotations, stats


def split_dataset(image_files, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    将图像文件列表按比例划分为 train/val/test

    Args:
        image_files: 图像文件名列表
        train_ratio, val_ratio, test_ratio: 划分比例
        random_seed: 随机种子

    Returns:
        train_files, val_files, test_files: 三个文件列表
    """
    random.seed(random_seed)

    shuffled_files = image_files.copy()
    random.shuffle(shuffled_files)

    total = len(shuffled_files)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = shuffled_files[:train_end]
    val_files = shuffled_files[train_end:val_end]
    test_files = shuffled_files[val_end:]

    return train_files, val_files, test_files


def create_directory_structure(output_dir):
    """
    创建 YOLO 格式的目录结构

    Args:
        output_dir: 输出根目录
    """
    dirs_to_create = [
        os.path.join(output_dir, 'images', 'train'),
        os.path.join(output_dir, 'images', 'val'),
        os.path.join(output_dir, 'images', 'test'),
        os.path.join(output_dir, 'labels', 'train'),
        os.path.join(output_dir, 'labels', 'val'),
        os.path.join(output_dir, 'labels', 'test'),
    ]

    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

    logging.info(f"Created directory structure at {output_dir}")


def organize_dataset(img_dir, ann_dir, output_dir, train_files, val_files, test_files,
                    class_mapping, class_names, img_width=2800, img_height=1024):
    """
    组织数据集到 YOLO 目录结构

    Args:
        img_dir: 源图像目录
        ann_dir: 源标注目录
        output_dir: 输出目录
        train_files, val_files, test_files: 划分后的文件列表
        class_mapping: 类别映射
        class_names: 类别名称列表
        img_width, img_height: 图像尺寸

    Returns:
        stats: 统计信息字典
    """
    stats = {
        'total_images': 0,
        'total_rectangles': 0,
        'total_bitmaps': 0,
        'empty_annotations': 0,
        'class_distribution': {name: 0 for name in class_names}
    }

    splits = [
        ('train', train_files),
        ('val', val_files),
        ('test', test_files)
    ]

    for split_name, file_list in splits:
        logging.info(f"Processing {split_name} split: {len(file_list)} images")

        for img_filename in file_list:
            src_img = os.path.join(img_dir, img_filename)
            dst_img = os.path.join(output_dir, 'images', split_name, img_filename)

            shutil.copy2(src_img, dst_img)
            stats['total_images'] += 1

            ann_filename = img_filename + '.json'
            ann_path = os.path.join(ann_dir, ann_filename)

            yolo_annotations, file_stats = process_annotation_file(
                ann_path, class_mapping, img_width, img_height
            )

            stats['total_rectangles'] += file_stats['rectangles']
            stats['total_bitmaps'] += file_stats['bitmaps']

            if len(yolo_annotations) == 0:
                stats['empty_annotations'] += 1

            label_filename = img_filename.replace('.bmp', '.txt')
            label_path = os.path.join(output_dir, 'labels', split_name, label_filename)

            with open(label_path, 'w') as f:
                for ann_line in yolo_annotations:
                    f.write(ann_line + '\n')
                    yolo_class_id = int(ann_line.split()[0])
                    class_name = class_names[yolo_class_id]
                    stats['class_distribution'][class_name] += 1

    return stats


def generate_yaml_and_classes(output_dir, class_names):
    """
    生成 data.yaml 和 classes.txt 配置文件

    Args:
        output_dir: 输出目录
        class_names: 类别名称列表
    """
    classes_path = os.path.join(output_dir, 'classes.txt')
    with open(classes_path, 'w', encoding='utf-8') as f:
        for name in class_names:
            f.write(f"{name}\n")

    logging.info(f"Generated classes.txt: {classes_path}")

    yaml_content = {
        'path': output_dir,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }

    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    logging.info(f"Generated data.yaml: {yaml_path}")


def print_statistics(stats, output_dir):
    """
    打印统计信息

    Args:
        stats: 统计信息字典
        output_dir: 输出目录
    """
    logging.info("=" * 60)
    logging.info("Conversion completed successfully!")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Total images processed: {stats['total_images']}")
    logging.info(f"Total rectangle annotations: {stats['total_rectangles']}")
    logging.info(f"Bitmap annotations ignored: {stats['total_bitmaps']}")
    logging.info(f"Images with no valid annotations: {stats['empty_annotations']}")
    logging.info("")
    logging.info("Class distribution:")
    for class_name, count in stats['class_distribution'].items():
        logging.info(f"  {class_name}: {count}")
    logging.info("=" * 60)


def main():
    """主函数"""
    SOURCE_DIR = "/data/chx25/wood-defect-detection-DatasetNinja" #修改这里的路径为数据集的真实路径
    OUTPUT_DIR = "/data/chx25/yolo_training_programn/wood_defect_yolo" #修改这里的路径为你想要保存YOLO格式数据集的路径
    META_PATH = os.path.join(SOURCE_DIR, "meta.json")
    IMG_DIR = os.path.join(SOURCE_DIR, "ds/img")
    ANN_DIR = os.path.join(SOURCE_DIR, "ds/ann")

    RANDOM_SEED = 42
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    IMG_WIDTH = 2800
    IMG_HEIGHT = 1024

    logging.info("Starting dataset conversion to YOLO format...")

    logging.info("Loading class mapping from meta.json...")
    class_mapping, class_names = load_class_mapping(META_PATH)
    logging.info(f"Loaded {len(class_names)} classes: {class_names}")

    logging.info("Scanning image files...")
    image_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.bmp')])
    logging.info(f"Found {len(image_files)} images")

    logging.info(f"Splitting dataset ({TRAIN_RATIO}:{VAL_RATIO}:{TEST_RATIO})...")
    train_files, val_files, test_files = split_dataset(
        image_files, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED
    )
    logging.info(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    logging.info("Creating output directory structure...")
    create_directory_structure(OUTPUT_DIR)

    logging.info("Processing and organizing dataset...")
    stats = organize_dataset(
        IMG_DIR, ANN_DIR, OUTPUT_DIR,
        train_files, val_files, test_files,
        class_mapping, class_names,
        IMG_WIDTH, IMG_HEIGHT
    )

    logging.info("Generating data.yaml and classes.txt...")
    generate_yaml_and_classes(OUTPUT_DIR, class_names)

    print_statistics(stats, OUTPUT_DIR)


if __name__ == "__main__":
    main()

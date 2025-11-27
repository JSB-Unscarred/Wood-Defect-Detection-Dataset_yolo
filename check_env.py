#!/usr/bin/env python3
"""
训练前环境检查脚本

功能：
- 验证Python依赖是否安装
- 检查数据集完整性
- 验证GPU可用性
- 测试配置加载
- 预估显存使用

"""

import sys
from pathlib import Path

print("=" * 80)
print("YOLOv11 训练环境检查")
print("=" * 80)

# 1. 检查Python版本
print("\n[1/6] 检查Python版本...")
py_version = sys.version_info
print(f"  Python版本: {py_version.major}.{py_version.minor}.{py_version.micro}")
if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
    print("  ✗ 警告: 推荐Python 3.8+")
else:
    print("  ✓ Python版本满足要求")

# 2. 检查必要的库
print("\n[2/6] 检查依赖库...")
required_packages = {
    'ultralytics': 'YOLOv11核心库',
    'pandas': '数据处理',
    'matplotlib': '可视化',
    'openpyxl': 'Excel导出',
    'torch': 'PyTorch',
    'cv2': 'OpenCV (opencv-python)',
}

missing_packages = []
for package, description in required_packages.items():
    try:
        if package == 'cv2':
            import cv2
        else:
            __import__(package)
        print(f"  ✓ {package:15s} - {description}")
    except ImportError:
        print(f"  ✗ {package:15s} - {description} (未安装)")
        missing_packages.append(package)

if missing_packages:
    print(f"\n  缺少依赖: {', '.join(missing_packages)}")
    print(f"  请运行: pip install -r requirements.txt")
else:
    print("\n  ✓ 所有依赖库已安装")

# 3. 检查GPU
print("\n[3/6] 检查GPU...")
try:
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"  ✓ 检测到 {gpu_count} 个GPU:")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"    GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

        # 检查CUDA版本
        cuda_version = torch.version.cuda
        print(f"  CUDA版本: {cuda_version}")
    else:
        print("  ✗ 警告: 未检测到GPU，将使用CPU训练（非常慢）")
except ImportError:
    print("  ✗ PyTorch未安装，无法检查GPU")

# 4. 检查数据集
print("\n[4/6] 检查数据集...")
dataset_root = Path("/data/chx25/yolo_training_programn/wood_defect_yolo")
data_yaml = dataset_root / "data.yaml"

if not dataset_root.exists():
    print(f"  ✗ 数据集目录不存在: {dataset_root}")
elif not data_yaml.exists():
    print(f"  ✗ data.yaml不存在: {data_yaml}")
else:
    print(f"  ✓ 数据集根目录: {dataset_root}")

    # 检查各个子目录
    subdirs = {
        'images/train': '训练图像',
        'images/val': '验证图像',
        'images/test': '测试图像',
        'labels/train': '训练标签',
        'labels/val': '验证标签',
        'labels/test': '测试标签',
    }

    for subdir, desc in subdirs.items():
        subdir_path = dataset_root / subdir
        if subdir_path.exists():
            file_count = len(list(subdir_path.glob('*')))
            print(f"  ✓ {desc:12s}: {file_count:5d} 文件")
        else:
            print(f"  ✗ {desc:12s}: 目录不存在")

# 5. 测试配置加载
print("\n[5/6] 测试配置加载...")
try:
    from config import TrainingConfig
    config = TrainingConfig()
    print(f"  ✓ 配置加载成功")
    print(f"    训练轮数: {config.epochs}")
    print(f"    批次大小: {config.batch}")
    print(f"    图像尺寸: {config.imgsz}")
    print(f"    矩形训练: {config.rect}")
    print(f"    Mosaic增强: {config.mosaic} (已禁用)" if config.mosaic == 0.0 else f"    Mosaic增强: {config.mosaic}")
except Exception as e:
    print(f"  ✗ 配置加载失败: {e}")

# 6. 显存估算
print("\n[6/6] 显存使用估算...")
try:
    from config import TrainingConfig
    config = TrainingConfig()

    # 粗略估算（基于经验公式）
    # YOLOv11s约7M参数，batch=16, imgsz=1600
    base_memory = 2.0  # 基础模型内存 (GB)
    batch_memory = config.batch * 0.3  # 每个batch约0.3GB (1600尺寸)
    total_estimated = base_memory + batch_memory

    print(f"  预估显存使用: ~{total_estimated:.1f} GB")
    print(f"  (batch={config.batch}, imgsz={config.imgsz})")

    if 'torch' in sys.modules and torch.cuda.is_available():
        available_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  可用显存: {available_memory:.1f} GB")

        if total_estimated > available_memory * 0.9:
            print(f"  ⚠ 警告: 显存可能不足，建议减小batch size或imgsz")
        else:
            print(f"  ✓ 显存充足")
    else:
        print(f"  (无法检测GPU显存)")

except Exception as e:
    print(f"  跳过显存估算: {e}")

# 总结
print("\n" + "=" * 80)
print("环境检查完成")
print("=" * 80)

if missing_packages:
    print("\n⚠ 发现问题: 缺少依赖库")
    print(f"  请运行: pip install -r requirements.txt")
    sys.exit(1)
elif not data_yaml.exists():
    print("\n⚠ 发现问题: 数据集配置不完整")
    sys.exit(1)
else:
    print("\n✓ 环境检查通过，可以开始训练！")
    print("\n启动训练命令:")
    print("  python3 train.py")
    print("\n监控训练（可选，在新终端）:")
    print("  tensorboard --logdir=model/wood_defect_yolo11s")
    sys.exit(0)

import os
import shutil
import random
from pathlib import Path

def create_directory_structure(base_path):
    """创建数据集所需的目录结构"""
    for split in ['train', 'test', 'val']:
        split_path = os.path.join(base_path, split)
        if not os.path.exists(split_path):
            os.makedirs(split_path)
            print(f"Created directory: {split_path}")

def split_dataset(source_dir, target_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    将数据集分割为训练集、验证集和测试集
    
    参数:
    source_dir: 源数据目录，包含所有room_*.txt文件
    target_dir: 目标数据目录，将包含train/val/test子目录
    train_ratio: 训练集比例
    val_ratio: 验证集比例
    test_ratio: 测试集比例
    random_seed: 随机种子，确保结果可重现
    """
    # 设置随机种子
    random.seed(random_seed)
    
    # 获取所有房间文件
    room_files = [f for f in os.listdir(source_dir) if f.startswith('room_') and f.endswith('.txt')]
    
    # 随机打乱文件列表
    random.shuffle(room_files)
    
    # 计算每个集合的大小
    total_files = len(room_files)
    train_size = int(total_files * train_ratio)
    val_size = int(total_files * val_ratio)
    
    # 分割文件列表
    train_files = room_files[:train_size]
    val_files = room_files[train_size:train_size + val_size]
    test_files = room_files[train_size + val_size:]
    
    # 创建目录结构
    create_directory_structure(target_dir)
    
    # 定义复制函数
    def copy_files(file_list, split_name):
        split_dir = os.path.join(target_dir, split_name)
        for file_name in file_list:
            source_path = os.path.join(source_dir, file_name)
            # 创建房间子目录
            room_name = file_name.split('.')[0]
            room_dir = os.path.join(split_dir, room_name)
            os.makedirs(room_dir, exist_ok=True)
            # 复制文件
            target_path = os.path.join(room_dir, file_name)
            shutil.copy2(source_path, target_path)
            print(f"Copied {file_name} to {split_name} set")
    
    # 复制文件到相应目录
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
    
    # 创建数据集信息文件
    create_dataset_info(target_dir, {
        'train': train_files,
        'val': val_files,
        'test': test_files
    })

def create_dataset_info(target_dir, split_files):
    """创建数据集信息文件"""
    info_file = os.path.join(target_dir, 'dataset_info.txt')
    with open(info_file, 'w') as f:
        f.write("Dataset Split Information\n")
        f.write("=======================\n\n")
        for split_name, files in split_files.items():
            f.write(f"{split_name.upper()} Set ({len(files)} files):\n")
            for file in sorted(files):
                f.write(f"- {file}\n")
            f.write("\n")

def main():
    # 设置源目录和目标目录
    source_directory = "scenes/data"  # 包含所有room_*.txt文件的目录
    target_directory = "scenes/t2n_processed"  # 将创建的数据集目录
    
    # 执行数据集分割
    split_dataset(source_directory, target_directory)
    
    print("\nDataset split completed!")
    print(f"Please check {target_directory} for the processed dataset.")

if __name__ == "__main__":
    main()

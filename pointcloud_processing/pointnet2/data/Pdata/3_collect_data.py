import os
import numpy as np
from tqdm import tqdm

def collect_point_data(input_dir, output_dir):
    """收集点云数据并按数据集分割转换为npy格式"""
    splits = ['train', 'test', 'val']
    
    for split in splits:
        # 为每个分割创建输出目录
        split_output_dir = os.path.join(output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        
        # 处理当前分割的数据
        split_dir = os.path.join(input_dir, split)
        rooms = os.listdir(split_dir)
        
        # 记录当前分割的文件路径
        anno_paths = []
        
        for room in tqdm(rooms, desc=f'Processing {split} set'):
            room_dir = os.path.join(split_dir, room)
            if not os.path.isdir(room_dir):
                continue
            
            txt_file = os.path.join(room_dir, f'{room}.txt')
            if not os.path.exists(txt_file):
                continue
            
            # 读取和处理数据
            data = np.loadtxt(txt_file)
            points = data[:, 0:6]   # x y z r g b
            labels = data[:, 6].astype(np.uint8)
            
            # 保存npy文件到对应的分割目录
            out_filename = f'{room}'
            np.save(os.path.join(split_output_dir, f'{out_filename}_point.npy'), points)
            np.save(os.path.join(split_output_dir, f'{out_filename}_label.npy'), labels)
            
            # 记录文件路径（使用相对路径）
            anno_paths.append(os.path.join(split, out_filename))
        
        # 为每个分割创建数据路径文件
        with open(os.path.join(output_dir, f'{split}_data_label.txt'), 'w') as f:
            for path in anno_paths:
                f.write(path + '\n')

def main():
    base_dir = 'scenes/t2n_processed'    # 分割好的txt数据目录
    output_dir = 'scenes/npy_data'        # npy文件输出目录
    
    collect_point_data(base_dir, output_dir)
    print("NPY files generation completed!")

if __name__ == '__main__':
    main()

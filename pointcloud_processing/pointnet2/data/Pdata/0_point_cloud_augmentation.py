import numpy as np
from pathlib import Path
import os

class PointCloudAugmentor:
    def __init__(self, input_file, output_dir):
        """
        初始化点云增强器
        Args:
            input_file: 输入点云文件路径
            output_dir: 输出目录
        """
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_point_cloud(self):
        """读取点云数据"""
        # 加载数据：x y z r g b label
        data = np.loadtxt(self.input_file)
        return data
    
    def random_rotation_matrix(self):
        """生成随机旋转矩阵"""
        angles = np.random.uniform(-np.pi, np.pi, 3)
        Rx = np.array([[1, 0, 0],
                      [0, np.cos(angles[0]), -np.sin(angles[0])],
                      [0, np.sin(angles[0]), np.cos(angles[0])]])
        
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                      [0, 1, 0],
                      [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                      [np.sin(angles[2]), np.cos(angles[2]), 0],
                      [0, 0, 1]])
        
        return np.dot(Rz, np.dot(Ry, Rx))
    
    def random_scale_matrix(self, min_scale=0.8, max_scale=1.2):
        """生成随机缩放矩阵"""
        scale = np.random.uniform(min_scale, max_scale)
        return np.eye(3) * scale
    
    def add_gaussian_noise(self, points, sigma=0.01):
        """添加高斯噪声"""
        noise = np.random.normal(0, sigma, points.shape)
        return points + noise
    
    def augment_single(self, points, add_noise=True):
        """对单个点云进行增强"""
        # 分离坐标和其他属性
        xyz = points[:, :3]
        features = points[:, 3:]  # RGB + label
        
        # 1. 随机旋转
        R = self.random_rotation_matrix()
        xyz_rotated = np.dot(xyz, R)
        
        # 2. 随机缩放
        S = self.random_scale_matrix()
        xyz_scaled = np.dot(xyz_rotated, S)
        
        # 3. 随机平移
        T = np.random.uniform(-2, 2, 3)
        xyz_translated = xyz_scaled + T
        
        # 4. 可选：添加高斯噪声
        if add_noise:
            xyz_translated = self.add_gaussian_noise(xyz_translated)
        
        # 组合变换后的点云
        augmented_points = np.column_stack([xyz_translated, features])
        return augmented_points
    
    def augment(self, num_augmented=100, add_noise=True):
        """生成多个增强样本"""
        # 读取原始点云
        original_points = self.load_point_cloud()
        
        # 生成多个增强样本
        for i in range(num_augmented):
            augmented_points = self.augment_single(original_points, add_noise)
            
            # 保存增强后的点云
            output_file = self.output_dir / f'room_{i+1:03d}.txt'
            np.savetxt(output_file, augmented_points, fmt='%.8f')
            
        print(f"已生成 {num_augmented} 个增强样本到 {self.output_dir}")

def main():
    # 设置输入输出路径
    input_file = 'scenes/demo_2.txt'  # 请修改为您的输入文件路径
    output_dir = 'scenes/data'  # 输出目录
    
    # 创建增强器并生成样本
    augmentor = PointCloudAugmentor(input_file, output_dir)
    augmentor.augment(num_augmented=500)  # 生成100个增强样本

if __name__ == '__main__':
    main()

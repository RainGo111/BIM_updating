import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import time
import warnings
warnings.filterwarnings('ignore')

class SimplePointCloudDiffDetector:
    
    def __init__(self, voxel_size=0.1, distance_threshold=0.2, max_points=30000):
        """
        参数:
            voxel_size: 体素下采样大小
            distance_threshold: 差异阈值
            max_points: 最大处理点数
        """
        self.voxel_size = voxel_size
        self.distance_threshold = distance_threshold
        self.max_points = max_points
    
    def load_and_preprocess(self, pc1_path, pc2_path):
        """加载并预处理点云"""
        print("加载点云文件...")
        pc1 = o3d.io.read_point_cloud(pc1_path)
        pc2 = o3d.io.read_point_cloud(pc2_path)
        
        print(f"原始点云 - PC1: {len(pc1.points):,} 点, PC2: {len(pc2.points):,} 点")
        
        # 体素下采样
        print(f"体素下采样 (voxel_size={self.voxel_size})...")
        pc1_down = pc1.voxel_down_sample(self.voxel_size)
        pc2_down = pc2.voxel_down_sample(self.voxel_size)
        
        print(f"下采样后 - PC1: {len(pc1_down.points):,} 点, PC2: {len(pc2_down.points):,} 点")
        
        # 进一步随机采样
        if len(pc1_down.points) > self.max_points:
            ratio = self.max_points / len(pc1_down.points)
            pc1_down = pc1_down.random_down_sample(ratio)
            print(f"进一步降采样 - PC1: {len(pc1_down.points):,} 点")
            
        if len(pc2_down.points) > self.max_points:
            ratio = self.max_points / len(pc2_down.points)
            pc2_down = pc2_down.random_down_sample(ratio)
            print(f"进一步降采样 - PC2: {len(pc2_down.points):,} 点")
        
        return pc1_down, pc2_down
    
    def detect_differences(self, pc1, pc2):
        """检测差异"""
        print("\n开始差异检测...")
        start_time = time.time()
        
        points1 = np.asarray(pc1.points)
        points2 = np.asarray(pc2.points)
        
        # 构建KD-Tree
        print("构建KD-Tree并计算距离...")
        tree2 = cKDTree(points2)
        distances, _ = tree2.query(points1, k=1)
        
        # 分类
        unchanged = distances < self.distance_threshold
        modified = (distances >= self.distance_threshold) & (distances < self.distance_threshold * 2)
        removed = distances >= self.distance_threshold * 2
        
        print(f"处理时间: {time.time() - start_time:.2f}秒")
        print(f"未变化: {np.sum(unchanged):,} 点 ({np.sum(unchanged)/len(points1)*100:.1f}%)")
        print(f"已修改: {np.sum(modified):,} 点 ({np.sum(modified)/len(points1)*100:.1f}%)")
        print(f"已删除: {np.sum(removed):,} 点 ({np.sum(removed)/len(points1)*100:.1f}%)")
        
        return distances, unchanged, modified, removed
    
    def create_colored_pointcloud(self, pc1, distances, unchanged, modified, removed):
        """创建彩色点云"""
        print("生成彩色结果...")
        
        # 复制点云
        pc_result = o3d.geometry.PointCloud(pc1)
        colors = np.zeros((len(pc1.points), 3))
        
        # 设置颜色
        colors[unchanged] = [0, 1, 0]      # 绿色 - 未变化
        colors[modified] = [1, 1, 0]       # 黄色 - 已修改
        colors[removed] = [1, 0, 0]        # 红色 - 已删除
        
        pc_result.colors = o3d.utility.Vector3dVector(colors)
        
        # 保存结果
        o3d.io.write_point_cloud("difference_result.ply", pc_result)
        print("结果已保存: difference_result.ply")
        
        return pc_result
    
    def simple_visualize(self, pc_result):
        """简单可视化"""
        print("\n显示结果...")
        print("颜色说明: 绿色=未变化, 黄色=已修改, 红色=已删除")
        print("关闭窗口继续...")
        
        # 确保点云有内容
        if len(pc_result.points) == 0:
            print("警告: 结果点云为空")
            return
        
        try:
            # 简单显示
            o3d.visualization.draw_geometries(
                [pc_result],
                window_name="Point Cloud Differences",
                width=800,
                height=600,
                left=50,
                top=50
            )
        except Exception as e:
            print(f"可视化失败: {e}")
            print("但结果文件已保存成功")
    
    def run_detection(self, pc1_path, pc2_path):
        """运行完整检测流程"""
        print("=" * 50)
        print("开始点云差异检测")
        print("=" * 50)
        
        try:
            # 1. 加载和预处理
            pc1, pc2 = self.load_and_preprocess(pc1_path, pc2_path)
            
            # 2. 检测差异
            distances, unchanged, modified, removed = self.detect_differences(pc1, pc2)
            
            # 3. 创建彩色结果
            pc_result = self.create_colored_pointcloud(pc1, distances, unchanged, modified, removed)
            
            # 4. 可视化
            self.simple_visualize(pc_result)
            
            print("\n" + "=" * 50)
            print("检测完成!")
            print("结果文件: difference_result.ply")
            print("=" * 50)
            
            return distances, pc_result
            
        except Exception as e:
            print(f"错误: {e}")
            return None, None


def main():
    
    # 文件路径
    pc1_path = "as_designed_model.ply"
    pc2_path = "as_built_model.ply"
    
    # 检查文件是否存在
    import os
    if not os.path.exists(pc1_path):
        print(f"错误: 找不到文件 {pc1_path}")
        return
    if not os.path.exists(pc2_path):
        print(f"错误: 找不到文件 {pc2_path}")
        return
    
    # 创建检测器 - 针对大型点云优化
    detector = SimplePointCloudDiffDetector(
        voxel_size=0.1,          # 体素
        distance_threshold=0.2,   # 差异阈值
        max_points=30000         # 最大处理点数
    )
    
    print(f"准备处理文件:")
    print(f"  as-designed: {pc1_path}")
    print(f"  as-built: {pc2_path}")
    print(f"参数设置:")
    print(f"  体素大小: {detector.voxel_size}")
    print(f"  差异阈值: {detector.distance_threshold}")
    print(f"  最大处理点数: {detector.max_points:,}")
    print()
    
    # 运行检测
    distances, result = detector.run_detection(pc1_path, pc2_path)
    
    if distances is not None:
        print(f"\n最终统计:")
        print(f"平均距离: {np.mean(distances):.3f}m")
        print(f"最大距离: {np.max(distances):.3f}m")
        print(f"处理的点数: {len(distances):,}")


if __name__ == "__main__":
    main()

# 参数调整
def run_with_custom_params():
    """使用自定义参数运行"""
    detector = SimplePointCloudDiffDetector(
        voxel_size=0.05,
        distance_threshold=0.15,
        max_points=100000
    )
    
    distances, result = detector.run_detection(
        "as_designed_model.ply", 
        "as_built_model.ply"
    )
    
    return distances, result


# 快速测试版本（更大的体素，更少的点）
def run_quick_test():
    detector = SimplePointCloudDiffDetector(
        voxel_size=0.001,          # 更大体素 = 更快
        distance_threshold=0.05,   # 更大阈值
        max_points=1000000         # 更少点数 = 更快
    )
    
    distances, result = detector.run_detection(
        "as_designed_model.ply", 
        "as_built_model.ply"
    )
    
    return distances, result

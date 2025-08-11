import open3d as o3d
import numpy as np
from pathlib import Path
import time
import sys

def load_and_visualize_point_cloud(file_path):
    """加载并可视化单个点云文件"""
    try:
        # 读取点云数据
        data = np.loadtxt(file_path)
        
        # 创建点云对象
        pcd = o3d.geometry.PointCloud()
        
        # 设置点的坐标
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])
        
        # 设置点的颜色
        colors = np.zeros((len(data), 3))
        pipe_mask = data[:, -1] == 1  # 管道标签为1
        colors[pipe_mask] = [1, 0, 0]   # 红色表示管道
        colors[~pipe_mask] = [0, 0, 1]  # 蓝色表示非管道
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # 可视化
        o3d.visualization.draw_geometries([pcd],
                                        window_name=f"Viewing: {Path(file_path).name}",
                                        width=1024,
                                        height=768)
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False
    
    return True

def visualize_point_clouds(directory):
    """可视化目录中的所有点云文件"""
    # 获取所有txt文件
    files = sorted(Path(directory).glob('*.txt'))
    
    if not files:
        print(f"No .txt files found in {directory}")
        return
    
    print(f"Found {len(files)} point cloud files.")
    print("按 'q' 关闭当前点云窗口并继续查看下一个")
    print("程序启动中...")
    time.sleep(2)  # 给用户时间阅读提示
    
    # 逐个显示点云
    for file_path in files:
        print(f"\n正在加载: {file_path.name}")
        if not load_and_visualize_point_cloud(str(file_path)):
            continue
            
def main():
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = 'scenes/data'  # 默认目录
    
    try:
        visualize_point_clouds(directory)
    except Exception as e:
        print(f"程序出错: {str(e)}")
    finally:
        print("\n可视化完成")

if __name__ == '__main__':
    main()

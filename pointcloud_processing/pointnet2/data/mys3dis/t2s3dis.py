import os
import numpy as np
from sklearn.neighbors import NearestNeighbors

def load_points(file_path):
    """加载点云文件"""
    return np.loadtxt(file_path)

def save_points(file_path, points):
    """保存点云文件"""
    np.savetxt(file_path, points, fmt="%.6f")

def rotate_point_cloud(points, angle_range=(-180, 180)):
    """随机旋转点云（仅对坐标部分进行旋转）"""
    angle = np.random.uniform(angle_range[0], angle_range[1])
    angle_rad = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    # 仅对坐标部分进行旋转
    points[:, :3] = np.dot(points[:, :3], rotation_matrix.T)
    return points

def translate_point_cloud(points, translation_range=(-0.5, 0.5)):
    """随机平移点云（仅对坐标部分进行平移）"""
    translation = np.random.uniform(translation_range[0], translation_range[1], size=(3,))
    points[:, :3] += translation
    return points

def scale_point_cloud(points, scale_range=(0.8, 1.2)):
    """随机缩放点云（仅对坐标部分进行缩放）"""
    scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= scale
    return points

def add_noise(points, noise_std=0.01):
    """添加高斯噪声（仅对坐标部分添加噪声）"""
    noise = np.random.normal(0, noise_std, points[:, :3].shape)
    points[:, :3] += noise
    return points

def augment_room(room_path, output_path, room_number, num_augmentations=500):
    """对单个房间点云进行数据增强"""
    try:
        # 加载整体点云
        room_file = f"room_{room_number}.txt"
        room_points = load_points(os.path.join(room_path, room_file))
        
        # 加载Annotations中的点云
        annotations_dir = os.path.join(room_path, "Annotations")
        annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith(".txt")]
        annotation_points = {}
        for file in annotation_files:
            annotation_points[file] = load_points(os.path.join(annotations_dir, file))
        
        # 计算这个房间的增强版本的起始编号
        existing_rooms = [d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d)) and d.startswith("room_")]
        if existing_rooms:
            existing_numbers = [int(room.split("_")[1]) for room in existing_rooms]
            start_index = max(existing_numbers) + 1
        else:
            start_index = 1

        for i in range(num_augmentations):
            current_room_number = start_index + i
            
            # 跳过原始房间号
            if current_room_number == room_number:
                continue
                
            # 随机生成增强参数
            angle = np.random.uniform(-180, 180)
            translation = np.random.uniform(-0.5, 0.5, size=(3,))
            scale = np.random.uniform(0.8, 1.2)
            noise_std = np.random.uniform(0.0, 0.02)
            
            # 对整体点云进行增强
            augmented_room_points = room_points.copy()
            augmented_room_points = rotate_point_cloud(augmented_room_points, angle_range=(angle, angle))
            augmented_room_points = translate_point_cloud(augmented_room_points, translation_range=(translation, translation))
            augmented_room_points = scale_point_cloud(augmented_room_points, scale_range=(scale, scale))
            augmented_room_points = add_noise(augmented_room_points, noise_std=noise_std)
            
            # 对Annotations中的点云进行相同的增强
            augmented_annotations = {}
            for file, points in annotation_points.items():
                augmented_points = points.copy()
                augmented_points = rotate_point_cloud(augmented_points, angle_range=(angle, angle))
                augmented_points = translate_point_cloud(augmented_points, translation_range=(translation, translation))
                augmented_points = scale_point_cloud(augmented_points, scale_range=(scale, scale))
                augmented_points = add_noise(augmented_points, noise_std=noise_std)
                augmented_annotations[file] = augmented_points
            
            # 创建新房间目录
            room_output_dir = os.path.join(output_path, f"room_{current_room_number}")
            if not os.path.exists(room_output_dir):
                os.makedirs(room_output_dir)
                os.makedirs(os.path.join(room_output_dir, "Annotations"), exist_ok=True)
                
                # 保存整体点云
                save_points(os.path.join(room_output_dir, f"room_{current_room_number}.txt"), augmented_room_points)
                
                # 保存Annotations中的点云
                for file, points in augmented_annotations.items():
                    save_points(os.path.join(room_output_dir, "Annotations", file), points)
                
                print(f"Generated room_{current_room_number}")
            
    except Exception as e:
        print(f"Error processing room_{room_number}: {str(e)}")
        raise

def augment_all_rooms(base_path, num_augmentations_per_room=199):
    """对多个房间进行数据增强"""
    try:
        # 获取所有房间文件夹并排序
        room_folders = [f for f in os.listdir(base_path) 
                       if f.startswith("room_") and os.path.isdir(os.path.join(base_path, f))]
        room_folders.sort(key=lambda x: int(x.split("_")[1]))
        
        total_rooms = len(room_folders)
        print(f"Found {total_rooms} rooms to process")
        
        # 对每个房间进行增强
        for room_folder in room_folders:
            try:
                room_number = int(room_folder.split("_")[1])
                room_path = os.path.join(base_path, room_folder)
                print(f"\nProcessing {room_path}")
                
                # 对当前房间进行增强
                augment_room(room_path, base_path, room_number, num_augmentations_per_room)
                
            except Exception as e:
                print(f"Error processing {room_folder}: {str(e)}")
                continue
        
        print("\nAugmentation completed for all rooms")
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    base_path = "Area_1"  # 包含所有房间文件夹的目录
    augment_all_rooms(base_path, num_augmentations_per_room=17)

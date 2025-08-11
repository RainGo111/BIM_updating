import os
import re

# 定义数据集根目录列表
DATA_ROOTS = ["Area_1", "Area_2"]  # 添加你需要处理的文件夹

# 定义输出文件路径
output_file = "my_anno_paths.txt"

# 按房间编号排序
def extract_room_number(room_folder):
    """从文件夹名称中提取房间编号"""
    match = re.match(r"room_(\d+)", room_folder)
    if match:
        return int(match.group(1))
    return -1  # 如果文件夹名称不符合 room_X 格式，返回 -1

# 生成Annotations路径并写入文件
with open(output_file, 'w') as f:
    for data_root in DATA_ROOTS:
        # 获取当前Area下的所有房间文件夹
        room_folders = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
        
        # 按房间编号排序
        room_folders_sorted = sorted(room_folders, key=extract_room_number)
        
        # 写入Annotations路径
        for room_folder in room_folders_sorted:
            anno_path = os.path.join(data_root, room_folder, "Annotations")
            if os.path.exists(anno_path):  # 确保Annotations文件夹存在
                f.write(f"{anno_path}\n")

print(f"Generated {output_file} with paths from {len(DATA_ROOTS)} areas.")

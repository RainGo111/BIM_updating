import open3d as o3d
import numpy as np


def visual_obj(path):
    with open(path, "r") as obj_file:
        points = []
        colors = []

        for line in obj_file.readlines():
            line = line.strip()
            line_list = line.split(" ")
            color = line_list[4:7]
            if color == ['38', '64', '140']:
                color = [str(38/255), str(64/255), str(140/255)]
            elif color == ['140', '76', '56']:
                color = [str(140/255), str(76/255), str(56/255)]
            elif color == ['140', '10', '79']:
                color = [str(140/255), str(10/255), str(79/255)]
            points.append(np.array(line_list[1:4]))
            colors.append(np.array(color))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])


def main():
    objt_file_path = "log/sem_seg/pointnet2_sem_seg/visual/Area_2_room_220_gt.obj"
    visual_obj(objt_file_path)


if __name__ == '__main__':
    main()

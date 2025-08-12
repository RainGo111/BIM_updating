# -*- coding: utf-8 -*-
"""
Core: compare as_built vs as_designed and write all differences back into the
as_designed coordinate space/result, with categories:
- ADDED   (仅在 as_built 出现的点)
- REMOVED (仅在 as_designed 出现的点)
- CHANGED (设计点与建成点存在匹配，但偏移超过 epsilon)
- UNCHANGED (变化小于 epsilon)

输出文件：
  1) as_designed_differences.ply
     - 以 as_designed 点云为主体，并根据类别着色
     - 同时把 as_built 的 “新增” 点附加进去（蓝色）
  2) change_summary.json
     - 统计各类别数量、参数与耗时等
  3) change_points.csv
     - 每个点一行，包含来源（as_designed/as_built）、在原始点云中的索引、坐标与类别
       便于后续筛选或与其他系统对接

基本假设：
  • 两个点云已经完成粗配准与精配准（处于同一坐标系）
  • 点云较大时，采用“非重叠均匀网格分块”以控制内存与加速 KDTree 查询

依赖：open3d, numpy, scipy（cKDTree）

命令示例：
  python pc_change_minimal.py as_designed_model.ply as_built_model.ply \
         --voxel 0.02 --grid 2.0 --match 0.05 --unchanged 0.015 --csv change_points.csv

重要参数说明：
  • --match：设计点与建成点匹配的最大半径，超过该半径即视作“无匹配”；
  • --unchanged：更严格的“未变”阈值（<= unchanged 视为 UNCHANGED，
                 位于 (unchanged, match] 视为 CHANGED）；
  • --grid：分块的网格边长（米），非重叠；
  • --min-block：若一个块内（设计+建成）总点数低于该阈值，则跳过该块（加速、降噪）。
"""
import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


@dataclass
class Params:
    """参数集合（便于在函数之间传递）"""
    voxel_size: float = 0.02      # 体素下采样尺寸（米）；0 代表不下采样
    grid_size: float = 2.0        # 非重叠分块网格边长（米）
    match_radius: float = 0.05    # 设计与建成点允许匹配的最大半径
    unchanged_eps: float = 0.015  # 更严格的“未变”阈值
    min_pts_block: int = 200      # 一个块内（设计+建成）最少点数，小于则跳过
    csv_path: str = "change_points.csv"  # CSV 输出路径


def load_downsample(path: str, voxel: float) -> o3d.geometry.PointCloud:
    """读取点云并按需进行体素下采样。"""
    pc = o3d.io.read_point_cloud(path)
    if voxel and voxel > 0:
        pc = pc.voxel_down_sample(voxel)
    return pc


def compute_bounds(p1: np.ndarray, p2: np.ndarray):
    """计算两云联合的包围盒（用于构建分块网格）。"""
    allp = np.vstack([p1, p2])
    return allp.min(0), allp.max(0)


def iter_blocks(minc, maxc, size):
    """遍历非重叠网格块，返回块索引与对应的 AABB（轴对齐包围盒）。"""
    dims = np.maximum(1, np.ceil((maxc - minc) / size).astype(int))
    for i in range(dims[0]):
        for j in range(dims[1]):
            for k in range(dims[2]):
                bmin = minc + np.array([i, j, k]) * size
                bmax = np.minimum(bmin + size, maxc)
                yield (i, j, k), bmin, bmax


def mask_in_box(pts: np.ndarray, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    """返回点是否在给定盒子内的布尔掩码。"""
    return np.all((pts >= bmin) & (pts <= bmax), axis=1)


def classify_block(block_idx,
                   des_pts, des_idx, built_pts, built_idx,
                   params: Params):
    """
    对单个块内的点进行分类。

    返回：
      design_labels: 与 des_idx 对齐的整型数组，取值 {0:UNCHANGED, 1:CHANGED, 2:REMOVED}
      added_points : (M,3) 仅在建成云中出现的“新增”点坐标
      added_indices: (M,)  上述“新增”点在原始 as_built 云中的索引

    逻辑：
      1) 针对设计点（des_pts），在建成树（tree_b）上做 1-NN 查询：
         - 距离 <= unchanged_eps 记为 UNCHANGED；
         - 距离在 (unchanged_eps, match_radius] 记为 CHANGED；
         - > match_radius 保持默认 REMOVED；
      2) 针对建成点（built_pts），在设计树（tree_d）上做 1-NN 查询：
         - 距离 > match_radius 记为 ADDED（即“新增”）。
    """
    # 默认全部标为 REMOVED（2）
    labels = np.full(len(des_idx), 2, dtype=np.uint8)
    added_points = np.empty((0, 3), dtype=np.float64)
    added_indices = np.empty((0,), dtype=np.int64)

    # 块内若两边都无点，直接返回
    if len(des_idx) == 0 and len(built_idx) == 0:
        return labels, added_points, added_indices

    # 块内若建成云无点，则设计点全部维持 REMOVED
    if len(built_idx) == 0:
        return labels, added_points, added_indices

    # 在建成点上建 KDTree，用于匹配设计点
    tree_b = cKDTree(built_pts)

    if len(des_idx) > 0:
        # 设计点寻找建成点最近邻
        dists, _ = tree_b.query(des_pts, k=1, workers=-1)
        # 未变
        unchanged_mask = dists <= params.unchanged_eps
        labels[unchanged_mask] = 0
        # 改变（但可匹配上）
        changed_mask = (dists > params.unchanged_eps) & (dists <= params.match_radius)
        labels[changed_mask] = 1
        # 其余保留为 REMOVED

    # 识别新增（ADDED）：找“建成点最近的设计点”是否超过匹配半径
    if len(des_idx) > 0:
        tree_d = cKDTree(des_pts)
        bdists, _ = tree_d.query(built_pts, k=1, workers=-1)
        added_mask = bdists > params.match_radius
    else:
        # 块内无设计点，建成点全为新增
        added_mask = np.ones(len(built_pts), dtype=bool)

    if np.any(added_mask):
        added_points = built_pts[added_mask]
        added_indices = built_idx[added_mask]

    return labels, added_points, added_indices


def run(designed_path: str, built_path: str, p: Params):
    """主流程：读取点云 -> 分块匹配 -> 汇总输出（PLY/JSON/CSV）。"""
    t0 = time.time()

    # 读取并下采样（若 voxel_size>0）
    des_pc = load_downsample(designed_path, p.voxel_size)
    built_pc = load_downsample(built_path, p.voxel_size)

    # 转为 numpy
    des = np.asarray(des_pc.points)
    built = np.asarray(built_pc.points)

    # 基于两云联合范围构建网格
    minc, maxc = compute_bounds(des, built)

    # 结果累加器（针对设计云：每个点一个标签；针对新增：收集建成云的新增点及其索引）
    N = len(des)
    design_labels = np.full(N, 2, dtype=np.uint8)  # 先全部置为 REMOVED
    processed = np.zeros(N, dtype=bool)            # 标记是否被分块访问（非重叠网格通常都会访问）
    added_all_points = []
    added_all_indices = []

    # 记录原始索引，确保 CSV 能写回“原始点云索引”
    des_idx_all = np.arange(N)
    built_idx_all = np.arange(len(built))

    block_count = 0
    for (i, j, k), bmin, bmax in iter_blocks(minc, maxc, p.grid_size):
        # 按块筛选点
        des_mask = mask_in_box(des, bmin, bmax)
        built_mask = mask_in_box(built, bmin, bmax)

        # 若块内点数过少，跳过以加速
        if des_mask.sum() + built_mask.sum() < p.min_pts_block:
            continue

        des_idx = des_idx_all[des_mask]
        built_idx = built_idx_all[built_mask]
        des_pts = des[des_mask]
        built_pts = built[built_mask]

        # 分类当前块
        lbls, added_pts, added_idx = classify_block((i, j, k), des_pts, des_idx, built_pts, built_idx, p)

        # 写回设计点标签
        design_labels[des_idx] = lbls
        processed[des_idx] = True

        # 汇总新增点
        if len(added_pts) > 0:
            added_all_points.append(added_pts)
            added_all_indices.append(added_idx)

        block_count += 1

    # 组装输出 PLY：先放设计点并按类别着色，再将新增点（建成）追加到同一云中
    colors = np.zeros((N, 3), dtype=np.float64)
    # 0: UNCHANGED -> 绿色
    colors[design_labels == 0] = [0.0, 0.8, 0.0]
    # 1: CHANGED   -> 橙红色
    colors[design_labels == 1] = [1.0, 0.3, 0.0]
    # 2: REMOVED   -> 紫色
    colors[design_labels == 2] = [0.6, 0.0, 0.8]

    out_points = [des]
    out_colors = [colors]

    # 汇总所有新增点与其索引
    added_points = np.vstack(added_all_points) if len(added_all_points) else np.empty((0, 3))
    added_indices = np.concatenate(added_all_indices).astype(int) if len(added_all_indices) else np.empty((0,), dtype=int)

    if len(added_points) > 0:
        out_points.append(added_points)
        # ADDED: 蓝色
        out_colors.append(np.tile([0.0, 0.5, 1.0], (len(added_points), 1)))

    out_pts = np.vstack(out_points)
    out_cols = np.vstack(out_colors)

    out_pc = o3d.geometry.PointCloud()
    out_pc.points = o3d.utility.Vector3dVector(out_pts)
    out_pc.colors = o3d.utility.Vector3dVector(out_cols)

    o3d.io.write_point_cloud("as_designed_differences.ply", out_pc)

    # JSON 摘要
    summary = {
        "designed_points": int(N),
        "built_points": int(len(built)),
        "blocks_processed": int(block_count),
        "unchanged": int((design_labels == 0).sum()),
        "changed": int((design_labels == 1).sum()),
        "removed": int((design_labels == 2).sum()),
        "added": int(len(added_points)),
        "params": {
            "voxel_size": p.voxel_size,
            "grid_size": p.grid_size,
            "match_radius": p.match_radius,
            "unchanged_eps": p.unchanged_eps,
            "min_pts_block": p.min_pts_block,
        },
        "runtime_sec": round(time.time() - t0, 2),
        "csv_path": p.csv_path,
    }
    with open("change_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # CSV 输出：单文件汇总所有类别，带来源信息
    # 列：source,index,x,y,z,category
    category_map = {0: "UNCHANGED", 1: "CHANGED", 2: "REMOVED"}
    with open(p.csv_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["source", "index", "x", "y", "z", "category"])

        # as_designed 行
        for idx in range(N):
            cat = int(design_labels[idx])
            writer.writerow(["as_designed", idx, float(des[idx, 0]), float(des[idx, 1]), float(des[idx, 2]), category_map[cat]])

        # as_built 的“新增”行
        for j in range(len(added_points)):
            pt = added_points[j]
            bi = int(added_indices[j]) if len(added_indices) > j else -1  # 兜底
            writer.writerow(["as_built", bi, float(pt[0]), float(pt[1]), float(pt[2]), "ADDED"])

    print("Saved: as_designed_differences.ply, change_summary.json, %s" % p.csv_path)
    print(summary)


def parse_args():
    """命令行参数解析。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("designed", type=str, help="as_designed_model.ply 的路径")
    ap.add_argument("built", type=str, help="as_built_model.ply 的路径")
    ap.add_argument("--voxel", type=float, default=0.02, help="体素下采样尺寸（米）；0 表示不下采样")
    ap.add_argument("--grid", type=float, default=2.0, help="分块网格边长（米）")
    ap.add_argument("--match", type=float, default=0.05, help="匹配半径（米），用于设计/建成最近邻匹配")
    ap.add_argument("--unchanged", type=float, default=0.015, help="未变阈值（米），越小越严格")
    ap.add_argument("--min-block", type=int, default=200, help="块内最少点数门槛（设计+建成），小于则跳过")
    ap.add_argument("--csv", type=str, default="change_points.csv", help="CSV 输出路径")
    return ap.parse_args()


if __name__ == "__main__":
    # 解析参数并组装 Params
    args = parse_args()
    p = Params(voxel_size=args.voxel,
               grid_size=args.grid,
               match_radius=args.match,
               unchanged_eps=args.unchanged,
               min_pts_block=args.min_block,
               csv_path=args.csv)

    # 执行主流程
    run(args.designed, args.built, p)

"""
分块检测
"""

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import time
import os
import warnings
from collections import defaultdict
import concurrent.futures
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
warnings.filterwarnings('ignore')

@dataclass
class Block:
    id: str
    bounds: np.ndarray  # [min_x, min_y, min_z, max_x, max_y, max_z]
    pc1_indices: List[int]
    pc2_indices: List[int]
    pc1_points: np.ndarray
    pc2_points: np.ndarray
    has_changes: bool = False
    change_score: float = 0.0

@dataclass
class BlockResult:
    block_id: str
    unchanged: np.ndarray
    minor_change: np.ndarray
    moderate_change: np.ndarray
    major_change: np.ndarray
    removed_indices: List[int]
    added_indices: List[int]
    removed_points: np.ndarray
    added_points: np.ndarray
    local_thresholds: Dict
    change_statistics: Dict

class BlockBasedPointCloudDetector:
    
    def __init__(self, 
                 voxel_size=0.02,
                 distance_threshold=0.05,
                 max_points=100000,
                 coarse_grid_size=2.0,      # 粗分块大小
                 fine_grid_size=0.5,       # 精细分块大小
                 min_points_per_block=30,  # 每块最少点数
                 change_threshold=0.1,     # 块变化阈值
                 overlap_ratio=0.1,        # 块重叠比例
                 parallel_processing=True): # 并行处理
        
        self.voxel_size = voxel_size
        self.distance_threshold = distance_threshold
        self.max_points = max_points
        self.coarse_grid_size = coarse_grid_size
        self.fine_grid_size = fine_grid_size
        self.min_points_per_block = min_points_per_block
        self.change_threshold = change_threshold
        self.overlap_ratio = overlap_ratio
        self.parallel_processing = parallel_processing
        
        # 基础阈值
        self.base_thresholds = {
            'unchanged': distance_threshold * 0.3,
            'minor_change': distance_threshold * 0.7,
            'moderate_change': distance_threshold * 1.5,
            'major_change': distance_threshold * 2.5,
        }
        
        # 新增/删除检测参数
        self.addition_params = {
            'search_radius': distance_threshold * 1.5,
            'min_neighbors': 3,
            'density_ratio': 0.4
        }
        
        print(f"初始化分块检测器:")
        print(f"  粗分块大小: {coarse_grid_size}m")
        print(f"  精细分块大小: {fine_grid_size}m")
        print(f"  最小块点数: {min_points_per_block}")
        print(f"  并行处理: {parallel_processing}")
    
    def load_and_preprocess(self, pc1_path, pc2_path):
        """加载并预处理点云"""
        print("=" * 70)
        print("步骤1: 加载和预处理点云")
        print("-" * 70)
        
        # 加载点云
        print("加载点云文件...")
        pc1 = o3d.io.read_point_cloud(pc1_path)
        pc2 = o3d.io.read_point_cloud(pc2_path)
        
        print(f"原始点云规模:")
        print(f"  PC1: {len(pc1.points):,} 点")
        print(f"  PC2: {len(pc2.points):,} 点")
        
        # 体素下采样
        print(f"体素下采样 (size={self.voxel_size}m)...")
        pc1_down = pc1.voxel_down_sample(self.voxel_size)
        pc2_down = pc2.voxel_down_sample(self.voxel_size)
        print(f"  下采样后 - PC1: {len(pc1_down.points):,} 点, PC2: {len(pc2_down.points):,} 点")
        
        # 限制点数（如果必要）
        if len(pc1_down.points) > self.max_points:
            print(f"  限制PC1点数到 {self.max_points:,}...")
            indices = np.random.choice(len(pc1_down.points), self.max_points, replace=False)
            pc1_down = pc1_down.select_by_index(indices)
            
        if len(pc2_down.points) > self.max_points:
            print(f"  限制PC2点数到 {self.max_points:,}...")
            indices = np.random.choice(len(pc2_down.points), self.max_points, replace=False)
            pc2_down = pc2_down.select_by_index(indices)
        
        print(f"最终点数 - PC1: {len(pc1_down.points):,}, PC2: {len(pc2_down.points):,}")
        
        return pc1_down, pc2_down
    
    def compute_bounds(self, points1, points2):
        """计算联合边界"""
        all_points = np.vstack([points1, points2])
        min_coords = np.min(all_points, axis=0)
        max_coords = np.max(all_points, axis=0)
        return min_coords, max_coords
    
    def create_coarse_blocks(self, points1, points2):
        """创建粗分块"""
        print("\n创建粗分块...")
        
        min_coords, max_coords = self.compute_bounds(points1, points2)
        
        # 计算网格维度
        grid_dims = np.ceil((max_coords - min_coords) / self.coarse_grid_size).astype(int)
        print(f"  粗网格维度: {grid_dims}")
        
        coarse_blocks = []
        block_count = 0
        
        # 遍历所有网格单元
        for i in range(grid_dims[0]):
            for j in range(grid_dims[1]):
                for k in range(grid_dims[2]):
                    # 计算块边界
                    block_min = min_coords + np.array([i, j, k]) * self.coarse_grid_size
                    block_max = block_min + self.coarse_grid_size
                    
                    # 添加重叠
                    overlap = self.coarse_grid_size * self.overlap_ratio
                    block_min_overlap = block_min - overlap
                    block_max_overlap = block_max + overlap
                    
                    # 查找块内的点
                    pc1_mask = np.all((points1 >= block_min_overlap) & (points1 <= block_max_overlap), axis=1)
                    pc2_mask = np.all((points2 >= block_min_overlap) & (points2 <= block_max_overlap), axis=1)
                    
                    pc1_indices = np.where(pc1_mask)[0]
                    pc2_indices = np.where(pc2_mask)[0]
                    
                    # 检查是否有足够的点
                    if len(pc1_indices) >= self.min_points_per_block or len(pc2_indices) >= self.min_points_per_block:
                        block = Block(
                            id=f"coarse_{i}_{j}_{k}",
                            bounds=np.array([*block_min, *block_max]),
                            pc1_indices=pc1_indices.tolist(),
                            pc2_indices=pc2_indices.tolist(),
                            pc1_points=points1[pc1_indices] if len(pc1_indices) > 0 else np.array([]).reshape(0,3),
                            pc2_points=points2[pc2_indices] if len(pc2_indices) > 0 else np.array([]).reshape(0,3)
                        )
                        coarse_blocks.append(block)
                        block_count += 1
        
        print(f"  创建了 {block_count} 个粗分块")
        return coarse_blocks
    
    def quick_block_assessment(self, block):
        # 如果一个块只在一个点云中有点，肯定有变化
        if len(block.pc1_points) == 0 or len(block.pc2_points) == 0:
            return True, 1.0
        
        # 计算点数比例
        count_ratio = abs(len(block.pc1_points) - len(block.pc2_points)) / max(len(block.pc1_points), len(block.pc2_points))
        
        # 快速距离检查
        if len(block.pc1_points) > 0 and len(block.pc2_points) > 0:
            tree2 = cKDTree(block.pc2_points)
            sample_size = min(100, len(block.pc1_points))  # 采样检查
            sample_indices = np.random.choice(len(block.pc1_points), sample_size, replace=False)
            sample_points = block.pc1_points[sample_indices]
            
            distances, _ = tree2.query(sample_points)
            avg_distance = np.mean(distances)
            change_score = max(count_ratio, avg_distance / self.distance_threshold)
        else:
            change_score = count_ratio
        
        has_changes = change_score > self.change_threshold
        return has_changes, change_score
    
    def filter_changed_blocks(self, coarse_blocks):
        """筛选有变化的粗分块"""
        print("筛选有变化的粗分块...")
        
        changed_blocks = []
        total_blocks = len(coarse_blocks)
        
        for block in coarse_blocks:
            has_changes, change_score = self.quick_block_assessment(block)
            block.has_changes = has_changes
            block.change_score = change_score
            
            if has_changes:
                changed_blocks.append(block)
        
        print(f"  {len(changed_blocks)}/{total_blocks} 个块有显著变化 ({len(changed_blocks)/total_blocks*100:.1f}%)")
        
        return changed_blocks
    
    def create_fine_blocks(self, coarse_block):
        """将粗分块进一步细分"""
        block_min = coarse_block.bounds[:3]
        block_max = coarse_block.bounds[3:]
        
        # 计算精细网格维度
        block_size = block_max - block_min
        fine_dims = np.ceil(block_size / self.fine_grid_size).astype(int)
        
        fine_blocks = []
        
        for i in range(fine_dims[0]):
            for j in range(fine_dims[1]):
                for k in range(fine_dims[2]):
                    # 计算精细块边界
                    fine_min = block_min + np.array([i, j, k]) * self.fine_grid_size
                    fine_max = fine_min + self.fine_grid_size
                    
                    # 确保不超出粗分块边界
                    fine_min = np.maximum(fine_min, block_min)
                    fine_max = np.minimum(fine_max, block_max)
                    
                    # 查找精细块内的点
                    if len(coarse_block.pc1_points) > 0:
                        pc1_mask = np.all((coarse_block.pc1_points >= fine_min) & (coarse_block.pc1_points <= fine_max), axis=1)
                        fine_pc1_indices = np.array(coarse_block.pc1_indices)[pc1_mask]
                        fine_pc1_points = coarse_block.pc1_points[pc1_mask]
                    else:
                        fine_pc1_indices = np.array([])
                        fine_pc1_points = np.array([]).reshape(0,3)
                    
                    if len(coarse_block.pc2_points) > 0:
                        pc2_mask = np.all((coarse_block.pc2_points >= fine_min) & (coarse_block.pc2_points <= fine_max), axis=1)
                        fine_pc2_indices = np.array(coarse_block.pc2_indices)[pc2_mask]
                        fine_pc2_points = coarse_block.pc2_points[pc2_mask]
                    else:
                        fine_pc2_indices = np.array([])
                        fine_pc2_points = np.array([]).reshape(0,3)
                    
                    # 检查是否有足够的点
                    if len(fine_pc1_points) >= 5 or len(fine_pc2_points) >= 5:
                        fine_block = Block(
                            id=f"{coarse_block.id}_fine_{i}_{j}_{k}",
                            bounds=np.array([*fine_min, *fine_max]),
                            pc1_indices=fine_pc1_indices.tolist(),
                            pc2_indices=fine_pc2_indices.tolist(),
                            pc1_points=fine_pc1_points,
                            pc2_points=fine_pc2_points
                        )
                        fine_blocks.append(fine_block)
        
        return fine_blocks
    
    def precise_block_comparison(self, block):
        """对单个精细块进行精确比较"""
        # 自适应阈值
        block_size = np.max(block.bounds[3:] - block.bounds[:3])
        size_factor = min(1.5, max(0.5, block_size / self.fine_grid_size))
        
        local_thresholds = {
            'unchanged': self.base_thresholds['unchanged'] * size_factor,
            'minor_change': self.base_thresholds['minor_change'] * size_factor,
            'moderate_change': self.base_thresholds['moderate_change'] * size_factor,
            'major_change': self.base_thresholds['major_change'] * size_factor,
        }
        
        # 初始化结果数组
        unchanged = np.array([], dtype=bool)
        minor_change = np.array([], dtype=bool)
        moderate_change = np.array([], dtype=bool)
        major_change = np.array([], dtype=bool)
        removed_indices = []
        added_indices = []
        removed_points = np.array([]).reshape(0,3)
        added_points = np.array([]).reshape(0,3)
        
        # 如果块中没有点，跳过
        if len(block.pc1_points) == 0 and len(block.pc2_points) == 0:
            return BlockResult(
                block_id=block.id,
                unchanged=unchanged, minor_change=minor_change,
                moderate_change=moderate_change, major_change=major_change,
                removed_indices=removed_indices, added_indices=added_indices,
                removed_points=removed_points, added_points=added_points,
                local_thresholds=local_thresholds,
                change_statistics={}
            )
        
        # 处理只有删除的情况
        if len(block.pc2_points) == 0:
            removed_indices = block.pc1_indices
            removed_points = block.pc1_points
            return BlockResult(
                block_id=block.id,
                unchanged=unchanged, minor_change=minor_change,
                moderate_change=moderate_change, major_change=major_change,
                removed_indices=removed_indices, added_indices=added_indices,
                removed_points=removed_points, added_points=added_points,
                local_thresholds=local_thresholds,
                change_statistics={'type': 'all_removed', 'count': len(removed_indices)}
            )
        
        # 处理只有新增的情况
        if len(block.pc1_points) == 0:
            added_indices = block.pc2_indices
            added_points = block.pc2_points
            return BlockResult(
                block_id=block.id,
                unchanged=unchanged, minor_change=minor_change,
                moderate_change=moderate_change, major_change=major_change,
                removed_indices=removed_indices, added_indices=added_indices,
                removed_points=removed_points, added_points=added_points,
                local_thresholds=local_thresholds,
                change_statistics={'type': 'all_added', 'count': len(added_indices)}
            )
        
        # 正常情况：两个点云都有点
        tree1 = cKDTree(block.pc1_points)
        tree2 = cKDTree(block.pc2_points)
        
        # PC1 -> PC2 距离计算
        distances_1to2, _ = tree2.query(block.pc1_points)
        
        # 分类PC1中的点
        unchanged = distances_1to2 < local_thresholds['unchanged']
        minor_change = (distances_1to2 >= local_thresholds['unchanged']) & \
                      (distances_1to2 < local_thresholds['minor_change'])
        moderate_change = (distances_1to2 >= local_thresholds['minor_change']) & \
                         (distances_1to2 < local_thresholds['moderate_change'])
        major_change = (distances_1to2 >= local_thresholds['moderate_change']) & \
                      (distances_1to2 < local_thresholds['major_change'])
        
        # 识别删除的点
        removed_mask = distances_1to2 >= local_thresholds['major_change']
        if np.any(removed_mask):
            removed_local_indices = np.where(removed_mask)[0]
            removed_indices = [block.pc1_indices[i] for i in removed_local_indices]
            removed_points = block.pc1_points[removed_mask]
        
        # 识别新增的点
        distances_2to1, _ = tree1.query(block.pc2_points)
        added_mask = distances_2to1 >= local_thresholds['major_change']
        
        if np.any(added_mask):
            added_local_indices = np.where(added_mask)[0]
            added_indices = [block.pc2_indices[i] for i in added_local_indices]
            added_points = block.pc2_points[added_mask]
        
        # 统计信息
        change_stats = {
            'unchanged': np.sum(unchanged),
            'minor_change': np.sum(minor_change),
            'moderate_change': np.sum(moderate_change),
            'major_change': np.sum(major_change),
            'removed': len(removed_indices),
            'added': len(added_indices),
            'total_pc1': len(block.pc1_points),
            'total_pc2': len(block.pc2_points)
        }
        
        return BlockResult(
            block_id=block.id,
            unchanged=unchanged, minor_change=minor_change,
            moderate_change=moderate_change, major_change=major_change,
            removed_indices=removed_indices, added_indices=added_indices,
            removed_points=removed_points, added_points=added_points,
            local_thresholds=local_thresholds,
            change_statistics=change_stats
        )
    
    def process_block_batch(self, blocks):
        """批量处理块"""
        results = []
        for block in blocks:
            fine_blocks = self.create_fine_blocks(block)
            for fine_block in fine_blocks:
                result = self.precise_block_comparison(fine_block)
                results.append(result)
        return results
    
    def detect_differences(self, pc1, pc2):
        """主要的分块差异检测方法"""
        print("\n" + "=" * 70)
        print("步骤2: 分层分块差异检测")
        print("-" * 70)
        
        start_time = time.time()
        
        points1 = np.asarray(pc1.points)
        points2 = np.asarray(pc2.points)
        
        # 第一阶段：粗分块
        coarse_blocks = self.create_coarse_blocks(points1, points2)
        
        # 第二阶段：筛选有变化的块
        changed_blocks = self.filter_changed_blocks(coarse_blocks)
        
        # 第三阶段：精细分块和详细比较
        print("\n创建精细分块并进行详细比较...")
        
        all_block_results = []
        
        if self.parallel_processing and len(changed_blocks) > 1:
            # 并行处理
            print(f"  使用并行处理 {len(changed_blocks)} 个变化块...")
            
            batch_size = max(1, len(changed_blocks) // 4)
            block_batches = [changed_blocks[i:i+batch_size] for i in range(0, len(changed_blocks), batch_size)]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_batch = {executor.submit(self.process_block_batch, batch): batch for batch in block_batches}
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    batch_results = future.result()
                    all_block_results.extend(batch_results)
        else:
            # 串行处理
            print(f"  串行处理 {len(changed_blocks)} 个变化块...")
            for i, block in enumerate(changed_blocks):
                print(f"    处理块 {i+1}/{len(changed_blocks)}: {block.id}")
                fine_blocks = self.create_fine_blocks(block)
                for fine_block in fine_blocks:
                    result = self.precise_block_comparison(fine_block)
                    all_block_results.append(result)
        
        print(f"  处理了 {len(all_block_results)} 个精细块")
        
        # 第四阶段：合并结果
        print("合并块结果...")
        merged_results = self.merge_block_results(points1, points2, all_block_results)
        
        print(f"总处理时间: {time.time() - start_time:.2f}秒")
        
        return merged_results
    
    def merge_block_results(self, points1, points2, block_results):
        """合并所有块的检测结果"""
        total_pc1 = len(points1)
        total_pc2 = len(points2)
        
        print("  正在收集块结果...")
        
        # 使用集合来更高效地处理重复索引
        removed_indices_set = set()
        added_indices_set = set()
        all_removed_points = []
        all_added_points = []
        
        # 统计信息
        block_statistics = []
        valid_blocks = 0
        
        for i, result in enumerate(block_results):
            if i % 500 == 0:
                print(f"    处理块 {i+1}/{len(block_results)}")
            
            # 收集删除和新增的索引
            if result.removed_indices:
                removed_indices_set.update(result.removed_indices)
            if result.added_indices:
                added_indices_set.update(result.added_indices)
            
            # 收集点数据
            if len(result.removed_points) > 0:
                all_removed_points.append(result.removed_points)
            if len(result.added_points) > 0:
                all_added_points.append(result.added_points)
            
            # 收集统计信息
            if result.change_statistics:
                block_statistics.append(result.change_statistics)
                valid_blocks += 1
        
        print(f"  有效块数: {valid_blocks}/{len(block_results)}")
        
        # 合并点数组
        print("  合并删除点...")
        if all_removed_points:
            merged_removed_points = np.vstack(all_removed_points)
        else:
            merged_removed_points = np.array([]).reshape(0,3)
            
        print("  合并新增点...")
        if all_added_points:
            merged_added_points = np.vstack(all_added_points)
        else:
            merged_added_points = np.array([]).reshape(0,3)
        
        # 转换集合为列表
        unique_removed_indices = list(removed_indices_set)
        unique_added_indices = list(added_indices_set)
        
        print("  创建全局分类数组...")
        # 简化的全局分类
        global_removed = np.zeros(total_pc1, dtype=bool)
        if unique_removed_indices:
            valid_removed_indices = [idx for idx in unique_removed_indices if 0 <= idx < total_pc1]
            global_removed[valid_removed_indices] = True
        
        # 计算基本距离分析
        global_unchanged = np.zeros(total_pc1, dtype=bool)
        global_minor = np.zeros(total_pc1, dtype=bool)
        global_moderate = np.zeros(total_pc1, dtype=bool) 
        global_major = np.zeros(total_pc1, dtype=bool)
        
        # 对未删除的点进行距离分析
        print("  计算距离分析...")
        non_removed_mask = ~global_removed
        non_removed_indices = np.where(non_removed_mask)[0]
        
        if len(non_removed_indices) > 0 and len(points2) > 0:
            tree2 = cKDTree(points2)
            distances = np.full(total_pc1, np.inf)
            
            # 分批处理
            batch_size = 10000
            for i in range(0, len(non_removed_indices), batch_size):
                batch_indices = non_removed_indices[i:i+batch_size]
                batch_points = points1[batch_indices]
                batch_distances, _ = tree2.query(batch_points)
                distances[batch_indices] = batch_distances
            
            # 分类
            global_unchanged[non_removed_mask] = distances[non_removed_mask] < self.base_thresholds['unchanged']
            global_minor[non_removed_mask] = ((distances[non_removed_mask] >= self.base_thresholds['unchanged']) & 
                                            (distances[non_removed_mask] < self.base_thresholds['minor_change']))
            global_moderate[non_removed_mask] = ((distances[non_removed_mask] >= self.base_thresholds['minor_change']) & 
                                               (distances[non_removed_mask] < self.base_thresholds['moderate_change']))
            global_major[non_removed_mask] = ((distances[non_removed_mask] >= self.base_thresholds['moderate_change']) & 
                                            (distances[non_removed_mask] < self.base_thresholds['major_change']))
        else:
            distances = np.full(total_pc1, np.inf)
            global_unchanged[non_removed_mask] = True
        
        # 统计结果
        print("\n分块检测统计:")
        print(f"  处理的块数: {len(block_results)}")
        print(f"  未变化: {np.sum(global_unchanged):,} ({np.sum(global_unchanged)/total_pc1*100:.1f}%)")
        print(f"  轻微变化: {np.sum(global_minor):,} ({np.sum(global_minor)/total_pc1*100:.1f}%)")
        print(f"  中等变化: {np.sum(global_moderate):,} ({np.sum(global_moderate)/total_pc1*100:.1f}%)")
        print(f"  重大变化: {np.sum(global_major):,} ({np.sum(global_major)/total_pc1*100:.1f}%)")
        print(f"  已删除: {len(unique_removed_indices):,} ({len(unique_removed_indices)/total_pc1*100:.1f}%)")
        print(f"  新增点: {len(unique_added_indices):,}")
        
        return {
            'distances': distances,
            'unchanged': global_unchanged,
            'minor_change': global_minor,
            'moderate_change': global_moderate,
            'major_change': global_major,
            'removed': global_removed,
            'removed_points': merged_removed_points,
            'new_points': merged_added_points,
            'pc1_points': points1,
            'pc2_points': points2,
            'thresholds': self.base_thresholds,
            'block_results': block_results[:100],  # 只保留前100个块结果
            'block_statistics': block_statistics,
            'unique_removed_indices': unique_removed_indices,
            'unique_added_indices': unique_added_indices,
            'processing_summary': {
                'total_blocks_processed': len(block_results),
                'valid_blocks': valid_blocks,
                'parallel_processing': self.parallel_processing,
                'coarse_grid_size': self.coarse_grid_size,
                'fine_grid_size': self.fine_grid_size
            }
        }
    
    def save_results(self, pc1, pc2, results):
        """保存分块检测结果"""
        print("\n" + "=" * 70)
        print("步骤3: 保存分块检测结果")
        print("-" * 70)
        
        try:
            # 1. 保存主要分类结果
            print("创建主要分类点云...")
            pc_result = o3d.geometry.PointCloud()
            pc_result.points = o3d.utility.Vector3dVector(results['pc1_points'])
            
            colors = np.zeros((len(results['pc1_points']), 3))
            colors[results['unchanged']] = [0, 0.8, 0]        # 绿色
            colors[results['minor_change']] = [0.5, 1, 0]     # 黄绿
            colors[results['moderate_change']] = [1, 0.8, 0]  # 橙色
            colors[results['major_change']] = [1, 0, 0]       # 红色
            colors[results['removed']] = [0.5, 0, 0.5]        # 紫色
            
            pc_result.colors = o3d.utility.Vector3dVector(colors)
            
            print("  保存 block_difference_result.ply...")
            o3d.io.write_point_cloud("block_difference_result.ply", pc_result)
            print("  ✓ 已保存")
            
            # 2. 保存新增点
            if len(results['new_points']) > 0:
                print(f"创建新增点云 ({len(results['new_points'])} 点)...")
                pc_new = o3d.geometry.PointCloud()
                pc_new.points = o3d.utility.Vector3dVector(results['new_points'])
                pc_new.paint_uniform_color([0, 0.5, 1])  # 蓝色
                
                print("  保存 block_new_points.ply...")
                o3d.io.write_point_cloud("block_new_points.ply", pc_new)
                print("  ✓ 已保存")
            
            # 3. 保存删除点
            if len(results['removed_points']) > 0:
                print(f"创建删除点云 ({len(results['removed_points'])} 点)...")
                
                # 分批保存大文件
                max_points_per_file = 50000
                if len(results['removed_points']) > max_points_per_file:
                    print(f"  点数较多，分批保存...")
                    num_batches = (len(results['removed_points']) + max_points_per_file - 1) // max_points_per_file
                    for i in range(num_batches):
                        start_idx = i * max_points_per_file
                        end_idx = min((i + 1) * max_points_per_file, len(results['removed_points']))
                        
                        batch_points = results['removed_points'][start_idx:end_idx]
                        pc_removed_batch = o3d.geometry.PointCloud()
                        pc_removed_batch.points = o3d.utility.Vector3dVector(batch_points)
                        pc_removed_batch.paint_uniform_color([0.7, 0, 0.7])  # 紫色
                        
                        filename = f"block_removed_points_part{i+1}.ply"
                        print(f"    保存 {filename} ({len(batch_points)} 点)...")
                        o3d.io.write_point_cloud(filename, pc_removed_batch)
                        print(f"    ✓ 已保存")
                else:
                    pc_removed = o3d.geometry.PointCloud()
                    pc_removed.points = o3d.utility.Vector3dVector(results['removed_points'])
                    pc_removed.paint_uniform_color([0.7, 0, 0.7])  # 紫色
                    
                    print("  保存 block_removed_points.ply...")
                    o3d.io.write_point_cloud("block_removed_points.ply", pc_removed)
                    print("  ✓ 已保存")
            
            # 4. 保存组合结果
            print("创建组合点云...")
            max_combined_points = 80000
            
            pc1_sample_size = min(len(results['pc1_points']), max_combined_points - len(results['new_points']))
            if pc1_sample_size < len(results['pc1_points']):
                print(f"  采样PC1点云 ({pc1_sample_size}/{len(results['pc1_points'])} 点)")
                sample_indices = np.random.choice(len(results['pc1_points']), pc1_sample_size, replace=False)
                sampled_pc1_points = results['pc1_points'][sample_indices]
                sampled_colors = colors[sample_indices]
            else:
                sampled_pc1_points = results['pc1_points']
                sampled_colors = colors
            
            all_points = [sampled_pc1_points]
            all_colors = [sampled_colors]
            
            if len(results['new_points']) > 0:
                all_points.append(results['new_points'])
                new_colors = np.tile([0, 0.5, 1], (len(results['new_points']), 1))
                all_colors.append(new_colors)
            
            if all_points:
                combined_points = np.vstack(all_points)
                combined_colors = np.vstack(all_colors)
                
                pc_combined = o3d.geometry.PointCloud()
                pc_combined.points = o3d.utility.Vector3dVector(combined_points)
                pc_combined.colors = o3d.utility.Vector3dVector(combined_colors)
                
                print("  保存 block_difference_combined.ply...")
                o3d.io.write_point_cloud("block_difference_combined.ply", pc_combined)
                print("  ✓ 已保存")
            
            return True
            
        except Exception as e:
            print(f"保存时出错: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_report(self, results):
        """生成分块检测报告"""
        print("\n生成分块检测报告...")
        
        try:
            report = []
            report.append("=" * 60)
            report.append("分层分块点云差异检测报告")
            report.append("=" * 60)
            report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            report.append("")
            
            # 基本信息
            total_pc1 = len(results['pc1_points'])
            total_pc2 = len(results['pc2_points'])
            processing_summary = results['processing_summary']
            
            report.append("点云规模:")
            report.append(f"  PC1: {total_pc1:,} 点")
            report.append(f"  PC2: {total_pc2:,} 点")
            report.append("")
            
            # 处理参数
            report.append("分块参数:")
            report.append(f"  粗分块大小: {processing_summary['coarse_grid_size']}m")
            report.append(f"  精细分块大小: {processing_summary['fine_grid_size']}m")
            report.append(f"  并行处理: {'是' if processing_summary['parallel_processing'] else '否'}")
            report.append(f"  处理的块数: {processing_summary['total_blocks_processed']}")
            report.append("")
            
            # 整体变化统计
            report.append("整体变化统计:")
            categories = ['unchanged', 'minor_change', 'moderate_change', 'major_change', 'removed']
            labels = ['未变化', '轻微变化', '中等变化', '重大变化', '已删除']
            
            for cat, label in zip(categories, labels):
                count = np.sum(results[cat])
                percentage = count / total_pc1 * 100 if total_pc1 > 0 else 0
                report.append(f"  {label}: {count:,} ({percentage:.2f}%)")
            
            report.append(f"  新增: {len(results['new_points']):,} 点")
            report.append("")
            
            # 检测质量评估
            report.append("检测质量评估:")
            change_ratio = (total_pc1 - np.sum(results['unchanged'])) / total_pc1 if total_pc1 > 0 else 0
            report.append(f"  总体变化率: {change_ratio*100:.1f}%")
            
            if change_ratio < 0.05:
                report.append("  评估: 变化很少，建议使用更敏感的参数")
            elif change_ratio > 0.5:
                report.append("  评估: 变化很多，建议检查配准质量或调整阈值")
            else:
                report.append("  评估: 变化程度适中，检测结果可信")
            
            report.append("")
            report.append("文件输出:")
            report.append("  - block_difference_result.ply (主要分类结果)")
            report.append("  - block_difference_combined.ply (组合视图)")
            if len(results['new_points']) > 0:
                report.append("  - block_new_points.ply (新增点)")
            if len(results['removed_points']) > 0:
                if len(results['removed_points']) > 50000:
                    report.append("  - block_removed_points_part*.ply (删除点-分批)")
                else:
                    report.append("  - block_removed_points.ply (删除点)")
            
            report_text = "\n".join(report)
            
            with open("block_difference_report.txt", "w", encoding="utf-8") as f:
                f.write(report_text)
            
            print("  ✓ 报告已保存到 block_difference_report.txt")
            
            return report_text
            
        except Exception as e:
            print(f"生成报告时出错: {e}")
            return None
    
    def run(self, pc1_path, pc2_path):
        """运行完整的分层分块差异检测流程"""
        print("\n" + "█" * 80)
        print("分层分块点云差异检测系统")
        print("█" * 80)
        
        # 检查文件
        if not os.path.exists(pc1_path):
            print(f"错误: 找不到文件 {pc1_path}")
            return None
        if not os.path.exists(pc2_path):
            print(f"错误: 找不到文件 {pc2_path}")
            return None
        
        print(f"\n输入文件:")
        print(f"  PC1 (设计): {pc1_path}")
        print(f"  PC2 (实际): {pc2_path}")
        
        print(f"\n分块策略参数:")
        print(f"  粗分块大小: {self.coarse_grid_size}m")
        print(f"  精细分块大小: {self.fine_grid_size}m")
        print(f"  最小块点数: {self.min_points_per_block}")
        print(f"  块重叠比例: {self.overlap_ratio}")
        print(f"  变化检测阈值: {self.change_threshold}")
        print(f"  并行处理: {'启用' if self.parallel_processing else '禁用'}")
        
        try:
            # 1. 加载和预处理
            pc1, pc2 = self.load_and_preprocess(pc1_path, pc2_path)
            
            # 2. 分层分块差异检测
            results = self.detect_differences(pc1, pc2)
            
            # 3. 保存结果
            success = self.save_results(pc1, pc2, results)
            
            if success:
                # 4. 生成报告
                self.generate_report(results)
                
                print("\n" + "█" * 80)
                print("✓ 分层分块检测完成！")
                print("\n核心改进特性:")
                print("  ✓ 两阶段分块策略 (粗分块 → 精细分块)")
                print("  ✓ 智能变化块筛选")
                print("  ✓ 自适应局部阈值")
                print("  ✓ 精确的新增/删除检测")
                print("  ✓ 并行处理提升效率")
                print("  ✓ 内存优化处理")
                print("\n生成的文件:")
                print("  - block_difference_result.ply (主要分类结果)")
                print("  - block_difference_combined.ply (组合视图)")
                if len(results['new_points']) > 0:
                    print(f"  - block_new_points.ply (新增点 - {len(results['new_points'])} 点)")
                if len(results['removed_points']) > 0:
                    if len(results['removed_points']) > 50000:
                        print(f"  - block_removed_points_part*.ply (删除点 - {len(results['removed_points'])} 点，分批保存)")
                    else:
                        print(f"  - block_removed_points.ply (删除点 - {len(results['removed_points'])} 点)")
                print("  - block_difference_report.txt (详细分析报告)")
                print("\n颜色编码:")
                print("  🟢 绿色 = 未变化")
                print("  🟡 黄绿 = 轻微变化") 
                print("  🟠 橙色 = 中等变化")
                print("  🔴 红色 = 重大变化")
                print("  🟣 紫色 = 删除点")
                print("  🔵 蓝色 = 新增点")
                print("█" * 80)
            
            return results
            
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """主函数 - 标准分块模式"""
    
    # 创建分层分块检测器
    detector = BlockBasedPointCloudDetector(
        voxel_size=0.02,           # 体素大小
        distance_threshold=0.05,   # 基础距离阈值
        max_points=100000,         # 最大处理点数
        coarse_grid_size=2.0,      # 粗分块大小 (2米)
        fine_grid_size=0.5,        # 精细分块大小 (0.5米)
        min_points_per_block=30,   # 每块最少点数
        change_threshold=0.1,      # 块变化阈值
        overlap_ratio=0.1,         # 块重叠比例
        parallel_processing=True   # 启用并行处理
    )
    
    # 运行检测
    results = detector.run("as_designed_model.ply", "as_built_model.ply")
    
    return results


def run_high_precision_blocks():
    """高精度分块模式"""
    print("\n运行高精度分块检测模式...")
    
    detector = BlockBasedPointCloudDetector(
        voxel_size=0.01,           # 更小的体素
        distance_threshold=0.02,   # 更严格的阈值
        max_points=150000,         # 更多点数
        coarse_grid_size=1.0,      # 更小的粗分块 (1米)
        fine_grid_size=0.2,        # 更小的精细分块 (0.2米)
        min_points_per_block=15,   # 更少的最小点数要求
        change_threshold=0.05,     # 更敏感的变化阈值
        overlap_ratio=0.15,        # 更大的重叠
        parallel_processing=True
    )
    
    results = detector.run("as_designed_model.ply", "as_built_model.ply")
    
    return results


def run_fast_blocks():
    """快速分块模式"""
    print("\n运行快速分块检测模式...")
    
    detector = BlockBasedPointCloudDetector(
        voxel_size=0.05,           # 较大的体素
        distance_threshold=0.1,    # 宽松的阈值
        max_points=50000,          # 较少点数
        coarse_grid_size=3.0,      # 较大的粗分块 (3米)
        fine_grid_size=1.0,        # 较大的精细分块 (1米)
        min_points_per_block=50,   # 更多的最小点数要求
        change_threshold=0.2,      # 较不敏感的变化阈值
        overlap_ratio=0.05,        # 较小的重叠
        parallel_processing=True
    )
    
    results = detector.run("as_designed_model.ply", "as_built_model.ply")
    
    return results


if __name__ == "__main__":
    print("分层分块点云差异检测系统")
    print("选择检测模式:")
    print("1. 标准分块模式 (推荐) - 平衡精度和效率")
    print("2. 高精度分块模式 - 最高精度但较慢")  
    print("3. 快速分块模式 - 快速预览")
    print("=" * 50)
    
    # 默认运行标准模式
    print("运行标准分块模式...")
    results = main()
    
    if results:
        print("\n✓ 检测完成！请查看生成的 .ply 文件和报告。")
    else:
        print("\n✗ 检测失败，请检查错误信息。")

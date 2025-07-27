#!/usr/bin/env python3
"""
批量计算FASTA文件中所有序列对的距离相关性并绘制热图
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import itertools
import typing
import collections
import os
import pickle
from pathlib import Path
# 添加以下导入
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
import matplotlib.patches as mpatches

# 数据路径
DATA_PATH = Path("/data2/IDR_LLM/my_DL/ProteinDistance/data")
PathType = str | os.PathLike[str]

def process_args() -> argparse.Namespace:
    processor = argparse.ArgumentParser(description="计算FASTA文件中所有序列对的距离相关性")
    processor.add_argument(
        "fasta_filepath",
        type=str,
        help="包含多个蛋白质序列的FASTA文件路径",
    )
    processor.add_argument(
        "--output_dir",
        type=str,
        default="./distance_analysis_output",
        help="输出目录路径"
    )
    processor.add_argument(
        "--plot_format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="图片保存格式"
    )
    # 新增参数
    processor.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="层次聚类的聚类数量，如果不指定则自动确定"
    )
    processor.add_argument(
        "--skip_pca",
        action="store_true",
        help="跳过PCA分析"
    )
    processor.add_argument(
        "--skip_clustering",
        action="store_true",
        help="跳过聚类分析"
    )
    return processor.parse_args()

def read_fasta_all(filepath: PathType) -> typing.Dict[str, str]:
    """
    读取FASTA文件中的所有序列
    
    Returns:
        Dict[seq_id, sequence]: 序列ID到序列的映射
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                # 保存前一个序列
                if current_id is not None:
                    sequences[current_id] = "".join(current_seq)
                # 开始新序列
                current_id = line[1:]  # 去掉 ">" 符号
                current_seq = []
            else:
                current_seq.append(line)
        
        # 保存最后一个序列
        if current_id is not None:
            sequences[current_id] = "".join(current_seq)
    
    return sequences

def get_aa_data(
    protein_sequence: str, 
    features: typing.Iterable[str], 
    index_data: dict[str, dict[str, float | int]],
) -> dict[str, list[float | int]]:
    """提取蛋白质序列的氨基酸特征数据"""
    aa_data = collections.defaultdict(list)
    for feat in features:
        for aa in protein_sequence:
            if aa in index_data[feat]:  # 检查氨基酸是否在数据中
                aa_data[feat].append(index_data[feat][aa])
            else:
                # 对于未知氨基酸，使用平均值
                avg_val = np.mean(list(index_data[feat].values()))
                aa_data[feat].append(avg_val)
    return aa_data

def seq_data_to_mat(aa_data: dict[str, list[float | int]]) -> np.ndarray:
    """将氨基酸数据转换为矩阵"""
    return np.array(list(aa_data.values()), dtype=np.float32)

def get_distance_correlation(seq_data_vec_1: np.ndarray, seq_data_vec_2: np.ndarray) -> float:
    """计算两个序列特征向量的距离相关性"""
    # 确保向量长度相同，如果不同则截断到较短的长度
    min_len = min(len(seq_data_vec_1), len(seq_data_vec_2))
    seq_data_vec_1 = seq_data_vec_1[:min_len]
    seq_data_vec_2 = seq_data_vec_2[:min_len]
    
    # 中心化向量
    centered_vec_1 = seq_data_vec_1 - seq_data_vec_1.mean()
    centered_vec_2 = seq_data_vec_2 - seq_data_vec_2.mean()
    
    # 计算距离相关性 (1 - cosine similarity)
    dot_product = centered_vec_1 @ centered_vec_2
    norm_product = np.linalg.norm(centered_vec_1) * np.linalg.norm(centered_vec_2)
    
    if norm_product == 0:
        return 1.0  # 如果向量为零向量，返回最大距离
    
    dxy = 1 - (dot_product / norm_product)
    return float(dxy)

def pad_sequences_fft(seq_data_mat_1: np.ndarray, seq_data_mat_2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    对两个序列的特征矩阵进行填充，使其长度相同，然后进行FFT变换
    """
    # 获取两个矩阵的长度
    len1 = seq_data_mat_1.shape[1]
    len2 = seq_data_mat_2.shape[1]
    max_len = max(len1, len2)
    
    # 用零填充到相同长度
    if len1 < max_len:
        padding = np.zeros((seq_data_mat_1.shape[0], max_len - len1))
        seq_data_mat_1 = np.hstack([seq_data_mat_1, padding])
    
    if len2 < max_len:
        padding = np.zeros((seq_data_mat_2.shape[0], max_len - len2))
        seq_data_mat_2 = np.hstack([seq_data_mat_2, padding])
    
    # 进行FFT变换
    seq_data_mat_1_fft = np.apply_along_axis(np.fft.fft, 1, seq_data_mat_1)
    seq_data_mat_2_fft = np.apply_along_axis(np.fft.fft, 1, seq_data_mat_2)
    
    return seq_data_mat_1_fft, seq_data_mat_2_fft

def run_pairwise(seq_1: str, seq_2: str, features: typing.Sequence[str], index_data: dict) -> float:
    """计算两个序列之间的距离相关性"""
    try:
        # 提取氨基酸特征数据
        seq_data_1 = get_aa_data(seq_1, features, index_data)
        seq_data_2 = get_aa_data(seq_2, features, index_data)
        
        # 转换为矩阵
        seq_data_mat_1 = seq_data_to_mat(seq_data_1)
        seq_data_mat_2 = seq_data_to_mat(seq_data_2)
        
        # 填充并进行FFT变换
        seq_data_mat_1_fft, seq_data_mat_2_fft = pad_sequences_fft(seq_data_mat_1, seq_data_mat_2)
        
        # 计算功率谱
        seq_data_vec_1 = np.absolute(seq_data_mat_1_fft.flatten()) ** 2
        seq_data_vec_2 = np.absolute(seq_data_mat_2_fft.flatten()) ** 2
        
        # 计算距离相关性
        dxy = get_distance_correlation(seq_data_vec_1, seq_data_vec_2)
        return dxy
        
    except Exception as e:
        print(f"计算序列对距离时出错: {e}")
        return np.nan

def compute_distance_matrix(sequences: dict, features: list, index_data: dict) -> tuple[np.ndarray, list]:
    """计算所有序列对的距离矩阵"""
    seq_ids = list(sequences.keys())
    n_seqs = len(seq_ids)
    distance_matrix = np.zeros((n_seqs, n_seqs))
    
    print(f"计算 {n_seqs} 个序列的距离矩阵...")
    
    for i in range(n_seqs):
        for j in range(n_seqs):
            if i == j:
                distance_matrix[i, j] = 0.0  # 自己与自己的距离为0
            elif i < j:
                # 只计算上三角矩阵
                seq_1 = sequences[seq_ids[i]]
                seq_2 = sequences[seq_ids[j]]
                distance = run_pairwise(seq_1, seq_2, features, index_data)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # 矩阵对称
                print(f"进度: {i+1}/{n_seqs}, {j+1}/{n_seqs}, 距离: {distance:.4f}")
    
    return distance_matrix, seq_ids

def plot_distance_heatmap(distance_matrix: np.ndarray, seq_ids: list, output_dir: str, format: str = "png", input_filename: str = ""):
    """绘制距离矩阵热图"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置图形大小
    plt.figure(figsize=(12, 10))
    
    # 绘制热图
    sns.heatmap(
        distance_matrix, 
        xticklabels=seq_ids, 
        yticklabels=seq_ids,
        annot=True if len(seq_ids) <= 10 else False,  # 序列太多时不显示数值
        fmt='.3f',
        cmap='viridis',
        cbar_kws={'label': 'Distance Correlation'}
    )
    
    plt.title('Pairwise Distance Correlation Matrix', fontsize=5)
    plt.xlabel('Sequence ID', fontsize=5)
    plt.ylabel('Sequence ID', fontsize=5)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图片 - 使用输入文件名作为前缀
    base_name = Path(input_filename).stem if input_filename else "distance_heatmap"
    output_path = os.path.join(output_dir, f"{base_name}_distance_heatmap.{format}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"热图已保存到: {output_path}")


"""

# 添加PCA分析函数
def perform_pca_analysis(distance_matrix: np.ndarray, seq_ids: list, output_dir: str, input_filename: str = ""):
    '''
    基于距离矩阵进行PCA分析
    
    Args:
        distance_matrix: 距离矩阵
        seq_ids: 序列ID列表
        output_dir: 输出目录
        input_filename: 输入文件名
    '''
    print("开始PCA分析...")
    
    # 由于距离矩阵是对称的，我们需要将其转换为特征向量
    # 方法1: 使用多维标度(MDS)的思想，将距离转换为坐标
    from sklearn.manifold import MDS
    
    # 使用MDS将距离矩阵转换为坐标
    mds = MDS(n_components=min(len(seq_ids)-1, 10), dissimilarity='precomputed', random_state=42)
    mds_coords = mds.fit_transform(distance_matrix)
    
    # 对MDS坐标进行PCA
    pca = PCA()
    pca_coords = pca.fit_transform(mds_coords)
    
    # 计算解释方差比例
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # 创建PCA结果DataFrame
    pca_df = pd.DataFrame(pca_coords, index=seq_ids)
    pca_df.columns = [f'PC{i+1}' for i in range(pca_coords.shape[1])]
    
    # 保存PCA结果
    base_name = Path(input_filename).stem if input_filename else "pca_analysis"
    pca_output_path = os.path.join(output_dir, f"{base_name}_pca_results.csv")
    pca_df.to_csv(pca_output_path)
    
    # 绘制PCA图形
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 解释方差图
    axes[0, 0].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    axes[0, 0].set_xlabel('主成分')
    axes[0, 0].set_ylabel('解释方差比例')
    axes[0, 0].set_title('PCA解释方差')
    
    # 在柱状图上添加数值
    for i, v in enumerate(explained_variance_ratio):
        axes[0, 0].text(i + 1, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. 累积解释方差图
    axes[0, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-')
    axes[0, 1].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95%')
    axes[0, 1].set_xlabel('主成分数量')
    axes[0, 1].set_ylabel('累积解释方差比例')
    axes[0, 1].set_title('累积解释方差')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. PC1 vs PC2 散点图
    scatter = axes[1, 0].scatter(pca_coords[:, 0], pca_coords[:, 1], alpha=0.7, s=50)
    axes[1, 0].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%})')
    axes[1, 0].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%})')
    axes[1, 0].set_title('PCA散点图 (PC1 vs PC2)')
    
    # 添加序列ID标签（如果序列不太多）
    if len(seq_ids) <= 50:
        for i, seq_id in enumerate(seq_ids):
            axes[1, 0].annotate(seq_id, (pca_coords[i, 0], pca_coords[i, 1]), 
                              xytext=(5, 5), textcoords='offset points', 
                              fontsize=8, alpha=0.7)
    
    # 4. PC3 vs PC1 散点图（如果有第三个主成分）
    if pca_coords.shape[1] >= 3:
        scatter2 = axes[1, 1].scatter(pca_coords[:, 0], pca_coords[:, 2], alpha=0.7, s=50)
        axes[1, 1].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%})')
        axes[1, 1].set_ylabel(f'PC3 ({explained_variance_ratio[2]:.1%})')
        axes[1, 1].set_title('PCA散点图 (PC1 vs PC3)')
        
        if len(seq_ids) <= 50:
            for i, seq_id in enumerate(seq_ids):
                axes[1, 1].annotate(seq_id, (pca_coords[i, 0], pca_coords[i, 2]), 
                                  xytext=(5, 5), textcoords='offset points', 
                                  fontsize=8, alpha=0.7)
    else:
        axes[1, 1].text(0.5, 0.5, '数据维度不足\n无法显示PC3', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    
    # 保存PCA图形
    pca_plot_path = os.path.join(output_dir, f"{base_name}_pca_analysis.pdf")
    plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"PCA分析结果已保存到: {pca_output_path}")
    print(f"PCA图形已保存到: {pca_plot_path}")
    
    # 输出PCA统计信息
    print(f"\n=== PCA分析统计 ===")
    print(f"总主成分数量: {len(explained_variance_ratio)}")
    print(f"前3个主成分解释的方差: {cumulative_variance[2]:.1%}" if len(cumulative_variance) >= 3 else f"前{len(cumulative_variance)}个主成分解释的方差: {cumulative_variance[-1]:.1%}")
    print(f"达到95%方差需要的主成分数量: {np.argmax(cumulative_variance >= 0.95) + 1}")
    
    return pca_df, explained_variance_ratio
"""

def perform_pca_analysis(distance_matrix: np.ndarray, seq_ids: list, output_dir: str, input_filename: str = ""):
    """
    基于距离矩阵进行PCA分析
    
    Args:
        distance_matrix: 距离矩阵
        seq_ids: 序列ID列表
        output_dir: 输出目录
        input_filename: 输入文件名
    """
    print("开始PCA分析...")
    
    # 由于距离矩阵是对称的，我们需要将其转换为特征向量
    # 方法1: 使用多维标度(MDS)的思想，将距离转换为坐标
    from sklearn.manifold import MDS
    
    # 使用MDS将距离矩阵转换为坐标
    mds = MDS(n_components=min(len(seq_ids)-1, 10), dissimilarity='precomputed', random_state=42)
    mds_coords = mds.fit_transform(distance_matrix)
    
    # 对MDS坐标进行PCA
    pca = PCA()
    pca_coords = pca.fit_transform(mds_coords)
    
    # 计算解释方差比例
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # 创建PCA结果DataFrame
    pca_df = pd.DataFrame(pca_coords, index=seq_ids)
    pca_df.columns = [f'PC{i+1}' for i in range(pca_coords.shape[1])]
    
    # 保存PCA结果
    base_name = Path(input_filename).stem if input_filename else "pca_analysis"
    pca_output_path = os.path.join(output_dir, f"{base_name}_pca_results.csv")
    pca_df.to_csv(pca_output_path)
    
    # 绘制PCA图形 - 原始4子图版本
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 解释方差图
    axes[0, 0].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    axes[0, 0].set_xlabel('principal component')   # 主成分
    axes[0, 0].set_ylabel('Proportion of explained variance') # 解释方差比例
    axes[0, 0].set_title('PCA explained variance') # PCA解释方差
    
    # 在柱状图上添加数值
    for i, v in enumerate(explained_variance_ratio):
        axes[0, 0].text(i + 1, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # 2. 累积解释方差图
    axes[0, 1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-')
    axes[0, 1].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95%')
    axes[0, 1].set_xlabel('Number of principal components') # 主成分数量
    axes[0, 1].set_ylabel('Cumulative proportion of explained variance')  # 累积解释方差比例
    axes[0, 1].set_title('Cumulative explained variance') # 累积解释方差
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. PC1 vs PC2 散点图
    scatter = axes[1, 0].scatter(pca_coords[:, 0], pca_coords[:, 1], alpha=0.7, s=50)
    axes[1, 0].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%})')
    axes[1, 0].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%})')
    axes[1, 0].set_title('PC1 vs PC2') # PCA散点图
    
    # 添加序列ID标签（如果序列不太多）
    if len(seq_ids) <= 50:
        for i, seq_id in enumerate(seq_ids):
            axes[1, 0].annotate(seq_id, (pca_coords[i, 0], pca_coords[i, 1]), 
                              xytext=(5, 5), textcoords='offset points', 
                              fontsize=5, alpha=0.7)
    
    # 4. PC3 vs PC1 散点图（如果有第三个主成分）
    if pca_coords.shape[1] >= 3:
        scatter2 = axes[1, 1].scatter(pca_coords[:, 0], pca_coords[:, 2], alpha=0.7, s=50)
        axes[1, 1].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%})')
        axes[1, 1].set_ylabel(f'PC3 ({explained_variance_ratio[2]:.1%})')
        axes[1, 1].set_title('PC1 vs PC3') # PCA散点图
        
        if len(seq_ids) <= 50:
            for i, seq_id in enumerate(seq_ids):
                axes[1, 1].annotate(seq_id, (pca_coords[i, 0], pca_coords[i, 2]), 
                                  xytext=(5, 5), textcoords='offset points', 
                                  fontsize=5, alpha=0.7)
    else:
        axes[1, 1].text(0.5, 0.5, '数据维度不足\n无法显示PC3', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    
    # 保存PCA图形
    pca_plot_path = os.path.join(output_dir, f"{base_name}_pca_analysis.pdf")
    plt.savefig(pca_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # === 新增：创建专门的PCA样本分类效果图 ===
    create_pca_classification_plots(pca_coords, seq_ids, explained_variance_ratio, 
                                   output_dir, base_name)
    
    print(f"PCA分析结果已保存到: {pca_output_path}")
    print(f"PCA图形已保存到: {pca_plot_path}")
    
    # 输出PCA统计信息
    print(f"\n=== PCA分析统计 ===")
    print(f"总主成分数量: {len(explained_variance_ratio)}")
    print(f"前3个主成分解释的方差: {cumulative_variance[2]:.1%}" if len(cumulative_variance) >= 3 else f"前{len(cumulative_variance)}个主成分解释的方差: {cumulative_variance[-1]:.1%}")
    print(f"达到95%方差需要的主成分数量: {np.argmax(cumulative_variance >= 0.95) + 1}")
    
    return pca_df, explained_variance_ratio

def create_pca_classification_plots(pca_coords: np.ndarray, seq_ids: list, 
                                   explained_variance_ratio: np.ndarray, 
                                   output_dir: str, base_name: str):
    """
    创建专门的PCA样本分类效果图
    
    Args:
        pca_coords: PCA坐标
        seq_ids: 序列ID列表
        explained_variance_ratio: 解释方差比例
        output_dir: 输出目录
        base_name: 基础文件名
    """
    # 根据样本数量决定图形布局
    n_samples = len(seq_ids)
    
    # 创建颜色映射
    # 如果序列ID包含特定模式，可以根据模式分组
    colors = plt.cm.tab20(np.linspace(0, 1, n_samples))
    
    # 尝试根据序列ID模式进行分组（例如top/bottom, cluster等）
    groups = detect_sample_groups(seq_ids)
    
    # === 图1: PC1 vs PC2 详细分类图 ===
    fig1, ax1 = plt.subplots(figsize=(14, 10))
    
    # 根据分组绘制散点图
    if groups:
        unique_groups = list(set(groups.values()))
        group_colors = plt.cm.Set1(np.linspace(0, 1, len(unique_groups)))
        
        for i, (group_name, color) in enumerate(zip(unique_groups, group_colors)):
            group_indices = [j for j, seq_id in enumerate(seq_ids) if groups[seq_id] == group_name]
            group_coords = pca_coords[group_indices]
            
            ax1.scatter(group_coords[:, 0], group_coords[:, 1], 
                       c=[color], s=80, alpha=0.7, label=group_name, 
                       edgecolors='black', linewidth=0.5)
        
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        # 如果没有明显分组，使用渐变色
        scatter = ax1.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                             c=range(n_samples), cmap='viridis', 
                             s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax1, label='sample index') # 样本索引
    
    # 添加样本标签
    for i, seq_id in enumerate(seq_ids):
        # 根据样本数量调整标签策略
        if n_samples <= 30:
            # 少量样本：显示完整ID
            ax1.annotate(seq_id, (pca_coords[i, 0], pca_coords[i, 1]), 
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=5, alpha=0.8, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        elif n_samples <= 100:
            # 中等样本：显示简化ID
            short_id = seq_id[:8] + '...' if len(seq_id) > 8 else seq_id
            ax1.annotate(short_id, (pca_coords[i, 0], pca_coords[i, 1]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=5, alpha=0.6)
        else:
            # 大量样本：只显示索引
            ax1.annotate(str(i), (pca_coords[i, 0], pca_coords[i, 1]), 
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=5, alpha=0.5)
    
    ax1.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%})', fontsize=5)
    ax1.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%})', fontsize=5)
    ax1.set_title('PC1 vs PC2', fontsize=5, fontweight='bold')  # PCA样本分类图
    ax1.grid(True, alpha=0.3)
    
    # 添加原点线
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # 保存PC1 vs PC2图
    pc12_plot_path = os.path.join(output_dir, f"{base_name}_pca_pc1_pc2_classification.pdf")
    plt.savefig(pc12_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # === 图2: 如果有PC3，绘制PC1 vs PC3图 ===
    if pca_coords.shape[1] >= 3:
        fig2, ax2 = plt.subplots(figsize=(14, 10))
        
        if groups:
            for i, (group_name, color) in enumerate(zip(unique_groups, group_colors)):
                group_indices = [j for j, seq_id in enumerate(seq_ids) if groups[seq_id] == group_name]
                group_coords = pca_coords[group_indices]
                
                ax2.scatter(group_coords[:, 0], group_coords[:, 2], 
                           c=[color], s=80, alpha=0.7, label=group_name,
                           edgecolors='black', linewidth=0.5)
            
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            scatter2 = ax2.scatter(pca_coords[:, 0], pca_coords[:, 2], 
                                 c=range(n_samples), cmap='viridis', 
                                 s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
            plt.colorbar(scatter2, ax=ax2, label='sample index')  # 样本索引
        
        # 添加样本标签
        for i, seq_id in enumerate(seq_ids):
            if n_samples <= 30:
                ax2.annotate(seq_id, (pca_coords[i, 0], pca_coords[i, 2]), 
                            xytext=(8, 8), textcoords='offset points',
                            fontsize=5, alpha=0.8,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            elif n_samples <= 100:
                short_id = seq_id[:8] + '...' if len(seq_id) > 8 else seq_id
                ax2.annotate(short_id, (pca_coords[i, 0], pca_coords[i, 2]), 
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=5, alpha=0.6)
            else:
                ax2.annotate(str(i), (pca_coords[i, 0], pca_coords[i, 2]), 
                            xytext=(3, 3), textcoords='offset points',
                            fontsize=5, alpha=0.5)
        
        ax2.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%})', fontsize=5)
        ax2.set_ylabel(f'PC3 ({explained_variance_ratio[2]:.1%})', fontsize=5)
        ax2.set_title('PC1 vs PC3', fontsize=5, fontweight='bold') # PCA样本分类图s
        ax2.grid(True, alpha=0.3)
        
        # 添加原点线
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # 保存PC1 vs PC3图
        pc13_plot_path = os.path.join(output_dir, f"{base_name}_pca_pc1_pc3_classification.pdf")
        plt.savefig(pc13_plot_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    # === 图3: 3D PCA图（如果有至少3个主成分）===
    if pca_coords.shape[1] >= 3:
        fig3 = plt.figure(figsize=(12, 9))
        ax3 = fig3.add_subplot(111, projection='3d')
        
        if groups:
            for i, (group_name, color) in enumerate(zip(unique_groups, group_colors)):
                group_indices = [j for j, seq_id in enumerate(seq_ids) if groups[seq_id] == group_name]
                group_coords = pca_coords[group_indices]
                
                ax3.scatter(group_coords[:, 0], group_coords[:, 1], group_coords[:, 2],
                           c=[color], s=60, alpha=0.7, label=group_name, 
                           edgecolors='black', linewidth=0.5)
            
            ax3.legend()
        else:
            scatter3d = ax3.scatter(pca_coords[:, 0], pca_coords[:, 1], pca_coords[:, 2],
                                   c=range(n_samples), cmap='viridis', s=60, alpha=0.8)
        
        # 只为少量样本添加3D标签（避免过于拥挤）
        if n_samples <= 20:
            for i, seq_id in enumerate(seq_ids):
                ax3.text(pca_coords[i, 0], pca_coords[i, 1], pca_coords[i, 2], 
                        seq_id, fontsize=5)
        
        ax3.set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%})')
        ax3.set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%})')
        ax3.set_zlabel(f'PC3 ({explained_variance_ratio[2]:.1%})')
        ax3.set_title('PCA 3D', fontsize=5, fontweight='bold') # PCA 3D样本分类图s
        
        # 保存3D图
        pca_3d_plot_path = os.path.join(output_dir, f"{base_name}_pca_3d_classification.pdf")
        plt.savefig(pca_3d_plot_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    print(f"PCA分类效果图已保存到: {output_dir}")
    
    # === 创建样本ID映射表 ===
    id_mapping_path = os.path.join(output_dir, f"{base_name}_sample_id_mapping.csv")
    mapping_df = pd.DataFrame({
        'Index': range(len(seq_ids)),
        'Sample_ID': seq_ids,
        'PC1': pca_coords[:, 0],
        'PC2': pca_coords[:, 1],
        'PC3': pca_coords[:, 2] if pca_coords.shape[1] >= 3 else np.nan,
        'Group': [groups.get(seq_id, 'Unknown') if groups else 'No_Group' for seq_id in seq_ids]
    })
    mapping_df.to_csv(id_mapping_path, index=False)
    print(f"样本ID映射表已保存到: {id_mapping_path}")

def detect_sample_groups(seq_ids: list) -> dict:
    """
    根据序列ID模式检测样本分组
    
    Args:
        seq_ids: 序列ID列表
    
    Returns:
        groups: 序列ID到分组名的映射字典
    """
    groups = {}
    
    # 检测常见的分组模式
    for seq_id in seq_ids:
        seq_id_lower = seq_id.lower()
        
        # 检测top/bottom模式
        if 'top' in seq_id_lower:
            groups[seq_id] = 'Top'
        elif 'bottom' in seq_id_lower:
            groups[seq_id] = 'Bottom'
        # 检测cluster模式
        elif 'cluster' in seq_id_lower or 'clust' in seq_id_lower:
            # 尝试提取cluster编号
            import re
            cluster_match = re.search(r'cluster[_\s]*(\d+)', seq_id_lower)
            if cluster_match:
                groups[seq_id] = f'Cluster_{cluster_match.group(1)}'
            else:
                groups[seq_id] = 'Cluster'
        # 检测其他数字模式
        elif any(char.isdigit() for char in seq_id):
            # 根据数字范围分组
            import re
            numbers = re.findall(r'\d+', seq_id)
            if numbers:
                num = int(numbers[0])
                if num <= 100:
                    groups[seq_id] = 'Group_1-100'
                elif num <= 200:
                    groups[seq_id] = 'Group_101-200'
                else:
                    groups[seq_id] = 'Group_200+'
        else:
            groups[seq_id] = 'Other'
    
    # 如果所有样本都被分到同一组，返回空字典（不分组）
    if len(set(groups.values())) == 1:
        return {}
    
    return groups



# 添加层次聚类分析函数
def perform_hierarchical_clustering(distance_matrix: np.ndarray, seq_ids: list, 
                                  output_dir: str, input_filename: str = "", 
                                  n_clusters: int = None):
    """
    基于距离矩阵进行层次聚类分析
    
    Args:
        distance_matrix: 距离矩阵
        seq_ids: 序列ID列表
        output_dir: 输出目录
        input_filename: 输入文件名
        n_clusters: 聚类数量，如果为None则自动确定
    """
    print("开始层次聚类分析...")
    
    # 将距离矩阵转换为压缩距离矩阵
    condensed_dist = squareform(distance_matrix)
    
    # 进行层次聚类
    linkage_matrix = linkage(condensed_dist, method='ward')
    
    # 如果没有指定聚类数量，使用肘部法则自动确定
    if n_clusters is None:
        # 计算不同聚类数量的组内平方和
        max_clusters = min(10, len(seq_ids) - 1)
        inertias = []
        cluster_range = range(2, max_clusters + 1)
        
        for k in cluster_range:
            clusters = fcluster(linkage_matrix, k, criterion='maxclust')
            # 计算组内平方和
            inertia = 0
            for cluster_id in range(1, k + 1):
                cluster_indices = np.where(clusters == cluster_id)[0]
                if len(cluster_indices) > 1:
                    cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                    inertia += np.sum(cluster_distances) / len(cluster_indices)
            inertias.append(inertia)
        
        # 使用肘部法则
        if len(inertias) > 1:
            diffs = np.diff(inertias)
            second_diffs = np.diff(diffs)
            if len(second_diffs) > 0:
                elbow_idx = np.argmax(second_diffs) + 2
                n_clusters = cluster_range[elbow_idx] if elbow_idx < len(cluster_range) else cluster_range[-1]
            else:
                n_clusters = 3
        else:
            n_clusters = 2
    
    # 获取聚类标签
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    
    # 创建聚类结果DataFrame
    cluster_df = pd.DataFrame({
        'Sequence_ID': seq_ids,
        'Cluster': cluster_labels
    })
    
    # 保存聚类结果
    base_name = Path(input_filename).stem if input_filename else "clustering_analysis"
    cluster_output_path = os.path.join(output_dir, f"{base_name}_clustering_results.csv")
    cluster_df.to_csv(cluster_output_path, index=False)
    
    # 创建可视化
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. 树状图
    dendrogram(linkage_matrix, labels=seq_ids, ax=axes[0, 0], 
               orientation='top', leaf_rotation=45)
    axes[0, 0].set_title(f'Hierarchical clustering dendrogram(clustering numbers:{n_clusters})') # 层次聚类树状图
    axes[0, 0].set_xlabel('Sequence ID') # 序列ID
    axes[0, 0].set_ylabel('distance') # 距离
    
    # 2. 带聚类标注的热图
    # 根据聚类重新排序
    cluster_order = np.argsort(cluster_labels)
    reordered_matrix = distance_matrix[np.ix_(cluster_order, cluster_order)]
    reordered_ids = [seq_ids[i] for i in cluster_order]
    reordered_labels = cluster_labels[cluster_order]
    
    # 创建聚类颜色映射
    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
    cluster_colors = [colors[cluster - 1] for cluster in reordered_labels]
    
    im = axes[0, 1].imshow(reordered_matrix, cmap='viridis', aspect='auto')
    axes[0, 1].set_title('Reordered distance matrix(grouped by cluster)') # 重排序的距离矩阵（按聚类分组）
    axes[0, 1].set_xticks(range(len(reordered_ids)))
    axes[0, 1].set_yticks(range(len(reordered_ids)))
    
    if len(reordered_ids) <= 50:
        axes[0, 1].set_xticklabels(reordered_ids, rotation=45, ha='right')
        axes[0, 1].set_yticklabels(reordered_ids)
    
    # 添加聚类边界线
    cluster_boundaries = []
    current_cluster = reordered_labels[0]
    for i, cluster in enumerate(reordered_labels):
        if cluster != current_cluster:
            cluster_boundaries.append(i - 0.5)
            current_cluster = cluster
    
    for boundary in cluster_boundaries:
        axes[0, 1].axhline(boundary, color='red', linewidth=2)
        axes[0, 1].axvline(boundary, color='red', linewidth=2)
    
    plt.colorbar(im, ax=axes[0, 1], label='distance') # 距离
    
    # 3. 聚类大小分布
    cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
    bars = axes[1, 0].bar(cluster_counts.index, cluster_counts.values, 
                         color=[colors[i-1] for i in cluster_counts.index])
    axes[1, 0].set_title('Cluster size distribution') # 聚类大小分布
    axes[1, 0].set_xlabel('Clustering ID') # 聚类ID
    axes[1, 0].set_ylabel('Number of sequences') # 序列数量
    
    # 在柱状图上添加数值
    for bar, count in zip(bars, cluster_counts.values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom')
    
    # 4. 如果进行了肘部法则分析，显示结果
    if 'inertias' in locals():
        axes[1, 1].plot(cluster_range, inertias, 'o-', label='sum of squares within classes') # 组内平方和
        axes[1, 1].axvline(n_clusters, color='red', linestyle='--', 
                          label=f'Number of selected clusters: {n_clusters}') # 选择的聚类数
        axes[1, 1].set_xlabel('Number of clusters') # 聚类数量
        axes[1, 1].set_ylabel('sum of squares within classes') # 组内平方和
        axes[1, 1].set_title('optimal number of clusters selected by elbow rule') # 肘部法则选择最优聚类数
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, f'A prespecified number of clusters was used: {n_clusters}', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=5)
        # 使用预聚类数
    
    plt.tight_layout()
    
    # 保存聚类图形
    cluster_plot_path = os.path.join(output_dir, f"{base_name}_clustering_analysis.pdf")
    plt.savefig(cluster_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"聚类分析结果已保存到: {cluster_output_path}")
    print(f"聚类图形已保存到: {cluster_plot_path}")
    
    # 输出聚类统计信息
    print(f"\n=== 层次聚类统计 ===")
    print(f"聚类数量: {n_clusters}")
    print(f"各聚类大小: {dict(cluster_counts)}")
    
    # 计算轮廓系数
    try:
        from sklearn.metrics import silhouette_score
        if len(np.unique(cluster_labels)) > 1:
            silhouette_avg = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
            print(f"平均轮廓系数: {silhouette_avg:.3f}")
    except ImportError:
        print("sklearn未安装，跳过轮廓系数计算")
    
    return cluster_df, linkage_matrix



def save_distance_matrix(distance_matrix: np.ndarray, seq_ids: list, output_dir: str, input_filename: str = ""):
    """保存距离矩阵为CSV文件"""
    # 创建DataFrame
    df = pd.DataFrame(distance_matrix, index=seq_ids, columns=seq_ids)
    
    # 保存CSV - 使用输入文件名作为前缀
    base_name = Path(input_filename).stem if input_filename else "distance_matrix"
    output_path = os.path.join(output_dir, f"{base_name}_distance_matrix.csv")
    df.to_csv(output_path)
    
    print(f"距离矩阵已保存到: {output_path}")
    
    return df

def main(args: argparse.Namespace) -> int:
    """主函数"""
    # 加载特征数据，此处的特征数据，主要是原始github仓库中的文件，需要绝对路径额外加载
    print("加载特征数据...")
    aa_features = pd.read_csv(DATA_PATH / "aa_features.csv")
    
    # if not Path("aaindex_bin").exists():
    #    raise FileNotFoundError("aaindex_bin not found. Please run `python3 split_aaindex_data.py` to generate it.")
    
    # 此处选择直接加载绝对路径的aaindex_bin，即/data2/IDR_LLM/my_DL/ProteinDistance/aaindex_bin
    with open("/data2/IDR_LLM/my_DL/ProteinDistance/aaindex_bin", "rb") as f:
        aa_data = pickle.load(f)
    
    # 加载额外特征
    long_range_contact_data = pd.read_csv(DATA_PATH / "long_range_contacts.csv")
    long_range_contact_data = dict(zip(long_range_contact_data.columns, long_range_contact_data.iloc[0]))
    
    relative_connectivity_data = pd.read_csv(DATA_PATH / "relative_connectivity.csv")
    relative_connectivity_data = dict(zip(relative_connectivity_data.columns, relative_connectivity_data.iloc[0]))
    
    aa_data["Nl"] = long_range_contact_data
    aa_data["Rk"] = relative_connectivity_data
    
    features = aa_features["ID"].tolist()
    
    # 读取所有序列
    print(f"读取FASTA文件: {args.fasta_filepath}")
    sequences = read_fasta_all(args.fasta_filepath)
    print(f"共读取 {len(sequences)} 个序列")
    
    if len(sequences) < 2:
        raise ValueError("FASTA文件中至少需要2个序列")
    
    # 计算距离矩阵
    distance_matrix, seq_ids = compute_distance_matrix(sequences, features, aa_data)
    
    # 保存距离矩阵
    save_distance_matrix(distance_matrix, seq_ids, args.output_dir, args.fasta_filepath)
    
    # 绘制热图
    plot_distance_heatmap(distance_matrix, seq_ids, args.output_dir, args.plot_format, args.fasta_filepath)
    
    # 新增：PCA分析
    print("\n" + "="*50)
    pca_results, explained_variance = perform_pca_analysis(
        distance_matrix, seq_ids, args.output_dir, args.fasta_filepath
    )
    
    # 新增：层次聚类分析
    print("\n" + "="*50)
    cluster_results, linkage_matrix = perform_hierarchical_clustering(
        distance_matrix, seq_ids, args.output_dir, args.fasta_filepath
    )
    
    # 输出统计信息
    print("\n=== 距离矩阵统计 ===")
    print(f"矩阵大小: {distance_matrix.shape}")
    print(f"最小距离: {np.min(distance_matrix[distance_matrix > 0]):.4f}")
    print(f"最大距离: {np.max(distance_matrix):.4f}")
    print(f"平均距离: {np.mean(distance_matrix[distance_matrix > 0]):.4f}")
    print(f"标准差: {np.std(distance_matrix[distance_matrix > 0]):.4f}")
    
    return 0

if __name__ == "__main__":
    args = process_args()
    raise SystemExit(main(args))


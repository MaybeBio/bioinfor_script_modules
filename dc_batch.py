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
    
    plt.title('Pairwise Distance Correlation Matrix', fontsize=16)
    plt.xlabel('Sequence ID', fontsize=12)
    plt.ylabel('Sequence ID', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # 保存图片 - 使用输入文件名作为前缀
    base_name = Path(input_filename).stem if input_filename else "distance_heatmap"
    output_path = os.path.join(output_dir, f"{base_name}_distance_heatmap.{format}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"热图已保存到: {output_path}")

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

# 主要是加了一个随机矩阵理论中马尔琴可-巴斯德分布的noise过滤

# version1
def PCA_analysis(obs_matrix,centered=True,clean_matrix=False,clean_threshold=1.0,write_PC_scores=False,write_path=''):
    """
        Args:
            obs_matrix: 观测矩阵，样本x特征（我们假设输入的数据，都是每一列是1个feature）
            centered: 是否中心化数据
            clean_matrix: 是否清理矩阵（降噪处理），也就是是否要使用MP分布
            clean_threshold: 清理矩阵的阈值，默认1.0，也就是对MP分布极限（MP limit）的一个缩放因子，1是没有区别，如果是乘上一个大的数，就是更严格地对特征值的一个过滤        
            write_PC_scores: 是否保留主成分分数，默认为False
            write_path: 主成分得分文件输出路径（如果write_PC_scores为True）
            
        Fun:
        传入观测矩阵，格式为(nsamples,nfeatures)，可选择是否进行均值中心化 + 标准化归一化（默认为True）；
            PCA分析，返回特征值、特征向量和主成分得分
    
    """
    import numpy as np
    import os

    # 首先是数据预处理，也就是数据中心化以及标准化
    # 注意是对每一列的特征进行处理，所以要注意axis参数（我之前博客中提到过这个轴的记忆口诀，就是行0行，列1列）
    if centered:
        mean_vec = np.mean(obs_matrix,axis=0)
        std_vec = np.std(obs_matrix,axis=0)
        
        # 然后就是归一化之后的数据
        z_matrix = (obs_matrix-mean_vec)/std_vec
    # 如果不进行标准化处理，则使用原始观测矩阵
    else:
        z_matrix = obs_matrix
    
    # 然后就是计算协方差矩阵
    # 计算协方差矩阵：X^T * X / (n-1)
    # 此处使用除以样本数-1，是为了获取无偏估计
    cov_mat = (z_matrix).T.dot(z_matrix)/ (z_matrix.shape[0]-1)

    # 在计算出协方差矩阵之后就是特征值分解，获取特征值和特征向量
    eig_vals,eig_vecs = np.linalg.eig(cov_mat)
    # 然后对特征值进行排序，降序排列
    key = np.argsort(eig_vals)[::-1]

    # 然后特征值和特征向量按照排序之后的索引重新组织，都是降序
    eig_vals,eig_vecs = eig_vals[key],eig_vecs[:,key]   

    # 如果需要清理矩阵，也就是降噪处理
    # 此处使用随机矩阵理论中的Marchenko-Pastur Distribution，也就是MP分布，来决定哪些特征值是我们想要的、哪些不是我们想要的，也就是降噪
    if clean_matrix:
        # 在计算MP分布的时候，我们需要sample/feature数的比值，实际上就是Accept ratio
        s_f_ratio = obs_matrix.shape[0]/(obs_matrix.shape[1])
        # 然后就是计算MP limit（上界），即符合MP分布的最大特征值，此处公式需要数学验证⚠️
        lamda_max = 1 + 1/s_f_ratio + 2/np.sqrt(s_f_ratio)

        # 依据MP分布的上界和阈值，找到需要清理的特征值
        # 注意这里的clean_threshold是一个缩放因子，默认1.0，也就是MP分布的极限
        # 如果需要更严格的清理，可以将clean_threshold设置为大于1的
        # 找到第1个小于阈值的特征值位置
        pos = np.where(eig_vals < clean_threshold*lamda_max)[0][0]
        # 重构清理后的协方差矩阵，保留前pos个主成分（因为是降序排列的）
        cleaned_cov = (eig_vecs[:,:pos].dot(np.diag(eig_vals[:pos]))).dot(eig_vecs[:,:pos].T)
        #重新计算清理后矩阵的特征值和特征向量
        eig_vals,eig_vecs = np.linalg.eig(cleaned_cov)
        # 再次按照特征值大小进行降序排列
        key = np.argsort(eig_vals)[::-1]
        eig_vals,eig_vecs = eig_vals[key],eig_vecs[:,key]

    # 计算主成分得分：原始数据投影到主成分空间
    pc_scores = (z_matrix.dot(eig_vecs)).astype(float)
    
    # 如果需要保存主成分得分
    if write_PC_scores:
        # 创建标签列:pc_1,pc_2...
        col_labels = ['pc_'+str(x+1) for x in range(len(eig_vals))]
        # 创建dataframe
        df = pd.DataFrame(pc_scores,column=col_labels)
        os.makedirs(write_path,exist_ok=True)
        # 构建文件名，包含清理标志
        file_name = write_path +'/pca_clean_' + str(clean_matrix)+".csv"
        # 保存为csv文件
        df.to_csv(file_name)
    # 返回特征值、特征向量、主成分得分
    return eig_vals,eig_vecs,pc_scores



#  version2，简化版本
def PCA_analysis(obs_matrix, centered=True, clean_matrix=False, clean_threshold=2.0):
    """
    PCA主成分分析函数
    """
    print("=== PCA分析开始 ===")
    print(f"输入数据形状: {obs_matrix.shape} (样本数: {obs_matrix.shape[0]}, 特征数: {obs_matrix.shape[1]})")
    
    # 数据预处理
    if centered:
        print("\n1. 数据中心化和标准化:")
        mean_vec = np.mean(obs_matrix, axis=0)
        std_vec = np.std(obs_matrix, axis=0)
        print(f"   原始数据均值: {mean_vec}")
        print(f"   原始数据标准差: {std_vec}")
        
        z_matrix = (obs_matrix - mean_vec) / std_vec
        print(f"   标准化后数据的前3行:\n{z_matrix[:3]}")
    else:
        z_matrix = obs_matrix

    # 计算协方差矩阵
    print("\n2. 计算协方差矩阵:")
    cov_mat = (z_matrix).T.dot(z_matrix) / (z_matrix.shape[0] - 1)
    print(f"   协方差矩阵形状: {cov_mat.shape}")
    print(f"   协方差矩阵:\n{cov_mat}")
    
    # 特征值分解
    print("\n3. 特征值分解:")
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    print(f"   原始特征值: {eig_vals}")
    print(f"   特征向量矩阵形状: {eig_vecs.shape}")
    
    # 按特征值大小降序排列
    key = np.argsort(eig_vals)[::-1]
    eig_vals, eig_vecs = eig_vals[key], eig_vecs[:, key]
    print(f"   排序后特征值: {eig_vals}")
    print(f"   解释方差比例: {eig_vals / np.sum(eig_vals)}")
    
    # 矩阵清理（如果需要）
    if clean_matrix:
        print("\n4. 矩阵清理（Marchenko-Pastur）:")
        q = obs_matrix.shape[0] / obs_matrix.shape[1]
        lambda_max = 1 + 1/q + 2/np.sqrt(q)
        print(f"   样本/特征比值 q: {q:.3f}")
        print(f"   Marchenko-Pastur上界: {lambda_max:.3f}")
        print(f"   清理阈值: {clean_threshold * lambda_max:.3f}")
        
        noise_indices = np.where(eig_vals < clean_threshold * lambda_max)[0]
        if len(noise_indices) > 0:
            pos = noise_indices[0]
            print(f"   保留前 {pos} 个主成分")
            cleaned_cov = (eig_vecs[:, :pos] @ np.diag(eig_vals[:pos])) @ eig_vecs[:, :pos].T
            eig_vals, eig_vecs = np.linalg.eig(cleaned_cov)
            key = np.argsort(eig_vals)[::-1]
            eig_vals, eig_vecs = eig_vals[key], eig_vecs[:, key]
            print(f"   清理后特征值: {eig_vals}")
    
    # 计算主成分得分
    print("\n5. 计算主成分得分:")
    pc_scores = (z_matrix @ eig_vecs).astype(float)
    print(f"   主成分得分形状: {pc_scores.shape}")
    print(f"   前3个样本的PC得分:\n{pc_scores[:3]}")
    
    return eig_vals, eig_vecs, pc_scores






==========================================================================================================================

# 使用示例如下：
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 创建示例数据集：蛋白质特征数据
np.random.seed(2025)

print("=== 创建示例数据集 ===")
# 假设我们有6个蛋白质样本，4个特征（电荷密度、疏水性、长度、柔性）
protein_names = ['Protein_A', 'Protein_B', 'Protein_C', 'Protein_D', 'Protein_E', 'Protein_F']
feature_names = ['Charge_Density', 'Hydrophobicity', 'Length', 'Flexibility']

# 创建具有内在相关性的数据
# 电荷密度和疏水性负相关，长度和柔性正相关
data = np.array([
    [0.8, 0.2, 100, 0.7],   # Protein_A: 高电荷，低疏水，中等长度，中高柔性
    [0.3, 0.9, 150, 0.4],   # Protein_B: 低电荷，高疏水，长，中低柔性
    [0.9, 0.1, 80, 0.8],    # Protein_C: 很高电荷，很低疏水，短，高柔性
    [0.2, 0.8, 200, 0.3],   # Protein_D: 很低电荷，高疏水，很长，低柔性
    [0.6, 0.4, 120, 0.6],   # Protein_E: 中电荷，中疏水，中长，中柔性
    [0.1, 1.0, 180, 0.2]    # Protein_F: 最低电荷，最高疏水，长，最低柔性
])

print(f"原始数据形状: {data.shape}")
print("\n原始数据:")
df_original = pd.DataFrame(data, index=protein_names, columns=feature_names)
print(df_original)

# 计算原始数据的相关性矩阵
print("\n原始数据特征间相关性:")
correlation_matrix = np.corrcoef(data.T)
print(pd.DataFrame(correlation_matrix, index=feature_names, columns=feature_names))

# 执行PCA分析
print("\n" + "="*60)
eig_vals, eig_vecs, pc_scores = PCA_analysis(data, centered=True, clean_matrix=False)

# 详细分析结果
print("\n=== PCA结果分析 ===")

# 1. 解释方差分析
explained_variance_ratio = eig_vals / np.sum(eig_vals)
cumulative_variance = np.cumsum(explained_variance_ratio)

print("1. 解释方差分析:")
for i, (val, ratio, cum) in enumerate(zip(eig_vals, explained_variance_ratio, cumulative_variance)):
    print(f"   PC{i+1}: 特征值={val:.3f}, 解释方差={ratio:.1%}, 累积解释方差={cum:.1%}")

# 2. 主成分的含义分析
print("\n2. 主成分载荷分析（特征在各PC上的权重）:")
loadings_df = pd.DataFrame(eig_vecs.T, columns=feature_names, index=[f'PC{i+1}' for i in range(len(eig_vals))])
print(loadings_df)

# 解释主成分的生物学意义
print("\n3. 主成分的生物学意义解释:")
for i in range(min(3, len(eig_vals))):  # 分析前3个主成分
    print(f"\n   PC{i+1} (解释方差: {explained_variance_ratio[i]:.1%}):")
    loadings = eig_vecs[:, i]
    
    # 找出载荷最大的特征
    max_idx = np.argmax(np.abs(loadings))
    max_feature = feature_names[max_idx]
    max_loading = loadings[max_idx]
    
    print(f"     主导特征: {max_feature} (载荷: {max_loading:.3f})")
    
    # 分析各特征的贡献
    for j, (feature, loading) in enumerate(zip(feature_names, loadings)):
        contribution = "正贡献" if loading > 0 else "负贡献"
        strength = "强" if abs(loading) > 0.5 else "中" if abs(loading) > 0.3 else "弱"
        print(f"     {feature}: {loading:.3f} ({strength}{contribution})")

# 3. 样本在主成分空间的分布
print("\n4. 样本在主成分空间的得分:")
pc_scores_df = pd.DataFrame(pc_scores, index=protein_names, columns=[f'PC{i+1}' for i in range(pc_scores.shape[1])])
print(pc_scores_df)

# 可视化分析
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 原始数据热图
sns.heatmap(df_original, annot=True, cmap='viridis', ax=axes[0,0])
axes[0,0].set_title('Original Data Heatmap')

# 2. 特征相关性热图
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            xticklabels=feature_names, yticklabels=feature_names, ax=axes[0,1])
axes[0,1].set_title('Feature Correlation Matrix')

# 3. 解释方差比例
axes[1,0].bar(range(1, len(eig_vals)+1), explained_variance_ratio, alpha=0.7, label='Individual')
axes[1,0].plot(range(1, len(eig_vals)+1), cumulative_variance, 'ro-', label='Cumulative')
axes[1,0].set_xlabel('Principal Component')
axes[1,0].set_ylabel('Explained Variance Ratio')
axes[1,0].set_title('Explained Variance by Principal Components')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# 4. 主成分得分散点图
scatter = axes[1,1].scatter(pc_scores[:, 0], pc_scores[:, 1], s=100, alpha=0.7)
for i, protein in enumerate(protein_names):
    axes[1,1].annotate(protein, (pc_scores[i, 0], pc_scores[i, 1]), 
                      xytext=(5, 5), textcoords='offset points')
axes[1,1].set_xlabel(f'PC1 ({explained_variance_ratio[0]:.1%} variance)')
axes[1,1].set_ylabel(f'PC2 ({explained_variance_ratio[1]:.1%} variance)')
axes[1,1].set_title('Samples in Principal Component Space')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

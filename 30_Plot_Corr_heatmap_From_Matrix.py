# 直接从矩阵中绘制热图，此处只用于绘制相关性热图
# 如果是输入的np.ndarray，需要自定义相关性指标，也就是feature names
# 如果输入的是pd.DataFrame，直接使用行列标签即可

def plot_matrix_heatmap(matrix,row_labels=None,col_labels=None,title="Matrix Heatmap",cmap="coolwarm",figsize=(6,5)):
    """
        Args:
            matrix: 2D numpy array 或 pd.DataFrame
            row_labels: 行标签列表（可选）
            col_labels: 列标签列表（可选）
            title: 图标题
            cmap: 颜色映射
            figsize: 图像大小
        
        Fun:
            绘制矩阵或DataFrame的热图，支持自定义行列标签。
    
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 创建新的图形对象
    plt.figure(figsize=figsize)
    # 通过values属性判断是numpy.ndarray还是pd.DataFrame
    if hasattr(matrix,'values'):
        # 有values属性，说明是pd.DataFrame
        # 直接使用values属性抽取其底层的numpy数组值
        data = matrix.values
        # 如果没有提供行、列标签，则使用DataFrame的index和columns
        if row_labels is None:
            row_labels = matrix.index
        if col_labels is None:
            col_labels = matrix.columns

    # 否则，直接使用numpy数组
    else:
        data = matrix

    # 绘制热图
    sns.heatmap(data,
                annot=True,
                fmt=".2f", # 显示数值，保留两位小数
                cmap=cmap, # 颜色映射
                xticklabels=col_labels,
                yticklabels=row_labels,
                square=True, # 使单元格呈正方形
                cbar=True #显示颜色条
                )
    # 设置图标标题
    plt.title(title)
    # 自动调整子图参数，适当间距
    plt.tight_layout()
    # 显示图形
    plt.show()


============》使用示例
# 假设我们有6个蛋白质样本，4个特征（电荷密度、疏水性、长度、柔性）
protein_names = ['Protein_A', 'Protein_B', 'Protein_C', 'Protein_D', 'Protein_E', 'Protein_F']
feature_names = ['Charge_Density', 'Hydrophobicity', 'Length', 'Flexibility']

data = np.asarray([
    [0.8, 0.2, 100, 0.7],   # Protein_A: 高电荷，低疏水，中等长度，中高柔性
    [0.3, 0.9, 150, 0.4],   # Protein_B: 低电荷，高疏水，长，中低柔性
    [0.9, 0.1, 80, 0.8],    # Protein_C: 很高电荷，很低疏水，短，高柔性
    [0.2, 0.8, 200, 0.3],   # Protein_D: 很低电荷，高疏水，很长，低柔性
    [0.6, 0.4, 120, 0.6],   # Protein_E: 中电荷，中疏水，中长，中柔性
    [0.1, 1.0, 180, 0.2]    # Protein_F: 最低电荷，最高疏水，长，最低柔性
])

correlation_matrix = np.corrcoef(data,rowvar=False)
print(pd.DataFrame(correlation_matrix,index=feature_names,columns=feature_names))

# 如果传入的是pd.DataFrame，则直接使用其行列标签
df = pd.DataFrame(correlation_matrix,index=feature_names,columns=feature_names)
plot_matrix_heatmap(df)

# 如果传入的是np.ndarray，则使用自定义的feature names
# 如果传入的是numpy.ndarray
feature_names = ['Charge_Density', 'Hydrophobicity', 'Length', 'Flexibility']
plot_matrix_heatmap(correlation_matrix,row_labels=feature_names,col_labels=feature_names)


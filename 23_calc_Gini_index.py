# 计算Gini index：

def get_Gini_index(seq):
    """
    计算Gini index，作为序列复杂度的一个指标。

    Args:
        seq (str): 输入的蛋白质序列字符串

    Returns:
        float: Gini index值
    """
    # 计算氨基酸频率
    amino_acid_counts = {}
    for aa in seq:
        if aa in amino_acid_counts:
            amino_acid_counts[aa] += 1
        else:
            amino_acid_counts[aa] = 1

    # 计算总氨基酸数
    total_count = len(seq)

    # 计算Gini index
    Gini_index = 1.0
    for count in amino_acid_counts.values():
        frequency = count / total_count
        if frequency > 0:  # 避免对数为零的情况
            Gini_index -= frequency * frequency

    return Gini_index

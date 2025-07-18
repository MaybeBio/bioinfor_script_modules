# 将蛋白质序列转换为带电氨基酸数值序列

# 1，使用列表生成式
def protein_sequence_to_charge_numeric(seq):
    """
    Args:
        seq: 输入的蛋白质序列字符串 
    Fun:
        将蛋白质序列转换为带电氨基酸的数值序列，使用以下映射：
        DE是-1，KRH是+1，其他氨基酸是0
    Returns:
        返回转换后的数值序列列表
    """
    charge_mapping = {
        'D': -1,  # 天冬氨酸
        'E': -1,  # 谷氨酸
        'K': 1,   # 赖氨酸
        'R': 1,   # 精氨酸
        'H': 1,   # 组氨酸
    }
    charge_numeric_seq = [ charge_mapping.get(aa,0) for aa in seq ]
    return charge_numeric_seq

# 2，
def protein_sequence_to_charge_numeric(seq):
    """
    Args:
        seq: 输入的蛋白质序列字符串 
    Fun:
        将蛋白质序列转换为带电氨基酸的数值序列，使用以下映射：
        DE是-1，KRH是+1，其他氨基酸是0
    Returns:
        返回转换后的数值序列列表
    """
    charge_mapping = {
        'D': -1,  # 天冬氨酸
        'E': -1,  # 谷氨酸
        'K': 1,   # 赖氨酸
        'R': 1,   # 精氨酸
        'H': 1,   # 组氨酸
    }
    numeric_seq = []
    for aa in seq:
        if aa in charge_mapping:
            numeric_seq.append(charge_mapping[aa])
        else:
            numeric_seq.append(0)
    return numeric_seq

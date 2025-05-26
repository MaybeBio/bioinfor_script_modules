# 计算1个分布的香农熵
# 常见情况是计算1个随机变量生成的序列：给定变量取值以及变量取值的频率，计算这个序列的香农熵

# 01
# 给定的字典中是频率的情况
def calc_shannon_entropy(seq:dict) -> float:
    """
    Args:
    seq:输入的氨基酸序列的字典，字符:频率(如果是频数可以转换为频率)
    
    Fun:
    计算香农熵，衡量序列的复杂性和信息量
    香农熵用于衡量1个随机序列中所包含的信息量或不确定性，
    熵越高，越无序，表明序列不确定性越大，信息量越大，复杂度越高
    （注意，如果每一个字符都等概率出现，则香农熵最大，信息量大，复杂度也高）
    统计学中信息量蕴藏在var中，高方差意味着信息量越大（全正相关）——》所以LCR是低熵值的区域
    
    """
    import math
    sample_entropy = 0
    for key in seq.keys():
        if seq[key] != 0:
            p = seq[key]
            sample_entropy -= p * math.log2(p)
    return sample_entropy

##############################################################################################################################################################################

# 02
# 还是频率
def calc_shannon_entropy(seq:dict) -> float:
    """
    Args:
    seq:输入的氨基酸序列的字典，字符:频率(如果是频数可以转换为频率)
    
    Fun:
    计算香农熵，衡量序列的复杂性和信息量
    
    """
    import math
    return sum([-(p * math.log2(p)) for p in seq.values() if p != 0]) 

##############################################################################################################################################################################

# 03 
# 如果给定的字典中是频数，需要先将频数转换为频率
def calc_shannon_entropy(seq:dict) -> float:
    """
    Args:
    seq:输入的氨基酸序列的字典，字符:频数(可以转换为频率)
    
    Fun:
    计算香农熵，衡量序列的复杂性和信息量
    """
    import math
    sample_entropy = 0
    for key in seq.keys():
        if seq[key] != 0:
            p = seq[key] / sum(seq.values())
            sample_entropy -= p * math.log2(p)
    return sample_entropy



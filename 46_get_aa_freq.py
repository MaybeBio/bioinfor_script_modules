# 获取序列频率分布

# 1
aa_alphabets = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
from typing import Dict
from collections import Counter
def get_aa_freq(sequence: str) -> Dict[str, float]:
    """
    Description
    ----------
        计算并返回蛋白质序列的氨基酸组成比例/频率字典
    
    Args
    ----------
        sequence (str): 蛋白质序列

    Returns
    ----------
        蛋白质序列的氨基酸组成比例/频率字典 (Dict[str, float])
    """
    aa_count = Counter(sequence)
    seq_length = len(sequence)
    aa_freq = { aa: aa_count.get(aa,0) / seq_length for aa in aa_alphabets }
    return aa_freq

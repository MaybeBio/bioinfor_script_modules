# 主要是用于统计1个粗粒度/符号简约序列的k-mer字典

# 1, 指定可迭代、需要统计的k-mer的k
# 可以将ks修改为Optional, 如果没有传入数值, 就返回string所有的k-mer字典, 也就是len(string)

from typing import Dict, Iterable
from collections import Counter
def get_kmer_counts(string:str, ks:Iterable[int]) -> Dict[int, Dict[str, int]]:
    """
    Description 
    ----------
        计算字符串中不同k-mer的出现次数, 支持多个k值;

    Args
    ----------
        string (str): 输入的字符串模式, 最好是简约化后的字符串, 也就是目标粗理化度层面
        ks (Iterable[int]): k-mer的长度列表, 可以是多个不同的k值, 可以是不同的迭代器类型, 比如list, set等, 或者是range(1,10)

    Returns
    ----------
        Dict[int, Dict[str, int]]: 包含不同k-mer出现次数的字典, 键为k值, 值为对应的k-mer计数字典;

    Notes
    ----------
    - 1, 注意传入的string只能是简约之后的电荷字符串
    - 2, 如果k值大于字符串长度, 则对应的k-mer计数字典为空
    - 3, 如果输入字符串为空, 则抛出ValueError异常
    
    Todos
    ----------
    - 1, 设计初衷是为了对单条字符串进行统计, 也就是片段内滑窗, 不跨界, 本函数面向单端, 但是后续需要对多个IDR区域进行统计等等,
    可以是对1个蛋白质内多片段进行统计, 也可以对多蛋白质进行批量统计, 跨端需要逐段调用再汇总
    """

    # 先处理异常情况
    if not string:
        raise ValueError("Input string cannot be empty.")
    
    # 先初始化结果存储字典
    kmer_dict: Dict[int, Dict[str, int]] = {k: Counter() for k in ks}

    for k in ks:
        if k > len(string):
            # 如果k值＞字符串长度, 则对应的k-mer计数字典为空
            kmer_dict[k] = {}
        else:
            # 每种k-mer下有多少个滑动窗口
            for i in range(len(string) - k + 1):
                # 提取k-mer
                kmer = string[i:i+k]
                kmer_dict[k][kmer] += 1
    
    return kmer_dict

=====================================================================================================

# 2, 不提供指定k-mer的k的情况下, 则统计所有的k-mer (也就是1-len(string))

def get_kmer_counts(string:str, ks:Optional[Iterable[int]] = None) -> Dict[int, Dict[str, int]]:
    """
    Description 
    ----------
        计算字符串中不同k-mer的出现次数, 支持多个k值;

    Args
    ----------
        string (str): 输入的字符串模式, 最好是简约化后的字符串, 也就是目标粗理化度层面
        ks (Optional[Iterable[int]]): k-mer的长度列表, 可以是多个不同的k值, 可以是不同的迭代器类型, 比如list, set等, 或者是range(1,10);
            可选参数, 如果没有提供, 就计算1-len(string)范围内的所有k值; 默认为None

    Returns
    ----------
        Dict[int, Dict[str, int]]: 包含不同k-mer出现次数的字典, 键为k值, 值为对应的k-mer计数字典;

    Notes
    ----------
    - 1, 注意传入的string只能是简约之后的电荷字符串
    - 2, 如果k值大于字符串长度, 则对应的k-mer计数字典为空
    - 3, 如果输入字符串为空, 则抛出ValueError异常
    
    Todos
    ----------
    - 1, 设计初衷是为了对单条字符串进行统计, 也就是片段内滑窗, 不跨界, 本函数面向单端, 但是后续需要对多个IDR区域进行统计等等,
    可以是对1个蛋白质内多片段进行统计, 也可以对多蛋白质进行批量统计, 跨端需要逐段调用再汇总
    """

    # 先处理异常情况
    if not string:
        raise ValueError("Input string cannot be empty.")

    if ks is not None:
        # 先初始化结果存储字典
        kmer_dict: Dict[int, Dict[str, int]] = {k: Counter() for k in ks}
        for k in ks:
            if k > len(string):
                # 如果k值＞字符串长度, 则对应的k-mer计数字典为空
                kmer_dict[k] = {}
            else:
                # 每种k-mer下有多少个滑动窗口
                for i in range(len(string) - k + 1):
                    # 提取k-mer
                    kmer = string[i:i+k]
                    kmer_dict[k][kmer] += 1
    else:
        # 如果没有提供ks参数, 则计算1-len(string)范围内的所有k值
        # 先初始化结果存储字典
        kmer_dict: Dict[int, Dict[str, int]] = {k: Counter() for k in range(1, len(string) + 1)}
        for k in range(1, len(string) + 1):
            for i in range(len(string) - k + 1):
                kmer = string[i:i+k]
                kmer_dict[k][kmer] += 1
    
    return kmer_dict

# 获取两个区域的交集
# 注意区域是连续的

# 1, 利用set特性

from typing import Tuple
def get_region_intersection(region1: Tuple[int, int], region2: Tuple[int, int]) -> Tuple[int, int]:
    """
    Description
    ----------
        计算两个区间坐标的交集, 如果没有交集则返回None;

    Args
    ----------
        region1 (Tuple[int, int]): 第一个区间坐标, 形式为(start, end);
        region2 (Tuple[int, int]): 第二个区间坐标, 形式为(start, end);

    Returns
    ----------
        Tuple[int, int] | None: 返回交集区间的坐标, 形式为(start, end), 如果没有交集则返回None;

    Example
    ----------
    >>> region1 = (5, 15)
    >>> region2 = (10, 20)
    >>> intersection = get_region_intersection(region1, region2)
    >>> print(intersection)
    (10, 15)
    """
  
    start1, end1 = region1
    start2, end2 = region2

    intersection_set = set(range(start1, end1+1)) & set(range(start2, end2+1))
    if intersection_set:
        return (min(intersection_set), max(intersection_set))
    else:
        return None

==================================================================================================================

# 2, 稍微用一点minmax函数

def get_region_intersection(region1: Tuple[int, int], region2: Tuple[int, int]) -> Tuple[int, int]:
    """
    Description
    ----------
        计算两个区间坐标的交集, 如果没有交集则返回None;

    Args
    ----------
        region1 (Tuple[int, int]): 第一个区间坐标, 形式为(start, end);
        region2 (Tuple[int, int]): 第二个区间坐标, 形式为(start, end);

    Returns
    ----------
        Tuple[int, int] | None: 返回交集区间的坐标, 形式为(start, end), 如果没有交集则返回None;

    Example
    ----------
    >>> region1 = (5, 15)
    >>> region2 = (10, 20)
    >>> intersection = get_region_intersection(region1, region2)
    >>> print(intersection)
    (10, 15)
    """
    start1, end1 = region1
    start2, end2 = region2

    # 计算交集的起始和结束位置 
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)

    # 检查是否有交集
    if intersection_start <= intersection_end:
        return (intersection_start, intersection_end)
    else:
        return None

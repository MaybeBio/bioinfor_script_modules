# 爬取各种数据库时, 如果需要爬取坐标, 可能某一个条目中能够爬到各种数据库来源的数值坐标range
# 比如说 xxx region: "1-5,3-7", 存为str key
# 最后清洗的时候可以合并区间

def consolidate_range_strings(range_str: str, min_len: int = 1) -> List[Tuple[int, int]]:
    """
    Description
    -----------
    对字符串形式的区间描述进行解析和合并, 比如说将 "1-5,3-7" 合并为 [(1, 7)]。
    
    Args
    ----
    range_str: str
        以逗号分隔的区间字符串, 每个区间由 "start-end" 形式表示 (e.g. "1-5,3-7,10-12")
    min_len: int
        合并后区间的最小长度, 默认1. 只有当合并后的区间长度 >= min_len 时才会被保留在结果中
    
    Returns
    -------
    List[Tuple[int, int]]: 合并后的区间列表, 每个区间以 (start, end) 的形式表示.
    """

    # 处理空值NA、空字符串
    if pd.isna(range_str) or not range_str.strip():
        return []
        
    intervals = []
    for part in range_str.split(','):
        if '-' in part:
            start, end = part.split('-')
            intervals.append([int(start), int(end)])
            
    if not intervals:
        return []
        
    intervals.sort(key=lambda x: x[0])
    # 升序之后从左到右逐2个合并区间
    merged = [intervals[0]]
    for current in intervals[1:]:
        previous = merged[-1]
        if current[0] <= previous[1] + 1:
            previous[1] = max(previous[1], current[1])
        else:
            merged.append(current)
            
    # 过滤掉长度小于 min_len 的区间(至少得有1个aa的长度)
    return [(m[0], m[1]) for m in merged if (m[1] - m[0] + 1) >= min_len]

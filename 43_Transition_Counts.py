# 对于一个简约/reduced的序列/符号/sign字符串, 统计其相邻字符转移频次(字典)
# 例如: "+-+-", 有(+-,2)、(-+,1) 转移字符事件+观测到的频次

# 1
def transition_counts(s: str):
  # 统计相邻转移，如 BA、AN、NB 等
  from collections import Counter
  return dict(Counter(a + b for a, b in zip(s, s[1:])))

====================================================================================================

# 2
def transition_counts(string: str) -> Dict[str, str]:
    """
    Description
    ----------
        统计字符串中相邻字符切换的模式及其出现次数;

    Args
    ----------
        string (str): 输入的字符串模式, 最好是简约化后的字符串, 也就是目标粗理化度层面
    
    
    Returns
    ----------
        Dict[str, int]: 包含相邻字符切换模式及其出现次数的字典, 键为切换模式(如 "AB"), 值为出现次数;

    Notes
    ----------
    - 1, 该函数适用于简约化后的字符串
    - 2, 如果输入的字符串长度小于2/没有切换, 则返回空字典, 不要返回None或者抛出error, 防止流程中统计中断
        
    Example
    ----------
    >>> pattern = "ABABAC"
    >>> counts = transition_counts(pattern)
    >>> print(counts)
    {"AB": 2, "BA": 2, "AC": 1}
    """
    if not string:
        raise ValueError("Input string cannot be empty.")
    if len(string) < 2:
        return {}
    transitions = [string[i] + string[i+1] for i in range(len(string)-1)]
    transition_counter = Counter(transitions)
    return dict(transition_counter)

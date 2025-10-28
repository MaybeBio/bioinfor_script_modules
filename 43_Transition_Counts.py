# 对于一个简约/reduced的序列/符号/sign字符串, 统计其相邻字符转移频次(字典)
# 例如: "+-+-", 有(+-,2)、(-+,1) 转移字符事件+观测到的频次

def transition_counts(s: str):
  # 统计相邻转移，如 BA、AN、NB 等
  from collections import Counter
  return dict(Counter(a + b for a, b in zip(s, s[1:])))

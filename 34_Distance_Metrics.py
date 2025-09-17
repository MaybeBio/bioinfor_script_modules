# 主要收集各种可度量的distance指标，可以衡量两个样本点远近、两个概率分布/两个向量/两个集合/两个序列的相似性
# 可用于定义构建loss function

# 1，余弦距离=1-余弦相似性
def Cos_dist(seq1,seq2):
  import numpy as np
  norm_seq1 = np.linalg.norm(seq1)
  norm_seq2 = np.linalg.norm(seq2)
  if norm_a == 0 or norm_b == 0:
    return 1
  cos_sim = np.dot(seq1,seq2)/(norm_a * norm_b)
  cos_dist = 1 - cos_sim
  return cos_dist

# np.ix_ 按index提取子矩阵
# pae[np.ix_(idx, idx)] 等价于 pae[idx][:, idx]（当 idx 是整数序列时），但 np.ix_ 在更复杂的多维切片/广播场景中更明确安全。
# 如果直接用布尔二维索引（pae[mask, mask]）也可，但 np.ix_ 常用于把整数索引数组扩展为行列索引的通用方法。

# 示例 1：简单整数索引
import numpy as np
pae = np.arange(36).reshape(6,6)  # 0..35 的 6x6 矩阵
idx = [0, 2, 5]
pae_sub = pae[np.ix_(idx, idx)]
# pae_sub =
# [[ 0,  2,  5],
#  [12, 14, 17],
#  [30, 32, 35]]


# 示例 2：从布尔 mask 得到 idx
mask = np.array([True, False, True, False, False, True])
idx = np.where(mask)[0]  # -> [0,2,5]
pae_sub = pae[np.ix_(idx, idx)]



# 示例 3：带重复索引（会重复行/列）
idx = [1, 1, 3]
pae_sub = pae[np.ix_(idx, idx)]
# 子矩阵会包含第1行/列两次

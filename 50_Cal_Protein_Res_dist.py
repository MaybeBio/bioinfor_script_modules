# 对于1个含有N个残基（默认等同于含有N个C α）的蛋白质序列
# 计算逐点，也就是两两点之间的距离，
# ⚠️所谓的计算两两残基之间的距离，实际上指的是计算这两个残基的C α之间的距离，因为1个残基有很多原子，每一个原子都有一个坐标
# 也就是对于1个蛋白质序列数组(N,3) ——> 返回1个(N,N)数组, 数组内容代表两个点之间的欧几里得几何距离

# 1, common 2 loop
# 普通逻辑的双层循环
import numpy as np
def _pairwise_dist_loop(coords):
    n = coords.shape[0] # 点的数量
    d = coords.shape[1] # 每个点的坐标维度，常规就是3，3维坐标
    # 初始化距离矩阵
    dist_matrix = np.zeros((n,n))

    # 双层循环: 遍历所有点对(i,j)
    for i in range(n):
        for j in range(n):
            diff_x = coords[i,0] - coords[j,0]
            diff_y = coords[i,1] - coords[j,1]
            diff_z = coords[i,2] - coords[j,2]
            # 计算欧式几何距离
            dist =  np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
            # 存入距离矩阵
            dist_matrix[i,j] = dist
    return dist_matrix

------------------------------------------------------------------------------------------------------------------

# 2, broadcasting 
# 使用广播机制，使运算更加方便快捷
# 处理速度优于法1
def _pairwise_dist(coords):
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))

# 运算示例, 比如说对于CTCF 727
import numpy as np
coords = np.zeros((727,3))
for i in range(727):
    coords[i] = chain[i+1]['CA'].coord
coords,coords.shape

# 调用
_pairwise_dist_loop(coords)
_pairwise_dist(coords)

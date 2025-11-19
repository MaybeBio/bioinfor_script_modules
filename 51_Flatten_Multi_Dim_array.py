# 将多维数组展平: ravel和flatten
# 首先声明两者所要实现的功能是一致的（将多维数组降位一维），两者的区别在于返回拷贝（copy）还是返回视图（view），numpy.flatten()返回一份拷贝，对拷贝所做的修改不会影响（reflects）原始矩阵，而numpy.ravel()返回的是视图（view，也颇有几分C/C++引用reference的意味），会影响（reflects）原始矩阵。
# 参考：https://blog.csdn.net/lanchunhui/article/details/50354978

import numpy as np

# 1. 创建一个常规的 2D 数组（内存连续）
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("原数组 arr：")
print(arr)
print("原数组内存地址：", arr.__array_interface__['data'][0])  # 查看数组数据的内存起始地址
print("-" * 50)

# 2. 用 ravel() 展平
raveled = arr.ravel()
print("ravel() 展平后：", raveled)
print("ravel() 结果内存地址：", raveled.__array_interface__['data'][0])
print("ravel() 是否和原数组共享内存？", raveled.base is arr)  # base 指向原数组（视图特征）
print("-" * 50)

# 3. 用 flatten() 展平
flattened = arr.flatten()
print("flatten() 展平后：", flattened)
print("flatten() 结果内存地址：", flattened.__array_interface__['data'][0])
print("flatten() 是否和原数组共享内存？", flattened.base is arr)  # base 为 None（拷贝特征）
print("-" * 50)

# 4. 验证：修改视图（raveled）会影响原数组吗？
raveled[0] = 99  # 修改 raveled 的第一个元素
print("修改 raveled[0] = 99 后：")
print("原数组 arr：", arr)  # 原数组被同步修改（因为 raveled 是视图）
print("raveled 数组：", raveled)
print("flattened 数组：", flattened)  # 不受影响（因为是独立拷贝）

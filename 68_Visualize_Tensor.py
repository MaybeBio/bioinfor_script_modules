def visualize_numpy_array(arr):
    """
    可视化NumPy数组的形状和结构
    """
    print("=" * 50)
    print(f"数组内容：\n{arr}")
    print("-" * 50)
    # 打印核心形状信息
    shape = arr.shape
    dim = arr.ndim
    size = arr.size
    print(f"数组形状 shape = {shape} → {dim}维数组")
    print(f"数组元素总数 size = {size}")
    print(f"维度含义：{' × '.join([f'第{i+1}维={s}' for i, s in enumerate(shape)])}")
    print("=" * 50)

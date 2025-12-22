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


=====================================================================================================

def visualize_numpy_array(arr):
    """
    可视化NumPy数组的形状和结构（层次化输出格式）
    """
    # 确保输入是numpy数组
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    
    # 定义分隔线样式，区分不同层级
    LINE_TOP = "=" * 60
    LINE_MID = "-" * 60
    LINE_SUB = "~" * 60

    print(LINE_TOP)
    print(f"【1. 数组基础信息】")
    print(LINE_SUB)
    # 打印数组内容，根据维度调整显示缩进
    if arr.ndim <= 2:
        print(f"数组内容：\n{arr}")
    else:
        print(f"高维数组内容（前3个维度切片）：\n{arr[:2] if arr.shape[0]>2 else arr}")
    print(LINE_MID)

    # 核心形状信息（层次化排版）
    shape = arr.shape
    dim = arr.ndim
    size = arr.size
    dtype = arr.dtype
    print(f"【2. 核心维度参数】")
    print(LINE_SUB)
    print(f"  ✅ 数组维度数  → {dim} 维")
    print(f"  ✅ 数组形状     → shape = {shape}")
    print(f"  ✅ 维度含义     → {' × '.join([f'第{i+1}维={s}' for i, s in enumerate(shape)])}")
    print(f"  ✅ 总元素数量   → size = {size}")
    print(f"  ✅ 数据类型     → dtype = {dtype}")
    print(LINE_TOP)

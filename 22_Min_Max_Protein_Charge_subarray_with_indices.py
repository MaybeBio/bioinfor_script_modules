# 对20、21模块的运用

# 1，蛋白质带电序列的最大连续子序列和问题

# 将蛋白质序列转换为带电氨基酸数值序列
def protein_sequence_to_charge_numeric(seq):
    """
    Args:
        seq: 输入的蛋白质序列字符串 
    Fun:
        将蛋白质序列转换为带电氨基酸的数值序列，使用以下映射：
        DE是-1，KRH是+1，其他氨基酸是0
    Returns:
        返回转换后的数值序列列表
    """
    charge_mapping = {
        'D': -1,  # 天冬氨酸
        'E': -1,  # 谷氨酸
        'K': 1,   # 赖氨酸
        'R': 1,   # 精氨酸
        'H': 1,   # 组氨酸
    }
    charge_numeric_seq = [ charge_mapping.get(aa,0) for aa in seq ]
    return charge_numeric_seq

def max_Protein_subarray_with_indices(seq):
    """
    Args:
        seq: 输入的蛋白质序列
    Fun:
        首先需要将输入的蛋白质序列转换为数值序列，
        使用动态规划方法找到最大子数组和及其起始和结束索引
    Returns:
        返回起始索引、结束索引（索引以1-index输出）,子数组本身以及最大子数组和，元组形式，子数组要重新组织为一个序列字符串
    
    """
    # 首先将蛋白质序列转换为数值序列
    nums = protein_sequence_to_charge_numeric(seq)
    if not nums:
        # 或者使用len(nums) == 0
        # 如果是一个空数组，应该返回
        # 起始和结束索引都为-1，子数组为空，最大和为0
        return (-1,-1,[],0)
    
    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    max_sum = dp[0]
    
    # 初始化起始和结束位置
    current_start = 0
    global_start = 0
    global_end = 0
    
    for i in range(1, n):
        # 计算 dp[i] 并判断是否选择单独开始
        if nums[i] > dp[i-1] + nums[i]:
            dp[i] = nums[i]
            current_start = i  # 新起点
        else:
            dp[i] = dp[i-1] + nums[i]
        
        # 更新全局最大值及其位置
        if dp[i] > max_sum:
            max_sum = dp[i]
            global_start = current_start
            global_end = i
    
    # 提取子数组
    subarray = nums[global_start : global_end + 1]
    # 但是数组不是我们想要的，我们想要的是相同坐标下对应的原始序列
    subarray_str = seq[global_start : global_end + 1]
    # 然后注意我们需要返回的是1-indexed的索引，因为我们已经提取出来了子序列，所以不需要0-index了
    global_start += 1  # 转换为1-indexed
    global_end += 1    # 转换为1-indexed

    return (global_start, global_end, subarray_str, max_sum)

===========================================================================================================================================================

# 2，蛋白质带电序列的最大、最小连续子序列和问题，对1的改进

# 将蛋白质序列转换为带电氨基酸数值序列
def protein_sequence_to_charge_numeric(seq):
    """
    Args:
        seq: 输入的蛋白质序列字符串 
    Fun:
        将蛋白质序列转换为带电氨基酸的数值序列，使用以下映射：
        DE是-1，KRH是+1，其他氨基酸是0
    Returns:
        返回转换后的数值序列列表
    """
    charge_mapping = {
        'D': -1,  # 天冬氨酸
        'E': -1,  # 谷氨酸
        'K': 1,   # 赖氨酸
        'R': 1,   # 精氨酸
        'H': 1,   # 组氨酸
    }
    charge_numeric_seq = [ charge_mapping.get(aa,0) for aa in seq ]
    return charge_numeric_seq

def min_max_Protein_Charge_subarray_with_indices(seq, find_min=False):
    """
    对一段数组/序列，寻找最大或最小连续子序列和所表征的子序列。

    Args:
        seq (str): 输入的蛋白质序列
        find_min (bool): 如果为 True，则寻找最小连续子序列和；否则寻找最大连续子序列和

    Returns:
        以元组形式返回
        start_index (int): 子序列的起始位置（1-based）
        end_index (int): 子序列的结束位置（1-based）
        subarray (list): 子序列
        max_sum (int): 最大或最小连续子序列和
    """
    # 首先将蛋白质序列转换为数值序列
    nums = protein_sequence_to_charge_numeric(seq)
    if not nums:
        # 或者使用len(nums) == 0
        # 如果是一个空数组，应该返回
        # 起始和结束索引都为-1，子数组为空，最大和为0
        return (-1,-1,[],0)

    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    result_sum = dp[0]
    
    # 初始化起始和结束位置
    current_start = 0
    global_start = 0
    global_end = 0

    for i in range(1, n):
        # 根据 find_min 决定是寻找最小还是最大子序列和
        if find_min:
            if nums[i] < dp[i-1] + nums[i]:
                dp[i] = nums[i]
                current_start = i  # 新起点
            else:
                dp[i] = dp[i-1] + nums[i]
            if dp[i] < result_sum:  # 更新全局最小值
                result_sum = dp[i]
                global_start = current_start
                global_end = i
        else:
            if nums[i] > dp[i-1] + nums[i]:
                dp[i] = nums[i]
                current_start = i  # 新起点
            else:
                dp[i] = dp[i-1] + nums[i]
            if dp[i] > result_sum:  # 更新全局最大值
                result_sum = dp[i]
                global_start = current_start
                global_end = i

    # 提取子数组
    subarray = nums[global_start : global_end + 1]
    # 但是数组不是我们想要的，我们想要的是相同坐标下对应的原始序列
    subarray_str = seq[global_start : global_end + 1]
    # 然后注意我们需要返回的是1-indexed的索引，因为我们已经提取出来了子序列，所以不需要0-index了
    global_start += 1  # 转换为1-indexed
    global_end += 1    # 转换为1-indexed
    
    return (global_start, global_end, subarray_str, result_sum)

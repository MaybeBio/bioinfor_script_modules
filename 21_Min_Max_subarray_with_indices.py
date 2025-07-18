# 对20_Max_subarray_with_indices.py的进一步补充
# 20_Max_subarray_with_indices.py是解决最大连续子序列的问题
# 本脚本模块同时解决最大连续子序列以及最小连续子序列的问题，参考https://blog.csdn.net/weixin_62528784/article/details/146420536

def max_subarray_with_indices(nums, find_min=False):
    """
    对一段数组/序列，寻找最大或最小连续子序列和所表征的子序列。

    Args:
        nums (list): 数值序列/列表
        find_min (bool): 如果为 True，则寻找最小连续子序列和；否则寻找最大连续子序列和

    Returns:
        max_sum (int): 最大或最小连续子序列和
        start_index (int): 子序列的起始位置（0-based）
        end_index (int): 子序列的结束位置（0-based）
        subarray (list): 子序列
    """
    if not nums:
        return 0, -1, -1, []  # 返回4个值，索引设为-1
    
        # 返回 0（默认和）、-1（无效起始索引）、-1（无效结束索引）、空列表（子数组）。
        # 避免解包错误以及类型冲突
        # 注意这里是对应result_sum, global_start, global_end, subarray

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
    return result_sum, global_start, global_end, subarray

# 最大连续子序列和/最大连续子数组和问题，参考https://blog.csdn.net/weixin_62528784/article/details/146371022

def max_subarray_with_indices(nums):
  """
  Args:
      nums：输入的数值列表list，比如说[-2, 1, -3, 4, -1, 2, 1, -5, 4]

  Fun:
      解决该输入数值列表的最大连续子序列和问题，
      返回该最大连续子序列和，起始索引（0-index，可以在结尾++转变为1-index），结束索引（0-index，可以在结尾++转变为1-index），以及提取出来的该最大连续子序列
      可以改为返回元组

  """
    if not nums:
        return 0, [], []
    
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
    return max_sum, global_start, global_end, subarray

# ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！下面是示例
# 示例
nums = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
result = max_subarray_with_indices(nums)
print(f"最大和: {result[0]}, 起始索引: {result[1]}, 结束索引: {result[2]}, 子数组: {result[3]}")
# 输出: 最大和: 6, 起始索引: 3, 结束索引: 6, 子数组: [4, -1, 2, 1]

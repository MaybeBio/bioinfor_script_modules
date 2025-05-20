# 获取1个字符串数组中每个字符串元素的公共最长子前缀
# 问题描述参考https://leetcode.cn/problems/longest-common-prefix/solutions/
# https://blog.csdn.net/weixin_62528784/article/details/148091302?sharetype=blogdetail&sharerId=148091302&sharerefer=PC&sharesource=weixin_62528784&spm=1011.2480.3001.8118

# 01
# 比较多个字符串，寻找多个字符串（字符串数组）的最长公共子前缀，包括2个
# 本质：先从任意1个字符串中初始化1个公共前缀数组，再循环比对
        def longestCommonPrefix(self, strs: List[str]) -> str:
            substring_0 = [] # 先初始化存储子串0的所有前缀
            substring_0_hit = {} # 我的想法是对于所有的前缀，构建1个计数count的hash表
            if len(strs) == 0: # 空列表判断，防止空列表index操作越界
                return ""
            for j in range(len(strs[0]): # 提取strs[0]的所有前缀子串
                substring_0.append(strs[0][0:j+1])
            for sub in substring_0: # 对strs[0]中的所有前缀子串，在strs中遍历获取hash表的计数
                substring_0_hit[sub] = 0
                for i in strs:
                    if i.startswith(sub):
                        substring_0_hit[sub] += 1
            for i in list(substring_0_hit.keys())[::-1]: # 逆序查找第1个hit符合len(strs)的key
                if substring_0_hit[i] == len(strs):
                    return i
            return ""  

##############################################################################################################################################################################


# 02
# 比较多个字符串，寻找多个字符串（字符串数组）的最长公共子前缀，包括2个
# 横向扫描，定义了2个函数

    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""
        
        prefix, count = strs[0], len(strs)
        for i in range(1, count):
            prefix = self.lcp(prefix, strs[i])
            if not prefix:
                break
        
        return prefix

    def lcp(self, str1, str2):
        length, index = min(len(str1), len(str2)), 0
        while index < length and str1[index] == str2[index]:
            index += 1
        return str1[:index]

##############################################################################################################################################################################

# 03
# 比较2个字符串，寻找2个字符串的最长公共子前缀

def longest_common_prefix_between2strings(str1:str,str2:str) -> str:
    """
    Args:
    str1:str数据类型为字符串，同理str2
    
    Fun:
    返回两个子字符串序列的最长公共子前缀，返回值为->str
    
    """
    
    # 首先获取两个str中的最短长度length，以及用于遍历比较两个数组中逐个元素的索引index
    length,index = min(len(str1),len(str2)),0
    prefix = "" # 先初始化前缀子串
    while index < length and str1[index] == str2[index]:
        # 当比对index没有超过字符串str长度，并且两个字符串对应索引值相等的时候，可以字符串累加为前缀
        prefix += str1[index] 
        # 注意循环变量的更新
        index += 1

    # 退出循环的时候，必然是当前字符条件不满足了，可以返回当前以及累积起来的子字符串前缀序列
    return prefix    

##############################################################################################################################################################################

# 04
# 同上03比较2个字符串

def longest_common_prefix_between2strings(str1:str,str2:str) -> str:
    length,index = min(len(str1),len(str2)),0
    prefix = "" # 先初始化前缀子串
    while index < length and str1[index] == str2[index]:
        # 当比对index没有超过字符串str长度，并且两个字符串对应索引值相等的时候
        index += 1

    # 退出循环的时候，必然是当前字符条件不满足了，可以返回当前以及累积起来的子字符串前缀序列，此处只使用当前索引index前的序列
    return str1[:index]    

##############################################################################################################################################################################

# 05

def longest_common_prefix_between2strings(str1:str,str2:str) -> str:
    prefix = ""
    for i in range(len(str1)):
        # 确保每个索引index都能够被str1或者是str2访问，并且逐个索引元素比较都相等，则可以字符串累加，本质上等同于法03
        if i < len(str2) and str1[i]==str2[i]:
            prefix += str1[i]
        else:
            break
    return prefix,str1[:i]  # 同样的，此处返回的prefix字符串累加值，其实和字符串切片str1[:i]是一样的，所以返回任意一个都行

# 01
def bootstrap_X(n_bootstrap,sample:list[float],real_param_to_test,alpha=0.05):
    """
    Args：
    n_bootstrap: 自助法抽样的次数
    sample: 需要进行自助法抽样的样本数据，此处设置为list浮点数列表类型
    real_param_to_test: 需要进行置信区间估计（显著性判断）的真实数据，比如说某个待检测样本的均值、方差、中位数、相关系数等
    alpha: 置信水平，默认值为0.05，注意此处使用百分位数对应置信水平（概率）
    参考https://en.wikipedia.org/wiki/Bootstrapping_(statistics)，只有当自助分布是对称的，并且以观察到的统计量为中心时，使用百分位数构造置信区间才合适，所以该算法运用前需要先对自助参数的分布形状做一个评估，即是否分布对称！！！！！！！！！！
    
    Fun：
    1,进行自助法抽样，返回一个包含n_bootstrap次抽样结果的列表；
    每次抽样计算一个统计量，此处命名为X，可以是均值、方差、中位数、相关系数等，比如说bootstrap_of_median函数，此处统一使用“X”指代
    2,利用百分位数法构造该参数X（比如说上面X指代median中位数）的置信区间;
    比如说置信水平为α，则使用(1-α/2)的百分位数和α/2的百分位数来构造置信区间，即α*100/2%和(1-α/2)*100%分位数；
    3,返回置信区间的下限和上限；
    4,顺便我们可以判断一下某个真实样本的参数数据（真实数据）是否在置信区间内，即是否显著，即real_param_to_test是否在置信区间内；
    
    """
    import numpy as np
    from scipy import stats
    
    #  注意下面都是以X代称需要估计的统计量
    bootstrap_X = [] # 存储自助法抽样获取的多次抽样结果，比如说是bootstrap_median
    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(sample,size=len(sample),replace=True)
        bootstrap_X.append(np.X(bootstrap_sample))  # np.X指代任何python实现中能够计算指标X的api函数，比如说均值用np.mean，中位数用np.median等

    ci_lower = np.percentile(bootstrap_X, alpha * 100 / 2)  # 置信区间下限
    ci_upper = np.percentile(bootstrap_X, (1 - alpha / 2) * 100)  # 置信区间上限
    
    # 下面是判断真实数据是否在置信区间内,即是否显著 
    is_significant = not (ci_lower <= real_param_to_test <= ci_upper)
    # 或者我们也可以使用
    # result = False if ci_lower <= observation_param <= ci_upper else True
    
    return ci_lower, ci_upper, is_significant
    # 也可以return result


##############################################################################################################################################################################
# 02
# ！！！比如说以自助法估计中位数统计量为例，构造相应百分位数的置信区间，用于判断抽取的某个样本中的中位数是否在置信区间内，以推断该样本是否来自于同一个抽样总体，即是否差异显著，见下面

def boostrap_median(n_bootstrap,smaple:list[float],real_param_to_test,alpha=0.05):
    import numpy as np
    import scipy as stats
    bootstrap_median=[]
    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(sample,size=len(sample),replace=True)
        bootstrap_median.append(np.median(bootstrap_sample))
    ci_lower = np.percentile(bootstrap_median,alpha*100/2)
    ci_upper = np.percentile(bootstrap_median,(1-alpha/2)*100)

    is_significant = not(ci_lower <= real_param_to_test <= ci_upper)
    return ci_lower,ci_upper,is_significant

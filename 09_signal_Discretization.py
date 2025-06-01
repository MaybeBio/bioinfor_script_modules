# 信号离散化，主要是在离散时间序列数据高上，当样本量不足的时候，如何通过提高采样频率、即通过插值方式，获取更多的伪序列数据
# 下面的seq_signal可以指代任何一维离散时间序列信号

# 01，假设原始采样频率是每个时序位置上采1个点，采样频率直观体现在倍数差异上
def signal_Discretization(seq_signal,sampling_freq):
        """
        Args:
        seq_signal:原始1D序列信号list，即seq_charge_generator的输出
        sammpling_freq:采样频率(Hz)，即每秒采样多少个点

        
        Fun:
        主要是将这个1D电荷序列信号进行离散化,
        根据给定的采样频率 sampling_freq，从原始信号中提取出符合该采样频率的离散点；
        本质上是扩充离散电荷的信号，然后运用非常粗暴的插值方法来获取新的离散信号值（这里直接就取离新插值点最近的右边的点的信号作为插值，比如说2.3就取3索引处的信号值作为插值信号）；
        参考：https://blog.csdn.net/xbinworld/article/details/65660665，其实可以做更复杂的插值方法，就是运用左右两边点的电荷信号，同时考虑距离作为权重
        X_是新的离散时间点（新的时间轴），Y_是对应的离散信号值（重采样之后的信号值）；      
        """
        # 假设原始信号的采样率为1 Hz，即每秒采样1个点（如果我们将原始序列视为是时序数据，那就是每1个位置采样1个点）
        original_length = len(seq_signal) # 原始信号长度，也就是原始信号x轴
        original_samppling_freq = 1.0 # 原始信号的采样率
        
        # 计算新的采样点数
        # original_sampling_freq（每个位置采多少样）对应original_length（多少个位置）采样数，提供新的采样频率实际上就是倍数关系，向下取整(正数上类似int)
        n = math.floor(original_length * sampling_freq / original_samppling_freq)
        
        # 生成新的时间轴(离散点)，即新的采样点，均匀
        X_ = np.linspace(0,original_length-1,n,endpoint=True)
        # 通过插值提取新的离散信号值
        # floor是下舍入取整，ceil是上舍入取整，参考：https://www.runoob.com/python/func-number-ceil.html
        # 向上还是向下影响不大，我们考虑电荷块信号的状态持续性或者是记忆性，说不清楚是向上还是向下？
        Y_ = np.asarray([seq_signal[math.ceil(point)] for point in X_])
        return X_, Y_
##############################################################################################################################################################################

# 02，离散化策略通01，但是取值的插值方式上参考https://blog.csdn.net/xbinworld/article/details/65660665使用线性插值
def signal_Discretization(seq_signal,sampling_freq):
        # 假设原始信号的采样率为1 Hz，即每秒采样1个点（如果我们将原始序列视为是时序数据，那就是每1个位置采样1个点）
        original_length = len(seq_signal) # 原始信号长度，也就是原始信号x轴
        original_samppling_freq = 1.0 # 原始信号的采样率
        
        # 计算新的采样点数
        # original_sampling_freq（每个位置采多少样）对应original_length（多少个位置）采样数，提供新的采样频率实际上就是倍数关系，向下取整(正数上类似int)
        n = math.floor(original_length * sampling_freq / original_samppling_freq)
        
        # 生成新的时间轴(离散点)，即新的采样点，均匀
        X_ = np.linspace(0,original_length-1,n,endpoint=True)
        # 通过插值提取新的离散信号值

        # 如果参考https://blog.csdn.net/xbinworld/article/details/65660665，使用更加复杂的插值方法的话
        # 首先就是取X_左边的点math.floor(point)和右边的点math.ceil(point)，
        # 应该判断一下，如果X_是整数，说明这个是原来的采样点，那么就直接取原来的点的值，也就是math.ceil/floor都可以（其实这个就是判断方法math.floor(point) == point）
        # 如果是新采样的点，也就是不是整数的话，比如说是25.7，那么需要获取左边的整数点math.floor(point)25和右边的整数点math.ceil(point)26，然后拿距离做一个权重
        def point_interpolation(point):
            return (math.ceil(point)-point)/(math.ceil(point)-math.floor(point))*seq_signal[math.floor(point)] + (point-math.floor(point))/(math.ceil(point)-math.floor(point))*seq_signal[math.ceil(point)]
        Y_interpolation = [ seq_signal[math.floor(point)] if math.floor(point) == point else point_interpolation(point) for point in X_ ]
        return X_, Y_interpolation

##############################################################################################################################################################################

# 03，同样离散化方式同01、02，只不过插值方式直接使用numpy的函数
def signal_Discretization(seq_signal,sampling_freq):
        original_length = len(seq_signal) 
        original_samppling_freq = 1.0 
        
        # 计算新的采样点
        n = math.floor(original_length * sampling_freq / original_samppling_freq)
        
        # 生成新的时间轴(离散点)，即新的采样点，均匀
        X_ = np.linspace(0,original_length-1,n,endpoint=True)
        # 通过插值提取新的离散信号值
        # 或者使用numpy中的插值方法numpy.interp()，参考：https://blog.csdn.net/qq_41688455/article/details/104352879
        Y_ = [np.interp(point,np.arange(len(seq_signal)), seq_signal) for point in X_]
        return X_,Y_

  ##############################################################################################################################################################################

  # 04，参考LCR-based_CBI/search_lcrs.py
  def signal_discretisation(seq_signal, sampling_frequency):
        # 原始计算n的方法，弃用！原理未知！！！！！！！！！！！！！！！！！！！！！！！！！
        # 从概念上看是每个位置采多少个样之后再除以2π，ω=2*pi*f，f是采样频率
        # 这取决于问题是从频率看，还是从角频率看了
        # n = math.floor(len(seq_signal) / (2 * math.pi /sampling_freq))
        n = math.floor(len(seq_signal) / (2 * math.pi / sampling_frequency))
        X_ = np.linspace(0, len(seq_signal) - 1, n, endpoint= True)
        Y_ = np.asarray([seq_signal[math.ceil(point)] for point in X_])
        return X_, Y_


##############################################################################################################################################################################
 # 05 废稿

    @staticmethod
    def signal_Discretization(charge_signal,sampling_freq):
        """
        Args:
        charge_signal:原始电荷信号list，即seq_charge_generator的输出
        sammpling_freq:采样频率(Hz)，即每秒采样多少个点

        
        Fun:
        主要是将这个1D电荷序列信号进行离散化,
        根据给定的采样频率 sampling_freq，从原始信号中提取出符合该采样频率的离散点；
        本质上是扩充离散电荷的信号，然后运用非常粗暴的插值方法来获取新的离散信号值（这里直接就取离新插值点最近的右边的点的信号作为插值，比如说2.3就取3索引处的信号值作为插值信号）；
        参考：https://blog.csdn.net/xbinworld/article/details/65660665，其实可以做更复杂的插值方法，就是运用左右两边点的电荷信号，同时考虑距离作为权重
        X_是新的离散时间点（新的时间轴），Y_是对应的离散信号值（重采样之后的信号值）；
    
        eg.
        其实后面也没有用上！！！！！！！！！！！！！！！！！！！！！
        ！！！！！！！！！！！！（这一段代码需要重新检测Q1，信号的离散化采样）
        
        """
        # 假设原始信号的采样率为1 Hz，即每秒采样1个点（如果我们将原始序列视为是时序数据，那就是每1个位置采样1个点）
        original_length = len(charge_signal) # 原始信号长度，也就是原始信号x轴
        original_samppling_freq = 1.0 # 原始信号的采样率
        
        # 计算新的采样点数
        # original_sampling_freq（每个位置采多少样）对应original_length（多少个位置）采样数，提供新的采样频率实际上就是倍数关系，向下取整(正数上类似int)
        n = math.floor(original_length * sampling_freq / original_samppling_freq)
        
        # 生成新的时间轴(离散点)，即新的采样点，均匀
        X_ = np.linspace(0,original_length-1,n,endpoint=True)
        # 通过插值提取新的离散信号值
        # floor是下舍入取整，ceil是上舍入取整，参考：https://www.runoob.com/python/func-number-ceil.html
        # 向上还是向下影响不大，我们考虑电荷块信号的状态持续性或者是记忆性，说不清楚是向上还是向下？
        Y_ = np.asarray([charge_signal[math.ceil(point)] for point in X_])
        return X_, Y_

        # 另法1：
        # 如果参考https://blog.csdn.net/xbinworld/article/details/65660665，使用更加复杂的插值方法的话
        # 首先就是取X_左边的点math.floor(point)和右边的点math.ceil(point)，
        # 应该判断一下，如果X_是整数，说明这个是原来的采样点，那么就直接取原来的点的值，也就是math.ceil/floor都可以（其实这个就是判断方法math.floor(point) == point）
        # 如果是新采样的点，也就是不是整数的话，比如说是25.7，那么需要获取左边的整数点math.floor(point)25和右边的整数点math.ceil(point)26，然后拿距离做一个权重
        """
        def point_interpolation(point):
            return (math.ceil(point)-point)/(math.ceil(point)-math.floor(point))*charge_signal[math.floor(point)] + (point-math.floor(point))/(math.ceil(point)-math.floor(point))*charge_signal[math.ceil(point)]
        Y_interpolation = [ charge_signal[math.floor(point)] if math.floor(point) == point else point_interpolation(point) for point in X_ ]
        return X_, Y_interpolation
        """
        
        # 另法2：
        # 或者使用numpy中的插值方法numpy.interp()，参考：https://blog.csdn.net/qq_41688455/article/details/104352879
        # Y_ = [np.interp(point,np.arange(len(charge_signal)), charge_signal) for point in X_]
    
        # 原始计算n的方法，弃用！原理未知！！！！！！！！！！！！！！！！！！！！！！！！！
        # 从概念上看是每个位置采多少个样之后再除以2π，ω=2*pi*f，f是采样频率
        # 这取决于问题是从频率看，还是从角频率看了
        # n = math.floor(len(charge_signal) / (2 * math.pi /sampling_freq))

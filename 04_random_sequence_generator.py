def random_seq_generator(length,background_freq,num_seq=1):
    """
    Args:
    length:生成1个随机序列的长度
    background_freq:背景频率，即生成的随机序列中每一个字符应该服从的概率分布字典，
    注意：seq中抽取的字符串顺序和背景频率中的字符串顺序需要对应，即字符串是["A","C","G","T"]，背景频率是{"A":0.25,"C":0.25,"G":0.25,"T":0.25}，即字符串中的ACGT和背景频率中的ACGT顺序一致！！！！！！！！！！！！
    注意：background_freq是该函数中random.choice()函数中的p参数，p参数是一个一维数组，表示每个元素被选中的概率，因为只能传入1个一维数组，也就是不能传{"A":0.25,"C":0.25,"G":0.25,"T":0.25}，只能传{0.25,0.25,0.25,0.25}
    所以我们可以在参数设置的时候传入1个字典dict，然后我们提取出其中的value作为random.choice()函数中的p参数所需要的1D-array形式；
    当然也可以直接提供整理好的1D-array形式的背景频率分布；
    
    num_seq:生成多少个随机序列，默认为1即生成1个随机序列

    Fun:
    生成指定num_seq个随机序列，长度为length，背景频率为background_freq，返回这么多个随机序列的列表，列表中的每个元素都是一个随机序列
    """
    # 随机序列需要设置seed，方便调试以及复现
    import numpy as np
    np.setseed(2025)
    
    seq = np.asarray(["character1","character2","charactern"])  # 比如说seq_DNA = np.asarray(["A","C","G","T"])
    
    random_seq = [list(np.random.choice(seq,size=length,replace=True,p=background_freq)) for i in range(num_seq)] # 默认是有放回抽样
    return random_seq
    
    
# 如果写的复杂点，就是提供用于抽样产生随机序列的字符表也设置成1个参数，可以扩写
def random_seq_generator(seq_symbol:list[str],length,background_freq:list[float],num_seq=1):
    """
    Args:
    seq_symbol:生成随机序列的字符表，比如说DNA的话，我们提供的是["A","C","G","T"]，
    那么我们只需要再将其进行array数组转换即可，即seq_DNA = np.asarray(["A","C","G","T"])，即np.asarray(seq_symbol)
    
    length:生成1个随机序列的长度
    background_freq:背景频率，即生成的随机序列中每一个字符应该服从的概率分布字典，
    注意：seq_symbol中抽取的字符串顺序和背景频率background_freq中的字符串顺序需要对应，即字符串是["A","C","G","T"]，背景频率是{"A":0.25,"C":0.25,"G":0.25,"T":0.25}，即字符串中的ACGT和背景频率中的ACGT顺序一致！！！！！！！！！！！！
    
    """
    import numpy as np 
    np.setseed(2025)
    
    seq_array = np.asarray(seq_symbol)  # 比如说seq_DNA = np.asarray(["A","C","G","T"])
    
    # 注意先将生成的序列转换为list，当然也可以不转换为list（因为列表生成式中每一个元素不一定得是list，可以是字符串str等其他类型）,注意是list()
    # 如果转换为list的话，实际上是将该子元素转换为1个可迭代对象，比如说list("ACGT")，那么实际上就是["A","C","G","T"]，也就是将字符串转换为列表，注意不是["ACGT"]
    # 然后对于每一个子元素，再用list进行整合，也就是最外面的[]，标准的列表生成式
    random_seq = [list(np.random.choice(seq_array,size=length,replace=True,p=background_freq) for i in range(num_seq))]
    return random_seq
    
    
# 如果不将每一个生成的随机序列转换为list对象的话，可以按照原样保存为str字符串对象

def random_seq_genrator(seq_symbol:list[str],length,background_freq:list[float])

"""
构建IDR序列特征数据集+label的脚本，
对于cell中的蛋白质数据，可以结合label数据，也就是相分离的分选系数数据；
对于不是cell中的蛋白质，本脚本同样可以直接构建IDR序列相关的特征数据，就算没有label，也可以做无监督ML等
IDR序列相关的特征，主要是集中于结构、电荷性质、aa组成、疏水性等物理化学性质方面。

主要修改点是使labels_df参数可选，当没有label数据时也能构建特征数据集，推广点说法，就是是否提供label数据；
如果有label就做监督学习，如果没有label就做无监督学习
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import ast
from localcider.sequenceParameters import SequenceParameters
import argparse



# 下面的get_featureX函数用于获取其他自定义的序列特征，有需要的时候可以手动取消注释添加
""""
def get_featureX(sequence):  
    '''
    Args:
        sequence (string): 主要针对一些在序列上计算的特征数据，所以输入参数就是序列，也可以输入其他
    
    Fun:
        这里可以添加一些其他的特征计算方法，有些参数并没有特定的实现方法；
        比如说localcider中已经实现的特征数据，并不一定符合我们序列分析的需求，
        或者是说我们有其他有需要分析的特征数据，
        以及从其他文献中获取的特征数据，都可以在这里添加
    '''
    # 比如说有这么一个特征，称之为featureX
    # 那么在计算出这个特征之后，就可以在下面定义的函数featurize_data中添加这一列：
    df['FeatureX'] = np.zeros(sample_points)

    # 然后在这里计算这个特征的值
    # 然后可以直接对原始序列数据中需要计算的col部分输入进行计算
    # 比如说某个feature就是针对全长IDR序列进行计算，那么就可以使用前面的IDR sequence Combined这一列数据进行计算
    # df['FeatureX'].iloc[idx] = calculate_featureX(sequence)
    df['FeatureX'].iloc[idx] = get_featureX(df['IDR Sequence Combined'].iloc[idx])
    df['FeatureX'].iloc[idx] = get_featureX(df["input_you_need"].iloc[idx])
"""

def featurize_data(df, labels_df=None):
    """
    Args:
        df:包含蛋白质IDR序列信息的DataFrame，可以直接由01_get_idrs.py脚本获得
        labels_df:包含蛋白质IDR分选系数的DataFrame，可以直接由cell文献的table S2获得(如果是cell文献中的蛋白质，是有这个数据的，如果是普通的无监督无label的蛋白质序列，是没有这个一个参数数据输入的，可以缺省)

    Fun:
        提取IDR序列的特征数据，主要是物理化学性质方面的特征数据，同样是返回一个DataFrame数据框
        另外可以考虑多增加点localcider中能够实现但是没有添加进来的特征数据（比如说也可以是一些基于序列数据的）
        

    Warnings！！！！！
        这里唯一需要注意的点就是：
        首先是带电残基aa名单定义的共识，主要是带正电的到底包不包括H，也就是组氨酸；
        另外就是等电点与PH的关系，实际上等电点就是PH，本质上还是那个带电的问题，因为本质上电荷的带电性质，本身也是看环境PH与等电点的比较；
        所以我们需要注意下面计算的各个只要是涉及到带电方面，而不是残基方面的特征，都需要注意这个残基定义的带电到底是正电还是负电，另外到底是否需要考虑到环境PH的影响

        目前的对策就是：
        1，识别哪些是计算涉及到带电残基带电信息本身，哪些只是涉及到特定氨基酸而已，但是并没有涉及到具体的带电性质的计算；
        2，对于那些涉及到带电残基计算的性质，仔细分析计算的算法细节，主要是判断是否需要计入H；
        如果是localcider等软件写的，看看自己能不能按照前面的get_featureX的方式来更新覆盖，也就是自己来写这个脚本；
        如果是没有实现的参数，就直接自己写模块来实现；
        选择哪种方式都可以，主要目的是为了统一；
        3，统一完带电氨基酸的定义以及识别之后，另外就是对等电点影响的考虑，主要是判断是否需要提供PH信息，
        如果需要的话，是哪些参数需要，以及需要的话，是提供什么值，比如说细胞核里的PH就是7.2

    """
    # 统计样本数
    sample_points = df['IDR Sequence List'].size
    print("Size of the dataset: ", str(sample_points), " IDRs ....")  

    # 下面构建数据集都是从列的方向，也就是从一个特征循环所有的样本方向
    
    # 收集IDR序列，合并的IDR序列，因为是序列，所以可以使用语言模型的处理技巧，后续可以使用ljw的方法转换为一些token特征
    df['IDR Sequence Combined'] = np.array(['']*sample_points)

    # IDR序列的一些特征数据，主要是计算一些特征数据
    df['IDR Count'] = np.zeros(sample_points) # IDR序列的个数
    df['Total IDR Length'] = np.zeros(sample_points) # IDR序列全长，应该是stitching起来了

    # 使用localcider计算提取的特征数据  
    # 一些关于局部电荷分布的特征数据
    df['Fraction Positive'] = np.zeros(sample_points) # 正电荷的比例
    df['Fraction Negative'] = np.zeros(sample_points) # 负电荷的比例
    df['Fraction Expanding'] = np.zeros(sample_points) # 对蛋白质结构扩展有帮助的氨基酸的比例，也就是膨胀残基比例，(E/D/R/K/P)对链延伸有帮助的残基比例
    df['FCR'] = np.zeros(sample_points) # FCR是Fraction of Charged Residues，带电残基的比例
    df['NCPR'] = np.zeros(sample_points) # NCPR是Net Charged Protein Residue，净电荷残基的比例
    df['Kappa'] = np.zeros(sample_points) # 反映蛋白质序列局部电荷分布模式不对称性的参数
    df['Omega'] = np.zeros(sample_points) # 反映蛋白质序列局部pro+带电残基分布模式不对称性的参数
    df['Isoelectric Point'] = np.zeros(sample_points) # 等电点pI值（本质上是PH值），蛋白质在该pH值下带净电荷为0
    df['Delta'] = np.zeros(sample_points) # 计算Kappa参数时，获取的当前蛋白质序列的电荷分布模式不对称性的未归一化的值，注意与Kappa的区别在于delta并没有归一化
    df['Delta Max'] = np.zeros(sample_points) # 同上，在计算Kappa参数时，获取的当前蛋白质正负电荷占比模式下，所有可能排布的序列下，电荷分布模式不对称性的可能最大值，用于对delta进行归一化
    df['SCD'] = np.zeros(sample_points) # Sequence charge decoration，序列电荷修饰，反映带电残基空间排布模式（也就是聚类模式）的参数

    # 一些关于蛋白质结构的特征数据    
    df['Uversky Hydropathy'] = np.zeros(sample_points) # 描述蛋白质序列的疏水性的归一化参数
    df['PPII Propensity'] = np.zeros(sample_points) # 描述蛋白质二级结构PPII，也就是II型聚pro螺旋倾向性的平均值

    # 只有当提供了label数据时才添加分选系数列
    if labels_df is not None:
        df['Log Partition Ratios'] = np.zeros(sample_points) # 分选系数的对数值，作为预测的label数据
        print("Label data (partition ratios) will be included in the feature matrix.")
    else:
        print("No label data provided. Only features will be computed.")

    # 然后是正式开始构建整个数据集
    for idx in range(sample_points):
        # 首先是合并所有的IDR序列，也就是stitching序列，再统计一下长度
        for idr_seq in df["IDR Sequence List"].iloc[idx]:
            df['IDR Sequence Combined'].iloc[idx] += str(idr_seq)
            df['IDR Count'].iloc[idx] += 1
        # 对前面已经合并之后的序列统计长度
        df['Total IDR Length'].iloc[idx] = len(df['IDR Sequence Combined'].iloc[idx])
        
        # 然后就是localcider的一些与电荷模式相关的特征数据
        # 我们从localcide中获取的所有IDR序列相关特征数据，都是对每一个蛋白质合并之后的IDR序列进行计算的
        obj = SequenceParameters(df['IDR Sequence Combined'].iloc[idx])
        df['Fraction Positive'].iloc[idx] = obj.get_fraction_positive() # 正电荷比例
        df['Fraction Negative'].iloc[idx] = obj.get_fraction_negative()
    
        # 需要注意一下，所计算的特征是否需要考虑PH的影响
        df['Fraction Expanding'].iloc[idx] = obj.get_fraction_expanding(pH=7.2) # 膨胀残基比例
        df['FCR'].iloc[idx] = obj.get_FCR(pH=7.2) # 带电残基比例
        df['NCPR'].iloc[idx] = obj.get_NCPR(pH=7.2) # 净电荷残基比例
        df['Omega'].iloc[idx] = obj.get_Omega() # Omega参数
        df['Kappa'].iloc[idx] = obj.get_kappa() # Kappa参数
        df['Isoelectric Point'].iloc[idx] = obj.get_isoelectric_point() # 等电点，也就是PH值
        df['Uversky Hydropathy'].iloc[idx] = obj.get_uversky_hydropathy() # 归一化之后的疏水性值
        df['PPII Propensity'].iloc[idx] = obj.get_PPII_propensity(mode='hilser') # PPII螺旋倾向性
        df['Delta'].iloc[idx] = obj.get_delta() # 计算kappa参数时的delta参数
        df['Delta Max'].iloc[idx] = obj.get_deltaMax() # 计算kappa参数时的delta max参数
        df['SCD'].iloc[idx] = obj.get_SCD() # 序列电荷修饰参数，这一个参数可以自己定义，也可以直接由localcider计算
        print('Finished featurizing ... ', str(idx+1) ,' IDRs ...')


        # 只有当提供了label数据时才尝试添加分选系数
        if labels_df is not None:
            df['Log Partition Ratios'].iloc[idx] = labels_df[labels_df['Uniprot ID'] 
                                                                    == df['Uniprot ID'].iloc[idx]]['Log2(average_P/S)']
    return df

if __name__ == "__main__":
    
    # 读入命令行参数
    parser = argparse.ArgumentParser(description='Run script to generate dataset of IDR features, Note that more features could be added in function get_featureX within the script on your own')
    parser.add_argument('--i', help="Name of input file containing lists of IDRs in each protein", required=True)
    # 修改：使partition ratios参数可选
    parser.add_argument('--l', help="Name of input file contaning labels (optional)", required=False, default=None)
    parser.add_argument('--o', help="Name of output file to export the data matrix into", required=True)
    args = parser.parse_args()
    
    idr_file_name = args.i
    label_file_name = args.l
    output_file_name = args.o

    # 读取IDR序列数据
    df = pd.read_csv(idr_file_name)
    # 将读入数据文件中的IDR序列从字符串转变为list列表
    df['IDR Sequence List'] = [ast.literal_eval(entry) for entry in df['IDR Sequence List']]
    
    # 我们这里需要排除那些没有IDR序列的蛋白质，无论是否是cell文献中的蛋白质，还是其他的蛋白质
    # 我们都只保留具有IDR序列信息的行（蛋白质）
    # 可以定义1个函数用判断元素是否为空列表list
    def is_empty_list(list):
        return len(list) == 0
    # 然后排除掉那些没有IDR序列的蛋白质行
    df = df[~df['IDR Sequence List'].apply(is_empty_list)]

    # 只有当提供了partition ratios文件时才读取分选系数数据
    labels_df = None
    if label_file_name is not None:
        labels_df = pd.read_csv(label_file_name)
        print(f"Loaded label data(partition ratios) from {label_file_name}")
    else:
        print("No label data provided. Only features will be computed.")

    # 然后进行特征数据的提取
    df = featurize_data(df, labels_df)

    # 保存数据矩阵到指定的输出文件
    df.to_csv(output_file_name, index=False)
    print("Data matrix saved to: ", output_file_name)
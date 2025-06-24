# 01

# 开头部分
import argparse # 用于解析命令行参数

# 结尾部分
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="为输入的蛋白质家族蛋序列建立多隐状态信号的1阶HMM模型")
    parser.add_argument("--fasta_file", type=str, help="需要输入的蛋白质家族的fasta序列文件路径")
    parser.add_argument("--n_states", type=int,default=3, help="隐状态编码个数，默认为3")
    parser.add_argument("--n_fits", type=int, default=1000, help="baum-welch算法中HMM模型拟合次数，默认为1000")
    parser.add_argument("--iter_n", type=int, default=100, help="baum-welch算法的最大迭代次数，默认为100，每次初始化的模型都要迭代该次数")
    parser.add_argument("--best_model_filename", type=str, help="最优HMM模型的保存文件名，建议是pkl格式文件，方便后续加载使用")
    
    args = parser.parse_args()

    first_order_HMM_model(args.fasta_file, args.best_model_filename, args.n_states, args.n_fits, args.iter_n)

===============================================================================================================================================================================

# 02

# 开头
import argparse 

# 结尾
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="该python命令行工具的帮助文档说明")
    parser.add_argument("--参数xx", type=指定参数数据类型,default=如果没有显式提供参数时默认值, help="对该参数的帮助文档说明")
    # 参数xx可以有多个
    
    args = parser.parse_args()

    要运行的主函数(args.参数xx) # 对应上面的参数xx


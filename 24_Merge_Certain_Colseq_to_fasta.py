# 主要是为了合并某一个df、或者是某一个输入csv（或者是其他分隔符文件）中具有list、dict、tuple等多值存储结构的某列
# 比如说一个蛋白质序列的csv文件，其中某一列是零散的功能结构域的子序字符串str的list数据类型，那么我想将这一列合并起来，然后获取一个蛋白质id+这一列合并功能结构域的fasta序列文件
# 那么我就可以使用这个函数，将这一列功能结构域进行合并，然后输出为fasta文件

# 1，对pandas df进行处理，其中的df可以换成csv文件，使用pandas读入，然后一些前置列名需要注意，在特定场景下使用需要修改

def merge_colseq_to_fasta(df,col_name,output_file):
    """
    Args:
        df: pandas DataFrame，包含蛋白质ID和对应序列列表，以及其他特征列(可以是由csv文件使用pandas读入进来)
        col_name: 需要合并的序列列名
        output_file: 输出的fasta文件路径
    Fun:
        原始输入文件中的某一列的值是一个list列表，有很多序列，将该指定列的所有list中的子序列合并为一个字符串，然后依然作为一行不变，并保存为fasta格式
    """
    import pandas as pd
    with open(output_file,"w") as f:
        for idx, row in df.iterrows():
            protein_id = row['Protein ID']
            seq_list_str = row[col_name]
            if seq_list_str == "[]":
                continue
            seq_list = eval(seq_list_str)  # 将字符串转换为列表
            seq = "".join(seq_list)
            f.write(f">{protein_id}\n{seq}\n")

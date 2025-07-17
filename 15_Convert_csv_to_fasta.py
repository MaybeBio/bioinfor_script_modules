# 现在尝试将1个csv文件中指定的2列转换为fasta格式，定义1个函数
# 默认输入的是csv文件，当然可以在pandas阅读中修改seq
# 另外默认第一行是列名，所以skiprows=1，可以进一步修改用于自动识别skip与否
def convert_csv_to_fasta(csv_file,id_col,seq_col,output_file):
    """
    Arg:
        csv_file: 包含蛋白质IDR和对应序列的文件（当然读入之后的df中有很多其他的列）
        id_col: 蛋白质ID所在的列名
        seq_col: 序列所在的列名
        output_file: 输出的fasta文件路径
    Fun:
        将指定的蛋白质ID和序列转换为fasta格式并保存到文件中，如果id_col有重名子，就按照顺序重命名为原来的id_1, id_2等
    """
    import pandas as pd
    from collections import Counter
    # 用于承接后面的fasta
    fasta_lines = []
    # 另外对于重复id也需要进行处理
    id_counter = {}

    # 统计重复id
    # 首先是Counter函数会统计每一个id的频数，然后构建一个id:频数的字典
    id_counts = Counter(df[id_col])
    # 然后对于这个字典，如果这个字典中的值也就是频数＞1，也就是有重复的，然后这里就会具体记录下来这个重复的蛋白质id，方便后来检索
    duplicates = {k:v for k,v in id_counts.items() if v > 1}

    # 主要是列名需要去除，当然可以做得复杂点就是自动判断是否要skiprows
    df = pd.read_csv(csv_file,skiprows=1)
    for idx,row in df.iterrows():
        protein_id = row[id_col]
        seq = row[seq_col]

        # 开始处理重复id
        # 逻辑是第1个重复的蛋白质是原名id，第2个是id_1，第3个是id_2，以此类推
        if protein_id in id_counter:
            # 如果这个蛋白质id已经存在于字典中，就说明是是第2次检索到了，因为下面的第一次检索是初始化记录
            id_counter[protein_id] += 1 # 增加计数，同时用于重命名的后缀
            unique_id = f"{protein_id}_{id_counter[protein_id]}"
        else:
            # 主要是对每一个蛋白质id的初始化，都初始化为0
            id_counter[protein_id] = 0
            unique_id = protein_id # 没有重复的，包括有重复但是第一个蛋白质的，都使用原名id

            fasta_lines.append(f">{unique_id}\n{seq}\n")
    # 将所有的fasta行写入到输出文件中
    with open(output_file,"w") as f:
        # writelines会原封不动地将列表中的每个字符串都写入文件
        f.writelines(fasta_lines)
        



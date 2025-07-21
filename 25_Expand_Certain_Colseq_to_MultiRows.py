# 将某个数据框中某列解包展开
# 比如说我有1个数据框，其中某一列col1的值是一个list等嵌合数据类型，里面是字符串str的list；我可以将这一个列表中的每一个字符串序列都展开成一行
# 比如说P49247有2条IDR序列，那么我就展开成两行，col1分别放IDR1以及IDR2；另外注意id列，那么我所展开的2行，需要P49247_1以及P49247_2来区分

def expand_col_seq(df,col_name):
    """
    将序列中的IDR Sequence List展开，并且注意id重名的问题
    比如说P49247有2条IDR序列，那么我所展开的2行，需要P49247_1以及P49247_2来区分
    """
    expand_rows = []
    id_counter = {}  # 用于记录每个ID的出现次数

    for _,row in df.iterrows():
        col_seqs = eval(row[col_name])
        base_id = row["Uniprot ID"]

        # 为每一个子序列创建一行
        for seq in col_seqs:
            # 首先进行base_id在id_counter中的初始化，无论是没有重复的id，还是有重复的id，的第一次
            if base_id not in id_counter:
                id_counter[base_id] = 0
            else:
                # 反之如果之前已经初始化过了，然后这次是≥第2次出现
                id_counter[base_id] += 1
            
            # 然后开始整理每一个IDR序列的id
            if id_counter[base_id] == 0:
                # 说明这个时候的序列是第一次出现，那么第一个id就不加后缀，直接使用原名
                unique_id = base_id 
            else:
                # 反之，就直接使用当前的计数器来添加后缀
                unique_id = f"{base_id}_{id_counter[base_id]}"

            # 这里按照自己需求来修改
            expand_rows.append({'Uniprot ID': unique_id,
                        'Log Partition Ratios': row["Log Partition Ratios"],
                        'IDR_sequence': seq})
    return pd.DataFrame(expand_rows)

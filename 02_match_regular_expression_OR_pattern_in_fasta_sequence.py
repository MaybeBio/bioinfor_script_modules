# 对每一个蛋白质等fasta序列文件，使用某种结构域的正则表达式等进行匹配，返回每个蛋白质序列中该蛋白质匹配的hits
# 参考https://github.com/iankorf/MCB185/blob/main/course/unit6-data.md

# 01 主要使用re.finditer（非重叠匹配，每次匹配的时候都将上一次匹配的终点作为下一次匹配的起点，所以本质上是非重叠匹配）

def main(fasta_file,output_file,regular_pattern="C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H"):
    """
    Funtion:
    使用前面定义的函数，解析fasta序列文件，然后使用正则表达式匹配蛋白质序列，最后将匹配到的蛋白质id、匹配到的序列、匹配到的序列的起始和终止位置写入到输出文件中
    
    Args:
    fasta_file: str, fasta格式的蛋白质序列文件的路径，要求同parse_fasta函数中的要求
    output_file: str, 输出文件的路径
    regular_pattern: str, 正则表达式的模式，用于匹配蛋白质序列，可以自定义，默认为"C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H"，即默认对C2H2 zinc finger domain进行匹配
    
    Returns:
    返回对输入fasta序列文件的正则表达式匹配结果，写入到输出文件中，即tsv文件，其中包含蛋白质id、匹配到的序列、匹配到的序列的起始和终止位置，1个蛋白质id可能有多个匹配到的序列，每个序列都会单独写入1行
    
    """
    protein_sequences = parse_fasta(fasta_file)
    # 解析批量蛋白质序列文件之后，获取蛋白质id+序列的字典
    
    with open(output_file, "w") as out:
        # 以写入模式打开输出文件
        out.write("Protein_ID\tStart\tEnd\tMatched_Sequence\n")
            
        for protein_id,current_protein_sequence in protein_sequences.items():
            current_protein_match = re.finditer(regular_pattern,current_protein_sequence)
            for match in current_protein_match:
                start = match.start() # 如果想要1-based的话，需要+1，即start = match.start()+1 
                end = match.end() # 如果想要1-based的话，需要+1，即end = match.end()+1
                matched_sequence = match.group()
                out.write(f"{protein_id}\t{start}\t{end}\t{matched_sequence}\n")

##############################################################################################################################################################################

# 02
# 或者只是作为一个单纯的module而言，不考虑main的文件操作输出

def main(fasta_file,regular_pattern="C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H"):
    """
    Funtion:
    使用前面01定义的函数，解析fasta序列文件，然后使用正则表达式匹配蛋白质序列，最后将匹配到的蛋白质id、匹配到的序列、匹配到的序列的起始和终止位置输出
    
    Args:
    fasta_file: str, fasta格式的蛋白质序列文件的路径，要求同01 parse_fasta函数中的要求
    regular_pattern: str, 正则表达式的模式，用于匹配蛋白质序列，可以自定义，默认为"C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H"，即默认对C2H2 zinc finger domain进行匹配
    
    Returns:
    返回对输入fasta序列文件的正则表达式匹配结果，其中包含蛋白质id、匹配到的序列、匹配到的序列的起始和终止位置，1个蛋白质id可能有多个匹配到的序列，每个序列都会单独写入1行
    
    """
    protein_sequences = parse_fasta(fasta_file)
    # 解析批量蛋白质序列文件之后，获取蛋白质id+序列的字典
            
    for protein_id,current_protein_sequence in protein_sequences.items():
        current_protein_match = re.finditer(regular_pattern,current_protein_sequence)
        for match in current_protein_match:
            start = match.start() # 如果想要1-based的话，需要+1，即start = match.start()+1 
            end = match.end() # 如果想要1-based的话，需要+1，即end = match.end()+1
            matched_sequence = match.group()
            print(f"{protein_id}\t{start}\t{end}\t{matched_sequence}\n")


##############################################################################################################################################################################

# 03 
# 考虑到正则表达式的参数显示/默认传入问题，我们还是对None值进行一个判断，03是对02做的一个小小的修改

def main(fasta_file,regular_pattern=None):
    """
    Funtion:
    使用前面01定义的函数，解析fasta序列文件，然后使用正则表达式匹配蛋白质序列，最后将匹配到的蛋白质id、匹配到的序列、匹配到的序列的起始和终止位置输出
    
    Args:
    fasta_file: str, fasta格式的蛋白质序列文件的路径，要求同01 parse_fasta函数中的要求
    regular_pattern: str, 正则表达式的模式，用于匹配蛋白质序列，可以自定义，默认为"C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H"，即默认对C2H2 zinc finger domain进行匹配
    
    Returns:
    返回对输入fasta序列文件的正则表达式匹配结果，其中包含蛋白质id、匹配到的序列、匹配到的序列的起始和终止位置，1个蛋白质id可能有多个匹配到的序列，每个序列都会单独写入1行
    
    """
    protein_sequences = parse_fasta(fasta_file)
    # 解析批量蛋白质序列文件之后，获取蛋白质id+序列的字典

    # 如果 regular_pattern 是 None，则使用默认值
    if regular_pattern is None:
        regular_pattern = "C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H"

    for protein_id,current_protein_sequence in protein_sequences.items():
        current_protein_match = re.finditer(regular_pattern,current_protein_sequence)
        for match in current_protein_match:
            start = match.start() # 如果想要1-based的话，需要+1，即start = match.start()+1 
            end = match.end() # 如果想要1-based的话，需要+1，即end = match.end()+1
            matched_sequence = match.group()
            print(f"{protein_id}\t{start}\t{end}\t{matched_sequence}\n")

##############################################################################################################################################################################

# 04
# 此处我们考虑重叠匹配的case（尽管这种情况在生物学中不太常见，尤其是我们没有听说过重叠的结构域case），使用re.search，因为这个函数只输出first匹配

import re

def find_overlapping_matches(pattern, sequence):
    """
    查找所有匹配项，包括重叠的部分。

    Args:
    pattern: str, 正则表达式模式。
    sequence: str, 要匹配的字符串。

    Returns:
    list: 包含所有匹配项的列表，每个匹配项是一个 (start, end, match) 元组。
    """
    matches = []
    start = 0
    while start < len(sequence):
        match = re.search(pattern, sequence[start:])
        if not match:
            break
        # 记录匹配的起点、终点和匹配的内容
        match_start = start + match.start()
        match_end = start + match.end()
        matches.append((match_start, match_end, match.group()))
        # 从下一个字符开始继续匹配，允许重叠
        start = match_start + 1
    return matches  

##############################################################################################################################################################################


# 05 
# 如果将整个仓库中01的解析fasta序列文件的模块以及此处02匹配正则表达式的模块合并在一起，合并为1个从批量batch蛋白质序列的fasta文件中解析获取蛋白质字典，然后对每一个蛋白质进行结构域模式匹配的模块

import re #正则表达式
import sys #提供参数
import argparse #用于参数解析，帮助文档

# 需要定义2个函数，1个是用于解析批量batch fasta序列文件的，也就是读入并整理fasta序列文件为1个字典
# 另外一个函数实际上就是主函数，调用fasta序列解读函数，然后再进行正则表达式的处理

def parse_fasta(fasta_file):
    """
    Function:
    从fasta序列文件中批量读取并处理蛋白质序列文件
    
    Args:
    fasta_file: str, fasta格式的蛋白质序列文件的路径,每一个蛋白质条目record的header最好是唯一的、连续的，经过处理之后的header字符串；
    比如说"ZK742.6        CE40734 WBGene00045277  status:Confirmed        UniProt:A3FPJ7  protein_id:ABN43077.1"，最好就处理打印出ZK742.6
    再比如说"sp|O60384|ZN861_HUMAN Putative zinc finger protein 861 OS=Homo sapiens OX=9606 GN=ZNF861P PE=5 SV=1"，最好也是处理打印出ZN861_HUMAN
    这些都可以在该函数中处理，也可以使用其他函数进行处理，然后获取clean id之后的fasta序列文件，再使用该函数；
    当然，如果不介意细节的话，可以直接将整个header都打印出来
    
    Returns:
    dict, 其中键是蛋白质的ID，值是蛋白质的氨基酸序列（字符串）
    
    """
    with open(fasta_file,"r") as file:
        sequences = {}
        # 存储最终蛋白质id+序列的字典
        current_id = None
        # 初始化每1个蛋白质record的id
        current_sequence = []
        # 初始化每一个蛋白质record的序列
        for line in file:
            line = line.strip() # 去除换行符，便于后续append之后真实合并
            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = ''.join(current_sequence)
                    current_sequence = []
                parts = line[1:].split()
                current_id = parts[0]
            else:
                current_sequence.append(line)
            # 第1次初始化的时候，因为没有蛋白质id，所以只执行parts以及current_id部分，实际上就是读取">"字符之后依据空格分隔的第1部分，初始化了1个蛋白质id
            # 然后接下去我们序列收集的做法都是这样的：在遇到下一个record的">"之前，我们将每一行的suquence都append起来，直到遇到下一个蛋白质id，我们才合并前1个蛋白质id的sequence
        if current_id:
            sequences[current_id] = ''.join(current_sequence)
        # 当然，前面的处理方法是在最后一个record之后，没有再遇到下一个蛋白质id的时候，我们需要再次合并最后一个蛋白质id的sequence，也就是最后1个蛋白质record需要注意一下
        
    return sequences    
    # 返回最终的蛋白质id+序列的字典       
    

def main(fasta_file,output_file,regular_pattern="C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H"):
    """
    Funtion:
    使用前面定义的函数，解析fasta序列文件，然后使用正则表达式匹配蛋白质序列，最后将匹配到的蛋白质id、匹配到的序列、匹配到的序列的起始和终止位置写入到输出文件中
    
    Args:
    fasta_file: str, fasta格式的蛋白质序列文件的路径，要求同parse_fasta函数中的要求
    output_file: str, 输出文件的路径
    regular_pattern: str, 正则表达式的模式，用于匹配蛋白质序列，可以自定义，默认为"C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H"，即默认对C2H2 zinc finger domain进行匹配
    
    Returns:
    返回对输入fasta序列文件的正则表达式匹配结果，写入到输出文件中，即tsv文件，其中包含蛋白质id、匹配到的序列、匹配到的序列的起始和终止位置，1个蛋白质id可能有多个匹配到的序列，每个序列都会单独写入1行
    
    """
    protein_sequences = parse_fasta(fasta_file)
    # 解析批量蛋白质序列文件之后，获取蛋白质id+序列的字典
    
    with open(output_file, "w") as out:
        # 以写入模式打开输出文件
        out.write("Protein_ID\tStart\tEnd\tMatched_Sequence\n")
            
        for protein_id,current_protein_sequence in protein_sequences.items():
            current_protein_match = re.finditer(regular_pattern,current_protein_sequence)
            for match in current_protein_match:
                start = match.start() # 如果想要1-based的话，需要+1，即start = match.start()+1 
                end = match.end() # 如果想要1-based的话，需要+1，即end = match.end()+1
                matched_sequence = match.group()
                out.write(f"{protein_id}\t{start}\t{end}\t{matched_sequence}\n")
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     "接受一个fasta文件，然后使用正则表达式匹配蛋白质序列，最后将匹配到的蛋白质id、匹配到的序列、匹配到的序列的起始和终止位置（注意是0-based）写入到输出文件中")
    parser.add_argument("--fasta_file", help=
                        "输入的蛋白质FASTA文件路径，要求是fasta格式，每一个蛋白质条目record的header最好是唯一的、连续的，经过处理之后的header字符串；不然就直接打印出整个header")
    parser.add_argument("--regular_pattern", 
                        help="正则表达式的模式，用于匹配蛋白质序列，可以自定义，默认为C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H",
                        default="C.{2,4}C.{3}[LIVMFYWC].{8}H.{3,5}H")
    parser.add_argument("--output_file", help="输出结构域正则表达式匹配文件,建议为tsv文件格式")

    args = parser.parse_args()
    main(args.fasta_file, args.output_file, args.regular_pattern)

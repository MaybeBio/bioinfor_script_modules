# 主要是处理批量蛋白质fasta序列文件，然后获取蛋白质id：序列的字典文件

# 01

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

##############################################################################################################################################################################
# 02

import gzip
import sys

def parse_fasta(filename):
    """
    从FASTA文件中解析蛋白质序列并返回一个字典，其中键是蛋白质的ID，值是氨基酸序列。
    
    Args:
    filename: str, FASTA格式的蛋白质序列文件路径。
              如果文件名为 '-'，则从标准输入读取；
              如果文件名以 '.gz' 结尾，则解压读取。
    
    Returns:
    dict: 包含蛋白质ID和对应氨基酸序列的字典。
    """
    # 打开文件
    if filename == '-':          
        fp = sys.stdin
    elif filename.endswith('.gz'): 
        fp = gzip.open(filename, 'rt')
    else:                          
        fp = open(filename)

    fasta_dict = {}
    name = None
    seqs = []

    while True:
        line = fp.readline()
        if line == '':  # 文件结束
            break
        line = line.rstrip()
        if line.startswith('>'):  # 处理 header 行
            if len(seqs) > 0:  # 保存前一个序列
                fasta_dict[name] = ''.join(seqs)
            # 提取简洁的 ID
            raw_header = line[1:]  # 去掉 '>'
            if " " in raw_header:
                name = raw_header.split()[0]  # 提取第一个字段作为 ID
            else:
                name = raw_header  # 如果没有空格，直接使用整个 header
            seqs = []  # 重置序列列表
        else:
            seqs.append(line)  # 收集序列行

    # 保存最后一个序列
    if name and len(seqs) > 0:
        fasta_dict[name] = ''.join(seqs)

    fp.close()
    return fasta_dict


##############################################################################################################################################################################
# 03

def parse_uniprot_fasta(path_to_fasta):
    
    """
    Argumrnts:
    filename:str,输入的fasta序列文件路径
    每个蛋白质序列的第1行类似于——》
    ">sp|A8K8V0|ZN785_HUMAN Zinc finger protein 785 OS=Homo sapiens OX=9606 GN=ZNF785 PE=1 SV=1"
    
    Returns:
    返回的是1个元组,每个元组元素是1个字典，包括键值对key-value
    键可以是蛋白质的id，或者是uniprot的accession号；
    值是该蛋白质序列的fasta文件
    
    """
    with open(path_to_fasta,"r") as fasta:
        protein_seqs = fasta.read().split(">")[1:]
        # fasta.read()是读取全部内容
        # split(">")是将读取的内容按">"分割成多个部分，返回的是一个列表，第1个是""空字符        
        # [1:]是从第2个元素开始，因为第1个元素是空字符，需要去除
        # 如果这里不用[1:]的话，可以在最后面del proteins_dict[""]
        
        proteins_dict = {} # 构造存储最终输出蛋白质id+序列的空字典
        for seq in protein_seqs:
            id_start = seq.find("|")+1
            id_end = seq.find("|",id_start)
            key = seq[id_start:id_end]
            
            fasta_start = seq.find("\n")
            # https://www.runoob.com/python/att-string-find.html，beg=id_start+1即在该开始索引处开始往后找第1个|符号
            value = seq[fasta_start:].replace("\n","")
            # 注意fasta序列中每一行都有换行符，每一行，所以需要置换
            # 据说uniprot数据库中有些序列会出现U，可以改成V，即value = seq[fasta_start:].replace("\n","").replace('U', 'V')
            proteins_dict[key] = value
        
    return proteins_dict
    








# 预测、注释IDR的方法有很多，方法收集主要来自于benchmark、SOTA
# 此处参考https://caid.idpcentral.org/methods
# https://caid.idpcentral.org/challenge/results
# 最新的benchmark是CAID3, 目前聚焦于AIUpred/Metapredict/fiDPnn, 但是部分server不可访问, 部分code未公开

======================================================================================================================================

#  法1: 使用MobiDB-lite算法, 直接从interproscan中的MobiDB数据库调用
# 注: 但是该方法太老, 旧version基本没有使用深度学习方法, 新版本未考究

"""
用于从全蛋白质序列fasta文件中获取每一个蛋白质的IDR序列
另外相比v1和v2，在appl参考的数据库上整合了更多的蛋白质功能、结构域注释信息

"""

from Bio import SeqIO # Biopython库
import subprocess
import numpy as np
import argparse
import pandas as pd
import itertools

def run_iprscan(fasta_file_name, out_file_name, out_file_format='tsv'):
    """
    Args:
        fasta_file_name (string): 输入的fasta文件名
        out_file_name (string): 输出的结果文件名
        out_file_format (string): 输出文件格式，默认为tsv
    
    Fun:
        运行iprscan命令行工具，分析蛋白质序列中的IDR区域，然后输出
    
    """
    # 因为iprscan.py一次只能处理100条序列，所以需要分批处理
    # 每次从原始fasta文件中收集1000条序列，分成10批次处理
    # 收集所有分批结果，合并后输出

    # 首先统计一下原始fasta文件中有多少条序列
    output_str = subprocess.check_output(['grep', '-c', '>', fasta_file_name]) 
    num_sequences = int(output_str.decode("utf-8"))
    # 计算需要将这个文件分成多少批次
    num_chunks = int(np.ceil(num_sequences/1000))

    create_new_file = 1
    chunk_number = 0
    counter = 0

    # 将大的fasta文件分批成小的包含1000条序列的fasta文件
    for sequence in SeqIO.parse(fasta_file_name, 'fasta'):
        # 创建新文件，用来存储1000条序列
        if create_new_file:
            chunk_file_name = fasta_file_name + '_chunk_' + str(chunk_number) + '.fasta'
            output_handle = open(chunk_file_name, 'w')
            chunk_number += 1
            create_new_file = 0
        # 写入序列到当前的fasta文件
        SeqIO.write(sequence, output_handle, 'fasta')
        counter += 1
        # 如果当前的fasta文件已经包含1000条序列，则关闭当前文件，准备下一个文件
        if counter == 1000:
            output_handle.close()
            create_new_file = 1

    # 如果最后一个文件没有满1000条序列，则关闭当前文件
    if not create_new_file:
        output_handle.close()

    # 现在开始处理每一个小的fasta文件，对每一个小文件运行iprscan
        cat_file_list = []
    del_file_list = []
    for chunk_number in range(num_chunks):
        chunk_file_name = fasta_file_name + '_chunk_' + str(chunk_number) + '.fasta'
        chunk_IDR_file_name = fasta_file_name + '_chunk_' + str(chunk_number) + '_IDRs'
        
        cat_file_list.append(chunk_IDR_file_name + '.tsv.tsv')
        del_file_list.append(chunk_file_name)
        del_file_list.append(chunk_IDR_file_name + '.tsv.tsv')
        
        # 原始参考的数据库只有MobiDBLite、PfamA
        command = ["python", "/data2/IDR_LLM/my_DL/iprscan5.py", 
                   "--email", "luxunisgod123@gmail.com", 
                   "--stype", "p",
                   "--sequence", chunk_file_name,
                   "--appl", "MobiDBLite,PfamA,SFLD,PrositeProfiles,PrositePatterns,Coils,Panther,SignalP,Phobius",
                   "--pollFreq", "10",
                   "--outfile", chunk_IDR_file_name,
                   "--outformat", out_file_format]
        result = subprocess.run(command, capture_output=True, text=True)

        # 检查命令是否成功执行
        if result.returncode == 0:
            # 打印输出结果
            print("Output:")
            print(result.stdout)
        else:
            # 打印错误信息
            print("An error occurred:", result.stderr)

    # 将所有的结果文件合并成一个tsv文件
    command = ["cat"] + cat_file_list
    with open(out_file_name, "w") as outfile:
        subprocess.run(command, stdout=outfile)

    # 删除所有的临时的chunk文件
    command = ["rm"] + del_file_list
    result = subprocess.run(command, capture_output=True, text=True)

    # 检查命令是否成功执行
    if result.returncode == 0:
        # 打印输出结果
        print("Output:")
        print(result.stdout)
    else:
        # 打印错误信息
        print("An error occurred:", result.stderr)

class IDR:
    """
    用于处理IDR区域的类，记录起点与终点
    """
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def check_and_extend(self,idr_obj):
        # 检查两个IDR区间是否有重叠或包含关系，如果有，则将它们合并成一个更大的区间（在IDR之间更新起点与终点）
        
        # case1:当前IDR被完全包含在传入的IDR中
        if idr_obj.start <= self.start and idr_obj.end >= self.end:
            # 如果传入的IDR完全包含当前IDR，就用传入IDR的范围替换当前IDR的范围
            # 将self序列的开始和结束更改为所提供的idr的开始和结束
            # 例如：当前IDR是[50-80]，传入IDR是[30-100]，结果当前IDR变成[30-100]
            self.start = idr_obj.start
            self.end = idr_obj.end
            return True
        # case2:传入IDR的末端与当前IDR有重叠
        elif idr_obj.end > self.start  and idr_obj.end <= self.end:
            # 将当前IDR的起始位置扩展到两者中更小的起始位置
            # 适当地更新self序列的开始
            # 例如：当前IDR是[50-80]，传入IDR是[30-60]，结果当前IDR变成[30-80]
            self.start = np.min([idr_obj.start, self.start])
            return True
        # case3:传入IDR的起始端与当前IDR有重叠
        elif idr_obj.start >= self.start and idr_obj.start < self.end:
            # 将当前IDR的结束位置扩展到两者中更大的结束位置
            # 适当地更新self序列的末尾
            # 例如：当前IDR是[50-80]，传入IDR是[70-100]，结果当前IDR变成[50-100]
            self.end = np.max([idr_obj.end, self.end])
            return True
        # case4:完全没有重叠
        # 这是一个全新的IDR
        # 两个IDR完全不重叠，返回False表示无法合并
        else:
            return False

    def check_and_exclude(self, pfam_start, pfam_end):
        # 检查self序列是否为所提供的pfam域的子集，也就是和pfam域有重叠，然后更新IDR起点与终点
        # return (is_exclude, is_overlap, new_idr)，返回值的含义：
        # is_exclude: 是否需要完全移除这个IDR
        # is_overlap: 是否存在重叠
        # new_idr: 如果IDR被分割，新产生的IDR对象

        # case1:IDR完全被Pfam结构域包含
        if pfam_start <= self.start and pfam_end >= self.end:
            # 整个IDR都在Pfam结构域内，应该完全移除
            # 例如：IDR[50-80]，Pfam[30-100] → 移除整个IDR
            return (True, True, None)
        # case2:Pfam结构域完全在IDR内部
        elif pfam_start > self.start and pfam_end < self.end:
            # IDR被Pfam结构域分割成两部分
            # 例如：IDR[30-100]，Pfam[50-70] → 分割成IDR[30-50]和IDR[70-100]
            new_idr = IDR(start=pfam_end, end=self.end)
            self.end = pfam_start
            return (False, True, new_idr)
        # case3:Pfam结构域的起始位置在IDR内
        elif pfam_start >= self.start and pfam_start <= self.end:
            # 截断IDR的末端
            # 例如：IDR[30-80]，Pfam[60-100] → 截断成IDR[30-60]
            self.end = pfam_start
            return (False, True, None)
        # case4:Pfam结构域的结束位置在IDR内
        elif pfam_end >= self.start and pfam_end <= self.end:
            # 截断IDR的前端
            # 例如：IDR[50-100]，Pfam[30-70] → 截断成IDR[70-100]
            self.start = pfam_end
            return (False, True, None)
        # case5:Pfam结构域完全不在IDR内,完全没有重叠
        else:
            # 两个区域不重叠，IDR保持不变
            return (False, False, None)

def get_disordered_sequences(filename,threshold=30):
    """
    函数读取InterProScan的输出文件，通过将所有预测拼接在一起，并排除与pfam域重叠的区域，构建由MobiDBLite预测的无序域列表
    其中threshold参数用于过滤掉长度小于一定阈值个氨基酸的IDR区域，默认为30
    """
    df = pd.read_csv(filename, delimiter='\t', 
                    names=['id', 'analysis', 'start', 'end'] , 
                    header=None, usecols=[0, 3, 6, 7])
        
    idr_dict = {}
    
    for protein in df['id'].unique():
        
        idr_dict[protein] = []

        # step1:IDR区域合并（Stitching）

        # 合并逻辑：
        # 提取所有MobiDBLite预测的IDR片段
        # 使用itertools.combinations生成所有IDR对的组合
        # 通过check_and_extend方法合并重叠或相邻的IDR
        # 重复直到没有更多可合并的IDR

        # 在文件中选择与无序序列预测相对应的行索引
        selected_idr_row_indices = df.index[(df['id']==protein)*(df['analysis']=='MobiDBLite')]

        # 检查这种蛋白质是否至少有一个无序预测
        if len(selected_idr_row_indices) > 0: 
            
            idr_start_end_list = []
            # 初始化一个存储IDR类对象的列表，该列表包含该蛋白质的IDR的起始和结束位置
            # 存储原始的IDR起始和结束位置列表
            for index in selected_idr_row_indices:
                current_idr_obj = IDR(start=df.iloc[index]['start'], end=df.iloc[index]['end'])
                idr_start_end_list.append(current_idr_obj)
            # 将idr_start_end_list中的所有idr拼接在一起
            stitching_flag = True
            while stitching_flag:
                # 如果只有一个idr，就没必要拼接
                if len(idr_start_end_list) == 1:
                    break
                # 如果有很多idr，比较idr_start_end_list中的每一对，检查是否有重叠或子集
                # combinations函数就是生成任意两个元素的组合
                for idr1, idr2 in itertools.combinations(idr_start_end_list, r=2):
                    stitching_flag = idr1.check_and_extend(idr2)
                    # 如果有重叠或子集，删除当前idr_object[j]，然后重新开始循环
                    if stitching_flag:
                        idr_start_end_list.remove(idr2)
                        break 

            # step2:排除与Pfam结构域重叠的IDR（Excluding）
            # 排除逻辑：

            # 对每个Pfam结构域，检查与IDR的重叠
            # 使用check_and_exclude方法处理重叠：
            # 完全包含：移除整个IDR
            # 部分重叠：截断IDR
            # 在IDR中间：分割成两个IDR
            
            # 选择与pfam域预测相对应的行索引
            selected_pfam_row_indices = df.index[(df['id']==protein)*(df['analysis']!='MobiDBLite')]
            # 排除与pfam域重叠的idr
            if len(selected_pfam_row_indices) > 0:             
                for index in selected_pfam_row_indices:
                    for idr in idr_start_end_list:
                        is_exclude, is_overlap, new_idr = idr.check_and_exclude(pfam_start=df.iloc[index]['start'],
                                                                                    pfam_end=df.iloc[index]['end'])
                        if is_exclude:
                            print("IDR removed because it is contained in a Pfam domain:", str(protein), 
                                    '[', str(idr.start), '-', str(idr.end), ']')
                            idr_start_end_list.remove(idr)
                            break
                        else:
                            if new_idr is not None:
                                idr_start_end_list.append(new_idr)
                                print("IDR broken into two because of overlap with Pfam domain:", 
                                        str(protein), '[', str(idr.start), '-', str(idr.end), ']',
                                        '[', str(new_idr.start), '-', str(new_idr.end), ']')
                            if is_overlap:
                                break
                                
            print('IDR list after stitching and excluding:')
            print(str(protein))
            print([str(idr.start) + '-' + str(idr.end) for idr in idr_start_end_list])

            # step3:长度过滤，排除长度小于一定阈值氨基酸的idr
            # 经过深思熟虑，我们决定将长度小于30个氨基酸的IDR排除，默认为30
            # 因为查阅一些文献，本来范围是定在20、25、30作为阈值等，但是考虑到后面长度要足够才能够在里面找block
            # 太短的区域通常不具有生物学意义
            for idr in idr_start_end_list:
                if idr.end - idr.start < threshold:
                    print("IDR removed because of small size:", str(protein), 
                         '[', str(idr.start), '-', str(idr.end), ']')
                else:
                    idr_dict[protein].append(idr) 
            print('------------------------------------------')

            # step4:排序
            
            # 排序逻辑：
            # 按照IDR在蛋白质序列中的起始位置进行排序
            # 确保IDR按照在序列中的出现顺序排列
            if len(idr_dict[protein]) > 0:
                sorted_indices = np.argsort([idr.start for idr in idr_dict[protein]])
                idr_dict[protein] = [idr_dict[protein][i] for i in sorted_indices]
    
    return idr_dict

def store_idr_data(idr_dict, fasta_file_name, out_file_name):
    """
    将处理后的IDR位置信息转换为实际的氨基酸序列，并保存到CSV文件中，最终输出的有蛋白质id、IDR起始位置、IDR结束位置和IDR序列，以及非IDR序列的位置和序列
    Args:
        idr_dict (dict): 包含蛋白质IDR信息的字典，键为蛋白质ID，值为IDR对象列表
        fasta_file_name (string): 输入的蛋白质全序列的fasta文件名
        out_file_name (string): 输出的包含IDR序列信息的CSV文件名
    """

    data = []
    
    for sequence in SeqIO.parse(fasta_file_name, 'fasta'):
        if sequence.id in idr_dict.keys():
            idr_sequence_list = []
            idr_positions_list = []  # 新增：存储IDR位置信息
            non_idr_sequence_list = []
            full_sequence = str(sequence.seq)
            
            # 收集所有IDR区域的坐标
            idr_regions = []
            for idr in idr_dict[sequence.id]:
                # 通过交叉引用idr_dict中的起始和结束位置和protein_sequences中的序列来提取IDR序列
                # Note: InterProScan uses 1-based indexing, Python uses 0-based indexing
                # 序列索引错误: InterProScan使用1-based索引，Python使用0-based索引，需要调整
                idr_sequence = str(sequence.seq[idr.start-1:idr.end])
                if idr_sequence != "":
                    idr_sequence_list.append(idr_sequence)
                    idr_positions_list.append(f"{idr.start}-{idr.end}")  # 新增：添加位置信息
                    # 记录IDR区域的坐标（转换为0-based索引）
                    idr_regions.append((idr.start-1, idr.end))
            
            # 收集非IDR区域的序列
            if len(idr_regions) > 0:
                # 按起始位置排序IDR区域
                idr_regions.sort(key=lambda x: x[0])
                
                # 提取非IDR区域
                current_pos = 0
                for start, end in idr_regions:
                    # 提取当前位置到IDR开始位置之间的序列
                    if current_pos < start:
                        non_idr_seq = full_sequence[current_pos:start]
                        if non_idr_seq != "":
                            non_idr_sequence_list.append(non_idr_seq)
                    # 更新当前位置到IDR结束位置
                    current_pos = end
                
                # 提取最后一个IDR之后的序列
                if current_pos < len(full_sequence):
                    non_idr_seq = full_sequence[current_pos:]
                    if non_idr_seq != "":
                        non_idr_sequence_list.append(non_idr_seq)
            else:
                # 如果没有IDR区域，整个序列都是非IDR
                non_idr_sequence_list.append(full_sequence)
            
            data.append([sequence.id.split('|')[1], sequence.id.split('|')[2], 
                        idr_sequence_list, idr_positions_list, non_idr_sequence_list])  # 修改：添加位置列表
        else:
            # 如果蛋白质不在idr_dict中，说明没有预测到IDR，整个序列都是非IDR
            full_sequence = str(sequence.seq)
            data.append([sequence.id.split('|')[1], sequence.id.split('|')[2], 
                        [], [], [full_sequence]])  # 修改：添加空的位置列表
    
    # 创建csv文件并存储数据
    df = pd.DataFrame(data, columns=['Uniprot ID', 'Protein ID', 'IDR Sequence List', 'IDR Positions List', 'Non-IDR Sequence List'])  # 修改：添加新列
    # 保存为csv文件       
    df.to_csv(out_file_name, index=False)  
    # df.to_csv(out_file_name, index=False, sep="\t") 可以保存为tsv文件

if __name__ == "__main__":
    # 读入命令行参数
    parser = argparse.ArgumentParser(description='Run script to extract IDRs and non-IDRs from full protein sequences in a fasta file')
    parser.add_argument('--i', help="Name of input fasta file containing full protein sequences", required=True)
    parser.add_argument('--r', help="Name of output file contaning IDR results from iprscan", required=True)
    parser.add_argument('--o', help="Name of output file contaning Uniprot IDs and IDR sequence lists, and non-IDR", required=True)
    parser.add_argument('--t', help="Threshold for minimum length of IDR sequences, default is 30", type=int, default=30)
    args = parser.parse_args()
    
    fasta_file_name = args.i
    iprscan_file_name = args.r
    output_file_name = args.o
    idr_threshold = args.t
    
    run_iprscan(fasta_file_name=fasta_file_name, out_file_name=iprscan_file_name)
    idr_dict = get_disordered_sequences(filename=iprscan_file_name, threshold=idr_threshold)
    store_idr_data(idr_dict=idr_dict, fasta_file_name=fasta_file_name, out_file_name=output_file_name)


======================================================================================================================================

# 法2: 使用AIUpred
# 参考仓库https://github.com/doszilab/AIUPred
# 下载解压之后

AIUpred_PATH = "/home/nicai_zht/software/AIUPred-2.1.2"
if AIUpred_PATH not in sys.path:
    sys.path.append(AIUpred_PATH)
import aiupred_lib

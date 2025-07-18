"""
Script to extract the IDRs from full protein sequences in a fasta file and generate
a dataset containing proteins, IDRs, and their partition ratios
"""

from Bio import SeqIO
import subprocess
import numpy as np
import argparse
import pandas as pd
import itertools

def run_iprscan(fasta_file_name, out_file_name, out_file_format='tsv'):
    """
    Function that takes in a fasta file and runs iprscan.py to identify IDRs on the sequence
    
    Inputs:
    -------
    fasta_file_name (string): Name of the fasta file containing full protein sequences
    out_file_format (string): Currently supports only 'tsv' 
    out_file_name (string): Name of output file that stores start and end positions of IDRs
    
    Outputs:
    --------
    None: This functions just runs iprscan.py writes out results to out_file_name
    """
    
    # Since iprscan.py takes only 100 sequences at a time, we collect only
    # 1000 sequences a time from a large fasta file and run iprscan.py
    # We_finally collate all the results and write it out in out_file_name
    
    assert fasta_file_name.split('.')[-1] == 'fasta', "Please supply a valid fasta file"
    
    # Count the number of sequences in the fasta file quickly using grep
    output_str = subprocess.check_output(['grep', '-c', '>', fasta_file_name]) 
    num_sequences = int(output_str.decode("utf-8"))
    # Count the number of files you need to chunk this large file into
    num_chunks = int(np.ceil(num_sequences/1000))
    
    create_new_file = 1
    chunk_number = 0
    counter = 0
    # Chunk the large fasta file into smaller files containing 1000 sequences
    for sequence in SeqIO.parse(fasta_file_name, 'fasta'):
        # Create new file to store 1000 sequences
        if create_new_file:
            chunk_file_name = fasta_file_name + '_chunk_' + str(chunk_number) + '.fasta'
            output_handle = open(chunk_file_name, 'w')
            chunk_number += 1
            create_new_file = 0
        # Write sequences to this file
        SeqIO.write(sequence, output_handle, 'fasta')
        counter += 1
        # Check if 1000 sequences have been reached. If yes, then create a new file.
        if counter % 1000 == 0: 
            output_handle.close()
            create_new_file = 1
    
    # Close the last file handle if it's still open
    if not create_new_file:
        output_handle.close()
    
    # Run iprscan for each of these files
    cat_file_list = []
    del_file_list = []
    for chunk_number in range(num_chunks):
        chunk_file_name = fasta_file_name + '_chunk_' + str(chunk_number) + '.fasta'
        chunk_IDR_file_name = fasta_file_name + '_chunk_' + str(chunk_number) + '_IDRs'
        
        cat_file_list.append(chunk_IDR_file_name + '.tsv.tsv')
        del_file_list.append(chunk_file_name)
        del_file_list.append(chunk_IDR_file_name + '.tsv.tsv')
        
        command = ["python", "/data2/IDR_LLM/my_DL/iprscan5.py", 
                   "--email", "luxunisgod123@gmail.com", 
                   "--stype", "p",
                   "--sequence", chunk_file_name,
                   "--appl", "MobiDBLite,PfamA",
                   "--pollFreq", "10",
                   "--outfile", chunk_IDR_file_name,
                   "--outformat", out_file_format]
        result = subprocess.run(command, capture_output=True, text=True)
        # Check if the command was successful
        if result.returncode == 0:
            # Print the output
            print("Output:")
            print(result.stdout)
        else:
            # Print an error message
            print("An error occurred:", result.stderr)
    
    # Append all the results into a single tsv file
    command = ["cat"] + cat_file_list
    with open(out_file_name, "w") as outfile:
        subprocess.run(command, stdout=outfile)
    
    # Delete all the chunk files that were created
    command = ["rm"] + del_file_list
    result = subprocess.run(command, capture_output=True, text=True)
    # Check if the command was successful
    if result.returncode == 0:
        # Print the output
        print("Output:")
        print(result.stdout)
    else:
        # Print an error message
        print("An error occurred:", result.stderr)


class IDR:
    """Class for keeping track of starting and ending positions of an IDR"""
    
    def __init__(self, start, end):
        self.start = start
        self.end = end
        
    def check_and_extend(self, idr_obj):
        # Check if the self sequence is a subset of the supplied idr
        if idr_obj.start <= self.start and idr_obj.end >= self.end:
            # If yes, change the start and end of the self sequence to the start and end of the supplied idr
            self.start = idr_obj.start
            self.end = idr_obj.end
            return True
        # Check if the end of the supplied sequence lies between the start and end of the self sequence
        elif idr_obj.end > self.start  and idr_obj.end <= self.end:
            # If yes, then there is overlap. Update the start of self sequence appropriately
            self.start = np.min([idr_obj.start, self.start])
            return True
        # Check if the start of the supplied sequence lies between the start and end of the self sequence
        elif idr_obj.start >= self.start and idr_obj.start < self.end:
            # If yes, then there is overlap. Update the end of self sequence appropriately
            self.end = np.max([idr_obj.end, self.end])
            return True
        # This is a completely new IDR
        else:
            return False
        
    def check_and_exclude(self, pfam_start, pfam_end):
        # Check if the self sequence is a subset of the supplied pfam domain
        if pfam_start <= self.start and pfam_end >= self.end:
            # If yes, remove this domain
            return (True, True, None)
        # Check if the the pfam domain is contained within the self sequence
        elif pfam_start > self.start and pfam_end < self.end:
            # If yes, then there is overlap. Update the start of self sequence appropriately
            new_idr = IDR(start=pfam_end, end=self.end)
            self.end = pfam_start
            return (False, True, new_idr)
        # Check if the start of the pfam domain lies between the start and end of the self sequence
        elif pfam_start >= self.start and pfam_start <= self.end:
            # If yes, then there is overlap. Update the end of self sequence appropriately
            self.end = pfam_start
            return (False, True, None)
        # Check if the end of the pfam domain lies between the start and end of the self sequence
        elif pfam_end >= self.start and pfam_end <= self.end:
            self.start = pfam_end
            return (False, True, None)
        # No overlap of the pfam domain with the idr whatsoever
        else:
            return (False, False, None)
        

def get_disordered_sequences(filename):
    """Function to read a file that is the output of InterProScan, construct a list of disordered domains predicted 
    by MobiDBLite by stitching together all predictions, and excluding regions that overlap with pfam domains """
    
    df = pd.read_csv(filename, delimiter='\t', 
                     names=['id', 'analysis', 'start', 'end'] , 
                     header=None, usecols=[0, 3, 6, 7])
    
    idr_dict = {}
    
    for protein in df['id'].unique():
        
        idr_dict[protein] = []
        
        # Step 1: Get the start and end positions of the unique IDRs by stitching together sequences
        # ------------------------------------------------------------------------------------------
        
        # Select the row indices in the file that correspond to predictions of disordered sequences
        selected_idr_row_indices = df.index[(df['id']==protein)*(df['analysis']=='MobiDBLite')]
        
        # Check if this protein has at least one disorder prediction
        if len(selected_idr_row_indices) > 0: 
            
            idr_start_end_list = []
            # Initialize a list that will store the IDR class objects containing
            # start and end positions of IDRs of this protein
            
            # Store the raw list of idr start and end positions
            for index in selected_idr_row_indices:
                current_idr_obj = IDR(start=df.iloc[index]['start'], end=df.iloc[index]['end'])
                idr_start_end_list.append(current_idr_obj)
            
            # Stich together all the IDRs in the idr_start_end_list
            stitching_flag = True
            while stitching_flag:
                # Nothing to stitch if there is only one idr
                if len(idr_start_end_list) == 1:
                    break
                # If there are many idrs, compare every pair in the idr_start_end_list and check for overlap or subset
                # combinations函数就是生成任意两个元素的组合
                for idr1, idr2 in itertools.combinations(idr_start_end_list, r=2):
                    stitching_flag = idr1.check_and_extend(idr2)
                    # If there is overlap or subset, remove the current idr_object[j], then restart the loops again
                    if stitching_flag:
                        idr_start_end_list.remove(idr2)
                        break 
            
            # Step 2: Exclude the idrs that overlap with pfam domains
            # -------------------------------------------------------
            
            # Select row indices that corresponds to pfam domain predictions
            selected_pfam_row_indices = df.index[(df['id']==protein)*(df['analysis']!='MobiDBLite')]
            
            # Exclude the idrs that overlap with pfam domains
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
            
            # Step 3: Exclude IDRs that are smaller than 20 amino acids long
            # The final list of start-end positions of IDRs
            for idr in idr_start_end_list:
                if idr.end - idr.start < 20:
                    print("IDR removed because of small size:", str(protein), 
                         '[', str(idr.start), '-', str(idr.end), ']')
                else:
                    idr_dict[protein].append(idr) 
            print('------------------------------------------')
            
            # Step 4: Sort the entries in idr_dict to arrange IDRs in the order of their 
            # occurrence along the sequence            
            if len(idr_dict[protein]) > 0:
                sorted_indices = np.argsort([idr.start for idr in idr_dict[protein]])
                idr_dict[protein] = [idr_dict[protein][i] for i in sorted_indices]
    
    return idr_dict


def store_idr_data(idr_dict, fasta_file_name, out_file_name):
    """Function that creates a csv file that contains information about protein IDs and
    their IDRs as a list, and also non-IDR sequences(这个是新增的功能)
    
    Inputs:
    -------
    idr_dict (dict): Dictionary that contains Uniprot IDs and IDR sequences of proteins
    fasta_file_name (string): Name of the fasta file that contains full sequences of proteins
    out_file_name (string): Name of the output csv file to store the list of IDR sequences
    
    Outputs:
    --------
    None
    """
    
    data = []
    
    for sequence in SeqIO.parse(fasta_file_name, 'fasta'):
        if sequence.id in idr_dict.keys():
            idr_sequence_list = []
            non_idr_sequence_list = []
            full_sequence = str(sequence.seq)
            
            # 收集所有IDR区域的坐标
            idr_regions = []
            for idr in idr_dict[sequence.id]:
                # Extract the IDR sequence by crossreferencing start and end positions in idr_dict with the 
                # sequences in protein_sequences
                # Note: InterProScan uses 1-based indexing, Python uses 0-based indexing
                # 序列索引错误: InterProScan使用1-based索引，Python使用0-based索引，需要调整
                idr_sequence = str(sequence.seq[idr.start-1:idr.end])
                if idr_sequence != "":
                    idr_sequence_list.append(idr_sequence)
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
                        idr_sequence_list, non_idr_sequence_list])
        else:
            # 如果蛋白质不在idr_dict中，说明没有预测到IDR，整个序列都是非IDR
            full_sequence = str(sequence.seq)
            data.append([sequence.id.split('|')[1], sequence.id.split('|')[2], 
                        [], [full_sequence]])
    
    # Create a csv file and store the data
    df = pd.DataFrame(data, columns=['Uniprot ID', 'Protein ID', 'IDR Sequence List', 'Non-IDR Sequence List'])
    # 还是保存为tsv文件好
    # df.to_csv(out_file_name, index=False, sep="\t") 
    # 依旧保存为csv文件       
    df.to_csv(out_file_name, index=False)          
    

if __name__ == "__main__":
    """ Run script to extract IDRs from full protein sequences in a fasta file and generate
    a dataset of proteins, their IDRs, and the corresponding partition ratios
    """    
    # Read command line arguments 
    parser = argparse.ArgumentParser(description='Run script to extract IDRs from full protein sequences in a fasta file')
    parser.add_argument('--i', help="Name of input fasta file containing full sequences", required=True)
    parser.add_argument('--r', help="Name of output file contaning IDR results from iprscan", required=True)
    parser.add_argument('--o', help="Name of output file contaning Uniprot IDs and IDR sequence lists", required=True)
    args = parser.parse_args()
    
    fasta_file_name = args.i
    iprscan_file_name = args.r
    output_file_name = args.o
    
    run_iprscan(fasta_file_name=fasta_file_name, out_file_name=iprscan_file_name)
    idr_dict = get_disordered_sequences(filename=iprscan_file_name)
    store_idr_data(idr_dict=idr_dict, fasta_file_name=fasta_file_name, out_file_name=output_file_name)
    
    
    
    
    
    

    
# 中性块可以是剩余
# 该方法暂时有问题，不能够避免overlap，可以定义某些feature作为主导dominate
# 或者不要在window level上merge，要在residue level上merge，再决策，也就是voting

def identify_charge_blocks(sequence,window_size=10,NCPR_threshold=5):
        """
        Args:
            window_size (int): 用于计算NCPR的滑动窗口大小，默认为10
            NCPR_threshold (float): 用于定义块的NCPR阈值，默认为0.5
        
        Fun:识别protein序列中charge feature的块
        """

        # 保存记录找到的块的（起点，终点）对
        Acidic_block = []
        Basic_block = []
        Neutral_block = []
        for i in range(len(sequence) - window_size + 1):
            window_seq = sequence[i:i+window_size]
            net_charge = sum([aa_charge.get(aa,0) for aa in window_seq])
            NCPR = net_charge / window_size
            
            if abs(net_charge) >= NCPR_threshold:
                if net_charge > 0:
                    # 保留窗口起点、终点、窗口大小、窗口内序列、窗口内序列净电荷值、NCPR值、窗口类型
                    # 此处起点、终点为0-based index，后续合并时再统一转换为1-based index
                    Basic_block.append((i,i+window_size-1,window_size,window_seq,net_charge,NCPR,"Basic"))

                else:
                    Acidic_block.append((i,i+window_size-1,window_size,window_seq,net_charge,NCPR,"Acidic"))
            
        # 合并重叠或相邻的电荷块
        # 用于保存已经合并的窗口起点终点元组对，注意每次比较都是比较当前元组的起点和上一个已经合并的元组（也就是merged_results[-1])的终点
        Acidic_block_merged = []
        Basic_block_merged = []

        # 先合并酸性带负电窗口
        if Acidic_block:
            # 非空则取第1个窗口，作为后续比对起点
            Acidic_block_merged.append(Acidic_block[0])
            for current_acidic in Acidic_block[1:]:
                last_acidic = Acidic_block_merged[-1]

                # 若当前窗口起点≤上一个已合并窗口终点+1，则说明两窗口重叠或相邻
                if current_acidic[0] <= last_acidic[1] + 1:
                    merged_start = last_acidic[0]
                    merged_end = max(last_acidic[1],current_acidic[1])
                    merged_seq = sequence[merged_start:merged_end + 1]
                    merged_net_charge = sum([aa_charge.get(aa,0) for aa in merged_seq])
                    merged_size = len(merged_seq)
                    if merged_size > 0:
                        merged_NCPR = merged_net_charge / merged_size
                    else:
                        merged_NCPR = 0

                    # 更新上1个已合并窗口的起点、终点、窗口大小、窗口内序列、窗口内序列净电荷值、NCPR值
                    # ⚠️注意此处将0-based index转换为1-based index，即真实物理坐标
                    Acidic_block_merged[-1] = (merged_start + 1,merged_end + 1,merged_size,merged_seq,merged_net_charge,merged_NCPR,"Acidic")
                
                # 否则没有重叠，直接添加当前窗口
                else:
                    # ⚠️注意此处将0-based index转换为1-based index，即真实物理坐标
                    Acidic_block_merged.append((current_acidic[0] + 1,current_acidic[1] + 1,current_acidic[2],current_acidic[3],current_acidic[4],current_acidic[5],current_acidic[6]))
        
        # 再合并碱性带正电窗口
        if Basic_block:
            # 非空则取第1个窗口，作为后续比对起点
            Basic_block_merged.append(Basic_block[0])
            for current_basic in Basic_block[1:]:
                last_basic = Basic_block_merged[-1]

                # 若当前窗口起点≤上一个已合并窗口终点+1，则说明两窗口重叠或相邻
                if current_basic[0] <= last_basic[1] + 1:
                    merged_start = last_basic[0]
                    merged_end = max(last_basic[1],current_basic[1])
                    merged_seq = sequence[merged_start:merged_end + 1]
                    merged_net_charge = sum([aa_charge.get(aa,0) for aa in merged_seq])
                    merged_size = len(merged_seq)
                    if merged_size > 0:
                        merged_NCPR = merged_net_charge / merged_size
                    else:
                        merged_NCPR = 0

                    # 更新上1个已合并窗口的起点、终点、窗口大小、窗口内序列、窗口内序列净电荷值、NCPR值
                    # ⚠️注意此处将0-based index转换为1-based index，即真实物理坐标
                    Basic_block_merged[-1] = (merged_start + 1,merged_end + 1,merged_size,merged_seq,merged_net_charge,merged_NCPR,"Basic")
                
                # 否则没有重叠，直接添加当前窗口
                else:
                    # ⚠️注意此处将0-based index转换为1-based index，即真实物理坐标
                    Basic_block_merged.append((current_basic[0] + 1,current_basic[1] + 1,current_basic[2],current_basic[3],current_basic[4],current_basic[5],current_basic[6])) 

        # 返回合并后的块列表,以字典返回
        return {
            "Acidic_blocks":Acidic_block_merged,
            "Basic_blocks":Basic_block_merged
        }    

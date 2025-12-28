# 序列重排、置换
# 新序列包含的元素与原序列完全一致（无新增、无缺失）；
# 元素的排列顺序与原序列不同（若原序列元素无重复，所有排列均为 “全新顺序”；若有重复，需排除与原顺序完全一致的情况，具体依场景而定）

# 1, 置换分析, 仅仅只是shuffle序列扰乱序列

import numpy as np
from typing import List, Optional
def generate_shuffled_sequences(sequence: str, n: int, seed: Optional[int] = None) -> List[str]:
        """
        Description
        ----------
            生成随机打乱的序列 (Permutation/Shuffle), 保持氨基酸组成不变, 用于置换检验Permutation Test.

        Args
        ----------
            sequence (str): 原始蛋白质序列
            n (int): 生成随机序列的数量
            seed (int, optional): 随机种子, 默认为None
        """
        rng = np.random.default_rng(seed)
        seq_list = list(sequence)
        shuffled_seqs = []
        for _ in range(n):
            perm = rng.permutation(seq_list)
            shuffled_seqs.append("".join(perm))
        return shuffled_seqs

===================================================================================================================

# 2, 对置换之后的序列所构建的零分布, 对真实观测值的构建的某一个统计量值所做的假设检验
# 也就是置换检验
def _perform_permutation_test(self, 
                                  sequence: str, 
                                  observed_blocks: List[Charge_Block],
                                  args_dict: Dict) -> Tuple[List[Charge_Block], float]:
        """
        Description
        ----------
            执行置换检验 (Permutation Test) 以评估电荷块的显著性

        Args
        ----
            sequence (str): 原始蛋白序列
            observed_blocks (List[Charge_Block]): 当前观测到的电荷块分布
            args_dict: 初始化中的charge_params参数
        """
        n_perm = args_dict['n_permutations']
        seed = args_dict['random_seed']
        
        # 生成随机序列
        shuffled_seqs = self._generate_shuffled_sequences(sequence, n_perm, seed)
        
        # 用于构建零假设分布的NCPR池(置换序列中所有块的NCPR值，不进行长度分层与阈值分配策略)
        pooled_acidic_ncpr: List[float] = []
        pooled_basic_ncpr: List[float] = []

        # 对于零分布, 只使用 'filter' 策略 (仅长度过滤)
        # 因为初次比较是长度过滤之后的观测序列块 vs 置换序列块, 所以零分布的块也只做长度过滤
        for seq in shuffled_seqs:
            # 递归调用, 关闭显著性计算以避免无限循环
            # 没有显著性计算就不需要permutation test和bootstrap test
            # 每一条序列获取1个原始的电荷块结果

            # 注意, 置换检验的null分布只做长度过滤, 因为本身就是为了评估观测块的显著性
            # 得到的置信度是用来第2次过滤
            # ⚠️ PTM修饰位点信息不保留, 因为计算映射比较复杂
            res = self.get_charge_blocks(
                sequence=seq,
                savgol_filter_window_length=args_dict['savgol_filter_window_length'],
                savgol_filter_polyorder=args_dict['savgol_filter_polyorder'],
                min_block_length=args_dict['min_block_length'],
                min_linker_length=args_dict['min_linker_length'],
                refine_strategy='filter', # 置换检验的null分布只做长度过滤
                phos_sites=None, # 随机序列不保留磷酸化位点位置信息
                compute_significance=False
            )      

            # 对于每一条序列的电荷块结果res
            # 收集所有块的NCPR值到pool中
            a_list = [ b.NCPR for b in res.Acidic_blocks]
            b_list = [ b.NCPR for b in res.Basic_blocks]

            # 每一个序列的所有块的NCPR值都加入pool
            pooled_acidic_ncpr.extend(a_list)
            pooled_basic_ncpr.extend(b_list)
        
        # 计算观测块的p值(基于统一的零假设分布, 也就是使用统一的pooled NCPR池)
        updated_blocks: List[Charge_Block] = []
        pvals: List[float] = []
        for b in observed_blocks:
            obs = b.NCPR
            p_val = 1.0
            
            # 依据观测序列块的类型, 选取对应的零假设分布池计算p值
            if b.block_type == "Acidic":
                # Acidic: NCRR值越负越极端
                if pooled_acidic_ncpr:
                    count = sum(1 for ncpr in pooled_acidic_ncpr if ncpr <= obs)
                    p_val = (count + 1) / (len(pooled_acidic_ncpr) + 1)
                else:
                    p_val = 1.0
            elif b.block_type == "Basic":
                # Basic: NCRR值越正越极端
                if pooled_basic_ncpr:
                    count = sum(1 for ncpr in pooled_basic_ncpr if ncpr >= obs)
                    p_val = (count + 1) / (len(pooled_basic_ncpr) + 1)
                else:
                    p_val = 1.0
            
            pvals.append(p_val)
            
            # 更新块信息
            # 原本的p_value为None -> 更新为p_val
            # 原本的每一个电荷块数据b -> 更新为 new_b
            new_b = replace(b, p_value=p_val)
            # 原本的电荷块分布数据observed_blocks -> updated_blocks
            updated_blocks.append(new_b)
        
        # 现在计算多重检验校正后的q值 (Benjamini-Hochberg FDR)
        # Benjamini-Hochberg FDR correction across all blocks
        # 参考: https://www.geeksforgeeks.org/data-science/benjamini-hochberg-procedure/
            
        _, p_adjusts, _, _ = multipletests(pvals, method="fdr_bh", is_sorted=False)

        # 增加矫正之后的p值(fdr/q_values)到结果中
        for i, block in enumerate(updated_blocks):
            updated_blocks[i] = replace(block, p_adjust=p_adjusts[i])

        # 返回最终的置换检验的结果
        return updated_blocks



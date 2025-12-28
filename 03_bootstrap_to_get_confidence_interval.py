# 01
def bootstrap_X(n_bootstrap,sample:list[float],real_param_to_test,alpha=0.05):
    """
    Args：
    n_bootstrap: 自助法抽样的次数
    sample: 需要进行自助法抽样的样本数据，此处设置为list浮点数列表类型
    real_param_to_test: 需要进行置信区间估计（显著性判断）的真实数据，比如说某个待检测样本的均值、方差、中位数、相关系数等
    alpha: 置信水平，默认值为0.05，注意此处使用百分位数对应置信水平（概率）
    参考https://en.wikipedia.org/wiki/Bootstrapping_(statistics)，只有当自助分布是对称的，并且以观察到的统计量为中心时，使用百分位数构造置信区间才合适，所以该算法运用前需要先对自助参数的分布形状做一个评估，即是否分布对称！！！！！！！！！！
    
    Fun：
    1,进行自助法抽样，返回一个包含n_bootstrap次抽样结果的列表；
    每次抽样计算一个统计量，此处命名为X，可以是均值、方差、中位数、相关系数等，比如说bootstrap_of_median函数，此处统一使用“X”指代
    2,利用百分位数法构造该参数X（比如说上面X指代median中位数）的置信区间;
    比如说置信水平为α，则使用(1-α/2)的百分位数和α/2的百分位数来构造置信区间，即α*100/2%和(1-α/2)*100%分位数；
    3,返回置信区间的下限和上限；
    4,顺便我们可以判断一下某个真实样本的参数数据（真实数据）是否在置信区间内，即是否显著，即real_param_to_test是否在置信区间内；
    
    """
    import numpy as np
    from scipy import stats
    
    #  注意下面都是以X代称需要估计的统计量
    bootstrap_X = [] # 存储自助法抽样获取的多次抽样结果，比如说是bootstrap_median
    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(sample,size=len(sample),replace=True)
        bootstrap_X.append(np.X(bootstrap_sample))  # np.X指代任何python实现中能够计算指标X的api函数，比如说均值用np.mean，中位数用np.median等

    ci_lower = np.percentile(bootstrap_X, alpha * 100 / 2)  # 置信区间下限
    ci_upper = np.percentile(bootstrap_X, (1 - alpha / 2) * 100)  # 置信区间上限
    
    # 下面是判断真实数据是否在置信区间内,即是否显著 
    is_significant = not (ci_lower <= real_param_to_test <= ci_upper)
    # 或者我们也可以使用
    # result = False if ci_lower <= observation_param <= ci_upper else True
    
    return ci_lower, ci_upper, is_significant
    # 也可以return result


##############################################################################################################################################################################
# 02
# ！！！比如说以自助法估计中位数统计量为例，构造相应百分位数的置信区间，用于判断抽取的某个样本中的中位数是否在置信区间内，以推断该样本是否来自于同一个抽样总体，即是否差异显著，见下面

def boostrap_median(n_bootstrap,smaple:list[float],real_param_to_test,alpha=0.05):
    import numpy as np
    import scipy as stats
    bootstrap_median=[]
    for i in range(n_bootstrap):
        bootstrap_sample = np.random.choice(sample,size=len(sample),replace=True)
        bootstrap_median.append(np.median(bootstrap_sample))
    ci_lower = np.percentile(bootstrap_median,alpha*100/2)
    ci_upper = np.percentile(bootstrap_median,(1-alpha/2)*100)

    is_significant = not(ci_lower <= real_param_to_test <= ci_upper)
    return ci_lower,ci_upper,is_significant



======================================================================================

# 3, 电荷块分析中使用自助法以及自助法评估边界稳定性的置信区间, 详情可以参考idr_pattern项目 charge_analyzer
@staticmethod
    def _generate_bootstrap_sequences(sequence: str, n: int, block_size: int = 5, seed: Optional[int] = None) -> List[Tuple[str, List[int]]]:
        """
        Description
        ----------
            生成Bootstrap重采样序列 (Block Bootstrap), 保持局部序列结构.
            返回: (new_sequence, original_indices)

        Args
        ----------
            sequence (str): 原始蛋白质序列
            n (int): 生成Bootstrap序列的数量
            block_size (int): 每个块的大小, 默认为5
            seed (int, optional): 随机种子, 默认为None
        """
        rng = np.random.default_rng(seed)
        L = len(sequence)
        if L == 0:
            return [("", [])] * n
            
        eff_block_size = min(block_size, L)
        bootstrap_data = []
        
        for _ in range(n):
            new_seq_parts = []
            indices = []
            current_len = 0
            while current_len < L:
                # 随机选择起始位置
                start_idx = rng.integers(0, L - eff_block_size + 1)
                end_idx = start_idx + eff_block_size
                
                block_seq = sequence[start_idx : end_idx]
                block_indices = list(range(start_idx, end_idx))
                
                new_seq_parts.append(block_seq)
                indices.extend(block_indices)
                current_len += len(block_seq)
            
            # 截断至原始长度
            full_seq = "".join(new_seq_parts)[:L]
            full_indices = indices[:L]
            bootstrap_data.append((full_seq, full_indices))
        return bootstrap_data

    def _perform_bootstrap_analysis(self, 
                                    sequence: str, 
                                    observed_blocks: List[Charge_Block],
                                    args_dict: Dict) -> Tuple[List[Charge_Block], List[Dict[str, float]]]:
        """
        Description
        ----------
            执行Bootstrap分析以评估残基分配置信度和电荷块稳定性
        """
        n_boot = args_dict['n_bootstraps']
        seed = args_dict['random_seed']
        L = len(sequence)
        
        # 生成Bootstrap序列及其映射
        # 允许通过 args_dict 指定 boot_block_size；若未指定，默认使用 savgol 窗口长度以保留相近尺度的局部结构
        bs = args_dict.get('boot_block_size') if args_dict is not None else None
        if bs is None:
            # 回退到 smoothing 窗口长度，确保至少为1且不超过序列长度
            bs = max(1, min(L, args_dict.get('savgol_filter_window_length', 5)))
        else:
            bs = max(1, min(L, int(bs)))

        bootstrap_data = self._generate_bootstrap_sequences(sequence, n_boot, block_size=bs, seed=seed)
        
        # 初始化投票计数器 [Acidic, Basic, Neutral]
        vote_counts = np.zeros((L, 3), dtype=int) 
        type_map = {"Acidic": 0, "Basic": 1, "Neutral": 2}
        
        for seq, mapping in bootstrap_data:
            # 递归调用, 关闭显著性计算
            res = self.get_charge_blocks(
                sequence=seq,
                savgol_filter_window_length=args_dict['savgol_filter_window_length'],
                savgol_filter_polyorder=args_dict['savgol_filter_polyorder'],
                min_block_length=args_dict['min_block_length'],
                min_linker_length=args_dict['min_linker_length'],
                refine_strategy=args_dict['refine_strategy'],
                phos_sites=None,
                compute_significance=False
            )
            
            # 统计每个残基的分类
            for b in res.Charge_blocks:
                b_type_idx = type_map.get(b.block_type, 2)
                # 映射回原始索引
                for i in range(b.start - 1, b.end):
                    if i < len(mapping):
                        orig_idx = mapping[i]
                        if 0 <= orig_idx < L:
                            vote_counts[orig_idx, b_type_idx] += 1
                            
        # 计算残基概率
        total_votes = np.sum(vote_counts, axis=1)
        probs = np.zeros((L, 3))
        mask = total_votes > 0
        probs[mask] = vote_counts[mask] / total_votes[mask, None]
        
        residue_confidence = []
        for i in range(L):
            residue_confidence.append({
                "Acidic": probs[i, 0],
                "Basic": probs[i, 1],
                "Neutral": probs[i, 2]
            })
            
        # 更新电荷块的置信度信息
        updated_blocks = []
        for b in observed_blocks:
            # 计算该块区域内残基被分类为该类型的平均概率
            block_probs = probs[b.start-1 : b.end, type_map[b.block_type]]
            avg_conf = np.mean(block_probs) if len(block_probs) > 0 else 0.0
            
            # 确定置信度等级
            conf_level = "Low"
            if b.p_value is not None and b.p_value < 0.01 and avg_conf > 0.9:
                conf_level = "High"
            elif (b.p_value is not None and b.p_value < 0.05) and avg_conf > 0.7:
                conf_level = "Medium"
                
            new_b = replace(b, boundary_confidence=avg_conf, confidence_level=conf_level)
            updated_blocks.append(new_b)
            
        return updated_blocks, residue_confidence

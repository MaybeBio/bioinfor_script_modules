# 序列重排、置换
# 新序列包含的元素与原序列完全一致（无新增、无缺失）；
# 元素的排列顺序与原序列不同（若原序列元素无重复，所有排列均为 “全新顺序”；若有重复，需排除与原顺序完全一致的情况，具体依场景而定）

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

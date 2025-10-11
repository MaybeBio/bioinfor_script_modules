# 获取连续简约字符pattern (Continuous reduced characters pattern)

# example 1, reference to py.35
    def get_charge_block_pattern(self, charge_blocks: dict) -> dict:
       
        # 获取电荷块信息
        charge_blocks = charge_blocks["Charge_blocks"]

       # 获取单一符号表示的电荷信号序列
        charge_pattern_original = ['0'] * len(self.sequence)  # 初始化为全中性
        for start, end, size, seq, net_charge, NCPR, block_type in charge_blocks:
            for i in range(start - 1, end):  # 转换为0-based index
                if block_type == "Acidic":
                    charge_pattern_original[i] = '-'
                elif block_type == "Basic":
                    charge_pattern_original[i] = "+"
        charge_pattern_original = "".join(charge_pattern_original)

        # 获取reduce合并之后简约的电荷信号序列
        charge_pattern_reduced = []
        charge_pattern_reduced = ['0'] * len(charge_blocks) # 初始化块数个全中性
        for idx,(start,end,size,seq,net_charge,NCPR,block_type) in enumerate(charge_blocks):
            if block_type == "Acidic":
                charge_pattern_reduced[idx] = '-'
            elif block_type == "Basic":
                charge_pattern_reduced[idx] = "+"
        charge_pattern_reduced = "".join(charge_pattern_reduced)

        # 以字典形式返回最后的两种结果
        charge_pattern = {
            "original_pattern": charge_pattern_original,
            "reduced_pattern": charge_pattern_reduced
        }
        return charge_pattern

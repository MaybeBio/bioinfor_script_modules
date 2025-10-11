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

================================================================================================================


def get_charge_block_pattern(self, block: Charge_Analysis_Result) -> Dict[str, str]:
        """
        Description:
            生成蛋白质电荷块信号的简约符号(sign)序列(reduced-alphabet),
        
        Args:
            block (Charge_Analysis_Result): Charge_Analysis_Result对象, 由get_charge_blocks_residue或get_charge_blocks_cell方法中间返回的结果
        
        Returns:
            dict: {"original_signal": str, "reduced_signal": str}
            返回原始电荷信号和简约电荷信号两种形式的字典
        
        Note:
            1, 在get_charge_blocks_residue方法中补充该方法的调用, 作为最终结果的pattern一部分返回
        """
        
        # 获取电荷块信息
        charge_blocks = block.Charge_blocks

       # 获取单一符号表示的电荷信号序列
        charge_pattern_original = ['0'] * len(self.sequence)  # 初始化为全中性
        for b in charge_blocks:
            for i in range(b.start - 1, b.end):  # 转换为0-based index
                if b.block_type == "Acidic":
                    charge_pattern_original[i] = '-'
                elif b.block_type == "Basic":
                    charge_pattern_original[i] = "+"
        charge_pattern_original = "".join(charge_pattern_original)

        # 获取reduce合并之后简约的电荷信号序列
        charge_pattern_reduced = []
        charge_pattern_reduced = ['0'] * len(charge_blocks) # 初始化块数个全中性
        for idx,b in enumerate(charge_blocks):
            if b.block_type == "Acidic":
                charge_pattern_reduced[idx] = '-'
            elif b.block_type == "Basic":
                charge_pattern_reduced[idx] = "+"
        charge_pattern_reduced = "".join(charge_pattern_reduced)

        # 以字典形式返回最后的两种结果
        charge_pattern = {
            "original_pattern": charge_pattern_original,
            "reduced_pattern": charge_pattern_reduced
        }
        return charge_pattern

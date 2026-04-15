# 将1个pdb结构文件，解析出其SIFTS注释层级的信息
# 即解析出其  每一个结构建模残基真正对应 uniprot序列层面的index、以及坐标

# 1.
# 简易版
# SIFTS信息已经内置在mmcif文件中，可以直接使用高级点的parser中获取残基级的SIFTS注释

import pandas as pd
from Bio.PDB import MMCIFParser
def pdb_to_residue_sifts(pdb_file: str) -> pd.DataFrame:
    parser = MMCIFParser()
    structure = parser.get_structure("structure", pdb_file)
    
    rows = []
    for model in structure:
        for chain in model:
            for residue in chain:
                # 我们只要α-C原子坐标, 前提得是这个残基有α-C原子, 也就是排除非标准氨基酸
                if residue.has_id("CA"):
                    res_id = residue.id[1]  # 获取残基编号
                    res_name = residue.get_resname()  # 获取残基名称
                    chain_id = chain.id  # 获取链ID
                    rows.append({
                        'pdb_id': structure.header['idcode'],
                        "chain_id": chain_id,
                        "residue_name": res_name,
                        "residue_number": res_id,
                        "CA_coord": residue['CA'].get_coord()
                    })
    
    return pd.DataFrame(rows)

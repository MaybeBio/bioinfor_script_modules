# 检验1个结构模型文件是否有效，通过检验其模型是否完好、是否有模型、是否可通过biopython模块解析

# 1.


from typing import *
from pathlib import Path
import warnings
from Bio.PDB import PDBParser, MMCIFParser

def validate_structural_file(file_loc: Union[str, Path], file_ext: str) -> bool:
    """
    Description
    -----------
    利用 Biopython 对目标结构文件进行尝试解析，检查其是否完整(无模型)或损坏(无法解析)，以此验证下载的结构文件的有效性
    """
    p_loc = Path(file_loc)
    if not p_loc.exists():
        return False
        
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            parser_engine = MMCIFParser(QUIET=True) if file_ext.lower() == '.cif' else PDBParser(QUIET=True)
            struct = parser_engine.get_structure("test_model", str(p_loc))
            models = list(struct.get_models())
            return len(models) > 0
    except Exception as e:
        print(f"Struct file invalid {p_loc.name}: {e}")
        return False


########################################################################################################3

# 2.
def validate_structural_file(file_loc: Union[str, Path], file_ext: str) -> bool:
    """
    Description
    -----------
    利用 Biopython 对目标结构文件进行尝试解析，检查其是否完整(无模型)或损坏(无法解析)，以此验证下载的结构文件的有效性
    """
    p_loc = Path(file_loc)
    if not p_loc.exists():
        return False
        
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            parser_engine = MMCIFParser(QUIET=True) if file_ext.lower() == '.cif' else PDBParser(QUIET=True)
            struct = parser_engine.get_structure("test_model", str(p_loc))
            models = list(struct.get_models())
            return len(models) > 0
    except Exception as e:
        logger.debug(f"Struct file invalid {p_loc.name}: {e}")
        return False

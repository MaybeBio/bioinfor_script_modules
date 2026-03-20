# 仅提供1个uniprot id, 获取其结构文件(cif或pdb文件), 可以是PDB、AlphaFoldDB等数据库中的结构文件

# 1. 如果是AlphaFoldDB
# 参考: https://alphafold.ebi.ac.uk/api-docs

import os
import requests
def fetch_pdb_from_alphafolddb(uniprot_id:str, pdb_save_dir:str = "."):
    """
    Description
    -----------
    从 AlphaFold 数据库获取指定 UniProt ID 的预测结构, 并将其保存为 CIF 格式文件到指定目录

    Args
    ----
    uniprot_id: str
        目标蛋白的 UniProt ID, 例如 "P49711"
    pdb_save_dir: str
        PDB 文件的保存目录, 默认为当前目录 "."
    
    """
    # 先创建目标文件夹
    os.makedirs(pdb_save_dir, exist_ok=True)
    api_url = f"https://alphafold.ebi.ac.uk/api/uniprot/summary/{uniprot_id}.json"
    response = requests.get(api_url).json()
    pdb_url = response['structures'][0]['summary']['model_url']
    # 继续下载PDB文件并保存到指定路径
    # 或者用subprocess+curl
    pdb_response = requests.get(pdb_url)
    if pdb_response.status_code == 200:
        pdb_save_file = os.path.join(pdb_save_dir, f"{uniprot_id}.cif")
        with open(pdb_save_file, "wb") as f:
            f.write(pdb_response.content)
        print(f"✅ PDB file for {uniprot_id} saved to {pdb_save_file}")
    else:
        print(f"❌ Failed to download PDB file for {uniprot_id}. HTTP status code: {pdb_response.status_code}")

# 仅提供1个uniprot id, 获取其结构文件(cif或pdb文件), 可以是PDB、AlphaFoldDB等数据库中的结构文件

# 1. 如果是AlphaFoldDB, 获取isoform1也就是主要蛋白质形态的cif文件
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


####################################################################################################################################################


# 2.1 直接从uniprot数据库中依据REST API获取每一个uniprot id下对应PDB原始数据库的PDB id, chain id以及坐标范围
# 暂时还拿不到结构, 只能拿到PDB id, 需要借助PDB api进一步获取该结构

import requests
import pandas as pd
def fetch_PDB_id_from_uniprot(uniprot_ac: str) -> pd.DataFrame:
    """   
    Description
    -----------
    提供uniprot_ac, 获取对应的PDB ID、链ID和范围信息,并返回一个DataFrame.

    Args
    ----
    uniprot_ac : str
        UniProt的访问号.
    
    Returns
    -------
    pd.DataFrame
        包含PDB ID、链ID和范围信息的DataFrame.

    Notes
    -----
    - 1. 如果是单个元素, 可以将返回的DataFrame另存为csv文件
    - 2. 如果是多个元素, 可以逐个uniprot_ac 调用该函数, 将输出的DataFrame合并后再另存为csv文件, 或者改写该函数, 
    直接输入1个uniprot_ac列表, 然后直接结果上entry_list全append, 最后一次性转换为DataFrame并返回.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_ac}"
    response = requests.get(url)

    entry_list = []
    if response.status_code == 200:
        res = response.json()
        for db_entry in res['uniProtKBCrossReferences']:
            if db_entry["database"] == "PDB":
                pdb_id = db_entry["id"]
                for property_entry in db_entry["properties"]:
                    if property_entry["key"] == "Chains":
                        # E/F/G -> E,F,G
                        chain_ids = ",".join(property_entry["value"].split("=")[0].split("/"))
                        ranges = property_entry["value"].split("=")[1]

                        # 添加到结果列表中
                        entry_list.append({
                            "pdb_id": pdb_id,
                            "chain_ids": chain_ids,
                            "ranges": ranges
                        })
        # 将结果列表转换为DataFrame
        df_pdb = pd.DataFrame(entry_list)
        return df_pdb
    else:
        print(f"Failed to fetch data for {uniprot_ac}. Status code: {response.status_code}")
        return pd.DataFrame()



# 2.2
# 同样简化写法, 借助xref或database:pdb有
def fetch_PDB_id_from_uniprot(uni_id: str) -> pd.DataFrame:
    url = f"https://rest.uniprot.org/uniprotkb/search?query=reviewed:true+AND+accession_id:{uni_id}&fields=xref_pdb"
    response = requests.get(url).json()
    res = response['results'][0]['uniProtKBCrossReferences']
    pdb_info_list = []
    for db_entry in res:
        pdb_id = db_entry["id"]
        for property_entry in db_entry["properties"]:
            if property_entry["key"] == "Chains":
                chain_ids = ",".join(property_entry["value"].split("=")[0].split("/"))
                ranges = property_entry["value"].split("=")[1]
                pdb_info_list.append((pdb_id, chain_ids, ranges))
    return pd.DataFrame(pdb_info_list, columns=["PDB ID", "Chain IDs", "Ranges"])

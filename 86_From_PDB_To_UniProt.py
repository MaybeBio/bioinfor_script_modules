# 和 py81相对
# py81是知道1个uniprot id，查询该蛋白质的所有相关涉及结构
# 本脚本是给出1个pdb id，拆分出所以entity，也就是给出所以uniprot id

import requests
def from_pdb_to_uniprot(pdb_id:str) -> list:
    """ 
    Description
    -----------
    输入1个PDB ID, 返回该PDB ID对应的所有Uniprot ID的列表(以蛋白质为主)

    Args
    ----
    pdb_id : str
        输入的PDB ID

    Returns
    -------
    list
        该PDB ID对应的所有Uniprot ID的列表(以蛋白质为主)
        
    """

    entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    response = requests.get(entry_url)
    if response.status_code != 200:
        print(f"查询失败，状态码: {response.status_code}")
        return None
    else:
        entry_data = response.json()
        all_entity_ids = entry_data["rcsb_entry_container_identifiers"]["polymer_entity_ids"]
        uniprot_ids = []
        for entity_id in all_entity_ids:
            entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
            entity_response = requests.get(entity_url)
            if entity_response.status_code != 200:
                print(f"查询实体数据失败，状态码: {entity_response.status_code}")
                continue
            else:
                entity_data = entity_response.json()
                if "uniprot_ids" in entity_data["rcsb_polymer_entity_container_identifiers"].keys():
                    uniprot_ids.extend(entity_data["rcsb_polymer_entity_container_identifiers"]["uniprot_ids"])
        return list(set(uniprot_ids))
    
    

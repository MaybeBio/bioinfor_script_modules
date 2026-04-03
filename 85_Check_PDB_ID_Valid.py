# 主要是检查PDB ID是否过期（相同需求比如说uniprot id也需要检查是否过期）
# 如果已过期，就用新的id替代
# 检查逻辑就是去removed/目录下查看是否有新的pdb id替代它


# 1. 
# 脚本参考/disobind/dataset/from_APIs_with_love.py

def get_superseding_pdb_id(pdb_id, max_trials=5, wait_time=30):
    url = f"https://data.rcsb.org/rest/v1/holdings/removed/{pdb_id}"
    import requests
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "id_codes_replaced_by" in data["rcsb_repository_holdings_removed"].keys():
            new_pdb_id = data["rcsb_repository_holdings_removed"]["id_codes_replaced_by"][0]
		    # If PDB ID has become obsolete and no new ID has been assigned (e.g. 8fg2).
        else:
            new_pdb_id = None
    else:
        new_pdb_id = pdb_id
    return [pdb_id, new_pdb_id]
get_superseding_pdb_id("8fg2")

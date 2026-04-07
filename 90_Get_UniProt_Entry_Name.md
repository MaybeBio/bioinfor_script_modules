<img width="1202" height="179" alt="image" src="https://github.com/user-attachments/assets/8f5a94b6-73f6-473f-9ef0-2ea40ec196f5" />

``` python
# data['proteinDescription']
"""
有 {'recommendedName': {'fullName': {'value': 'Transcriptional repressor CTCF'}},
 'alternativeNames': [{'fullName': {'value': '11-zinc finger protein'}},
  {'fullName': {'value': 'CCCTC-binding factor'}},
  {'fullName': {'value': 'CTCFL paralog'}}]}
""" 

# uniprot entry name获取函数
import httpx

def get_uniprot_entry_name( uniprot_id: str): 
    """
	Description
	-----------
	获取uniprot id对应的蛋白质名称
	
	Args
	----
	uniprot_id: str
        uniprot id, e.g. P04637
	"""
    # 注意P49711是 uniprot primary accession number，CTCF_HUMAN是 uniprot entry name
    import httpx
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        response = httpx.get(url)
        response.raise_for_status()  # 检查请求是否成功
        data = response.json()
        entry_name = data.get("primaryAccession", None)  # 获取entry name
        if "recommendedName" in data['proteinDescription'].keys():
            prot_name = data['proteinDescription']['recommendedName']['fullName']['value']
        elif "alternativeNames" in data['proteinDescription'].keys():
            prot_name = data['proteinDescription']['alternativeNames'][0]['fullName']['value']
        return [entry_name, prot_name]
    except httpx.HTTPError as e:
        print(f"Error fetching UniProt data for {uniprot_id}: {e}")
        return None


get_uniprot_entry_name("P49711")
# ['P49711', 'Transcriptional repressor CTCF']


```

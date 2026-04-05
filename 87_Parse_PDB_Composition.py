# 给定1个pdb结构，解析其entity实体组成，到底有几个组分、分别是多少链，是同源多聚体还是单体结构等

# 1.
# 零散，未封装成函数

import requests
import pandas as pd 


def send_request( url, _format = "json", max_trials = 10, wait_time = 5 ):
	"""
	Send a request to the server to fetch the data.
		For Httpresponse 404 returns "not_found".
		For Httpresponse 400 returns "bad_request".
	
	Input:
	----------
	url --> URL of the server from where to fetch the data.
	_format --> output format for the server reponse (json or text).
				If None, the response is returned.
	max_trial --> in case retrieval fails, try again uptil max_trials.
	wait_time --> wait some time before sending another request to the server.
		This is a variant of the Exponential backoff algorithm.


	Returns:
	----------
	Currently, will return either of:
		_format = None: response object.
		_format = json: return a JSON dict.
		_format = text: return the response in text format.
	"""
	for trial in range( 0, max_trials ):
		try:
			response = requests.get( url )

			# Resource not found.
			if response.status_code == 404:
				if trial > ( max_trials/2 ):
					continue
				else:
					return "not_found"

			# Bad request.
			elif response.status_code == 400:
				if trial != ( max_trials - 1 ):
					continue
				else:
					return "bad_request"

			elif response.status_code == 200:
				if _format == None:
					return response
				elif _format == "json":
					return response.json()
				else:
					return response.text
				break
			
			else:
				raise Exception( f"--> Encountered status code: {response.status_code}\n" )
		except Exception as e:
			if trial != max_trials-1:
			# 	print( f"Trial {trial}: Exception {e} \t --> {url}" )
				continue
			else:
				return "not_found"

def get_pdb_entry_info( entry_id, max_trials = 10, wait_time = 5 ):
	"""
	Retrieve PDB Entry level info from PDB REST API.

	Input:
	----------
	entry_id --> PDB ID.
	max_trial --> in case retrieval fails, try again uptil max_trials.
	wait_time --> wait some time before sending another request to the server.

	Returns:
	----------
	entry_data --> dict containing entry level info for the specified entry_id.
	"""
	entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{entry_id}"
	entry_data = send_request( entry_url, _format = "json", 
								max_trials = max_trials, 
								wait_time = wait_time )

	return entry_data


def get_pdb_entity_info( entry_id, entity_id, max_trials = 10, wait_time = 5 ):
	"""
	Retrieve PDB Entity level info from PDB REST API for the entry_id and entity_id.

	Input:
	----------
	entry_id --> PDB ID.
	entity_id --> Entity ID for associated with a PDB entry.
	max_trial --> in case retrieval fails, try again uptil max_trials.
	wait_time --> wait some time before sending another request to the server.

	Returns:
	----------
	entity_data --> dict containing entity level info for the specified entry_id and entry_id.
	"""
	entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{entry_id}/{entity_id}"
	entity_data = send_request( entity_url, _format = "json", 
								max_trials = max_trials, 
								wait_time = wait_time )

	return entity_data


# 下面是实际数据处理阶段

# 我们使用实际pbd id来检查, 比如所7W1M
# 我们将最难的模块一步一步来拆解

entry_id =  "7W1M"

# ❓存储pdb id，然后是链、asym信息是？ 然后是uniprot id
df = pd.DataFrame( columns = ["PDB ID", "Entity ID", "Asym ID", "Auth Asym ID", "Uniprot ID"] )

# ❓存储所有实体和链信息，看看哪些是chimeric的，哪些是non-protein的
np_entity, chimeric, total_chains = [], [], 0
all_uni_ids = []

# 存储entry_data和 entity_data
entry_data, entity_data = [None, None]
if entry_data == None:
    # 先获取entry_data，这是1个字典
    entry_data = get_pdb_entry_info( entry_id )

if entry_data == "not_found":
    print( "PDB ID not found." )
else:
    # 开始收集数据
    all_entity_ids = entry_data["rcsb_entry_container_identifiers"]["polymer_entity_ids"]

    # ❓
    # 不知道在存什么
    entity_ids = []
    asym_ids, auth_asym_ids, uniprot_ids = {}, {}, {}

    # 存储entity_data的字典
    entity_dict = {} if entity_data == None else entity_data
    
    # ❓
    row_idx = 0

    # 然后开始就是对每个entity id进行循环，获取entity_data
    for entity_id in all_entity_ids:
        # 获取entity_data
        if entity_data == None:
            entity_dict[entity_id] = get_pdb_entity_info( entry_id, entity_id )
    
            if entity_dict[entity_id] == "not_found":
                print( "Entity data not found." )
                continue
        
        # 加上每一个entity本身的chain id数
        total_chains += len( entity_dict[entity_id]["rcsb_polymer_entity_container_identifiers"]["asym_ids"] )
        
        if "uniprot_ids" in entity_dict[entity_id]["rcsb_polymer_entity_container_identifiers"].keys():
            asym_ids = entity_dict[entity_id]["rcsb_polymer_entity_container_identifiers"]["asym_ids"]
            auth_asym_ids = entity_dict[entity_id]["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"]
            uniprot_ids = entity_dict[entity_id]["rcsb_polymer_entity_container_identifiers"]["uniprot_ids"]

            all_uni_ids.extend( uniprot_ids )

            # Remove chimeric entities.
            if len( uniprot_ids ) > 1: 
                # ⚠️ 我们这里人工定义1个变量
                coverage = []
                for i in range( len( uniprot_ids ) ):
                    coverage_list = entity_dict[entity_id]["rcsb_polymer_entity_container_identifiers"]['reference_sequence_identifiers']
                    for j in coverage_list:
                        if j['database_accession'] == uniprot_ids[i]:
                            coverage[i] = j['entity_sequence_coverage']
                            
                # 然后收集齐了每个uniprot id的coverage之后，我们就可以判断是否是chimera了
                chimera = any([c != coverage[0]   for c in coverage])
                    
            else:
                chimera = False

            # 如果不是chimera，我们就把它的信息存到dataframe里
            if not chimera:

                # ⚠️ 有多个chain吗？按理来说只有1个chain 
                for i in range( len( asym_ids ) ):
                    df.loc[row_idx] = [
                            entry_id, # 7w1m
                            entity_id, # chain id，比如说1
                            asym_ids[i], # A
                            auth_asym_ids[i], # A
                            ",".join( uniprot_ids ) # 这里指的就是非嵌合体但是还有多个uniprot id的情况，比如说是真同源
                                ]
                    # 这里看样子是每一个chain都要存一行吗？如果是的话，那就对了
                    row_idx += 1 
            else:
                # 嵌合体是针对entity层级来说的，
                # ⚠️ 如果有嵌合体，比如说是某一个pdb的第3个entity，我们就记录下来，然后排除
                chimeric.append( f"{entry_id}_{entity_id}" )
        
        # 如果没有uniprot id的话，我们就把它记录为non-protein entity，因为我们只要蛋白质
        else:
            np_entity.append( f"{entry_id}_{entity_id}" ) 
    

# 后续要分析，主要是查看前面输出的7个变量

# df, entry_data, entity_dict, chimeric, np_entity, total_chains, all_uni_ids       





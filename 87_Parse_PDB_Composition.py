# 给定1个pdb结构entry，解析其entity实体组成，到底有几个组分、分别是多少链，是同源多聚体还是单体结构等

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


#####################################################################################################

# 2.
# 封装好之后

import httpx 
import pandas as pd
from io import StringIO
from Bio import SeqIO


def parse_pdb_entry(entry_id):
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
				response = httpx.get( url )

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


	def get_uniprot_seq( uni_id, max_trials = 5, wait_time = 5, return_id = False ):
		"""
		Obtain Uniprot seq for the specified Uniprot ID.

		Input:
		----------
		uni_id --> Uniprot accession.
		max_trial --> in case retrieval fails, try again uptil max_trials.
		wait_time --> wait some time before sending another request to the server.
		
		Returns:
		----------
		Sequence for the given Uni ID.
			Uni ID is returned if specified.
		An empty list will be returned if couldn't find the sequence for the given Uni ID.
		"""
		url = f"http://www.uniprot.org/uniprot/{uni_id}.fasta"
		
		data = send_request( url, _format = "text", max_trials = max_trials, wait_time = wait_time )
		if data == "not_found" or data == "bad_request":
			return [uni_id, []] if return_id else []
		
		else:
			seq_record = [str( record.seq ) for record in SeqIO.parse( StringIO( data ), 'fasta' )]

			if seq_record == []:
				return [uni_id, []] if return_id else []
			else:
				return [uni_id, seq_record[0]] if return_id else seq_record[0]

	def is_chimera( uni_ids ):
		"""
		Chimeric proteins will have >1 Uniprot IDs,
			each belonging to a distinct protein.
		Check if all Uniprot IDs are the same or not.
			All should have the same sequence.

		Input:
		----------
		uni_ids --> list of Uniprot IDs.

		Returns:
		----------
		chimera --> (bool) True any of the protein sequences do not match.
		"""
		sequences = [get_uniprot_seq( id_ ) for id_ in uni_ids]
		sequences = [seq for seq in sequences if seq != []]
		
		chimera = any( [seq != sequences[0] for seq in sequences] )
		# print( chimera )
		# exit()

		return chimera

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

	def from_pdb_rest_api_with_love( entry_id, max_trials = 10, wait_time = 5, custom = [None, None] ):
		"""
		Obtain Chain and UniProt IDs using the PDB ID.

		Input:
		----------
		entry_id --> PDB ID.
		max_trials --> maximum no. of attempts to fetch info. from the URL.
		wait_time --> waiting time before sending a request to the server again.
		custom --> If not None, will use the PDB Entry and Entity data dicts provided.

		Returns:
		----------
		0 if error obtaining info from REST API 
		None if entity information or entry information does not exist
		tuple of 
		- dataframe 
		- entry_data
		- entity_data 
		- list of entities with chimeric chains
		- list of entities with non-protein chains 

		"""
		df = pd.DataFrame( columns = ["PDB ID", "Entity ID", "Asym ID", "Auth Asym ID", "Uniprot ID"] )
		np_entity, chimeric, total_chains = [], [], 0
		all_uni_ids = []

		entry_data, entity_data = custom
		# If pre-existing data file does not exist.
		if entry_data == None:  # TODO there is no else statement to use pre-existing data file?
			entry_data = get_pdb_entry_info( entry_id, max_trials = max_trials, wait_time = wait_time )

		# PDB ID does not exist.
		if entry_data == "not_found": # TODO this will depend on previous condition?
			return None
	
		else:
			# Initialize an empty dataframe to store all relevant info for the PDB ID.
			# Obtain the entity ids in the PDB.
			all_entity_ids = entry_data["rcsb_entry_container_identifiers"]["polymer_entity_ids"]

			entity_ids = []
			asym_ids, auth_asym_ids, uniprot_ids = {}, {}, {}
			
			# Dict to store entity data from PDB API for each entity ID.
			# Use existing one else create anew.
			entity_dict = {} if entity_data == None else entity_data  # TODO this can be taken inside the for loop. Redundant.

			row_idx = 0

			# All the IDs are strings.
			# For all the entities in the PDB.
			for entity_id in all_entity_ids:
				# If using pre-downloaded data, do not download again.
				if entity_data == None:
					entity_dict[entity_id] = get_pdb_entity_info( entry_id, entity_id, max_trials = max_trials, wait_time = wait_time )

					if entity_dict[entity_id] == "not_found":
						continue

				# Count all chains that exist in the PDB - protein and non-protein.
				total_chains += len( entity_dict[entity_id]["rcsb_polymer_entity_container_identifiers"]["asym_ids"] )

				if "uniprot_ids" in entity_dict[entity_id]["rcsb_polymer_entity_container_identifiers"].keys():
					asym_ids = entity_dict[entity_id]["rcsb_polymer_entity_container_identifiers"]["asym_ids"]
					auth_asym_ids = entity_dict[entity_id]["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"]
					uniprot_ids = entity_dict[entity_id]["rcsb_polymer_entity_container_identifiers"]["uniprot_ids"]

					all_uni_ids.extend( uniprot_ids )

					# Remove chimeric entities.
					if len( uniprot_ids ) > 1:
						chimera = is_chimera( uniprot_ids )
					else:
						chimera = False

					if not chimera:

						for i in range( len( asym_ids ) ):
							df.loc[row_idx] = [
									entry_id, 
									entity_id,
									asym_ids[i],
									auth_asym_ids[i],
									",".join( uniprot_ids )
										]
							row_idx += 1
					else:
						chimeric.append( f"{entry_id}_{entity_id}" )

				else:
					np_entity.append( f"{entry_id}_{entity_id}" )
		return df, entry_data, entity_dict, chimeric, np_entity, total_chains, all_uni_ids
		# return [entity_ids, asym_ids, auth_asym_ids, uniprot_ids, [entry_data, entity_dict]]

	return from_pdb_rest_api_with_love(entry_id)

# 示例
df, entry_data, entity_dict, chimeric, np_entity, total_chains, all_uni_ids = parse_pdb_entry("7W1M")

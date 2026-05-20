# 对于pdb结构文件，解析每一条chain也就是每一个entity，比如说给出每一个entity对应的uniprot id等

# 1. 解析本地pdb结构文件
# 给出每一个entity的uniprot id

    def _extract_chain_to_uniprot_from_cif(self, pdb_id: str) -> Dict[str, str]:
        """中文说明: 从本地 mmCIF 文件解析 chain -> uniprot 的映射（不走 PDBe API）。"""

        cif_path = self.structures_dir / f"{pdb_id}.cif"
        if not cif_path.exists():
            return {}

        try:
            blob = MMCIF2Dict(str(cif_path))
        except Exception:
            return {}

        ref_ids = self._as_list(blob.get("_struct_ref.id"))
        ref_db_names = self._as_list(blob.get("_struct_ref.db_name"))
        ref_accs = self._as_list(blob.get("_struct_ref.pdbx_db_accession"))

        ref_to_acc: Dict[str, str] = {}
        for rid, dbn, acc in zip(ref_ids, ref_db_names, ref_accs):
            if str(dbn).strip().upper() != "UNP":
                continue
            rid_s = str(rid).strip()
            acc_s = str(acc).strip()
            if rid_s and acc_s:
                ref_to_acc[rid_s] = acc_s.split("-")[0]

        seq_ref_ids = self._as_list(blob.get("_struct_ref_seq.ref_id"))
        seq_strands = self._as_list(blob.get("_struct_ref_seq.pdbx_strand_id"))
        seq_accs = self._as_list(blob.get("_struct_ref_seq.pdbx_db_accession"))

        chain_to_uniprot: Dict[str, str] = {}
        for rid, strands, seq_acc in zip(seq_ref_ids, seq_strands, seq_accs):
            rid_s = str(rid).strip()
            uni = ref_to_acc.get(rid_s, str(seq_acc).strip().split("-")[0])
            if not uni:
                continue
            for chain_id in str(strands).split(","):
                cid = chain_id.strip()
                if cid and cid not in chain_to_uniprot:
                    chain_to_uniprot[cid] = uni
        return chain_to_uniprot



#########################################################################################################################################################

# 2. 通过PDBe api

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

# print( get_pdb_entry_info( "6vu3" ) )
# exit()


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
					return None

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

################################################################################################################

# 3.

def fetch_pdb_entity_details(pdb_id: str) -> Tuple[Dict, bool, bool, int, List[str]]:
    """
    Description
    -----------
    解构 PDB 复合物的全部组装链信息 
    
    Args
    ----
    pdb_id: str
    
    Returns
    -------
    Tuple 包含:
        - entity_composition (Dict): {entity_id: {"uni_ids": [], "asym_ids": [], "auth_ids": []}}
        - has_chimera (bool): 该PDB结构中是否存在嵌合体(多个Uniprot挤进同一条链)
        - has_non_protein (bool): 该结构是否有除了水之外的配体等非蛋白质实体
        - valid_protein_chain_count (int): 有效抗体/蛋白质的链数量
        - universe_uniprots (List): 结构中提及过的所有 UniProt ID(平面汇总)
    """

    # initialization
    all_uni_ids = []
    entity_compose_df = pd.DataFrame(columns=["PDB ID", "Entity ID", "Asym ID", "Auth Asym ID", "Uniprot ID"])
    chimeric_entities, np_entities, total_chains = [], [], 0

    # get_pdb_entry_info()
    entry_uri = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
    entry_data = http_client.execute(entry_uri, resp_type="json")
    
    if entry_data in ("NOT_FOUND", "BAD_REQUEST", None) or type(entry_data) is str:
        return ({}, False, False, 0, [])

    # all_entity_ids
    all_entity = entry_data.get("rcsb_entry_container_identifiers", {}).get("polymer_entity_ids", [])

    entity_dict = {}
    row_index = 0

    for entity in all_entity:
        # get_pdb_entity_info()
        entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.lower()}/{entity}"
        entity_data = http_client.execute(entity_url, resp_type="json")
        if entity_data in ("NOT_FOUND", "BAD_REQUEST", None) or not isinstance(entity_data, dict):
            continue
        entity_dict[entity] = entity_data

        # count all chains in this PDB entry - protein or non-protein
        total_chains += len(entity_data.get("rcsb_polymer_entity_container_identifiers", {}).get("asym_ids", []))
        
        if "uniprot_ids" in entity_data.get("rcsb_polymer_entity_container_identifiers", {}).keys():
            asym_ids = entity_data.get("rcsb_polymer_entity_container_identifiers", {}).get("asym_ids", [])
            auth_asym_ids = entity_data.get("rcsb_polymer_entity_container_identifiers", {}).get("auth_asym_ids", [])
            uniprot_ids = entity_data.get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids", [])
            
            all_uni_ids.extend(uniprot_ids)
            
            # remove chimeric entities, which have multiple uniprot ids mapped to the same chain, because they are hard to handle and may cause IDR boundary issues.
            if len(uniprot_ids) > 1:
                # ⚠️ 我们有很多方法可以检测1个entity是否为chimera，可以通过所有序列比对（disobind）、或者从entity_data中的uniprot 覆盖度等（csdn）
                # 我们这里从简，单纯从数目上判断
                has_chimera = True
            else:
                has_chimera = False

            # we only deal with non-chimeric entities
            if not has_chimera:
                for i in range(len(asym_ids)):
                    entity_compose_df.loc[row_index] = [
                        pdb_id.lower(),
                        entity,
                        asym_ids[i],
                        auth_asym_ids[i],
                        ",".join(uniprot_ids)
                    ]
                    row_index += 1
            else:
                chimeric_entities.append(f"{pdb_id.lower()}_{entity}")

        # non-protein entity
        np_entities.append(f"{pdb_id.lower()}_{entity}")

    return (
        entity_compose_df,  
        entry_data,
        entity_dict,
        chimeric_entities,
        np_entities,
        total_chains,
        all_uni_ids
    )

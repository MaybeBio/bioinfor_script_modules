# 检查1个给定的pdb id是否有效，是否过期，或更新为新的id数据

# 1.
# 可以尝试使用除了requests之外其他的http客户端，比如说httpx

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


def get_superseding_pdb_id( pdb_id, max_trials = 10, wait_time = 5 ):
	"""
	Obtain PDB IDs that supersedes the input PDB ID.

	Input:
	----------
	entry_id --> PDB ID.
	max_trials --> maximum no. of attempts to fetch info. from the URL.
	wait_time --> waiting time before sending a request to the server again.

	Returns:
	----------
	new_pdb_id --> superseeding PDB ID.
	"""
	url = f"https://data.rcsb.org/rest/v1/holdings/removed/{pdb_id}"
	data = send_request( url, 
						_format = "json", max_trials = 10, wait_time = wait_time  )
	
	# PDB ID is still active. No superseeding PDB ID exists.
	if data == "not_found":
		new_pdb_id = pdb_id
		# break

	else:
		if "id_codes_replaced_by" in data["rcsb_repository_holdings_removed"].keys():
			new_pdb_id = data["rcsb_repository_holdings_removed"]["id_codes_replaced_by"][0]
		# If PDB ID has become obsolete and no new ID has been assigned (e.g. 8fg2).
		else:
			new_pdb_id = None

	return new_pdb_id

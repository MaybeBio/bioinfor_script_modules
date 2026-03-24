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


##################################################################################################################################################

# 3.
# 从PDB结构数据库中直接依据API获取uniprot对应的PDB id

def fetch_PDB_id_from_uniprot(uni_id: str) -> set:
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity_groups/{uni_id}"
    response = requests.get(url).json()
    all_pdb_ids = response['rcsb_group_container_identifiers']['group_member_ids']
    pdb_ids = set(id[:4] for id in all_pdb_ids if not id.startswith("AF"))
    return pdb_ids 

###################################################################################################################################################

# 4.
# 合并法2以及法3, 可以将来源数据库RCSB PDB或者是Uniprot作为参数传入, 然后提供两个来源的pdb id, 可以将两个ID 合并起来作为防御性校验

import requests
from typing import *
def fetch_pdbID_for_uniprotID(uniprot_ac:str, source:Literal["uniprot", "rcsb"] = "uniprot") -> set:
    """ 
    Description
    -----------
    给定一个UniProt ID, 返回对应的PDB ID集合.

    Args
    ----
    uniprot_ac : str
        UniProt的访问号.
    source : Literal["uniprot", "rcsb"]
        指定使用哪个数据库的API来获取PDB ID. 可选值为 "uniprot" 或 "rcsb", 默认为 "uniprot".
        可以增加一个参数, 比如说是merge
        
    Returns
    -------
    set
        包含对应PDB ID的集合.
    """
    # 两个不同的api接口
    uniprot_url = f"https://rest.uniprot.org/uniprotkb/search?query=reviewed:true+AND+accession_id:{uniprot_ac}&fields=xref_pdb"
    rcsb_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity_groups/{uniprot_ac}"

    if source == "uniprot":
        response = requests.get(uniprot_url).json()
        res = response['results'][0]['uniProtKBCrossReferences']
        pdb_ids = set()
        for db_entry in res:
            pdb_ids.add(db_entry["id"])
    elif source == "rcsb":
        response = requests.get(rcsb_url).json()
        all_pdb_ids = response['rcsb_group_container_identifiers']['group_member_ids']
        pdb_ids = set(id[:4] for id in all_pdb_ids if not id.startswith("AF"))
    return pdb_ids


############################################################################################################################################################

# 5. 对4的改进
class APIFetcher:
    """
    Description
    -----------
    处理所有向外 HTTP 调用的安全会话层, 
    封闭了带指数退避(Exponential Backoff)特色的重试机制.
    
    Notes
    -----
    - 1. 使用 requests.Session() 保持连接池，比每次 requests.get 更快。
    """
    def __init__(self, retries: int = 10, backoff_factor: float = 1.0, timeout_sec: int = 10):
        """  
        Description 
        -----------
        初始化 APIFetcher 实例，设定重试次数、退避因子和请求超时.

        Args
        ----
        retries: int
            最大重试次数, 默认10次.
        backoff_factor: float
            每次重试之间的退避时间乘数, 默认1.0秒. 1.0*(1, 2, 3...) 秒的等待时间.
        timeout_sec: int
            每次请求的超时时间, 默认10秒.
        """

        self.retries = retries
        self.backoff = backoff_factor
        self.timeout = timeout_sec

        # self.session = requests.Session() + session.get(url)
        # HTTP请求基于TCP协议, 每次requests.get(url)都会建立新的链接:
        # 建立TCP连接-发送HTTP请求-接收响应-关闭TCP连接
        # 使用 requests.Session() 可以复用TCP连接, 避免每次请求都重新建立连接的开销, 提升效率.
        # 后续用同1个session发送请求时, 直接复用已建立的连接, 跳过"建立/关闭连接"的耗时步骤
        # 尤其适用于高并发调用同一域名API场景, 比如说重试、批量请求等.
        self.session = requests.Session()

    def execute(self, url: str, resp_type: str = "json") -> Any:
        """
        Description
        -----------
        执行请求并根据设定格式返回响应.
        
        Args
        ----
        url: str
            目标API的URL连接.
        resp_type: str
            期望解析的形式(response type)，支持 "json", "text", "raw". 即拿到HTTP响应后, 按什么格式解析并返回数据
            
        Returns
        -------
        包含设定类型的数据对象, 
        若 404 或资源无效返回特殊标志 "NOT_FOUND", 若为 400 返回 "BAD_REQUEST".

        Notes
        -----
        - 1. HTTP status code: 参考https://en.wikipedia.org/wiki/List_of_HTTP_status_codes,
        200(OK)请求成功, 400(Bad Request)请求格式错误, 404(Not Found)资源不存在.
        对于其他未显式匹配的状态码, 以及抛出的网络异常, 此处以触发重试机制为主, 不设置特殊返回值, 最终重试耗尽后返回 "NOT_FOUND" 以示失败.
        - 2. Uniprot中还有500 code, 参考: https://www.uniprot.org/help/rest-api-headers
        """
        for attempt in range(self.retries):
            try:
                res = self.session.get(url, timeout=self.timeout)
                # 成功响应
                if res.status_code == 200:
                    # 按照响应格式要求解析返回数据
                    if resp_type == "json":
                        return res.json()
                    elif resp_type == "text":
                        return res.text
                    elif resp_type == "raw":
                        return res.content
                    else:
                        # None 或未知格式默认返回原始响应对象
                        logger.warning(f"Unknown response type requested: {resp_type}. Defaulting to Res object.")
                        return res
                
                # 如果资源不存在且重试已过半
                if res.status_code == 404 and attempt > (self.retries // 2):
                    return "NOT_FOUND"
                # 如果请求格式错误且重试已接近尾声
                elif res.status_code == 400 and attempt >= (self.retries - 2):
                    return "BAD_REQUEST"

            except requests.RequestException:
                if attempt == self.retries - 1:
                    logger.info(f"[API_FAILED] Dead after {self.retries} attempts for: {url}")
                    return "NOT_FOUND"
            
            # Wait and backoff
            time.sleep(self.backoff * (attempt + 1))
        return "NOT_FOUND"

# Global instanced fetcher 全局API请求实例(全局变量)，全局复用同1个requests.Session()
http_client = APIFetcher()

# ⚠️ 这里开始才是正式内容
def fetch_pdbID_for_uniprotID(accession: str, source:Literal["uniprot", "rcsb"] = "uniprot") -> List[str]:
    """ 
    Description
    -----------
    给定一个UniProt ID, 返回对应的PDB ID的列表.

    Args
    ----
    uniprot_ac : str
        UniProt的访问号.
    source : Literal["uniprot", "rcsb"]
        指定使用哪个数据库的API来获取PDB ID. 可选值为 "uniprot" 或 "rcsb", 默认为 "uniprot".
        
    Returns
    -------
    List[str]:
        包含所有合法PDB ID的列表, 集合处理之后确保都是唯一的. 若无对应PDB或发生错误则返回空列表.
    """
    # 处理一下一些uniprot异构物id
    core_id = accession.split("-")[0].strip()
    # 两个不同的api接口
    uniprot_url = f"https://rest.uniprot.org/uniprotkb/search?query=reviewed:true+AND+accession_id:{core_id}&fields=xref_pdb"
    rcsb_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity_groups/{core_id}"

    if source == "uniprot":
        response = http_client.execute(uniprot_url, resp_type="json")
        if response in ("NOT_FOUND", "BAD_REQUEST") or not isinstance(response, dict):
            return []
        
        try:
            results = response.get("results", [])
            if not results:
                return []
            pdb_ids = set()
            for res in results:
                xrefs = res.get("uniProtKBCrossReferences", [])
                for xref in xrefs:
                    if xref.get("database") == "PDB":
                        # 如果为了防止大小写重复, 可以设置为.upper
                        pdb_ids.add(xref.get("id"))
            return list(pdb_ids)
        except Exception:
            return []
    
    elif source == "rcsb":
        response = http_client.execute(rcsb_url, resp_type="json")
        if response in ("NOT_FOUND", "BAD_REQUEST") or not isinstance(response, dict):
            return []
        try:
            container = response.get("rcsb_group_container_identifiers", {})
            group_members = container.get("group_member_ids", [])
            # "1A0N_1" -> "1A0N", 同时排除掉AlphaFold模型(通常以 "AF" 开头的PDB ID)
            valid_pdbs = [val[:4] for val in group_members if isinstance(val, str) and "AF" not in val]
            return list(set(valid_pdbs))
        except Exception:
            return []

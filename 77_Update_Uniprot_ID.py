# 用的依然是uniprot 的REST API, 参考https://www.uniprot.org/help/programmatic_access
# 相比脚本76
# 优化如下：
# 1. http请求不直接requests.get, 优化TCP连接池, 复用TCP连接
# 2. 针对uniprot ID变更情况进行防御性编程
# 主要用于从一些其他数据库挖掘uniprot id时下游检查id时效性, 是否更新等, 在元数据上的处理
# 参考 https://blog.csdn.net/weixin_62528784/article/details/158963913?spm=1001.2014.3001.5502

import requests
import time
import logging
from typing import *
logger = logging.getLogger(__name__)

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
                    logger.debug(f"[API_FAILED] Dead after {self.retries} attempts for: {url}")
                    return "NOT_FOUND"
            
            # Wait and backoff
            time.sleep(self.backoff * (attempt + 1))
        return "NOT_FOUND"

# Global instanced fetcher 全局API请求实例(全局变量)，全局复用同1个requests.Session()
http_client = APIFetcher()


def fetch_uniprot_metadata(accession: str) -> Optional[Union[Dict, int]]:
    """
    Description
    -----------
    获取指定的 UniProt ID 的底层 JSON 元数据记录,
    主要用于探查序列是否存在及等变更情况, 便于及时更新从数据库中挖掘到的uniprot id.
    参考: https://www.uniprot.org/help/api
    
    Args
    ----
    accession: str
        UniProt访问号 (e.g. "P49711")
        
    Returns
    -------
    Dict: 如果记录正常，返回 JSON 字典.
    int: 返回 0 表示序列已过期(Inactive).
    None: 没有找到网页或请求崩了.

    Notes
    ------
    - 1. 非200的http响应返回一律被视为失败, 有200返回的, 可能情况有3种:
        a. 正常记录: 直接返回JSON内容.
        b. 过期记录: JSON中会有 "entryType": "Inactive" 字段, 返回0以示过期. 一般表明这个记录被废弃了, 可能是删除、条目拆分, 前者不需要记录, 后者拆分之后的数据可能会发生IDR边界坐标漂移, 也不需要记录
        c. 返回内容正常, 但是实际是303 重定向的数据, 这种id也要更新
    """

    target_url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    content = http_client.execute(target_url, resp_type="json")
    
    # 防御性编程, 判断响应内容是否有效
    if content in ("NOT_FOUND", "BAD_REQUEST") or content is None:
        return None
        
    # 判断响应内容以及是否包含过期标志
    # 200 但 条目删除、拆分
    if content.get("entryType") == "Inactive":
        return 0
    # 200 但正常、重定向合并(⚠️ 这个需要处理)
    return content

def resolve_uniprot_accession(accession: str, redirect_ok: bool = True) -> Optional[str]:
    """
    Description
    -----------
    解析给定的 UniProt ID 并返回其最新、最权威的 Primary Accession.
    处理由于数据库更新、条目合并导致的旧 ID 问题 (底层通过 requests 自动处理 303 重定向).
    
    Args
    ----
    accession: str
        目标或可疑的过时 UniProt 编号
    redirect_ok: bool
        是否接受重定向后的新ID, 默认True
        
    Returns
    -------
    str: 若有效，返回最新的 Primary Accession (可能与输入一致, 也可能是重定向后的新ID)
    None: 若彻底废弃 (Inactive) 或不存在，则返回 None. 

    Notes
    -----
    - 1. 和前面的 fetch_uniprot_metadata 函数配合使用, 同样的操作逻辑.
    首先是非200的, 视为失败;
    200的, 如果是拆分, 因为边界漂移我们舍弃; 如果是删除, 我们舍弃;
    问题就在于如果是重定向, 这个比较难判断旧IDR坐标是否适用于新序列, 因为更新之后的ID可能序列也更新了, 
    所以我们这里加一个"刻舟求剑"的选项参数, 让用户自己选择是否接受重定向的ID, 但默认是接受的, 因为不接受的话就等于放弃了这个ID, 这也是我们这个函数的主要目的.
    """
    # content是json配置字典
    payload = fetch_uniprot_metadata(accession)
    # 非200, 或200但 条目删除、拆分, 都视为无效
    if payload is None or payload == 0:
        return None
    
    # 获取真实被访问的Primary Accession, 可能与输入相同, 也可能是重定向后的新ID
    primary_acc = payload.get("primaryAccession", accession)

    # 依据redirect_ok参数决定是否接受重定向后的ID, 默认接受
    if redirect_ok:
        return primary_acc
    else:
        if primary_acc != accession:
            logger.info(f"[BOUNDARY_DRIFT_RISK] {accession} redirected to {primary_acc}. Dropped to ensure coordinate safety.")
            return None
        else:
            return primary_acc


# 应用示例:
resolve_uniprot_accession("P49711", redirect_ok=True)

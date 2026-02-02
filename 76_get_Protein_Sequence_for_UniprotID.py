# 给定1个uniprot accession ID，获取对应蛋白质序列

import requests, time
def get_Uniprot_protein_fasta(query: str, format: str, compressed: bool, size: int) -> list[str]:
    """
    Description:
        根据给定的查询参数, 从Uniprot数据库中获取蛋白质的fasta序列, 支持分页请求;
        
    Args:
        query (str): 查询字符串, 用于指定搜索条件, 可以是单个序列数据， 也可以是批量查询;
        format (str): 返回数据的格式, 默认为"fasta";
        compressed (bool): 是否请求压缩格式的数据, 默认为False;
        size (int): 每页返回的记录数/单次请求返回的序列数, 默认为500
    
    Returns:
        list[str]: 包含所有获取到的fasta序列的列表
    """

    base = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query,
        "format": format,
        "compressed": compressed,
        "size": size
    }
    # 用于存储分页请求返回的fasta序列片段
    fasta_parts = []
    url = base
    # 分页请求循环，直到没有下一页
    while True:
        # 首次请求时使用params字典，后续请求的"下一页"url已包含所有必要参数，因此不再重复传入params参数
        # 请求超时设置为60秒
        r = requests.get(url, params=params if url == base else None, timeout=60)
        # 检查是否请求成功，否则抛出异常
        r.raise_for_status()
        
        # 解析响应内容，将当前请求返回的fasta序列添加到列表中
        fasta_parts.append(r.text)

        # 解析下一页url
        # 先尝试获取 “next” 对应的子字典，若不存在则返回空字典；再从子字典中获取 “url”，若不存在则返回None（表示无下一页）
        next_url = r.links.get("next", {}).get("url")
        # 若无下一页，则跳出循环
        if not next_url:
            break
        # 更新url为下一页的url，继续请求
        # next_url已包含所有参数，无需重复传递
        url, params = next_url, None
        # 为避免过于频繁请求，稍作延时
        time.sleep(0.2)

    # 返回所有获取到的fasta序列片段, 拼接为一个完整的字符串
    fasta_str = "".join(fasta_parts)
    # 最好分割为一个accession对应一个字符串的列表返回
    fasta_list = fasta_str.strip().split("\n>")

    # 假设我们都是检索一条序列
    fasta_entry = fasta_list[0]
    # 然后分割每一条序列
    fasta_seq = "".join(fasta_entry.split("\n")[1:])

    return fasta_seq	

# 使用示例
result = get_Uniprot_protein_fasta(query="P49711", format="fasta", compressed=False, size=500)

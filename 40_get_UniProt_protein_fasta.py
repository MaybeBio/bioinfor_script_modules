# 主要借助uniprot数据库所提供的api访问
# 参考api使用文档:https://www.uniprot.org/help/programmatic_access

# 1, 直接构造search的query url, 翻页访问

import requests, time
def get_Uniprot_protein_fasta(query: str, params_dict: dict, output_file: str) -> list[str]:
    """
    Description:
        根据给定的查询参数, 从Uniprot数据库中获取蛋白质的fasta序列, 支持分页请求;
        
    Args:
        query (str): 查询字符串, 用于指定搜索条件, 可以是单个序列数据，也可以是批量查询;
        params_dict (dict): 其他请求参数的字典, 一般包括以下键值对:
            format (str): 返回数据的格式, 默认为"fasta";
            compressed (bool): 是否请求压缩格式的数据, 默认为False;
            size (int): 每页返回的记录数/单次请求返回的序列数, 默认为500
        output_file (str): 输出文件路径, 用于保存获取到的fasta序列;

    Returns:
        list[str]: 包含所有获取到的fasta序列的列表
    
    Notes:
        1, params_dict中的键值对会被添加到请求参数中, query参数会单独传递, 虽然参数单独传递, 但是最后都合并到URL中进行请求。
        2, query参数参考https://www.uniprot.org/help/query-fields, params_dict参数则比较固定, 参考Args部分说明;
        3, query参数一般建议带上物种, organism_id:9606; 以及是否reviewed:true等过滤条件
    """

    base = "https://rest.uniprot.org/uniprotkb/search"
    # 将查询参数添加到params_dict中
    params_dict["query"] = query
    # 初始化请求url为基础url, 同时准备一个列表用于存储获取到的fasta序列片段
    pages, url = [], base
    
    # 分页请求循环，直到没有下一页
    while True:
        # 首次请求时使用params字典，后续请求的"下一页"url已包含所有必要参数，因此不再重复传入params参数
        # 请求超时设置为60秒
        r = requests.get(url, params=params_dict if url == base else None, timeout=60)
        # 检查是否请求成功，否则抛出异常
        r.raise_for_status()
        
        # 解析响应内容，将当前请求返回的fasta序列添加到列表中
        pages.append(r.text)

        # 解析下一页url
        # 先尝试获取 “next” 对应的子字典，若不存在则返回空字典；再从子字典中获取 “url”，若不存在则返回None（表示无下一页）
        next_url = r.links.get("next", {}).get("url")
        # 若无下一页，则跳出循环
        if not next_url:
            break
        # 更新url为下一页的url，继续请求
        # next_url已包含所有参数，无需重复传递
        url = next_url
        # 为避免过于频繁请求，稍作延时
        time.sleep(0.2)

    # 返回所有获取到的fasta序列片段, 拼接为一个完整的字符串
    all_fasta = "".join(pages).strip()

    if not all_fasta:
        return []

    # 存储单个条目、每一行line的缓冲区
    records, buffer = [], []
    for line in all_fasta.splitlines():
        if line.startswith(">"):
            # 说明新开了一个序列
            # 如果前面已经有序列了, 则合并并保存前面的序列(buffer，保存时补上换行符,其实就是将lines重新转回str)
            if buffer:
                records.append("\n".join(buffer))
            # 如果前面没有新的序列
            buffer = [line]
        else:
            buffer.append(line)
    
    # 处理最后一个序列
    if buffer:
        records.append("\n".join(buffer))

    # 我们可以再将每一个条目分别提取出来
    fasta_dict = {}
    for each_fasta in records:
        protein_id = each_fasta.split("|")[1]
        protein_seq =  "".join(line.strip() for line in each_fasta.splitlines()[1:])
        fasta_dict[protein_id] = protein_seq
        
    # 将结果保存到一个指定的输出文件中
    with open(output_file,"w") as file:
        file.write(all_fasta)
    return fasta_dict

# 示例, example, 可依据自己的需求构建更加复杂的url
query = "(organism_id:9606) AND (reviewed:true) AND ((xref:prosite-PS00028) OR (xref:prosite-PS50157))"
params_dict = {
    "format": "fasta",
    "compressed": False,
    "size": 500
}

get_Uniprot_protein_fasta(query=query, params_dict=params_dict, output_file="c2h2_zf_PROSITE.fasta")


================================================================================================================================

# 2, 使用stream接口

import requests, time
def get_Uniprot_protein_fasta(query: str, params_dict: dict, output_file: str) -> list[str]:
    """
    Description:
        根据给定的查询参数, 从Uniprot数据库中获取蛋白质的fasta序列, 支持分页请求;
        
    Args:
        query (str): 查询字符串, 用于指定搜索条件, 可以是单个序列数据，也可以是批量查询;
        params_dict (dict): 其他请求参数的字典, 一般包括以下键值对:
            format (str): 返回数据的格式, 默认为"fasta";
            compressed (bool): 是否请求压缩格式的数据, 默认为False;
            size (int): 每页返回的记录数/单次请求返回的序列数, 默认为500
        output_file (str): 输出文件路径, 用于保存获取到的fasta序列;

    Returns:
        list[str]: 包含所有获取到的fasta序列的列表
    
    Notes:
        1, params_dict中的键值对会被添加到请求参数中, query参数会单独传递, 虽然参数单独传递, 但是最后都合并到URL中进行请求。
        2, query参数参考https://www.uniprot.org/help/query-fields, params_dict参数则比较固定, 参考Args部分说明;
        3, query参数一般建议带上物种, organism_id:9606; 以及是否reviewed:true等过滤条件
    """

    base = "https://rest.uniprot.org/uniprotkb/stream"
    # 将查询参数添加到params_dict中
    params_dict["query"] = query
    # 初始化请求url为基础url, 同时准备一个列表用于存储获取到的fasta序列片段
    pages, url = [], base
    
    # 使用流式端点进行请求
    r = requests.get(url, params=params_dict, stream=True, timeout=60)
    r.raise_for_status()

    if not r.text.strip():
        return []

    # 存储单个条目、每一行line的缓冲区
    records, buffer = [], []

    # 对于stream, 建议使用iter_lines方法逐行读取内容, 不使用splitlines方法
    for line in r.iter_lines(decode_unicode=True):
        if line.startswith(">"):
            # 说明新开了一个序列
            # 如果前面已经有序列了, 则合并并保存前面的序列(buffer，保存时补上换行符,其实就是将lines重新转回str)
            if buffer:
                records.append("\n".join(buffer))
            # 如果前面没有新的序列
            buffer = [line]
        else:
            buffer.append(line)
        
    # 处理最后一个序列
    if buffer:
        records.append("\n".join(buffer))

    # 我们可以再将每一个条目分别提取出来
    fasta_dict = {}
    for each_fasta in records:
        protein_id = each_fasta.split("|")[1]
        protein_seq =  "".join(line.strip() for line in each_fasta.splitlines()[1:])
        fasta_dict[protein_id] = protein_seq
        
    # 将结果保存到一个指定的输出文件中
    with open(output_file,"w") as file:
        file.write(r.text)
    return fasta_dict

# 示例, example, 可依据自己的需求构建更加复杂的url
query = "(organism_id:9606) AND (reviewed:true) AND ((xref:prosite-PS00028) OR (xref:prosite-PS50157))"
params_dict = {
    "format": "fasta",
    "compressed": False
}

get_Uniprot_protein_fasta(query=query, params_dict=params_dict, output_file="uniprot_human_zinc_finger.fasta")


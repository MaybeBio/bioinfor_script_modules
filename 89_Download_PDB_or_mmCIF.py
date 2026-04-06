# 给定1个pdb id，下载其结构数据，优先cif后pdb
# 如果是天然单体数据，保底可以再用alphafoldDB数据库兜底


# 1. 
def download_pdb(pdb_id:str , file_dir:str) -> list:
    """
    Description
    -----------
    下载PDB结构文件
    
    Args
    ----
    pdb_id: str
        PDB结构ID
    file_dir: str
        文件保存路径的目录, 注意是目录而不是文件

    Returns
    -------
    list
        包含文件类型和PDB ID的列表
        
    Notes
    -----
    - 1. 后续可以考虑使用biopandas处理, https://github.com/BioPandas/biopandas?tab=readme-ov-file
    """
    
    # 将PDB ID转换为小写, 便于查询
    pdb_id = pdb_id.lower()
    
    
    # 先写1个下载http请求的函数，用于下文下载使用
    def _download_file(url, file_path):
        """   
        Description
        -----------
        下载文件的函数
        
        Args
        ----
        url: str
            文件下载链接
        file_path: str
            文件保存路径, 注意是文件而不是目录
        """
        import requests
        
        try:
            response = requests.get(url)
            response.raise_for_status()  # 检查请求是否成功
            
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"文件已成功下载: {file_path}")
            # 下载成功之后返回True
            return True
        except requests.exceptions.RequestException as e:
            print(f"下载文件时发生错误: {e}")
            return False
    
    # 构建url
    try:
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        success = _download_file(url, f"{file_dir}/{pdb_id}.cif")
        # 如果下载成功
        if success:
            return ["cif", pdb_id]
        else:
            # 如果cif下载失败，可以试一下pdb下载
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            success = _download_file(url, f"{file_dir}/{pdb_id}.pdb")
            if success:
                return ["pdb", pdb_id]
            else:
                print(f"无法下载PDB结构文件: {pdb_id}")
                return [None, pdb_id]
    except Exception as e:
        print(f"发生错误: {e}")
        return [None, pdb_id]
        
    

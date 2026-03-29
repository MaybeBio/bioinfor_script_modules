# 搭配pypaperflow工具抓取pubmed文献之后使用
# 1个简单的搭建文献语料背景知识库的方法，最简单最原始的方法，就是将所有要查询的文献合并成1个超级大的文本

def Parse_and_Merge_pubmed_meta(paper_dir, output_txt):
    """
    Description
    ----------
    在使用pypaperflow抓取pubmed文献meta数据之后, 遍历指定目录下的所有子目录(按年份和PMID组织), 读取每篇文章的meta JSON文件,
    将所有文章的meta信息提取并合并成一个统一的文本文件, 方便后续分析和阅读

    Args
    ----
    paper_dir: pypaperflow抓取文献的输出目录路径
    output_txt: 抓取文献meta数据合并输出的txt路径
    """

    import os
    import json

    count = 0
    with open(output_txt, "w") as out_f:
        # 按照年份排序遍历，避免乱序
        if not os.path.exists(paper_dir):
            print(f"错误: 目录不存在 {paper_dir}")
        else:
            year_list = sorted(os.listdir(paper_dir))
            for year in year_list:
                year_dir = os.path.join(paper_dir, year)
                
                # 跳过非文件夹项（如系统隐藏文件）
                if not os.path.isdir(year_dir):
                    continue
                    
                pmid_list = os.listdir(year_dir)
                for pmid in pmid_list:
                    pmid_dir = os.path.join(year_dir, pmid)
                    
                    # 再次检查是否为文件夹
                    if not os.path.isdir(pmid_dir):
                        continue
                        
                    meta_json = os.path.join(pmid_dir, f"{pmid}.json")
                    if not os.path.exists(meta_json):
                        continue
                    
                    try:
                        with open(meta_json, "r") as f:
                            meta_dict = json.load(f)

                        # --- 开始写入 ---
                        # 使用 .get(key, {}) 防止 content 为空导致报错
                        content_data = meta_dict.get('content', {})
                        identity_data = meta_dict.get('identity', {})
                        source_data = meta_dict.get('source', {})

                        out_f.write(f"PMID: {identity_data.get('pmid', 'N/A')}\n")
                        out_f.write(f"Title: {identity_data.get('title', 'N/A')}\n")
                        out_f.write(f"Journal: {source_data.get('journal_title', 'N/A')}\n")
                        out_f.write(f"Publication Date: {source_data.get('pub_date', 'N/A')}\n")
                        
                        out_f.write("Abstract:\n")
                        abstract = content_data.get('abstract')
                        
                        # 处理摘要为空的情况 (NoneType error)
                        if abstract:
                            for sentence in abstract.split('.'):
                                if sentence.strip():  # 避免空行
                                    out_f.write(sentence.strip() + '.\n')
                        else:
                            out_f.write("No Abstract Available.\n")
                            
                        out_f.write("Keywords:\n")
                        out_f.write(f"{content_data.get('keywords', [])}\n")
                        out_f.write("MeSH Terms:\n")
                        out_f.write(f"{content_data.get('mesh_terms', [])}\n")
                        out_f.write("\n" + "="*50 + "\n\n") # 加个显眼的分隔符
                        
                        count += 1
                        
                    except Exception as e:
                        print(f"⚠️ 处理 PMID {pmid} 时出错，已跳过。错误信息: {e}")

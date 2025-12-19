# 从文本中提取链接, 比如说是文献摘要/正文之类

import re

def extract_urls_from_text(text: str) -> List[Dict[str, str]]:
    """
    从文本中提取 URL，并尝试分类（GitHub, General, etc.）
    """
    if not text:
        return []

    # 匹配 URL 的正则 (比较宽松，能匹配大部分 http/https/ftp/www)
    url_pattern = r'(https?://[^\s,;>)]+|www\.[^\s,;>)]+|ftp://[^\s,;>)]+)'
    
    found_urls = re.findall(url_pattern, text)
    
    results = []
    seen = set() # 去重

    for url in found_urls:
        # 清洗末尾的标点符号 (比如句号结尾的 url.)
        url = url.rstrip('.')
        
        if url in seen:
            continue
        seen.add(url)

        # 简单分类
        category = "General"
        if "github.com" in url:
            category = "GitHub"
        elif "gitlab.com" in url:
            category = "GitLab"
        elif "zenodo.org" in url:
            category = "Zenodo"
        elif "figshare.com" in url:
            category = "Figshare"
        elif "huggingface.co" in url:
            category = "HuggingFace"

        results.append({
            "url": url,
            "source": "abstract_mining", # 标记来源
            "category": category
        })
    
    return results

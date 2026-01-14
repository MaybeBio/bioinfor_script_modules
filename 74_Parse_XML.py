# 解析xml
from bs4 import BeautifulSoup
import json
import bs4

def parse_section_recursive(sec_element):
    """
    递归解析章节，处理嵌套结构
    """
    section_data = {
        "title": "N/A",
        "content": [],   # 存放本层级的段落文本
        "subsections": [] # 存放子章节
    }

    # 1. 提取标题
    title_node = sec_element.find("title", recursive=False)
    if title_node:
        section_data["title"] = title_node.get_text().strip()

    # 2. 提取本层级的直接内容 (Paragraphs)
    # 使用 recursive=False 确保只获取当前层级的 p，不获取子章节里的 p
    direct_paragraphs = sec_element.find_all("p", recursive=False)
    section_data["content"] = [p.get_text().strip() for p in direct_paragraphs]

    # 3. 递归寻找子章节 (Sub-sections)
    sub_sections = sec_element.find_all("sec", recursive=False)
    for sub in sub_sections:
        # 递归调用
        child_data = parse_section_recursive(sub)
        section_data["subsections"].append(child_data)

    return section_data

def parse_pmc_xml_to_json(soup):
    
    paper_structure = {
        "title": soup.find('article-title').get_text() if soup.find('article-title') else "N/A",
        "body": []
    }

    body = soup.find('body')
    if body:
        # 只查找第一层级的 sec
        top_level_sections = body.find_all("sec", recursive=False)
        for sec in top_level_sections:
            paper_structure["body"].append(parse_section_recursive(sec))
            
    return paper_structure

# --- 使用示例 ---

# 假设 xml_str 是我们的原始 XML 数据
# xml_str = handle.read() 
# result = parse_pmc_xml_to_json(xml_str)

# 打印结果看看结构 (美化输出)
# print(json.dumps(result, indent=2, ensure_ascii=False))

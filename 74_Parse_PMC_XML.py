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
print(json.dumps(result, indent=2, ensure_ascii=False))

================================================================================================================================================

# 2, 
# 解析Entrez efetch返回的xml内容（pmc_full_xml）

soup = BeautifulSoup(pmc_full_xml, 'xml')
articles = soup.find_all('article')

for article in articles:
    parsed_json = self._parse_soup_to_json(article)

# 其中_parse_soup_to_json函数
def _parse_soup_to_json(self, article_soup: Any) -> Dict[str, Any]:
        """
        Parses a single <article> soup object into hierarchical JSON.
        """
        paper_structure = {
            "title": "N/A",
            "body": []
        }
        
        # Title
        title_node = article_soup.find('article-title')
        if title_node:
            paper_structure["title"] = title_node.get_text().strip()
            
        # Body
        body = article_soup.find('body')
        if body:
            # we only parse top-level sections here
            top_level_sections = body.find_all("sec", recursive=False)
            for sec in top_level_sections:
                paper_structure["body"].append(self._parse_section_recursive(sec))
                
        return paper_structure



====================================================================================================================================================

# 3,

def _parse_soup_to_json(self, article_soup: Any) -> Dict[str, Any]:
        """
        Parses a single <article> soup object into hierarchical JSON.
        """
        paper_structure = {
            "title": "N/A",
            "body": []
        }
        
        # Title
        title_node = article_soup.find('article-title')
        if title_node:
            paper_structure["title"] = title_node.get_text().strip()
            
        # Strategy: Parse Body First
        body = article_soup.find('body')
        sections_found = False
        
        if body:
            # 1. Try finding top-level sections (standard JATS)
            top_level_sections = body.find_all("sec", recursive=False)
            
            # 2. If no top-level sections, try finding ALL sections (loose structure)
            if not top_level_sections:
                 top_level_sections = body.find_all("sec")
            
            if top_level_sections:
                for sec in top_level_sections:
                    # Prevent parsing sub-sections as top-level if we used find_all(recursive=True)
                    # Simple heuristic: only parse if parent is body or parent is not another sec
                    if sec.parent.name == 'body' or sec.parent.name != 'sec':
                         paper_structure["body"].append(self._parse_section_recursive(sec))
                         sections_found = True
            
            # 3. If still no sections found (e.g. Letter to Editor), look for direct paragraphs
            if not sections_found:
                 direct_paragraphs = body.find_all("p", recursive=False)
                 if direct_paragraphs:
                     content = [p.get_text().strip() for p in direct_paragraphs]
                     paper_structure["body"].append({
                         "title": "Main Text",
                         "content": content,
                         "subsections": []
                     })
                     sections_found = True

        # Strategy: If Body is empty or missing, check Abstract for structured content
        # Many bioinformatics application notes put "Motivation", "Results" in Abstract
        if not sections_found:
            abstract = article_soup.find('abstract')
            if abstract:
                # Treat abstract sections as body sections
                abs_sections = abstract.find_all("sec")
                if abs_sections:
                    for sec in abs_sections:
                        paper_structure["body"].append(self._parse_section_recursive(sec))
                else:
                    # Flat abstract
                    abstract_text = abstract.get_text().strip()
                    if abstract_text:
                        paper_structure["body"].append({
                            "title": "Abstract-Only Content",
                            "content": [abstract_text],
                            "subsections": []
                        })

        return paper_structure






======================================================================================================================================================

# 4, 摘要要获取

    def _parse_soup_to_json(self, article_soup: Any) -> Dict[str, Any]:
        """
        Parses a single <article> soup object into hierarchical JSON.
        Now explicitly captures Abstract + Body.
        """
        paper_structure = {
            "title": "N/A",
            "body": []
        }
        
        # 1. Title
        title_node = article_soup.find('article-title')
        if title_node:
            paper_structure["title"] = title_node.get_text().strip()

        # 2. Abstract (Always try to fetch this first)
        # Abstract is usually under <front><article-meta><abstract>
        abstract_node = article_soup.find('abstract')
        if abstract_node:
            # Check if abstract has sections (structured abstract)
            abs_sections = abstract_node.find_all("sec")
            if abs_sections:
                # Structured abstract: add each sec
                for sec in abs_sections:
                    # Often abstract sections don't have titles in standard ways, handling varies
                    # But _parse_section_recursive handles <title> if present.
                    parsed_sec = self._parse_section_recursive(sec)
                    if parsed_sec["title"] == "N/A":
                        # Attempt to get a label if title is missing (common in some XMLs)
                         parsed_sec["title"] = "Abstract Section"
                    paper_structure["body"].append(parsed_sec)
            else:
                # Unstructured abstract: treat as one block
                # Get all text clean
                abs_text = abstract_node.get_text(separator=' ', strip=True)
                if abs_text:
                    paper_structure["body"].append({
                        "title": "Abstract",
                        "content": [abs_text], 
                        "subsections": []
                    })
            
        # 3. Body
        body = article_soup.find('body')
        sections_found = False
        
        if body:
            # 1. Try finding top-level sections (standard JATS)
            top_level_sections = body.find_all("sec", recursive=False)
            
            # 2. If no top-level sections, try finding ALL sections (loose structure)
            if not top_level_sections:
                 top_level_sections = body.find_all("sec")
            
            if top_level_sections:
                for sec in top_level_sections:
                    # Prevent parsing sub-sections as top-level if we used find_all(recursive=True)
                    if sec.parent and sec.parent.name == 'sec':
                        continue 

                    paper_structure["body"].append(self._parse_section_recursive(sec))
                    sections_found = True
            
            # 3. If still no sections found (e.g. Letter to Editor), look for direct paragraphs
            if not sections_found:
                 direct_paragraphs = body.find_all("p", recursive=False)
                 if direct_paragraphs:
                     content = [p.get_text().strip() for p in direct_paragraphs]
                     paper_structure["body"].append({
                         "title": "Main Text", 
                         "content": content,
                         "subsections": []
                     })
                     sections_found = True

        # Fallback: If no body and no abstract was found earlier (very rare empty paper)
        if not sections_found and not abstract_node:
             # Try to see if there is anything at all?
             pass

        return paper_structure




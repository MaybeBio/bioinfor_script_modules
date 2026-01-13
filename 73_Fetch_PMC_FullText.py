# db="pmc", 获取其中文本数据

# 此处只使用retmode="xml", 不使用retmode="text", rettype="medline" 以及Medline.parse
# 因为后者无法获取全文数据, 还是只能获取一些元数据
'''
from Bio import Medline
handle = Entrez.efetch(db="pmc", id=results[0]['LinkSetDb'][0]['Link'][0]['Id'], retmode="text", rettype="medline")
records = Medline.parse(handle)
records = list(records)
handle.close()
records
'''

handle = Entrez.efetch(db="pmc", id="OUR_PMC_ID", retmode="xml")
# handle句柄是一个流, 读过一次指针就到了末尾, 所以下面的两种读法只能选一种
xml1 = handle.read()
xml2 = Entrez.read(handle)
handle.close()

# 1, 对于Entrez.read返回的对象, 也就是xml2

def parse_biopython_object(xml_record):
    """
    解析 Biopython 解析后的字典对象 (xml2)
    """
    print(f"=== 基于 Biopython 对象 (xml2) 的全文解析 ===")
    
    # xml2 通常是一个列表，取第一个记录
    if isinstance(xml_record, list):
        record = xml_record[0]
    else:
        record = xml_record

    # 递归函数
    def print_section_dict(sec_dict, level=1):
        # 1. 获取标题
        # Biopython 中 title 可能是字符串，也可能是包含格式的复杂对象，用 str() 强转
        title = sec_dict.get('title', 'No Title')
        print(f"\n{'#' * level} {str(title)}")
        
        # 2. 获取段落 (注意：所有的 p 都在这里，顺序可能与原文不同如果它们被其他标签因为打断)
        if 'p' in sec_dict:
            for p in sec_dict['p']:
                # Biopython 的 StringElement 可能包含属性，转为字符串
                print(f"  [P] {str(p)
                               }")

        # 3. 递归子章节
        if 'sec' in sec_dict:
            for sub_sec in sec_dict['sec']:
                print_section_dict(sub_sec, level + 1)

    # 入口检查
    if 'body' in record:
        body = record['body']
        if 'sec' in body:
            for section in body['sec']:
                print_section_dict(section)
        else:
            print("Body 中未发现顶级 sec 结构")
    else:
        print("记录中未发现 body 部分")

# 假设 xml2 是 Entrez.read() 返回的对象
parse_biopython_object(xml2)

================================================================================================================================================================

# 2, 对于.read()返回的对象

import xml.etree.ElementTree as ET

def parse_raw_xml_content(xml_string):
    """
    解析原始 XML 字符串，提取分层全文
    """
    try:
        # 1. 解析 XML 字符串
        root = ET.fromstring(xml_string)
        
        # 2. 寻找 article-body 节点 (JATS 格式通常是 body)
        # 注意: 这里的 find 路径可能需要根据实际 XML 命名空间调整，这里用通用写法
        body = root.find(".//body")
        
        if body is None:
            print("未找到 Body 节点")
            return

        print(f"=== 基于原始 XML (xml1) 的全文解析 ===")

        # 定义递归函数处理章节
        def process_element(element, level=0):
            # 如果是章节 (sec)
            if element.tag == 'sec':
                # 提取标题
                title_node = element.find('title')
                title_text = "".join(title_node.itertext()) if title_node is not None else "无标题章节"
                print(f"\n{'#' * (level + 1)} {title_text}")
                
            # 如果是段落 (p)
            elif element.tag == 'p':
                # itertext() 能自动处理嵌套标签，如 "This is <bold>bold</bold> text" -> "This is bold text"
                text = "".join(element.itertext()).strip()
                if text:
                    print(f"  [P] {text[:100]}...") # 只打印前100字预览

            # 递归遍历所有子节点 (保持原文顺序)
            # 这一点比 Biopython 对象优势巨大，Biopython 分离了不同类型的标签
            for child in element:
                # 如果是 sec，递归层级+1；如果是 p 或其他，层级不变
                next_level = level + 1 if element.tag == 'sec' else level
                process_element(child, next_level)

        # 开始遍历 body 的直系子节点
        for child in body:
            process_element(child)
            
    except Exception as e:
        print(f"解析出错: {e}")

# 假设 xml1 是我们的字符串数据
parse_raw_xml_content(xml1) 


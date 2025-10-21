# 获取蛋白质的注释信息, 比如说是功能结构域、翻译后修饰位点等等

# 1, 从uniprot数据库扒取, 主要是借助rest api, 从请求返回的json格式数据中解包

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import requests,time

@dataclass(frozen=True)
class UniProt_Feature:
    """
    Description
    ----------
        UniProt特征类, 存储从UniProt数据库获取的蛋白质特征信息

    Args
    ----------
        feature_type (str): 特征类型, 例如'DOMAIN','REGION','MOTIF'等
        description (str): 特征描述信息
        start (int): 特征起始位置(1-based)
        end (int): 特征结束位置(1-based)
    """
    feature_type: str
    start: int
    end: int
    description: Optional[str] = None

@dataclass(frozen=True)
class UniProt_Feature_Analysis_Result:
    """
    Description
    ----------
        UniProt特征分析结果类, 存储从UniProt数据库获取的蛋白质特征信息列表

    Args
    ----------
        features (List[UniProt_Feature]): UniProt特征列表

    Returns
    ----------
        Dict[str, List[UniProt_Feature]]: 以特征类型为键, 特征列表为值的字典, 也就是features按类型分类存储;
        所以同一类型的feature会被存储在同一个列表中
    """
    features: Dict[str, List[UniProt_Feature]]


def get_uniprot_features(protein_id: str) -> UniProt_Feature_Analysis_Result:
    """
    Description
    ----------
        从UniProt数据库获取指定蛋白质的特征信息

    Args
    ----------
        protein_id (str): UniProt蛋白质ID

    Returns
    ----------
        UniProt_Feature_Analysis_Result: UniProt特征分析结果对象
    """
    url = f"https://rest.uniprot.org/uniprotkb/{protein_id}"
    # 指定返回JSON格式
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers, timeout=10)
    if response.status_code == 200:
        data = response.json()
        
        # 初始化features字典, feature_type作为键, 对应的UniProt_Feature列表作为值
        features: Dict[str, List[UniProt_Feature]] = {}
        for feature in data.get("features", []):
            f_type = feature.get("type")
            f_description = feature.get("description", "")
            f_start = feature["location"]["start"]["value"]
            f_end = feature["location"]["end"]["value"]
            uni_feature = UniProt_Feature(
                feature_type=f_type,
                start=f_start,
                end=f_end,
                description=f_description
            )
            # 如果该feature_type还未在字典中, 则初始化一个空列表
            if f_type not in features:
                features[f_type] = []
            # 将当前feature添加到对应类型的列表中
            features[f_type].append(uni_feature)
        return UniProt_Feature_Analysis_Result(features=features)


# 示例分析:
result = get_uniprot_features("P49711")
for ftype, flist in result.features.items():
    print(f"Feature Type: {ftype}")
    for feature in flist:
        print(f"  Start: {feature.start}, End: {feature.end}, Description: {feature.description}")


=====================================================================================================================

# 2, 深度使用api, 
# 参考:https://www.uniprot.org/help/sequence_annotation
# 参考: https://www.uniprot.org/help/general_annotation

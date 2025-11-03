# 递归展平字典, 举例见method1里的example

# 1
from typing import Dict, Any
def flatten_dict_to_featureDict(input_dict: Dict) -> Dict[str, Any]:
    """
    Description
    ----------
        将一个嵌套的字典展开为一个扁平化的字典, 适用于将复杂的嵌套字典转换为机器学习特征字典;

    Args
    ----------
        input_dict (Dict): 输入的嵌套字典;

    Returns
    ----------
        Dict: 扁平化后的字典;

    Notes
    ----------
    - 1, 为了防止展开之后的键值对冲突, 也为了便于识别新键值对的来源, 展开之后的键名使用短横线号"_"链接各级键名;
    比如说, 输入字典为 {"a": 1, "b": {"c": 2, "d": {"e": 3}}}, 则展开后的字典为 {"a": 1, "b_c": 2, "b_d_e": 3};
    - 2, 本函数假设输入字典的值要么是标量数值类型, 要么是字典类型, 不考虑其他复杂数据类型;除了字典类型之外的值均视为标量数值类型, 后续可以在展平之后再进行数据类型转换(提取之类)


    Example
    ----------
    >>> nested_dict = {
    ...     "a": 1,
    ...     "b": {
    ...         "c": 2,
    ...         "d": {
    ...             "e": 3
    ...         }
    ...     }
    ... }
    >>> flat_dict = flatten_dict_to_featureDict(nested_dict)
    >>> print(flat_dict)
    {'a': 1, 'b_c': 2, 'b_d_e': 3}
    """

    # 初始化新字典
    flat_dict = {}
    for key, value in input_dict.items():
        # 首先判断子值是否为字典
        if isinstance(value, dict):
            # 如果是则递归调用本函数然后展开为子字典
            sub_dict = flatten_dict_to_featureDict(value)
            # 假设我们获取了子字典sub_dict, 现在需要完善一下子字典的键名
            for sub_key, sub_value in sub_dict.items():
                # 使用下划线链接键名, 存到新字典中
                flat_dict[f"{key}_{sub_key}"] = sub_value
        # 如果不是字典, 暂时认为是scalar数值类型就直接添加
        else:
            flat_dict[key] = value
    return flat_dict

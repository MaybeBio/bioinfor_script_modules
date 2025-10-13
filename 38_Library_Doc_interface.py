# 主要是小众python库的文档查看
# 此处以aiupred_lib为例

import sys
import os
import inspect  # 核心：用于提取对象信息和文档

# 导入目标库
import aiupred_lib  # 确保这行不报错，否则需检查路径或库安装

def get_object_doc(obj, obj_name: str) -> str:
    """提取对象的文档信息（docstring + 参数列表）"""
    # 1. 提取docstring（若没有文档，返回提示）
    docstring = inspect.getdoc(obj) or "⚠️ 无官方文档"
    
    # 2. 提取参数列表（针对函数/方法）
    try:
        # 获取签名（参数名、默认值等）
        sig = inspect.signature(obj)
        params_info = f"参数列表: {str(sig)}"
    except (ValueError, TypeError):
        # 非函数/方法（如类本身），无参数列表
        params_info = "❌ 非可调用对象，无参数列表"
    
    # 3. 组合文档信息
    return f"""
【{obj_name}】
{params_info}
------------------------------
文档说明:
{docstring}
======================================================================"""


def is_callable_member(member) -> bool:
    """过滤：仅保留「可调用对象」（函数、类、类方法等），排除普通变量"""
    # 排除内置特殊成员（如 __name__、__doc__）
    if inspect.ismodule(member):
        return False  # 排除子模块（若库包含子模块，可按需调整）
    # 保留：函数、类、实例方法、类方法、静态方法
    return (inspect.isfunction(member) 
            or inspect.isclass(member) 
            or inspect.ismethod(member) 
            or inspect.ismethoddescriptor(member))


def print_all_methods_with_docs(library) -> None:
    """主函数：输出库中所有可调用成员及其文档"""
    print(f"=== 开始输出 {library.__name__} 库的所有方法及文档 ===")
    print(f"库路径: {library.__file__}\n")

    # 1. 获取库的所有成员名称（过滤掉内置特殊成员，如 __init__）
    all_member_names = [name for name in dir(library) if not name.startswith("__")]

    # 2. 遍历成员，分类处理
    for member_name in all_member_names:
        # 获取成员对象（如函数、类）
        member = getattr(library, member_name)
        
        # 过滤：仅处理可调用对象
        if not is_callable_member(member):
            continue

        # 场景A：成员是「类」（需进一步输出类的内部方法）
        if inspect.isclass(member):
            print(f"📦 类: {member_name}")
            print(get_object_doc(member, member_name))  # 输出类本身的文档
            
            # 提取类的所有方法（过滤内置特殊方法）
            class_methods = [
                meth_name for meth_name in dir(member) 
                if not meth_name.startswith("__")
            ]
            for meth_name in class_methods:
                meth = getattr(member, meth_name)
                if is_callable_member(meth):  # 确保是方法（而非类变量）
                    print(f"  🔧 方法: {member_name}.{meth_name}")
                    print(get_object_doc(meth, f"{member_name}.{meth_name}"))

        # 场景B：成员是「顶层函数」（直接输出）
        else:
            print(f"🔧 顶层函数: {member_name}")
            print(get_object_doc(member, member_name))

    print(f"=== {library.__name__} 库所有方法及文档输出完毕 ===")


# 执行主函数，输出结果
if __name__ == "__main__":
    print_all_methods_with_docs(aiupred_lib)

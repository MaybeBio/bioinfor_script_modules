# 对小众第三方库的属性、方法以及文档的提取

import inspect
import importlib
import sys
import os
from typing import Any, List, Dict, Optional

def inspect_library(
    library_name: str,
    output_path: Optional[str] = None,
    include_private: bool = False,
    include_imported: bool = False
):
    """
    Dynamically inspect a Python library and generate a documentation report.
    
    Args:
        library_name: The import name of the library (e.g., 'numpy', 'Bio.PDB').
        output_path: Path to save the Markdown report. If None, prints to stdout.
        include_private: Whether to include members starting with '_'.
        include_imported: Whether to include members imported from other modules.
    """
    
    # 1. 动态导入库
    try:
        module = importlib.import_module(library_name)
    except ImportError as e:
        print(f"❌ Error: Could not import library '{library_name}'. Reason: {e}")
        return
    except Exception as e:
        print(f"❌ Error: An unexpected error occurred while importing '{library_name}': {e}")
        return

    lines = []
    lines.append(f"# Documentation for `{library_name}`")
    lines.append(f"**File Path:** `{getattr(module, '__file__', 'Built-in/Unknown')}`\n")
    
    doc = inspect.getdoc(module)
    if doc:
        lines.append("## Module Docstring")
        lines.append(f"```text\n{doc}\n```\n")

    lines.append("## Contents")

    # 2. 获取所有成员
    # 使用 dir() 获取所有名称，然后 getattr 获取对象
    # 优先检查 __all__ 属性（如果存在）, 一般是公开API, 否则使用 dir()
    if hasattr(module, "__all__"):
        all_names = module.__all__
        using_all = True
    else:
        all_names = dir(module)
        using_all = False
    
    members_data = []

    for name in all_names:
        if not include_private and not using_all and name.startswith("_"):
            continue
        
        try:
            obj = getattr(module, name)
        except AttributeError:
            continue

        # 过滤掉从其他模块导入的成员（除非指定包含）
        if not include_imported:
            # __module__ 属性指示对象所属的模块, 通过检查它是否与库名匹配来判断
            # __module__ 是python中对象的一个特殊属性, 核心作用是记录对象最初被定义(创建)的模块名称
            # 简单来说，__module__相当于某一个对象的出生证明, 无论这个对象被导入到哪个模块、被传递到哪里，__module__永远指向它"原本所在的模块", 不会随导入/传递行为改变 
            # 如果 obj.__module__ 不以 library_name 开头，说明它是导入的
            obj_module = getattr(obj, "__module__", None)
            if obj_module and not obj_module.startswith(library_name):
                # 特殊情况：如果使用了 __all__，通常意味着作者希望导出它，即使它是导入的
                if not using_all:
                    continue

        members_data.append((name, obj))




        # 通过检查 __module__ 属性
        obj_module = getattr(obj, "__module__", None)
        if not include_imported and obj_module and not obj_module.startswith(library_name):
             # 特殊处理：有些库会在 __init__ 中暴露子模块，这种通常需要保留
             pass 

        members_data.append((name, obj))

    # 3. 分类处理 (Classes vs Functions vs Others)
    classes = []
    functions = []
    others = []

    for name, obj in members_data:
        if inspect.isclass(obj):
            classes.append((name, obj))
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
            functions.append((name, obj))
        else:
            others.append((name, obj))

    # --- Helper: 获取签名和文档 ---
    def get_info(obj):
        try:
            sig = str(inspect.signature(obj))
        except (ValueError, TypeError):
            # 尝试获取内置函数的签名
            sig = getattr(obj, "__text_signature__", "(...)")
            if sig is None: sig = "(...)"
        
        doc = inspect.getdoc(obj) or "No documentation available."
        return sig, doc

    # 4. 生成 Markdown 内容
    
    # --- Functions ---
    if functions:
        lines.append("### 🔧 Functions")
        for name, func in functions:
            sig, doc = get_info(func)
            lines.append(f"#### `{name}{sig}`")
            lines.append(f"> {doc.splitlines()[0] if doc else ''}") # 仅显示第一行简介
            lines.append(f"<details><summary>Full Docstring</summary>\n\n```text\n{doc}\n```\n</details>\n")

    # --- Classes ---
    if classes:
        lines.append("### 📦 Classes")
        for name, cls in classes:
            sig, doc = get_info(cls)
            lines.append(f"#### `class {name}{sig}`")
            lines.append(f"{doc.splitlines()[0] if doc else ''}\n")
            
            # Inspect Class Methods
            methods = inspect.getmembers(cls, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x))
            if methods:
                lines.append("| Method | Signature | Description |")
                lines.append("| :--- | :--- | :--- |")
                for m_name, m_obj in methods:
                    if not include_private and m_name.startswith("_") and m_name != "__init__":
                        continue
                    m_sig, m_doc = get_info(m_obj)
                    short_doc = m_doc.splitlines()[0] if m_doc else "-"
                    # Escape pipes for markdown table
                    short_doc = short_doc.replace("|", "\|")
                    lines.append(f"| **{m_name}** | `{m_sig}` | {short_doc} |")
            lines.append("\n")

    # --- Output ---
    content = "\n".join(lines)
    
    if output_path:
        # 自动补全后缀
        if not output_path.endswith(".md"):
            output_path += ".md"
        
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Documentation saved to: {os.path.abspath(output_path)}")
        except IOError as e:
            print(f"❌ Error writing file: {e}")
    else:
        # Print to console (simplified)
        print(content)

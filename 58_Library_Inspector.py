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


=========================================================================================================================================================

# 2,
# 新增网络重要性分析

import inspect
import importlib
import sys
import os
import pkgutil
from collections import Counter, defaultdict
from typing import Any, List, Dict, Optional, Tuple

def inspect_library(
    library_name: str,
    output_path: Optional[str] = None,
    include_private: bool = False,
    include_imported: bool = False
):
    """
    Dynamically inspect a Python library, analyze dependencies, and generate a report.
    """
    
    # 1. 动态导入主库
    try:
        main_module = importlib.import_module(library_name)
    except ImportError as e:
        print(f"❌ Error: Could not import library '{library_name}'. Reason: {e}")
        return
    except Exception as e:
        print(f"❌ Error: An unexpected error occurred while importing '{library_name}': {e}")
        return

    lines = []
    lines.append(f"# Documentation for `{library_name}`")
    lines.append(f"**File Path:** `{getattr(main_module, '__file__', 'Built-in/Unknown')}`\n")
    
    doc = inspect.getdoc(main_module)
    if doc:
        lines.append("## Module Docstring")
        lines.append(f"```text\n{doc}\n```\n")

    # ==========================================
    # Phase 1: Dependency & Importance Analysis (The "PageRank" Logic)
    # ==========================================
    print(f"🔍 Analyzing dependencies for '{library_name}' (this may take a moment)...")
    
    internal_modules_rank = Counter() # 内部模块重要性（被内部其他模块引用的次数）
    external_libs_rank = Counter()    # 外部库重要性（被内部模块引用的次数）
    dependency_graph = defaultdict(set) # 记录谁引用了谁: graph[importer] = {imported_1, imported_2}

    # 获取所有子模块列表
    submodules = [main_module]
    if hasattr(main_module, "__path__"):
        # 递归查找所有子模块
        for importer, modname, ispkg in pkgutil.walk_packages(main_module.__path__, main_module.__name__ + "."):
            try:
                # 尝试导入子模块以分析其依赖
                sub_mod = importlib.import_module(modname)
                submodules.append(sub_mod)
            except Exception:
                # 某些模块可能因为缺少依赖或环境问题无法导入，跳过
                continue

    # 分析每个模块的 imports
    for mod in submodules:
        current_mod_name = mod.__name__
        
        # 检查该模块的全局变量（即 imports 和定义的类/函数）
        for name, obj in inspect.getmembers(mod):
            # 获取对象的定义模块
            obj_module = getattr(obj, "__module__", None)
            
            if not obj_module:
                continue
            
            # 忽略自身引用
            if obj_module == current_mod_name:
                continue

            # 记录依赖关系
            dependency_graph[current_mod_name].add(obj_module)

            if obj_module.startswith(library_name):
                # 这是一个内部引用 (Internal Dependency)
                # 比如在 Bio.PDB 中引用了 Bio.File
                # 我们只记录模块级别的引用，避免统计过于细碎
                internal_modules_rank[obj_module] += 1
            else:
                # 这是一个外部引用 (External Dependency)
                # 提取顶级包名，例如 'numpy.core.multiarray' -> 'numpy'
                top_level_pkg = obj_module.split('.')[0]
                # 排除标准库中的一些常见干扰项（可选）
                if top_level_pkg not in ['builtins', 'sys', 'os', 'typing']:
                    external_libs_rank[top_level_pkg] += 1

    # --- 生成分析报告 ---
    lines.append("## 📊 Architecture & Importance Analysis")
    lines.append("Based on import frequency across all submodules (PageRank-lite).")

    # 1. 外部依赖排行
    lines.append("### 🌍 Top External Dependencies")
    lines.append("Which 3rd-party libraries does this project rely on the most?")
    if external_libs_rank:
        lines.append("| Library | Usage Count | Importance Bar |")
        lines.append("| :--- | :--- | :--- |")
        for lib, count in external_libs_rank.most_common(10):
            bar = "█" * (count // 2 if count > 1 else 1) # 简单的 ASCII 条形图
            lines.append(f"| **{lib}** | {count} | `{bar}` |")
    else:
        lines.append("_No significant external dependencies detected._")
    lines.append("\n")

    # 2. 内部核心模块排行
    lines.append("### 🧠 Core Internal Modules")
    lines.append("These modules are heavily imported by other parts of the library. They likely contain the core logic/utilities.")
    if internal_modules_rank:
        lines.append("| Internal Module | In-Degree (Refs) | Importance Bar |")
        lines.append("| :--- | :--- | :--- |")
        for mod, count in internal_modules_rank.most_common(10):
            # 简化显示，去掉公共前缀
            short_name = mod.replace(library_name + ".", "")
            bar = "▓" * (count // 2 if count > 1 else 1)
            lines.append(f"| **{short_name}** | {count} | `{bar}` |")
    else:
        lines.append("_No internal cross-references detected._")
    lines.append("\n")

    # 3. 可视化 (Mermaid Graph)
    lines.append("### 🕸️ Dependency Visualization")
    lines.append("Copy the code below into a Mermaid live editor or view in GitHub/VSCode.")
    
    mermaid_lines = ["graph TD"]
    # 为了避免图表过大，只显示最重要的连接
    top_internal = set(x[0] for x in internal_modules_rank.most_common(15))
    
    for source, targets in dependency_graph.items():
        # 只显示源头是核心模块，或者目标是核心模块的关系
        if source not in top_internal and len(targets.intersection(top_internal)) == 0:
            continue
            
        short_source = source.replace(library_name + ".", "")
        # 限制节点名称长度
        short_source = short_source.split('.')[-1] if '.' in short_source else short_source
        
        for target in targets:
            if target.startswith(library_name):
                if target in top_internal:
                    short_target = target.split('.')[-1]
                    mermaid_lines.append(f"    {short_source} --> {short_target}")
            else:
                # 外部库只显示前几名
                top_pkg = target.split('.')[0]
                if external_libs_rank[top_pkg] > 2: # 阈值：引用超过2次才显示
                    mermaid_lines.append(f"    {short_source} -.-> {top_pkg}[{top_pkg}]")

    lines.append("<details><summary>Show Mermaid Graph</summary>\n")
    lines.append("```mermaid")
    lines.append("\n".join(mermaid_lines))
    lines.append("```\n</details>\n")


    # ==========================================
    # Phase 2: Surface Level Inspection (Original Logic)
    # ==========================================
    lines.append("## 📑 Top-Level API Contents")

    # 2. 获取所有成员
    if hasattr(main_module, "__all__"):
        all_names = main_module.__all__
        using_all = True
    else:
        all_names = dir(main_module)
        using_all = False
    
    members_data = []

    for name in all_names:
        if not include_private and not using_all and name.startswith("_"):
            continue
        
        try:
            obj = getattr(main_module, name)
        except AttributeError:
            continue

        # 过滤掉从其他模块导入的成员（除非指定包含）
        obj_module = getattr(obj, "__module__", None)
        is_imported = False
        if obj_module and not obj_module.startswith(library_name):
            is_imported = True
        
        if not include_imported and is_imported:
             # 特殊处理：如果使用了 __all__，通常意味着作者希望导出它
             if not using_all:
                 continue

        members_data.append((name, obj, is_imported))

    # 3. 分类处理
    classes = []
    functions = []
    
    for name, obj, is_imported in members_data:
        # 标记导入的成员
        display_name = name + (" (imported)" if is_imported else "")
        
        if inspect.isclass(obj):
            classes.append((display_name, obj))
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
            functions.append((display_name, obj))

    # --- Helper: 获取签名和文档 ---
    def get_info(obj):
        try:
            sig = str(inspect.signature(obj))
        except (ValueError, TypeError):
            sig = getattr(obj, "__text_signature__", "(...)")
            if sig is None: sig = "(...)"
        
        doc = inspect.getdoc(obj) or "No documentation available."
        return sig, doc

    # 4. 生成 Markdown 内容 (Functions & Classes)
    if functions:
        lines.append("### 🔧 Functions")
        for name, func in functions:
            sig, doc = get_info(func)
            lines.append(f"#### `{name}{sig}`")
            lines.append(f"> {doc.splitlines()[0] if doc else ''}")
            lines.append(f"<details><summary>Full Docstring</summary>\n\n```text\n{doc}\n```\n</details>\n")

    if classes:
        lines.append("### 📦 Classes")
        for name, cls in classes:
            sig, doc = get_info(cls)
            lines.append(f"#### `class {name}{sig}`")
            lines.append(f"{doc.splitlines()[0] if doc else ''}\n")
            
            methods = inspect.getmembers(cls, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x))
            if methods:
                lines.append("| Method | Signature | Description |")
                lines.append("| :--- | :--- | :--- |")
                for m_name, m_obj in methods:
                    if not include_private and m_name.startswith("_") and m_name != "__init__":
                        continue
                    m_sig, m_doc = get_info(m_obj)
                    short_doc = m_doc.splitlines()[0] if m_doc else "-"
                    short_doc = short_doc.replace("|", "\|")
                    lines.append(f"| **{m_name}** | `{m_sig}` | {short_doc} |")
            lines.append("\n")

    # --- Output ---
    content = "\n".join(lines)
    
    if output_path:
        if not output_path.endswith(".md"):
            output_path += ".md"
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Documentation saved to: {os.path.abspath(output_path)}")
        except IOError as e:
            print(f"❌ Error writing file: {e}")
    else:
        print(content)


========================================================================================================================================================

# 3, 网络分析, 但是简单的库没效果

import inspect
import importlib
import sys
import os
import pkgutil
from collections import Counter, defaultdict
from typing import Any, List, Dict, Optional, Tuple

# 尝试导入 networkx 进行高级网络分析
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

def inspect_library(
    library_name: str,
    output_path: Optional[str] = None,
    include_private: bool = False,
    include_imported: bool = False
):
    """
    Dynamically inspect a Python library, analyze dependencies using Network Analysis, and generate a report.
    """
    
    # --- 1. 动态导入主库 (带 sys.argv 保护) ---
    _old_argv = sys.argv
    sys.argv = [sys.argv[0]]

    submodules = []
    main_module = None

    try:
        try:
            main_module = importlib.import_module(library_name)
            submodules.append(main_module)
        except ImportError as e:
            print(f"❌ Error: Could not import library '{library_name}'. Reason: {e}")
            return
        except Exception as e:
            print(f"❌ Error: An unexpected error occurred while importing '{library_name}': {e}")
            return

        print(f"🔍 Analyzing dependencies for '{library_name}' (Network Analysis Phase)...")
        
        if hasattr(main_module, "__path__"):
            for importer, modname, ispkg in pkgutil.walk_packages(main_module.__path__, main_module.__name__ + "."):
                try:
                    sub_mod = importlib.import_module(modname)
                    submodules.append(sub_mod)
                except Exception:
                    continue
    finally:
        sys.argv = _old_argv

    lines = []
    lines.append(f"# Documentation for `{library_name}`")
    lines.append(f"**File Path:** `{getattr(main_module, '__file__', 'Built-in/Unknown')}`\n")
    
    doc = inspect.getdoc(main_module)
    if doc:
        lines.append("## Module Docstring")
        lines.append(f"```text\n{doc}\n```\n")

    # ==========================================
    # Phase 1: Network Construction & Analysis
    # ==========================================
    
    # 使用 NetworkX 构建有向图
    # 节点：模块名
    # 边：引用关系 (Importer -> Imported)
    G = nx.DiGraph() if HAS_NETWORKX else None
    
    internal_modules_rank = Counter() 
    external_libs_rank = Counter()    
    dependency_graph = defaultdict(set) 

    for mod in submodules:
        current_mod_name = mod.__name__
        if HAS_NETWORKX:
            G.add_node(current_mod_name, type='internal')
        
        for name, obj in inspect.getmembers(mod):
            obj_module = getattr(obj, "__module__", None)
            
            if not obj_module: continue
            if obj_module == current_mod_name: continue

            # 记录基础数据
            dependency_graph[current_mod_name].add(obj_module)

            # 区分内部和外部
            if obj_module.startswith(library_name):
                internal_modules_rank[obj_module] += 1
                if HAS_NETWORKX:
                    G.add_edge(current_mod_name, obj_module)
            else:
                top_level_pkg = obj_module.split('.')[0]
                if top_level_pkg not in ['builtins', 'sys', 'os', 'typing']:
                    external_libs_rank[top_level_pkg] += 1
                    if HAS_NETWORKX:
                        # 外部库作为节点加入，标记为 external
                        G.add_node(top_level_pkg, type='external')
                        G.add_edge(current_mod_name, top_level_pkg)

    lines.append("## 📊 Network & Architecture Analysis")
    
    if not HAS_NETWORKX:
        lines.append("> ⚠️ `networkx` is not installed. Advanced metrics (PageRank, Centrality) are disabled.")
        lines.append("> Install it via `pip install networkx` to see them.\n")

    # --- 1. 外部依赖 (Sinks) ---
    lines.append("### 🌍 Top External Dependencies")
    if external_libs_rank:
        lines.append("| Library | Usage Count |")
        lines.append("| :--- | :--- |")
        for lib, count in external_libs_rank.most_common(10):
            lines.append(f"| **{lib}** | {count} |")
    else:
        lines.append("_No significant external dependencies._")
    lines.append("\n")

    # --- 2. 网络指标分析 (Network Metrics) ---
    if HAS_NETWORKX and len(G.nodes) > 0:
        lines.append("### 🕸️ Network Metrics (Advanced)")
        lines.append("Using Graph Theory to identify critical components.")

        # A. PageRank (权威性)
        try:
            pagerank = nx.pagerank(G, alpha=0.85)
            sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            
            lines.append("#### 👑 Top Modules by PageRank (Authority)")
            lines.append("These modules are the 'most important' based on the network structure. If core modules rely on them, they get a higher score.")
            lines.append("| Rank | Module | Score | Type |")
            lines.append("| :--- | :--- | :--- | :--- |")
            
            for i, (node, score) in enumerate(sorted_pr[:10]):
                node_type = "Internal" if node.startswith(library_name) else "External"
                short_name = node.replace(library_name + ".", "")
                lines.append(f"| {i+1} | `{short_name}` | {score:.4f} | {node_type} |")
            lines.append("\n")
        except Exception as e:
            lines.append(f"> Could not calculate PageRank: {e}\n")

        # B. Betweenness Centrality (桥梁/枢纽)
        try:
            # 只计算内部子图的介数中心性，看看谁是内部的“胶水”
            internal_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'internal']
            sub_G = G.subgraph(internal_nodes)
            betweenness = nx.betweenness_centrality(sub_G)
            sorted_bt = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

            lines.append("#### 🌉 Top Bridges (Betweenness Centrality)")
            lines.append("These modules act as 'bridges' connecting different parts of the library. They control information flow.")
            lines.append("| Rank | Module | Score |")
            lines.append("| :--- | :--- | :--- |")
            
            count = 0
            for node, score in sorted_bt:
                if score == 0: continue # 忽略没有桥接作用的
                if count >= 8: break
                short_name = node.replace(library_name + ".", "")
                lines.append(f"| {count+1} | `{short_name}` | {score:.4f} |")
                count += 1
            if count == 0:
                lines.append("_No significant bridge nodes detected (flat structure)._")
            lines.append("\n")

        except Exception as e:
            lines.append(f"> Could not calculate Betweenness: {e}\n")

    # --- 3. 可视化 (Mermaid) ---
    lines.append("### 🗺️ Dependency Map")
    
    mermaid_lines = ["graph TD"]
    # 定义样式类
    mermaid_lines.append("    classDef core fill:#f96,stroke:#333,stroke-width:2px;")
    mermaid_lines.append("    classDef external fill:#9cf,stroke:#333,stroke-width:1px;")
    mermaid_lines.append("    classDef normal fill:#fff,stroke:#333,stroke-width:1px;")

    # 筛选要显示的节点（避免图太大）
    # 策略：显示 PageRank 前 20 的节点 + 它们的一级连接
    if HAS_NETWORKX:
        top_nodes = set(n for n, s in sorted_pr[:20])
    else:
        top_nodes = set(x[0] for x in internal_modules_rank.most_common(20))

    # 构建 Mermaid 边
    edges_to_draw = set()
    
    # 遍历图中的边
    source_data = G.edges() if HAS_NETWORKX else []
    if not HAS_NETWORKX:
        # 回退逻辑
        for src, targets in dependency_graph.items():
            for tgt in targets:
                source_data.append((src, tgt))

    for u, v in source_data:
        # 过滤：只显示涉及 Top 节点的边
        if u in top_nodes or v in top_nodes:
            # 简化名称
            short_u = u.replace(library_name + ".", "").split('.')[-1]
            short_v = v.replace(library_name + ".", "").split('.')[-1]
            
            # 处理外部库名称
            if not v.startswith(library_name):
                short_v = v.split('.')[0] # 只取 numpy, 不取 numpy.core
            
            # 避免自环
            if short_u == short_v: continue
            
            # 生成唯一的边 ID 防止重复
            edge_id = f"{short_u}->{short_v}"
            if edge_id in edges_to_draw: continue
            edges_to_draw.add(edge_id)

            # 决定箭头样式
            arrow = "-.->" if not v.startswith(library_name) else "-->"
            mermaid_lines.append(f"    {short_u}{arrow}{short_v}")
            
            # 样式应用
            if u in top_nodes and u.startswith(library_name):
                mermaid_lines.append(f"    class {short_u} core;")
            elif not u.startswith(library_name):
                mermaid_lines.append(f"    class {short_u} external;")
            
            if v in top_nodes and v.startswith(library_name):
                mermaid_lines.append(f"    class {short_v} core;")
            elif not v.startswith(library_name):
                mermaid_lines.append(f"    class {short_v} external;")

    lines.append("<details><summary>Show Mermaid Graph</summary>\n")
    lines.append("```mermaid")
    lines.append("\n".join(mermaid_lines))
    lines.append("```\n</details>\n")

    # ==========================================
    # Phase 2: Surface Level Inspection
    # ==========================================
    lines.append("## 📑 Top-Level API Contents")

    if hasattr(main_module, "__all__"):
        all_names = main_module.__all__
        using_all = True
    else:
        all_names = dir(main_module)
        using_all = False
    
    members_data = []

    for name in all_names:
        if not include_private and not using_all and name.startswith("_"):
            continue
        
        try:
            obj = getattr(main_module, name)
        except AttributeError:
            continue

        obj_module = getattr(obj, "__module__", None)
        is_imported = False
        if obj_module and not obj_module.startswith(library_name):
            is_imported = True
        
        if not include_imported and is_imported:
             if not using_all:
                 continue

        members_data.append((name, obj, is_imported))

    classes = []
    functions = []
    
    for name, obj, is_imported in members_data:
        display_name = name + (" (imported)" if is_imported else "")
        
        if inspect.isclass(obj):
            classes.append((display_name, obj))
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
            functions.append((display_name, obj))

    def get_info(obj):
        try:
            sig = str(inspect.signature(obj))
        except (ValueError, TypeError):
            sig = getattr(obj, "__text_signature__", "(...)")
            if sig is None: sig = "(...)"
        
        doc = inspect.getdoc(obj) or "No documentation available."
        return sig, doc

    if functions:
        lines.append("### 🔧 Functions")
        for name, func in functions:
            sig, doc = get_info(func)
            lines.append(f"#### `{name}{sig}`")
            lines.append(f"> {doc.splitlines()[0] if doc else ''}")
            lines.append(f"<details><summary>Full Docstring</summary>\n\n```text\n{doc}\n```\n</details>\n")

    if classes:
        lines.append("### 📦 Classes")
        for name, cls in classes:
            sig, doc = get_info(cls)
            lines.append(f"#### `class {name}{sig}`")
            lines.append(f"{doc.splitlines()[0] if doc else ''}\n")
            
            methods = inspect.getmembers(cls, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x))
            if methods:
                lines.append("| Method | Signature | Description |")
                lines.append("| :--- | :--- | :--- |")
                for m_name, m_obj in methods:
                    if not include_private and m_name.startswith("_") and m_name != "__init__":
                        continue
                    m_sig, m_doc = get_info(m_obj)
                    short_doc = m_doc.splitlines()[0] if m_doc else "-"
                    short_doc = short_doc.replace("|", "\\|")
                    lines.append(f"| **{m_name}** | `{m_sig}` | {short_doc} |")
            lines.append("\n")

    # --- Output ---
    content = "\n".join(lines)
    
    if output_path:
        if not output_path.endswith(".md"):
            output_path += ".md"
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                print(f"❌ Error creating directory {output_dir}: {e}")
                return

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Documentation saved to: {os.path.abspath(output_path)}")
        except IOError as e:
            print(f"❌ Error writing file: {e}")
    else:
        print(content)



==============================================================================================================================================

# 4, 类继承的需求新增

import inspect
import importlib
import sys
import os
import pkgutil
from collections import Counter, defaultdict
from typing import Any, List, Dict, Optional, Tuple

# 尝试导入 networkx 进行高级网络分析
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

def inspect_library(
    library_name: str,
    output_path: Optional[str] = None,
    include_private: bool = False,
    include_imported: bool = False
):
    """
    Dynamically inspect a Python library, analyze dependencies using Network Analysis, and generate a report.
    """
    
    # --- 1. 动态导入主库 (带 sys.argv 保护) ---
    _old_argv = sys.argv
    sys.argv = [sys.argv[0]]

    submodules = []
    main_module = None

    try:
        try:
            main_module = importlib.import_module(library_name)
            submodules.append(main_module)
        except ImportError as e:
            print(f"❌ Error: Could not import library '{library_name}'. Reason: {e}")
            return
        except Exception as e:
            print(f"❌ Error: An unexpected error occurred while importing '{library_name}': {e}")
            return

        print(f"🔍 Analyzing dependencies for '{library_name}' (Network Analysis Phase)...")
        
        if hasattr(main_module, "__path__"):
            for importer, modname, ispkg in pkgutil.walk_packages(main_module.__path__, main_module.__name__ + "."):
                try:
                    sub_mod = importlib.import_module(modname)
                    submodules.append(sub_mod)
                except Exception:
                    continue
    finally:
        sys.argv = _old_argv

    lines = []
    lines.append(f"# Documentation for `{library_name}`")
    lines.append(f"**File Path:** `{getattr(main_module, '__file__', 'Built-in/Unknown')}`\n")
    
    doc = inspect.getdoc(main_module)
    if doc:
        lines.append("## Module Docstring")
        lines.append(f"```text\n{doc}\n```\n")

    # ==========================================
    # Phase 1: Network Construction & Analysis
    # ==========================================
    
    G = nx.DiGraph() if HAS_NETWORKX else None
    
    internal_modules_rank = Counter() 
    external_libs_rank = Counter()    
    dependency_graph = defaultdict(set) 

    for mod in submodules:
        current_mod_name = mod.__name__
        if HAS_NETWORKX:
            G.add_node(current_mod_name, type='internal')
        
        for name, obj in inspect.getmembers(mod):
            obj_module = getattr(obj, "__module__", None)
            
            if not obj_module: continue
            if obj_module == current_mod_name: continue

            dependency_graph[current_mod_name].add(obj_module)

            if obj_module.startswith(library_name):
                internal_modules_rank[obj_module] += 1
                if HAS_NETWORKX:
                    G.add_edge(current_mod_name, obj_module)
            else:
                top_level_pkg = obj_module.split('.')[0]
                if top_level_pkg not in ['builtins', 'sys', 'os', 'typing']:
                    external_libs_rank[top_level_pkg] += 1
                    if HAS_NETWORKX:
                        G.add_node(top_level_pkg, type='external')
                        G.add_edge(current_mod_name, top_level_pkg)

    lines.append("## 📊 Network & Architecture Analysis")
    
    if not HAS_NETWORKX:
        lines.append("> ⚠️ `networkx` is not installed. Advanced metrics are disabled.\n")

    # --- 1. 外部依赖 ---
    lines.append("### 🌍 Top External Dependencies")
    if external_libs_rank:
        lines.append("| Library | Usage Count |")
        lines.append("| :--- | :--- |")
        for lib, count in external_libs_rank.most_common(10):
            lines.append(f"| **{lib}** | {count} |")
    else:
        lines.append("_No significant external dependencies._")
    lines.append("\n")

    # --- 2. 网络指标分析 ---
    if HAS_NETWORKX and len(G.nodes) > 0:
        lines.append("### 🕸️ Network Metrics (Advanced)")
        
        # PageRank
        try:
            pagerank = nx.pagerank(G, alpha=0.85)
            sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            
            lines.append("#### 👑 Top Modules by PageRank (Authority)")
            lines.append("| Rank | Module | Score | Type |")
            lines.append("| :--- | :--- | :--- | :--- |")
            
            for i, (node, score) in enumerate(sorted_pr[:10]):
                node_type = "Internal" if node.startswith(library_name) else "External"
                short_name = node.replace(library_name + ".", "")
                lines.append(f"| {i+1} | `{short_name}` | {score:.4f} | {node_type} |")
            lines.append("\n")
        except Exception:
            pass

        # Betweenness
        try:
            internal_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'internal']
            if len(internal_nodes) > 2: # 只有节点够多时才计算介数
                sub_G = G.subgraph(internal_nodes)
                betweenness = nx.betweenness_centrality(sub_G)
                sorted_bt = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)

                lines.append("#### 🌉 Top Bridges (Betweenness Centrality)")
                lines.append("| Rank | Module | Score |")
                lines.append("| :--- | :--- | :--- |")
                
                count = 0
                for node, score in sorted_bt:
                    if score == 0: continue
                    if count >= 8: break
                    short_name = node.replace(library_name + ".", "")
                    lines.append(f"| {count+1} | `{short_name}` | {score:.4f} |")
                    count += 1
                if count == 0:
                    lines.append("_No significant bridge nodes detected (flat structure)._\n")
            else:
                lines.append("_Structure is too simple (single module) for Betweenness Centrality analysis._\n")

        except Exception:
            pass

    # --- 3. 可视化 (Mermaid) ---
    lines.append("### 🗺️ Dependency & Architecture Map")
    
    mermaid_lines = ["graph TD"]
    mermaid_lines.append("    classDef core fill:#f96,stroke:#333,stroke-width:2px;")
    mermaid_lines.append("    classDef external fill:#9cf,stroke:#333,stroke-width:1px;")
    mermaid_lines.append("    classDef normal fill:#fff,stroke:#333,stroke-width:1px;")

    # 判断是否为简单结构（单文件）
    is_simple_structure = len(submodules) < 2

    if is_simple_structure:
        lines.append("> ℹ️ **Structure Note:** This library appears to be a single module. The graph below visualizes **Class Inheritance** to show internal architecture.\n")
    
    # 筛选节点
    if HAS_NETWORKX:
        top_nodes = set(n for n, s in sorted_pr[:20])
    else:
        top_nodes = set(x[0] for x in internal_modules_rank.most_common(20))

    edges_to_draw = set()
    
    # A. 绘制模块依赖 (原逻辑)
    source_data = G.edges() if HAS_NETWORKX else []
    if not HAS_NETWORKX:
        for src, targets in dependency_graph.items():
            for tgt in targets:
                source_data.append((src, tgt))

    for u, v in source_data:
        if u in top_nodes or v in top_nodes:
            short_u = u.replace(library_name + ".", "").split('.')[-1]
            short_v = v.replace(library_name + ".", "").split('.')[-1]
            if not v.startswith(library_name): short_v = v.split('.')[0]
            
            if short_u == short_v: continue
            edge_id = f"{short_u}->{short_v}"
            if edge_id in edges_to_draw: continue
            edges_to_draw.add(edge_id)

            arrow = "-.->" if not v.startswith(library_name) else "-->"
            mermaid_lines.append(f"    {short_u}{arrow}{short_v}")
            
            # 样式
            if u.startswith(library_name): mermaid_lines.append(f"    class {short_u} core;")
            else: mermaid_lines.append(f"    class {short_u} external;")
            
            if v.startswith(library_name): mermaid_lines.append(f"    class {short_v} core;")
            else: mermaid_lines.append(f"    class {short_v} external;")

    # B. (新增) 绘制类继承关系 (针对单文件模块增强)
    if is_simple_structure:
        for mod in submodules:
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                # 只分析定义在当前库中的类
                if getattr(obj, "__module__", "").startswith(library_name):
                    for base in obj.__bases__:
                        base_name = base.__name__
                        if base_name == 'object': continue
                        
                        # 绘制继承箭头: Class --|> Base
                        mermaid_lines.append(f"    {name} --|> {base_name}")
                        mermaid_lines.append(f"    class {name} core;")
                        
                        # 如果基类是外部的（比如 torch.nn.Module），标记为 external
                        if base.__module__.split('.')[0] != library_name:
                            mermaid_lines.append(f"    class {base_name} external;")
                        else:
                            mermaid_lines.append(f"    class {base_name} core;")

    lines.append("<details><summary>Show Mermaid Graph</summary>\n")
    lines.append("```mermaid")
    lines.append("\n".join(mermaid_lines))
    lines.append("```\n</details>\n")

    # ==========================================
    # Phase 2: Surface Level Inspection
    # ==========================================
    lines.append("## 📑 Top-Level API Contents")

    if hasattr(main_module, "__all__"):
        all_names = main_module.__all__
        using_all = True
    else:
        all_names = dir(main_module)
        using_all = False
    
    members_data = []

    for name in all_names:
        if not include_private and not using_all and name.startswith("_"):
            continue
        
        try:
            obj = getattr(main_module, name)
        except AttributeError:
            continue

        obj_module = getattr(obj, "__module__", None)
        is_imported = False
        if obj_module and not obj_module.startswith(library_name):
            is_imported = True
        
        if not include_imported and is_imported:
             if not using_all:
                 continue

        members_data.append((name, obj, is_imported))

    classes = []
    functions = []
    
    for name, obj, is_imported in members_data:
        display_name = name + (" (imported)" if is_imported else "")
        
        if inspect.isclass(obj):
            classes.append((display_name, obj))
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
            functions.append((display_name, obj))

    def get_info(obj):
        try:
            sig = str(inspect.signature(obj))
        except (ValueError, TypeError):
            sig = getattr(obj, "__text_signature__", "(...)")
            if sig is None: sig = "(...)"
        
        doc = inspect.getdoc(obj) or "No documentation available."
        return sig, doc

    if functions:
        lines.append("### 🔧 Functions")
        for name, func in functions:
            sig, doc = get_info(func)
            lines.append(f"#### `{name}{sig}`")
            lines.append(f"> {doc.splitlines()[0] if doc else ''}")
            lines.append(f"<details><summary>Full Docstring</summary>\n\n```text\n{doc}\n```\n</details>\n")

    if classes:
        lines.append("### 📦 Classes")
        for name, cls in classes:
            sig, doc = get_info(cls)
            lines.append(f"#### `class {name}{sig}`")
            lines.append(f"{doc.splitlines()[0] if doc else ''}\n")
            
            methods = inspect.getmembers(cls, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x))
            if methods:
                lines.append("| Method | Signature | Description |")
                lines.append("| :--- | :--- | :--- |")
                for m_name, m_obj in methods:
                    if not include_private and m_name.startswith("_") and m_name != "__init__":
                        continue
                    m_sig, m_doc = get_info(m_obj)
                    short_doc = m_doc.splitlines()[0] if m_doc else "-"
                    short_doc = short_doc.replace("|", "\\|")
                    lines.append(f"| **{m_name}** | `{m_sig}` | {short_doc} |")
            lines.append("\n")

    # --- Output ---
    content = "\n".join(lines)
    
    if output_path:
        if not output_path.endswith(".md"):
            output_path += ".md"
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                print(f"❌ Error creating directory {output_dir}: {e}")
                return

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Documentation saved to: {os.path.abspath(output_path)}")
        except IOError as e:
            print(f"❌ Error writing file: {e}")
    else:
        print(content)



===================================================================================================================================================

# 5, 类继承, 变量处理逻辑

import inspect
import importlib
import sys
import os
import pkgutil
import ast  # 新增：用于静态代码分析
from collections import Counter, defaultdict
from typing import Any, List, Dict, Optional, Tuple, Set

# 尝试导入 networkx 进行高级网络分析
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# --- Helper: AST 分析器，用于提取函数内部逻辑 ---
class FunctionFlowVisitor(ast.NodeVisitor):
    def __init__(self):
        self.calls = []      # 调用的函数
        self.assignments = [] # 变量赋值
        self.returns = []    # 返回值

    def visit_Call(self, node):
        # 提取函数调用，例如 self.model(x) 或 np.array(x)
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = getattr(node.func.value, 'id', 'obj') + "." + node.func.attr
        
        if func_name:
            # 尝试获取参数名，用于展示数据流
            args = []
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    args.append(arg.id)
            self.calls.append((func_name, args))
        self.generic_visit(node)

    def visit_Assign(self, node):
        # 提取赋值，例如 y = f(x)
        targets = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                targets.append(target.id)
        
        # 如果赋值的右边是一个函数调用
        if isinstance(node.value, ast.Call):
            self.visit(node.value) # 让 visit_Call 处理右边
            # 关联最后一次调用到这个变量 (简化处理)
            if self.calls:
                last_call, args = self.calls[-1]
                self.assignments.append((targets, last_call))
        self.generic_visit(node)

    def visit_Return(self, node):
        if isinstance(node.value, ast.Name):
            self.returns.append(node.value.id)
        elif isinstance(node.value, ast.Tuple):
            self.returns.append("Tuple(...)")
        else:
            self.returns.append("Expression")
        self.generic_visit(node)

def generate_function_flowchart(func_obj) -> str:
    """
    使用 AST 分析函数源码，生成 Mermaid 流程图代码
    """
    try:
        source = inspect.getsource(func_obj)
        # 去除缩进，否则 ast.parse 会报错
        source = inspect.cleandoc(source)
        tree = ast.parse(source)
    except (OSError, TypeError, IndentationError, SyntaxError):
        return ""

    visitor = FunctionFlowVisitor()
    visitor.visit(tree)

    # 如果函数太简单（没有调用也没有赋值），就不画图了
    if not visitor.calls and not visitor.assignments:
        return ""

    # 构建 Mermaid
    lines = ["flowchart LR"]
    
    # 1. 输入参数
    sig = inspect.signature(func_obj)
    params = list(sig.parameters.keys())
    if params:
        lines.append(f"    Input[Input: {', '.join(params)}]:::input")
    
    # 2. 逻辑流
    # 简化策略：按顺序连接调用
    prev_node = "Input" if params else None
    
    step_idx = 0
    for func_name, args in visitor.calls:
        step_id = f"Step{step_idx}"
        arg_str = f"({', '.join(args)})" if args else ""
        
        # 检查这个调用是否被赋值给了变量
        assigned_var = None
        for targets, call_name in visitor.assignments:
            if call_name == func_name:
                assigned_var = ", ".join(targets)
                break
        
        label = f"{func_name}{arg_str}"
        if assigned_var:
            label += f"<br/>⬇<br/>{assigned_var}"
            
        lines.append(f"    {step_id}({label}):::process")
        
        if prev_node:
            lines.append(f"    {prev_node} --> {step_id}")
        prev_node = step_id
        step_idx += 1

    # 3. 返回值
    if visitor.returns:
        ret_label = ", ".join(visitor.returns)
        lines.append(f"    Return([Return: {ret_label}]):::output")
        if prev_node:
            lines.append(f"    {prev_node} --> Return")

    # 样式定义
    lines.append("    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;")
    lines.append("    classDef process fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;")
    lines.append("    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,rx:10,ry:10;")
    
    return "\n".join(lines)


def inspect_library(
    library_name: str,
    output_path: Optional[str] = None,
    include_private: bool = False,
    include_imported: bool = False
):
    """
    Dynamically inspect a Python library, analyze dependencies using Network Analysis, and generate a report.
    """
    
    # --- 1. 动态导入主库 (带 sys.argv 保护) ---
    _old_argv = sys.argv
    sys.argv = [sys.argv[0]]

    submodules = []
    main_module = None

    try:
        try:
            main_module = importlib.import_module(library_name)
            submodules.append(main_module)
        except ImportError as e:
            print(f"❌ Error: Could not import library '{library_name}'. Reason: {e}")
            return
        except Exception as e:
            print(f"❌ Error: An unexpected error occurred while importing '{library_name}': {e}")
            return

        print(f"🔍 Analyzing dependencies for '{library_name}' (Network Analysis Phase)...")
        
        if hasattr(main_module, "__path__"):
            for importer, modname, ispkg in pkgutil.walk_packages(main_module.__path__, main_module.__name__ + "."):
                try:
                    sub_mod = importlib.import_module(modname)
                    submodules.append(sub_mod)
                except Exception:
                    continue
    finally:
        sys.argv = _old_argv

    lines = []
    lines.append(f"# Documentation for `{library_name}`")
    lines.append(f"**File Path:** `{getattr(main_module, '__file__', 'Built-in/Unknown')}`\n")
    
    doc = inspect.getdoc(main_module)
    if doc:
        lines.append("## Module Docstring")
        lines.append(f"```text\n{doc}\n```\n")

    # ==========================================
    # Phase 1: Network Construction & Analysis
    # ==========================================
    
    G = nx.DiGraph() if HAS_NETWORKX else None
    
    internal_modules_rank = Counter() 
    external_libs_rank = Counter()    
    dependency_graph = defaultdict(set) 

    for mod in submodules:
        current_mod_name = mod.__name__
        if HAS_NETWORKX:
            G.add_node(current_mod_name, type='internal')
        
        for name, obj in inspect.getmembers(mod):
            obj_module = getattr(obj, "__module__", None)
            
            if not obj_module: continue
            if obj_module == current_mod_name: continue

            dependency_graph[current_mod_name].add(obj_module)

            if obj_module.startswith(library_name):
                internal_modules_rank[obj_module] += 1
                if HAS_NETWORKX:
                    G.add_edge(current_mod_name, obj_module)
            else:
                top_level_pkg = obj_module.split('.')[0]
                if top_level_pkg not in ['builtins', 'sys', 'os', 'typing']:
                    external_libs_rank[top_level_pkg] += 1
                    if HAS_NETWORKX:
                        G.add_node(top_level_pkg, type='external')
                        G.add_edge(current_mod_name, top_level_pkg)

    lines.append("## 📊 Network & Architecture Analysis")
    
    if not HAS_NETWORKX:
        lines.append("> ⚠️ `networkx` is not installed. Advanced metrics are disabled.\n")

    # --- 1. 外部依赖 ---
    lines.append("### 🌍 Top External Dependencies")
    if external_libs_rank:
        lines.append("| Library | Usage Count |")
        lines.append("| :--- | :--- |")
        for lib, count in external_libs_rank.most_common(10):
            lines.append(f"| **{lib}** | {count} |")
    else:
        lines.append("_No significant external dependencies._")
    lines.append("\n")

    # --- 2. 网络指标分析 ---
    if HAS_NETWORKX and len(G.nodes) > 0:
        lines.append("### 🕸️ Network Metrics (Advanced)")
        
        # PageRank
        try:
            pagerank = nx.pagerank(G, alpha=0.85)
            sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            
            lines.append("#### 👑 Top Modules by PageRank (Authority)")
            lines.append("| Rank | Module | Score | Type |")
            lines.append("| :--- | :--- | :--- | :--- |")
            
            for i, (node, score) in enumerate(sorted_pr[:10]):
                node_type = "Internal" if node.startswith(library_name) else "External"
                short_name = node.replace(library_name + ".", "")
                lines.append(f"| {i+1} | `{short_name}` | {score:.4f} | {node_type} |")
            lines.append("\n")
        except Exception:
            pass

    # --- 3. 可视化 (Mermaid) ---
    lines.append("### 🗺️ Dependency & Architecture Map")
    
    mermaid_lines = ["graph TD"]
    mermaid_lines.append("    classDef core fill:#f96,stroke:#333,stroke-width:2px;")
    mermaid_lines.append("    classDef external fill:#9cf,stroke:#333,stroke-width:1px;")
    
    # 筛选节点
    if HAS_NETWORKX:
        top_nodes = set(n for n, s in sorted_pr[:20])
    else:
        top_nodes = set(x[0] for x in internal_modules_rank.most_common(20))

    edges_to_draw = set()
    
    # A. 绘制模块依赖
    source_data = G.edges() if HAS_NETWORKX else []
    if not HAS_NETWORKX:
        for src, targets in dependency_graph.items():
            for tgt in targets:
                source_data.append((src, tgt))

    for u, v in source_data:
        if u in top_nodes or v in top_nodes:
            short_u = u.replace(library_name + ".", "").split('.')[-1]
            short_v = v.replace(library_name + ".", "").split('.')[-1]
            if not v.startswith(library_name): short_v = v.split('.')[0]
            
            if short_u == short_v: continue
            edge_id = f"{short_u}->{short_v}"
            if edge_id in edges_to_draw: continue
            edges_to_draw.add(edge_id)

            arrow = "-.->" if not v.startswith(library_name) else "-->"
            mermaid_lines.append(f"    {short_u}{arrow}{short_v}")
            
            if u.startswith(library_name): mermaid_lines.append(f"    class {short_u} core;")
            else: mermaid_lines.append(f"    class {short_u} external;")
            
            if v.startswith(library_name): mermaid_lines.append(f"    class {short_v} core;")
            else: mermaid_lines.append(f"    class {short_v} external;")

    # B. (增强) 绘制类继承关系 - 对所有模块生效
    # 收集所有类
    all_classes = []
    for mod in submodules:
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if getattr(obj, "__module__", "").startswith(library_name):
                all_classes.append((name, obj))

    # 如果类不是特别多，就画出来
    if len(all_classes) < 50: 
        for name, obj in all_classes:
            for base in obj.__bases__:
                base_name = base.__name__
                if base_name == 'object': continue
                
                # 绘制继承箭头: Class --|> Base
                mermaid_lines.append(f"    {name} --|> {base_name}")
                mermaid_lines.append(f"    class {name} core;")
                
                if base.__module__.split('.')[0] != library_name:
                    mermaid_lines.append(f"    class {base_name} external;")
                else:
                    mermaid_lines.append(f"    class {base_name} core;")

    lines.append("<details><summary>Show Mermaid Graph</summary>\n")
    lines.append("```mermaid")
    lines.append("\n".join(mermaid_lines))
    lines.append("```\n</details>\n")

    # ==========================================
    # Phase 2: Surface Level Inspection & Logic Flow
    # ==========================================
    lines.append("## 📑 Top-Level API Contents & Logic Flow")

    if hasattr(main_module, "__all__"):
        all_names = main_module.__all__
        using_all = True
    else:
        all_names = dir(main_module)
        using_all = False
    
    members_data = []

    for name in all_names:
        if not include_private and not using_all and name.startswith("_"):
            continue
        
        try:
            obj = getattr(main_module, name)
        except AttributeError:
            continue

        obj_module = getattr(obj, "__module__", None)
        is_imported = False
        if obj_module and not obj_module.startswith(library_name):
            is_imported = True
        
        if not include_imported and is_imported:
             if not using_all:
                 continue

        members_data.append((name, obj, is_imported))

    classes = []
    functions = []
    
    for name, obj, is_imported in members_data:
        display_name = name + (" (imported)" if is_imported else "")
        
        if inspect.isclass(obj):
            classes.append((display_name, obj))
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
            functions.append((display_name, obj))

    def get_info(obj):
        try:
            sig = str(inspect.signature(obj))
        except (ValueError, TypeError):
            sig = getattr(obj, "__text_signature__", "(...)")
            if sig is None: sig = "(...)"
        
        doc = inspect.getdoc(obj) or "No documentation available."
        return sig, doc

    if functions:
        lines.append("### 🔧 Functions")
        for name, func in functions:
            sig, doc = get_info(func)
            lines.append(f"#### `{name}{sig}`")
            lines.append(f"> {doc.splitlines()[0] if doc else ''}")
            
            # --- 新增：逻辑流可视化 ---
            flow_chart = generate_function_flowchart(func)
            if flow_chart:
                lines.append("\n**Logic Flow:**")
                lines.append("```mermaid")
                lines.append(flow_chart)
                lines.append("```\n")
            
            lines.append(f"<details><summary>Full Docstring</summary>\n\n```text\n{doc}\n```\n</details>\n")

    if classes:
        lines.append("### 📦 Classes")
        for name, cls in classes:
            sig, doc = get_info(cls)
            lines.append(f"#### `class {name}{sig}`")
            lines.append(f"{doc.splitlines()[0] if doc else ''}\n")
            
            methods = inspect.getmembers(cls, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x))
            if methods:
                lines.append("| Method | Signature | Description |")
                lines.append("| :--- | :--- | :--- |")
                for m_name, m_obj in methods:
                    if not include_private and m_name.startswith("_") and m_name != "__init__":
                        continue
                    m_sig, m_doc = get_info(m_obj)
                    short_doc = m_doc.splitlines()[0] if m_doc else "-"
                    short_doc = short_doc.replace("|", "\\|")
                    lines.append(f"| **{m_name}** | `{m_sig}` | {short_doc} |")
            lines.append("\n")

    # --- Output ---
    content = "\n".join(lines)
    
    if output_path:
        if not output_path.endswith(".md"):
            output_path += ".md"
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                print(f"❌ Error creating directory {output_dir}: {e}")
                return

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Documentation saved to: {os.path.abspath(output_path)}")
        except IOError as e:
            print(f"❌ Error writing file: {e}")
    else:
        print(content)

======================================================================================================================================================

# 6, 暂时保留的版本

import inspect
import importlib
import sys
import os
import pkgutil
import ast
from collections import Counter, defaultdict
from typing import Any, List, Dict, Optional, Tuple, Set
import json

# 尝试导入 networkx 进行高级网络分析
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# --- Helper: 1. 单函数逻辑分析 (微观) ---
# (保留之前的 LogicNode, AdvancedFlowVisitor, generate_function_flowchart 代码，此处省略以节省篇幅，请保留原有的类定义)
# ...existing code...
class LogicNode:
    """表示流程图中的一个节点"""
    def __init__(self, id, label, node_type="process"):
        self.id = id
        self.label = label
        self.node_type = node_type # input, process, output
        self.edges_in = [] # List of (source_id, var_name)

class AdvancedFlowVisitor(ast.NodeVisitor):
    """
    解析函数源码，构建数据流向图。
    追踪变量的 生产(Definition) -> 消费(Usage) 链条。
    """
    def __init__(self):
        self.nodes = []
        self.current_producers = {} # var_name -> node_id (记录当前变量是由哪个节点产生的)
        self.counter = 0

    def _get_id(self):
        self.counter += 1
        return f"Node{self.counter}"

    def _resolve_inputs(self, input_vars: List[str]) -> List[Tuple[str, str]]:
        """查找输入变量的来源节点"""
        edges = []
        for var in input_vars:
            if var in self.current_producers:
                source_id = self.current_producers[var]
                edges.append((source_id, var))
        return edges

    def _extract_names(self, node) -> List[str]:
        """从 AST 节点中提取所有变量名 (用于查找输入)"""
        names = []
        class NameCollector(ast.NodeVisitor):
            def visit_Name(self, n):
                if isinstance(n.ctx, ast.Load):
                    names.append(n.id)
            def visit_Attribute(self, n):
                # 尝试捕获 self.xxx
                if isinstance(n.value, ast.Name) and n.value.id == 'self':
                    names.append(f"self.{n.attr}")
                self.generic_visit(n)
        
        if node:
            NameCollector().visit(node)
        return list(set(names)) # 去重

    def visit_FunctionDef(self, node):
        # 1. 处理输入参数 (Input Node)
        args = []
        arg_labels = []
        
        # 提取参数和类型注解
        all_args = node.args.args + node.args.kwonlyargs
        if node.args.vararg: all_args.append(node.args.vararg)
        if node.args.kwarg: all_args.append(node.args.kwarg)

        for arg in all_args:
            var_name = arg.arg
            args.append(var_name)
            
            # 尝试获取类型注解
            ann = ""
            if arg.annotation:
                try:
                    if hasattr(ast, 'unparse'):
                        ann = ": " + ast.unparse(arg.annotation)
                    else:
                        ann = ": " + str(arg.annotation)
                except: pass
            arg_labels.append(f"{var_name}{ann}")
            
        if args:
            node_id = "Input"
            # Mermaid 节点标签
            label = "Input\\n" + "\\n".join(arg_labels)
            logic_node = LogicNode(node_id, label, node_type="input")
            self.nodes.append(logic_node)
            
            # 注册这些变量的生产者为 Input 节点
            for arg in args:
                self.current_producers[arg] = node_id
                # 同时也注册 self.arg (针对 __init__ 这种常见模式的简化处理)
                if 'self' in args:
                    self.current_producers[f"self.{arg}"] = node_id
        
        # 继续遍历函数体
        for item in node.body:
            self.visit(item)

    def visit_Assign(self, node):
        self._handle_assign(node, node.targets)

    def visit_AnnAssign(self, node):
        # 处理带类型的赋值: x: int = value
        if node.value:
            self._handle_assign(node, [node.target], annotation=node.annotation)

    def _handle_assign(self, node, targets, annotation=None):
        # 1. 分析输入 (右值)
        input_vars = self._extract_names(node.value)
        
        # 2. 确定操作标签 (Label)
        label = "Assign"
        if isinstance(node.value, ast.Call):
            func_name = self._get_func_name(node.value)
            label = f"Call: {func_name}"
        elif isinstance(node.value, ast.BinOp):
            op = type(node.value.op).__name__
            label = f"Op: {op}"
        elif isinstance(node.value, ast.Constant):
             label = f"Const: {node.value.value}"
        
        # 3. 分析输出 (左值)
        outputs = []
        output_labels = []
        for target in targets:
            if isinstance(target, ast.Name):
                var_name = target.id
                outputs.append(var_name)
                
                ann_str = ""
                if annotation and hasattr(ast, 'unparse'):
                    try: ann_str = ": " + ast.unparse(annotation)
                    except: pass
                output_labels.append(f"{var_name}{ann_str}")
            elif isinstance(target, ast.Attribute):
                # 处理 self.x = ...
                if isinstance(target.value, ast.Name) and target.value.id == 'self':
                    var_name = f"self.{target.attr}"
                    outputs.append(var_name)
                    output_labels.append(var_name)

        if outputs:
            node_id = self._get_id()
            full_label = f"{label}\\n⬇\\n{', '.join(output_labels)}"
            
            logic_node = LogicNode(node_id, full_label)
            # 建立连线：找到输入变量的上一个生产者
            logic_node.edges_in = self._resolve_inputs(input_vars)
            
            self.nodes.append(logic_node)
            
            # 更新生产者表
            for out in outputs:
                self.current_producers[out] = node_id

    def visit_Expr(self, node):
        # 处理独立的函数调用 (无赋值)，例如 print(), model.eval()
        if isinstance(node.value, ast.Call):
            input_vars = self._extract_names(node.value)
            func_name = self._get_func_name(node.value)
            
            node_id = self._get_id()
            logic_node = LogicNode(node_id, f"Call: {func_name}")
            logic_node.edges_in = self._resolve_inputs(input_vars)
            
            self.nodes.append(logic_node)
            # 这种调用通常有副作用，但没有显式返回值变量，所以不更新 current_producers

    def visit_Return(self, node):
        input_vars = []
        ret_str = "None"
        if node.value:
            input_vars = self._extract_names(node.value)
            if hasattr(ast, 'unparse'):
                try: ret_str = ast.unparse(node.value)
                except: pass
            else:
                ret_str = "Expression"
        
        node_id = "Return"
        logic_node = LogicNode(node_id, f"Return\\n{ret_str}", node_type="output")
        logic_node.edges_in = self._resolve_inputs(input_vars)
        self.nodes.append(logic_node)

    def _get_func_name(self, node):
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return getattr(node.func.value, 'id', 'obj') + "." + node.func.attr
        return "func"

def generate_function_flowchart(func_obj) -> str:
    """
    使用高级 AST 分析生成 Mermaid 数据流图
    """
    try:
        source = inspect.getsource(func_obj)
        source = inspect.cleandoc(source)
        tree = ast.parse(source)
    except (OSError, TypeError, IndentationError, SyntaxError):
        return ""

    visitor = AdvancedFlowVisitor()
    visitor.visit(tree)

    if not visitor.nodes:
        return ""

    # 构建 Mermaid
    lines = ["flowchart TD"] # 使用自顶向下布局，适合展示流程
    
    # 样式定义
    lines.append("    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;")
    lines.append("    classDef process fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;")
    lines.append("    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,rx:10,ry:10;")
    
    # 绘制节点
    for node in visitor.nodes:
        # 转义标签中的引号
        safe_label = node.label.replace('"', "'")
        
        shape_start, shape_end = "(", ")"
        if node.node_type == "input": shape_start, shape_end = "[", "]"
        if node.node_type == "output": shape_start, shape_end = "([", "])"
        
        lines.append(f'    {node.id}{shape_start}"{safe_label}"{shape_end}:::{node.node_type}')
        
        # 绘制连线
        for source_id, var_name in node.edges_in:
            # 在连线上显示变量名，展示数据流动
            lines.append(f"    {source_id} -- {var_name} --> {node.id}")

    return "\n".join(lines)

# --- Helper: 2. 全局调用图分析 (宏观) ---

class GlobalCallGraphVisitor(ast.NodeVisitor):
    """
    分析整个模块的 AST，构建函数之间的调用关系图。
    """
    def __init__(self, known_functions: Set[str]):
        self.known_functions = known_functions # 库中定义的所有函数名集合
        self.calls = [] # List of (caller, callee, arg_names)
        self.current_function = "Main_Script" # 默认为顶层脚本

    def visit_FunctionDef(self, node):
        prev_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = prev_function

    def visit_Call(self, node):
        # 提取被调用的函数名
        callee_name = ""
        if isinstance(node.func, ast.Name):
            callee_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # 处理 self.method() 或 module.func()
            callee_name = node.func.attr
        
        if callee_name:
            # 提取参数名 (用于展示数据流)
            args = []
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    args.append(arg.id)
            
            # 只有当被调用的函数是我们库里的函数时，才记录（避免画出 print, len 等内置函数）
            # 或者如果它是 self.xxx 调用，我们也记录（假设是类内部调用）
            if callee_name in self.known_functions or (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self'):
                self.calls.append((self.current_function, callee_name, args))
        
        self.generic_visit(node)

def generate_global_call_graph(modules: List[Any], library_name: str) -> str:
    """
    生成全局函数调用图 (Global Call Graph)
    """
    # 1. 收集所有定义的函数名 (建立白名单)
    known_functions = set()
    for mod in modules:
        for name, obj in inspect.getmembers(mod):
            if inspect.isfunction(obj) or inspect.ismethod(obj):
                known_functions.add(name)
            elif inspect.isclass(obj):
                for m_name, m_obj in inspect.getmembers(obj):
                    if inspect.isfunction(m_obj) or inspect.ismethod(m_obj):
                        known_functions.add(m_name)

    # 2. 遍历所有源码进行 AST 分析
    visitor = GlobalCallGraphVisitor(known_functions)
    
    for mod in modules:
        try:
            source = inspect.getsource(mod)
            tree = ast.parse(source)
            visitor.visit(tree)
        except Exception:
            continue

    if not visitor.calls:
        return ""

    # 3. 构建 Mermaid 图
    lines = ["graph TD"]
    lines.append("    classDef main fill:#f9f,stroke:#333,stroke-width:2px;")
    lines.append("    classDef func fill:#fff,stroke:#333,stroke-width:1px;")
    
    edges = set()
    
    for caller, callee, args in visitor.calls:
        # 忽略递归调用
        if caller == callee: continue
        
        # 格式化边
        edge_label = ""
        if args:
            edge_label = f"|{', '.join(args)}|"
        
        edge_str = f"    {caller} -->{edge_label} {callee}"
        
        if edge_str not in edges:
            edges.add(edge_str)
            lines.append(edge_str)
            
            if caller == "main" or caller == "Main_Script":
                lines.append(f"    class {caller} main;")
            else:
                lines.append(f"    class {caller} func;")
            lines.append(f"    class {callee} func;")

    return "\n".join(lines)


def convert_md_to_html(md_content: str, title: str) -> str:
    """
    将 Markdown 内容转换为带有 Mermaid 渲染支持的 HTML。
    """
    # 简单的 Markdown -> HTML 转换 (为了不引入 heavy 依赖如 markdown 库，我们做简单的替换)
    # 注意：这里主要为了渲染 Mermaid 和基本结构。
    # 如果需要完美的 Markdown 渲染，建议用户安装 `markdown` 库，但这里我们用轻量级方案。
    
    html_content = md_content.replace("\n", "<br>\n")
    
    # 处理代码块 (简单的处理，防止 mermaid 代码被破坏)
    # 我们需要把 ```mermaid ... ``` 转换成 <div class="mermaid"> ... </div>
    
    parts = md_content.split("```")
    final_html_body = []
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # 普通文本
            # 简单的格式化处理
            text = part.replace("<", "&lt;").replace(">", "&gt;")
            
            # 处理标题
            lines = text.split('\n')
            formatted_lines = []
            for line in lines:
                if line.startswith('# '): formatted_lines.append(f"<h1>{line[2:]}</h1>")
                elif line.startswith('## '): formatted_lines.append(f"<h2>{line[3:]}</h2>")
                elif line.startswith('### '): formatted_lines.append(f"<h3>{line[4:]}</h3>")
                elif line.startswith('#### '): formatted_lines.append(f"<h4>{line[5:]}</h4>")
                elif line.startswith('**') and line.endswith('**'): formatted_lines.append(f"<b>{line[2:-2]}</b><br>")
                elif line.startswith('> '): formatted_lines.append(f"<blockquote>{line[2:]}</blockquote>")
                elif line.startswith('|'): formatted_lines.append(f"<p style='font-family:monospace; white-space: pre;'>{line}</p>") # 简单处理表格
                else: formatted_lines.append(f"{line}<br>")
            
            final_html_body.append("\n".join(formatted_lines))
        else:
            # 代码块
            if part.startswith("mermaid"):
                # Mermaid 图表
                graph_code = part[7:].strip() # 去掉 'mermaid'
                final_html_body.append(f'<div class="mermaid">\n{graph_code}\n</div>')
            else:
                # 普通代码块
                lang = part.split('\n')[0]
                code = part[len(lang):].strip()
                final_html_body.append(f'<pre style="background:#f4f4f4; padding:10px; border-radius:5px;"><code>{code}</code></pre>')

    body_str = "\n".join(final_html_body)

    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 1200px; margin: 0 auto; color: #333; }}
        h1, h2, h3 {{ color: #24292e; border-bottom: 1px solid #eaecef; padding-bottom: .3em; }}
        code {{ background-color: #f6f8fa; padding: 0.2em 0.4em; border-radius: 3px; font-family: monospace; }}
        pre {{ background-color: #f6f8fa; padding: 16px; overflow: auto; border-radius: 6px; }}
        blockquote {{ border-left: 4px solid #dfe2e5; color: #6a737d; padding-left: 1em; margin-left: 0; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
        th, td {{ border: 1px solid #dfe2e5; padding: 6px 13px; }}
        th {{ background-color: #f6f8fa; font-weight: 600; }}
        tr:nth-child(2n) {{ background-color: #f6f8fa; }}
        .mermaid {{ margin: 20px 0; text-align: center; }}
        details {{ margin-bottom: 10px; border: 1px solid #e1e4e8; border-radius: 6px; padding: 8px; }}
        summary {{ cursor: pointer; font-weight: bold; outline: none; }}
    </style>
</head>
<body>
    {body_str}

    <!-- 引入 Mermaid.js -->
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
</body>
</html>
    """
    return html_template


def inspect_library(
    library_name: str,
    output_path: Optional[str] = None,
    include_private: bool = False,
    include_imported: bool = False
):
    # ...existing code...
    # (保持之前的 inspect_library 逻辑不变，直到 Phase 2 之前)
    
    # --- 1. 动态导入主库 (带 sys.argv 保护) ---
    _old_argv = sys.argv
    sys.argv = [sys.argv[0]]

    submodules = []
    main_module = None

    try:
        try:
            main_module = importlib.import_module(library_name)
            submodules.append(main_module)
        except ImportError as e:
            print(f"❌ Error: Could not import library '{library_name}'. Reason: {e}")
            return
        except Exception as e:
            print(f"❌ Error: An unexpected error occurred while importing '{library_name}': {e}")
            return

        print(f"🔍 Analyzing dependencies for '{library_name}' (Network Analysis Phase)...")
        
        if hasattr(main_module, "__path__"):
            for importer, modname, ispkg in pkgutil.walk_packages(main_module.__path__, main_module.__name__ + "."):
                try:
                    sub_mod = importlib.import_module(modname)
                    submodules.append(sub_mod)
                except Exception:
                    continue
    finally:
        sys.argv = _old_argv

    lines = []
    lines.append(f"# Documentation for `{library_name}`")
    lines.append(f"**File Path:** `{getattr(main_module, '__file__', 'Built-in/Unknown')}`\n")
    
    doc = inspect.getdoc(main_module)
    if doc:
        lines.append("## Module Docstring")
        lines.append(f"```text\n{doc}\n```\n")

    # ==========================================
    # Phase 1: Network Construction & Analysis
    # ==========================================
    # (保留原有的 Phase 1 代码，此处省略)
    # ...existing code...
    G = nx.DiGraph() if HAS_NETWORKX else None
    internal_modules_rank = Counter() 
    external_libs_rank = Counter()    
    dependency_graph = defaultdict(set) 

    for mod in submodules:
        current_mod_name = mod.__name__
        if HAS_NETWORKX:
            G.add_node(current_mod_name, type='internal')
        for name, obj in inspect.getmembers(mod):
            obj_module = getattr(obj, "__module__", None)
            if not obj_module: continue
            if obj_module == current_mod_name: continue
            dependency_graph[current_mod_name].add(obj_module)
            if obj_module.startswith(library_name):
                internal_modules_rank[obj_module] += 1
                if HAS_NETWORKX: G.add_edge(current_mod_name, obj_module)
            else:
                top_level_pkg = obj_module.split('.')[0]
                if top_level_pkg not in ['builtins', 'sys', 'os', 'typing']:
                    external_libs_rank[top_level_pkg] += 1
                    if HAS_NETWORKX:
                        G.add_node(top_level_pkg, type='external')
                        G.add_edge(current_mod_name, top_level_pkg)

    lines.append("## 📊 Network & Architecture Analysis")
    if not HAS_NETWORKX: lines.append("> ⚠️ `networkx` is not installed. Advanced metrics are disabled.\n")
    lines.append("### 🌍 Top External Dependencies")
    if external_libs_rank:
        lines.append("| Library | Usage Count |")
        lines.append("| :--- | :--- |")
        for lib, count in external_libs_rank.most_common(10):
            lines.append(f"| **{lib}** | {count} |")
    else:
        lines.append("_No significant external dependencies._")
    lines.append("\n")
    
    # (保留原有的 Network Metrics 和 Dependency Map 代码)
    # ...existing code...
    if HAS_NETWORKX and len(G.nodes) > 0:
        lines.append("### 🕸️ Network Metrics (Advanced)")
        try:
            pagerank = nx.pagerank(G, alpha=0.85)
            sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            lines.append("#### 👑 Top Modules by PageRank (Authority)")
            lines.append("| Rank | Module | Score | Type |")
            lines.append("| :--- | :--- | :--- | :--- |")
            for i, (node, score) in enumerate(sorted_pr[:10]):
                node_type = "Internal" if node.startswith(library_name) else "External"
                short_name = node.replace(library_name + ".", "")
                lines.append(f"| {i+1} | `{short_name}` | {score:.4f} | {node_type} |")
            lines.append("\n")
        except Exception: pass

    lines.append("### 🗺️ Dependency & Architecture Map")
    mermaid_lines = ["graph TD"]
    mermaid_lines.append("    classDef core fill:#f96,stroke:#333,stroke-width:2px;")
    mermaid_lines.append("    classDef external fill:#9cf,stroke:#333,stroke-width:1px;")
    if HAS_NETWORKX: top_nodes = set(n for n, s in sorted_pr[:20])
    else: top_nodes = set(x[0] for x in internal_modules_rank.most_common(20))
    edges_to_draw = set()
    source_data = G.edges() if HAS_NETWORKX else []
    if not HAS_NETWORKX:
        for src, targets in dependency_graph.items():
            for tgt in targets: source_data.append((src, tgt))
    for u, v in source_data:
        if u in top_nodes or v in top_nodes:
            short_u = u.replace(library_name + ".", "").split('.')[-1]
            short_v = v.replace(library_name + ".", "").split('.')[-1]
            if not v.startswith(library_name): short_v = v.split('.')[0]
            if short_u == short_v: continue
            edge_id = f"{short_u}->{short_v}"
            if edge_id in edges_to_draw: continue
            edges_to_draw.add(edge_id)
            arrow = "-.->" if not v.startswith(library_name) else "-->"
            mermaid_lines.append(f"    {short_u}{arrow}{short_v}")
            if u.startswith(library_name): mermaid_lines.append(f"    class {short_u} core;")
            else: mermaid_lines.append(f"    class {short_u} external;")
            if v.startswith(library_name): mermaid_lines.append(f"    class {short_v} core;")
            else: mermaid_lines.append(f"    class {short_v} external;")
    
    all_classes = []
    for mod in submodules:
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if getattr(obj, "__module__", "").startswith(library_name):
                all_classes.append((name, obj))
    if len(all_classes) < 50: 
        for name, obj in all_classes:
            for base in obj.__bases__:
                base_name = base.__name__
                if base_name == 'object': continue
                mermaid_lines.append(f"    {name} --|> {base_name}")
                mermaid_lines.append(f"    class {name} core;")
                if base.__module__.split('.')[0] != library_name: mermaid_lines.append(f"    class {base_name} external;")
                else: mermaid_lines.append(f"    class {base_name} core;")
    lines.append("<details><summary>Show Mermaid Graph</summary>\n")
    lines.append("```mermaid")
    lines.append("\n".join(mermaid_lines))
    lines.append("```\n</details>\n")

    # ==========================================
    # Phase 1.5: Global Call Graph (新增：宏观逻辑流)
    # ==========================================
    lines.append("## 🚀 Global Execution Flow")
    lines.append("This graph visualizes how data flows between functions across the entire project.")
    lines.append("It traces function calls to show the high-level logic pipeline.")
    
    global_call_graph = generate_global_call_graph(submodules, library_name)
    if global_call_graph:
        lines.append("```mermaid")
        lines.append(global_call_graph)
        lines.append("```\n")
    else:
        lines.append("_No internal function calls detected (or code structure is too dynamic)._\n")

    # ==========================================
    # Phase 2: Surface Level Inspection & Logic Flow
    # ==========================================
    lines.append("## 📑 Top-Level API Contents & Logic Flow")

    if hasattr(main_module, "__all__"):
        all_names = main_module.__all__
        using_all = True
    else:
        all_names = dir(main_module)
        using_all = False
    
    members_data = []

    for name in all_names:
        if not include_private and not using_all and name.startswith("_"):
            continue
        try: obj = getattr(main_module, name)
        except AttributeError: continue
        obj_module = getattr(obj, "__module__", None)
        is_imported = False
        if obj_module and not obj_module.startswith(library_name): is_imported = True
        if not include_imported and is_imported:
             if not using_all: continue
        members_data.append((name, obj, is_imported))

    classes = []
    functions = []
    for name, obj, is_imported in members_data:
        display_name = name + (" (imported)" if is_imported else "")
        if inspect.isclass(obj): classes.append((display_name, obj))
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj): functions.append((display_name, obj))

    def get_info(obj):
        try: sig = str(inspect.signature(obj))
        except (ValueError, TypeError):
            sig = getattr(obj, "__text_signature__", "(...)")
            if sig is None: sig = "(...)"
        doc = inspect.getdoc(obj) or "No documentation available."
        return sig, doc

    if functions:
        lines.append("### 🔧 Functions")
        for name, func in functions:
            sig, doc = get_info(func)
            lines.append(f"#### `{name}{sig}`")
            lines.append(f"> {doc.splitlines()[0] if doc else ''}")
            
            lines.append(f"<details><summary>Full Docstring</summary>\n\n```text\n{doc}\n```\n</details>\n")

            flow_chart = generate_function_flowchart(func)
            if flow_chart:
                lines.append("\n**Logic Flow:**")
                lines.append("```mermaid")
                lines.append(flow_chart)
                lines.append("```\n")

    if classes:
        lines.append("### 📦 Classes")
        for name, cls in classes:
            sig, doc = get_info(cls)
            lines.append(f"#### `class {name}{sig}`")
            lines.append(f"{doc.splitlines()[0] if doc else ''}\n")
            
            methods = inspect.getmembers(cls, predicate=lambda x: inspect.isfunction(x) or inspect.ismethod(x))
            if methods:
                lines.append("| Method | Signature | Description |")
                lines.append("| :--- | :--- | :--- |")
                for m_name, m_obj in methods:
                    if not include_private and m_name.startswith("_") and m_name != "__init__":
                        continue
                    m_sig, m_doc = get_info(m_obj)
                    short_doc = m_doc.splitlines()[0] if m_doc else "-"
                    short_doc = short_doc.replace("|", "\\|")
                    lines.append(f"| **{m_name}** | `{m_sig}` | {short_doc} |")
            lines.append("\n")

    # --- Output ---
    content = "\n".join(lines)
    
    if output_path:
        # 1. 保存 Markdown (原逻辑)
        md_path = output_path
        if not md_path.endswith(".md"):
            md_path += ".md"
        
        output_dir = os.path.dirname(md_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                print(f"❌ Error creating directory {output_dir}: {e}")
                return

        try:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Markdown report saved to: {os.path.abspath(md_path)}")
            
            # 2. (新增) 自动生成 HTML 版本
            html_path = md_path.replace(".md", ".html")
            html_content = convert_md_to_html(content, f"Analysis Report: {library_name}")
            
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"📊 Interactive HTML report saved to: {os.path.abspath(html_path)}")
            print(f"   (Open the HTML file in your browser to see rendered charts)")
            
        except IOError as e:
            print(f"❌ Error writing file: {e}")
    else:
        print(content)



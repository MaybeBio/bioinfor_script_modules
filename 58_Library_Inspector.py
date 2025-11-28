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

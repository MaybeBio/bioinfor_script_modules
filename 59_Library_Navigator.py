import inspect # 查看类、对象内部结构属性
import importlib # 动态导入模块
import sys
import os
import pkgutil # 遍历包/模块(查找子模块用)
import ast # 将 Python 代码解析为抽象语法树（AST），实现代码分析 / 重构
import re
from collections import Counter, defaultdict
from typing import Any, List, Dict, Optional, Tuple, Set
import json
import html # 设计到Memarid md转html, html的一些语言的转义

# 尝试导入 networkx 进行高级网络分析
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# --- Helper: ID Sanitizer ---
def sanitize_id(name: str) -> str:
    """
    将任意字符串转换为合法的 Mermaid 节点 ID。
    Mermaid ID 只能包含字母、数字、下划线。
    """
    # 将点号、空格、特殊符号替换为下划线
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # 避免数字开头
    if clean and clean[0].isdigit():
        clean = "_" + clean
    return clean


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
        # 1. 清洗 ID
        safe_node_id = sanitize_id(node.id)
        
        # 2. 转义 Label 中的特殊字符
        # 将双引号转义为单引号，防止破坏 Mermaid 语法
        safe_label = node.label.replace('"', "'")
        
        shape_start, shape_end = "(", ")"
        if node.node_type == "input": shape_start, shape_end = "[", "]"
        if node.node_type == "output": shape_start, shape_end = "([", "])"
        
        # 使用引号包裹 Label，确保特殊字符（如空格、=）被正确显示
        lines.append(f'    {safe_node_id}{shape_start}"{safe_label}"{shape_end}:::{node.node_type}')
        
        for source_id, var_name in node.edges_in:
            safe_source_id = sanitize_id(source_id)
            # 连线 Label 也要清洗，去掉可能破坏语法的字符
            safe_var = var_name.replace('"', "'").replace('|', '/')
            lines.append(f'    {safe_source_id} -- "{safe_var}" --> {safe_node_id}')

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
        
        # 【关键修复】生成安全的 ID
        caller_id = sanitize_id(caller)
        callee_id = sanitize_id(callee)

        # 格式化边
        edge_label = ""
        if args:
            # 截断过长的参数列表，防止图表爆炸
            arg_str = '<br>'.join(args) 
            # if len(arg_str) > 20:
                # arg_str = arg_str[:17] + "..." # ！！！！！！！！！！！
            # 移除可能破坏 Mermaid 语法的字符
            arg_str = arg_str.replace('"', "'").replace('|', '/')
            edge_label = f"|{arg_str}|"
        
        # 使用 ID[Label] 的格式
        # 这样 ID 是安全的（无点号），Label 可以包含点号
        edge_str = f'    {caller_id}["{caller}"] -->{edge_label} {callee_id}["{callee}"]'
        
        if edge_str not in edges:
            edges.add(edge_str)
            lines.append(edge_str)
            
            # 应用样式到 ID
            if caller == "main" or caller == "Main_Script":
                lines.append(f"    class {caller_id} main;")
            else:
                lines.append(f"    class {caller_id} func;")
            lines.append(f"    class {callee_id} func;")

    return "\n".join(lines)


def convert_md_to_html(md_content: str, title: str) -> str:
    """
    将 Markdown 内容转换为带有 Mermaid 渲染支持的 HTML。
    修复了标签转义问题，确保 Mermaid 代码能被正确解析。
    """
    parts = md_content.split("```")
    final_html_body = []
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # === 普通文本块 ===
            # 1. 先进行 HTML 转义，防止正文中的 < > 破坏页面结构
            text = html.escape(part)
            
            # 2. 【关键修复】还原我们生成的特定 HTML 标签
            # 因为我们在 inspect_library 中手动添加了这些标签，所以这里要“反转义”回来
            text = text.replace("&lt;details&gt;", "<details>")
            text = text.replace("&lt;/details&gt;", "</details>")
            text = text.replace("&lt;summary&gt;", "<summary>")
            text = text.replace("&lt;/summary&gt;", "</summary>")
            
            # 3. 简单的 Markdown 格式化
            lines = text.split('\n')
            formatted_lines = []
            for line in lines:
                if line.startswith('# '): formatted_lines.append(f"<h1>{line[2:]}</h1>")
                elif line.startswith('## '): formatted_lines.append(f"<h2>{line[3:]}</h2>")
                elif line.startswith('### '): formatted_lines.append(f"<h3>{line[4:]}</h3>")
                elif line.startswith('#### '): formatted_lines.append(f"<h4>{line[5:]}</h4>")
                elif line.startswith('**') and line.endswith('**'): formatted_lines.append(f"<b>{line[2:-2]}</b><br>")
                # 注意：html.escape 后，> 变成了 &gt;
                elif line.startswith('&gt; '): formatted_lines.append(f"<blockquote>{line[5:]}</blockquote>")
                elif line.startswith('|'): formatted_lines.append(f"<p style='font-family:monospace; white-space: pre;'>{line}</p>")
                else: formatted_lines.append(f"{line}<br>")
            
            final_html_body.append("\n".join(formatted_lines))
        else:
            # === 代码块 ===
            if part.startswith("mermaid"):
                # Mermaid 图表
                graph_code = part[7:].strip()
                
                # 【关键修复】对 Mermaid 代码进行 HTML 转义
                # 这样 A-->B 中的 > 会变成 &gt;，<br> 会变成 &lt;br&gt;
                # 浏览器解析 HTML 后，Mermaid 引擎读取到的就是原始的 A-->B 和 <br> 字符串
                # 这能完美解决 Syntax error 问题
                escaped_code = html.escape(graph_code)
                
                final_html_body.append(f'<div class="mermaid" style="overflow-x: auto;">\n{escaped_code}\n</div>')
            else:
                # 普通代码
                lang = part.split('\n')[0]
                code = part[len(lang):].strip()
                escaped_code = html.escape(code)
                final_html_body.append(f'<pre style="background:#f4f4f4; padding:10px; border-radius:5px;"><code>{escaped_code}</code></pre>')

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
        
        /* 优化边标签样式 */
        .edgeLabel {{
            font-size: 11px !important;
            background-color: rgba(255, 255, 255, 0.9) !important;
            padding: 2px !important;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    {body_str}

    <!-- 引入 Mermaid.js -->
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ 
            startOnLoad: true, 
            theme: 'default',
            flowchart: {{ 
                useMaxWidth: false, 
                htmlLabels: true,
                rankSpacing: 150, 
                nodeSpacing: 100,
                curve: 'basis' 
            }} 
        }});
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
    
    # --- 【关键修改】使用纯数字 ID 映射 ---
    # 1. 收集所有需要绘制的节点名称
    nodes_to_map = set()
    
    # 收集依赖关系中的节点
    source_data = G.edges() if HAS_NETWORKX else []
    if not HAS_NETWORKX:
        for src, targets in dependency_graph.items():
            for tgt in targets: source_data.append((src, tgt))
            
    dependency_edges = []
    for u, v in source_data:
        if u in top_nodes or v in top_nodes:
            # 简化名称逻辑
            short_u = u.replace(library_name + ".", "").split('.')[-1]
            short_v = v.replace(library_name + ".", "").split('.')[-1]
            if not v.startswith(library_name): short_v = v.split('.')[0]
            
            if short_u == short_v: continue
            
            nodes_to_map.add(u)
            nodes_to_map.add(v)
            dependency_edges.append((u, v, short_u, short_v))

    # 收集继承关系中的节点
    inheritance_edges = []
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
                
                # 这里的 name 和 base_name 已经是短名称了，但为了 ID 映射，我们需要唯一标识
                # 简单起见，类名直接作为唯一标识（假设没有重名类，或者不关心重名）
                nodes_to_map.add(name)
                nodes_to_map.add(base_name)
                
                # 记录基类的模块信息用于判断是否 external
                base_module = base.__module__.split('.')[0]
                inheritance_edges.append((name, base_name, base_module))

    # 2. 构建 ID 映射表
    # id_map: {"NodeName": "id_0", "OtherNode": "id_1", ...}
    id_map = {name: f"id_{i}" for i, name in enumerate(nodes_to_map)}

    # 3. 绘制依赖关系
    edges_drawn = set()
    for u, v, label_u, label_v in dependency_edges:
        uid = id_map[u]
        vid = id_map[v]
        
        edge_key = f"{uid}->{vid}"
        if edge_key in edges_drawn: continue
        edges_drawn.add(edge_key)
        
        arrow = "-.->" if not v.startswith(library_name) else "-->"
        
        # 【优化】在箭头前后增加空格，确保语法解析更稳定
        mermaid_lines.append(f'    {uid}["{label_u}"] {arrow} {vid}["{label_v}"]')
        
        if u.startswith(library_name): mermaid_lines.append(f"    class {uid} core;")
        else: mermaid_lines.append(f"    class {uid} external;")
        
        if v.startswith(library_name): mermaid_lines.append(f"    class {vid} core;")
        else: mermaid_lines.append(f"    class {vid} external;")

    # 4. 绘制继承关系
    for cls_name, base_name, base_mod in inheritance_edges:
        cid = id_map[cls_name]
        bid = id_map[base_name]
        
        # 【关键修复】将 --|> (类图语法) 改为 ==> (流程图粗箭头语法)
        # 这样既修复了 Syntax Error，又能通过粗线条在视觉上区分继承关系
        mermaid_lines.append(f'    {cid}["{cls_name}"] ==> {bid}["{base_name}"]')
        
        mermaid_lines.append(f"    class {cid} core;")
        
        if base_mod != library_name:
            mermaid_lines.append(f"    class {bid} external;")
        else:
            mermaid_lines.append(f"    class {bid} core;")

    # 直接输出 Mermaid 代码块
    lines.append("```mermaid")
    lines.append("\n".join(mermaid_lines))
    lines.append("```\n")

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



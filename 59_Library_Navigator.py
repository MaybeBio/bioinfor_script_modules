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


========================================================================================================================
# 2, 


import inspect # 查看类、对象内部结构属性
import importlib # 动态导入模块
import sys
import os
import pkgutil # 遍历包/模块(查找子模块用)
import ast # 将 Python 代码解析为抽象语法树（AST），实现代码分析 / 重构
import re
from collections import Counter, defaultdict, deque
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
        # 新增：用于依赖闭包计算的邻接表
        self.dependency_map = defaultdict(set) # caller -> set(callees)

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
                self.dependency_map[self.current_function].add(callee_name)
        
        self.generic_visit(node)

# --- Helper: 3. 入口分析 (Navigator: How to Drive) ---
class EntryAnalysisVisitor(ast.NodeVisitor):
    """
    分析代码中的入口点，特别是 argparse 的使用。
    """
    def __init__(self):
        self.has_main_block = False
        self.args = [] # List of (arg_name, help_text)

    def visit_If(self, node):
        # 检测 if __name__ == "__main__":
        try:
            if (isinstance(node.test, ast.Compare) and 
                isinstance(node.test.left, ast.Name) and 
                node.test.left.id == "__name__" and 
                isinstance(node.test.comparators[0], ast.Constant) and 
                node.test.comparators[0].value == "__main__"):
                self.has_main_block = True
        except: pass
        self.generic_visit(node)

    def visit_Call(self, node):
        # 检测 parser.add_argument(...)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            arg_name = "Unknown"
            help_text = ""
            
            # 提取参数名 (通常是第一个参数)
            if node.args:
                if isinstance(node.args[0], ast.Constant):
                    arg_name = node.args[0].value
            
            # 提取 help 信息
            for kw in node.keywords:
                if kw.arg == 'help' and isinstance(kw.value, ast.Constant):
                    help_text = kw.value.value
            
            self.args.append((arg_name, help_text))
        
        self.generic_visit(node)

# --- Helper: 4. 依赖闭包计算 (Navigator: Extraction Guide) ---
def get_dependency_closure(target_func: str, dependency_map: Dict[str, Set[str]]) -> Set[str]:
    """
    计算目标函数的依赖闭包（即运行该函数所需的所有其他函数）。
    """
    closure = set()
    queue = deque([target_func])
    visited = set()

    while queue:
        current = queue.popleft()
        if current in visited: continue
        visited.add(current)
        closure.add(current)

        if current in dependency_map:
            for dep in dependency_map[current]:
                if dep not in visited:
                    queue.append(dep)
    return closure

# --- Helper: 5. 模块分类器 (Navigator: Architecture) ---
def classify_module(module_obj) -> str:
    """
    根据模块导入的外部库推测其角色。
    """
    try:
        source = inspect.getsource(module_obj)
    except:
        return "Unknown"
    
    # 简单的关键词匹配
    if "torch" in source or "tensorflow" in source or "keras" in source:
        return "Model / AI"
    if "flask" in source or "django" in source or "fastapi" in source:
        return "Web / API"
    if "pandas" in source or "numpy" in source or "csv" in source:
        return "Data Processing"
    if "matplotlib" in source or "seaborn" in source or "plotly" in source:
        return "Visualization"
    if "argparse" in source or "click" in source:
        return "Interface / CLI"
    
    return "Utility / Core"


def generate_global_call_graph(modules: List[Any], library_name: str) -> Tuple[str, Dict[str, Set[str]]]:
    """
    生成全局函数调用图 (Global Call Graph) 并返回依赖映射
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
        return "", {}

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

    return "\n".join(lines), visitor.dependency_map


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
    # Phase 0: Navigator - How to Drive (新增)
    # ==========================================
    lines.append("## 🚦 Navigator: How to Drive")
    lines.append("This section helps you understand how to run this library from the command line or entry points.")
    
    entry_visitor = EntryAnalysisVisitor()
    try:
        source = inspect.getsource(main_module)
        entry_visitor.visit(ast.parse(source))
    except: pass

    if entry_visitor.has_main_block:
        lines.append("- ✅ **Entry Point Detected**: This module contains an `if __name__ == '__main__':` block, meaning it can be run directly.")
    else:
        lines.append("- ℹ️ **No Direct Entry Point**: This module seems to be a library intended for import, not direct execution.")
        
        # --- 新增：针对 Import 库的智能导航 ---
        lines.append("\n### 🐍 Python API Usage (Inferred)")
        lines.append("Since no CLI entry point was found, here are the likely **Python API entry points** for your script:")
        
        # 1. 预先构建调用图以辅助判断 (提前运行部分 Phase 1 逻辑)
        # 我们需要知道哪些函数是 "Top Level" (没有被内部调用的函数)
        temp_known_funcs = set()
        temp_calls = []
        if hasattr(main_module, "__path__"):
             # 简单扫描一下主模块和直接子模块
            scan_modules = [main_module] + submodules # 限制数量避免太慢submodules[:10]
        else:
            scan_modules = [main_module]

        for m in scan_modules:
            for n, o in inspect.getmembers(m):
                if inspect.isfunction(o) or inspect.ismethod(o): temp_known_funcs.add(n)
        
        temp_visitor = GlobalCallGraphVisitor(temp_known_funcs)
        for m in scan_modules:
            try: temp_visitor.visit(ast.parse(inspect.getsource(m)))
            except: pass
            
        # 计算入度 (被调用次数)
        in_degree = Counter()
        for caller, callee, _ in temp_visitor.calls:
            in_degree[callee] += 1

        # 2. 寻找潜在的 API 入口
        api_candidates = []
        candidates_source = []
        
        if hasattr(main_module, "__all__"):
            candidates_source = main_module.__all__
        else:
            candidates_source = [x for x in dir(main_module) if not x.startswith("_")]
            
        for name in candidates_source:
            try:
                obj = getattr(main_module, name)
                # 只关注函数和类
                if inspect.isfunction(obj):
                    api_candidates.append((name, "Function", obj))
                elif inspect.isclass(obj):
                    # 忽略异常类
                    if issubclass(obj, Exception): continue
                    api_candidates.append((name, "Class", obj))
            except: continue
            
        # 3. 增强评分机制
        def score_candidate(item):
            name, kind, obj = item
            score = 0
            name_lower = name.lower()
            
            # A. 关键词评分 (业务逻辑优先)
            # 核心动词
            if any(v in name_lower for v in ['predict', 'calculate', 'analyze', 'solve', 'run', 'process', 'train', 'evaluate', 'generate']): score += 15
            # 入口动词
            elif any(v in name_lower for v in ['main', 'start', 'init', 'load', 'create', 'make']): score += 10
            # 转换/工具动词
            elif any(v in name_lower for v in ['convert', 'parse', 'read', 'write', 'save', 'plot', 'show']): score += 5
            
            # B. 类型评分
            if kind == 'Class':
                # 模型/核心类加分
                if any(n in name_lower for n in ['model', 'engine', 'client', 'api', 'runner', 'predictor']): score += 8
            
            # C. 拓扑评分 (关键改进)
            # 如果一个函数是 Public 的，且在内部几乎没被调用 (入度低)，说明它是给外部用的
            if kind == 'Function':
                degree = in_degree.get(name, 0)
                if degree == 0: score += 10  # 纯顶层接口
                elif degree < 3: score += 5  # 低耦合接口
                else: score -= 5             # 被内部大量调用，可能是底层工具函数
            
            # D. 复杂度评分 (代码行数)
            try:
                lines_of_code = len(inspect.getsource(obj).splitlines())
                if lines_of_code > 50: score += 5 # 逻辑复杂的通常是主要接口
                elif lines_of_code < 3: score -= 5 # 只有一两行的通常是 wrapper
            except: pass

            return score

        api_candidates.sort(key=score_candidate, reverse=True)
        
        # 4. 生成代码片段

        '''
        if api_candidates: 
            lines.append("```python")
            lines.append(f"import {library_name}")
            lines.append("")
            lines.append("# Likely entry points (Ranked by relevance):")
            # 展示前 8 个，增加覆盖率
            for name, kind, obj in api_candidates: 
                try:
                    sig = str(inspect.signature(obj))
                except: sig = "(...)"
                
                # 增加一行简短注释 (取 docstring 第一行)
                doc = inspect.getdoc(obj)
                doc_summary = f"  # {doc.splitlines()[0]}" if doc else ""
                # 截断过长的注释
                if len(doc_summary) > 60: doc_summary = doc_summary[:57] + "..."
                
                lines.append(f"# {kind}: {library_name}.{name}{sig}{doc_summary}")
            lines.append("```")
        else:
            lines.append("_No obvious public API members detected._")
        '''

        # --- 新增：人性化格式输出 ---
        if api_candidates:
            lines.append("\n#### 🚀 Top Recommended Entry Points")
            lines.append("| Type | API | Description |")
            lines.append("| :--- | :--- | :--- |")
            
            count = 0
            for name, kind, obj in api_candidates:
                # 过滤掉得分太低的 (可能是底层工具)
                if score_candidate((name, kind, obj)) < 0 and count > 5: continue
                # if count >= 10: break # 最多显示 10 个核心 API
                
                # 1. 提取签名并美化参数
                try:
                    sig = inspect.signature(obj)
                    params = []
                    for p_name, p in sig.parameters.items():
                        if p.default == inspect.Parameter.empty:
                            # 必填参数加粗
                            params.append(f"**{p_name}**")
                        else:
                            # 可选参数普通字体
                            params.append(f"{p_name}")
                    
                    # 重新组装签名，避免过长
                    sig_str = f"({', '.join(params)})"
                    # if len(sig_str) > 60: # 如果参数太长，截断
                    #    sig_str = f"({', '.join(params[:3])}, ...)"
                except: 
                    sig_str = "(...)"

                # 2. 提取并清洗文档
                doc = inspect.getdoc(obj)
                doc_summary = doc.splitlines()[0] if doc else "No description."
                # if len(doc_summary) > 80: doc_summary = doc_summary[:77] + "..."
                
                # 3. 图标区分
                icon = "ƒ" if kind == "Function" else "C"
                
                lines.append(f"| `{icon}` | **{library_name}.{name}**{sig_str} | {doc_summary} |")
                count += 1
            
            lines.append("\n> **Note:** Bold parameters are required. Others are optional.")
            
            # --- 新增：多维度代码片段生成 ---
            lines.append("\n#### 🧩 Code Snippets (Auto-Generated)")
            lines.append("```python")
            lines.append(f"import {library_name}")
            lines.append("")
            
            # 1. 提取 Top Functions (最多 10 个)
            top_funcs = [x for x in api_candidates if x[1] == 'Function'][:10]
            if top_funcs:
                lines.append("# --- Top Ranked Functions ---")
                for i, (name, _, obj) in enumerate(top_funcs):
                    try:
                        sig = inspect.signature(obj)
                        args = []
                        for p_name, p in sig.parameters.items():
                            if p.default == inspect.Parameter.empty and p_name != 'self':
                                args.append(f"{p_name}=...") 
                            elif p.default != inspect.Parameter.empty:
                                # 可选参数注释掉，提示用户存在
                                # args.append(f"{p_name}={p.default}") 
                                pass
                        
                        # 如果参数太多，换行显示
                        if len(args) > 3:
                            args_str = ",\n    ".join(args)
                            call_str = f"{library_name}.{name}(\n    {args_str}\n)"
                        else:
                            call_str = f"{library_name}.{name}({', '.join(args)})"
                            
                        lines.append(f"# {i+1}. {name}")
                        lines.append(f"result_{i+1} = {call_str}")
                        lines.append("")
                    except: pass

            # 2. 提取 Top Class (最多 10 个)
            top_classes = [x for x in api_candidates if x[1] == 'Class'][:10]
            if top_classes:
                lines.append("# --- Core Classes Initialization ---")
                for i, (name, _, obj) in enumerate(top_classes):
                    try:
                        # 尝试获取 __init__ 的签名
                        sig = inspect.signature(obj)
                        args = []
                        for p_name, p in sig.parameters.items():
                            if p.default == inspect.Parameter.empty and p_name != 'self':
                                args.append(f"{p_name}=...")
                        
                        call_str = f"{library_name}.{name}({', '.join(args)})"
                        instance_name = name.lower().replace("model", "_model").replace("class", "_obj")
                        
                        lines.append(f"# {i+1}. {name}")
                        lines.append(f"{instance_name} = {call_str}")
                        lines.append("")
                    except: pass
            
            lines.append("```")

        else:
            lines.append("_No obvious public API members detected._")

    if entry_visitor.args:
        lines.append("\n### ⌨️ CLI Arguments (Detected)")
        lines.append("| Argument | Help Text |")
        lines.append("| :--- | :--- |")
        for arg, help_text in entry_visitor.args:
            lines.append(f"| `{arg}` | {help_text} |")
    else:
        lines.append("\n_No explicit `argparse` configuration detected in the main module._")
    lines.append("\n")

    # ==========================================
    # Phase 1: Network Construction & Analysis
    # ==========================================

    G = nx.DiGraph() if HAS_NETWORKX else None
    internal_modules_rank = Counter() 
    external_libs_rank = Counter()    
    dependency_graph = defaultdict(set) 
    module_roles = {} # module_name -> role

    for mod in submodules:
        current_mod_name = mod.__name__
        
        # 模块角色分类
        role = classify_module(mod)
        module_roles[current_mod_name] = role

        if HAS_NETWORKX:
            G.add_node(current_mod_name, type='internal', role=role)
        
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
    
    sorted_pr = []
    if HAS_NETWORKX and len(G.nodes) > 0:
        lines.append("### 🕸️ Network Metrics (Advanced)")
        try:
            pagerank = nx.pagerank(G, alpha=0.85)
            sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            lines.append("#### 👑 Top Modules by PageRank (Authority)")
            lines.append("| Rank | Module | Score | Type | Role |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for i, (node, score) in enumerate(sorted_pr[:10]):
                node_type = "Internal" if node.startswith(library_name) else "External"
                role = module_roles.get(node, "External Lib")
                short_name = node.replace(library_name + ".", "")
                lines.append(f"| {i+1} | `{short_name}` | {score:.4f} | {node_type} | {role} |")
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
                
                nodes_to_map.add(name)
                nodes_to_map.add(base_name)
                
                base_module = base.__module__.split('.')[0]
                inheritance_edges.append((name, base_name, base_module))

    # 2. 构建 ID 映射表
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
        mermaid_lines.append(f'    {uid}["{label_u}"] {arrow} {vid}["{label_v}"]')
        
        if u.startswith(library_name): mermaid_lines.append(f"    class {uid} core;")
        else: mermaid_lines.append(f"    class {uid} external;")
        
        if v.startswith(library_name): mermaid_lines.append(f"    class {vid} core;")
        else: mermaid_lines.append(f"    class {vid} external;")

    # 4. 绘制继承关系
    for cls_name, base_name, base_mod in inheritance_edges:
        cid = id_map[cls_name]
        bid = id_map[base_name]
        
        mermaid_lines.append(f'    {cid}["{cls_name}"] ==> {bid}["{base_name}"]')
        mermaid_lines.append(f"    class {cid} core;")
        
        if base_mod != library_name:
            mermaid_lines.append(f"    class {bid} external;")
        else:
            mermaid_lines.append(f"    class {bid} core;")

    lines.append("```mermaid")
    lines.append("\n".join(mermaid_lines))
    lines.append("```\n")

    # ==========================================
    # Phase 1.5: Global Call Graph & Extraction Guide (新增)
    # ==========================================
    lines.append("## 🚀 Global Execution Flow & Extraction Guide")
    lines.append("This graph visualizes how data flows between functions across the entire project.")
    
    global_call_graph, dependency_map = generate_global_call_graph(submodules, library_name)
    if global_call_graph:
        lines.append("```mermaid")
        lines.append(global_call_graph)
        lines.append("```\n")
        
        # --- Navigator: Extraction Guide ---
        lines.append("### ✂️ Navigator: Snippet Extractor")
        lines.append("Want to use a specific function without the whole library? Here is the **Dependency Closure** for key functions.")
        
        # 选取几个重要的函数进行分析 (如果有 PageRank，选排名高的；否则选调用量大的)
        # 这里简单起见，选取 dependency_map 中作为 caller 出现次数最多的前 3 个函数
        top_funcs = sorted(dependency_map.keys(), key=lambda k: len(dependency_map[k]), reverse=True)[:3]
        
        if top_funcs:
            for func in top_funcs:
                closure = get_dependency_closure(func, dependency_map)
                lines.append(f"#### To extract `{func}`:")
                lines.append(f"> You need these **{len(closure)}** components:")
                lines.append(f"`{', '.join(sorted(list(closure)))}`")
                lines.append("")
        else:
            lines.append("_Not enough call data to generate extraction guide._")

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


============================================================================================================


# 3,  

import inspect # Inspect live objects and class internals
import importlib # Dynamically import modules
import sys
import os
import pkgutil # Walk through packages/modules (used to find submodules)
import ast # Parse Python code into Abstract Syntax Trees (AST) for code analysis/refactoring
import re
from collections import Counter, defaultdict, deque
from typing import Any, List, Dict, Optional, Tuple, Set
import json
import html # Used for Mermaid MD to HTML conversion, escaping HTML characters


# Attempt to import networkx for advanced network analysis
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


# --- Helper: ID Sanitizer ---
def sanitize_id(name: str) -> str:
    """
    Description
    ----------
    Convert an arbitrary string into a valid Mermaid node ID. Mermaid node IDs
    should only contain letters, digits and underscores.

    Args
    -----
    name : str
        The input string to sanitize into a Mermaid-safe identifier.

    Returns
    --------
    str
        A sanitized identifier containing only [A-Za-z0-9_] and guaranteed not
        to start with a digit (an underscore is prepended if necessary).

    Notes
    -------
    - 1, Non-alphanumeric characters (including dots, spaces and symbols) are
      replaced with underscores.
    - 2, If the resulting identifier starts with a digit, a leading underscore is
      added to ensure it is a valid ID for Mermaid.
    - 3, Empty input returns an empty string.
    """
    # Replace dots, Spaces, and special symbols with underscores
    clean = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Avoid starting with a digit
    if clean and clean[0].isdigit():
        clean = "_" + clean
    return clean


# --- Helper: 1. Single-function Logic Analysis (Micro) ---
class LogicNode:
    """Represents a node in a function flowchart.

    Description
    ----------
    Models a single element in a data/logic flow diagram extracted from a
    function's AST. Nodes denote inputs, processing steps, or outputs and
    are used to assemble Mermaid flowcharts showing data production and use.

    Attributes
    ----------
    id : str
        Unique identifier for the node (used when rendering the diagram).
    label : str
        Human-readable label displayed inside the node.
    node_type : str
        One of "input", "process", or "output" — controls node shape/style.
    edges_in : List[Tuple[str, str]]
        Incoming edges as (source_node_id, variable_name) pairs indicating
        which node produced each input variable.
    """
    def __init__(self, id, label, node_type="process"):
        self.id = id
        self.label = label
        self.node_type = node_type # input, process, output
        self.edges_in = [] # List of (source_id, var_name)

class AdvancedFlowVisitor(ast.NodeVisitor):
    """
    Description
    ----------
    Parses function source code to build a data flow graph.
    Tracks the chain of variable production (Definition) -> consumption (Usage).
    """
    def __init__(self):
        """
        Description
        ----------
        Initialize visitor state used to accumulate flow nodes and track current producers.

        Args
        -----
        None

        Returns
        --------
        None

        Notes
        -------
        - Initializes:
          - self.nodes: list of LogicNode instances discovered
          - self.current_producers: mapping var_name -> node_id for latest producer
          - self.counter: integer counter for generating unique node ids
        """
        self.nodes = []
        self.current_producers = {} # var_name -> node_id (Tracks which node currently produces each variable)
        
        self.counter = 0

    def _get_id(self):
        """
        Description
        ----------
        Generate a new unique internal node identifier.

        Args
        -----
        None

        Returns
        --------
        str
            A new unique node id string (e.g., "Node1").

        Notes
        -------
        - Increments an internal counter on each call.
        """
        self.counter += 1
        return f"Node{self.counter}"

    def _resolve_inputs(self, input_vars: List[str]) -> List[Tuple[str, str]]:
        """
        Description
        ----------
        Resolve which previously created nodes produced the given input variables.

        Args
        -----
        input_vars : List[str]
            Variable names referenced on the right-hand side of an expression.

        Returns
        --------
        List[Tuple[str, str]]
            List of (source_node_id, var_name) pairs for known producers.

        Notes
        -------
        - Only returns producers that were previously recorded in self.current_producers.
        """
        edges = []
        for var in input_vars:
            if var in self.current_producers:
                source_id = self.current_producers[var]
                edges.append((source_id, var))
        return edges

    def _extract_names(self, node) -> List[str]:
        """
        Description
        ----------
        Extract variable names referenced inside an AST node.

        Args
        -----
        node : ast.AST
            The AST node to inspect for Name and Attribute usage.

        Returns
        --------
        List[str]
            Unique list of variable names found (includes simple names and 'self.attr' for attributes).

        Notes
        -------
        - Only collects names used in load (read) context.
        - Captures simple `Name` nodes and `self.attr` attributes; other attributes are visited recursively.
        """
        names = []
        class NameCollector(ast.NodeVisitor):
            def visit_Name(self, n):
                if isinstance(n.ctx, ast.Load):
                    names.append(n.id)
            def visit_Attribute(self, n):
                # Attempt to capture self.xxx
                if isinstance(n.value, ast.Name) and n.value.id == 'self':
                    names.append(f"self.{n.attr}")
                self.generic_visit(n)
        
        if node:
            NameCollector().visit(node)
        return list(set(names)) # Remove duplicates

    def visit_FunctionDef(self, node):
        """
        Description
        ----------
        Handle a FunctionDef AST node: register input parameter node and traverse body.

        Args
        -----
        node : ast.FunctionDef
            The function definition AST node.

        Returns
        --------
        None

        Notes
        -------
        - Creates an "Input" LogicNode if the function has parameters and marks parameters as produced
          by that Input node so subsequent assignments can reference them as inputs.
        - Continues traversal into the function body to collect assignments, calls and return nodes.
        """
        args = []
        arg_labels = []
        
        # Extract parameters and type annotations
        all_args = node.args.args + node.args.kwonlyargs
        if node.args.vararg: all_args.append(node.args.vararg)
        if node.args.kwarg: all_args.append(node.args.kwarg)

        for arg in all_args:
            var_name = arg.arg
            args.append(var_name)
            
            # Attempt to get type annotation
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
            # Mermaid node label
            label = "Input\\n" + "\\n".join(arg_labels)
            logic_node = LogicNode(node_id, label, node_type="input")
            self.nodes.append(logic_node)
            
            # Register these variables as produced by the Input node
            for arg in args:
                self.current_producers[arg] = node_id
                # Also register self.arg (simplified handling for common __init__ pattern)
                if 'self' in args:
                    self.current_producers[f"self.{arg}"] = node_id
        
        # Continue traversing the function body
        for item in node.body:
            self.visit(item)

    def visit_Assign(self, node):
        """
        Description
        ----------
        Handle simple assignment AST nodes by delegating to the assignment handler.

        Args
        -----
        node : ast.Assign
            The assignment AST node.

        Returns
        --------
        None
        """
        self._handle_assign(node, node.targets)

    def visit_AnnAssign(self, node):
        """
        Description
        ----------
        Handle annotated assignment (PEP 526) such as `x: int = value`.

        Args
        -----
        node : ast.AnnAssign
            Annotated assignment AST node.

        Returns
        --------
        None
        """
        # Handle annotated assignment: x: int = value
        if node.value:
            self._handle_assign(node, [node.target], annotation=node.annotation)

    def _handle_assign(self, node, targets, annotation=None):
        """
        Description
        ----------
        Core handler for assignment-like statements. Analyzes RHS inputs, determines operation label,
        creates a LogicNode representing the assignment/call/op, and updates producer mapping.

        Args
        -----
        node : ast.AST
            The original AST node for context (Assign or AnnAssign).
        targets : List[ast.AST]
            Left-hand side target nodes.
        annotation : Optional[ast.AST]
            Optional annotation for the target (for AnnAssign).

        Returns
        --------
        None

        Notes
        -------
        - Recognizes Calls, BinaryOps and Constants on the RHS to provide a richer label.
        - Supports simple target types: Name and self.Attribute.
        - Updates self.current_producers so later statements can resolve dependencies.
        """
        # 1. Analyze inputs (right-hand side)
        input_vars = self._extract_names(node.value)
        
        # 2. Determine operation label
        label = "Assign"
        if isinstance(node.value, ast.Call):
            func_name = self._get_func_name(node.value)
            label = f"Call: {func_name}"
        elif isinstance(node.value, ast.BinOp):
            op = type(node.value.op).__name__
            label = f"Op: {op}"
        elif isinstance(node.value, ast.Constant):
             label = f"Const: {node.value.value}"
        
        # 3. Analyze outputs (left-hand side)
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
                # Handle self.x = ...
                if isinstance(target.value, ast.Name) and target.value.id == 'self':
                    var_name = f"self.{target.attr}"
                    outputs.append(var_name)
                    output_labels.append(var_name)

        if outputs:
            node_id = self._get_id()
            full_label = f"{label}\\n⬇\\n{', '.join(output_labels)}"
            
            logic_node = LogicNode(node_id, full_label)
            # Establish edges: find the previous producers of input variables
            logic_node.edges_in = self._resolve_inputs(input_vars)
            
            self.nodes.append(logic_node)
            
            # Update producers table
            for out in outputs:
                self.current_producers[out] = node_id

    def visit_Expr(self, node):
        """
        Description
        ----------
        Handle expression statements, commonly used for standalone function calls with side-effects.

        Args
        -----
        node : ast.Expr
            The expression AST node.

        Returns
        --------
        None

        Notes
        -------
        - Creates a LogicNode for the call but does not update producers because such calls usually
          don't assign to variables.
        """
        # Handle standalone function calls (no assignment), e.g., print(), model.eval()
        if isinstance(node.value, ast.Call):
            input_vars = self._extract_names(node.value)
            func_name = self._get_func_name(node.value)
            
            node_id = self._get_id()
            logic_node = LogicNode(node_id, f"Call: {func_name}")
            logic_node.edges_in = self._resolve_inputs(input_vars)
            
            self.nodes.append(logic_node)
            # Such calls usually have side effects but no explicit return variables, so current_producers is not updated

    def visit_Return(self, node):
        """
        Description
        ----------
        Handle return statements by creating an output LogicNode that links to its input expressions.

        Args
        -----
        node : ast.Return
            The return statement AST node.

        Returns
        --------
        None

        Notes
        -------
        - Attempts to stringify the returned expression when possible for clearer labels.
        """
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
        """
        Description
        ----------
        Derive a human-readable function name for an ast.Call node.

        Args
        -----
        node : ast.Call
            The call AST node whose target name should be extracted.

        Returns
        --------
        str
            A best-effort string representing the function being called (e.g., "foo" or "obj.method").

        Notes
        -------
        - For attribute calls returns "<owner>.<attr>" if possible; otherwise returns a fallback "func".
        """
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return getattr(node.func.value, 'id', 'obj') + "." + node.func.attr
        return "func"

def generate_function_flowchart(func_obj) -> str:
    """
    Description
    ----------
    Generate a Mermaid data flow diagram for a single Python function using AST analysis.

    Args
    -----
    func_obj : callable
        The function object to analyze.

    Returns
    --------
    str
        Mermaid markup representing the function's data flow graph (empty string on failure).

    Notes
    -------
    - Uses inspect.getsource and ast.parse to build an AST, then visits it with AdvancedFlowVisitor.
    - Returns an empty string if source extraction or parsing fails (e.g., builtins or dynamic functions).
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

    # Build Mermaid
    lines = ["flowchart TD"] # Use top-down layout, suitable for showing flow
    
    # Style definitions
    lines.append("    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;")
    lines.append("    classDef process fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;")
    lines.append("    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,rx:10,ry:10;")
    
    # Draw nodes
    for node in visitor.nodes:
        # 1. Sanitize ID
        safe_node_id = sanitize_id(node.id)
        
        # 2. Escape special characters in Label
        # Escape double quotes to single quotes to prevent breaking Mermaid syntax
        safe_label = node.label.replace('"', "'")
        
        shape_start, shape_end = "(", ")"
        if node.node_type == "input": shape_start, shape_end = "[", "]"
        if node.node_type == "output": shape_start, shape_end = "([", "])"
        
        # Use quotes around Label to ensure special characters (like spaces, =) are displayed correctly
        lines.append(f'    {safe_node_id}{shape_start}"{safe_label}"{shape_end}:::{node.node_type}')
        
        for source_id, var_name in node.edges_in:
            safe_source_id = sanitize_id(source_id)
            # Edge labels also need to be sanitized to remove characters that might break syntax
            safe_var = var_name.replace('"', "'").replace('|', '/')
            lines.append(f'    {safe_source_id} -- "{safe_var}" --> {safe_node_id}')

    return "\n".join(lines)

# --- Helper: 2. Global Call Graph Analysis (Macro) ---

class GlobalCallGraphVisitor(ast.NodeVisitor):
    """
    Description
    ----------
    Analyze the entire module's AST to build a call graph between functions.
    Traverses the AST to find function definitions and function calls, linking callers to callees.
    """
    def __init__(self, known_functions: Set[str]):
        """
        Description
        ----------
        Initialize the visitor with a set of known internal functions.

        Args
        -----
        known_functions : Set[str]
            A set of function names defined within the library being analyzed. Used to filter out built-in or external calls.

        Returns
        --------
        None

        Notes
        -------
        - Initializes `self.calls` to store the graph edges.
        - Initializes `self.dependency_map` for closure calculation.
        - Sets `self.current_function` to "Main_Script" to capture top-level calls.
        """
        self.known_functions = known_functions # Set of all function names defined in the library
        self.calls = [] # List of (caller, callee, arg_names)
        self.current_function = "Main_Script" # Default to top-level script
        # New: adjacency list for dependency closure calculation
        self.dependency_map = defaultdict(set) # caller -> set(callees)

    def visit_FunctionDef(self, node):
        """
        Description
        ----------
        Handle function definitions to track the current caller context.

        Args
        -----
        node : ast.FunctionDef
            The function definition AST node.

        Returns
        --------
        None

        Notes
        -------
        - Updates `self.current_function` to the name of the function being visited.
        - Restores the previous function name after visiting the body (handling nested functions).
        """
        prev_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = prev_function

    def visit_Call(self, node):
        """
        Description
        ----------
        Handle function calls to record dependencies between the current function and the callee.

        Args
        -----
        node : ast.Call
            The function call AST node.

        Returns
        --------
        None

        Notes
        -------
        - Extracts the callee name (handling simple names and attributes like `self.method`).
        - Extracts argument names for visualization.
        - Filters calls: only records calls to functions in `known_functions` or `self.*` calls.
        - Updates `self.calls` and `self.dependency_map`.
        """
        # Extract the name of the called function
        callee_name = ""
        if isinstance(node.func, ast.Name):
            callee_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handle self.method() or module.func()
            callee_name = node.func.attr
        
        if callee_name:
            # Extract argument names (for data flow visualization)
            args = []
            for arg in node.args:
                if isinstance(arg, ast.Name):
                    args.append(arg.id)
            
            # Only record calls to functions defined in our library (to avoid including built-ins like print, len)
            # Or if it's a self.xxx call, we also record it (assuming it's an internal class call)
            if callee_name in self.known_functions or (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'self'):
                self.calls.append((self.current_function, callee_name, args))
                self.dependency_map[self.current_function].add(callee_name)
        
        self.generic_visit(node)

# --- Helper: 3. Entry Analysis (Navigator: How to Drive) ---
class EntryAnalysisVisitor(ast.NodeVisitor):
    """
    Description
    ----------
    Analyze entry points in the code, especially the use of argparse.
    """
    def __init__(self):
        """
        Description
        ----------
        Initialize visitor state for entry point analysis.

        Args
        -----
        None

        Returns
        --------
        None

        Notes
        -------
        - Initializes `self.has_main_block` to False.
        - Initializes `self.args` to store discovered CLI arguments.
        """
        self.has_main_block = False
        self.args = [] # List of (arg_name, help_text)

    def visit_If(self, node):
        """
        Description
        ----------
        Detect `if __name__ == "__main__":` blocks to identify script entry points.

        Args
        -----
        node : ast.If
            The If statement AST node.

        Returns
        --------
        None

        Notes
        -------
        - Sets `self.has_main_block` to True if the standard main guard is found.
        """
        # Detect if __name__ == "__main__":
        try:
            if (isinstance(node.test, ast.Compare) and 
                isinstance(node.test.left, ast.Name) and 
                node.test.left.id == "__name__" and 
                isinstance(node.test.comparators[0], ast.Constant) and 
                node.test.comparators[0].value == "__main__"):
                self.has_main_block = True
        except: pass
        self.generic_visit(node)

    def visit_Call(self, node):
        """
        Description
        ----------
        Detect `parser.add_argument(...)` calls to extract CLI argument definitions.

        Args
        -----
        node : ast.Call
            The function call AST node.

        Returns
        --------
        None

        Notes
        -------
        - Extracts the argument name (e.g., '--input') and help text.
        - Appends found arguments to `self.args`.
        """
        # Detect parser.add_argument(...)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            arg_name = "Unknown"
            help_text = ""
            
            # Extract argument name (usually the first argument)
            if node.args:
                if isinstance(node.args[0], ast.Constant):
                    arg_name = node.args[0].value
            
            # Extract help information
            for kw in node.keywords:
                if kw.arg == 'help' and isinstance(kw.value, ast.Constant):
                    help_text = kw.value.value
            
            self.args.append((arg_name, help_text))
        
        self.generic_visit(node)

# --- Helper: 4. Dependency Closure Calculation (Navigator: Extraction Guide) ---
def get_dependency_closure(target_func: str, dependency_map: Dict[str, Set[str]]) -> Set[str]:
    """
    Description
    ----------
    Calculate the dependency closure of the target function (i.e., all other functions required to run it).

    Args
    -----
    target_func : str
        The name of the function to analyze.
    dependency_map : Dict[str, Set[str]]
        Adjacency list representing the call graph (caller -> callees).

    Returns
    --------
    Set[str]
        A set of function names representing the transitive closure of dependencies.

    Notes
    -------
    - Uses Breadth-First Search (BFS) to traverse the dependency graph.
    """
    closure = set()
    queue = deque([target_func])
    visited = set()

    while queue:
        current = queue.popleft()
        if current in visited: continue
        visited.add(current)
        closure.add(current)

        if current in dependency_map:
            for dep in dependency_map[current]:
                if dep not in visited:
                    queue.append(dep)
    return closure

# --- Helper: 5. Module Classifier (Navigator: Architecture) ---
def classify_module(module_obj) -> str:
    """
    Description
    ----------
    Infer the role of a module based on the external libraries it imports or keywords in source.

    Args
    -----
    module_obj : module
        The module object to classify.

    Returns
    --------
    str
        A string category (e.g., "Model / AI", "Web / API", "Utility / Core").

    Notes
    -------
    - Scans source code for specific library imports (torch, flask, pandas, etc.).
    - Returns "Unknown" if source cannot be retrieved.
    """
    try:
        source = inspect.getsource(module_obj)
    except:
        return "Unknown"
    
    # Simple keyword matching
    if "torch" in source or "tensorflow" in source or "keras" in source:
        return "Model / AI"
    if "flask" in source or "django" in source or "fastapi" in source:
        return "Web / API"
    if "pandas" in source or "numpy" in source or "csv" in source:
        return "Data Processing"
    if "matplotlib" in source or "seaborn" in source or "plotly" in source:
        return "Visualization"
    if "argparse" in source or "click" in source:
        return "Interface / CLI"
    
    return "Utility / Core"


def generate_global_call_graph(modules: List[Any], library_name: str) -> Tuple[str, Dict[str, Set[str]]]:
    """
    Description
    ----------
    Generate a global function call graph and return the dependency mapping.

    Args
    -----
    modules : List[Any]
        List of module objects to analyze.
    library_name : str
        The name of the library being inspected (used for filtering).

    Returns
    --------
    Tuple[str, Dict[str, Set[str]]]
        - str: Mermaid graph definition string.
        - Dict[str, Set[str]]: Dependency map (caller -> set of callees).

    Notes
    -------
    - 1. Collects all defined function names to create a whitelist.
    - 2. Traverses all source code using `GlobalCallGraphVisitor`.
    - 3. Constructs a Mermaid graph string with proper styling and ID sanitization.
    """
    # 1. Collect all defined function names (whitelist)
    known_functions = set()
    for mod in modules:
        for name, obj in inspect.getmembers(mod):
            if inspect.isfunction(obj) or inspect.ismethod(obj):
                known_functions.add(name)
            elif inspect.isclass(obj):
                for m_name, m_obj in inspect.getmembers(obj):
                    if inspect.isfunction(m_obj) or inspect.ismethod(m_obj):
                        known_functions.add(m_name)

    # 2. Traverse all source code for AST analysis
    visitor = GlobalCallGraphVisitor(known_functions)
    
    for mod in modules:
        try:
            source = inspect.getsource(mod)
            tree = ast.parse(source)
            visitor.visit(tree)
        except Exception:
            continue

    if not visitor.calls:
        return "", {}

    # 3. Construct Mermaid graph
    lines = ["graph TD"]
    lines.append("    classDef main fill:#f9f,stroke:#333,stroke-width:2px;")
    lines.append("    classDef func fill:#fff,stroke:#333,stroke-width:1px;")
    
    edges = set()
    
    for caller, callee, args in visitor.calls:
        # Ignore recursive calls
        if caller == callee: continue
        
        # [Key Fix] Generate a secure ID
        caller_id = sanitize_id(caller)
        callee_id = sanitize_id(callee)

        # Format edge
        edge_label = ""
        if args:
            # Truncate long argument lists to prevent graph explosion
            arg_str = '<br>'.join(args) 
            # if len(arg_str) > 20:
                # arg_str = arg_str[:17] + "..." # ！！！！！！！！！！！
            # Remove characters that may break Mermaid syntax
            arg_str = arg_str.replace('"', "'").replace('|', '/')
            edge_label = f"|{arg_str}|"
        
        # Use ID[Label] format
        # This way, the ID is safe (no dots), and the label can contain dots
        edge_str = f'    {caller_id}["{caller}"] -->{edge_label} {callee_id}["{callee}"]'
        
        if edge_str not in edges:
            edges.add(edge_str)
            lines.append(edge_str)
            
            # Apply styles to IDs
            if caller == "main" or caller == "Main_Script":
                lines.append(f"    class {caller_id} main;")
            else:
                lines.append(f"    class {caller_id} func;")
            lines.append(f"    class {callee_id} func;")

    return "\n".join(lines), visitor.dependency_map


def convert_md_to_html(md_content: str, title: str) -> str:
    """
    Description
    ----------
    Convert Markdown content to HTML with Mermaid rendering support.
    Fixes tag escaping issues to ensure Mermaid code is correctly parsed.

    Args
    -----
    md_content : str
        The raw Markdown content string.
    title : str
        The title for the generated HTML page.

    Returns
    --------
    str
        A complete HTML string containing the rendered content and Mermaid.js integration.

    Notes
    -------
    - Splits content by code blocks to handle escaping differently for text vs code.
    - Manually restores specific HTML tags (details, summary) used in the report.
    - Escapes Mermaid diagram code to prevent browser parsing errors before Mermaid.js runs.
    """
    parts = md_content.split("```")
    final_html_body = []
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # === Plain text block ===
            # 1. First, perform HTML escaping to prevent < > in the text from breaking the page structure
            text = html.escape(part)
            
            # 2. [Key Fix] Restore the specific HTML tags we generated
            # Because we manually added these tags in inspect_library, we need to "unescape" them here
            text = text.replace("&lt;details&gt;", "<details>")
            text = text.replace("&lt;/details&gt;", "</details>")
            text = text.replace("&lt;summary&gt;", "<summary>")
            text = text.replace("&lt;/summary&gt;", "</summary>")
            
            # 3. Simple Markdown formatting
            lines = text.split('\n')
            formatted_lines = []
            for line in lines:
                if line.startswith('# '): formatted_lines.append(f"<h1>{line[2:]}</h1>")
                elif line.startswith('## '): formatted_lines.append(f"<h2>{line[3:]}</h2>")
                elif line.startswith('### '): formatted_lines.append(f"<h3>{line[4:]}</h3>")
                elif line.startswith('#### '): formatted_lines.append(f"<h4>{line[5:]}</h4>")
                elif line.startswith('**') and line.endswith('**'): formatted_lines.append(f"<b>{line[2:-2]}</b><br>")
                # Note: after html.escape, > becomes &gt;
                elif line.startswith('&gt; '): formatted_lines.append(f"<blockquote>{line[5:]}</blockquote>")
                elif line.startswith('|'): formatted_lines.append(f"<p style='font-family:monospace; white-space: pre;'>{line}</p>")
                else: formatted_lines.append(f"{line}<br>")
            
            final_html_body.append("\n".join(formatted_lines))
        else:
            # === Code block ===
            if part.startswith("mermaid"):
                # Mermaid diagram
                graph_code = part[7:].strip()
                
                # [Key Fix] Perform HTML escaping on Mermaid code
                # This way, > in A-->B becomes &gt;, <br> becomes &lt;br&gt;
                # After the browser parses the HTML, the Mermaid engine reads the original A-->B and <br> strings
                # This perfectly solves the Syntax error problem
                escaped_code = html.escape(graph_code)
                
                final_html_body.append(f'<div class="mermaid" style="overflow-x: auto;">\n{escaped_code}\n</div>')
            else:
                # Plain code
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
        
        /* Optimize edge label styles */
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

    <!-- Import Mermaid.js -->
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ 
            startOnLoad: true, 
            maxTextSize: 1000000,  
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
    """
    Description
    ----------
    Main entry point to inspect a Python library.
    Performs dynamic import, AST analysis, dependency graph construction, and report generation.

    Args
    -----
    library_name : str
        The importable name of the library to inspect (e.g., "pandas", "requests").
    output_path : Optional[str]
        Path to save the generated Markdown report. If None, prints to stdout.
    include_private : bool
        If True, includes private members (starting with _) in the report.
    include_imported : bool
        If True, includes members imported from other libraries in the API list.

    Returns
    --------
    None

    Notes
    -------
    - Phase 0: Navigator - Analyzes entry points (CLI or API) and suggests usage.
    - Phase 1: Network Analysis - Builds dependency graphs (internal & external).
    - Phase 2: API Inspection - Lists functions/classes and generates logic flowcharts.
    - Generates both Markdown and HTML reports if output_path is provided.
    """
    # --- 1. Dynamically import the main library (with sys.argv protection) ---
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
                # --- 1. Skip test modules (key modification) ---
                # Libraries like pandas contain many test files that often reference missing dependencies (e.g., pyarrow), causing crashes
                # Test code is usually not the focus of architecture analysis
                mod_parts = modname.split('.')
                if 'tests' in mod_parts or 'test' in mod_parts or 'conftest' in mod_parts:
                    continue

                try:
                    sub_mod = importlib.import_module(modname)
                    submodules.append(sub_mod)
                except (KeyboardInterrupt, SystemExit):
                    # Allow users to interrupt with Ctrl+C
                    raise
                except BaseException: 
                    # --- 2. Catch BaseException (key modification) ---
                    # Exceptions raised by pytest.skip() inherit from BaseException, not Exception
                    # This allows ignoring import errors caused by missing optional dependencies
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
    # Phase 0: Navigator - How to Drive (New)
    # ==========================================
    lines.append("## 🚦 Navigator: How to Drive")
    lines.append("This section helps you understand how to run this library from the command line or entry points.")
    
    entry_visitor = EntryAnalysisVisitor()
    try:
        source = inspect.getsource(main_module)
        entry_visitor.visit(ast.parse(source))
    except: pass

    if entry_visitor.has_main_block:
        lines.append("- ✅ **Entry Point Detected**: This module contains an `if __name__ == '__main__':` block, meaning it can be run directly.")
    else:
        lines.append("- ℹ️ **No Direct Entry Point**: This module seems to be a library intended for import, not direct execution.")
        
        # --- New: Intelligent navigation for import libraries ---
        lines.append("\n### 🐍 Python API Usage (Inferred)")
        lines.append("Since no CLI entry point was found, here are the likely **Python API entry points** for your script:")
        
        # 1. Pre-build call graph to assist judgment (partially run Phase 1 logic in advance)
        # We need to know which functions are "Top Level" (not called internally)
        temp_known_funcs = set()
        temp_calls = []
        if hasattr(main_module, "__path__"):
             # Simple scan of the main module and direct submodules
            scan_modules = [main_module] + submodules # Limit number to avoid being too slow submodules[:10]
        else:
            scan_modules = [main_module]

        for m in scan_modules:
            for n, o in inspect.getmembers(m):
                if inspect.isfunction(o) or inspect.ismethod(o): temp_known_funcs.add(n)
        
        temp_visitor = GlobalCallGraphVisitor(temp_known_funcs)
        for m in scan_modules:
            try: temp_visitor.visit(ast.parse(inspect.getsource(m)))
            except: pass
            
        # Calculate in-degree (number of times called)
        in_degree = Counter()
        for caller, callee, _ in temp_visitor.calls:
            in_degree[callee] += 1

        # 2. Find potential API entry points
        api_candidates = []
        candidates_source = []
        
        if hasattr(main_module, "__all__"):
            candidates_source = main_module.__all__
        else:
            candidates_source = [x for x in dir(main_module) if not x.startswith("_")]
            
        for name in candidates_source:
            try:
                obj = getattr(main_module, name)
                # Only focus on functions and classes
                if inspect.isfunction(obj):
                    api_candidates.append((name, "Function", obj))
                elif inspect.isclass(obj):
                    # Ignore exception classes
                    if issubclass(obj, Exception): continue
                    api_candidates.append((name, "Class", obj))
            except: continue
            
        # 3. Enhanced scoring mechanism
        def score_candidate(item):
            """
            Description
            ----------
            Calculate a relevance score for an API candidate to determine if it's a likely entry point.

            Args
            -----
            item : Tuple[str, str, Any]
                A tuple containing (name, kind, object), where kind is 'Function' or 'Class'.

            Returns
            --------
            int
                A score indicating the likelihood of being a main entry point (higher is better).

            Notes
            -------
            - Scoring factors:
              - Keywords in name (verbs like predict, run, load).
              - Class names (Model, Engine).
              - Topological usage (low in-degree suggests public API).
              - Code complexity (lines of code).
            """
            name, kind, obj = item
            score = 0
            name_lower = name.lower()
            
            # A. Keyword scoring (business logic priority)
            # Core verbs
            if any(v in name_lower for v in ['predict', 'calculate', 'analyze', 'solve', 'run', 'process', 'train', 'evaluate', 'generate']): score += 15
            # Entry verbs
            elif any(v in name_lower for v in ['main', 'start', 'init', 'load', 'create', 'make']): score += 10
            # Conversion/tool verbs
            elif any(v in name_lower for v in ['convert', 'parse', 'read', 'write', 'save', 'plot', 'show']): score += 5
            
            # B. Type scoring
            if kind == 'Class':
                # Model/core class bonus
                if any(n in name_lower for n in ['model', 'engine', 'client', 'api', 'runner', 'predictor']): score += 8
            
            # C. Topological scoring (key improvement)
            # If a function is public and rarely called internally (low in-degree), it is likely intended for external use
            if kind == 'Function':
                degree = in_degree.get(name, 0)
                if degree == 0: score += 10  # Pure top-level interface
                elif degree < 3: score += 5  # Low coupling interface
                else: score -= 5             # Heavily called internally, likely a low-level utility function
            
            # D. Complexity scoring (lines of code)
            try:
                lines_of_code = len(inspect.getsource(obj).splitlines())
                if lines_of_code > 50: score += 5 # Complex logic often indicates main interface
                elif lines_of_code < 3: score -= 5 # Wrappers usually have only one or two lines
            except: pass

            return score

        api_candidates.sort(key=score_candidate, reverse=True)
        
        # 4. Generate code snippets

        '''
        if api_candidates: 
            lines.append("```python")
            lines.append(f"import {library_name}")
            lines.append("")
            lines.append("# Likely entry points (Ranked by relevance):")
            # Show top 8 for better coverage
            for name, kind, obj in api_candidates: 
                try:
                    sig = str(inspect.signature(obj))
                except: sig = "(...)"
                
                # Add a brief comment line (take the first line of the docstring)
                doc = inspect.getdoc(obj)
                doc_summary = f"  # {doc.splitlines()[0]}" if doc else ""
                # Truncate overly long comments
                if len(doc_summary) > 60: doc_summary = doc_summary[:57] + "..."
                
                lines.append(f"# {kind}: {library_name}.{name}{sig}{doc_summary}")
            lines.append("```")
        else:
            lines.append("_No obvious public API members detected._")
        '''

        # --- New: User-friendly formatted output ---
        if api_candidates:
            lines.append("\n#### 🚀 Top Recommended Entry Points")
            lines.append("| Type | API | Description |")
            lines.append("| :--- | :--- | :--- |")
            
            count = 0
            for name, kind, obj in api_candidates:
                # Filter out candidates with low scores (likely low-level utilities)
                if score_candidate((name, kind, obj)) < 0 and count > 5: continue
                # if count >= 10: break # Show up to 10 core APIs
                
                # 1. Extract signature and beautify parameters
                try:
                    sig = inspect.signature(obj)
                    params = []
                    for p_name, p in sig.parameters.items():
                        if p.default == inspect.Parameter.empty:
                            # Required parameters in bold
                            params.append(f"**{p_name}**")
                        else:
                            # Optional parameters in regular font
                            params.append(f"{p_name}")
                    
                    # Reassemble signature to avoid excessive length
                    sig_str = f"({', '.join(params)})"
                    # if len(sig_str) > 60: # If parameters are too long, truncate
                    #    sig_str = f"({', '.join(params[:3])}, ...)"
                except: 
                    sig_str = "(...)"

                # 2. Extract and clean documentation
                doc = inspect.getdoc(obj)
                doc_summary = doc.splitlines()[0] if doc else "No description."
                # if len(doc_summary) > 80: doc_summary = doc_summary[:77] + "..."
                
                # 3. Icon differentiation
                icon = "ƒ" if kind == "Function" else "C"
                
                lines.append(f"| `{icon}` | **{library_name}.{name}**{sig_str} | {doc_summary} |")
                count += 1
            
            lines.append("\n> **Note:** Bold parameters are required. Others are optional.")
            
            # --- New: Multi-dimensional code snippet generation ---
            lines.append("\n#### 🧩 Code Snippets (Auto-Generated)")
            lines.append("```python")
            lines.append(f"import {library_name}")
            lines.append("")
            
            # 1. Extract Top Functions (up to 10)
            top_funcs = [x for x in api_candidates if x[1] == 'Function'][:10]
            if top_funcs:
                lines.append("# --- Top Ranked Functions ---")
                for i, (name, _, obj) in enumerate(top_funcs):
                    try:
                        sig = inspect.signature(obj)
                        args = []
                        for p_name, p in sig.parameters.items():
                            if p.default == inspect.Parameter.empty and p_name != 'self':
                                args.append(f"{p_name}=...") 
                            elif p.default != inspect.Parameter.empty:
                                # Optional parameters are commented out to indicate their presence
                                # args.append(f"{p_name}={p.default}") 
                                pass
                        
                        # If there are too many parameters, display them on multiple lines
                        if len(args) > 3:
                            args_str = ",\n    ".join(args)
                            call_str = f"{library_name}.{name}(\n    {args_str}\n)"
                        else:
                            call_str = f"{library_name}.{name}({', '.join(args)})"
                            
                        lines.append(f"# {i+1}. {name}")
                        lines.append(f"result_{i+1} = {call_str}")
                        lines.append("")
                    except: pass

            # 2. Extract Top Classes (up to 10)
            top_classes = [x for x in api_candidates if x[1] == 'Class'][:10]
            if top_classes:
                lines.append("# --- Core Classes Initialization ---")
                for i, (name, _, obj) in enumerate(top_classes):
                    try:
                        # Attempt to get the signature of __init__
                        sig = inspect.signature(obj)
                        args = []
                        for p_name, p in sig.parameters.items():
                            if p.default == inspect.Parameter.empty and p_name != 'self':
                                args.append(f"{p_name}=...")
                        
                        call_str = f"{library_name}.{name}({', '.join(args)})"
                        instance_name = name.lower().replace("model", "_model").replace("class", "_obj")
                        
                        lines.append(f"# {i+1}. {name}")
                        lines.append(f"{instance_name} = {call_str}")
                        lines.append("")
                    except: pass
            
            lines.append("```")

        else:
            lines.append("_No obvious public API members detected._")

    if entry_visitor.args:
        lines.append("\n### ⌨️ CLI Arguments (Detected)")
        lines.append("| Argument | Help Text |")
        lines.append("| :--- | :--- |")
        for arg, help_text in entry_visitor.args:
            lines.append(f"| `{arg}` | {help_text} |")
    else:
        lines.append("\n_No explicit `argparse` configuration detected in the main module._")
    lines.append("\n")

    # ==========================================
    # Phase 1: Network Construction & Analysis
    # ==========================================

    G = nx.DiGraph() if HAS_NETWORKX else None
    internal_modules_rank = Counter() 
    external_libs_rank = Counter()    
    dependency_graph = defaultdict(set) 
    module_roles = {} # module_name -> role

    for mod in submodules:
        current_mod_name = mod.__name__
        
        # Module role classification
        role = classify_module(mod)
        module_roles[current_mod_name] = role

        if HAS_NETWORKX:
            G.add_node(current_mod_name, type='internal', role=role)
        
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
    
    sorted_pr = []
    if HAS_NETWORKX and len(G.nodes) > 0:
        lines.append("### 🕸️ Network Metrics (Advanced)")
        try:
            pagerank = nx.pagerank(G, alpha=0.85)
            sorted_pr = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)
            lines.append("#### 👑 Top Modules by PageRank (Authority)")
            lines.append("| Rank | Module | Score | Type | Role |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for i, (node, score) in enumerate(sorted_pr[:10]):
                node_type = "Internal" if node.startswith(library_name) else "External"
                role = module_roles.get(node, "External Lib")
                short_name = node.replace(library_name + ".", "")
                lines.append(f"| {i+1} | `{short_name}` | {score:.4f} | {node_type} | {role} |")
            lines.append("\n")
        except Exception: pass

    lines.append("### 🗺️ Dependency & Architecture Map")
    mermaid_lines = ["graph TD"]
    mermaid_lines.append("    classDef core fill:#f96,stroke:#333,stroke-width:2px;")
    mermaid_lines.append("    classDef external fill:#9cf,stroke:#333,stroke-width:1px;")
    
    if HAS_NETWORKX: top_nodes = set(n for n, s in sorted_pr[:20])
    else: top_nodes = set(x[0] for x in internal_modules_rank.most_common(20))
    
    # --- New: Use pure numeric ID mapping ---
    # 1. Collect all node names to be drawn
    nodes_to_map = set()
    
    # Collect nodes from dependency relationships
    source_data = G.edges() if HAS_NETWORKX else []
    if not HAS_NETWORKX:
        for src, targets in dependency_graph.items():
            for tgt in targets: source_data.append((src, tgt))
            
    dependency_edges = []
    for u, v in source_data:
        if u in top_nodes or v in top_nodes:
            # Simplify naming logic
            short_u = u.replace(library_name + ".", "").split('.')[-1]
            short_v = v.replace(library_name + ".", "").split('.')[-1]
            if not v.startswith(library_name): short_v = v.split('.')[0]
            
            if short_u == short_v: continue
            
            nodes_to_map.add(u)
            nodes_to_map.add(v)
            dependency_edges.append((u, v, short_u, short_v))

    # Collect nodes from inheritance relationships
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
                
                nodes_to_map.add(name)
                nodes_to_map.add(base_name)
                
                base_module = base.__module__.split('.')[0]
                inheritance_edges.append((name, base_name, base_module))

    # 2. Build ID mapping table
    id_map = {name: f"id_{i}" for i, name in enumerate(nodes_to_map)}

    # 3. Draw dependency relationships
    edges_drawn = set()
    for u, v, label_u, label_v in dependency_edges:
        uid = id_map[u]
        vid = id_map[v]
        
        edge_key = f"{uid}->{vid}"
        if edge_key in edges_drawn: continue
        edges_drawn.add(edge_key)
        
        arrow = "-.->" if not v.startswith(library_name) else "-->"
        mermaid_lines.append(f'    {uid}["{label_u}"] {arrow} {vid}["{label_v}"]')
        
        if u.startswith(library_name): mermaid_lines.append(f"    class {uid} core;")
        else: mermaid_lines.append(f"    class {uid} external;")
        
        if v.startswith(library_name): mermaid_lines.append(f"    class {vid} core;")
        else: mermaid_lines.append(f"    class {vid} external;")

    # 4. Draw inheritance relationships
    for cls_name, base_name, base_mod in inheritance_edges:
        cid = id_map[cls_name]
        bid = id_map[base_name]
        
        mermaid_lines.append(f'    {cid}["{cls_name}"] ==> {bid}["{base_name}"]')
        mermaid_lines.append(f"    class {cid} core;")
        
        if base_mod != library_name:
            mermaid_lines.append(f"    class {bid} external;")
        else:
            mermaid_lines.append(f"    class {bid} core;")

    lines.append("```mermaid")
    lines.append("\n".join(mermaid_lines))
    lines.append("```\n")

    # ==========================================
    # Phase 1.5: Global Call Graph & Extraction Guide (New)
    # ==========================================
    lines.append("## 🚀 Global Execution Flow & Extraction Guide")
    lines.append("This graph visualizes how data flows between functions across the entire project.")
    
    global_call_graph, dependency_map = generate_global_call_graph(submodules, library_name)
    if global_call_graph:
        lines.append("```mermaid")
        lines.append(global_call_graph)
        lines.append("```\n")
        
        # --- Navigator: Extraction Guide ---
        lines.append("### ✂️ Navigator: Snippet Extractor")
        lines.append("Want to use a specific function without the whole library? Here is the **Dependency Closure** for key functions.")
        
        # Select several important functions for analysis (if PageRank is available, select the top-ranked; otherwise, select the most called)
        # For simplicity, select the top 3 functions that appear most frequently as callers in the dependency_map
        top_funcs = sorted(dependency_map.keys(), key=lambda k: len(dependency_map[k]), reverse=True)[:3]
        
        if top_funcs:
            for func in top_funcs:
                closure = get_dependency_closure(func, dependency_map)
                lines.append(f"#### To extract `{func}`:")
                lines.append(f"> You need these **{len(closure)}** components:")
                lines.append(f"`{', '.join(sorted(list(closure)))}`")
                lines.append("")
        else:
            lines.append("_Not enough call data to generate extraction guide._")

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
        """
        Description
        ----------
        Extract signature and docstring summary for a Python object.

        Args
        -----
        obj : Any
            The function or class object to inspect.

        Returns
        --------
        Tuple[str, str]
            A tuple containing (signature_string, docstring_summary).

        Notes
        -------
        - Handles cases where signature extraction fails (e.g., built-ins) by falling back to `__text_signature__` or default.
        """
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
        # 1. Save Markdown (original logic)
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
            
            # 2. (New) Automatically generate HTML version
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

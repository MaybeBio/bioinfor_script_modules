# 单函数分析逻辑

# 1, v1
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




===============================================================================================================


# 2, v2
# --- Helper: 1. 单函数逻辑分析 (微观) ---
class LogicNode:
    """表示流程图中的一个节点"""
    def __init__(self, id, label, node_type="process", detail=None):
        self.id = id
        self.label = label
        self.node_type = node_type # input, process, output, decision, loop
        self.detail = detail # 额外的类型或描述信息
        self.edges_in = [] # List of (source_id, var_name)

class AdvancedFlowVisitor(ast.NodeVisitor):
    """
    解析函数源码，构建数据流向图。
    重点增强：类型提取、关键算子识别、控制流可视化。
    """
    def __init__(self):
        self.nodes = []
        self.current_producers = {} # var_name -> node_id
        self.counter = 0
        self.in_control_flow = False # 标记是否在 if/loop 内部

    def _get_id(self):
        self.counter += 1
        return f"Node{self.counter}"

    def _resolve_inputs(self, input_vars: List[str]) -> List[Tuple[str, str]]:
        edges = []
        for var in input_vars:
            if var in self.current_producers:
                source_id = self.current_producers[var]
                edges.append((source_id, var))
        return edges

    def _extract_names(self, node) -> List[str]:
        names = []
        class NameCollector(ast.NodeVisitor):
            def visit_Name(self, n):
                if isinstance(n.ctx, ast.Load):
                    names.append(n.id)
            def visit_Attribute(self, n):
                if isinstance(n.value, ast.Name) and n.value.id == 'self':
                    names.append(f"self.{n.attr}")
                self.generic_visit(n)
        if node: NameCollector().visit(node)
        return list(set(names))

    def visit_FunctionDef(self, node):
        # 1. Input Node (增强类型显示)
        args = []
        arg_labels = []
        
        all_args = node.args.args + node.args.kwonlyargs
        if node.args.vararg: all_args.append(node.args.vararg)
        if node.args.kwarg: all_args.append(node.args.kwarg)

        for arg in all_args:
            var_name = arg.arg
            args.append(var_name)
            ann = ""
            if arg.annotation:
                try:
                    if hasattr(ast, 'unparse'): ann = ": " + ast.unparse(arg.annotation)
                    else: ann = ": " + str(arg.annotation)
                except: pass
            arg_labels.append(f"{var_name}{ann}")
            
        if args:
            node_id = "Input"
            # 使用 HTML 标签增强显示效果 (Mermaid 支持)
            label = "<b>Input Data</b><br/>" + "<br/>".join([f"• {l}" for l in arg_labels])
            logic_node = LogicNode(node_id, label, node_type="input")
            self.nodes.append(logic_node)
            
            for arg in args:
                self.current_producers[arg] = node_id
                if 'self' in args: self.current_producers[f"self.{arg}"] = node_id
        
        # 遍历函数体
        for item in node.body:
            self.visit(item)

    def visit_Assign(self, node):
        self._handle_assign(node, node.targets)

    def visit_AnnAssign(self, node):
        if node.value:
            self._handle_assign(node, [node.target], annotation=node.annotation)

    def _handle_assign(self, node, targets, annotation=None):
        input_vars = self._extract_names(node.value)
        
        # --- 智能过滤与标签生成 ---
        label = "Assign"
        node_type = "process"
        
        # 1. 函数调用 (重点关注)
        if isinstance(node.value, ast.Call):
            func_name = self._get_func_name(node.value)
            # 忽略简单的类型转换，如 int(), str(), float()
            if func_name in ['int', 'str', 'float', 'list', 'tuple', 'len']:
                return 
            
            label = f"<b>Call:</b> {func_name}"
            # 如果是 torch/numpy/model 调用，标记为核心计算
            if any(x in func_name for x in ['torch', 'np', 'model', 'predict', 'forward']):
                node_type = "core_process"

        # 2. 运算操作
        elif isinstance(node.value, ast.BinOp):
            op = type(node.value.op).__name__
            label = f"<b>Op:</b> {op}"
        
        # 3. 常量赋值 (通常忽略，除非是配置)
        elif isinstance(node.value, ast.Constant):
            return 

        # --- 输出变量处理 ---
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
                if isinstance(target.value, ast.Name) and target.value.id == 'self':
                    var_name = f"self.{target.attr}"
                    outputs.append(var_name)
                    output_labels.append(var_name)

        if outputs:
            node_id = self._get_id()
            # 组合 Label：操作 + 结果
            full_label = f"{label}<br/>⬇<br/>" + ", ".join(output_labels)
            
            logic_node = LogicNode(node_id, full_label, node_type=node_type)
            logic_node.edges_in = self._resolve_inputs(input_vars)
            self.nodes.append(logic_node)
            
            for out in outputs:
                self.current_producers[out] = node_id

    # --- 新增：控制流可视化 ---
    def visit_If(self, node):
        # 简单展示条件判断
        test_code = "Condition"
        if hasattr(ast, 'unparse'):
            try: test_code = ast.unparse(node.test)
            except: pass
            
        node_id = self._get_id()
        label = f"<b>Decision</b><br/>If {test_code}?"
        logic_node = LogicNode(node_id, label, node_type="decision")
        
        # 尝试连接输入的变量
        input_vars = self._extract_names(node.test)
        logic_node.edges_in = self._resolve_inputs(input_vars)
        self.nodes.append(logic_node)
        
        # 递归访问子节点 (简化处理，不完全模拟分支合并，只展示流程存在)
        self.generic_visit(node)

    def visit_Return(self, node):
        input_vars = []
        ret_str = "None"
        if node.value:
            input_vars = self._extract_names(node.value)
            if hasattr(ast, 'unparse'):
                try: ret_str = ast.unparse(node.value)
                except: pass
            else: ret_str = "Expression"
        
        node_id = "Return"
        # 增强 Return 节点的显示
        label = f"<b>Output / Return</b><br/>{ret_str}"
        logic_node = LogicNode(node_id, label, node_type="output")
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
    lines = ["flowchart TD"]
    
    # 优化样式：更现代、更柔和的配色
    lines.append("    classDef input fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,rx:5,ry:5;")
    lines.append("    classDef process fill:#fff,stroke:#bdbdbd,stroke-width:1px;")
    lines.append("    classDef core_process fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,rx:5,ry:5;") # 核心计算高亮
    lines.append("    classDef decision fill:#f3e5f5,stroke:#7b1fa2,stroke-width:1px,rx:5,ry:5,stroke-dasharray: 5 5;")
    lines.append("    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,rx:5,ry:5;")
    
    # 绘制节点
    for node in visitor.nodes:
        safe_node_id = sanitize_id(node.id)
        # 允许 HTML 标签
        safe_label = node.label.replace('"', "'")
        
        shape_start, shape_end = "[", "]"
        if node.node_type == "input": shape_start, shape_end = "([", "])"
        if node.node_type == "output": shape_start, shape_end = "([", "])"
        if node.node_type == "decision": shape_start, shape_end = "{{", "}}"
        
        lines.append(f'    {safe_node_id}{shape_start}"{safe_label}"{shape_end}:::{node.node_type}')
        
        for source_id, var_name in node.edges_in:
            safe_source_id = sanitize_id(source_id)
            safe_var = var_name.replace('"', "'").replace('|', '/')
            # 虚线连接 Decision，实线连接数据流
            arrow = "-.->" if "Decision" in source_id else "-->"
            lines.append(f'    {safe_source_id} {arrow} "{safe_var}" {safe_node_id}')

    return "\n".join(lines)

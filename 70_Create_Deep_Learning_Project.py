# 新建1个深度学习项目
# 文件组织参考:
'''
project_root/
│
├── data/                   # 存放原始数据和处理后的数据
│   ├── raw/
│   └── processed/
│
├── configs/                # 配置文件 (超参数、路径)
│   └── config.yaml
│
├── src/                    # 核心源代码
│   ├── __init__.py
│   ├── dataset.py          # 1. 数据处理 (Data Loader)
│   ├── model.py            # 2. 模型定义 (Network Architecture)
│   ├── trainer.py          # 3. 训练逻辑 (Training Loop)
│   ├── utils.py            # 4. 辅助函数 (Metrics, Logging)
│   └── predict.py          # 5. 推理/预测逻辑
│
├── main.py                 # 项目入口 (整合所有模块)
├── requirements.txt        # 依赖库
└── README.md
'''


import os
import sys

def create_project_structure(project_name):
    # 如果没提供名字，默认叫 MyNeuralProject
    if not project_name:
        project_name = "MyNeuralProject"
        
    root_dir = project_name
    
    # 1. 定义目录结构列表
    dirs = [
        os.path.join(root_dir, "data", "raw"),
        os.path.join(root_dir, "data", "processed"),
        os.path.join(root_dir, "configs"),
        os.path.join(root_dir, "src"),
    ]
    
    # 2. 定义可以通过 touch 创建的空文件（或带这类文件名）
    files = [
        os.path.join(root_dir, "configs", "config.yaml"),
        os.path.join(root_dir, "src", "__init__.py"),
        os.path.join(root_dir, "src", "dataset.py"),
        os.path.join(root_dir, "src", "model.py"),
        os.path.join(root_dir, "src", "trainer.py"),
        os.path.join(root_dir, "src", "utils.py"),
        os.path.join(root_dir, "src", "predict.py"),
        os.path.join(root_dir, "main.py"),
        os.path.join(root_dir, "requirements.txt"),
        os.path.join(root_dir, "README.md"),
    ]

    print(f"正在创建项目骨架: {project_name} ...")
    
    # 创建目录
    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
            print(f"  [目录] {d} 创建成功")
        except OSError as e:
            print(f"  [错误] 创建目录 {d} 失败: {e}")

    # 创建空文件
    for f in files:
        if not os.path.exists(f):
            try:
                with open(f, 'w', encoding='utf-8') as file:
                    # 可以在这里写入一些基础注释
                    if f.endswith(".py"):
                        file.write(f"# {os.path.basename(f)}\n")
                    elif f.endswith(".md"):
                        file.write(f"# {project_name}\n\n")
                print(f"  [文件] {f} 创建成功")
            except OSError as e:
                 print(f"  [错误] 创建文件 {f} 失败: {e}")
        else:
            print(f"  [跳过] 文件 {f} 已存在")
            
    print(f"\n项目结构生成完毕! 请进入目录开始开发:\ncd {project_name}")

if __name__ == "__main__":
    # 从命令行参数获取项目名称
    p_name = sys.argv[1] if len(sys.argv) > 1 else "MyNeuralProject"
    create_project_structure(p_name)


# 使用示例
# 将上述code封装为1个py脚本, create_project_structure.py
python create_project_structure.py NAME
  

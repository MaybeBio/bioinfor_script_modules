# 为每一个项目创建规范的层次化组织文件夹

import os
import argparse

def create_project_structure(project_name):
    """
        Args：
            project_name (str): 新项目的名称
        Fun：
            为新项目创建一个结构化的目录，注意该项目是建立在当前文件夹下，也就是pwd下
    """
    
    # Define the directory structure
    structure = {
        'data': {
            'raw': None,
            'processed': None,
            'external': None
        },
        'notebooks': None,  # 改为空目录，不预设文件
        'src': {
            'data': None,
            'features': None,
            'models': None,
            'evaluation': None,
            'utils': {
            '__init__.py': None
            }
        },
        'scripts': None,
        'models': None,
        'config': None,
        'tests': None,
        'reports': {
            'figures': None,
            'results': None,
            'final_report.md': None
        },
        'requirements.txt': None,
        'environment.yml': None,
        'setup.py': None,
        'Dockerfile': None,
        'README.md': None
    }

    # 创建项目根目录
    project_path = os.path.abspath(project_name)
    os.makedirs(project_path, exist_ok=True)
    print(f"Created project directory: {project_path}")

    def create_structure(base_path, structure):
        """
            Args：
                base_path (str): 基础路径
                structure (dict): 目录结构字典
            Fun:
                根据给定的结构在指定路径下创建目录和文件(递归创建目录结构)
        """
        for name, content in structure.items():
            path = os.path.join(base_path, name)
            
            # 如果内容为None且名称不包含'.'，则创建目录
            if content is None:
                if '.' in name:
                    # 包含点的视为文件
                    open(path, 'a').close()
                    print(f"Created file: {path}")
                else:
                    # 不包含点的视为目录
                    os.makedirs(path, exist_ok=True)
                    print(f"Created directory: {path}")
            else:
                # content为字典，创建目录并递归
                os.makedirs(path, exist_ok=True)
                print(f"Created directory: {path}")
                if isinstance(content, dict):
                    create_structure(path, content)

    # 创建目录结构
    create_structure(project_path, structure)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a structured project directory.')
    parser.add_argument('project_name', help='Name of the project directory to create')
    args = parser.parse_args()
  
    create_project_structure(args.project_name)


# 使用示例
# 将该代码文件命名为Create_project_structure.py
python Create_project_structure.py "my_proj"

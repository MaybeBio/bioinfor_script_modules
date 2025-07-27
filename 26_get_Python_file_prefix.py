# 使用python，获取某个文件的前缀

# 局部前缀：比如说"path/to/my_document.txt"，获取的是my_document
# 全局前缀：还是上面那个例子，就是path/to/my_document

# 1，
# 局部前缀

import os

def get_file_prefix(filepath):
  """
  获取文件名的前缀（不包括后缀名）。

  Args:
    filepath: 文件路径。

  Returns:
    文件名的前缀。
  """
  filename = os.path.basename(filepath)  # 获取文件名，如果传入的是完整路径
  prefix = os.path.splitext(filename)[0]
  return prefix

def get_file_prefix(file_path):
  """
  Args:
      file_path:文件的绝对路径
  Fun:
      获取文件的前缀名（局部前缀）
  """
  import os 
  # 获取文件名，包括后缀
  filename = os.path.basename(file_path)
  # 获取前缀
  prefix = os.path.splittext(filename)[0]
  # 或者也可以直接使用分隔符
  # prefix = filename.split('.')[0]
  return prefix

##############################################################################################################################################################################

# 2
# 全局前缀
def get_file_prefix_string(filename):
  return filename.split(".")[0]
  



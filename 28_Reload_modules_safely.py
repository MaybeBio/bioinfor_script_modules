# 主要是为了安全地重新导入或者是加载模块
# 因为一般第一次import模块的时候，这个时候模块会缓存在系统的内存中，也就是缓存在sys.modules字典中，此时即使修改了模块代码源文件，但是再次import其实还是使用的旧版本的代码
# 所以需要安全地重新地加载该模块

# 所谓模块，比如说我在/path1/下面有一个moduleA.py，然后其中定义了一个子函数def Normalize():
# 然后我就想在其他的代码文件中使用这个子函数来处理，比如说归一化其他的数据，那么这个时候我就需要在这个文件中导入模块A也就是moduleA的Normalize函数
# 效果就是：
import sys
sys.append("/path1/") # 注意如果要使用/path1/moduleA.py，是添加这个模块文件的文件夹
# 然后我们就可以正常加载该模块文件中定义的子函数了
from moduleA import Normalize

# 或者是更一般的写法如下
import moduleA as A
import A.Normalize as Normalize

# 后续可以通过sys.append以及remove函数来处理这些模块导入路径，以及可以在sys.modules中查看已经导入的模块名


# 1️⃣
# 1，更加安全地重新加载该模块moduleA，即使我在源文件moduleA.py中多次修改迭代更新了该文件，也依旧能够在我想要导入的文件中加载最新版的moduleA
import sys
import importlib

def reload_module(module_path, module_name):
    """安全地重新加载模块"""
    # 添加路径
    if module_path not in sys.path:
        sys.path.append(module_path)
    
    # 重新加载或导入
    if module_name in sys.modules:
        # 重新加载已存在的模块
        importlib.reload(sys.modules[module_name])
    else:
        # 首次导入
        exec(f"import {module_name}")
    
    return sys.modules[module_name]


# ✅✅✅✅✅✅使用示例
# ccs = reload_module('/data2/IDR_LLM/my_DL/scripts', 'charge_chi_score')
xid = reload_module('/data2/IDR_LLM/my_DL/scripts', 'chi_score_analysis')
# 其实就是类似于安全地加载并导入模块chi_score_analysis，效果类似于
sys.append("/data2/IDR_LLM/my_DL/scripts")
import chi_score_analysis as xid



##############################################################################################################################################################################

# 2，对1中定义函数的一个改版

import importlib
import sys
from types import ModuleType

def reload_module(module_path, module_name):
    """改进的模块重新加载函数"""
    # 添加路径（避免重复）
    if module_path not in sys.path:
        sys.path.insert(0, module_path)  # 使用insert(0, ...)优先搜索
    
    try:
        # 如果模块已经导入，重新加载
        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            # 首次导入
            module = importlib.import_module(module_name)
        
        return module
    except Exception as e:
        print(f"重新加载模块 {module_name} 失败: {e}")
        return None


===================================================================================================================================

# 3，更简单的方法
# notebook开头的sys.append一般是固定了的
# 我只需要importlib.reload(被修改了源码的该模块)即可

import importlib
import utils.data_generation_and_treatment as dgen
import utils.cluster_tools as ct

# 修改完代码后，重新加载模块
# ⚠️对，我只需要importlib.reload(被修改了源码的该模块)即可！
importlib.reload(dgen)
importlib.reload(ct)




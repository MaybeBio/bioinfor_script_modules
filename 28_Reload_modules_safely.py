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

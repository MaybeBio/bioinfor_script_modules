# 下面的程序都是完整的python api接口代码，并不是单纯的函数模块，当然，可以写成一个函数模块，但是暂且记录原始的全代码

# 收集的代码在：01_get_idrs.py、01_get_idrs_v2.py

# 原始版本01_get_idrs.py命令行如下：
python3 /data2/IDR_LLM/my_DL/01_get_idrs.py --help
usage: 01_get_idrs.py [-h] --i I --r R --o O

Run script to extract IDRs from full protein sequences in a fasta file

options:
  -h, --help  show this help message and exit
  --i I       Name of input fasta file containing full sequences # 多个蛋白质序列的fasta输入文件
  --r R       Name of output file contaning IDR results from iprscan # 包含-appl多个参考数据库结果、未整理合并的中间数据（即蛋白质多结构域中间数据）
  --o O       Name of output file contaning Uniprot IDs and IDR sequence lists # 输出的被PfamA结构域排除之后的MobiDB-lite识别出来的IDR stitching结构域，也就是最终输出的IDR序列文件

# v2版本01_get_idrs_v2.py的代码命令行同样如下：
python3 /data2/IDR_LLM/my_DL/01_get_idrs_v2.py --help
usage: 01_get_idrs_v2.py [-h] --i I --r R --o O

Run script to extract IDRs from full protein sequences in a fasta file

options:
  -h, --help  show this help message and exit
  --i I       Name of input fasta file containing full sequences
  --r R       Name of output file contaning IDR results from iprscan
  --o O       Name of output file contaning Uniprot IDs and IDR sequence lists

# 前期使用以v2为主：主要输出的就是uniprot蛋白质id（accession序列号）+IDR序列list+non-IDR序列list

# 更新中文版之后的代码在01_get_idrs_v3.py
# v3版本除了输出每个蛋白质的IDR、non-IDR序列list之外，还输出IDR stitching序列文件的坐标，即起点与终点坐标
# 另外还设置了一个用于过滤IDR长度的阈值参数，默认为30，也就长度＜30aa，不认为是本程序所要寻找的IDR


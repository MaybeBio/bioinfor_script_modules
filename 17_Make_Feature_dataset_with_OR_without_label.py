# 同样整理的是整体py脚本接口，可以写成一个单独的函数模块，暂时记录全局代码

# 代码收集在02_make_dataset.py、02_make_dataset_v2.py

# v1版本的02_make_dataset.py命令行如下：
# 主要是此处的label文件，也就是--p的命令行输入，是专门针对v1版本问题的partition ratios文件，所以普适性不是很强
python3 /data2/IDR_LLM/my_DL/02_make_dataset.py --help
usage: 02_make_dataset.py [-h] --i I --p P --o O

Run script to generate dataset of IDR features

options:
  -h, --help  show this help message and exit
  --i I       Name of input file containing lists of IDRs in each protein
  --p P       Name of input file contaning log partition ratios
  --o O       Name of output file to export the data matrix into

# v2版本的将--p参数改成了--l参数，即label参数，可以提供，也可以不提供
# 如果提供了，就是有监督的学习，否则是无监督的学习
python3 /data2/IDR_LLM/my_DL/02_make_dataset_v2.py --help
usage: 02_make_dataset_v2.py [-h] --i I [--l L] --o O

Run script to generate dataset of IDR features, Note that more features could be added in function get_featureX within the script on your own

options:
  -h, --help  show this help message and exit
  --i I       Name of input file containing lists of IDRs in each protein
  --l L       Name of input file contaning labels (optional), default is None
  --o O       Name of output file to export the data matrix into


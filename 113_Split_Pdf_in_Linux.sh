# 划分/编辑pdf文件

# 1. 原始，自带
pdfseparate
pdfunite

# 2. pdftk
# 安装
sudo apt-get update
sudo apt-get install pdftk
# 先查看pdf页数多大，便于后续估计
pdfinfo 数据结构\ 用面向对象方法与C++语言描述\ 第3版-殷人昆.pdf | grep Pages
# 比如说Pages:           516
# 分成5分，凑100pages
pdftk  数据结构\ 用面向对象方法与C++语言描述\ 第3版-殷人昆.pdf   cat 1-100 output 数据结构-殷人昆-3rd_1_100.pdf
# ...
pdftk  数据结构\ 用面向对象方法与C++语言描述\ 第3版-殷人昆.pdf   cat 401-516  output 数据结构-殷人昆-3rd_401_516.pdf

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


#############################################################################################################################

# 3. pdftk批量处理，输入pdf文件+固定拆分页数
#!/bin/bash
# 判断传入参数数量，必须是 2 个：PDF文件  单份页数
if [ $# -ne 2 ]; then
    echo "使用方式：$0 目标文件.pdf  每几份拆分页数"
    echo "示例：$0 test.pdf 50"
    exit 1
fi

PDF_FILE="$1"
PER_PAGE="$2"

# 校验输入是否为数字
if ! [[ "$PER_PAGE" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误：第二个参数必须是正整数页数"
    exit 1
fi

# 校验文件存在
if [ ! -f "$PDF_FILE" ]; then
    echo "错误：文件 $PDF_FILE 不存在"
    exit 1
fi

# 获取PDF总页数
TOTAL=$(pdftk "$PDF_FILE" dump_data | grep NumberOfPages | awk '{print $2}')
echo "文件总页数：$TOTAL，按每 ${PER_PAGE} 页拆分"

START=1
while [ "$START" -le "$TOTAL" ]; do
    END=$(( START + PER_PAGE - 1 ))
    if [ "$END" -gt "$TOTAL" ]; then
        END="$TOTAL"
    fi
    OUTPUT="${PDF_FILE%.pdf}_${START}-${END}.pdf"
    pdftk "$PDF_FILE" cat ${START}-${END} output "$OUTPUT"
    echo "生成：$OUTPUT"
    START=$(( START + PER_PAGE ))
done

echo "✅ 全部拆分完毕"


# 最后执行
# 每50页拆分
./split_pdf.sh 文档.pdf 50

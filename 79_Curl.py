# -O (大写)	保留文件原始名称，直接下载到当前目录	不想改名，仅下载即可
# -o (小写)	自定义保存路径 / 文件名	需要重命名或指定保存目录

# 1
# 最简单的下载，文件名为 test-file.zip，保存在当前终端所在目录
curl -O https://example.com/test-file.zip

# 2
# 将下载的文件重命名为 my-file.zip，保存在当前目录
curl -o my-file.zip https://example.com/test-file.zip

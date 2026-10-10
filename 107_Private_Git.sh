# 共用服务器, 不用--global

# 1. 当前项目当前仓库
# 设置仅当前项目的 Git 身份
git config user.name "你的名字"
git config user.email "你的邮箱"

# 2. 一次性操作
# 查看
# git config user.name
# git config user.email
# git config --list
# 再操作在其他地方
git add . && git -c user.name="你的名字" -c user.email="你的邮箱" commit -m "提交说明" && git push

# 3. 如果是新仓库
git init
git branch -M main
git remote add origin 新建的远程仓库地址(gh存疑)
git add . && git -c user.name="你的名字" -c user.email="你的邮箱" commit -m "提交说明" && git push -u origin main



# 4. 如果我们连本地仓库配置都不想写入，一次commit临时指定身份，完全不写配置文件（适合极短期一次性操作）
git -c user.name="Your Name" -c user.email="you@example.com" commit -m "prepare initial structure"



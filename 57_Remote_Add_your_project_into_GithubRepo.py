# 将你电脑上的某个文件夹项目远程添加到github仓库中
# 如果是大文件: 
# (1)Git LFS
# (2)或者撤销包含大文件的提交将大文件移出 Git 跟踪列表
# (3)告诉 Git 以后忽略它们,在项目根目录创建或编辑 .gitignore 文件
# (4)如果懒的话: 直接暴力删除源文件, 本来我们只上传脚本到github, 数据文件放云盘或本地, 重新push
# (5)如果是一些ipynb比较大, 都是一些图标或者输出, 尽量都clear output, 让输出不要超过20MB左右

1, cd /该项目文件夹 
git init
2, 检查并提交代码
git add .
git commit -m "Initial commit"
# git commit -m "Initial commit: Release v0.1.0"
3, 确保分支名为main, 重命名主分支为 main
git branch -M main
4, 配置远程仓库地址 origin, 让你的本地 Git 仓库知道要把代码推送到哪里
可以在网页端github上创建同名仓库, 只关注public或private, 其余markdown、gitignore等都先不用管
5, 添加远程仓库地址
git remote add origin https://github.com/MaybeBio/同名仓库.git
6, 尝试推送
git push -u origin main


# 如果已经commit过了, 第2次或者是后续更新推送
7, 查看哪些文件被修改了/查看当前状态
git status
8, 添加修改到暂存区(添加所有修改的文件)
git add .
9, 填写本次更新的说明信息
git commit -m "Update README with new documentation and examples"
10, 推送到远程仓库
git push

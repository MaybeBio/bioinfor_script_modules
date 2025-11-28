# 将你电脑上的某个文件夹项目远程添加到github仓库中
# 如果是大文件: 
# (1)Git LFS
# (2)或者撤销包含大文件的提交将大文件移出 Git 跟踪列表
# (3)告诉 Git 以后忽略它们,在项目根目录创建或编辑 .gitignore 文件
# (4)如果懒的话: 直接暴力删除源文件, 本来我们只上传脚本到github, 数据文件放云盘或本地, 重新push

1, cd /该项目文件夹 
git init
2, 检查并提交代码
git add .
git commit -m "Initial commit"
3, 确保分支名为main
git branch -M main
4, 配置远程仓库地址 origin, 让你的本地 Git 仓库知道要把代码推送到哪里
可以在网页端github上创建同名仓库, 只关注public或private, 其余markdown、gitignore等都先不用管
5, 添加远程仓库地址
git remote add origin https://github.com/MaybeBio/同名仓库.git
6, 尝试推送
git push -u origin main

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


#################################

# 如果已经commit过了, 第2次或者是后续更新推送
7, 查看哪些文件被修改了/查看当前状态
git status
8, 添加修改到暂存区(添加所有修改的文件)
git add .
9, 填写本次更新的说明信息
git commit -m "Update README with new documentation and examples"
10, 推送到远程仓库
git push


##################################

# 如果已经commit过了, 但是在push时因为文件很大而导致失败
# 可以不上传更新大的文件，只更新其余的

"""


如果**不在乎**最近那次导致失败的 commit 信息，可以直接强行回退到成功 push 之前的状态，重新提交。

1.  **回退到上一个版本（HEAD~1 指向由于大文件失败的那个 commit）：**
    ```bash
    git reset --soft HEAD~1
    ```
2.  **强制从暂存区中移除该文件夹（不删除本地文件）：**
    ```bash
    git rm -r --cached data/processed/disobind_subsample
    ```
3.  **确保文件夹已在 `.gitignore` 中（防止再次被 add）：**
    ```bash
    echo "data/processed/disobind_subsample/" >> .gitignore
    git add .gitignore
    ```
4.  **重新提交剩下的文件：**
    ```bash
    git commit -m "chore: remove large folder and update gitignore"
    ```
5.  **推送：**
    ```bash
    git push origin main
    ```

核心问题是 **“大文件已经存在于本地 .git 文件夹的历史对象中”**。

执行 `git reset --soft HEAD~1`，然后确保 `git status` 里那个大文件夹不再是绿色（暂存态），而是红色（未追踪态）或直接被 `.gitignore` 忽略，再重新 commit 即可。

"""

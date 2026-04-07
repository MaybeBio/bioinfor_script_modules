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

# 如果已经commit过了, 但是在push时因为文件很多而导致失败
# 可以不上传更新大的文件，只更新其余的

"""
由于 disobind_subsample 文件夹太大导致上传失败，现在需要将其从 Git 的暂存区（Staging Area）撤回，并确保它不会被包含在提交中。

可以按照以下步骤在终端中操作：

### 1. 从暂存区移除该文件夹
运行以下命令，这会将该文件夹从 Git 的跟踪中移除，但**不会删除本地磁盘上的实际文件**。

```bash
cd /data2/IDRInter
git rm -r --cached data/processed/disobind_subsample
```

### 2. 将其添加到 `.gitignore`（防止以后再次被误添加）
为了避免下次执行 `git add .` 时再次把这个大文件夹加进去，建议将其路径写入 `.gitignore` 文件：

```bash
echo "data/processed/disobind_subsample/" >> .gitignore
```

### 3. 如果你已经做过了 Commit（可选）
如果之前已经执行过 `git commit`（只是在 `push` 时失败），那么该文件夹的信息已经进入了本地历史记录。需要撤销最后一次提交但不删除代码更改：

```bash
# 撤回到上一个版本，保留所有代码更改（此时变回 add 之前的状态）
git reset --soft HEAD~1

# 然后重复第1步的命令来剔除大文件夹
git rm -r --cached data/processed/disobind_subsample
```

### 总结
执行完第 1 步和第 2 步后，你可以重新进行提交：
```bash
git add .
git commit -m "Remove large folder from tracking and update gitignore"
git push
```

这样，Git 就会忽略这个巨大的文件夹，只上传其他必要的文件。

"""

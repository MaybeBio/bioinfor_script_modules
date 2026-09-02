# 最近的一次commit提交信息有误，如何修改覆盖
# 把最后一次的 Commit 记录从历史中“抹去”，并用包含新修改（或新日志）的 Commit 替代它

# 1. 在本地修改信息
git commit --amend -m "这里写正确的commit信息" 

# 2. 强推到远程仓库
git push origin <当前分支名> --force
# 或者简写为 git push -f 

# 如果是更早之前的某次 commit，上述忽略

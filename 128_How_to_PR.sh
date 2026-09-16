# 参考：https://github.com/PaddlePaddle/book/wiki/%E5%A6%82%E4%BD%95%E6%8F%90%E4%BA%A4Pull-Request
# 如何提交Pull Request?

# 1. 先fork别人的仓库，生成一个在自己目录下的Github仓库
# 2. 克隆仓库到本地
git clone xxx
# 3. 配置远程地址（指向上游远程仓库）
# origin 默认是我们clone下来的仓库，fork
# upstream 是我们手动新增，指向原始别人的仓库（约定俗成upstream，当然也可以叫source或者改成其他之类的）
# 目的是为了让我们的本地git知道原始仓库在哪里，下面才能fetch它的代码
# git remote -v 检查过upstream有就行
git remote add upstream 官方仓库地址

# ---------------------------
# 【之后每次开发新PR都执行下面】
# ---------------------------

# 4. 拉取上游最新消息（拉取官方最新提交到本地，不改动本地文件）
# 只是拿一份最新快照，看看原始仓库现在变成什么样了
git fetch upstream

# 5. 切换到我们本地的main主分支
# 我们需要一个本地 “干净的镜像分支”，专门用来同步上游原始仓库，不在这个分支写任何业务代码、不提交任何自己的修改
# 这个分支，要和上游对应的分支一一对应：上游只有 main，那我们本地就用 main 作为这个干净镜像分支
# 不是因为我们只有 main，也不是单纯因为上游只有 main
# 是一一对应关系：上游仓库有 main，我们本地就拿自己的 main，专门用来镜像上游 main
# 这个本地 main，唯一职责：同步上游，保持干净。不能在这里写我们的功能代码
# 所以操作顺序：checkout main → 切到这个干净镜像分支，准备更新它
git checkout main

# 6. 把本地main更新为上游最新（将本地main对齐到 upstream/main，原仓库最新main）
# 把本地 main 的基底，对齐到刚刚 fetch 下来的上游 main 最新提交
# 本地 main 现在就和原始仓库 main 一模一样，干净、最新
# 这里才是怕上游有更新的核心操作
# 当然，这一步更新的，仅仅只是我们本地的main，不会自动推送到我们fork仓库的origin/main
# 我们一般只需要维护本地main即可，无需让fork仓库main也同步（因为重要的是下面我们新建的分支）
git rebase upstream/main

# 7. 基于最新main创建分支开发（在【已经是最新代码的本地main】上新建我们的功能分支）
# ⚠️ 从最新 main 拉出独立分支干活，永远不要直接在 main 上写代码！
# 不要在本地 main 写代码提交！main 分支只用来同步上游，纯干净镜像。所有改动全部放新建的 feature 分支
# main 分支不能开发，所以新建分支（基于当前所在分支，此时一般是已经同步到最新的main分支）
# 所有改动起点就是上游最新代码，后续提 PR 冲突最少
git checkout -b my-fix
# 8. 编码、commit（改代码，提交）
git add .
git commit -m "fix: xxx问题"
# 9. push，提PR（把我们的分支推送到我们自己的fork仓库（origin））
git push origin my-fix
# 10. 推送完成后，去 GitHub 网页，就能看到提示：Compare & pull request，点它创建 PR，目标分支选原仓库的 main
# 11. 在被merge PR之后，我们可以删除提交的该分支，以及本地fork的仓库

# 我们再理一下：
# upstream/main：我们 fetch 下来缓存的「原始仓库 main」快照
# main：我们电脑上本地分支
# origin/main：我们 fork 仓库（github/gitee 服务器上）的 main

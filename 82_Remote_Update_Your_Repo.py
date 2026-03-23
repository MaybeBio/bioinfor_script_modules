# 基础操作参考py 57、61脚本
# 假设我们已经有了1个github仓库，如何远程拉取并修改它，然后再上传？

# 1. 克隆（拉取）远程仓库到本地
git clone https://github.com/你的用户名/Zerotier_Tailscale_Moniter.git

# 2. 进入仓库目录并进行修改
# 在文件夹里对你的代码进行修改、添加或删除操作。

# 3. 查看文件变更状态（可选但建议），确认自己改了哪些文件：
git status

# 4. 将需要的变更添加到暂存区
# （1）正常将所有改动过的文件准备提交
git add .

# （2）如果只想提交特定文件
git add 文件夹/文件名

# （3）使用 .gitignore 分离模板和真实文件
# 永久忽略该文件，除非以后永远不想提交它的任何修改
# 打开仓库根目录下的 .gitignore 文件（如果没有就新建一个）
# 直接在.gitignore 文件中写入该文件的相对路径，是相对于 .gitignore 文件所在目录的路径
# 再从git暂存区移除追踪，本地文件保留
git rm --cached zt-monitor.sh
# 此时再
git add .

# （4）临时忽略本次上传（仅针对本次修改，后续仍可正常追踪）
# 首先
# 处理已被git追踪的文件，如果这个文件之前提交过，比如说执行过git add+git commit，即使加了.gitignore，git仍会显示它的修改，必须先移除追踪，也就是从git暂存区移除
# 如何查看某个文件是否在git文件追踪列表中很简单，直接 git status 本文件，查看是否modified等红字
# 接着
# 比如所我只是这次不想上传zt-monitor.sh脚本，但以后可能还需要提交这个脚本的其他修改，那么我们不用改.gitignore，也不用git rm
# 只需要撤销暂存+让git暂时忽略修改
git restore --staged zt-monitor.sh
git update-index --assume-unchanged zt-monitor.sh
# 此时再
git add .
# 后续如果想要恢复对这个文件的追踪
# git update-index --no-assume-unchanged zt-monitor.sh


  # 5. 提交变更并添加说明
  # 总之前面的文件都已经 git add .了，无论是什么情况
  git commit -m "xxxx"

  # 6. 推送到远程 GitHub 仓库
  # 将本地的提交推送到远程的默认分支（通常是 main 或 master）：
  git push origin main

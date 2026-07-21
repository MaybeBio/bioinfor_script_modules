# 获取某个仓库某个子文件、子文件夹

# 1. 远程仓库直接导出（无需克隆仓库，轻量化首选）
git archive --remote=仓库地址 分支:仓库内路径 | tar -x

# ⚠️ 注意git版本更新 https://www.datacamp.com/tutorial/git-update
# 注意更新docker源，git版本限制可以直接上游PPA（ppa:git-core/ppa） https://git-scm.com/install/linux
# 目前 git version 2.54.0
# 例如lightning仓库下的某一个子文件夹 https://github.com/Lightning-AI/pytorch-lightning/tree/master/examples/fabric/reinforcement_learning
# 仓库地址已知，分支为master，仓库内相对路径也已知
git archive --remote=https://github.com/Lightning-AI/pytorch-lightning.git master:examples/fabric/reinforcement_learning | tar -x

# 2. 或使用 https://download-directory.github.io/?url=

# 3. 浅克隆 + 稀疏检出
git clone --depth 1 仓库地址
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/Lightning-AI/pytorch-lightning.git
cd pytorch-lightning
git sparse-checkout set examples/fabric/reinforcement_learning

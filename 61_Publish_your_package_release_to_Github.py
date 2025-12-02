# 将本地 Python 项目发布到 GitHub 并创建 Release 版本的简易步骤指南

### 第一阶段：本地 Git 初始化
在你的 Linux 终端中，进入项目目录 LibInspector

1.  初始化 Git 仓库
cd /data2/LibInspector
git init

2.  创建.gitignore文件(非常重要)
防止将垃圾文件（如编译缓存、虚拟环境）上传到 GitHub。
创建文件.gitignore 并填入以下内容：略, 到时候灵活变动

3.  提交代码到本地仓库
git add .
git commit -m "Initial commit: Release v0.1.0"

### 第二阶段：在 GitHub 创建远程仓库

1.  登录你的 GitHub 账号
2.  点击右上角的 + 号，选择 New repository
3.  Repository name 填写 LibInspector
4.  Description 可以填写：略
5.  Public/Private 选择 Public
6.  重要提示：不要勾选 "Add a README file"、".gitignore" 或 "license"。因为你本地已经有了这些文件，勾选会导致冲突
7.  点击Create repository

### 第三阶段：关联并推送代码

在 GitHub 创建完仓库后，你会看到一个页面包含仓库地址（例如 https://github.com/YourUsername/LibInspector.git）
在你的终端执行以下命令：

1.  重命名主分支为 main (GitHub 默认标准)
git branch -M main

2.  添加远程仓库地址 
git remote add origin https://github.com/YourUsername/LibInspector.git

3.  推送到 GitHub
git push -u origin main


### 第四阶段：构建安装包 (可选但推荐)

为了让 Release 看起来更专业，你可以上传构建好的 .whl 和 .tar.gz 文件，这样别人可以直接下载安装

1.  安装构建工具
pip install build

2.  构建项目
python -m build
# 这会在你的目录下生成一个 dist/ 文件夹，里面包含 lib_inspector-0.1.0-py3-none-any.wh` 和 lib_inspector-0.1.0.tar.gz。


### 第五阶段：在 GitHub 创建 Release

# 有两种方案

1. 最简单（只打 Tag，不上传构建包）, 触发 GitHub 自动生成源码压缩包

# 1. 本地打标签
git tag v0.1.0

# 2. 推送标签到远程
git push origin v0.1.0


2. 构建whl

# 1. 先构建 (还是建议构建一下)
python -m build

# 2. 创建 Release 并上传 dist 目录下的所有文件, 这里需要用到gh工具
# --title: 标题
# --notes: 说明文字
gh release create v0.1.0 dist/* --title "v0.1.0 Initial Release" --notes "First release with CLI support."
(这里需要gh登入, 但是gh不确定)

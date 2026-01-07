# 典的 Git 同步场景。
# 简单来说，不仅你的本地有新东西（你的新代码），远程仓库也有新东西（你在网页端的修改），
# Git 需要你先把远程的新东西“拉”下来跟你的“合并”之后，才允许你往上推。

# 1. 拉取并合并远程修改
git pull origin main
# (注：如果你的默认分支叫 master，请把 main 换成 master)

# 运行之后会有提示
hint:   git config pull.rebase false  # merge ——》一般选这个
hint:   git config pull.rebase true   # rebase
hint:   git config pull.ff only       # fast-forward only

# 如果没有冲突：Git 会自动弹出一个编辑器让你确认合并信息。直接保存并关闭即可（vim 按 :wq，nano 按 Ctrl+X 然后 Y）。然后你就合并成功了

'''
执行后会出现两种情况：

情况 A：自动合并成功（Most Likely）

如果你修改的文件和网页端修改的文件不一样（或者不是同一行），Git 会自动把两份修改合并在一起。
可能会弹出一个编辑器让你输入“Merge message”，通常直接保存退出即可（如果是 vim，按 :wq 回车；如果是 nano，按 Ctrl+O 回车然后 Ctrl+X）。
情况 B：出现冲突 (Conflict)

如果你们改了同一行代码，Git 会提示 CONFLICT。
你需要打开 VS Code 中标红的文件，选择“保留双方”或“采用当前/传入更改”，保存文件。
然后执行 git add . 和 git commit -m "解决冲突"。

'''

# 2. 推送所有更改
# 合并完成后（无论是从情况A 还是 B 出来），你的本地现在就同时拥有了“网页端的修改”和“你本地的更新”。
# 现在可以放心地推送到远程了：

git push origin main

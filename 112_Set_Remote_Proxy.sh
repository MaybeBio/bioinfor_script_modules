# 远程 Linux 内部程序走本地代理
# 通过 SSH 远程端口转发，把 Windows 本地的 7892 代理端口映射到远程 Linux 主机上
# ⚠️ 查看vpn端口、ssh config中设置端口、每次init shell重新export代理

# Windows 本地代理客户端(clash verge、beibei) ，例如http代理端口7892
# 因为经常变，所以时刻查看一下
# vpn中局域网代理无需处理

# 1. 配置.ssh/config
####
# Host 你的主机别名
#    HostName 远程服务器IP/域名
#    User 远程登录用户名
#    Port 22
#    # 将本地 7892 代理端口，转发到远程主机的 7892 端口
#    RemoteForward 7892 127.0.0.1:7892
####

# 2. 设置 HTTP/HTTPS 代理
export http_proxy=http://127.0.0.1:7892
export https_proxy=http://127.0.0.1:7892

# 取消代理
unset http_proxy https_proxy

# 详情见https://github.com/MaybeBio/GhResearcher
# 比如说动态追踪科研圈的自动化
# 因为经常性远程推送更新用户和组织名单
# 1个简易的命令如下：

curl -f -o protein_dl_user.txt https://raw.githubusercontent.com/MaybeBio/GhResearcher/refs/heads/main/tests/protein_dl_user.txt && \
ghresearcher monitor -f protein_dl_user.txt --since $(date -d "1 day ago" +%Y-%m-%d) --expand-commits && \
rm -f protein_dl_user.txt

# 1. alias：简略，但处理不了位置参数
# 2. 封装shell函数到 bashrc或zshrc中，可以定制log查看：自动化保存log，自行查询不保存：
# 3. tmux send‑keys：固定tmux终端会话，指定2到该tmux会话，每日早上9点接受动态
# 4. 2的执行方式可以自动化，crontab或者systemd定时，可以保存在本地再push更新，或者，使用github-action！
# 方案4其实就是定时执行 + Git 仓库持久化存储

##########################################################################################################################################################

# 方案2原始版本 1️⃣
ghfollow() {
    # 默认天数回溯1
    local days=1
    # 传参类似 ghfollw -2, 代表回溯2天
    # 只要传入参数，就会覆盖默认值，解析出2天这种
    if [[ $# -ge 1 ]]; then
        days="${1#-}"
    fi

    # 存几个变量，/tmp中的唯一临时文件，后续删除
    local tmp_raw=$(mktemp)
    local log_dir="$HOME/ghfollow_log"
    
    # 日志存档形式，什么时候记录的、存的是什么时候的信息
    mkdir -p "${log_dir}"
    local date_folder=$(date +%Y%m%d)
    local target_log_dir="${log_dir}/${date_folder}"
    mkdir -p "${target_log_dir}"
    # 时分秒到位，同一天可以多次运行，当然本身gh输出就是有时间戳的，不记录时间其实也没有多少问题
    # $HOME/ghfollow_log/2026-08-13/090000_past1days.log
    local logfile="${target_log_dir}/$(date +%H%M%S)_past${days}days.log"

    # 执行命令并保存日志
    curl -f -o "${tmp_raw}" https://raw.githubusercontent.com/MaybeBio/GhResearcher/refs/heads/main/tests/protein_dl_user.txt && \
    ghresearcher monitor -f "${tmp_raw}" --since "$(date -d "${days} day ago" +%Y-%m-%d)" --expand-commits 2>&1 | tee "${logfile}"

    rm -f "${tmp_raw}"
    echo "✅日志已保存：${logfile}"
}

# 使用示例，然后查看log文件
ghfollow -1
ghfollow -3

#####################################################################################################################################

# 方案2改进版本2️⃣
# 平时手动跑不带参数，只屏幕输出，不写磁盘log（堆积起来麻烦），一般就只有自动化定时推送设置log
# 所以加一个参数 --log
# ⚠️ 因为涉及到下载github上的文件，所以proxy先处理好（不然curl一直卡着）

ghfollow() {
    local days=1
    local enable_log=0

    # 解析参数，支持 -N 回溯天数，以及 --log 开启日志
    # 正常情况下只需要处理2个参数：log标志和出传入的天数
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --log)
                enable_log=1
                shift
                ;;
            -*)
                days="${1#-}"
                shift
                ;;
            *)
                shift
                ;;
        esac
    done

    local tmp_raw=$(mktemp)
    local logfile=""

    # 只有开启日志的时候，才创建目录、生成日志路径
    if [[ ${enable_log} -eq 1 ]]; then
        local log_dir="$HOME/ghfollow_log"
        local date_folder=$(date +%Y%m%d)
        local target_log_dir="${log_dir}/${date_folder}"
        mkdir -p "${target_log_dir}"
        logfile="${target_log_dir}/$(date +%H%M%S)_past${days}days.log"
    fi

    if [[ ${enable_log} -eq 1 ]]; then
        # 定时模式：tee，屏幕打印同时写入日志文件
        curl -f -o "${tmp_raw}" https://raw.githubusercontent.com/MaybeBio/GhResearcher/refs/heads/main/tests/protein_dl_user.txt && \
        ghresearcher monitor -f "${tmp_raw}" --since "$(date -d "${days} day ago" +%Y-%m-%d)" --expand-commits 2>&1 | tee "${logfile}"
        echo "✅日志已保存：${logfile}"
    else
        # 手动交互模式：只输出终端，不写磁盘，无文件堆积
        curl -f -o "${tmp_raw}" https://raw.githubusercontent.com/MaybeBio/GhResearcher/refs/heads/main/tests/protein_dl_user.txt && \
        ghresearcher monitor -f "${tmp_raw}" --since "$(date -d "${days} day ago" +%Y-%m-%d)" --expand-commits
    fi

    rm -f "${tmp_raw}"
}

# 使用示例
# 手动执行，不看log
ghfollw -3
# 手动执行，但是存个log后续再看
ghfollow -2 --log
# 如果要在crontab或systemd中定时调用，都带上--log，这样每天至少1个log，起来就看

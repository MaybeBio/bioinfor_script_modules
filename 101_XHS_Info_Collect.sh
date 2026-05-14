# 在创建一些skill、或者是科研定制化、高度经验化的prompt的时候，或者是需要一些独特的思考视角来增强prompt的时候
# 可以参考小红书上的一些经验、帖子，
# 使用CLI工具 https://github.com/jackwener/xhs-cli

# 1.
# 目前每一条阅读记录都需要下面方式才能够展开正文
# 见issue：https://github.com/jackwener/xiaohongshu-cli/issues/55
xhs read 1 --json | jq -r '.data.items[0].note_card.desc // ""' | less


######################################################################################################

# 2.
# 如果要批量处理
# 我们只要正文desc内容

#!/bin/zsh
set -euo pipefail

if ! command -v xhs >/dev/null 2>&1; then
  echo "xhs command not found in PATH" >&2
  exit 1
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 \"QUERY\" PAGES [OUTPUT] [READ_COUNT]" >&2
  echo "Example: $0 \"如何将自己的想法转变为1个课题\" 5 开题.md 20" >&2
  exit 1
fi

query="$1"
pages="$2"
output="${3:-开题.md}"
read_count="${4:-20}"

typeset -i PAGES=$pages
typeset -i READ_COUNT=$read_count

has_jq=0
if command -v jq >/dev/null 2>&1; then
  has_jq=1
elif ! command -v python >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
  echo "Need either jq or python/python3 to parse JSON." >&2
  exit 1
fi

parse_title() {
  local json="$1"
  if (( has_jq )); then
    printf "%s" "$json" | jq -r '.data.items[0].note_card.title // "无标题"'
  else
    local pybin="python"
    command -v python3 >/dev/null 2>&1 && pybin="python3"
    printf "%s" "$json" | "$pybin" -c 'import sys,json; d=json.load(sys.stdin); print(d.get("data",{}).get("items",[{}])[0].get("note_card",{}).get("title","无标题"))'
  fi
}

parse_desc() {
  local json="$1"
  if (( has_jq )); then
    printf "%s" "$json" | jq -r '.data.items[0].note_card.desc // ""'
  else
    local pybin="python"
    command -v python3 >/dev/null 2>&1 && pybin="python3"
    printf "%s" "$json" | "$pybin" -c 'import sys,json; d=json.load(sys.stdin); print(d.get("data",{}).get("items",[{}])[0].get("note_card",{}).get("desc",""))'
  fi
}

# Truncate/initialize output file
: > "$output"

for ((page=1; page<=PAGES; page++)); do
  printf "\n\n# Search: %s - Page %d\n\n" "$query" "$page" >> "$output"

  # Keep context for read indices
  xhs search "$query" --sort popular --page "$page" >/dev/null

  for ((i=1; i<=READ_COUNT; i++)); do
    if json="$(xhs read "$i" --json 2>/dev/null)"; then
      title="$(parse_title "$json")"
      desc="$(parse_desc "$json")"

      if [[ -n "$desc" ]]; then
        printf "\n\n---\n\n## Page %d - Item %d\n\n### %s\n\n%s\n" "$page" "$i" "$title" "$desc" >> "$output"
      else
        printf "\n\n---\n\n## Page %d - Item %d\n\n[desc empty]\n" "$page" "$i" >> "$output"
      fi
    else
      printf "\n\n---\n\n## Page %d - Item %d\n\n[xhs read %d failed]\n" "$page" "$i" "$i" >> "$output"
    fi
    sleep 0.3
  done
done

echo "Done. Output: $output"

# 简单用法示例
chmod +x xhs.sh
zsh xhs.sh "如何将自己的想法转变为1个课题, 1个idea, 找到研究方向" 5 开题.md 20


#######################################################################################

# 3.
# 有反爬

#!/usr/bin/env zsh
set -euo pipefail

# 简化检验：需 jq 或 python
command -v xhs >/dev/null 2>&1 || { echo "xhs not found"; exit 1; }

QUERY="$1"; PAGES="$2"; OUTPUT="${3:-开题.md}"; READ_COUNT="${4:-20}"
: > "$OUTPUT"

rand_sleep() { sleep_time=$(awk -v a=0.8 -v b=2 'BEGIN{srand(); print a + (b-a)*rand()}'); sleep $sleep_time; }

read_with_retry() {
  local idx=$1
  local tries=0 max=5 backoff=2
  while (( tries < max )); do
    # capture both stdout+stderr
    out_and_err="$(xhs read "$idx" --json 2>&1)" || true
    if printf "%s" "$out_and_err" | grep -qi "Captcha triggered"; then
      tries=$((tries+1))
      wait_secs=$(( backoff ** tries ))
      echo "Captcha detected for item $idx, retry $tries/$max — sleeping ${wait_secs}s" >&2
      sleep $wait_secs
      continue
    fi
    # if output looks like json, return it
    if printf "%s" "$out_and_err" | head -1 | grep -q '^{'; then
      printf "%s" "$out_and_err"
      return 0
    fi
    # otherwise treat as failure and retry a little
    tries=$((tries+1))
    sleep 1
  done
  return 1
}

for ((page=1; page<=PAGES; page++)); do
  echo "# Search: $QUERY — Page $page" >> "$OUTPUT"
  xhs search "$QUERY" --sort popular --page "$page" >/dev/null || true
  for ((i=1; i<=READ_COUNT; i++)); do
    if json="$(read_with_retry $i)"; then
      # 提取 title & desc（优先用 jq）
      if command -v jq >/dev/null 2>&1; then
        title=$(printf "%s" "$json" | jq -r '.data.items[0].note_card.title // "无标题"')
        desc=$(printf "%s" "$json" | jq -r '.data.items[0].note_card.desc // ""')
      else
        title=$(printf "%s" "$json" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("data",{}).get("items",[{}])[0].get("note_card",{}).get("title","无标题"))')
        desc=$(printf "%s" "$json" | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d.get("data",{}).get("items",[{}])[0].get("note_card",{}).get("desc",""))')
      fi
      printf "\n\n---\n\n## Page %d - Item %d\n\n### %s\n\n%s\n" "$page" "$i" "$title" "$desc" >> "$OUTPUT"
    else
      printf "\n\n---\n\n## Page %d - Item %d\n\n[xhs read %d failed or captcha persisted]\n" "$page" "$i" "$i" >> "$OUTPUT"
    fi
    rand_sleep
  done
  # 每翻页后做更长等待，降低触发概率
  sleep 5
done

echo "Done. Output: $OUTPUT"

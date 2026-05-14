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

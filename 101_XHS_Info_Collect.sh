# 在创建一些skill、或者是科研定制化、高度经验化的prompt的时候，或者是需要一些独特的思考视角来增强prompt的时候
# 可以参考小红书上的一些经验、帖子，
# 使用CLI工具 https://github.com/jackwener/xhs-cli

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

# Truncate/initialize output file
: > "$output"

for ((page=1; page<=PAGES; page++)); do
  printf "\n\n# Search: %s — Page %d\n\n" "$query" "$page" >> "$output"

  # Run search (prints to stdout; xhs keeps the search context)
  xhs search "$query" --sort popular --page "$page"

  for ((i=1; i<=READ_COUNT; i++)); do
    printf "\n\n---\n\n## Page %d — Item %d\n\n" "$page" "$i" >> "$output"
    if ! xhs read "$i" >> "$output" 2>/dev/null; then
      printf "[xhs read %d failed]\n" "$i" >> "$output"
    fi
    sleep 0.3
  done
done

echo "Done. Output: $output"

# 简单用法示例
# chmod +x xhs_batch_read.zsh
# ./xhs_batch_read.zsh "如何将自己的想法转变为1个课题, 1个idea, 找到研究方向" 3 开题.md 20

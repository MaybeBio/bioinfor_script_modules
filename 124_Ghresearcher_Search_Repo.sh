# seen in ghresearcher repo
ghresearcher search --config examples/search_idr_repos.yaml --updated ">=$(date -d '7 days ago' +%Y-%m-%d)" --jq \                                                                            
  '.[] | [.fullName, .language, .stargazersCount, (.pushedAt | fromdateiso8601 + 8*3600 | strftime("%Y-%m-%d %H:%M:%S")), (.createdAt | fromdateiso8601 + 8*3600 | strftime("%Y-%m-%d %H:%M:%S")), .visibility, .url, .description] | @csv' \                                      
  | tail -n +3 > repo.csv

# 比如说一个gse超级序列号，里面有很多gsm样本号，我们无需一个一个样本页面点进去查看，直接拉取全部view即可
for g in $(seq 6047774 6047796); do
  echo "===== GSМ$g"
  curl -s "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM$g&targ=self&form=text&view=brief" | grep -iE "^!Sample_title|^!Sample_characteristics|^!Sample_source" | head -8
done

# Entrez Direct下载pubmed文献, 提取综述, 做embedding 

from Bio import Entrez
Entrez.email = "luxunisgod123@gmail.com"     # Always tell NCBI who you are

# 粗略的搜索
term1 = 'alphafold[All Fields] OR "AlphaFold"[All Fields]'
handle1 = Entrez.esearch(db="pubmed", term=term1, usehistory="y", retmax=2705)
result1 = Entrez.read(handle1)
handle1.close()

# count = 270可以在retmax=0中看到

webnev = result1['WebEnv']
query_key = result1['QueryKey']

# 比如说我们分10批次下载这2705篇文章的摘要
batch_size = 271
count = int(result1['Count'])
out_handle = open("alphafold_pubmed_abstracts.txt", "w")
for start in range(0, count, batch_size):
    end = min(count, start + batch_size)
    print("Downloading records %i to %i" % (start + 1, end))
    handle = Entrez.efetch(db="pubmed", rettype="abstract", retmode="text",
                           retstart=start, retmax=batch_size,
                           webenv=webnev, query_key=query_key)
    data = handle.read()
    handle.close()
    out_handle.write(data)
out_handle.close()

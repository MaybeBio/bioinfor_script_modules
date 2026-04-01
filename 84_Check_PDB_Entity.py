# 对于每一个PDB结构, 检查一下里面有多少个entity
# 比如说有多少个蛋白质，有多少个非蛋白质，如果是蛋白质的话那么每一条chain对应的是什么蛋白质（对应的是什么uniprot id）
# 比如说下面这种

"""
✅ 查询成功！PDB ID: 7W1M
该结构内所有的链总数 (包含非蛋白): 8
提取到的蛋白质链 (Chains) 数量: 6
⚠️ 注意！还包含非蛋白链 (如DNA/RNA小分子): ['7W1M_6', '7W1M_7']

--- 亚基组成的 DataFrame 详情表 ---
PDB ID	Entity ID	Asym ID	Auth Asym ID	Uniprot ID
0	7W1M	1	A	A	Q14683
1	7W1M	2	B	B	Q9UQE7
2	7W1M	3	C	C	O60216
3	7W1M	4	D	D	Q8WVM7
4	7W1M	5	E	E	Q6KC79
5	7W1M	8	H	H	P49711

"""

# 1.
# 这里没有封装成1个函数，而是打包写成了1个python脚本
# ⚠️ 然后需要注意的是我们的任务本质上还是发送请求，然后进行数据批量的获取与处理
# 所以要注意数据量一旦非常多的时候，要权衡各种策略
# ⚠️ 比如说我有十几万个pdb id，然后pdb官网提供了各种访问请求，有rest api，也有graphQL，
# 如果用rest api逐个请求就会非常慢（比如说之前uniprot+pdb两个 域名网站的请求，其中uniprot网站暂时不知道如何优化限速问题，但是pdb我们可以使用此处的graphQL）
# 此处数据库是pdb，我们可以使用graphQL
# 参考：https://www.rcsb.org/docs/programmatic-access/web-apis-overview#data-api

import pandas as pd
import requests
import time
import os
from tqdm import tqdm
from multiprocessing import Pool

# ================= 配置参数 =================
INPUT_PDB_LIST_CSV = "我们的PDB ID输入文件(csv输入文件, 每一行1个pdb id)"
OUTPUT_CSV = "/data2/pdb_chains_master_GraphQL.csv"
ERROR_LOG = "/data2/graphql_failed_log.txt"

WORKERS = 32
BATCH_SIZE = 100
# ============================================

def construct_graphql_query(pdb_id_list):
    ids_string = '["' + '", "'.join(pdb_id_list) + '"]'
    query = f"""
    {{
      entries(entry_ids: {ids_string}) {{
        rcsb_id
        polymer_entities {{
          rcsb_id
          rcsb_polymer_entity_container_identifiers {{
            auth_asym_ids
            reference_sequence_identifiers {{
              database_accession
              database_name
            }}
          }}
        }}
      }}
    }}
    """
    return {"query": query}


def fetch_graphql_batch(pdb_chunk):
    url = "https://data.rcsb.org/graphql"
    max_retries = 5

    df_list = []

    for attempt in range(max_retries):
        try:
            response = requests.post(
                url,
                json=construct_graphql_query(pdb_chunk),
                timeout=30
            )

            if response.status_code != 200:
                time.sleep(2 ** attempt)
                continue

            data = response.json()
            if "data" in data and "entries" in data["data"]:
                entries = data["data"]["entries"]
                if entries is None:
                    return pd.DataFrame()

                for entry in entries:
                    pdb_id = entry.get("rcsb_id")
                    if not pdb_id:
                        continue

                    entities = entry.get("polymer_entities", [])
                    for ent in entities:
                        ent_id_full = ent.get("rcsb_id", "_")
                        ent_id = ent_id_full.split("_")[1] if "_" in ent_id_full else ent_id_full

                        identifiers = ent.get("rcsb_polymer_entity_container_identifiers", {})
                        chains = identifiers.get("auth_asym_ids", [])
                        refs = identifiers.get("reference_sequence_identifiers", [])

                        uni_id = None
                        if refs:
                            for ref in refs:
                                if ref.get("database_name") == "UniProt":
                                    uni_id = ref.get("database_accession")
                                    break

                        for c in chains:
                            df_list.append({
                                "PDB ID": pdb_id,
                                "Entity ID": ent_id,
                                "Auth Asym ID": c,
                                "Uniprot ID": uni_id
                            })

                return pd.DataFrame(df_list)

        except Exception:
            time.sleep(1)

    return pd.DataFrame()


def main():
    if not os.path.exists(INPUT_PDB_LIST_CSV):
        print(f"找不到 {INPUT_PDB_LIST_CSV}，请检查路径。")
        return

    df = pd.read_csv(INPUT_PDB_LIST_CSV)
    if "PDB ID" not in df.columns:
        print("CSV 里找不到 'PDB ID' 列，请检查文件格式。")
        return

    all_pdbs = list(set([str(x).strip().upper() for x in df["PDB ID"].dropna().tolist()]))
    print(f"解析到去重后的总 PDB 数量：{len(all_pdbs)}")

    processed = set()
    if os.path.exists(OUTPUT_CSV):
        existing_df = pd.read_csv(OUTPUT_CSV)
        if "PDB ID" in existing_df.columns:
            processed = set(existing_df["PDB ID"].unique())
            print(f"断点续传：已跳过 {len(processed)} 个现存 PDB。")

    pdbs_to_do = sorted(list(set(all_pdbs) - processed))
    print(f"本次实际派发请求数量：{len(pdbs_to_do)}")
    if len(pdbs_to_do) == 0:
        return

    chunks = [pdbs_to_do[i:i + BATCH_SIZE] for i in range(0, len(pdbs_to_do), BATCH_SIZE)]
    print(f"生成 {len(chunks)} 个 GraphQL 批处理数据块，每块 {BATCH_SIZE} 个。")

    header = not os.path.exists(OUTPUT_CSV)

    with Pool(WORKERS) as pool:
        with tqdm(total=len(chunks), desc="GraphQL Fetching") as pbar:
            for result_df in pool.imap_unordered(fetch_graphql_batch, chunks):
                if not result_df.empty:
                    result_df.to_csv(OUTPUT_CSV, mode="a", index=False, header=header)
                    header = False
                pbar.update(1)

    print(f"全部完成，输出文件：{OUTPUT_CSV}")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"总耗时: {(time.time() - start_time)/60:.2f} 分钟")

# 将以上程序封装在1个py脚本中，或者在1个独立的notebook cell中，然后再执行


# 执行之后的效果如下:
"""

PDB ID	Entity ID	Auth Asym ID	Uniprot ID
0	1ENA	1	A	P00644
1	1ENC	1	A	P00644
2	1ENH	1	A	P02836
3	1ENO	1	A	P80030
4	1ENP	1	A	P80030

"""

# ⚠️ 然后下一步就是进行统计分析，具体就是查看每一个pbd结构中的uniprot id的分布
# 比如说绘制1个直方图

import matplotlib.pyplot as plt
import seaborn as sns

# 统计每个 PDB 对应多少个独立 Uniprot ID
pdb_uniprot_counts = (
    df_pdb_map.dropna(subset=["Uniprot ID"])
              .groupby("PDB ID")["Uniprot ID"]
              .nunique()
)

print(pdb_uniprot_counts.describe())

# 画直方图
plt.figure(figsize=(12, 6))
sns.histplot(pdb_uniprot_counts, bins=100, kde=False, color="steelblue")
plt.xlabel("Number of unique Uniprot IDs per PDB")
plt.ylabel("Number of PDB entries")
plt.title("Distribution of Uniprot IDs per PDB")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()

# 如果要限制x轴的绘制范围

plt.figure(figsize=(12, 6))
sns.histplot(pdb_uniprot_counts, bins=200, kde=False, color="steelblue")
plt.xlim(0, 10)
plt.xlabel("Number of unique Uniprot IDs per PDB")
plt.ylabel("Number of PDB entries")
plt.title("Distribution of Uniprot IDs per PDB")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


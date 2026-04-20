# colabfold, 参考https://github.com/YoshitakaMo/localcolabfold
# 安装简单，消费级gpu就能够跑

# 1.

#!/bin/bash

INPUTDIR="~/alphafold/localcolabfold/input/"
OUTPUTBASE="~/alphafold/localcolabfold/output/"
RANDOMSEED=0

export PATH="~/alphafold/localcolabfold/.pixi/envs/default/bin:$PATH"

# 检查目录
if [ ! -d "$INPUTDIR" ]; then
    echo "ERROR: INPUTDIR not found: $INPUTDIR"
    exit 1
fi

mkdir -p "$OUTPUTBASE"

for fasta in "$INPUTDIR"/*.fasta
do
    # 防止空匹配
    [ -e "$fasta" ] || continue

    name=$(basename "$fasta" .fasta)
    outdir="${OUTPUTBASE}/${name}"

    mkdir -p "$outdir"

    echo "Running $name ..."

    colabfold_batch \
      --num-recycle 3 \
      --amber \
      --templates \
      --use-gpu-relax \
      --num-models 5 \
      --model-order 1,2,3,4,5 \
      --random-seed ${RANDOMSEED} \
      "$fasta" \
      "$outdir"

done


#########################################################################################################################

# 2.

#!/bin/bash

# INPUTFILE="1BJP_1" # 输入文件, 不包含扩展名, 此示例中为1BJP_1.fasta
INPUTFILE="CTCF_IDR"
OUTPUTDIR="${INPUTFILE}" # 输出目录, 可以与输入文件相同, 也可以不同
RANDOMSEED=0 

export PATH="/mnt/sdb/zht/localcolabfold/.pixi/envs/default/bin:${PATH}"

colabfold_batch \
  --num-recycle 3 \
  --amber \
  --templates \
  --use-gpu-relax \
  --num-models 5 \
  --model-order 1,2,3,4,5 \
  --random-seed ${RANDOMSEED} \
  ${INPUTFILE}.fasta \
  ${OUTPUTDIR}

def download_fq(sra_file,num_threads,Outdir):
  """


  """
  import subprocess
  with open(sra_file,"r") as sras:
    for sra in sras:
      sra = sra.stripe("\n")
      print (f"currently downloading {sra}\n")
      prefetch = f"prefetch {sra} --maxsize 50G -O {Outdir}"
      subprocess.run(prefetch,check=True,shell=True)
      print(f"generating fastq for {sra}\n")
      fasterq-dump = f"fasterq-dump {sra} -e {num_threads}"
      subprocess.run(fasterq_dump,check=True,shell=True)




=======================

# 如果是在shell中，可以直接操作

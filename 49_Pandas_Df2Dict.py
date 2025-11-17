# 将pandas中的dataframe转化为dict
# key为protein_id, value为其他
c2h2_zfp_NCR_af3_dict = c2h2_zfp_NCR_af3.set_index('protein_id')[['motif_consensus', 'zf_count', 'sequence']].to_dict('index')

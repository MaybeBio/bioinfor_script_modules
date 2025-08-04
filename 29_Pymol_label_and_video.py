# ⚠️使用场景：
# AF3获取的model_0.cif置信度最高，其结构坐标可以放到pymol中进行可视化
# 主要是标注自己所研究的感兴趣的region，label自定义
# 然后制作简短的视频video，可在ppt上播放

# 1,外部设置
cd /mnt/sdb/zht/project/uniprot/pymol_chimerax

# 打开日志，记录所有操作指令
log_open ctcf_2

# 2,首先是内部全局的设置
set assembly,""

# 3,获取对应蛋白质的CIF文件
# 加载 fold_ctcf_2copy_model_0.cif 文件并命名为 fold_ctcf_2copy_model_0
load /mnt/sdb/zht/project/uniprot/pymol_chimerax/fold_ctcf_2copy_model_0.cif, fold_ctcf_2copy_model_0

# 4,对获取的原始CIF文件中的蛋白质进行区域选择
#有编号用resi,无编号用pepseqls

# 标注酸碱性区域，酸红碱蓝

# 标注碱性区域
select Basic1_chainA, fold_ctcf_2copy_model_0 and chain A and resi 20-29
color blue, Basic1_chainA
# 在区域中心添加整体标注
label (fold_ctcf_2copy_model_0 and chain A and resi 24 and name CA), "Basic1 (Chain A)"

select Basic1_chainB, fold_ctcf_2copy_model_0 and chain B and resi 20-29
color blue, Basic1_chainB
label (fold_ctcf_2copy_model_0 and chain B and resi 24 and name CA), "Basic1 (Chain B)"

select Basic2_chainA, fold_ctcf_2copy_model_0 and chain A and resi 199-215
color blue, Basic2_chainA
label (fold_ctcf_2copy_model_0 and chain A and resi 207 and name CA), "Basic2 (Chain A)"

select Basic2_chainB, fold_ctcf_2copy_model_0 and chain B and resi 199-215
color blue, Basic2_chainB
label (fold_ctcf_2copy_model_0 and chain B and resi 207 and name CA), "Basic2 (Chain B)"

select Basic3_chainA, fold_ctcf_2copy_model_0 and chain A and resi 250-267
color blue, Basic3_chainA
label (fold_ctcf_2copy_model_0 and chain A and resi 258 and name CA), "Basic3 (Chain A)"

select Basic3_chainB, fold_ctcf_2copy_model_0 and chain B and resi 250-267
color blue, Basic3_chainB
label (fold_ctcf_2copy_model_0 and chain B and resi 258 and name CA), "Basic3 (Chain B)"

select Basic4_chainA, fold_ctcf_2copy_model_0 and chain A and resi 292-301
color blue, Basic4_chainA
label (fold_ctcf_2copy_model_0 and chain A and resi 296 and name CA), "Basic4 (Chain A)"

select Basic4_chainB, fold_ctcf_2copy_model_0 and chain B and resi 292-301
color blue, Basic4_chainB
label (fold_ctcf_2copy_model_0 and chain B and resi 296 and name CA), "Basic4 (Chain B)"

select Basic5_chainA, fold_ctcf_2copy_model_0 and chain A and resi 336-350
color blue, Basic5_chainA
label (fold_ctcf_2copy_model_0 and chain A and resi 343 and name CA), "Basic5 (Chain A)"

select Basic5_chainB, fold_ctcf_2copy_model_0 and chain B and resi 336-350
color blue, Basic5_chainB
label (fold_ctcf_2copy_model_0 and chain B and resi 343 and name CA), "Basic5 (Chain B)"

select Basic6_chainA, fold_ctcf_2copy_model_0 and chain A and resi 363-375
color blue, Basic6_chainA
label (fold_ctcf_2copy_model_0 and chain A and resi 369 and name CA), "Basic6 (Chain A)"

select Basic6_chainB, fold_ctcf_2copy_model_0 and chain B and resi 363-375
color blue, Basic6_chainB
label (fold_ctcf_2copy_model_0 and chain B and resi 369 and name CA), "Basic6 (Chain B)"

select Basic7_chainA, fold_ctcf_2copy_model_0 and chain A and resi 391-403
color blue, Basic7_chainA
label (fold_ctcf_2copy_model_0 and chain A and resi 397 and name CA), "Basic7 (Chain A)"

select Basic7_chainB, fold_ctcf_2copy_model_0 and chain B and resi 391-403
color blue, Basic7_chainB
label (fold_ctcf_2copy_model_0 and chain B and resi 397 and name CA), "Basic7 (Chain B)"

select Basic8_chainA, fold_ctcf_2copy_model_0 and chain A and resi 485-496
color blue, Basic8_chainA
label (fold_ctcf_2copy_model_0 and chain A and resi 491 and name CA), "Basic8 (Chain A)"

select Basic8_chainB, fold_ctcf_2copy_model_0 and chain B and resi 485-496
color blue, Basic8_chainB
label (fold_ctcf_2copy_model_0 and chain B and resi 491 and name CA), "Basic8 (Chain B)"

select Basic9_chainA, fold_ctcf_2copy_model_0 and chain A and resi 508-518
color blue, Basic9_chainA
label (fold_ctcf_2copy_model_0 and chain A and resi 513 and name CA), "Basic9 (Chain A)"

select Basic9_chainB, fold_ctcf_2copy_model_0 and chain B and resi 508-518
color blue, Basic9_chainB
label (fold_ctcf_2copy_model_0 and chain B and resi 513 and name CA), "Basic9 (Chain B)"

select Basic10_chainA, fold_ctcf_2copy_model_0 and chain A and resi 590-607
color blue, Basic10_chainA
label (fold_ctcf_2copy_model_0 and chain A and resi 598 and name CA), "Basic10 (Chain A)"

select Basic10_chainB, fold_ctcf_2copy_model_0 and chain B and resi 590-607
color blue, Basic10_chainB
label (fold_ctcf_2copy_model_0 and chain B and resi 598 and name CA), "Basic10 (Chain B)"

select Basic11_chainA, fold_ctcf_2copy_model_0 and chain A and resi 645-659
color blue, Basic11_chainA
label (fold_ctcf_2copy_model_0 and chain A and resi 652 and name CA), "Basic11 (Chain A)"

select Basic11_chainB, fold_ctcf_2copy_model_0 and chain B and resi 645-659
color blue, Basic11_chainB
label (fold_ctcf_2copy_model_0 and chain B and resi 652 and name CA), "Basic11 (Chain B)"

# 标注酸性区域
select Acidic1_chainA, fold_ctcf_2copy_model_0 and chain A and resi 222-236
color red, Acidic1_chainA
label (fold_ctcf_2copy_model_0 and chain A and resi 229 and name CA), "Acidic1 (Chain A)"

select Acidic1_chainB, fold_ctcf_2copy_model_0 and chain B and resi 222-236
color red, Acidic1_chainB
label (fold_ctcf_2copy_model_0 and chain B and resi 229 and name CA), "Acidic1 (Chain B)"

select Acidic2_chainA, fold_ctcf_2copy_model_0 and chain A and resi 607-635
color red, Acidic2_chainA
label (fold_ctcf_2copy_model_0 and chain A and resi 621 and name CA), "Acidic2 (Chain A)"

select Acidic2_chainB, fold_ctcf_2copy_model_0 and chain B and resi 607-635
color red, Acidic2_chainB
label (fold_ctcf_2copy_model_0 and chain B and resi 621 and name CA), "Acidic2 (Chain B)"

select Acidic3_chainA, fold_ctcf_2copy_model_0 and chain A and resi 691-707
color red, Acidic3_chainA
label (fold_ctcf_2copy_model_0 and chain A and resi 699 and name CA), "Acidic3 (Chain A)"

select Acidic3_chainB, fold_ctcf_2copy_model_0 and chain B and resi 691-707
color red, Acidic3_chainB
label (fold_ctcf_2copy_model_0 and chain B and resi 699 and name CA), "Acidic3 (Chain B)"

#最后找到一个好位点:
bg_color white
ray 1000,1000

# 5,旋转各种位点，如果找到好的位点可以写入日志，并拍图
get_view
# png ctcf2.png
#至于要不要show其他形状看自己

# 6,一切处理完毕之后
log_close

# 7,创建旋转动画
mset 1 x120  
# 表示从第1帧到第120帧，创建一个长度为120帧的动画

util.mroll 1, 120, 1  
# 1：起始帧，表示动画从第1帧开始。
# 120：结束帧，表示动画在第120帧结束。
# 1：循环次数，表示动画循环1次。

mplay
# 播放动画

# 保存动画为PNG文件序列
# ⚠️先在所需目录下构建ctcf_rotation文件夹
mpng ctcf_rotation/ctcf2
#将动画的每一帧保存为PNG文件，表示保存的PNG文件将以 ctcf_rotation/frame 前缀开头，并附加帧编号，文件名为frame0001.png、frame0002.png等。

# ⚠️./ctcf2%04d.png可以换成前面自定义存储png动画帧的文件夹，另外分辨率也可以选择1080p，即1920:1080，或者其他等
ffmpeg -framerate 10 -i ./ctcf2%04d.png -vf "scale=640:464" -c:v libx264 -crf 18 -pix_fmt yuv420p ctcf_2.mp4
# 保存动画为MP4文件,10表示每秒10帧，-vf "scale=640:464"表示将视频的分辨率调整为640x464，-c:v libx264表示使用libx264编码器，-crf 18表示设置视频质量，-pix_fmt yuv420p表示设置像素格式为yuv420p。
# ./ctcf2%04d.png表示当前文件夹中输入的PNG文件序列，%04d表示帧编号，会自动匹配所有以 ctcf2 开头，以 .png 结尾的文件。

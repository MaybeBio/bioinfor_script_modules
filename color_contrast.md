

import matplotlib.pyplot as plt
import numpy as np

# 创建模拟的红蓝色相关性热图
np.random.seed(42)
correlation_matrix = np.random.randn(50, 50) * 0.5
# 添加一些结构化的相关性模式
for i in range(50):
    for j in range(50):
        if abs(i-j) < 5:
            correlation_matrix[i,j] += 0.7
        elif abs(i-j) < 10:
            correlation_matrix[i,j] += 0.3

# 限制相关系数范围在-1到1之间
correlation_matrix = np.clip(correlation_matrix, -1, 1)

# 模拟边界位置
boundaries = [12, 25, 38]

# 创建颜色对比图
colors_to_test = [
    ('yellow', 'Yellow (原始)'),
    ('gold', 'Gold (深黄色)'),
    ('orange', 'Orange (橙色)'),
    ('lime', 'Lime (亮绿色)'),
    ('white', 'White (白色)'),
    ('black', 'Black (黑色)'),
    ('#FF4500', 'OrangeRed (橙红色)'),
    ('magenta', 'Magenta (紫红色)')
]

# 创建子图
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.ravel()

for idx, (color, label) in enumerate(colors_to_test):
    ax = axes[idx]
    
    # 绘制热图
    im = ax.imshow(correlation_matrix, cmap='RdBu_r', 
                   norm=plt.Normalize(-1, 1), aspect='auto')
    
    # 绘制边界线
    for boundary in boundaries:
        ax.axvline(boundary, color=color, alpha=1, linewidth=3)
        ax.axhline(boundary, color=color, alpha=1, linewidth=3)
    
    ax.set_title(f'{label}', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 50)
    
    # 只在第一个子图显示坐标轴标签
    if idx == 0:
        ax.set_xlabel('Residue Position')
        ax.set_ylabel('Residue Position')
    else:
        ax.set_xticks([])
        ax.set_yticks([])

# 添加总标题
fig.suptitle('边界线颜色在红蓝色相关性热图上的显示效果对比', 
             fontsize=16, fontweight='bold', y=0.98)

# 添加颜色条
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Pearson Correlation\nCoefficient', 
               rotation=0, labelpad=20, fontsize=10)

plt.tight_layout()
plt.subplots_adjust(right=0.9)
plt.show()

# 打印显眼程度评估
print("颜色显眼程度评估 (基于红蓝色背景):")
print("=" * 50)
visibility_ranking = [
    ("Lime (亮绿色)", "★★★★★", "与红蓝都形成强烈对比"),
    ("White (白色)", "★★★★★", "在所有区域都很显眼"),
    ("Gold (深黄色)", "★★★★☆", "比普通黄色更深，对比度好"),
    ("OrangeRed (橙红色)", "★★★★☆", "与蓝色形成互补色对比"),
    ("Black (黑色)", "★★★★☆", "经典选择，稳定显眼"),
    ("Magenta (紫红色)", "★★★☆☆", "在白色区域显眼"),
    ("Orange (橙色)", "★★★☆☆", "在蓝色区域显眼"),
    ("Yellow (原始)", "★★☆☆☆", "在红色区域不够显眼")
]

for color, stars, comment in visibility_ranking:
    print(f"{color:<20} {stars} - {comment}")


![Uploading image.png…]()

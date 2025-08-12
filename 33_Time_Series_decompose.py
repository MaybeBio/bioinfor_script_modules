# 参考：https://mp.weixin.qq.com/s/Tqe0YamfDz5-xabEaiSCFQ
# 时间序列分析+protein sequence
# 时间序列分解
# FFT filter noise同本质

# 1，乘法model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 示例蛋白质序列
protein_sequence = "MKRHDVKRHSDE"

# 定义电荷规则
charge_dict = {
    'K': 1, 'R': 1, 'H': 1,
    'D': -1, 'E': -1,
    'A': 0, 'C': 0, 'F': 0, 'G': 0, 'I': 0, 'L': 0, 'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
}

# 将蛋白质序列转换为电荷信号序列
charge_signal = [charge_dict[aa] for aa in protein_sequence]

# 创建DataFrame
df = pd.DataFrame({'position': range(1, len(charge_signal) + 1), 'charge': charge_signal})

# 绘制原始电荷信号序列
plt.plot(df['position'], df['charge'], linestyle=':', marker='o', markersize=5)
plt.xlabel('Position')
plt.ylabel('Charge')
plt.title('Charge Signal Sequence')
plt.show()

# 使用statsmodels进行时间序列分解
# 由于序列较短，可能需要调整周期
result = seasonal_decompose(df['charge'], model='additive', period=3)

# 绘制分解结果
result.plot()
plt.show()

# 提取趋势、季节性和随机成分
trend = result.trend
seasonal = result.seasonal
residual = result.resid

# 可视化趋势成分
plt.plot(df['position'], trend, label='Trend')
plt.xlabel('Position')
plt.ylabel('Charge')
plt.title('Trend Component')
plt.legend()
plt.show()

# 可视化季节性成分
plt.plot(df['position'], seasonal, label='Seasonality')
plt.xlabel('Position')
plt.ylabel('Charge')
plt.title('Seasonal Component')
plt.legend()
plt.show()

# 可视化随机成分
plt.plot(df['position'], residual, label='Residual')
plt.xlabel('Position')
plt.ylabel('Charge')
plt.title('Residual Component')
plt.legend()
plt.show()



=================================================================================================================================================================
# 2，加性model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import signal
import seaborn as sns

def analyze_charge_regions(protein_sequence, window_size=5, threshold=0.3):
    """
    Analyze the charge-enriched regions in protein sequences
    
    Parameters:
    - protein_sequence: 蛋白质序列
    - window_size: 滑动窗口大小
    - threshold: 富集区域阈值
    """
    
    # 定义电荷规则
    charge_dict = {
        'K': 1, 'R': 1, 'H': 1,      # 正电荷
        'D': -1, 'E': -1,             # 负电荷
        'A': 0, 'C': 0, 'F': 0, 'G': 0, 'I': 0, 'L': 0, 
        'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'S': 0, 'T': 0, 
        'V': 0, 'W': 0, 'Y': 0        # 中性
    }
    
    # 转换为电荷信号
    charge_signal = [charge_dict.get(aa, 0) for aa in protein_sequence]
    sequence_length = len(charge_signal)
    
    # 创建DataFrame
    df = pd.DataFrame({
        'position': range(1, sequence_length + 1), 
        'charge': charge_signal,
        'amino_acid': list(protein_sequence)
    })
    
    # 1. 绘制原始电荷信号
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0,0].plot(df['position'], df['charge'], linestyle='-', marker='o', markersize=6, linewidth=2)
    axes[0,0].set_xlabel('Position')
    axes[0,0].set_ylabel('Charge')
    axes[0,0].set_title('Original Charge Signal Sequence')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 滑动窗口平滑
    df['smoothed_charge'] = df['charge'].rolling(window=window_size, center=True).mean()
    
    axes[0,1].plot(df['position'], df['charge'], linestyle=':', alpha=0.5, label='Original')
    axes[0,1].plot(df['position'], df['smoothed_charge'], linestyle='-', linewidth=2, label=f'Smoothed (window={window_size})')
    axes[0,1].set_xlabel('Position')
    axes[0,1].set_ylabel('Charge')
    axes[0,1].set_title('Smoothed Charge Signal')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 时间序列分解
    if sequence_length >= 6:  # 确保有足够的数据点
        # 动态调整周期
        if sequence_length < 20:
            period = max(3, sequence_length // 4)
        elif sequence_length < 50:
            period = sequence_length // 5
        else:
            period = 10
            
        # 使用线性插值处理NaN值
        charge_interpolated = df['charge'].interpolate()
        
        try:
            result = seasonal_decompose(charge_interpolated, 
                                      model='additive', 
                                      period=period, 
                                      extrapolate_trend='freq')
            
            # 绘制趋势成分
            axes[1,0].plot(df['position'], result.trend, linewidth=2, color='red')
            axes[1,0].set_xlabel('Position')
            axes[1,0].set_ylabel('Charge')
            axes[1,0].set_title('Trend Component (Long-term charge pattern)')
            axes[1,0].grid(True, alpha=0.3)
            
            # 绘制季节性成分
            axes[1,1].plot(df['position'], result.seasonal, linewidth=2, color='green')
            axes[1,1].set_xlabel('Position')
            axes[1,1].set_ylabel('Charge')
            axes[1,1].set_title('Seasonal Component (Periodic patterns)')
            axes[1,1].grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"时间序列分解失败: {e}")
            axes[1,0].text(0.5, 0.5, 'Decomposition failed\nSequence too short', 
                          transform=axes[1,0].transAxes, ha='center', va='center')
            axes[1,1].text(0.5, 0.5, 'Decomposition failed\nSequence too short', 
                          transform=axes[1,1].transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.show()
    
    # 4. 识别电荷富集区域
    identify_charge_regions(df, threshold=threshold)
    
    return df

def identify_charge_regions(df, threshold=0.3):
    """Identify the regions rich in positive and negative charges"""
    
    # 计算累积电荷
    df['cumulative_charge'] = df['charge'].cumsum()
    
    # 使用滑动窗口计算局部电荷密度
    window = min(5, len(df)//3)  # 动态窗口大小
    df['local_charge_density'] = df['charge'].rolling(window=window, center=True).mean()
    
    # 识别富集区域
    positive_regions = []
    negative_regions = []
    
    # 寻找连续的正/负电荷区域
    current_pos_start = None
    current_neg_start = None
    
    for i, row in df.iterrows():
        charge_density = row['local_charge_density']
        
        if pd.notna(charge_density):
            # 正电荷富集区域
            if charge_density > threshold:
                if current_pos_start is None:
                    current_pos_start = row['position']
                if current_neg_start is not None:
                    negative_regions.append((current_neg_start, row['position']-1))
                    current_neg_start = None
            
            # 负电荷富集区域
            elif charge_density < -threshold:
                if current_neg_start is None:
                    current_neg_start = row['position']
                if current_pos_start is not None:
                    positive_regions.append((current_pos_start, row['position']-1))
                    current_pos_start = None
            
            # 中性区域
            else:
                if current_pos_start is not None:
                    positive_regions.append((current_pos_start, row['position']-1))
                    current_pos_start = None
                if current_neg_start is not None:
                    negative_regions.append((current_neg_start, row['position']-1))
                    current_neg_start = None
    
    # 处理序列末尾的区域
    if current_pos_start is not None:
        positive_regions.append((current_pos_start, df['position'].iloc[-1]))
    if current_neg_start is not None:
        negative_regions.append((current_neg_start, df['position'].iloc[-1]))
    
    # 可视化结果
    plt.figure(figsize=(12, 8))
    
    # 原始信号
    plt.subplot(3, 1, 1)
    plt.plot(df['position'], df['charge'], 'ko-', markersize=6, linewidth=2)
    plt.xlabel('Position')
    plt.ylabel('Charge')
    plt.title('Original Charge Signal')
    plt.grid(True, alpha=0.3)
    
    # 局部电荷密度
    plt.subplot(3, 1, 2)
    plt.plot(df['position'], df['local_charge_density'], 'b-', linewidth=2, label='Local Charge Density')
    plt.axhline(y=threshold, color='r', linestyle='--', alpha=0.7, label=f'Positive threshold (+{threshold})')
    plt.axhline(y=-threshold, color='r', linestyle='--', alpha=0.7, label=f'Negative threshold (-{threshold})')
    
    # 标记富集区域
    for start, end in positive_regions:
        plt.axvspan(start, end, alpha=0.3, color='red', label='Positive region' if (start, end) == positive_regions[0] else "")
    
    for start, end in negative_regions:
        plt.axvspan(start, end, alpha=0.3, color='blue', label='Negative region' if (start, end) == negative_regions[0] else "")
    
    plt.xlabel('Position')
    plt.ylabel('Local Charge Density')
    plt.title(f'Charge Enrichment Regions (window={window}, threshold=±{threshold})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 累积电荷
    plt.subplot(3, 1, 3)
    plt.plot(df['position'], df['cumulative_charge'], 'g-', linewidth=2)
    plt.xlabel('Position')
    plt.ylabel('Cumulative Charge')
    plt.title('Cumulative Charge Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 输出结果
    print("="*50)
    print("CHARGE ENRICHMENT ANALYSIS RESULTS")
    print("="*50)
    print(f"Sequence: {''.join(df['amino_acid'].tolist())}")
    print(f"Length: {len(df)} amino acids")
    print(f"Analysis parameters: window={window}, threshold=±{threshold}")
    print()
    
    if positive_regions:
        print("POSITIVE CHARGE ENRICHMENT REGIONS:")
        for i, (start, end) in enumerate(positive_regions, 1):
            region_seq = ''.join(df[(df['position'] >= start) & (df['position'] <= end)]['amino_acid'].tolist())
            avg_charge = df[(df['position'] >= start) & (df['position'] <= end)]['local_charge_density'].mean()
            print(f"  Region {i}: Position {start}-{end}, Sequence: {region_seq}, Avg density: {avg_charge:.2f}")
    else:
        print("No significant positive charge enrichment regions found.")
    
    print()
    
    if negative_regions:
        print("NEGATIVE CHARGE ENRICHMENT REGIONS:")
        for i, (start, end) in enumerate(negative_regions, 1):
            region_seq = ''.join(df[(df['position'] >= start) & (df['position'] <= end)]['amino_acid'].tolist())
            avg_charge = df[(df['position'] >= start) & (df['position'] <= end)]['local_charge_density'].mean()
            print(f"  Region {i}: Position {start}-{end}, Sequence: {region_seq}, Avg density: {avg_charge:.2f}")
    else:
        print("No significant negative charge enrichment regions found.")

# 使用示例
protein_sequences = [
"MEGDAVEAIVEESETFIKGKERKTYQRRREGGQEEDACHLPQNQTDGGEVVQDVNSSVQMVMMEQLDPTLLQMKTEVMEGTVAPEAEAAVDDTQIITLQVVNMEEQPINIGELQLVQVPVPVTVPVATTSVEELQGAYENEVSKEGLAESEPMICHTLPLPEGFQVVKVGANGEVETLEQGELPPQEDPSWQKDPDYQPPAKKTKKTKKSKLRYTEEGKDVDVSVYDFEEEQQEGLLSEVNAEKVVGNMKPPKPTKIKKKGVKKTFQCELCSYTCPRRSNLDRHMKSHTDERPHKCHLCGRAFRTVTLLRNHLNTHTGTRPHKCPDCDMAFVTSGELVRHRRYKHTHEKPFKCSMCDYASVEVSKLKRHIRSHTGERPFQCSLCSYASRDTYKLKRHMRTHSGEKPYECYICHARFTQSGTMKMHILQKHTENVAKFHCPHCDTVIARKSDLGVHLRKQHSYIEQGKKCRYCDAVFHERYALIQHQKSHKNEKRFKCDQCDYACRQERHMIMHKRTHTGEKPYACSHCDKTFRQKQLLDMHFKRYHDPNFVPAAFVCSKCGKTFTRRNTMARHADNCAGPDGVEGENGGETKKSKRGRKRKMRSKKEDSSDSENAEPDLDDNEDEEEPAVEIEPEPEPQPVTPAPPPAKKRRGRPPGRTNQPKQNQPTAIIQVEDQNTGAIENIIVEVKKEPDAEPAEGEEEEAQPAATDAPNGDLTPEMILSMMDR"
]

for seq in protein_sequences:
    print(f"\n{'='*60}")
    print(f"ANALYZING SEQUENCE: {seq}")
    print(f"{'='*60}")
    
    # 分析序列，可以调整参数
    df = analyze_charge_regions(
        seq, 
        window_size=3,      # 滑动窗口大小
        threshold=0.2       # 富集阈值（可根据需要调整）
    )

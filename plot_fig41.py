import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['Microsoft JhengHei']   # Windows 中文字型
rcParams['axes.unicode_minus'] = False

markets  = ['SPY', 'QQQ', '0050.TW']
arima    = [0.79, 1.04, 0.87]
lstm     = [2.67, 4.43, 3.20]      # 單層 LSTM（新版 6 維）
ensemble = [8.44, 11.09, 8.63]     # 隨機森林（集成樹代表）

x = np.arange(len(markets)); w = 0.25
fig, ax = plt.subplots(figsize=(8, 5))
b1 = ax.bar(x - w, arima,    w, label='ARIMA（統計）',              color='#2ca77f')
b2 = ax.bar(x,     lstm,     w, label='LSTM（單層，深度學習）',      color='#2b7fd4')
b3 = ax.bar(x + w, ensemble, w, label='集成樹模型（以隨機森林為代表）', color='#d4552b')

for b in (b1, b2, b3):
    ax.bar_label(b, fmt='%.2f', padding=3, fontsize=9)

ax.set_ylabel('測試集 MAPE（%）')
ax.set_xticks(x); ax.set_xticklabels(markets)
ax.set_ylim(0, 13)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('fig4-1_mape.png', dpi=200)
plt.show()
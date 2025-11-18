# 檔案名稱: analyze_results.py (修正版)

# --- 步驟 1: 匯入必要的函式庫 ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 步驟 1.1: 設定中文字體 ---
# 解決 Matplotlib 中文顯示問題
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 指定支援中文的字體
plt.rcParams['axes.unicode_minus'] = False # 解決負號顯示問題

print("--- 開始分析實驗結果 ---")

# --- 步驟 2: 讀取您的實驗結果 ---
try:
    df = pd.read_csv('experiment_results.csv')
    print("成功讀取 experiment_results.csv 檔案。")
except FileNotFoundError:
    print("錯誤: 找不到 'experiment_results.csv' 檔案。")
    exit()

# --- 步驟 3: 繪製「圖一：模型橫評」---
print("\n--- 正在生成圖一：模型橫評 (SPY) ---")
spy_results = df[df['Ticker'] == 'SPY']

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 7)) # 把圖表拉寬一點，讓模型名字顯示更清楚

# 繪製柱狀圖，並將模型按照準確率從高到低排序
spy_results_sorted = spy_results.sort_values('Accuracy', ascending=False)
sns.barplot(x='Model', y='Accuracy', data=spy_results_sorted, palette='viridis')

plt.title('圖一：不同模型在 SPY ETF 上的準確率比較', fontsize=18)
plt.xlabel('模型 (Model)', fontsize=14)
plt.ylabel('準確率 (Accuracy)', fontsize=14)
plt.xticks(rotation=15) # 將 X 軸標籤稍微旋轉，避免重疊
plt.ylim(0.50, 0.60) # Y 軸範圍
plt.show()


# --- 步驟 4: 繪製「圖二：市場橫評」---
# 【修正點】我們選擇表現較好的 DoubleLayerBiLSTM 來進行比較
MODEL_FOR_COMPARISON = 'DoubleLayerBiLSTM' 
print(f"\n--- 正在生成圖二：市場橫評 ({MODEL_FOR_COMPARISON}) ---")

# 【修正點】修改篩選條件
model_comparison_results = df[df['Model'] == MODEL_FOR_COMPARISON]

if model_comparison_results.empty:
    print(f"錯誤: 在結果中找不到模型 '{MODEL_FOR_COMPARISON}' 的數據，無法生成圖二。")
else:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Ticker', y='Accuracy', data=model_comparison_results, palette='plasma')
    plt.title(f'圖二：{MODEL_FOR_COMPARISON} 模型在不同市場 ETF 上的準確率比較', fontsize=16)
    plt.xlabel('ETF 標的 (Ticker)', fontsize=12)
    plt.ylabel('準確率 (Accuracy)', fontsize=12)
    plt.ylim(0.50, 0.60)
    plt.show()

print("\n--- 分析圖表生成完畢 ---")
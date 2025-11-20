# 檔案名稱: analyze_results.py (F1-Score 最終版)
# 功能: 讀取實驗結果，繪製以 F1-Score 為核心指標的比較圖，並直接儲存。

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("--- 開始分析實驗結果 (F1-Score 模式) ---")

# --- 步驟 2: 讀取您的實驗結果 ---
try:
    df = pd.read_csv('experiment_results.csv')
    print("成功讀取 experiment_results.csv 檔案。")
except FileNotFoundError:
    print("錯誤: 找不到 'experiment_results.csv' 檔案。")
    exit()

# --- 步驟 3: 繪製並儲存「圖 4-1：模型橫評」---
print("\n--- 正在生成圖 4-1：模型橫評 (SPY, F1-Score) ---")
spy_results = df[df['Ticker'] == 'SPY']

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 7))

# 【核心修改】根據 F1-Score 進行排序和繪圖
spy_results_sorted = spy_results.sort_values('F1-Score', ascending=False)
sns.barplot(x='Model', y='F1-Score', data=spy_results_sorted, palette='viridis', hue='Model', legend=False)

# 【核心修改】圖表標題與軸標籤全部更新為 F1-Score
plt.title('Figure 4-1: Model F1-Score Comparison on SPY ETF', fontsize=16)
plt.xlabel('Model', fontsize=14)
plt.ylabel('F1-Score', fontsize=14)
plt.xticks(rotation=15)

# 動態設定Y軸，確保所有柱子可見
y_min = spy_results_sorted['F1-Score'].min() - 0.05 
plt.ylim(bottom=y_min)

# 使用論文中的圖表編號命名
output_filename_1 = 'figure_4-1_model_comparison_F1.png'
plt.savefig(output_filename_1, dpi=300, bbox_inches='tight')
print(f"圖 4-1 已成功儲存為 '{output_filename_1}'")
plt.close()

# --- 步驟 4: 繪製並儲存「圖 4-2：市場橫評」---
# 【核心修改】選擇在SPY上 F1-Score 最高的模型 DoubleLayerLSTM
MODEL_FOR_COMPARISON = 'DoubleLayerLSTM' 
print(f"\n--- 正在生成圖 4-2：市場橫評 ({MODEL_FOR_COMPARISON}, F1-Score) ---")

model_comparison_results = df[df['Model'] == MODEL_FOR_COMPARISON]

if model_comparison_results.empty:
    print(f"錯誤: 在結果中找不到模型 '{MODEL_FOR_COMPARISON}' 的數據。")
else:
    plt.figure(figsize=(10, 6))
    # 【核心修改】根據 F1-Score 繪圖
    sns.barplot(x='Ticker', y='F1-Score', data=model_comparison_results, palette='plasma', hue='Ticker', legend=False)
    
    # 【核心修改】圖表標題與軸標籤全部更新為 F1-Score
    plt.title(f'Figure 4-2: {MODEL_FOR_COMPARISON} F1-Score Across Different Markets', fontsize=16)
    plt.xlabel('ETF Ticker', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    
    # 動態設定Y軸
    y_min_market = model_comparison_results['F1-Score'].min() - 0.05
    plt.ylim(bottom=y_min_market)
    
    # 使用論文中的圖表編號命名
    output_filename_2 = 'figure_4-2_market_comparison_F1.png'
    plt.savefig(output_filename_2, dpi=300, bbox_inches='tight')
    print(f"圖二已成功儲存為 '{output_filename_2}'")
    plt.close()

print("\n--- 分析圖表生成完畢 ---")
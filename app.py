# --- 1. 匯入所有必要的函式庫 ---
import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # 新增：用於繪製雙軸圖
from joblib import load
from pathlib import Path
import os
import openai
from dotenv import load_dotenv

# --- 2. 初始化設定 ---

# 載入 .env 檔案中的環境變數
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# 設定 Streamlit 頁面
st.set_page_config(page_title="AEGIS 氣候金融風險分析平台", page_icon="🌍", layout="wide")

# --- 3. 核心功能函式 (帶有快取功能) ---
@st.cache_data(ttl=3600)
def load_and_prepare_data(ticker):
    """從 SQLite 讀取資料並準備特徵"""
    DB_FILE = Path("etf_data.db")
    if not DB_FILE.exists():
        return None
    
    conn = sqlite3.connect(DB_FILE)
    try:
        # 1. 名稱轉換 (例如 "0050.TW" -> "0050_tw")
        table_name = ticker.lower().replace('.', '_')

        # 2. 查詢表格
        query = 'SELECT * FROM "{}"'.format(table_name)
        df = pd.read_sql_query(query, conn, index_col='Date', parse_dates=['Date'])
    except Exception as e:
        return None
    finally:
        conn.close()
    
    # 特徵工程
    df["Return"] = df["Close"].pct_change()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["Volatility"] = df["Return"].rolling(20).std()
    
    def calc_rsi(s, period=14):
        delta = s.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.rolling(period).mean() / loss.rolling(period).mean()
        return 100 - (100 / (1 + rs))
    
    df["RSI"] = calc_rsi(df["Close"])
    df.dropna(inplace=True)
    return df

@st.cache_resource
def load_model(ticker):
    """載入對應的模型 (包含檔名容錯處理)"""
    # 1. 嘗試直接載入 (例如 rf_SPY.joblib)
    path1 = Path("models") / f"rf_{ticker}.joblib"
    
    # 2. 嘗試將 . 換成 _ (解決 rf_0050.TW.joblib vs rf_0050_TW.joblib 的問題)
    safe_ticker = ticker.replace('.', '_')
    path2 = Path("models") / f"rf_{safe_ticker}.joblib"
    
    # 3. 嘗試全小寫 (以防萬一檔名是 rf_0050_tw.joblib)
    path3 = Path("models") / f"rf_{safe_ticker.lower()}.joblib"

    # 依序檢查檔案是否存在
    if path1.exists():
        return load(path1)['model']
    elif path2.exists():
        return load(path2)['model']
    elif path3.exists():
        return load(path3)['model']
    
    return None

def get_climate_gpt_summary(ticker, ticker_pred, ticker_conf, krbn_pred, krbn_conf, latest_data, correlation):
    """生成極簡版氣候金融風險摘要"""
    if not openai.api_key:
        return "錯誤：OpenAI API Key 未設定。"
    
    # 修改後的 Prompt：強調「極度精簡」與「條列式」
    prompt = f"""
    你是專業操盤手。請根據數據直接給出 **3 點關鍵操作結論**，**嚴禁廢話**，總字數控制在 100 字內：

    [市場數據]
    - 碳權 (KRBN) 預測：{krbn_pred} (信心 {krbn_conf:.0f}%)
    - 標的 ({ticker}) 預測：{ticker_pred} (信心 {ticker_conf:.0f}%)
    - 相關係數：{correlation:.2f} (負值=避險有效, 正值=同步波動)
    - 標的 RSI：{latest_data['RSI']:.0f}

    [輸出格式]
    1. 🎯 **避險判斷**：(一句話判定 KRBN 是否能保護 {ticker})
    2. ⚡ **趨勢訊號**：(解讀兩者方向一致或相反的意義)
    3. 💡 **操作建議**：(直接給出加碼、減碼或觀望建議)
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, # 降低溫度，讓回答更收斂、更精準
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"摘要生成失敗：{e}"

# --- 4. Streamlit 介面佈局 ---

st.title("🌍 AEGIS：基於生成式 AI 之碳權與美股雙軌分析平台")

with st.sidebar:
    st.header("⚙️ 控制面板")
    # KRBN 已經在後台自動跑了，這裡選的是使用者想看的「主標的」
    selected_ticker = st.selectbox("請選擇投資標的 ETF:", ("SPY", "QQQ", "0050.TW"))
    st.info("系統已自動載入 KRBN (碳權) 數據進行交叉比對。")

# 1. 載入主標的資料
main_data = load_and_prepare_data(selected_ticker)
main_model = load_model(selected_ticker)

# 2. 載入碳權 (KRBN) 資料 - 這是雙軌分析的關鍵
krbn_ticker = "KRBN"
krbn_data = load_and_prepare_data(krbn_ticker)
krbn_model = load_model(krbn_ticker)

if main_data is not None and main_model is not None:
    # --- 預測主標的 ---
    features = ["MA20", "MA60", "Volatility", "RSI"]
    latest_main = main_data[features].iloc[[-1]]
    
    main_pred_val = main_model.predict(latest_main)[0]
    main_prob = main_model.predict_proba(latest_main)[0]
    
    main_pred_str = "看漲 ▲" if main_pred_val == 1 else "看跌 ▼"
    main_delta_color = "normal" if main_pred_val == 1 else "inverse"
    main_conf_score = main_prob[main_pred_val] * 100

    # --- 預測碳權 (KRBN) & 計算相關性 ---
    correlation = 0.0
    corr_desc = "資料不足"
    corr_color = "gray"

    # 如果還沒訓練 KRBN 模型，這裡給個預設值避免報錯
    if krbn_data is not None and krbn_model is not None:
        latest_krbn = krbn_data[features].iloc[[-1]]
        krbn_pred_val = krbn_model.predict(latest_krbn)[0]
        krbn_prob = krbn_model.predict_proba(latest_krbn)[0]
        
        krbn_pred_str = "看漲 ▲" if krbn_pred_val == 1 else "看跌 ▼"
        krbn_delta_color = "normal" if krbn_pred_val == 1 else "inverse"
        krbn_conf_score = krbn_prob[krbn_pred_val] * 100

        # === 新增功能：計算相關係數 ===
        # 確保兩個資料集的日期對齊
        common_index = main_data.index.intersection(krbn_data.index)
        if len(common_index) > 60:
            df_main_aligned = main_data.loc[common_index]
            df_krbn_aligned = krbn_data.loc[common_index]
            
            # 計算 60 日滾動相關係數的最後一筆值
            correlation = df_main_aligned['Close'].rolling(window=60).corr(df_krbn_aligned['Close']).iloc[-1]
            
            # 定義相關性描述與顏色
            if correlation > 0.5:
                corr_desc = "高度正相關 (同步波動)"
                corr_color = "#ff4b4b" # 紅色：對避險來說是壞事
            elif correlation < -0.3:
                corr_desc = "負相關 (具避險效果)"
                corr_color = "#09ab3b" # 綠色：對避險來說是好事
            else:
                corr_desc = "低度相關 (走勢脫鉤)"
                corr_color = "gray"
    else:
        krbn_pred_str = "資料不足"
        krbn_delta_color = "off"
        krbn_conf_score = 0.0

    # --- 介面顯示區 ---
    
    # 區塊 1: 雙軌預測儀表板
    st.header("📊 雙軌趨勢預測儀表板")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"標的：{selected_ticker}")
        st.metric(label="AI 趨勢預測", value=main_pred_str, delta=f"信心度 {main_conf_score:.1f}%", delta_color=main_delta_color)
    
    with col2:
        st.subheader(f"碳權指標：{krbn_ticker}")
        st.metric(label="碳價趨勢預測", value=krbn_pred_str, delta=f"信心度 {krbn_conf_score:.1f}%", delta_color=krbn_delta_color)

    # 顯示相關係數指標
    if krbn_data is not None:
        st.markdown(f"#### 🔗 {selected_ticker} 與 碳權 (60日) 相關係數： <span style='color:{corr_color};font-size:20px'>{correlation:.2f} ({corr_desc})</span>", unsafe_allow_html=True)

    st.markdown("---")

    # 區塊 2: 雙軸走勢對照圖 (升級版)
    st.subheader(f"📈 {selected_ticker} vs. 碳權 (KRBN) 走勢對照")
    
    if krbn_data is not None:
        # 使用 make_subplots 建立雙軸圖
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 加入主標的 (左軸)
        fig.add_trace(
            go.Scatter(x=df_main_aligned.index, y=df_main_aligned['Close'], name=f"{selected_ticker} 收盤價"),
            secondary_y=False
        )

        # 加入 KRBN (右軸) - 設定為虛線或不同顏色以示區別
        fig.add_trace(
            go.Scatter(x=df_krbn_aligned.index, y=df_krbn_aligned['Close'], name="KRBN 碳權價格",
                       line=dict(color='rgba(255, 99, 71, 0.7)', dash='dot')), # 番茄紅虛線
            secondary_y=True
        )

        # 設定標題與軸名稱
        fig.update_layout(height=500, hovermode="x unified", title_text="股價與碳價趨勢對比 (觀察蹺蹺板效應)")
        fig.update_yaxes(title_text=f"{selected_ticker} 價格", secondary_y=False)
        fig.update_yaxes(title_text="KRBN 價格 ($)", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        # 如果沒有 KRBN 資料，退回顯示單一圖表
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=main_data.index, y=main_data['Close'], mode='lines', name='收盤價'))
        fig.add_trace(go.Scatter(x=main_data.index, y=main_data['MA20'], mode='lines', name='20日均線', line=dict(dash='dot', color='orange')))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # 區塊 3: 氣候金融 AI 摘要
    st.subheader("🤖 AEGIS 氣候金融風險摘要")
    
    if st.button("生成 AI 分析報告"):
        with st.spinner("正在整合碳權數據、相關係數與市場走勢，生成氣候風險評估中..."):
            # 準備傳給 GPT 的純文字參數 (包含 correlation)
            summary = get_climate_gpt_summary(
                ticker=selected_ticker,
                ticker_pred=main_pred_str,
                ticker_conf=main_conf_score,
                krbn_pred=krbn_pred_str,
                krbn_conf=krbn_conf_score,
                latest_data=main_data.iloc[-1],
                correlation=correlation
            )
            st.success("分析完成！")
            st.info(summary)
    else:
        st.write("點擊按鈕以啟動 GenAI 進行雙軌關聯分析。")

else:
    st.error("系統錯誤：無法載入資料庫或模型檔案。")
    st.warning("請確認：1. 是否已執行 update_database.py？ 2. 是否已執行 train_rf.py 訓練所有模型（含 KRBN）？")
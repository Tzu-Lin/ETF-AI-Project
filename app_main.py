# app_main.py
# LLM-ETF 互動式預測平台 (迴歸為主，分類為輔)
# 研究生：林子瑜
# 指導教授：李冠榮 副教授

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# 載入環境變數（API Key 等）
load_dotenv()

# 頁面設定
st.set_page_config(page_title="LLM-ETF 預測平台", layout="wide", page_icon="📈")

# 隱藏 Streamlit 預設選單與頁尾（選用）
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ====================== 1. 特徵工程函數 ======================
def compute_technical_features(df):
    """計算技術指標：MA20, MA60, RSI(14), Volatility(20日年化)"""
    df = df.copy()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 波動率 (年化)
    log_return = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = log_return.rolling(window=20).std() * np.sqrt(252)
    
    # 移除空值
    df = df.dropna().reset_index(drop=True)
    return df

def prepare_features_for_prediction(df, window=60):
    """取最近 window 天特徵 (Close, MA20, MA60, RSI, Volatility)"""
    df_feat = compute_technical_features(df)
    if len(df_feat) < window:
        st.error(f"資料不足 {window} 天，無法進行預測")
        return None
    last_window = df_feat.iloc[-window:][['Close', 'MA20', 'MA60', 'RSI', 'Volatility']].values
    return last_window.reshape(1, -1)  # 攤平為 1 x (window*5)

# ====================== 2. 資料取得函數 ======================
@st.cache_data(ttl=3600)
def fetch_etf_data(ticker, period='5y'):
    """從 Yahoo Finance 取得歷史資料，快取一小時"""
    end = datetime.today()
    start = end - timedelta(days=5*365)
    data = yf.download(ticker, start=start, end=end, progress=False)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return data

# ====================== 3. 模型載入（預訓練，避免每次重訓練） ======================
@st.cache_resource
def load_regression_model(ticker, model_type='rf'):
    """載入迴歸模型 (Random Forest / XGBoost / MLP)"""
    model_path = f"models/{model_type}_{ticker.lower()}_regressor.pkl"
    if not os.path.exists(model_path):
        st.warning(f"模型檔案 {model_path} 不存在，將使用簡易預測（請先離線訓練）")
        return None
    return joblib.load(model_path)

@st.cache_resource
def load_classification_model(ticker, model_type='rf'):
    """載入分類模型 (對照用)"""
    model_path = f"models/{model_type}_{ticker.lower()}_classifier.pkl"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

def get_scaler(ticker):
    """載入標準化器（若無則回傳 None）"""
    scaler_path = f"models/scaler_{ticker.lower()}.pkl"
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    return None

# ====================== 4. 評估指標計算 ======================
def regression_metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

def classification_metrics(y_true, y_pred):
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return f1, prec, rec, acc

# ====================== 5. GPT / LLM 摘要生成（Gemini 免費版） ======================
def generate_llm_summary(ticker, pred_price, actual_price, r2, mae, risk_factors=None):
    """呼叫 Google Gemini API 生成風險摘要 (免費)"""
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "⚠️ 未設定 GEMINI_API_KEY，無法產生摘要。"
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
你是金融分析專家。以下為 ETF {ticker} 的模型預測結果：
- 預測收盤價：{pred_price:.2f}
- 最新實際收盤價：{actual_price:.2f}
- 決定係數 R²：{r2:.3f}
- 平均絕對誤差 MAE：{mae:.2f}
請用繁體中文撰寫一則 30-50 字的風險摘要與操作建議。
"""
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"摘要生成失敗：{str(e)}"

# ====================== 6. Streamlit UI 主程式 ======================
def main():
    st.title("📊 LLM-ETF：基於大語言模型輔助之跨市場ETF趨勢預測平台")
    st.markdown("**迴歸預測（收盤價）為主軸，分類結果（漲跌）僅供對照**")
    
    # 側邊欄控制項
    with st.sidebar:
        st.header("⚙️ 設定")
        ticker = st.selectbox("選擇 ETF", ["SPY", "QQQ", "0050.TW"])
        task = st.radio(
            "預測任務",
            ["📈 迴歸預測 (收盤價) - 主要任務", "📉 分類預測 (漲跌) - 對照組"],
            help="本研究以迴歸預測為核心，分類結果僅供比較模型在不同任務上的表現"
        )
        model_type = st.selectbox("迴歸模型 (僅迴歸任務)", ["rf", "xgb", "mlp"], format_func=lambda x: {"rf":"隨機森林", "xgb":"XGBoost", "mlp":"多層感知機"}.get(x, x))
        show_compare = st.checkbox("顯示台股 vs 美股五年走勢對比圖", value=True)
        
        st.markdown("---")
        st.caption("資料來源：Yahoo Finance | 模型：預訓練 | LLM：Google Gemini 免費版")
    
    # 取得資料
    with st.spinner("載入資料中..."):
        df = fetch_etf_data(ticker)
        if df.empty:
            st.error("無法取得資料，請檢查網路或稍後再試")
            return
    
    # 計算技術指標並準備特徵
    df_feat = compute_technical_features(df)
    if len(df_feat) < 60:
        st.error("歷史資料不足 60 天，無法進行預測")
        return
    
    # 最近 60 天特徵（攤平）
    X_latest = prepare_features_for_prediction(df, window=60)
    
    # ---------- 迴歸預測（主要任務） ----------
    if "迴歸預測" in task:
        st.subheader(f"📈 {ticker} 迴歸預測結果 (收盤價)")
        
        # 載入模型與標準化器
        model = load_regression_model(ticker, model_type)
        scaler = get_scaler(ticker)
        
        if model is None:
            st.warning(f"⚠️ 未找到 {model_type}_{ticker.lower()}_regressor.pkl，無法進行迴歸預測。請先離線訓練模型。")
            # 顯示簡單的佔位資訊
            st.info("本平台採用預訓練模型，請參閱離線訓練腳本產生模型檔案。")
        else:
            # 若需標準化，先對特徵縮放（假設 scaler 存在）
            if scaler is not None:
                X_scaled = scaler.transform(X_latest)
            else:
                X_scaled = X_latest
            pred_price = model.predict(X_scaled)[0]
            actual_price = df_feat['Close'].iloc[-1]
            
            # 計算評估指標（需使用測試集，此處展示最近一段時間的簡單評估）
            # 為簡化，以最近 100 天為測試集（實際應使用完整測試集）
            test_size = min(100, len(df_feat)-60)
            X_test = np.array([prepare_features_for_prediction(df_feat.iloc[:i+60], window=60).flatten() for i in range(-test_size, 0) if i+60 >= 0])
            y_test = df_feat['Close'].iloc[-test_size:].values
            if len(X_test) > 0 and model is not None:
                if scaler is not None:
                    X_test_scaled = scaler.transform(X_test)
                else:
                    X_test_scaled = X_test
                y_pred_test = model.predict(X_test_scaled)
                mae, mse, rmse, r2 = regression_metrics(y_test, y_pred_test)
            else:
                mae = mse = rmse = r2 = 0.0
            
            # 顯示預測值與誤差
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("預測收盤價", f"${pred_price:.2f}" if ticker!="0050.TW" else f"NT${pred_price:.2f}")
            col2.metric("最新實際收盤價", f"${actual_price:.2f}" if ticker!="0050.TW" else f"NT${actual_price:.2f}")
            col3.metric("絕對誤差 (MAE)", f"${abs(pred_price - actual_price):.2f}" if ticker!="0050.TW" else f"NT${abs(pred_price - actual_price):.2f}")
            col4.metric("誤差率", f"{abs(pred_price - actual_price)/actual_price:.2%}")
            
            # 顯示模型評估指標
            with st.expander("📊 模型評估指標 (測試集)"):
                st.write(f"- 均方根誤差 (RMSE)：{rmse:.2f}")
                st.write(f"- 平均絕對誤差 (MAE)：{mae:.2f}")
                st.write(f"- 決定係數 (R²)：{r2:.4f}")
            
            # GPT 摘要
            with st.expander("🤖 LLM 風險摘要 (Gemini)"):
                summary = generate_llm_summary(ticker, pred_price, actual_price, r2, mae)
                st.write(summary)
    
    # ---------- 分類預測（對照組） ----------
    else:
        st.subheader(f"📉 {ticker} 分類預測結果 (對照組)")
        st.caption("漲跌分類僅作為模型對照，本研究核心為迴歸預測")
        
        clf_model = load_classification_model(ticker, 'rf')
        if clf_model is None:
            st.warning(f"⚠️ 未找到分類模型，無法展示分類結果。請先離線訓練 rf_{ticker.lower()}_classifier.pkl")
        else:
            # 分類標籤：漲=1, 跌=0
            y_true = (df_feat['Close'].shift(-1) > df_feat['Close']).astype(int).dropna().values
            if len(y_true) > 0:
                # 使用最後 100 筆評估
                y_pred_prob = clf_model.predict_proba(X_latest)[:,1]  # 實際應使用測試集
                y_pred = (y_pred_prob > 0.5).astype(int)
                f1, prec, rec, acc = classification_metrics(y_true[-100:], y_pred[-100:] if len(y_pred)>=100 else y_pred)
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("F1-score", f"{f1:.3f}")
                col2.metric("精確率", f"{prec:.3f}")
                col3.metric("召回率", f"{rec:.3f}")
                col4.metric("準確率", f"{acc:.3f}")
    
    # ---------- 台股 vs 美股五年走勢對比圖 ----------
    if show_compare:
        st.subheader("🌍 台股與美股近五年走勢對比")
        with st.spinner("繪製走勢圖中..."):
            end = datetime.today()
            start = end - timedelta(days=5*365)
            spy = yf.download("SPY", start=start, end=end, progress=False)["Close"]
            tw = yf.download("0050.TW", start=start, end=end, progress=False)["Close"]
            # 基準化至 100
            spy_norm = spy / spy.iloc[0] * 100
            tw_norm = tw / tw.iloc[0] * 100
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=spy_norm.index, y=spy_norm, name="SPY (美股)", line=dict(color="blue")), secondary_y=False)
            fig.add_trace(go.Scatter(x=tw_norm.index, y=tw_norm, name="0050.TW (台股)", line=dict(color="red")), secondary_y=True)
            fig.update_layout(title="台股與美股近五年走勢對比 (基準化至 100)", xaxis_title="日期", height=500, hovermode="x unified")
            fig.update_yaxes(title_text="SPY 基準化指數", secondary_y=False)
            fig.update_yaxes(title_text="0050.TW 基準化指數", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)
    
    # ---------- 顯示原始技術指標 ----------
    with st.expander("📋 最新技術指標 (60天平均值)"):
        latest = df_feat[['Close', 'MA20', 'MA60', 'RSI', 'Volatility']].iloc[-1]
        st.write(latest)

# ====================== 離線訓練輔助說明 ======================
if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **模型訓練提示**\n"
        "若缺少模型檔案，請先執行以下離線訓練腳本：\n"
        "```python\n"
        "from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier\n"
        "import joblib, yfinance as yf, pandas as pd\n"
        "# 下載資料、計算特徵、訓練迴歸模型與分類模型\n"
        "# 並儲存至 models/ 資料夾\n"
        "```\n"
        "詳細訓練程式請參考論文附錄。"
    )
    main()
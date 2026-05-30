# app_main.py
# LLM-ETF 互动式预测平台 (回归为主，分类为辅)
# 研究生：林子瑜
# 指导教授：李冠荣 副教授

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="LLM-ETF 预测平台", layout="wide", page_icon="📈")

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ====================== 1. 特徵工程 ======================
def compute_technical_features(df):
    df = df.copy()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    log_ret = np.log(df['Close'] / df['Close'].shift(1))
    df['Volatility'] = log_ret.rolling(20).std() * np.sqrt(252)
    return df.dropna().reset_index(drop=True)

def prepare_features_for_prediction(df, window=60, flatten=True):
    """
    准备预测用的特征矩阵
    flatten=True  -> 展平为 (1, window*5) 用于随机森林等
    flatten=False -> 保留 (1, window, 5) 用于 LSTM
    """
    df_feat = compute_technical_features(df)
    if len(df_feat) < window:
        return None
    last_window = df_feat.iloc[-window:][['Close', 'MA20', 'MA60', 'RSI', 'Volatility']].values
    if flatten:
        return last_window.reshape(1, -1)   # (1,300)
    else:
        return last_window.reshape(1, window, 5)

# ====================== 2. 资料取得 ======================
@st.cache_data(ttl=3600)
def fetch_etf_data(ticker, period_years=5):
    end = datetime.today()
    start = end - timedelta(days=period_years*365)
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return data

# ====================== 3. 模型载入 ======================
@st.cache_resource
def load_regression_model(ticker, model_type='rf'):
    safe_ticker = ticker.lower().replace(".", "_")
    path = f"models/{model_type}_{safe_ticker}_regressor.pkl"
    if not os.path.exists(path):
        return None
    obj = joblib.load(path)
    if hasattr(obj, 'predict'):
        return obj
    if isinstance(obj, dict) and 'model' in obj and hasattr(obj['model'], 'predict'):
        return obj['model']
    return None

@st.cache_resource
def load_classification_model(ticker, model_type='rf'):
    safe_ticker = ticker.lower().replace(".", "_")
    original_ticker = ticker.replace(".", "_")
    candidates = [
        f"models/{model_type}_{original_ticker}.joblib",
        f"models/{model_type}_{safe_ticker}.joblib",
        f"models/{model_type}_{safe_ticker}_classifier.pkl"
    ]
    for path in candidates:
        if os.path.exists(path):
            obj = joblib.load(path)
            if isinstance(obj, dict) and 'model' in obj:
                model = obj['model']
                if hasattr(model, 'predict_proba'):
                    return model
            elif hasattr(obj, 'predict_proba'):
                return obj
    return None

def get_scaler(ticker):
    safe_ticker = ticker.lower().replace(".", "_")
    path = f"models/scaler_{safe_ticker}.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return None

# ====================== 4. 评估指标 ======================
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

# ====================== 5. Gemini 摘要 ======================
def generate_llm_summary(ticker, pred_price, actual_price, r2, mae):
    try:
        import google.generativeai as genai
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "⚠️ 未设定 API Key"
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"ETF {ticker} 预测收盘价 {pred_price:.2f}，实际 {actual_price:.2f}，R²={r2:.3f}，MAE={mae:.2f}。请用繁体中文给 30 字风险摘要。"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"摘要失败：{e}"

# ====================== 6. 主程式 ======================
def main():
    st.title("📊 LLM-ETF：基於大語言模型輔助之跨市場ETF趨勢預測平台")
    st.markdown("**迴歸預測（收盤價）為主軸，分類結果（漲跌）僅供對照**")
    
    with st.sidebar:
        st.header("⚙️ 設定")
        ticker = st.selectbox("選擇 ETF", ["SPY", "QQQ", "0050.TW"])
        task = st.radio("預測任務", ["📈 迴歸預測 (收盤價) - 主要任務", "📉 分類預測 (漲跌) - 對照組"])
        st.caption("資料：Yahoo Finance | 模型：預訓練 | LLM：Gemini 免費版")
    
    # 取得资料
    df = fetch_etf_data(ticker, period_years=5)
    if df.empty:
        st.error("無法取得資料")
        return
    df_feat = compute_technical_features(df)
    if len(df_feat) < 60:
        st.error("少於 60 天")
        return
    
    # ---------- 迴歸任務 ----------
    if "迴歸預測" in task:
        st.subheader(f"📈 {ticker} 迴歸預測")
        model = load_regression_model(ticker, 'rf')
        scaler = get_scaler(ticker)
        if model is None:
            st.warning("迴歸模型不存在，請確認 models/ 內有對應的 .pkl 檔案")
            return
        
        X_latest = prepare_features_for_prediction(df, 60, flatten=True)
        if X_latest is None:
            st.error("特徵不足")
            return
        
        # 標準化（如果有 scaler）
        if scaler:
            try:
                X_scaled = scaler.transform(X_latest)
            except Exception as e:
                st.error(f"標準化失敗：{e}。請檢查 scaler 是否與特徵維度匹配。")
                return
        else:
            X_scaled = X_latest
        
        pred_price = float(model.predict(X_scaled)[0])
        actual_price = float(df_feat['Close'].iloc[-1])
        
        # 簡易測試集評估
        test_size = min(100, len(df_feat)-60)
        X_test, y_test = [], []
        for offset in range(1, test_size+1):
            idx = len(df_feat) - offset
            if idx >= 60:
                feat = prepare_features_for_prediction(df_feat.iloc[:idx], 60, flatten=True)
                if feat is not None:
                    X_test.append(feat.flatten())
                    y_test.append(float(df_feat.iloc[idx]['Close']))
        if X_test:
            Xt = np.array(X_test)
            yt = np.array(y_test)
            if scaler:
                Xt = scaler.transform(Xt)
            y_pred_test = model.predict(Xt)
            mae, mse, rmse, r2 = regression_metrics(yt, y_pred_test)
        else:
            mae = mse = rmse = r2 = 0.0
        
        col1, col2, col3, col4 = st.columns(4)
        cur = "NT$" if ticker=="0050.TW" else "$"
        col1.metric("預測收盤價", f"{cur}{pred_price:.2f}")
        col2.metric("最新實際", f"{cur}{actual_price:.2f}")
        col3.metric("絕對誤差", f"{cur}{abs(pred_price-actual_price):.2f}")
        col4.metric("誤差率", f"{abs(pred_price-actual_price)/actual_price:.2%}")
        
        with st.expander("📊 模型評估指標 (測試集)"):
            st.write(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        with st.expander("🤖 LLM 風險摘要 (Gemini)"):
            st.write(generate_llm_summary(ticker, pred_price, actual_price, r2, mae))
    
    # ---------- 分類任務（對照組） ----------
    else:
        st.subheader(f"📉 {ticker} 分類預測 (對照組)")
        clf = load_classification_model(ticker, 'rf')
        if clf is None:
            st.warning("分類模型不存在或無法讀取（請檢查 models/ 內的 .joblib 檔案）")
            return
        
        X_latest = prepare_features_for_prediction(df, 60, flatten=True)   # 預設展平
        if X_latest is None:
            st.error("特徵不足")
            return
        
        # 嘗試預測，若發生特徵數錯誤則提示使用者
        try:
            # 若模型有 n_features_in_ 屬性，檢查是否與 X_latest 維度一致
            if hasattr(clf, 'n_features_in_'):
                expected_n = clf.n_features_in_
                actual_n = X_latest.shape[1]
                if expected_n != actual_n:
                    st.error(f"模型期望 {expected_n} 個特徵，但輸入為 {actual_n} 個。可能原因是訓練時未展平？請檢查模型訓練方式。")
                    return
            prob = clf.predict_proba(X_latest)[0][1]
            st.metric("上漲機率", f"{prob:.2%}")
            st.write("**預測結果：上漲**" if prob>0.5 else "**預測結果：下跌**")
        except Exception as e:
            st.error(f"分類預測失敗：{e}\n可能是特徵維度不一致。建議改用迴歸任務為主要分析。")
            st.stop()
        
        # 測試集評估
        test_size = min(100, len(df_feat)-60)
        y_true, y_pred = [], []
        for offset in range(1, test_size+1):
            idx = len(df_feat) - offset
            if idx >= 60:
                feat = prepare_features_for_prediction(df_feat.iloc[:idx], 60, flatten=True)
                if feat is not None:
                    true_lbl = 1 if df_feat.iloc[idx]['Close'] > df_feat.iloc[idx-1]['Close'] else 0
                    try:
                        prob_val = clf.predict_proba(feat)[0][1]
                        pred_lbl = 1 if prob_val>0.5 else 0
                        y_true.append(true_lbl)
                        y_pred.append(pred_lbl)
                    except:
                        pass
        if y_true:
            f1, prec, rec, acc = classification_metrics(y_true, y_pred)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("F1-score", f"{f1:.3f}")
            col2.metric("精確率", f"{prec:.3f}")
            col3.metric("召回率", f"{rec:.3f}")
            col4.metric("準確率", f"{acc:.3f}")
        else:
            st.info("測試集樣本不足，無法計算分類指標")
    
    # 顯示最新技術指標（簡化）
    with st.expander("📋 最新技術指標"):
        st.write(df_feat[['Close','MA20','MA60','RSI','Volatility']].iloc[-1])

if __name__ == "__main__":
    main()
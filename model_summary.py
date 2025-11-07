# --- 1. 定義模型超參數 ---
# 這個區塊是新增的
TIMESTEPS = 30     # 設定滑動窗口大小為 30 天
N_FEATURES = 4     # 假設我們有 4 個特徵: Return, Volume_Change, Intraday_Range_Pct, RSI
print(f"模型設定：一次觀察 {TIMESTEPS} 天的數據，每天有 {N_FEATURES} 個特徵。")
# --- 2. LSTM Autoencoder 模型架構 ---
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
inputs = Input(shape=(TIMESTEPS, N_FEATURES))
# 編碼器 (Encoder)
encoder = LSTM(64, activation='relu')(inputs)
# 解碼器 (Decoder)
decoder = RepeatVector(TIMESTEPS)(encoder)
decoder = LSTM(64, activation='relu', return_sequences=True)(decoder)
output = TimeDistributed(Dense(N_FEATURES))(decoder)
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer='adam', loss='mae')

# 顯示模型架構
model.summary()
import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 載入 .env 檔案中的環境變數
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 2. 檢查金鑰是否存在
if not OPENAI_API_KEY:
    print("❌ .env 中沒有找到 OPENAI_API_KEY，請檢查檔案內容。")
    exit(1)

# 3. 嘗試用這個金鑰建立 OpenAI 客戶端並發送一個簡單請求
print("✅ 找到 OPENAI_API_KEY，正在測試連線...")
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", # 使用一個較便宜且快速的模型來測試
        messages=[{"role": "user", "content": "說 'API 測試成功'"}],
        max_tokens=20
    )
    print(f"✅ API 測試成功，回應內容：{response.choices[0].message.content}")
except Exception as e:
    print(f"❌ API 連線失敗：{e}")
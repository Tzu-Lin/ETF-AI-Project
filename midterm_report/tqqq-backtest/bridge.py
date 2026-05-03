"""
bridge.py — 財經數位分身 Agent 中樞腳本 (HTTP 服務版)
功能：
  1. 提供 /execute 端點，接收 n8n 的 JSON 指令
  2. 唯讀載入 AEGIS 數據
  3. 透過 tsx 執行 TQQQ 選擇權回測工具
  4. 呼叫遠端 Ollama (Hermes-2-Pro) 進行策略分析
  5. 將 AI 結論萃取為技能 Markdown，存入 ./memories/tqqq_defense.md
  6. 回傳結構化結果給 n8n
使用方式：
  在 hermes-lab 環境下執行：
  uvicorn bridge:app --host 0.0.0.0 --port 5000
  確保環境變數 OLLAMA_BASE_URL 已設為哥哥的 Tailscale IP
"""
import os
import json
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------- 設定區 ----------------------------
# 遠端 Ollama API（從環境變數讀取，方便切換）
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://100.64.0.1:11434")
OLLAMA_MODEL = "hermes-2-pro:latest"

# 回測工具 CLI 檔案位置 (相對於本檔案或絕對路徑)
# 注意：目前 bridge.py 位於 tqqq-backtest 目錄內，所以 runBacktestCli.ts 就在同一層
BACKTEST_TS_FILE = Path("runBacktestCli.ts")  # 如果放在上一層請改成 ../tqqq-backtest/runBacktestCli.ts

# AEGIS 數據目錄 (唯讀)
AEGIS_DATA_DIR = Path("../AEGIS/data")

# 技能記憶固定輸出檔案
MEMORIES_DIR = Path("./memories")
MEMORIES_DIR.mkdir(exist_ok=True)
SKILL_FILE = MEMORIES_DIR / "tqqq_defense.md"

# Logging
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("bridge")

# FastAPI 實例
app = FastAPI(title="TQQQ Digital Twin Bridge", version="1.0.0")

# ---------------------------- Pydantic 請求模型 ----------------------------
class N8nRequest(BaseModel):
    backtest_params: Dict[str, Any] = {}   # 例如 {"putDelta": 0.3, "cashReservePercentage": 0.2}
    user_message: str = ""                 # 使用者額外訊息

# ---------------------------- AEGIS 數據讀取 (唯讀) ----------------------------
def load_aegis_data() -> Dict[str, pd.DataFrame]:
    """從 AEGIS data 目錄唯讀載入 CSV 數據，回傳字典"""
    data = {}
    if not AEGIS_DATA_DIR.exists():
        logger.warning(f"AEGIS 數據目錄不存在: {AEGIS_DATA_DIR}")
        return data
    for file in AEGIS_DATA_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(file, parse_dates=True, index_col=0)
            name = file.stem
            data[name] = df
            logger.info(f"讀取 AEGIS 數據: {file} (shape={df.shape})")
        except Exception as e:
            logger.error(f"讀取 {file} 失敗: {e}")
    return data

def build_market_snapshot(data: Dict[str, pd.DataFrame]) -> str:
    """從 AEGIS 數據中擷取最新數值，組成市場快照文字"""
    snapshot_lines = []
    if "etf_tqqq" in data:
        try:
            latest = data["etf_tqqq"].iloc[-1]
            snapshot_lines.append(f"TQQQ 最新資料: {latest.to_dict()}")
        except Exception:
            pass
    if "carbon" in data:
        try:
            latest = data["carbon"].iloc[-1]
            snapshot_lines.append(f"碳權指數: {latest.to_dict()}")
        except Exception:
            pass
    return "\n".join(snapshot_lines) if snapshot_lines else "無即時市場數據"

# ---------------------------- 回測工具執行 ----------------------------
def run_tqqq_backtest(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用 tsx 執行回測 CLI，透過 STDIN 傳遞 JSON 參數，避免命令列轉義問題。
    """
    if not BACKTEST_TS_FILE.exists():
        raise FileNotFoundError(f"找不到回測腳本: {BACKTEST_TS_FILE}")

    params_json = json.dumps(params)  # 緊湊 JSON，無換行

    # 使用 npx.cmd 並以 shell=True 確保找到 npm 全域工具
    cmd = f'npx.cmd tsx "{BACKTEST_TS_FILE}"'
    logger.info(f"執行回測 (stdin): {cmd}")

    try:
        # 用 Popen 以便寫入 stdin
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(BACKTEST_TS_FILE.parent),
            shell=True
        )
        stdout_data, stderr_data = proc.communicate(input=params_json, timeout=120)

        if proc.returncode != 0:
            logger.error(f"回測工具錯誤: {stderr_data}")
            raise RuntimeError(f"回測失敗: {stderr_data}")

        if not stdout_data.strip():
            logger.error("回測工具無輸出")
            raise RuntimeError("回測工具無輸出")

        result = json.loads(stdout_data)
        logger.info("回測執行成功")
        return result

    except subprocess.TimeoutExpired:
        raise RuntimeError("回測執行逾時 (120 秒)")
    except json.JSONDecodeError as e:
        logger.error(f"回測輸出非 JSON: {stdout_data}")
        raise RuntimeError(f"回測輸出格式錯誤: {e}")

# ---------------------------- 遠端 Ollama 呼叫 (含 Mock 切換) ----------------------------
USE_MOCK_HERMES = os.getenv("USE_MOCK_HERMES", "false").lower() == "true"

def ask_hermes(prompt: str) -> str:
    """發送生成請求到遠端 Ollama，或使用模擬回應（開發測試用）"""

    # Mock 模式：直接回傳假的分析結果，不連網路
    if USE_MOCK_HERMES:
        logger.info("⚡ 使用 Mock Hermes 回應 (未連接遠端 AI)")
        return """## 策略評估與建議（模擬）
目前 TQQQ 隱含波動率偏高，根據回測結果，最大回撤已達 12.3%，建議立即提高現金儲備比例至 25%，並將 putDelta 調降至 0.2，以降低下檔風險。

```skill
# 高波動防守技能
- **技能名稱**：高波動 Wheel 防守模組
- **觸發條件**：當 VIX > 30 或 最大回撤 > 10%
- **建議行動**：
  - 降低 putDelta 至 0.20
  - 提高 cashReservePercentage 至 25%
  - 暫停開倉直到 IV 回落至 25 以下
- **參考數據**：TQQQ IV 45%，歷史最大回撤 15%
"""

    # 真實連線模式
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 2048}
    }
    logger.info(f"發送請求至 Hermes: {prompt[:80]}...")
    try:
        resp = requests.post(url, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("response", "")
        logger.info(f"Hermes 回覆 (前 100 字): {answer[:100]}...")
        return answer
    except requests.RequestException as e:
        logger.error(f"呼叫 Ollama 失敗: {e}")
        raise RuntimeError(f"遠端 AI 服務不可用: {e}")

# ---------------------------- 技能儲存 ----------------------------
def save_skill_to_markdown(analysis_text: str, meta: Dict[str, Any]) -> Path:
    """
    將 AI 分析結論儲存為固定技能檔 tqqq_defense.md。
    內容包含時間戳、回測指標、及 AI 分析全文。
    """
    # 建立 Markdown 內容
    content = f"""# TQQQ 防禦策略技能 (自動生成)

**生成時間:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**相關參數:** {json.dumps(meta.get("parameters", {}), indent=2)}
**最大回撤:** {meta.get("maxDrawdown", "N/A")}
**勝率:** {meta.get("winRate", "N/A")}
**IRR:** {meta.get("irr", "N/A")}

---

## AI 分析報告

{analysis_text}
"""
    # 寫入固定檔案 (每次覆蓋，保持最新技能)
    with open(SKILL_FILE, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"技能已更新至 {SKILL_FILE}")
    return SKILL_FILE

# ---------------------------- 核心執行邏輯 ----------------------------
def execute_strategy_cycle(params: Dict[str, Any], user_msg: str) -> Dict[str, Any]:
    """
    完整流程：
    1. 讀取 AEGIS 數據
    2. 執行回測
    3. 發送給 Hermes 分析
    4. 儲存技能
    5. 回傳整合結果
    """
    # 1. 唯讀數據
    aegis_data = load_aegis_data()
    market_snapshot = build_market_snapshot(aegis_data)

    # 2. 回測
    backtest_result = run_tqqq_backtest(params)
    # 提取關鍵指標（根據你的回測輸出鍵值調整，這裡給出安全取值）
    max_dd = backtest_result.get("maxDrawdown", "N/A")
    win_rate = backtest_result.get("winRate", "N/A")
    irr = backtest_result.get("irr", "N/A")

    # 3. 組合 prompt
    prompt = f"""
你是一位專業的財經分析師，負責管理一個 TQQQ 選擇權 Wheel 策略的數位分身。

目前市場數據摘要：
{market_snapshot}

剛剛執行的 TQQQ 回測結果：
- 策略參數：{json.dumps(params, ensure_ascii=False)}
- 最大回撤 (Max Drawdown): {max_dd}
- 勝率 (Win Rate): {win_rate}
- 內部報酬率 (IRR): {irr}
- 完整回測結果：{json.dumps(backtest_result, ensure_ascii=False)}

使用者額外訊息：{user_msg}

請根據以上資訊提供：
1. 當前策略的風險評估與改善建議。
2. 針對目前市場環境，動態調整 putDelta 與現金儲備比例的具體建議。
3. 最後請將你的分析濃縮成一個「技能」，用專屬標記 ```skill 包圍，技能內容需包含：
   - 技能名稱
   - 觸發條件
   - 建議行動 (具體參數調整或避險動作)
   - 參考數據

請全程使用繁體中文回答。
"""
    # 4. 呼叫遠端 Hermes (或 Mock)
    ai_analysis = ask_hermes(prompt)

    # 5. 儲存技能 (固定檔案)
    meta = {
        "parameters": params,
        "maxDrawdown": max_dd,
        "winRate": win_rate,
        "irr": irr
    }
    skill_path = save_skill_to_markdown(ai_analysis, meta)

    return {
        "status": "ok",
        "backtest_result": backtest_result,
        "ai_analysis": ai_analysis,
        "skill_file": str(skill_path)
    }

# ---------------------------- API 端點 (供 n8n 呼叫) ----------------------------
@app.post("/execute")
async def handle_n8n_request(request: N8nRequest):
    logger.info("接收到 n8n 請求")
    try:
        result = execute_strategy_cycle(
            params=request.backtest_params or {"putDelta": 0.3, "cashReservePercentage": 0.2},
            user_msg=request.user_message or ""
        )
        return result
    except Exception as e:
        logger.exception("執行週期發生錯誤")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
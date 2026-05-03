import subprocess
import json

# 1. 準備乾淨的 Python 字典 (這就是未來 n8n 會傳給你的資料)
params = {
    "startDate": "2021-01-01",
    "endDate": "2024-01-01",
    "initialCapital": 3000000,
    "putDTE": 42,
    "putDelta": 0.3,
    "callDTE": 42,
    "callDelta": 0.2,
    "sellCallAboveCostBasisOnly": True,
    "enableCashTactic": True,
    "cashReservePercentage": 30,
    "dipTriggers": [{"drop": 70, "use": 10}],
    "enableRecurringInvestment": False,
    "exchangeRate": 31,
    "optionFee": 0.5,
    "slippage": 2,
    "ivAdjustmentFactor": 1
}

# 2. 自動將 Python 字典轉成合法的 JSON 字串
params_json = json.dumps(params)

# 3. 呼叫我們剛寫好的 TypeScript CLI
cmd = ["npx", "tsx", "runBacktestCli.ts", "--params", params_json]

print("正在執行回測，請稍候...")

# 執行指令並擷取輸出 (shell=True 可以避免 Windows 路徑問題)
result = subprocess.run(cmd, capture_output=True, text=True, shell=True)

# 4. 印出結果
print("\n=== 回測成功！擷取到的 JSON 如下 ===")
print(result.stdout)

if result.stderr:
    print("\n=== 錯誤訊息 ===")
    print(result.stderr)
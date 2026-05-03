// services/geminiService.ts (已修改為呼叫遠端 Ollama)

import { StrategyParameters } from "../types";

export const getStrategyAnalysis = async (params: StrategyParameters): Promise<string> => {
    // 1. 定義 Ollama 的連線資訊
    // 注意：這裡預設使用環境變數，如果沒設定，就退回 localhost。
    // 未來你在 .env 中設定 OLLAMA_BASE_URL 即可指向哥哥的 Tailscale IP
    const ollamaBaseUrl = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
    const model = 'hermes-2-pro'; // 替換為哥哥電腦上安裝的模型名稱

    // 2. 組合提示詞 (這部分保留你哥原本的心血，因為這段 Prompt 寫得很好)
    const recurringInvestmentDetails = params.enableRecurringInvestment
        ? `
            - 每月定期定額：${params.monthlyInvestmentTWD.toLocaleString()} 台幣
            - 假設匯率：1:${params.exchangeRate}，並在資金足夠時買入 100 股 TQQQ 進行備兌
        ` : ``;

    const cashTacticDetails = params.enableCashTactic 
        ? `使用現金儲備戰術：保留 ${params.cashReservePercentage}% 的總資產作為現金。
         - 平時投資比例：${100 - params.cashReservePercentage}%
         - 抄底觸發條件：
             - 條件 1：跌幅 ${params.dipTriggers[0].drop}%，動用儲備 ${params.dipTriggers[0].use}%
             - 條件 2：跌幅 ${params.dipTriggers[1].drop}%，動用儲備 ${params.dipTriggers[1].use}%
             - 條件 3：跌幅 ${params.dipTriggers[2].drop}%，動用儲備 ${params.dipTriggers[2].use}%
        ` : `純策略，無現金儲備（100% 資金投入）。`;

    const prompt = `
        你是一位專業的量化金融分析師，精通選擇權交易與美股 ETF 投資。
        請根據以下回測設定，提供一份針對 TQQQ 的策略分析與建議，請使用 Markdown 格式輸出。

        本次分析涵蓋四種策略的對比：
        1.  **純選擇權輪轉 (Pure Options Wheel)**: 100% 資金執行 Sell Put 與 Sell Call。
        2.  **純持有 (Pure Buy & Hold)**: 100% 資金買入並持有 TQQQ。
        3.  **選擇權輪轉 + 現金戰術 (Options Wheel + Cash Reserve)**: ${100 - params.cashReservePercentage}% 資金做選擇權，${params.cashReservePercentage}% 保留現金等待 TQQQ 暴跌抄底。
        4.  **持有 + 現金戰術 (Buy & Hold + Cash Reserve)**: ${100 - params.cashReservePercentage}% 資金買入並持有，${params.cashReservePercentage}% 保留現金等待 TQQQ 暴跌抄底。

        **【回測參數設定】**
        - 期間：${params.startDate} 至 ${params.endDate}
        - 初始資金：${params.initialCapital.toLocaleString()} TWD (匯率 1:${params.exchangeRate} 計算)
        - 選擇權天期：DTE ${params.putDTE} (Put), ${params.callDTE} (Call)
        - Put Delta 目標：${params.putDelta}
        - Call Delta 目標：${params.callDelta}
        
        ${recurringInvestmentDetails}
        ${cashTacticDetails}
        
        **【分析重點要求】**
        - 評估在當前設定下，100% 資金輪轉與加入現金戰術的風險報酬差異。
        - 針對 TQQQ 的高波動特性，這樣的 Delta 設定是否合理？有什麼潛在風險？
        - 如果面臨大熊市（例如 2022 年），Covered Call 的防禦力是否足夠？
        - 給予操作上的具體改善建議。

        **請直接給出分析內容，不需重複我的參數設定。**
    `;

    // 3. 發送請求到 Ollama
    try {
        const response = await fetch(`${ollamaBaseUrl}/api/generate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: model,
                prompt: prompt,
                stream: false // 設定為 false，讓它一次把完整文字吐回來
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data.response; // Ollama 的回傳結構中，回覆文字放在 response 欄位

    } catch (error) {
        console.error("Error getting strategy analysis from Ollama:", error);
        return "無法取得 AI 分析結果，請確認 Ollama 伺服器是否正常連線，且模型已啟動。";
    }
};
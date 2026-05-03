// runBacktestCli.ts
// 從命令列參數 --params 或標準輸入讀取 JSON 參數，執行回測並輸出 JSON 結果

import { runBacktest } from './services/backtestService';

async function main() {
  try {
    let params: any;

    // 先檢查命令列參數
    const args = process.argv.slice(2);
    const paramsIndex = args.indexOf('--params');
    if (paramsIndex !== -1 && paramsIndex + 1 < args.length) {
      params = JSON.parse(args[paramsIndex + 1]);
    } else {
      // 如果沒有 --params，則從標準輸入讀取
      let inputData = '';
      process.stdin.setEncoding('utf-8');
      for await (const chunk of process.stdin) {
        inputData += chunk;
      }
      if (!inputData.trim()) {
        throw new Error('無提供參數：請使用 --params 或透過 stdin 傳入 JSON');
      }
      params = JSON.parse(inputData);
    }

    const result = await runBacktest(params);
    process.stdout.write(JSON.stringify(result));
    process.exit(0);
  } catch (error: any) {
    process.stderr.write(JSON.stringify({ error: error.message || error.toString() }));
    process.exit(1);
  }
}

main();
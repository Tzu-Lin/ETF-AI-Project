import React from 'react';
// FIX: Changed import of BacktestResult to come from types.ts instead of backtestService.ts to fix module export error.
import { BacktestResult } from '../types';
import NetWorthChart from './NetWorthChart';
import HoldingsChart from './HoldingsChart';
import TradeLogTable from './TradeLogTable';
import AnnualSummaryTable from './AnnualSummaryTable';
import { marked } from 'marked';


interface ResultsDisplayProps {
  results: BacktestResult | null;
  geminiAnalysis: string;
  isAnalyzing: boolean;
}

const formatTwd = (value: number) => {
    if (isNaN(value)) return 'N/A';
    return new Intl.NumberFormat('zh-TW', {
        style: 'currency',
        currency: 'TWD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
    }).format(value);
};

const formatDate = (date: Date): string => {
    if (!date || isNaN(date.getTime())) return '';
    return date.toLocaleDateString('en-CA'); // YYYY-MM-DD
};

const StatCard: React.FC<{ title: string; value: string; className?: string; children?: React.ReactNode }> = ({ title, value, className, children }) => (
    <div className={`bg-gray-800 p-4 rounded-lg shadow-md flex flex-col justify-between ${className}`}>
        <div>
            <h4 className="text-sm text-gray-400 font-medium">{title}</h4>
            <p className="text-2xl font-bold text-white">{value}</p>
        </div>
        {children && <div className="text-xs text-gray-400 mt-1 space-y-1">{children}</div>}
    </div>
);


const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ results, geminiAnalysis, isAnalyzing }) => {
  if (!results || results.optionsHistory.length === 0) {
    return (
        <div className="mt-6 bg-gray-800 p-6 rounded-lg text-center">
            <p className="text-gray-400">請設定參數並執行回測以查看結果。</p>
        </div>
    );
  }

  const { 
    optionsHistory, buyAndHoldHistory, optionsWithCashHistory, buyAndHoldWithCashHistory,
    tradeLog, optionsWithCashTradeLog, buyAndHoldTradeLog, buyAndHoldWithCashTradeLog,
    annualSummaries, optionsWithCashAnnualSummaries,
    analytics, buyAndHoldAnalytics, optionsWithCashAnalytics, buyAndHoldWithCashAnalytics, 
    priceHistory,
    totalTwdInvested,
    exchangeRate,
  } = results;
  
  const analysisHtml = geminiAnalysis ? marked.parse(geminiAnalysis) as string : '';

  const showCashStrategies = optionsWithCashHistory.length > 0;

  const finalOptionsNetWorth = optionsHistory.length > 0 ? optionsHistory[optionsHistory.length - 1].netWorth : 0;
  const finalBuyAndHoldNetWorth = buyAndHoldHistory.length > 0 ? buyAndHoldHistory[buyAndHoldHistory.length - 1].netWorth : 0;
  const finalOptionsWithCashNetWorth = optionsWithCashHistory.length > 0 ? optionsWithCashHistory[optionsWithCashHistory.length - 1].netWorth : 0;
  const finalBuyAndHoldWithCashNetWorth = buyAndHoldWithCashHistory.length > 0 ? buyAndHoldWithCashHistory[buyAndHoldWithCashHistory.length - 1].netWorth : 0;

  return (
    <div className="mt-6 space-y-6">
      
      {/* --- Overall Stats --- */}
      <div className="space-y-4 bg-gray-900/50 p-4 rounded-lg">
          <h3 className="text-lg font-semibold text-white">回測總覽</h3>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <StatCard title="總投入資金 (TWD)" value={`${formatTwd(totalTwdInvested)}`} />
          </div>
      </div>

      {/* --- Key Metrics --- */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Pure Strategies */}
          <div className="space-y-4 bg-gray-900/50 p-4 rounded-lg">
              <h3 className="text-lg font-semibold text-white">100% 資金策略</h3>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <StatCard title="輪動策略總回報" value={`${analytics.totalReturn.toFixed(2)}%`}>
                      <span>最終資產: {formatTwd(finalOptionsNetWorth * exchangeRate)}</span>
                  </StatCard>
                  <StatCard title="輪動策略 CAGR" value={`${analytics.cagr.toFixed(2)}%`} />
                  <StatCard title="輪動策略最大回撤" value={`${analytics.maxDrawdown.toFixed(2)}%`} >
                      <>
                        <span>{formatTwd(analytics.drawdownPeakUSD * exchangeRate)} → {formatTwd(analytics.drawdownTroughUSD * exchangeRate)}</span>
                        <div className="text-gray-400 mt-1 flex flex-col items-start text-xs">
                          <span>{formatDate(analytics.drawdownPeakDate)}</span>
                          <span className="leading-none mx-2">↓</span>
                          <span>{formatDate(analytics.drawdownTroughDate)}</span>
                        </div>
                      </>
                  </StatCard>
              </div>
               <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4">
                  <StatCard title="買入持有總回報" value={`${buyAndHoldAnalytics.totalReturn.toFixed(2)}%`}>
                      <span>最終資產: {formatTwd(finalBuyAndHoldNetWorth * exchangeRate)}</span>
                  </StatCard>
                  <StatCard title="買入持有 CAGR" value={`${buyAndHoldAnalytics.cagr.toFixed(2)}%`} />
                  <StatCard title="買入持有最大回撤" value={`${buyAndHoldAnalytics.maxDrawdown.toFixed(2)}%`} >
                     <>
                        <span>{formatTwd(buyAndHoldAnalytics.drawdownPeakUSD * exchangeRate)} → {formatTwd(buyAndHoldAnalytics.drawdownTroughUSD * exchangeRate)}</span>
                        <div className="text-gray-400 mt-1 flex flex-col items-start text-xs">
                          <span>{formatDate(buyAndHoldAnalytics.drawdownPeakDate)}</span>
                          <span className="leading-none mx-2">↓</span>
                          <span>{formatDate(buyAndHoldAnalytics.drawdownTroughDate)}</span>
                        </div>
                     </>
                  </StatCard>
              </div>
          </div>

          {/* Strategies with Cash Reserve */}
          {showCashStrategies && (
            <div className="space-y-4 bg-gray-900/50 p-4 rounded-lg">
                <h3 className="text-lg font-semibold text-white">帶現金儲備的策略</h3>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <StatCard title="輪動+現金總回報" value={`${optionsWithCashAnalytics.totalReturn.toFixed(2)}%`}>
                      <span>最終資產: {formatTwd(finalOptionsWithCashNetWorth * exchangeRate)}</span>
                    </StatCard>
                    <StatCard title="輪動+現金 CAGR" value={`${optionsWithCashAnalytics.cagr.toFixed(2)}%`} />
                    <StatCard title="輪動+現金最大回撤" value={`${optionsWithCashAnalytics.maxDrawdown.toFixed(2)}%`}>
                       <>
                        <span>{formatTwd(optionsWithCashAnalytics.drawdownPeakUSD * exchangeRate)} → {formatTwd(optionsWithCashAnalytics.drawdownTroughUSD * exchangeRate)}</span>
                        <div className="text-gray-400 mt-1 flex flex-col items-start text-xs">
                           <span>{formatDate(optionsWithCashAnalytics.drawdownPeakDate)}</span>
                           <span className="leading-none mx-2">↓</span>
                           <span>{formatDate(optionsWithCashAnalytics.drawdownTroughDate)}</span>
                        </div>
                       </>
                    </StatCard>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4">
                    <StatCard title="買入持有+現金總回報" value={`${buyAndHoldWithCashAnalytics.totalReturn.toFixed(2)}%`}>
                       <span>最終資產: {formatTwd(finalBuyAndHoldWithCashNetWorth * exchangeRate)}</span>
                    </StatCard>
                    <StatCard title="買入持有+現金 CAGR" value={`${buyAndHoldWithCashAnalytics.cagr.toFixed(2)}%`} />
                    <StatCard title="買入持有+現金最大回撤" value={`${buyAndHoldWithCashAnalytics.maxDrawdown.toFixed(2)}%`} >
                        <>
                           <span>{formatTwd(buyAndHoldWithCashAnalytics.drawdownPeakUSD * exchangeRate)} → {formatTwd(buyAndHoldWithCashAnalytics.drawdownTroughUSD * exchangeRate)}</span>
                           <div className="text-gray-400 mt-1 flex flex-col items-start text-xs">
                            <span>{formatDate(buyAndHoldWithCashAnalytics.drawdownPeakDate)}</span>
                            <span className="leading-none mx-2">↓</span>
                            <span>{formatDate(buyAndHoldWithCashAnalytics.drawdownTroughDate)}</span>
                           </div>
                        </>
                    </StatCard>
                </div>
            </div>
          )}
      </div>


      {/* --- Charts --- */}
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
        <h3 className="text-lg font-bold text-white mb-4">淨值曲線</h3>
        <NetWorthChart 
            optionsHistory={optionsHistory} 
            buyAndHoldHistory={buyAndHoldHistory}
            optionsWithCashHistory={optionsWithCashHistory}
            buyAndHoldWithCashHistory={buyAndHoldWithCashHistory}
            priceHistory={priceHistory} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
              <h3 className="text-lg font-bold text-white mb-4">持倉結構 (純輪動策略)</h3>
              <HoldingsChart data={optionsHistory} />
          </div>
           <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
                <h3 className="text-lg font-bold text-white mb-4">Gemini 策略分析</h3>
                {isAnalyzing ? (
                    <div className="flex justify-center items-center h-full">
                        <p className="text-gray-400">分析中...</p>
                    </div>
                ) : (
                    <div className="prose prose-invert prose-sm max-h-72 overflow-y-auto" dangerouslySetInnerHTML={{ __html: analysisHtml }} />
                )}
            </div>
      </div>

      {/* --- Tables --- */}
       <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
        <h3 className="text-lg font-bold text-white mb-4">年度績效 (純輪動策略)</h3>
        <AnnualSummaryTable data={annualSummaries} />
      </div>

      <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
        <h3 className="text-lg font-bold text-white mb-4">交易日誌 (純輪動策略)</h3>
        <TradeLogTable log={tradeLog} />
      </div>

      {buyAndHoldTradeLog && buyAndHoldTradeLog.length > 0 && (
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
            <h3 className="text-lg font-bold text-white mb-4">交易日誌 (純買入持有策略)</h3>
            <TradeLogTable log={buyAndHoldTradeLog} />
          </div>
      )}

      {showCashStrategies && optionsWithCashAnnualSummaries && optionsWithCashAnnualSummaries.length > 0 && (
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
            <h3 className="text-lg font-bold text-white mb-4">年度績效 (輪動+現金策略)</h3>
            <AnnualSummaryTable data={optionsWithCashAnnualSummaries} />
          </div>
      )}

      {showCashStrategies && optionsWithCashTradeLog && optionsWithCashTradeLog.length > 0 && (
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
            <h3 className="text-lg font-bold text-white mb-4">交易日誌 (輪動+現金策略)</h3>
            <TradeLogTable log={optionsWithCashTradeLog} />
          </div>
      )}

      {showCashStrategies && buyAndHoldWithCashTradeLog && buyAndHoldWithCashTradeLog.length > 0 && (
          <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
            <h3 className="text-lg font-bold text-white mb-4">交易日誌 (買入持有+現金策略)</h3>
            <TradeLogTable log={buyAndHoldWithCashTradeLog} />
          </div>
      )}
    </div>
  );
};

export default ResultsDisplay;
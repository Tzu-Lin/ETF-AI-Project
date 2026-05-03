import React, { useState } from 'react';
import { TradeLog, TradeType } from '../types';

interface TradeLogTableProps {
  log: TradeLog[];
}

const formatCurrency = (value: number | undefined) => {
    if(value === undefined) return '—';
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
    }).format(value);
};

const formatShares = (value: number | undefined) => {
    if(value === undefined) return '—';
    return value.toFixed(2);
}

const TradeLogTable: React.FC<TradeLogTableProps> = ({ log }) => {
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 10;
    const reversedLog = [...log].reverse();

    const indexOfLastItem = currentPage * itemsPerPage;
    const indexOfFirstItem = indexOfLastItem - itemsPerPage;
    const currentItems = reversedLog.slice(indexOfFirstItem, indexOfLastItem);
    const totalPages = Math.ceil(reversedLog.length / itemsPerPage);
    
    // Check if this log is for a strategy that includes a cash reserve by checking if any entry has the property
    const hasReserve = log.some(entry => entry.reserveCash !== undefined);

  return (
    <div>
        <div className="overflow-x-auto">
            <table className="min-w-full text-sm text-left text-gray-400">
                <thead className="text-xs text-gray-300 uppercase bg-gray-700">
                <tr>
                    <th scope="col" className="px-4 py-3">日期</th>
                    <th scope="col" className="px-4 py-3">類型</th>
                    <th scope="col" className="px-4 py-3">股價</th>
                    <th scope="col" className="px-4 py-3">詳細資訊</th>

                    {hasReserve ? (
                        <>
                            <th scope="col" className="px-4 py-3">主要現金</th>
                            <th scope="col" className="px-4 py-3">抄底儲備金</th>
                        </>
                    ) : (
                        <th scope="col" className="px-4 py-3">當前現金</th>
                    )}
                    
                    <th scope="col" className="px-4 py-3">當前持股</th>
                    <th scope="col" className="px-4 py-3">當前市值</th>
                    <th scope="col" className="px-4 py-3">選擇權收益</th>
                    <th scope="col" className="px-4 py-3">總資產 / 淨值</th>
                </tr>
                </thead>
                <tbody>
                {currentItems.map((entry, index) => (
                    <tr key={index} className="border-b border-gray-700 hover:bg-gray-700/50">
                    <td className="px-4 py-2">{entry.date.toLocaleDateString()}</td>
                    <td className="px-4 py-2">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                            entry.type === TradeType.SELL_PUT || entry.type === TradeType.SELL_CALL ? 'bg-blue-900 text-blue-300' : 
                            entry.type === TradeType.PUT_ASSIGNED || entry.type === TradeType.CALL_ASSIGNED ? 'bg-red-900 text-red-300' : 
                            entry.type === TradeType.PUT_EXPIRED || entry.type === TradeType.CALL_EXPIRED || entry.type === TradeType.RECOVERY_SELL ? 'bg-green-900 text-green-300' :
                            entry.type === TradeType.DIP_BUY ? 'bg-emerald-900 text-emerald-300' :
                            entry.type === TradeType.REBALANCE || entry.type === TradeType.RECURRING_INVESTMENT ? 'bg-indigo-900 text-indigo-300' :
                            entry.type === TradeType.EARLY_CLOSE ? 'bg-purple-900 text-purple-300' :
                            entry.type === TradeType.SIMULATION_END ? 'bg-gray-600 text-gray-200' :
                            'bg-gray-900 text-gray-300'
                        }`}>
                            {entry.type}
                        </span>
                    </td>
                    <td className="px-4 py-2 font-mono">{new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(entry.stockPrice)}</td>
                    <td className="px-4 py-2 text-xs max-w-xs truncate" title={entry.details}>{entry.details}</td>
                    
                    {hasReserve ? (
                        <>
                            <td className="px-4 py-2 font-mono">{formatCurrency(entry.cash - (entry.reserveCash || 0))}</td>
                            <td className="px-4 py-2 font-mono">{formatCurrency(entry.reserveCash)}</td>
                        </>
                    ) : (
                        <td className="px-4 py-2 font-mono">{formatCurrency(entry.cash)}</td>
                    )}

                    <td className="px-4 py-2 font-mono">{formatShares(entry.stockShares)}</td>
                    <td className="px-4 py-2 font-mono">{formatCurrency(entry.stockValue)}</td>
                    <td className={`px-4 py-2 font-mono ${entry.optionsProfit && entry.optionsProfit > 0 ? 'text-green-400' : entry.optionsProfit && entry.optionsProfit < 0 ? 'text-red-400' : ''}`}>{formatCurrency(entry.optionsProfit)}</td>
                    <td className="px-4 py-2 font-mono">{formatCurrency(entry.netWorth)}</td>
                    </tr>
                ))}
                </tbody>
            </table>
        </div>
        <div className="flex justify-between items-center mt-4 text-sm">
            <button
                onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
                disabled={currentPage === 1}
                className="px-3 py-1 bg-gray-700 rounded disabled:opacity-50"
            >
                上一頁
            </button>
            <span>第 {currentPage} 頁 / 共 {totalPages} 頁</span>
            <button
                onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
                disabled={currentPage === totalPages}
                className="px-3 py-1 bg-gray-700 rounded disabled:opacity-50"
            >
                下一頁
            </button>
        </div>
    </div>
  );
};

export default TradeLogTable;
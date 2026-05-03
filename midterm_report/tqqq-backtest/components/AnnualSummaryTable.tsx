

import React from 'react';
import { AnnualSummary } from '../types';

interface AnnualSummaryTableProps {
  data: AnnualSummary[];
}

const formatCurrency = (value: number) => {
    if (value === 0) return '—';
    return new Intl.NumberFormat('zh-TW', {
        style: 'currency',
        currency: 'TWD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0,
    }).format(value);
};

const AnnualSummaryTable: React.FC<AnnualSummaryTableProps> = ({ data }) => {
  return (
    <div className="overflow-x-auto max-h-96">
      <table className="min-w-full text-sm text-left text-gray-400">
        <thead className="text-xs text-gray-300 uppercase bg-gray-700 sticky top-0">
          <tr>
            <th scope="col" className="px-4 py-3">年份</th>
            <th scope="col" className="px-4 py-3">年初淨值</th>
            <th scope="col" className="px-4 py-3">年末淨值</th>
            <th scope="col" className="px-4 py-3">年化報酬率</th>
            <th scope="col" className="px-4 py-3">選擇權總收益</th>
            <th scope="col" className="px-4 py-3">股票總收益</th>
          </tr>
        </thead>
        <tbody>
          {data.map((summary) => (
            <tr key={summary.year} className="border-b border-gray-700 hover:bg-gray-700/50">
              <td className="px-4 py-2 font-semibold text-white">{summary.year}</td>
              <td className="px-4 py-2">{formatCurrency(summary.startNetWorth)}</td>
              <td className="px-4 py-2">{formatCurrency(summary.endNetWorth)}</td>
              <td className={`px-4 py-2 font-semibold ${summary.annualReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {summary.annualReturn.toFixed(2)}%
              </td>
              <td className={`px-4 py-2 font-mono ${summary.totalOptionsProfit > 0 ? 'text-green-400' : summary.totalOptionsProfit < 0 ? 'text-red-400' : ''}`}>
                {formatCurrency(summary.totalOptionsProfit)}
              </td>
              <td className={`px-4 py-2 font-mono ${summary.totalStockProfit > 0 ? 'text-green-400' : summary.totalStockProfit < 0 ? 'text-red-400' : ''}`}>
                {formatCurrency(summary.totalStockProfit)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default AnnualSummaryTable;
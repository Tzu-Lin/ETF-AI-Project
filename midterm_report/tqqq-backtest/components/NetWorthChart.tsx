
import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { PortfolioSnapshot, DailyPrice } from '../types';

interface NetWorthChartProps {
  optionsHistory: PortfolioSnapshot[];
  buyAndHoldHistory: PortfolioSnapshot[];
  optionsWithCashHistory: PortfolioSnapshot[];
  buyAndHoldWithCashHistory: PortfolioSnapshot[];
  priceHistory: DailyPrice[];
}

const formatCurrency = (value: number) => {
    if (value >= 1e6) return `$${(value / 1e6).toFixed(1)}M`;
    if (value >= 1e3) return `$${(value / 1e3).toFixed(0)}K`;
    return `$${value.toFixed(0)}`;
};

const CustomTooltip: React.FC<any> = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-gray-700 border border-gray-600 p-3 rounded-lg shadow-xl">
        <p className="label text-gray-300">{`${new Date(label).toLocaleDateString()}`}</p>
        {payload.map((p: any) => (
             <p key={p.name} style={{ color: p.color }}>
                {`${p.name}: ${
                    p.dataKey === 'stockPrice' 
                    ? new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(p.value)
                    : new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits:0, maximumFractionDigits:0 }).format(p.value)
                }`}
            </p>
        ))}
      </div>
    );
  }
  return null;
};


const NetWorthChart: React.FC<NetWorthChartProps> = ({ 
    optionsHistory, 
    buyAndHoldHistory, 
    optionsWithCashHistory,
    buyAndHoldWithCashHistory,
    priceHistory 
}) => {
    const combinedData = optionsHistory.map((optionsPoint, index) => ({
        date: optionsPoint.date.getTime(),
        optionsNetWorth: optionsPoint.netWorth,
        buyAndHoldNetWorth: buyAndHoldHistory[index]?.netWorth,
        optionsWithCashNetWorth: optionsWithCashHistory[index]?.netWorth,
        buyAndHoldWithCashNetWorth: buyAndHoldWithCashHistory[index]?.netWorth,
        stockPrice: priceHistory[index]?.price
    }));

  return (
    <div style={{ width: '100%', height: 400 }}>
        <ResponsiveContainer>
        <LineChart data={combinedData} margin={{ top: 5, right: 20, left: 30, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#4A5568" />
            <XAxis 
                dataKey="date" 
                tickFormatter={(unixTime) => new Date(unixTime).toLocaleDateString('en-US', { year: '2-digit', month: 'short' })}
                stroke="#A0AEC0"
                domain={['dataMin', 'dataMax']}
                type="number"
            />
            <YAxis 
                yAxisId="left"
                tickFormatter={formatCurrency} 
                stroke="#A0AEC0" 
                width={80} 
                domain={['auto', 'auto']}
            />
            <YAxis 
                yAxisId="right"
                orientation="right"
                stroke="#A0AEC0"
                tickFormatter={(value) => `$${value.toFixed(1)}`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line yAxisId="left" type="monotone" dataKey="optionsNetWorth" name="純輪動策略" stroke="#38BDF8" strokeWidth={2} dot={false} />
            <Line yAxisId="left" type="monotone" dataKey="buyAndHoldNetWorth" name="純買入持有" stroke="#F472B6" strokeWidth={2} dot={false} />
            {optionsWithCashHistory.length > 0 && 
              <Line yAxisId="left" type="monotone" dataKey="optionsWithCashNetWorth" name="輪動+現金" stroke="#34D399" strokeWidth={2} dot={false} />
            }
            {buyAndHoldWithCashHistory.length > 0 &&
              <Line yAxisId="left" type="monotone" dataKey="buyAndHoldWithCashNetWorth" name="買入持有+現金" stroke="#FBBF24" strokeWidth={2} dot={false} />
            }
            <Line yAxisId="right" type="monotone" dataKey="stockPrice" name="TQQQ 股價" stroke="#9CA3AF" strokeWidth={1} dot={false} strokeDasharray="5 5" />
        </LineChart>
        </ResponsiveContainer>
    </div>
  );
};

export default NetWorthChart;


import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { PortfolioSnapshot } from '../types';

interface HoldingsChartProps {
  data: PortfolioSnapshot[];
}

const formatCurrency = (value: number) => {
    if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(0)}K`;
    return value.toFixed(0);
};

const CustomTooltip: React.FC<any> = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    const total = payload.reduce((sum: number, entry: any) => sum + entry.value, 0);
    return (
      <div className="bg-gray-700 border border-gray-600 p-3 rounded-lg shadow-xl">
        <p className="label text-gray-300">{`${new Date(label).toLocaleDateString()}`}</p>
        {payload.map((p: any) => (
             <p key={p.name} style={{ color: p.color }}>
                {`${p.name}: ${new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits:0, maximumFractionDigits:0 }).format(p.value)}`}
            </p>
        ))}
         <p className="font-bold text-white mt-2">總計: {new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits:0, maximumFractionDigits:0 }).format(total)}</p>
      </div>
    );
  }
  return null;
};

const HoldingsChart: React.FC<HoldingsChartProps> = ({ data }) => {
    const chartData = data.map(item => ({
        date: item.date.getTime(),
        '現金': item.cash,
        'TQQQ 股票': item.stockValue,
    }));

  return (
    <div style={{ width: '100%', height: 300 }}>
      <ResponsiveContainer>
        <AreaChart data={chartData} margin={{ top: 5, right: 20, left: 30, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#4A5568" />
          <XAxis 
            dataKey="date" 
            tickFormatter={(unixTime) => new Date(unixTime).getFullYear().toString()}
            stroke="#A0AEC0"
            type="number"
            domain={['dataMin', 'dataMax']}
          />
          <YAxis 
            tickFormatter={formatCurrency} 
            stroke="#A0AEC0" 
            width={80}
            stackStrategy="expand"
          />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          <Area type="monotone" dataKey="現金" stackId="1" stroke="#38BDF8" fill="#38BDF8" fillOpacity={0.6} />
          <Area type="monotone" dataKey="TQQQ 股票" stackId="1" stroke="#F472B6" fill="#F472B6" fillOpacity={0.6} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default HoldingsChart;
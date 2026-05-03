import React, { useState, useMemo } from 'react';
import { StrategyParameters as StrategyParametersType } from '../types';

interface StrategyParametersProps {
  parameters: StrategyParametersType;
  onParametersChange: (name: keyof StrategyParametersType, value: string | number | boolean) => void;
  onDipTriggerChange: (index: number, field: 'drop' | 'use', value: string) => void;
  onRunBacktest: () => void;
  onAnalyzeStrategy: () => void;
  isLoading: boolean;
  isAnalyzing: boolean;
}

const InputField: React.FC<{
    label: string;
    name: keyof StrategyParametersType | string;
    type: string;
    value: string | number;
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
    unit?: string;
    min?: number;
    max?: number;
    step?: number;
    title?: string;
}> = ({ label, name, type, value, onChange, unit, min, max, step, title }) => (
    <div className="flex flex-col">
        <label htmlFor={name as string} className="mb-1 text-sm font-medium text-gray-300" title={title}>{label}</label>
        <div className="flex items-center">
            <input
                type={type}
                id={name as string}
                name={name as string}
                value={value}
                onChange={onChange}
                min={min}
                max={max}
                step={step}
                className="w-full bg-gray-700 border border-gray-600 rounded-md px-3 py-2 text-white focus:ring-blue-500 focus:border-blue-500"
            />
            {unit && <span className="ml-2 text-gray-400">{unit}</span>}
        </div>
    </div>
);


const StrategyParameters: React.FC<StrategyParametersProps> = ({
  parameters,
  onParametersChange,
  onDipTriggerChange,
  onRunBacktest,
  onAnalyzeStrategy,
  isLoading,
  isAnalyzing,
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showRecurring, setShowRecurring] = useState(true);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    let val: string | number | boolean = value;

    if (type === 'checkbox') {
        val = checked;
    } else if (type === 'number' && value !== '') {
        let numValue = parseFloat(value);
        if (name === 'putDelta' || name === 'callDelta') {
            // Enforce positive value for delta inputs
            numValue = Math.abs(numValue);
        }
        val = numValue;
    }
    
    onParametersChange(name as keyof StrategyParametersType, val);
  };
  
  const dipUseTotal = useMemo(() => {
    return parameters.dipTriggers.reduce((sum, trigger) => sum + trigger.use, 0);
  }, [parameters.dipTriggers]);

  const isDipUseInvalid = useMemo(() => {
      // The sum of percentages used from the reserve pool should not exceed 100%.
      return dipUseTotal > 100;
  }, [dipUseTotal]);

  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg space-y-6">
      
      <div>
        <h3 className="text-lg font-bold text-white mb-3">回測控制</h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <InputField label="初始資金 (TWD)" name="initialCapital" type="number" value={parameters.initialCapital} onChange={handleChange} unit="台幣" />
          <InputField label="開始日期" name="startDate" type="date" value={parameters.startDate} onChange={handleChange} />
          <InputField label="結束日期" name="endDate" type="date" value={parameters.endDate} onChange={handleChange} />
        </div>
      </div>

      <div>
        <button onClick={() => setShowRecurring(!showRecurring)} className="text-blue-400 text-sm hover:underline w-full text-left mb-2">
          {showRecurring ? '▼ 隱藏定期定額 & 匯率設定' : '► 顯示定期定額 & 匯率設定'}
        </button>
        {showRecurring && (
          <div className="bg-gray-900/50 p-4 rounded-md space-y-4">
            <div className="flex items-center space-x-3">
              <input
                type="checkbox"
                id="enableRecurringInvestment"
                name="enableRecurringInvestment"
                checked={parameters.enableRecurringInvestment}
                onChange={handleChange}
                className="h-5 w-5 rounded bg-gray-700 border-gray-600 text-blue-500 focus:ring-blue-500"
              />
              <label htmlFor="enableRecurringInvestment" className="font-medium text-gray-200">啟用定期定額</label>
            </div>
            {parameters.enableRecurringInvestment && (
              <div className="mt-4 pt-4 border-t border-gray-700/50 grid grid-cols-1 sm:grid-cols-2 gap-4">
                <InputField label="每月投入 (TWD)" name="monthlyInvestmentTWD" type="number" value={parameters.monthlyInvestmentTWD} onChange={handleChange} unit="台幣" />
                <InputField label="美金匯率" name="exchangeRate" type="number" value={parameters.exchangeRate} onChange={handleChange} min={1} step={0.1} title="1 美金兌換多少台幣" />
              </div>
            )}
          </div>
        )}
      </div>

      <div>
        <h3 className="text-lg font-bold text-white mb-3">現金儲備策略 (用於基準比較)</h3>
        <div className="flex items-center space-x-3 bg-gray-900/50 p-3 rounded-md">
          <input
            type="checkbox"
            id="enableCashTactic"
            name="enableCashTactic"
            checked={parameters.enableCashTactic}
            onChange={handleChange}
            className="h-5 w-5 rounded bg-gray-700 border-gray-600 text-blue-500 focus:ring-blue-500"
          />
          <label htmlFor="enableCashTactic" className="font-medium text-gray-200">啟用現金儲備進行策略比較</label>
        </div>
        
        {parameters.enableCashTactic && (
            <div className="mt-4 border-l-2 border-blue-500 pl-4 space-y-4">
                <p className="text-sm text-gray-400">
                    啟用後，將額外模擬兩種保留部分現金用於抄底的策略。動用的儲備是基於 **初始設定的現金儲備總額** 的百分比。
                </p>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <InputField label="現金儲備比例" name="cashReservePercentage" type="number" value={parameters.cashReservePercentage} onChange={handleChange} min={0} max={100} unit="%" />
                </div>
                 {isDipUseInvalid && (
                    <div className="bg-red-900/50 border border-red-700 text-red-300 px-4 py-2 rounded-md text-sm">
                        <strong>警告：</strong>「動用儲備」的總和 ({dipUseTotal.toFixed(2)}%) 已超過 100%。這表示您計劃動用的資金超過了總儲備金，請調整參數。
                    </div>
                )}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {parameters.dipTriggers.map((trigger, index) => (
                        <div key={index} className="bg-gray-700/50 p-4 rounded-lg">
                            <h4 className="font-bold text-white mb-2">抄底觸發點 {index + 1}</h4>
                            <div className="space-y-3">
                                <InputField label="從 ATH 下跌 (%)" name={`drop-${index}`} type="number" value={trigger.drop} onChange={(e) => onDipTriggerChange(index, 'drop', e.target.value)} min={0} unit="%" />
                                <InputField label="動用儲備 (%)" name={`use-${index}`} type="number" value={trigger.use} onChange={(e) => onDipTriggerChange(index, 'use', e.target.value)} min={0} unit="%" title={`佔總儲備金的 ${trigger.use}%`} />
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        )}
      </div>

      <div>
        <button onClick={() => setShowAdvanced(!showAdvanced)} className="text-blue-400 text-sm hover:underline">
          {showAdvanced ? '隱藏進階選項' : '顯示進階選項'}
        </button>
        {showAdvanced && (
          <div className="mt-3 bg-gray-900/50 p-4 rounded-md space-y-6">
            <div>
              <h3 className="text-lg font-bold text-white mb-3">選擇權輪動策略</h3>
               <div 
                  className="flex items-center space-x-3 mb-4"
                  title="啟用後，只有當股價高於您的持股成本價時，才會賣出掩護性買權 (Covered Call)。這可以避免您的股票在虧損狀態下被賣出，但可能會在股價反彈期間減少權利金收入。"
                >
                    <input
                        type="checkbox"
                        id="sellCallAboveCostBasisOnly"
                        name="sellCallAboveCostBasisOnly"
                        checked={parameters.sellCallAboveCostBasisOnly}
                        onChange={handleChange}
                        className="h-5 w-5 rounded bg-gray-700 border-gray-600 text-blue-500 focus:ring-blue-500"
                    />
                    <label htmlFor="sellCallAboveCostBasisOnly" className="font-medium text-gray-200">僅在成本價之上賣出 Call</label>
                </div>
              <p className="text-sm text-gray-400 mb-2">Delta 請輸入正數。</p>
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4">
                <InputField label="Put DTE" name="putDTE" type="number" value={parameters.putDTE} onChange={handleChange} min={1} unit="天" />
                <InputField label="Put Delta" name="putDelta" type="number" value={parameters.putDelta} onChange={handleChange} min={0.01} max={0.99} step={0.01} />
                <InputField label="Call DTE" name="callDTE" type="number" value={parameters.callDTE} onChange={handleChange} min={1} unit="天" />
                <InputField label="Call Delta" name="callDelta" type="number" value={parameters.callDelta} onChange={handleChange} min={0.01} max={0.99} step={0.01} />
              </div>
            </div>
            <div>
              <h3 className="text-lg font-bold text-white mb-3">真實性參數</h3>
               <p className="text-sm text-gray-400 mb-2">
                  此處的參數用於模擬更真實的交易環境。
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <InputField label="選擇權合約費" name="optionFee" type="number" value={parameters.optionFee} onChange={handleChange} min={0} step={0.01} unit="$/合約" />
                <InputField label="交易滑價" name="slippage" type="number" value={parameters.slippage} onChange={handleChange} min={0} step={0.1} unit="%" />
                <InputField 
                    label="IV 動態調整因子" 
                    name="ivAdjustmentFactor" 
                    type="number" 
                    value={parameters.ivAdjustmentFactor} 
                    onChange={handleChange} 
                    min={0} 
                    step={0.1}
                    title="控制IV因應股價下跌的增長幅度。0=無調整, 1=線性增長, 2=更劇烈增長。"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-col sm:flex-row items-center justify-end gap-4 pt-4 border-t border-gray-700">
        <button
          onClick={onAnalyzeStrategy}
          disabled={isLoading || isAnalyzing}
          className="w-full sm:w-auto px-6 py-2 bg-gray-600 text-white font-semibold rounded-md hover:bg-gray-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isAnalyzing ? '分析中...' : 'Gemini 分析策略'}
        </button>
        <button
          onClick={onRunBacktest}
          disabled={isLoading || isAnalyzing || (parameters.enableCashTactic && isDipUseInvalid)}
          className="w-full sm:w-auto px-8 py-2 bg-blue-600 text-white font-bold rounded-md hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          title={isDipUseInvalid ? "請修正抄底儲備參數後再執行" : ""}
        >
          {isLoading ? '執行中...' : '執行回測'}
        </button>
      </div>
    </div>
  );
};

export default StrategyParameters;
import React, { useState, useCallback } from 'react';
import StrategyParameters from './StrategyParameters';
import ResultsDisplay from './ResultsDisplay';
// FIX: Moved BacktestResult import from backtestService to types to resolve module export error.
import { StrategyParameters as StrategyParametersType, BacktestResult }  from '../types';
import { runBacktest } from '../services/backtestService';
import { getStrategyAnalysis } from '../services/geminiService';
import { tqqqHistoricalDataRaw } from '../data/tqqqData';

// Helper function to get the last date from the data and format it for the date input
const getLastAvailableDate = (): string => {
    if (!tqqqHistoricalDataRaw || tqqqHistoricalDataRaw.length === 0) {
        // A sensible fallback if data is somehow empty
        const today = new Date();
        return today.toISOString().split('T')[0];
    }
    const lastEntry = tqqqHistoricalDataRaw[tqqqHistoricalDataRaw.length - 1];
    // Convert YYYY/MM/DD to YYYY-MM-DD for the HTML date input
    return lastEntry.date.replace(/\//g, '-');
};


const Dashboard: React.FC = () => {
    const [parameters, setParameters] = useState<StrategyParametersType>({
        startDate: '2021-01-01',
        endDate: getLastAvailableDate(),
        initialCapital: 3000000,
        putDTE: 42,
        putDelta: 0.3,
        callDTE: 42,
        callDelta: 0.20,
        sellCallAboveCostBasisOnly: true, // Default to current logic
        enableCashTactic: true, // Enable by default to show the 4 strategies
        cashReservePercentage: 30,
        dipTriggers: [
            { drop: 70, use: 10 },
            { drop: 75, use: 10 },
            { drop: 80, use: 10 },
        ],
        // Recurring investment parameters
        enableRecurringInvestment: true,
        monthlyInvestmentTWD: 20000,
        exchangeRate: 31,
        // Realism parameters with defaults
        optionFee: 0.50,
        slippage: 2,
        ivAdjustmentFactor: 1, // Default for the new dynamic IV model
    });
    
    const [results, setResults] = useState<BacktestResult | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [geminiAnalysis, setGeminiAnalysis] = useState('');

    const handleParametersChange = useCallback((name: keyof StrategyParametersType, value: string | number | boolean) => {
        setParameters(prev => ({ ...prev, [name]: value }));
    }, []);

    const handleDipTriggerChange = useCallback((index: number, field: 'drop' | 'use', value: string) => {
        const numericValue = parseFloat(value);
        if (isNaN(numericValue) || numericValue < 0) return;

        setParameters(prev => {
            const newTriggers = [...prev.dipTriggers];
            newTriggers[index] = { ...newTriggers[index], [field]: numericValue };
            return { ...prev, dipTriggers: newTriggers };
        });
    }, []);

    const handleRunBacktest = useCallback(() => {
        setIsLoading(true);
        setResults(null);
        setTimeout(() => {
            try {
                const backtestData = runBacktest(parameters);
                setResults(backtestData);
            } catch (error) {
                console.error("Backtest failed:", error);
            } finally {
                setIsLoading(false);
            }
        }, 50);
    }, [parameters]);
    
    const handleAnalyzeStrategy = useCallback(async () => {
        setIsAnalyzing(true);
        setGeminiAnalysis('');
        try {
            const analysis = await getStrategyAnalysis(parameters);
            setGeminiAnalysis(analysis);
        } catch (error) {
            console.error("Gemini analysis failed:", error);
            setGeminiAnalysis("Failed to get analysis. See console for details.");
        } finally {
            setIsAnalyzing(false);
        }
    }, [parameters]);

    return (
        <main className="p-4 sm:p-6 md:p-8">
            <header className="mb-6">
                <h1 className="text-3xl font-bold text-white">TQQQ 選擇權輪動策略回測</h1>
                <p className="text-gray-400 mt-1">比較四種策略：純輪動、純買入持有，以及它們各自帶有現金儲備的版本。</p>
            </header>

            <StrategyParameters 
                parameters={parameters}
                onParametersChange={handleParametersChange}
                onDipTriggerChange={handleDipTriggerChange}
                onRunBacktest={handleRunBacktest}
                onAnalyzeStrategy={handleAnalyzeStrategy}
                isLoading={isLoading}
                isAnalyzing={isAnalyzing}
            />
            
            {isLoading && (
                <div className="mt-6 bg-gray-800 p-6 rounded-lg text-center">
                    <p className="text-white text-lg animate-pulse">正在執行回測...</p>
                </div>
            )}
            
            {!isLoading && <ResultsDisplay results={results} geminiAnalysis={geminiAnalysis} isAnalyzing={isAnalyzing} />}

        </main>
    );
};

export default Dashboard;
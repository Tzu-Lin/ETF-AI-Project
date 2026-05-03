import { tqqqHistoricalDataRaw } from '../data/tqqqData';
import { blackScholesPrice, findStrikeForDelta } from '../utils/blackScholes';
import { DailyPrice, PortfolioSnapshot, StrategyParameters, TradeLog, TradeType, AnnualSummary, BacktestResult, Analytics } from '../types';

const parseDate = (dateStr: string): Date => {
  const [year, month, day] = dateStr.split('/').map(Number);
  return new Date(year, month - 1, day);
};

export const tqqqHistoricalData: DailyPrice[] = tqqqHistoricalDataRaw.map(d => ({
    date: parseDate(d.date),
    price: d.price
}));

interface OpenPosition {
    type: 'PUT' | 'CALL';
    strike: number;
    expiryDate: Date;
    premium: number;
    contractSize: number;
    entryDate: Date;
}

const findClosestTradingDay = (targetDate: Date, priceData: DailyPrice[]): Date => {
    if (priceData.length === 0) return targetDate;
    let closestDate = priceData[0].date;
    let minDiff = Math.abs(targetDate.getTime() - closestDate.getTime());

    for (const day of priceData) {
        const diff = Math.abs(targetDate.getTime() - day.date.getTime());
        if (diff < minDiff) {
            minDiff = diff;
            closestDate = day.date;
        }
    }
    return closestDate;
};

// Calculates the Internal Rate of Return (IRR) for a series of cash flows.
// This provides a money-weighted rate of return, accurate for recurring investments.
const calculateIrr = (cashflows: { amount: number; date: Date }[]): number => {
    if (cashflows.length < 2) return 0;
    
    const hasPositive = cashflows.some(cf => cf.amount > 0);
    const hasNegative = cashflows.some(cf => cf.amount < 0);
    if (!hasPositive || !hasNegative) return 0;

    const sortedFlows = cashflows.sort((a, b) => a.date.getTime() - b.date.getTime());
    const startDate = sortedFlows[0].date;

    const timeInYears = (date: Date) => (date.getTime() - startDate.getTime()) / (365.25 * 24 * 60 * 60 * 1000);

    const npv = (rate: number): number => {
        if (rate <= -1) rate = -0.99999; // Prevent Math.pow with negative base
        return sortedFlows.reduce((acc, flow) => {
            return acc + flow.amount / Math.pow(1 + rate, timeInYears(flow.date));
        }, 0);
    };

    const derivative = (rate: number): number => {
         if (rate <= -1) rate = -0.99999;
        return sortedFlows.reduce((acc, flow) => {
            const t = timeInYears(flow.date);
            if (t === 0) return acc;
            return acc - (flow.amount * t) / Math.pow(1 + rate, t + 1);
        }, 0);
    };
    
    let guess = 0.1; // Initial guess for IRR (10%)
    const maxIterations = 100;
    const tolerance = 1e-7;

    for (let i = 0; i < maxIterations; i++) {
        const npvValue = npv(guess);
        const derivativeValue = derivative(guess);

        if (Math.abs(npvValue) < tolerance) {
            return guess * 100; // Return as percentage
        }
        
        if (derivativeValue === 0) break;
        
        guess = guess - npvValue / derivativeValue;
    }
    
    return Math.abs(npv(guess)) < 1e-5 ? guess * 100 : 0; // Return 0 if not converged
};


const calculateAnalytics = (
    history: PortfolioSnapshot[], 
    totalInvestedCapital: number, 
    cashflows: { amount: number; date: Date }[]
): Analytics => {
    if (history.length === 0) return { totalReturn: 0, cagr: 0, sharpeRatio: 0, maxDrawdown: 0, drawdownPeakUSD: 0, drawdownTroughUSD: 0, drawdownPeakDate: new Date(), drawdownTroughDate: new Date() };
    
    const finalNetWorth = history[history.length - 1].netWorth;
    
    // Simple return: (Final Value / Total Capital In) - 1. Kept for basic reference.
    const totalReturn = totalInvestedCapital > 0 ? (finalNetWorth / totalInvestedCapital - 1) * 100 : 0;
    
    // CAGR: Use IRR for a money-weighted return, which is more accurate for portfolios with cash flows.
    const finalCashflows = [...cashflows, { amount: finalNetWorth, date: history[history.length - 1].date }];
    const cagr = calculateIrr(finalCashflows);
    
    let peakUSD = history[0].netWorth;
    let peakDate = history[0].date;
    let maxDrawdown = 0;
    let drawdownPeakUSD = peakUSD;
    let drawdownTroughUSD = peakUSD;
    let drawdownPeakDate = peakDate;
    let drawdownTroughDate = peakDate;


    for (const s of history) {
        if (s.netWorth > peakUSD) {
            peakUSD = s.netWorth;
            peakDate = s.date;
        }
        const drawdown = peakUSD > 0 ? (peakUSD - s.netWorth) / peakUSD : 0;
        if (drawdown > maxDrawdown) {
            maxDrawdown = drawdown;
            drawdownPeakUSD = peakUSD;
            drawdownTroughUSD = s.netWorth;
            drawdownPeakDate = peakDate;
            drawdownTroughDate = s.date;
        }
    }

    return { 
        totalReturn, 
        cagr, 
        sharpeRatio: 0, 
        maxDrawdown: maxDrawdown * 100,
        drawdownPeakUSD,
        drawdownTroughUSD,
        drawdownPeakDate,
        drawdownTroughDate
    };
};


// Annual market parameters based on user-provided data
const annualParameters: { [year: number]: { iv: number; riskFreeRate: number } } = {
    2010: { iv: 0.518, riskFreeRate: 0.002 },
    2011: { iv: 0.557, riskFreeRate: 0.001 },
    2012: { iv: 0.409, riskFreeRate: 0.001 },
    2013: { iv: 0.327, riskFreeRate: 0.001 },
    2014: { iv: 0.327, riskFreeRate: 0.001 },
    2015: { iv: 0.384, riskFreeRate: 0.001 },
    2016: { iv: 0.363, riskFreeRate: 0.003 },
    2017: { iv: 0.255, riskFreeRate: 0.013 },
    2018: { iv: 0.382, riskFreeRate: 0.025 },
    2019: { iv: 0.354, riskFreeRate: 0.018 },
    2020: { iv: 0.674, riskFreeRate: 0.005 },
    2021: { iv: 0.453, riskFreeRate: 0.009 },
    2022: { iv: 0.589, riskFreeRate: 0.028 },
    2023: { iv: 0.389, riskFreeRate: 0.040 },
    2024: { iv: 0.359, riskFreeRate: 0.045 },
    2025: { iv: 0.478, riskFreeRate: 0.0425 },
};

const calculateAnnualSummaries = (
    history: PortfolioSnapshot[],
    tradeLogs: TradeLog[],
    initialCapitalUSD: number,
    exchangeRate: number
): AnnualSummary[] => {
    const annualSummariesMap: Map<number, {
        snapshots: PortfolioSnapshot[];
        tradeLogs: TradeLog[];
    }> = new Map();

    history.forEach(snapshot => {
        const year = snapshot.date.getFullYear();
        if (!annualSummariesMap.has(year)) {
            annualSummariesMap.set(year, { snapshots: [], tradeLogs: [] });
        }
        annualSummariesMap.get(year)!.snapshots.push(snapshot);
    });

    tradeLogs.forEach(log => {
        const year = log.date.getFullYear();
        if (annualSummariesMap.has(year)) {
            annualSummariesMap.get(year)!.tradeLogs.push(log);
        }
    });
    
    const annualSummaries: AnnualSummary[] = [];
    const sortedYears = Array.from(annualSummariesMap.keys()).sort();
    let lastYearEndNetWorth: number = initialCapitalUSD;

    for (const year of sortedYears) {
        const yearData = annualSummariesMap.get(year)!;
        const startNetWorth = lastYearEndNetWorth;
        const endNetWorth = yearData.snapshots[yearData.snapshots.length - 1].netWorth;
        
        const totalOptionsProfit = yearData.tradeLogs.reduce((sum, log) => sum + (log.optionsProfit || 0), 0);
        
        const totalProfit = endNetWorth - startNetWorth;
        const totalStockProfit = totalProfit - totalOptionsProfit;

        annualSummaries.push({
            year,
            startNetWorth,
            endNetWorth,
            annualReturn: startNetWorth > 0 ? (endNetWorth / startNetWorth - 1) * 100 : 0,
            totalOptionsProfit,
            totalStockProfit,
        });

        lastYearEndNetWorth = endNetWorth;
    }

    return annualSummaries.map(summary => ({
        ...summary,
        startNetWorth: summary.startNetWorth * exchangeRate,
        endNetWorth: summary.endNetWorth * exchangeRate,
        totalOptionsProfit: summary.totalOptionsProfit * exchangeRate,
        totalStockProfit: summary.totalStockProfit * exchangeRate,
    }));
};


export const runBacktest = (params: StrategyParameters): BacktestResult => {
    const tradeLog: TradeLog[] = []; // P1: Pure Options
    const p2_tradeLog: TradeLog[] = []; // P2: Pure B&H
    const p3_tradeLog: TradeLog[] = []; // P3: Options + Cash
    const p4_tradeLog: TradeLog[] = []; // P4: B&H + Cash

    const optionsHistory: PortfolioSnapshot[] = [];
    const buyAndHoldHistory: PortfolioSnapshot[] = [];
    const optionsWithCashHistory: PortfolioSnapshot[] = [];
    const buyAndHoldWithCashHistory: PortfolioSnapshot[] = [];
    
    const EARLY_CLOSE_PROFIT_TARGET = 0.75;

    const dataSlice = tqqqHistoricalData.filter(d => 
        d.date >= new Date(params.startDate) && d.date <= new Date(params.endDate)
    );

    const emptyAnalytics = { totalReturn: 0, cagr: 0, sharpeRatio: 0, maxDrawdown: 0, drawdownPeakUSD: 0, drawdownTroughUSD: 0, drawdownPeakDate: new Date(), drawdownTroughDate: new Date() };
    if (dataSlice.length === 0) {
        return { 
            optionsHistory: [], buyAndHoldHistory: [], optionsWithCashHistory: [], buyAndHoldWithCashHistory: [], 
            tradeLog: [], optionsWithCashTradeLog: [], buyAndHoldTradeLog: [], buyAndHoldWithCashTradeLog: [], 
            annualSummaries: [], optionsWithCashAnnualSummaries: [], priceHistory: [],
            analytics: emptyAnalytics, buyAndHoldAnalytics: emptyAnalytics, optionsWithCashAnalytics: emptyAnalytics, buyAndHoldWithCashAnalytics: emptyAnalytics,
            totalTwdInvested: params.initialCapital,
            exchangeRate: params.exchangeRate,
        };
    }
    
    const initialCapitalUSD = params.initialCapital / params.exchangeRate;
    const reserveRatio = params.enableCashTactic ? params.cashReservePercentage / 100 : 0;
    
    // Cash flow tracking for IRR calculation
    const p1_cashflows: { amount: number; date: Date }[] = [{ amount: -initialCapitalUSD, date: dataSlice[0].date }];
    const p2_cashflows: { amount: number; date: Date }[] = [{ amount: -initialCapitalUSD, date: dataSlice[0].date }];
    const p3_cashflows: { amount: number; date: Date }[] = [{ amount: -initialCapitalUSD, date: dataSlice[0].date }];
    const p4_cashflows: { amount: number; date: Date }[] = [{ amount: -initialCapitalUSD, date: dataSlice[0].date }];

    // Investment Tracking
    let totalTwdInvested = params.initialCapital;
    let accumulatedTwdSavings = 0;
    let lastInvestmentMonth = -1; // Use -1 to trigger investment in the first month

    let allTimeHigh = dataSlice[0].price;

    // --- STRATEGY 1: Pure Options Wheel ---
    let p1_cash = initialCapitalUSD;
    let p1_stockShares = 0;
    let p1_openPosition: OpenPosition | null = null;
    let p1_assignmentPrice: number | null = null;
    let p1_totalInvestedUSD = initialCapitalUSD;
    
    // --- STRATEGY 2: Pure Buy & Hold ---
    let p2_stockShares = initialCapitalUSD / dataSlice[0].price;
    let p2_totalInvestedUSD = initialCapitalUSD;

    // --- STRATEGE 3: Options Wheel + Cash Reserve ---
    const p3_initialCapitalUSD = initialCapitalUSD;
    const p3_initialReserveForDipBuy = p3_initialCapitalUSD * reserveRatio;
    let p3_mainCash = p3_initialCapitalUSD * (1 - reserveRatio);
    let p3_reserveCash = p3_initialCapitalUSD * reserveRatio;
    let p3_stockShares = 0;
    let p3_openPosition: OpenPosition | null = null;
    let p3_assignmentPrice: number | null = null;
    let p3_allTimeHigh = dataSlice[0].price;
    const p3_dipTriggers = params.dipTriggers.map(t => ({...t, triggered: false}));
    let p3_inDip = false;
    let p3_totalInvestedUSD = initialCapitalUSD;

    // --- STRATEGY 4: Buy & Hold + Cash Reserve ---
    const p4_initialCapitalUSD = initialCapitalUSD;
    const p4_initialReserveForDipBuy = p4_initialCapitalUSD * reserveRatio;
    let p4_cashReserve = p4_initialCapitalUSD * reserveRatio;
    const p4_initialInvestedCapital = p4_initialCapitalUSD * (1 - reserveRatio);
    let p4_stockShares = p4_initialInvestedCapital / dataSlice[0].price;
    let p4_allTimeHigh = dataSlice[0].price;
    const p4_dipTriggers = params.dipTriggers.map(t => ({...t, triggered: false}));
    let p4_inDip = false;
    let p4_totalInvestedUSD = initialCapitalUSD;

    for (const currentData of dataSlice) {
        const currentDate = currentData.date;
        const currentPrice = currentData.price;
        const currentYear = currentDate.getFullYear();
        
        allTimeHigh = Math.max(allTimeHigh, currentPrice);
        
        const p1_netWorth = () => p1_cash + p1_stockShares * currentPrice;
        const p3_netWorth = () => p3_mainCash + p3_reserveCash + p3_stockShares * currentPrice;

        // --- Recurring Investment Logic (runs once per month) ---
        if (params.enableRecurringInvestment) {
            const currentMonth = currentDate.getMonth();
            if (currentMonth !== lastInvestmentMonth) {
                totalTwdInvested += params.monthlyInvestmentTWD;
                accumulatedTwdSavings += params.monthlyInvestmentTWD;
                lastInvestmentMonth = currentMonth;

                const costOf100SharesUSD = currentPrice * 100;
                const costOf100SharesTWD = costOf100SharesUSD * params.exchangeRate;

                if (accumulatedTwdSavings >= costOf100SharesTWD) {
                    const lotsToTransfer = Math.floor(accumulatedTwdSavings / costOf100SharesTWD);
                    const twdToSpend = lotsToTransfer * costOf100SharesTWD;
                    const usdToAdd = lotsToTransfer * costOf100SharesUSD;
                    
                    accumulatedTwdSavings -= twdToSpend;

                    // Log cash flow for IRR calculation
                    p1_cashflows.push({ amount: -usdToAdd, date: currentDate });
                    p2_cashflows.push({ amount: -usdToAdd, date: currentDate });
                    p3_cashflows.push({ amount: -usdToAdd, date: currentDate });
                    p4_cashflows.push({ amount: -usdToAdd, date: currentDate });

                    // P1: Options
                    p1_cash += usdToAdd;
                    p1_totalInvestedUSD += usdToAdd;

                    // P2: B&H
                    const p2_sharesBought = usdToAdd / currentPrice;
                    p2_stockShares += p2_sharesBought;
                    p2_totalInvestedUSD += usdToAdd;
                    const p2_details = `投入 ${lotsToTransfer} 批資金 (共 ${usdToAdd.toFixed(0)} USD)，買入 ${p2_sharesBought.toFixed(2)} 股`;
                    p2_tradeLog.push({ 
                        date: currentDate, 
                        type: TradeType.RECURRING_INVESTMENT, 
                        stockPrice: currentPrice, 
                        details: p2_details, 
                        cash: 0,
                        stockShares: p2_stockShares, 
                        stockValue: p2_stockShares * currentPrice, 
                        netWorth: p2_stockShares * currentPrice 
                    });
                    
                    if (params.enableCashTactic) {
                        // P3: Options + Cash - Add new funds and rebalance cash pools
                        p3_mainCash += usdToAdd;
                        p3_totalInvestedUSD += usdToAdd;

                        const p3_currentNetWorth = p3_mainCash + p3_reserveCash + (p3_stockShares * currentPrice);
                        const p3_targetReserve = p3_currentNetWorth * reserveRatio;
                        const p3_transferAmount = p3_targetReserve - p3_reserveCash;

                        if (p3_transferAmount > 0 && p3_mainCash >= p3_transferAmount) {
                            p3_mainCash -= p3_transferAmount;
                            p3_reserveCash += p3_transferAmount;
                        } else if (p3_transferAmount < 0 && p3_reserveCash >= Math.abs(p3_transferAmount)) {
                            p3_mainCash -= p3_transferAmount; // Adds to mainCash due to double negative
                            p3_reserveCash += p3_transferAmount; // Subtracts from reserveCash
                        }

                        // P4: B&H + Cash - Add new funds and rebalance the entire portfolio
                        p4_cashReserve += usdToAdd;
                        p4_totalInvestedUSD += usdToAdd;

                        const p4_currentNetWorth = (p4_stockShares * currentPrice) + p4_cashReserve;
                        const p4_targetReserve = p4_currentNetWorth * reserveRatio;
                        const p4_capitalForStock = p4_currentNetWorth - p4_targetReserve;

                        p4_cashReserve = p4_targetReserve;
                        p4_stockShares = p4_capitalForStock / currentPrice;
                        const p4_details = `投入 ${lotsToTransfer} 批資金 (共 ${usdToAdd.toFixed(0)} USD) 並重平衡資產。`;
                        p4_tradeLog.push({
                            date: currentDate,
                            type: TradeType.RECURRING_INVESTMENT,
                            stockPrice: currentPrice,
                            details: p4_details,
                            cash: p4_cashReserve,
                            stockShares: p4_stockShares,
                            stockValue: p4_stockShares * currentPrice,
                            netWorth: p4_currentNetWorth,
                            reserveCash: p4_cashReserve
                        });
                    }

                    const details = `投入 ${lotsToTransfer} 批資金 (共 ${usdToAdd.toFixed(0)} USD)`;
                    tradeLog.push({ date: currentDate, type: TradeType.RECURRING_INVESTMENT, stockPrice: currentPrice, details, cash: p1_cash, stockShares: p1_stockShares, stockValue: p1_stockShares * currentPrice, netWorth: p1_netWorth() });
                    if (params.enableCashTactic) {
                        p3_tradeLog.push({ date: currentDate, type: TradeType.RECURRING_INVESTMENT, stockPrice: currentPrice, details, cash: p3_mainCash + p3_reserveCash, stockShares: p3_stockShares, stockValue: p3_stockShares * currentPrice, netWorth: p3_netWorth(), reserveCash: p3_reserveCash });
                    }
                }
            }
        }
        
        // Use year-specific IV and risk-free rate, with a fallback for years outside the defined range
        const defaultParams = annualParameters[2023]; // Use a recent year as a fallback
        const paramsForYear = annualParameters[currentYear] || defaultParams;
        const baseIV = paramsForYear.iv;
        const RISK_FREE_RATE = paramsForYear.riskFreeRate;

        // --- Dynamic IV Calculation ---
        const drawdown = (allTimeHigh - currentPrice) / allTimeHigh;
        const dynamicIV = baseIV * (1 + (drawdown * params.ivAdjustmentFactor));
        
        // --- P3 & P4: ATH Recovery and Rebalance Logic ---
        if (params.enableCashTactic) {
            const isNewAthP3 = currentPrice > p3_allTimeHigh;
            if (p3_inDip && isNewAthP3) {
                const proceeds = p3_stockShares * currentPrice;
                const details = `P3: New ATH reached at ${currentPrice.toFixed(2)}. Sold all ${p3_stockShares.toFixed(2)} shares.`;
                p3_mainCash += proceeds;
                const soldShares = p3_stockShares;
                p3_stockShares = 0;
                p3_assignmentPrice = null; 
                p3_tradeLog.push({ date: currentDate, type: TradeType.RECOVERY_SELL, stockPrice: currentPrice, details, cash: p3_mainCash + p3_reserveCash, stockShares: p3_stockShares, stockValue: p3_stockShares * currentPrice, netWorth: p3_netWorth(), stockProfit: soldShares > 0 ? proceeds - (p1_assignmentPrice || currentPrice) * soldShares : 0, reserveCash: p3_reserveCash });
                
                const totalNetWorth = p3_mainCash + p3_reserveCash;
                p3_mainCash = totalNetWorth * (1 - reserveRatio);
                p3_reserveCash = totalNetWorth * reserveRatio;
                p3_tradeLog.push({ date: currentDate, type: TradeType.REBALANCE, stockPrice: currentPrice, details: `P3: Rebalanced portfolio to ${100 - params.cashReservePercentage}:${params.cashReservePercentage} after ATH.`, cash: p3_mainCash + p3_reserveCash, stockShares: p3_stockShares, stockValue: p3_stockShares * currentPrice, netWorth: totalNetWorth, reserveCash: p3_reserveCash });
                
                p3_inDip = false;
                p3_dipTriggers.forEach(t => t.triggered = false);
            }
            p3_allTimeHigh = Math.max(p3_allTimeHigh, currentPrice);

            const isNewAthP4 = currentPrice > p4_allTimeHigh;
            if(p4_inDip && isNewAthP4) {
                const totalNetWorth = (p4_stockShares * currentPrice) + p4_cashReserve;
                p4_cashReserve = totalNetWorth * reserveRatio;
                const investableCapital = totalNetWorth * (1 - reserveRatio);
                p4_stockShares = investableCapital / currentPrice;
                p4_inDip = false;
                p4_dipTriggers.forEach(t => t.triggered = false);
                p4_tradeLog.push({
                    date: currentDate,
                    type: TradeType.REBALANCE,
                    stockPrice: currentPrice,
                    details: `P4: New ATH. Rebalanced portfolio to ${100 - params.cashReservePercentage}:${params.cashReservePercentage}.`,
                    cash: p4_cashReserve,
                    stockShares: p4_stockShares,
                    stockValue: p4_stockShares * currentPrice,
                    netWorth: totalNetWorth,
                    reserveCash: p4_cashReserve
                });
            }
            p4_allTimeHigh = Math.max(p4_allTimeHigh, currentPrice);

            // --- P3 Daily Cash Reserve Rebalancing ---
            // We only rebalance if we are NOT in a dip-buying phase.
            // During a dip, the reserve is meant to be spent down.
            if (!p3_inDip) {
                const totalNetWorth = p3_netWorth();
                if (totalNetWorth > 0) {
                    const currentReserveRatio = p3_reserveCash / totalNetWorth;
                    const targetReserveRatio = reserveRatio;
                    const REPLENISH_THRESHOLD_RATIO = 0.20;

                    if (targetReserveRatio > 0 && currentReserveRatio < REPLENISH_THRESHOLD_RATIO) {
                        const neededReserve = totalNetWorth * targetReserveRatio;
                        const transferAmount = neededReserve - p3_reserveCash;

                        if (transferAmount > 0 && p3_mainCash >= transferAmount) {
                            p3_mainCash -= transferAmount;
                            p3_reserveCash += transferAmount;
                            const details = `P3: Reserve replenished. Ratio (${(currentReserveRatio * 100).toFixed(1)}%) fell to/below 20%. Topped up to ${params.cashReservePercentage}%.`;
                            p3_tradeLog.push({
                                date: currentDate,
                                type: TradeType.REBALANCE,
                                stockPrice: currentPrice,
                                details,
                                cash: p3_mainCash + p3_reserveCash,
                                stockShares: p3_stockShares,
                                stockValue: p3_stockShares * currentPrice,
                                netWorth: totalNetWorth,
                                reserveCash: p3_reserveCash
                            });
                        }
                    }
                }
            }
        }

        // --- P1: Pure Options Logic ---
        {
            let p1_optionsProfit = 0;
            if (p1_openPosition) {
                const timeToExpiryMs = p1_openPosition.expiryDate.getTime() - currentDate.getTime();
                if (timeToExpiryMs > 0) {
                    const timeToExpiryYears = timeToExpiryMs / (1000 * 60 * 60 * 24 * 365.25);
                    const currentOptionPrice = blackScholesPrice(currentPrice, p1_openPosition.strike, timeToExpiryYears, dynamicIV, RISK_FREE_RATE, p1_openPosition.type);
                    const profitPercentage = (p1_openPosition.premium - currentOptionPrice) / p1_openPosition.premium;

                    if (profitPercentage >= EARLY_CLOSE_PROFIT_TARGET) {
                        const slippageAdjustedBuybackPrice = currentOptionPrice * (1 + params.slippage / 100);
                        const commission = params.optionFee * p1_openPosition.contractSize;
                        const buybackCost = (slippageAdjustedBuybackPrice * 100 * p1_openPosition.contractSize) + commission;
                        p1_cash -= buybackCost;
                        p1_optionsProfit += (p1_openPosition.premium * 100 * p1_openPosition.contractSize) - (currentOptionPrice * 100 * p1_openPosition.contractSize);
                        const details = `P1: Closed ${p1_openPosition.type} @ ${p1_openPosition.strike.toFixed(2)} early for ${(profitPercentage * 100).toFixed(0)}% profit.`;
                        tradeLog.push({ date: currentDate, type: TradeType.EARLY_CLOSE, stockPrice: currentPrice, details, cash: p1_cash, stockShares: p1_stockShares, stockValue: p1_stockShares * currentPrice, netWorth: p1_netWorth(), optionsProfit: p1_optionsProfit });
                        p1_openPosition = null;
                    }
                }

                if (p1_openPosition && currentDate >= p1_openPosition.expiryDate) {
                    if (p1_openPosition.type === 'PUT') {
                        if (currentPrice < p1_openPosition.strike) {
                            const cost = p1_openPosition.strike * 100 * p1_openPosition.contractSize;
                            if (p1_cash >= cost) {
                                p1_cash -= cost;
                                p1_stockShares += 100 * p1_openPosition.contractSize;
                                p1_assignmentPrice = p1_openPosition.strike;
                                tradeLog.push({ date: currentDate, type: TradeType.PUT_ASSIGNED, stockPrice: currentPrice, details: `P1: Assigned ${p1_openPosition.contractSize} PUT @ ${p1_openPosition.strike}.`, cash: p1_cash, stockShares: p1_stockShares, stockValue: p1_stockShares * currentPrice, netWorth: p1_netWorth() });
                            }
                        } else {
                            tradeLog.push({ date: currentDate, type: TradeType.PUT_EXPIRED, stockPrice: currentPrice, details: `P1: PUT @ ${p1_openPosition.strike} expired.`, cash: p1_cash, stockShares: p1_stockShares, stockValue: p1_stockShares * currentPrice, netWorth: p1_netWorth() });
                        }
                    } else { // CALL
                        if (currentPrice > p1_openPosition.strike) {
                            const sharesToSell = 100 * p1_openPosition.contractSize;
                            if (p1_stockShares >= sharesToSell) {
                                p1_cash += p1_openPosition.strike * sharesToSell;
                                p1_stockShares -= sharesToSell;
                                p1_assignmentPrice = null;
                                tradeLog.push({ date: currentDate, type: TradeType.CALL_ASSIGNED, stockPrice: currentPrice, details: `P1: Called away ${sharesToSell} shares @ ${p1_openPosition.strike}.`, cash: p1_cash, stockShares: p1_stockShares, stockValue: p1_stockShares * currentPrice, netWorth: p1_netWorth() });
                            }
                        } else {
                            tradeLog.push({ date: currentDate, type: TradeType.CALL_EXPIRED, stockPrice: currentPrice, details: `P1: CALL @ ${p1_openPosition.strike} expired.`, cash: p1_cash, stockShares: p1_stockShares, stockValue: p1_stockShares * currentPrice, netWorth: p1_netWorth() });
                        }
                    }
                    p1_openPosition = null;
                }
            }
            
            if (!p1_openPosition) {
                // Priority 1: Sell a covered call if conditions are met.
                if (p1_stockShares >= 100) {
                    const sellCallCondition = !params.sellCallAboveCostBasisOnly || p1_assignmentPrice === null || currentPrice > p1_assignmentPrice;
                    if (sellCallCondition) {
                        const strike = findStrikeForDelta(currentPrice, params.callDelta, params.callDTE / 365.0, dynamicIV, RISK_FREE_RATE, 'CALL');
                        const premium = blackScholesPrice(currentPrice, strike, params.callDTE / 365.0, dynamicIV, RISK_FREE_RATE, 'CALL');
                        const slippageAdjPremium = premium * (1 - params.slippage / 100);
                        const contractSize = Math.floor(p1_stockShares / 100);
                        const commission = params.optionFee * contractSize;
                        const totalPremium = (slippageAdjPremium * 100 * contractSize) - commission;
                        p1_cash += totalPremium;
                        p1_optionsProfit += totalPremium;
                        const expiry = new Date(currentDate.getTime() + params.callDTE * 24 * 60 * 60 * 1000);
                        p1_openPosition = { type: 'CALL', strike, expiryDate: findClosestTradingDay(expiry, dataSlice), premium, contractSize, entryDate: currentDate };
                        const details = `P1: Sold ${contractSize} CALL @ ${strike.toFixed(2)} for $${slippageAdjPremium.toFixed(2)}/share.`;
                        tradeLog.push({ date: currentDate, type: TradeType.SELL_CALL, stockPrice: currentPrice, details, cash: p1_cash, stockShares: p1_stockShares, stockValue: p1_stockShares * currentPrice, netWorth: p1_netWorth(), optionsProfit: p1_optionsProfit });
                    }
                }
                
                // Priority 2: If no call was sold (e.g., price below assignment), use available cash to sell a put.
                if (!p1_openPosition) {
                    const strike = findStrikeForDelta(currentPrice, -params.putDelta, params.putDTE / 365.0, dynamicIV, RISK_FREE_RATE, 'PUT');
                    const requiredCollateral = strike * 100;
                    const contractSize = Math.floor(p1_cash / requiredCollateral);
                    if (contractSize > 0) {
                        const premium = blackScholesPrice(currentPrice, strike, params.putDTE / 365.0, dynamicIV, RISK_FREE_RATE, 'PUT');
                        const slippageAdjPremium = premium * (1 - params.slippage / 100);
                        const commission = params.optionFee * contractSize;
                        const totalPremium = (slippageAdjPremium * 100 * contractSize) - commission;
                        p1_cash += totalPremium;
                        p1_optionsProfit += totalPremium;
                        const expiry = new Date(currentDate.getTime() + params.putDTE * 24 * 60 * 60 * 1000);
                        p1_openPosition = { type: 'PUT', strike, expiryDate: findClosestTradingDay(expiry, dataSlice), premium, contractSize, entryDate: currentDate };
                        const details = `P1: Sold ${contractSize} PUT @ ${strike.toFixed(2)} for $${slippageAdjPremium.toFixed(2)}/share.`;
                        tradeLog.push({ date: currentDate, type: TradeType.SELL_PUT, stockPrice: currentPrice, details, cash: p1_cash, stockShares: p1_stockShares, stockValue: p1_stockShares * currentPrice, netWorth: p1_netWorth(), optionsProfit: p1_optionsProfit });
                    }
                }
            }
        }
        
        // --- P3: Options + Cash Reserve Logic ---
        if (params.enableCashTactic) {
            const dropPercentage = (p3_allTimeHigh - currentPrice) / p3_allTimeHigh * 100;
            for (const trigger of p3_dipTriggers) {
                if (dropPercentage >= trigger.drop && !trigger.triggered) {
                    const amountToSpend = p3_initialReserveForDipBuy * (trigger.use / 100);
                    if (p3_reserveCash >= amountToSpend) {
                        p3_reserveCash -= amountToSpend;
                        const sharesBought = amountToSpend / currentPrice;
                        p3_stockShares += sharesBought;
                        trigger.triggered = true;
                        p3_inDip = true; 
                        p3_tradeLog.push({
                            date: currentDate, type: TradeType.DIP_BUY, stockPrice: currentPrice,
                            details: `P3: Dip buy @ ${dropPercentage.toFixed(1)}% drop. Bought ${sharesBought.toFixed(2)} shares.`,
                            cash: p3_mainCash + p3_reserveCash, stockShares: p3_stockShares, stockValue: p3_stockShares * currentPrice, netWorth: p3_netWorth(),
                            reserveCash: p3_reserveCash
                        });
                    }
                }
            }

            let p3_optionsProfit = 0;
            if (p3_openPosition) {
                 const timeToExpiryMs = p3_openPosition.expiryDate.getTime() - currentDate.getTime();
                 if (timeToExpiryMs > 0) {
                    const timeToExpiryYears = timeToExpiryMs / (1000 * 60 * 60 * 24 * 365.25);
                    const currentOptionPrice = blackScholesPrice(currentPrice, p3_openPosition.strike, timeToExpiryYears, dynamicIV, RISK_FREE_RATE, p3_openPosition.type);
                    const profitPercentage = (p3_openPosition.premium - currentOptionPrice) / p3_openPosition.premium;
                    if (p3_openPosition.premium > 0 && profitPercentage >= EARLY_CLOSE_PROFIT_TARGET) {
                        const buybackCost = (currentOptionPrice * (1 + params.slippage / 100) * 100 * p3_openPosition.contractSize) + (params.optionFee * p3_openPosition.contractSize);
                        p3_mainCash -= buybackCost;
                        p3_optionsProfit += (p3_openPosition.premium * 100 * p3_openPosition.contractSize) - (currentOptionPrice * 100 * p3_openPosition.contractSize);
                        const details = `P3: Closed ${p3_openPosition.type} @ ${p3_openPosition.strike.toFixed(2)} early for ${(profitPercentage * 100).toFixed(0)}% profit.`;
                        p3_tradeLog.push({ date: currentDate, type: TradeType.EARLY_CLOSE, stockPrice: currentPrice, details, cash: p3_mainCash + p3_reserveCash, stockShares: p3_stockShares, stockValue: p3_stockShares * currentPrice, netWorth: p3_netWorth(), optionsProfit: p3_optionsProfit, reserveCash: p3_reserveCash });
                        p3_openPosition = null;
                    }
                 }
                 if (p3_openPosition && currentDate >= p3_openPosition.expiryDate) {
                     if (p3_openPosition.type === 'PUT') {
                        if (currentPrice < p3_openPosition.strike) {
                            const cost = p3_openPosition.strike * 100 * p3_openPosition.contractSize;
                            if (p3_mainCash >= cost) {
                                p3_mainCash -= cost;
                                p3_stockShares += 100 * p3_openPosition.contractSize;
                                p3_assignmentPrice = p3_openPosition.strike;
                                p3_tradeLog.push({ date: currentDate, type: TradeType.PUT_ASSIGNED, stockPrice: currentPrice, details: `P3: Assigned ${p3_openPosition.contractSize} PUT @ ${p3_openPosition.strike}.`, cash: p3_mainCash + p3_reserveCash, stockShares: p3_stockShares, stockValue: p3_stockShares * currentPrice, netWorth: p3_netWorth(), reserveCash: p3_reserveCash });
                            }
                        } else {
                            p3_tradeLog.push({ date: currentDate, type: TradeType.PUT_EXPIRED, stockPrice: currentPrice, details: `P3: PUT @ ${p3_openPosition.strike} expired.`, cash: p3_mainCash + p3_reserveCash, stockShares: p3_stockShares, stockValue: p3_stockShares * currentPrice, netWorth: p3_netWorth(), reserveCash: p3_reserveCash });
                        }
                    } else { // CALL
                        if (currentPrice > p3_openPosition.strike) {
                            const sharesToSell = 100 * p3_openPosition.contractSize;
                            if (p3_stockShares >= sharesToSell) {
                                p3_mainCash += p3_openPosition.strike * sharesToSell;
                                p3_stockShares -= sharesToSell;
                                p3_assignmentPrice = null;
                                p3_tradeLog.push({ date: currentDate, type: TradeType.CALL_ASSIGNED, stockPrice: currentPrice, details: `P3: Called away ${sharesToSell} shares @ ${p3_openPosition.strike}.`, cash: p3_mainCash + p3_reserveCash, stockShares: p3_stockShares, stockValue: p3_stockShares * currentPrice, netWorth: p3_netWorth(), reserveCash: p3_reserveCash });
                            }
                        } else {
                             p3_tradeLog.push({ date: currentDate, type: TradeType.CALL_EXPIRED, stockPrice: currentPrice, details: `P3: CALL @ ${p3_openPosition.strike} expired.`, cash: p3_mainCash + p3_reserveCash, stockShares: p3_stockShares, stockValue: p3_stockShares * currentPrice, netWorth: p3_netWorth(), reserveCash: p3_reserveCash });
                        }
                    }
                    p3_openPosition = null;
                 }
            }

            if (!p3_openPosition) {
                if (p3_stockShares >= 100) {
                    const sellCallCondition = !params.sellCallAboveCostBasisOnly || p3_assignmentPrice === null || currentPrice > p3_assignmentPrice;
                    if (sellCallCondition) {
                        const strike = findStrikeForDelta(currentPrice, params.callDelta, params.callDTE / 365.0, dynamicIV, RISK_FREE_RATE, 'CALL');
                        const premium = blackScholesPrice(currentPrice, strike, params.callDTE / 365.0, dynamicIV, RISK_FREE_RATE, 'CALL');
                        const slippageAdjPremium = premium * (1 - params.slippage / 100);
                        const contractSize = Math.floor(p3_stockShares / 100);
                        const totalPremium = (slippageAdjPremium * 100 * contractSize) - (params.optionFee * contractSize);
                        p3_mainCash += totalPremium;
                        p3_optionsProfit += totalPremium;
                        const expiry = new Date(currentDate.getTime() + params.callDTE * 24 * 60 * 60 * 1000);
                        p3_openPosition = { type: 'CALL', strike, expiryDate: findClosestTradingDay(expiry, dataSlice), premium, contractSize, entryDate: currentDate };
                        const details = `P3: Sold ${contractSize} CALL @ ${strike.toFixed(2)} for $${slippageAdjPremium.toFixed(2)}/share.`;
                        p3_tradeLog.push({ date: currentDate, type: TradeType.SELL_CALL, stockPrice: currentPrice, details, cash: p3_mainCash + p3_reserveCash, stockShares: p3_stockShares, stockValue: p3_stockShares * currentPrice, netWorth: p3_netWorth(), optionsProfit: p3_optionsProfit, reserveCash: p3_reserveCash });
                    }
                }
                
                if (!p3_openPosition) {
                    const strike = findStrikeForDelta(currentPrice, -params.putDelta, params.putDTE / 365.0, dynamicIV, RISK_FREE_RATE, 'PUT');
                    const requiredCollateral = strike * 100;
                    const contractSize = Math.floor(p3_mainCash / requiredCollateral);
                    if (contractSize > 0) {
                        const premium = blackScholesPrice(currentPrice, strike, params.putDTE / 365.0, dynamicIV, RISK_FREE_RATE, 'PUT');
                        const slippageAdjPremium = premium * (1 - params.slippage / 100);
                        const totalPremium = (slippageAdjPremium * 100 * contractSize) - (params.optionFee * contractSize);
                        p3_mainCash += totalPremium;
                        p3_optionsProfit += totalPremium;
                        const expiry = new Date(currentDate.getTime() + params.putDTE * 24 * 60 * 60 * 1000);
                        p3_openPosition = { type: 'PUT', strike, expiryDate: findClosestTradingDay(expiry, dataSlice), premium, contractSize, entryDate: currentDate };
                        const details = `P3: Sold ${contractSize} PUT @ ${strike.toFixed(2)} for $${slippageAdjPremium.toFixed(2)}/share.`;
                        p3_tradeLog.push({ date: currentDate, type: TradeType.SELL_PUT, stockPrice: currentPrice, details, cash: p3_mainCash + p3_reserveCash, stockShares: p3_stockShares, stockValue: p3_stockShares * currentPrice, netWorth: p3_netWorth(), optionsProfit: p3_optionsProfit, reserveCash: p3_reserveCash });
                    }
                }
            }
        }

        // --- P4: B&H + Cash Reserve Logic ---
        if (params.enableCashTactic) {
            const dropPercentage = (p4_allTimeHigh - currentPrice) / p4_allTimeHigh * 100;
            for (const trigger of p4_dipTriggers) {
                if (dropPercentage >= trigger.drop && !trigger.triggered) {
                    const amountToSpend = p4_initialReserveForDipBuy * (trigger.use / 100);
                    if (p4_cashReserve >= amountToSpend) {
                        p4_cashReserve -= amountToSpend;
                        const sharesBought = amountToSpend / currentPrice;
                        p4_stockShares += sharesBought;
                        trigger.triggered = true;
                        p4_inDip = true;
                        p4_tradeLog.push({
                            date: currentDate,
                            type: TradeType.DIP_BUY,
                            stockPrice: currentPrice,
                            details: `P4: Dip buy @ ${dropPercentage.toFixed(1)}% drop. Bought ${sharesBought.toFixed(2)} shares.`,
                            cash: p4_cashReserve,
                            stockShares: p4_stockShares,
                            stockValue: p4_stockShares * currentPrice,
                            netWorth: (p4_stockShares * currentPrice) + p4_cashReserve,
                            reserveCash: p4_cashReserve,
                        });
                    }
                }
            }
        }
        
        // --- Record Snapshots ---
        optionsHistory.push({ date: currentDate, cash: p1_cash, stockShares: p1_stockShares, stockPrice: currentPrice, stockValue: p1_stockShares * currentPrice, netWorth: p1_netWorth() });

        const p2_netWorth = p2_stockShares * currentPrice;
        buyAndHoldHistory.push({ date: currentDate, cash: 0, stockShares: p2_stockShares, stockPrice: currentPrice, stockValue: p2_netWorth, netWorth: p2_netWorth });

        if(params.enableCashTactic) {
            optionsWithCashHistory.push({ date: currentDate, cash: p3_mainCash, reserveCash: p3_reserveCash, stockShares: p3_stockShares, stockPrice: currentPrice, stockValue: p3_stockShares * currentPrice, netWorth: p3_netWorth() });
            
            const p4_stockValue = p4_stockShares * currentPrice;
            const p4_netWorth = p4_cashReserve + p4_stockValue;
            buyAndHoldWithCashHistory.push({ date: currentDate, cash: p4_cashReserve, stockShares: p4_stockShares, stockPrice: currentPrice, stockValue: p4_stockValue, netWorth: p4_netWorth });
        }
    }

    const finalDataPoint = dataSlice[dataSlice.length - 1];
    const finalPrice = finalDataPoint.price;
    const finalDate = finalDataPoint.date;

    const finalP1NetWorth = optionsHistory.length > 0 ? optionsHistory[optionsHistory.length - 1].netWorth : initialCapitalUSD;
    tradeLog.push({ date: finalDate, type: TradeType.SIMULATION_END, stockPrice: finalPrice, details: 'End of simulation period.', cash: p1_cash, stockShares: p1_stockShares, stockValue: p1_stockShares * finalPrice, netWorth: finalP1NetWorth });
    
    if (p2_tradeLog.length > 0) {
        const finalP2NetWorth = buyAndHoldHistory.length > 0 ? buyAndHoldHistory[buyAndHoldHistory.length - 1].netWorth : initialCapitalUSD;
        p2_tradeLog.push({ date: finalDate, type: TradeType.SIMULATION_END, stockPrice: finalPrice, details: 'End of simulation period.', cash: 0, stockShares: p2_stockShares, stockValue: p2_stockShares * finalPrice, netWorth: finalP2NetWorth });
    }

    if (params.enableCashTactic && optionsWithCashHistory.length > 0) {
        const finalP3NetWorth = optionsWithCashHistory[optionsWithCashHistory.length - 1].netWorth;
        p3_tradeLog.push({ date: finalDate, type: TradeType.SIMULATION_END, stockPrice: finalPrice, details: 'End of simulation period.', cash: p3_mainCash + p3_reserveCash, stockShares: p3_stockShares, stockValue: p3_stockShares * finalPrice, netWorth: finalP3NetWorth, reserveCash: p3_reserveCash });
    }
    
    if (params.enableCashTactic && p4_tradeLog.length > 0) {
        const finalP4NetWorth = buyAndHoldWithCashHistory.length > 0 ? buyAndHoldWithCashHistory[buyAndHoldWithCashHistory.length - 1].netWorth : initialCapitalUSD;
        // FIX: Corrected variable name from p4_reserveCash to p4_cashReserve.
        p4_tradeLog.push({ date: finalDate, type: TradeType.SIMULATION_END, stockPrice: finalPrice, details: 'End of simulation period.', cash: p4_cashReserve, stockShares: p4_stockShares, stockValue: p4_stockShares * finalPrice, netWorth: finalP4NetWorth, reserveCash: p4_cashReserve });
    }
    
    const annualSummaries = calculateAnnualSummaries(optionsHistory, tradeLog, initialCapitalUSD, params.exchangeRate);
    const optionsWithCashAnnualSummaries = params.enableCashTactic 
        ? calculateAnnualSummaries(optionsWithCashHistory, p3_tradeLog, p3_initialCapitalUSD, params.exchangeRate)
        : [];

    return {
        optionsHistory, 
        buyAndHoldHistory, 
        optionsWithCashHistory, 
        buyAndHoldWithCashHistory, 
        tradeLog, 
        optionsWithCashTradeLog: p3_tradeLog, 
        buyAndHoldTradeLog: p2_tradeLog,
        buyAndHoldWithCashTradeLog: p4_tradeLog,
        annualSummaries, 
        optionsWithCashAnnualSummaries,
        priceHistory: dataSlice,
        totalTwdInvested,
        exchangeRate: params.exchangeRate,
        analytics: calculateAnalytics(optionsHistory, p1_totalInvestedUSD, p1_cashflows),
        buyAndHoldAnalytics: calculateAnalytics(buyAndHoldHistory, p2_totalInvestedUSD, p2_cashflows),
        optionsWithCashAnalytics: calculateAnalytics(optionsWithCashHistory, p3_totalInvestedUSD, p3_cashflows),
        buyAndHoldWithCashAnalytics: calculateAnalytics(buyAndHoldWithCashHistory, p4_totalInvestedUSD, p4_cashflows)
    };
};
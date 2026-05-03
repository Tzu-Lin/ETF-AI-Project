// FIX: Removed self-import of DailyPrice which caused a declaration conflict.
export interface DailyPrice {
  date: Date;
  price: number;
}

export interface PortfolioSnapshot {
  date: Date;
  cash: number;
  stockShares: number;
  stockPrice: number;
  stockValue: number;
  netWorth: number;
  reserveCash?: number; // Optional reserve cash for combined strategies
}

export enum TradeType {
  SELL_PUT = 'Sell Put',
  SELL_CALL = 'Sell Call',
  PUT_ASSIGNED = 'Put Assigned',
  CALL_ASSIGNED = 'Call Assigned',
  PUT_EXPIRED = 'Put Expired',
  CALL_EXPIRED = 'Call Expired',
  DIP_BUY = 'Dip Buy',
  RECOVERY_SELL = 'Recovery Sell',
  REBALANCE = 'Rebalance',
  RECURRING_INVESTMENT = '定期定額投入',
  EARLY_CLOSE = 'Early Close',
  SIMULATION_END = 'Simulation End',
}

export interface TradeLog {
  date: Date;
  type: TradeType;
  stockPrice: number;
  details: string;
  cash: number; // For pure strategy, this is all cash. For cash strategy, this is TOTAL cash (main + reserve).
  stockShares: number;
  stockValue: number;
  netWorth: number;
  optionsProfit?: number;
  stockProfit?: number;
  reserveCash?: number; // Cash set aside for dip buys. Only for cash-enabled strategies.
}

export interface AnnualSummary {
  year: number;
  startNetWorth: number;
  endNetWorth: number;
  annualReturn: number;
  totalOptionsProfit: number;
  totalStockProfit: number;
}

export interface StrategyParameters {
  startDate: string;
  endDate: string;
  initialCapital: number; // This is now in TWD
  putDTE: number;
  putDelta: number;
  callDTE: number;
  callDelta: number;
  sellCallAboveCostBasisOnly: boolean; // New parameter to control Call selling logic
  enableCashTactic: boolean; // This now controls the n% cash reserve for both benchmarks
  cashReservePercentage: number;
  dipTriggers: {
    drop: number;
    use: number;
  }[];
  // Recurring Investment
  enableRecurringInvestment: boolean;
  monthlyInvestmentTWD: number;
  exchangeRate: number;
  // Realism parameters
  optionFee: number;
  slippage: number;
  ivAdjustmentFactor: number; // New parameter for dynamic IV
}

export interface Analytics {
    totalReturn: number;
    cagr: number;
    sharpeRatio: number; // Placeholder
    maxDrawdown: number;
    drawdownPeakUSD: number;
    drawdownTroughUSD: number;
    drawdownPeakDate: Date;
    drawdownTroughDate: Date;
}

export interface BacktestResult {
    // Four strategy histories
    optionsHistory: PortfolioSnapshot[];
    buyAndHoldHistory: PortfolioSnapshot[];
    optionsWithCashHistory: PortfolioSnapshot[];
    buyAndHoldWithCashHistory: PortfolioSnapshot[];

    tradeLog: TradeLog[]; // Log for the primary options strategy
    optionsWithCashTradeLog: TradeLog[]; // Log for the options strategy with cash reserve
    buyAndHoldTradeLog: TradeLog[]; // Log for the pure Buy & Hold strategy
    buyAndHoldWithCashTradeLog: TradeLog[]; // Log for the Buy & Hold strategy with cash reserve

    annualSummaries: AnnualSummary[]; // Summary for the primary options strategy (NOW IN TWD)
    optionsWithCashAnnualSummaries: AnnualSummary[]; // Summary for the options with cash strategy (NOW IN TWD)
    priceHistory: DailyPrice[];

    // Analytics for each of the four strategies
    analytics: Analytics;
    buyAndHoldAnalytics: Analytics;
    optionsWithCashAnalytics: Analytics;
    buyAndHoldWithCashAnalytics: Analytics;

    // Total cost basis
    totalTwdInvested: number;
    exchangeRate: number;
}
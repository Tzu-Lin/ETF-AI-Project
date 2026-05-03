// utils/blackScholes.ts

// Standard normal distribution CDF approximation
const normCdf = (x: number): number => {
    // Correct approximation for standard normal CDF
    let z = x;
    let t = 1 / (1 + 0.2316419 * Math.abs(z));
    let d = 0.3989423 * Math.exp(-z * z / 2);
    let prob = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    if (z > 0) {
        return 1 - prob;
    }
    return prob;
};


/**
 * Calculates option price using Black-Scholes model.
 * @param spot Current stock price
 * @param strike Strike price
 * @param timeToExpiryYears Time to expiry in years
 * @param iv Implied volatility (e.g., 0.60)
 * @param riskFreeRate Risk-free interest rate (e.g., 0.05 for 5%)
 * @param type 'CALL' or 'PUT'
 * @returns The theoretical price of the option
 */
export const blackScholesPrice = (
    spot: number,
    strike: number,
    timeToExpiryYears: number,
    iv: number,
    riskFreeRate: number,
    type: 'CALL' | 'PUT'
): number => {
    if (timeToExpiryYears <= 0) {
        return type === 'CALL' ? Math.max(0, spot - strike) : Math.max(0, strike - spot);
    }
    if (iv <= 0) {
        return type === 'CALL' ? Math.max(0, spot - strike) : Math.max(0, strike - spot);
    }


    const d1 = (Math.log(spot / strike) + (riskFreeRate + (iv * iv) / 2) * timeToExpiryYears) / (iv * Math.sqrt(timeToExpiryYears));
    const d2 = d1 - iv * Math.sqrt(timeToExpiryYears);

    if (type === 'CALL') {
        return spot * normCdf(d1) - strike * Math.exp(-riskFreeRate * timeToExpiryYears) * normCdf(d2);
    } else { // PUT
        return strike * Math.exp(-riskFreeRate * timeToExpiryYears) * normCdf(-d2) - spot * normCdf(-d1);
    }
};

/**
 * Calculates option delta using Black-Scholes model.
 */
export const blackScholesDelta = (
    spot: number,
    strike: number,
    timeToExpiryYears: number,
    iv: number,
    riskFreeRate: number,
    type: 'CALL' | 'PUT'
): number => {
    if (timeToExpiryYears <= 0 || iv <= 0) {
        if (type === 'CALL') {
            return spot > strike ? 1 : 0;
        } else { // PUT
            return spot < strike ? -1 : 0;
        }
    }

    const d1 = (Math.log(spot / strike) + (riskFreeRate + (iv * iv) / 2) * timeToExpiryYears) / (iv * Math.sqrt(timeToExpiryYears));
    
    if (type === 'CALL') {
        return normCdf(d1);
    } else { // PUT
        return normCdf(d1) - 1;
    }
};

/**
 * Finds the strike price for a given delta.
 */
export const findStrikeForDelta = (
    spot: number,
    targetDelta: number,
    timeToExpiryYears: number,
    iv: number,
    riskFreeRate: number,
    type: 'CALL' | 'PUT'
): number => {
    // Use an iterative search (bisection method) to find the strike
    let lowStrike = spot * 0.2; // Start wider
    let highStrike = spot * 2.5;
    let midStrike = spot;
    let calculatedDelta = 0;

    for (let i = 0; i < 50; i++) { // 50 iterations for precision
        midStrike = (lowStrike + highStrike) / 2;
        calculatedDelta = blackScholesDelta(spot, midStrike, timeToExpiryYears, iv, riskFreeRate, type);

        if (Math.abs(calculatedDelta - targetDelta) < 0.001) {
            break; // Exit loop if close enough
        }

        if (type === 'PUT') {
            // For puts, delta becomes more negative as strike increases.
            // Target is ~ -0.3
            if (calculatedDelta < targetDelta) {
                // calculatedDelta is too negative (e.g., -0.4). Strike is too high. Lower the upper bound.
                highStrike = midStrike;
            } else {
                // calculatedDelta is not negative enough (e.g., -0.2). Strike is too low. Raise the lower bound.
                lowStrike = midStrike;
            }
        } else { // CALL
            // For calls, delta decreases as strike increases.
            if (calculatedDelta > targetDelta) {
                lowStrike = midStrike;
            } else {
                highStrike = midStrike;
            }
        }
    }
    // Round to nearest $0.50 to simulate real strike prices
    return Math.round(midStrike * 2) / 2;
};

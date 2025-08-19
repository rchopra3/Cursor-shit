-- Financial Portfolio Analysis Dashboard - Advanced SQL Queries
-- This file contains complex SQL queries for financial analysis

-- Query 1: Find the best and worst performing stock each month
-- This query calculates monthly returns and ranks stocks by performance
WITH monthly_returns AS (
    SELECT 
        strftime('%Y-%m', date) as month,
        ticker,
        company_name,
        AVG(daily_return) as avg_daily_return,
        SUM(daily_return) as total_return,
        COUNT(*) as trading_days
    FROM calculated_metrics cm
    JOIN assets a ON cm.ticker = a.ticker
    WHERE daily_return IS NOT NULL
    GROUP BY strftime('%Y-%m', date), ticker
),
ranked_monthly AS (
    SELECT 
        month,
        ticker,
        company_name,
        total_return,
        avg_daily_return,
        trading_days,
        ROW_NUMBER() OVER (PARTITION BY month ORDER BY total_return DESC) as best_rank,
        ROW_NUMBER() OVER (PARTITION BY month ORDER BY total_return ASC) as worst_rank
    FROM monthly_returns
)
SELECT 
    month,
    MAX(CASE WHEN best_rank = 1 THEN ticker END) as best_performer,
    MAX(CASE WHEN best_rank = 1 THEN company_name END) as best_company,
    MAX(CASE WHEN best_rank = 1 THEN total_return END) as best_return,
    MAX(CASE WHEN worst_rank = 1 THEN ticker END) as worst_performer,
    MAX(CASE WHEN worst_rank = 1 THEN company_name END) as worst_company,
    MAX(CASE WHEN worst_rank = 1 THEN total_return END) as worst_return
FROM ranked_monthly
GROUP BY month
ORDER BY month DESC;

-- Query 2: Calculate correlation of daily returns between different pairs of stocks
-- This query computes rolling correlations between stock pairs
WITH stock_returns AS (
    SELECT 
        date,
        ticker,
        daily_return
    FROM calculated_metrics
    WHERE daily_return IS NOT NULL
),
correlation_pairs AS (
    SELECT 
        a.ticker as stock1,
        b.ticker as stock2,
        a.daily_return as return1,
        b.daily_return as return2,
        a.date
    FROM stock_returns a
    JOIN stock_returns b ON a.date = b.date AND a.ticker < b.ticker
),
correlation_calc AS (
    SELECT 
        stock1,
        stock2,
        COUNT(*) as n,
        AVG(return1) as mean1,
        AVG(return2) as mean2,
        SUM((return1 - AVG(return1) OVER (PARTITION BY stock1, stock2)) * 
            (return2 - AVG(return2) OVER (PARTITION BY stock1, stock2))) as covariance,
        SUM(POWER(return1 - AVG(return1) OVER (PARTITION BY stock1, stock2), 2)) as var1,
        SUM(POWER(return2 - AVG(return2) OVER (PARTITION BY stock1, stock2), 2)) as var2
    FROM correlation_pairs
    GROUP BY stock1, stock2
)
SELECT 
    stock1,
    stock2,
    n as data_points,
    ROUND(
        CASE 
            WHEN var1 * var2 > 0 THEN covariance / SQRT(var1 * var2)
            ELSE NULL
        END, 4
    ) as correlation_coefficient
FROM correlation_calc
WHERE n >= 30  -- Only show correlations with sufficient data
ORDER BY ABS(correlation_coefficient) DESC;

-- Query 3: Calculate hypothetical portfolio value over time
-- This query assumes equal investment in each asset and tracks portfolio growth
WITH portfolio_daily AS (
    SELECT 
        date,
        COUNT(DISTINCT ticker) as num_assets,
        AVG(close_price) as avg_price,
        SUM(close_price) as total_value,
        AVG(daily_return) as portfolio_return
    FROM daily_prices dp
    JOIN assets a ON dp.ticker = a.ticker
    GROUP BY date
),
portfolio_cumulative AS (
    SELECT 
        date,
        num_assets,
        avg_price,
        total_value,
        portfolio_return,
        -- Calculate cumulative return starting from 1.0
        EXP(SUM(LN(1 + portfolio_return)) OVER (ORDER BY date) as cumulative_return,
        -- Calculate portfolio value assuming $10,000 initial investment
        10000 * EXP(SUM(LN(1 + portfolio_return)) OVER (ORDER BY date) as portfolio_value
    FROM portfolio_daily
    WHERE portfolio_return IS NOT NULL
)
SELECT 
    date,
    num_assets,
    ROUND(avg_price, 2) as avg_stock_price,
    ROUND(total_value, 2) as total_stock_value,
    ROUND(portfolio_return * 100, 2) as daily_return_pct,
    ROUND((cumulative_return - 1) * 100, 2) as total_return_pct,
    ROUND(portfolio_value, 2) as portfolio_value_10k
FROM portfolio_cumulative
ORDER BY date DESC
LIMIT 100;

-- Query 4: Risk-adjusted performance metrics by stock
-- This query calculates Sharpe ratio and other risk metrics for each stock
WITH stock_metrics AS (
    SELECT 
        ticker,
        company_name,
        sector,
        COUNT(*) as trading_days,
        AVG(daily_return) as mean_return,
        STDDEV(daily_return) as std_return,
        MIN(daily_return) as min_return,
        MAX(daily_return) as max_return,
        AVG(volatility_30d) as avg_volatility
    FROM calculated_metrics cm
    JOIN assets a ON cm.ticker = a.ticker
    WHERE daily_return IS NOT NULL
    GROUP BY ticker
)
SELECT 
    ticker,
    company_name,
    sector,
    trading_days,
    ROUND(mean_return * 100, 2) as mean_return_pct,
    ROUND(std_return * 100, 2) as std_return_pct,
    ROUND(min_return * 100, 2) as min_return_pct,
    ROUND(max_return * 100, 2) as max_return_pct,
    ROUND(avg_volatility * 100, 2) as avg_volatility_pct,
    -- Sharpe ratio (assuming risk-free rate of 0.02)
    ROUND((mean_return - 0.02/252) / std_return, 2) as sharpe_ratio
FROM stock_metrics
ORDER BY sharpe_ratio DESC;

-- Query 5: Sector performance comparison
-- This query analyzes performance by sector
SELECT 
    sector,
    COUNT(DISTINCT ticker) as num_stocks,
    ROUND(AVG(mean_return) * 100, 2) as avg_return_pct,
    ROUND(AVG(std_return) * 100, 2) as avg_volatility_pct,
    ROUND(AVG(sharpe_ratio), 2) as avg_sharpe_ratio
FROM (
    SELECT 
        ticker,
        sector,
        AVG(daily_return) as mean_return,
        STDDEV(daily_return) as std_return,
        (AVG(daily_return) - 0.02/252) / STDDEV(daily_return) as sharpe_ratio
    FROM calculated_metrics cm
    JOIN assets a ON cm.ticker = a.ticker
    WHERE daily_return IS NOT NULL
    GROUP BY ticker, sector
)
GROUP BY sector
ORDER BY avg_sharpe_ratio DESC;

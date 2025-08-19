-- Financial Portfolio Analysis Dashboard Database Schema
-- This schema creates a normalized database structure for storing stock data and calculated metrics

-- Drop existing tables if they exist (for clean setup)
DROP TABLE IF EXISTS calculated_metrics;
DROP TABLE IF EXISTS daily_prices;
DROP TABLE IF EXISTS assets;

-- Assets table: stores basic information about each stock
CREATE TABLE assets (
    ticker VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(100) NOT NULL,
    sector VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Daily prices table: stores OHLCV data for each stock
CREATE TABLE daily_prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    open_price DECIMAL(10,4),
    high_price DECIMAL(10,4),
    low_price DECIMAL(10,4),
    close_price DECIMAL(10,4),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ticker) REFERENCES assets(ticker),
    UNIQUE(date, ticker)
);

-- Calculated metrics table: stores derived financial metrics
CREATE TABLE calculated_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    daily_return DECIMAL(8,6),
    volatility_30d DECIMAL(8,6),
    moving_avg_50d DECIMAL(10,4),
    sharpe_ratio DECIMAL(8,6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (ticker) REFERENCES assets(ticker),
    UNIQUE(date, ticker)
);

-- Create indexes for better query performance
CREATE INDEX idx_daily_prices_date ON daily_prices(date);
CREATE INDEX idx_daily_prices_ticker ON daily_prices(ticker);
CREATE INDEX idx_calculated_metrics_date ON calculated_metrics(date);
CREATE INDEX idx_calculated_metrics_ticker ON calculated_metrics(ticker);

-- Insert sample assets data
INSERT INTO assets (ticker, company_name, sector) VALUES
    ('AAPL', 'Apple Inc.', 'Technology'),
    ('MSFT', 'Microsoft Corporation', 'Technology'),
    ('GOOGL', 'Alphabet Inc.', 'Technology'),
    ('AMZN', 'Amazon.com Inc.', 'Consumer Cyclical'),
    ('META', 'Meta Platforms Inc.', 'Technology');

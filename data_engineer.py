#!/usr/bin/env python3
"""
Financial Portfolio Analysis Dashboard - Data Engineering Module

This module handles:
1. Fetching historical stock data using yfinance
2. Calculating financial metrics (returns, volatility, moving averages)
3. Storing data in SQLite database
4. Data validation and cleaning

Author: Portfolio Analysis Team
Date: 2024
"""

import sqlite3
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Tuple
import warnings

# Suppress yfinance warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialDataEngineer:
    """Main class for financial data engineering operations."""
    
    def __init__(self, db_path: str = 'portfolio.db'):
        """
        Initialize the data engineer.
        
        Args:
            db_path (str): Path to SQLite database
        """
        self.db_path = db_path
        self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        self.conn = None
        
    def connect_db(self) -> None:
        """Establish database connection."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            logger.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
    
    def close_db(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def initialize_database(self) -> None:
        """Initialize database with schema and sample assets."""
        try:
            with open('schema.sql', 'r') as f:
                schema_sql = f.read()
            
            cursor = self.conn.cursor()
            cursor.executescript(schema_sql)
            self.conn.commit()
            logger.info("Database initialized successfully")
            
        except FileNotFoundError:
            logger.error("Schema file not found. Please ensure schema.sql exists.")
            raise
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise
    
    def fetch_stock_data(self, ticker: str, period: str = '2y') -> pd.DataFrame:
        """
        Fetch historical stock data for a given ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            period (str): Time period for data (e.g., '2y' for 2 years)
            
        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        try:
            logger.info(f"Fetching data for {ticker}")
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if data.empty:
                logger.warning(f"No data retrieved for {ticker}")
                return pd.DataFrame()
            
            # Reset index to make date a column
            data.reset_index(inplace=True)
            data['ticker'] = ticker
            
            # Rename columns to match database schema
            data.rename(columns={
                'Date': 'date',
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Close': 'close_price',
                'Volume': 'volume'
            }, inplace=True)
            
            # Convert date to string format for SQLite
            data['date'] = data['date'].dt.strftime('%Y-%m-%d')
            
            logger.info(f"Retrieved {len(data)} records for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def calculate_financial_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate financial metrics from price data.
        
        Args:
            data (pd.DataFrame): OHLCV data
            
        Returns:
            pd.DataFrame: Data with calculated metrics
        """
        if data.empty:
            return pd.DataFrame()
        
        try:
            # Calculate daily returns
            data['daily_return'] = data['close_price'].pct_change()
            
            # Calculate 30-day rolling volatility
            data['volatility_30d'] = data['daily_return'].rolling(window=30).std()
            
            # Calculate 50-day moving average
            data['moving_avg_50d'] = data['close_price'].rolling(window=50).mean()
            
            # Calculate Sharpe ratio (assuming risk-free rate of 2% annually)
            risk_free_rate = 0.02 / 252  # Daily risk-free rate
            data['sharpe_ratio'] = (data['daily_return'] - risk_free_rate) / data['volatility_30d']
            
            # Remove rows with NaN values (first 50 days due to moving average calculation)
            data = data.dropna()
            
            logger.info(f"Calculated financial metrics for {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return pd.DataFrame()
    
    def store_daily_prices(self, data: pd.DataFrame) -> None:
        """
        Store daily price data in database.
        
        Args:
            data (pd.DataFrame): OHLCV data with calculated metrics
        """
        if data.empty:
            return
        
        try:
            cursor = self.conn.cursor()
            
            # Prepare data for insertion
            price_data = data[['date', 'ticker', 'open_price', 'high_price', 
                              'low_price', 'close_price', 'volume']].values.tolist()
            
            # Use parameterized query to prevent SQL injection
            insert_query = """
                INSERT OR REPLACE INTO daily_prices 
                (date, ticker, open_price, high_price, low_price, close_price, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            cursor.executemany(insert_query, price_data)
            self.conn.commit()
            
            logger.info(f"Stored {len(price_data)} daily price records")
            
        except sqlite3.Error as e:
            logger.error(f"Error storing daily prices: {e}")
            raise
    
    def store_calculated_metrics(self, data: pd.DataFrame) -> None:
        """
        Store calculated financial metrics in database.
        
        Args:
            data (pd.DataFrame): Data with calculated metrics
        """
        if data.empty:
            return
        
        try:
            cursor = self.conn.cursor()
            
            # Prepare metrics data for insertion
            metrics_data = data[['date', 'ticker', 'daily_return', 'volatility_30d', 
                                'moving_avg_50d', 'sharpe_ratio']].values.tolist()
            
            # Use parameterized query to prevent SQL injection
            insert_query = """
                INSERT OR REPLACE INTO calculated_metrics 
                (date, ticker, daily_return, volatility_30d, moving_avg_50d, sharpe_ratio)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            
            cursor.executemany(insert_query, metrics_data)
            self.conn.commit()
            
            logger.info(f"Stored {len(metrics_data)} calculated metrics records")
            
        except sqlite3.Error as e:
            logger.error(f"Error storing calculated metrics: {e}")
            raise
    
    def process_all_tickers(self) -> None:
        """Process all tickers: fetch data, calculate metrics, and store in database."""
        logger.info("Starting data processing for all tickers")
        
        try:
            for ticker in self.tickers:
                logger.info(f"Processing {ticker}...")
                
                # Fetch data
                data = self.fetch_stock_data(ticker)
                if data.empty:
                    continue
                
                # Calculate metrics
                data_with_metrics = self.calculate_financial_metrics(data)
                if data_with_metrics.empty:
                    continue
                
                # Store data
                self.store_daily_prices(data_with_metrics)
                self.store_calculated_metrics(data_with_metrics)
                
                logger.info(f"Successfully processed {ticker}")
                
        except Exception as e:
            logger.error(f"Error processing tickers: {e}")
            raise
        
        logger.info("Data processing completed for all tickers")
    
    def get_data_summary(self) -> Dict[str, int]:
        """
        Get summary of data in database.
        
        Returns:
            Dict[str, int]: Summary statistics
        """
        try:
            cursor = self.conn.cursor()
            
            # Count records in each table
            cursor.execute("SELECT COUNT(*) FROM assets")
            asset_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM daily_prices")
            price_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM calculated_metrics")
            metrics_count = cursor.fetchone()[0]
            
            # Get date range
            cursor.execute("SELECT MIN(date), MAX(date) FROM daily_prices")
            date_range = cursor.fetchone()
            
            summary = {
                'assets': asset_count,
                'daily_prices': price_count,
                'calculated_metrics': metrics_count,
                'date_range': date_range
            }
            
            return summary
            
        except sqlite3.Error as e:
            logger.error(f"Error getting data summary: {e}")
            return {}

def main():
    """Main function to run the data engineering pipeline."""
    logger.info("Starting Financial Portfolio Analysis Data Engineering Pipeline")
    
    # Initialize data engineer
    engineer = FinancialDataEngineer()
    
    try:
        # Connect to database
        engineer.connect_db()
        
        # Initialize database (this will create tables and insert sample assets)
        engineer.initialize_database()
        
        # Process all tickers
        engineer.process_all_tickers()
        
        # Get and display summary
        summary = engineer.get_data_summary()
        logger.info("Data Engineering Pipeline Summary:")
        logger.info(f"Assets: {summary.get('assets', 0)}")
        logger.info(f"Daily Prices: {summary.get('daily_prices', 0)}")
        logger.info(f"Calculated Metrics: {summary.get('calculated_metrics', 0)}")
        if summary.get('date_range'):
            logger.info(f"Date Range: {summary['date_range'][0]} to {summary['date_range'][1]}")
        
        logger.info("Data Engineering Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    
    finally:
        # Always close database connection
        engineer.close_db()

if __name__ == "__main__":
    main()

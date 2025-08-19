#!/usr/bin/env python3
"""
Financial Portfolio Analysis Dashboard - Analysis Script

This script provides comprehensive analysis of a portfolio of technology stocks including:
- Data loading and exploration
- Performance analysis and risk metrics
- Correlation analysis
- Portfolio construction and backtesting
- Visualization dashboard
- Export of results

Author: Portfolio Analysis Team
Date: 2024
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure plotting style
plt.style.use('default')
sns.set_palette("husl")

class FinancialPortfolioAnalyzer:
    """Main class for financial portfolio analysis."""
    
    def __init__(self, db_path: str = 'portfolio.db'):
        """
        Initialize the portfolio analyzer.
        
        Args:
            db_path (str): Path to SQLite database
        """
        self.db_path = db_path
        self.conn = None
        self.assets_df = None
        self.prices_df = None
        self.performance_df = None
        
    def connect_database(self):
        """Establish connection to the portfolio database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            print("✅ Successfully connected to portfolio.db")
            return True
        except sqlite3.Error as e:
            print(f"❌ Database connection error: {e}")
            return False
    
    def close_database(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("✅ Database connection closed")
    
    def load_data(self):
        """Load all relevant data from the database."""
        try:
            # Load assets information
            assets_query = """
                SELECT * FROM assets
                ORDER BY ticker
            """
            
            # Load daily prices with calculated metrics
            prices_query = """
                SELECT 
                    dp.date,
                    dp.ticker,
                    dp.open_price,
                    dp.high_price,
                    dp.low_price,
                    dp.close_price,
                    dp.volume,
                    cm.daily_return,
                    cm.volatility_30d,
                    cm.moving_avg_50d,
                    cm.sharpe_ratio
                FROM daily_prices dp
                LEFT JOIN calculated_metrics cm ON dp.date = cm.date AND dp.ticker = cm.ticker
                ORDER BY dp.date DESC, dp.ticker
            """
            
            self.assets_df = pd.read_sql_query(assets_query, self.conn)
            self.prices_df = pd.read_sql_query(prices_query, self.conn)
            
            # Convert date column to datetime
            self.prices_df['date'] = pd.to_datetime(self.prices_df['date'])
            
            print(f"✅ Loaded {len(self.assets_df)} assets")
            print(f"✅ Loaded {len(self.prices_df)} price records")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def explore_data(self):
        """Perform comprehensive data exploration."""
        print("\n🔍 DATA EXPLORATION SUMMARY")
        print("=" * 50)
        
        # Date range
        print(f"📅 Date Range: {self.prices_df['date'].min().strftime('%Y-%m-%d')} to {self.prices_df['date'].max().strftime('%Y-%m-%d')}")
        print(f"📊 Total Trading Days: {self.prices_df['date'].nunique()}")
        print(f"🏢 Number of Stocks: {self.prices_df['ticker'].nunique()}")
        
        # Missing values
        print("\n❓ Missing Values:")
        missing_data = self.prices_df.isnull().sum()
        for col, missing in missing_data[missing_data > 0].items():
            print(f"  {col}: {missing}")
        
        # Summary statistics by stock
        print("\n📈 Summary Statistics by Stock:")
        summary_stats = self.prices_df.groupby('ticker').agg({
            'close_price': ['count', 'mean', 'std', 'min', 'max'],
            'daily_return': ['mean', 'std', 'min', 'max'],
            'volume': ['mean', 'std']
        }).round(4)
        
        print(summary_stats)
        
        return summary_stats
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics for each stock."""
        print("\n📊 Calculating Performance Metrics...")
        
        performance_data = []
        
        for ticker in self.prices_df['ticker'].unique():
            stock_data = self.prices_df[self.prices_df['ticker'] == ticker].sort_values('date')
            
            if len(stock_data) > 0 and 'daily_return' in stock_data.columns:
                # Calculate cumulative returns
                stock_data = stock_data.copy()
                stock_data['cumulative_return'] = (1 + stock_data['daily_return']).cumprod()
                
                # Calculate annualized metrics
                trading_days = len(stock_data)
                total_return = stock_data['cumulative_return'].iloc[-1] - 1
                annualized_return = (1 + total_return) ** (252 / trading_days) - 1
                annualized_volatility = stock_data['daily_return'].std() * np.sqrt(252)
                
                # Calculate Sharpe ratio (assuming 2% risk-free rate)
                risk_free_rate = 0.02
                sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
                
                # Calculate maximum drawdown
                cumulative_returns = stock_data['cumulative_return']
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min()
                
                performance_data.append({
                    'ticker': ticker,
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'annualized_volatility': annualized_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'trading_days': trading_days
                })
        
        self.performance_df = pd.DataFrame(performance_data)
        
        # Add company names
        self.performance_df = self.performance_df.merge(
            self.assets_df[['ticker', 'company_name', 'sector']], on='ticker'
        )
        
        print("✅ Performance metrics calculated successfully!")
        
        # Display performance metrics
        display_cols = ['ticker', 'company_name', 'sector', 'total_return', 'annualized_return', 
                       'annualized_volatility', 'sharpe_ratio', 'max_drawdown']
        
        print("\n📊 PERFORMANCE METRICS SUMMARY")
        print("=" * 50)
        print(self.performance_df[display_cols].round(4).to_string(index=False))
        
        return self.performance_df
    
    def create_visualization_dashboard(self):
        """Create a comprehensive visualization dashboard."""
        print("\n🎨 Creating Visualization Dashboard...")
        
        # Set up the plotting style
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Stock Price Evolution
        ax1 = plt.subplot(4, 2, 1)
        for ticker in self.prices_df['ticker'].unique():
            stock_data = self.prices_df[self.prices_df['ticker'] == ticker].sort_values('date')
            plt.plot(stock_data['date'], stock_data['close_price'], label=ticker, linewidth=2)
        
        plt.title('Stock Price Evolution Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Close Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 2. Cumulative Returns
        ax2 = plt.subplot(4, 2, 2)
        for ticker in self.prices_df['ticker'].unique():
            stock_data = self.prices_df[self.prices_df['ticker'] == ticker].sort_values('date')
            if 'daily_return' in stock_data.columns:
                cumulative_returns = (1 + stock_data['daily_return']).cumprod()
                plt.plot(stock_data['date'], cumulative_returns, label=ticker, linewidth=2)
        
        plt.title('Cumulative Returns Over Time', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 3. Daily Returns Distribution
        ax3 = plt.subplot(4, 2, 3)
        for ticker in self.prices_df['ticker'].unique():
            stock_data = self.prices_df[self.prices_df['ticker'] == ticker]
            if 'daily_return' in stock_data.columns:
                returns = stock_data['daily_return'].dropna()
                plt.hist(returns, bins=30, alpha=0.6, label=ticker, density=True)
        
        plt.title('Distribution of Daily Returns', fontsize=14, fontweight='bold')
        plt.xlabel('Daily Return')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Volatility Comparison
        ax4 = plt.subplot(4, 2, 4)
        if 'volatility_30d' in self.prices_df.columns:
            volatility_data = self.prices_df.groupby('ticker')['volatility_30d'].mean().sort_values(ascending=False)
            plt.bar(volatility_data.index, volatility_data.values, alpha=0.7)
            plt.title('Average 30-Day Volatility by Stock', fontsize=14, fontweight='bold')
            plt.xlabel('Stock Ticker')
            plt.ylabel('Average Volatility')
            plt.grid(True, alpha=0.3)
        
        # 5. Sharpe Ratio Comparison
        ax5 = plt.subplot(4, 2, 5)
        sharpe_data = self.performance_df.sort_values('sharpe_ratio', ascending=False)
        plt.barh(sharpe_data['ticker'], sharpe_data['sharpe_ratio'], alpha=0.7)
        plt.title('Sharpe Ratio by Stock', fontsize=14, fontweight='bold')
        plt.xlabel('Sharpe Ratio')
        plt.ylabel('Stock Ticker')
        plt.grid(True, alpha=0.3)
        
        # 6. Maximum Drawdown
        ax6 = plt.subplot(4, 2, 6)
        drawdown_data = self.performance_df.sort_values('max_drawdown')
        plt.barh(drawdown_data['ticker'], drawdown_data['max_drawdown'], alpha=0.7)
        plt.title('Maximum Drawdown by Stock', fontsize=14, fontweight='bold')
        plt.xlabel('Maximum Drawdown')
        plt.ylabel('Stock Ticker')
        plt.grid(True, alpha=0.3)
        
        # 7. Volume Analysis
        ax7 = plt.subplot(4, 2, 7)
        volume_data = self.prices_df.groupby('ticker')['volume'].mean().sort_values(ascending=False)
        plt.bar(volume_data.index, volume_data.values, alpha=0.7)
        plt.title('Average Trading Volume by Stock', fontsize=14, fontweight='bold')
        plt.xlabel('Stock Ticker')
        plt.ylabel('Average Volume')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 8. Sector Performance
        ax8 = plt.subplot(4, 2, 8)
        sector_performance = self.performance_df.groupby('sector')['annualized_return'].mean().sort_values(ascending=False)
        plt.bar(sector_performance.index, sector_performance.values, alpha=0.7)
        plt.title('Average Annualized Return by Sector', fontsize=14, fontweight='bold')
        plt.xlabel('Sector')
        plt.ylabel('Average Annualized Return')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('portfolio_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print("✅ Dashboard saved as 'portfolio_dashboard.png'")
        return fig
    
    def analyze_correlations(self):
        """Analyze correlations between different stocks."""
        print("\n🔗 Analyzing Stock Correlations...")
        
        # Create returns matrix
        returns_matrix = self.prices_df.pivot(index='date', columns='ticker', values='daily_return')
        
        # Calculate correlation matrix
        correlation_matrix = returns_matrix.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
        plt.title('Correlation Matrix of Daily Returns', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print("✅ Correlation heatmap saved as 'correlation_heatmap.png'")
        
        # Display correlation statistics
        print("\n📊 CORRELATION ANALYSIS")
        print("=" * 40)
        
        # Find highest and lowest correlations
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                ticker1 = correlation_matrix.columns[i]
                ticker2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                correlations.append((ticker1, ticker2, corr_value))
        
        correlations_df = pd.DataFrame(correlations, columns=['Stock1', 'Stock2', 'Correlation'])
        correlations_df = correlations_df.sort_values('Correlation', ascending=False)
        
        print("\n🔗 Stock Pair Correlations:")
        print(correlations_df.round(4).to_string(index=False))
        
        return correlation_matrix, correlations_df
    
    def test_portfolio_strategies(self):
        """Test different portfolio allocation strategies."""
        print("\n📊 Testing Portfolio Strategies...")
        
        tickers = self.prices_df['ticker'].unique()
        
        # Strategy 1: Equal Weight
        equal_weights = {ticker: 1/len(tickers) for ticker in tickers}
        
        # Strategy 2: Market Cap Weighted (simplified - using average price as proxy)
        avg_prices = self.prices_df.groupby('ticker')['close_price'].mean()
        total_value = avg_prices.sum()
        market_cap_weights = {ticker: price/total_value for ticker, price in avg_prices.items()}
        
        # Strategy 3: Risk Parity (simplified - inverse volatility)
        volatilities = self.prices_df.groupby('ticker')['daily_return'].std()
        inv_vol = 1 / volatilities
        total_inv_vol = inv_vol.sum()
        risk_parity_weights = {ticker: vol/total_inv_vol for ticker, vol in inv_vol.items()}
        
        strategies = {
            'Equal Weight': equal_weights,
            'Market Cap Weighted': market_cap_weights,
            'Risk Parity': risk_parity_weights
        }
        
        results = {}
        
        for strategy_name, weights in strategies.items():
            portfolio_returns, metrics, _ = self.construct_portfolio(weights)
            results[strategy_name] = metrics
        
        results_df = pd.DataFrame(results).T
        
        print("\n📊 PORTFOLIO STRATEGY COMPARISON")
        print("=" * 50)
        
        print("\n📈 Portfolio Performance by Strategy:")
        print(results_df.round(4).to_string())
        
        # Show weights for each strategy
        print("\n⚖️ Portfolio Weights by Strategy:")
        for strategy_name, weights in strategies.items():
            print(f"\n{strategy_name}:")
            for ticker, weight in weights.items():
                print(f"  {ticker}: {weight:.4f}")
        
        return results_df, strategies
    
    def construct_portfolio(self, weights):
        """Construct and backtest a portfolio with given weights."""
        # Create portfolio returns
        portfolio_returns = pd.DataFrame()
        
        for ticker in weights.keys():
            stock_data = self.prices_df[self.prices_df['ticker'] == ticker].sort_values('date')
            if 'daily_return' in stock_data.columns:
                portfolio_returns[ticker] = stock_data['daily_return'] * weights[ticker]
        
        # Calculate portfolio daily returns
        portfolio_returns['portfolio_return'] = portfolio_returns.sum(axis=1)
        
        # Calculate cumulative portfolio value
        portfolio_returns['cumulative_value'] = (1 + portfolio_returns['portfolio_return']).cumprod()
        
        # Calculate portfolio metrics
        total_return = portfolio_returns['cumulative_value'].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        annualized_volatility = portfolio_returns['portfolio_return'].std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / annualized_volatility
        
        # Calculate maximum drawdown
        cumulative_values = portfolio_returns['cumulative_value']
        running_max = cumulative_values.expanding().max()
        drawdown = (cumulative_values - running_max) / running_max
        max_drawdown = drawdown.min()
        
        portfolio_metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        return portfolio_returns, portfolio_metrics, weights
    
    def generate_insights(self):
        """Generate key insights from the analysis."""
        print("\n🔍 Generating Key Insights...")
        
        print("\n🔍 KEY INSIGHTS AND RECOMMENDATIONS")
        print("=" * 60)
        
        # Best and worst performers
        best_performer = self.performance_df.loc[self.performance_df['annualized_return'].idxmax()]
        worst_performer = self.performance_df.loc[self.performance_df['annualized_return'].idxmin()]
        
        print(f"🏆 Best Performer: {best_performer['ticker']} ({best_performer['company_name']})")
        print(f"   Annualized Return: {best_performer['annualized_return']:.2%}")
        print(f"   Sharpe Ratio: {best_performer['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {best_performer['max_drawdown']:.2%}")
        
        print(f"\n📉 Worst Performer: {worst_performer['ticker']} ({worst_performer['company_name']})")
        print(f"   Annualized Return: {worst_performer['annualized_return']:.2%}")
        print(f"   Sharpe Ratio: {worst_performer['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {worst_performer['max_drawdown']:.2%}")
        
        # Risk analysis
        print(f"\n⚠️ Risk Analysis:")
        avg_volatility = self.performance_df['annualized_volatility'].mean()
        print(f"   Average Portfolio Volatility: {avg_volatility:.2%}")
        
        # Portfolio recommendations
        print(f"\n💡 Portfolio Recommendations:")
        print(f"   1. Consider overweighting {best_performer['ticker']} for growth")
        print(f"   2. Monitor {worst_performer['ticker']} for potential recovery or exit")
        print(f"   3. Diversification benefits: {len(self.performance_df)} stocks across {self.performance_df['sector'].nunique()} sectors")
        
        return {
            'best_performer': best_performer,
            'worst_performer': worst_performer,
            'avg_volatility': avg_volatility
        }
    
    def export_results(self):
        """Export analysis results to CSV files."""
        print("\n📁 Exporting Results...")
        
        try:
            # Create outputs directory if it doesn't exist
            os.makedirs('outputs', exist_ok=True)
            
            # Export price data
            self.prices_df.to_csv('outputs/portfolio_prices.csv', index=False)
            print("✅ Exported price data to outputs/portfolio_prices.csv")
            
            # Export performance metrics
            self.performance_df.to_csv('outputs/portfolio_performance.csv', index=False)
            print("✅ Exported performance metrics to outputs/portfolio_performance.csv")
            
            print("\n📁 All results exported successfully to outputs/ directory!")
            
        except Exception as e:
            print(f"❌ Error exporting results: {e}")
    
    def run_complete_analysis(self):
        """Run the complete financial analysis pipeline."""
        print("🚀 Starting Financial Portfolio Analysis Pipeline")
        print("=" * 60)
        
        try:
            # Step 1: Connect to database
            if not self.connect_database():
                return False
            
            # Step 2: Load data
            if not self.load_data():
                return False
            
            # Step 3: Data exploration
            self.explore_data()
            
            # Step 4: Calculate performance metrics
            self.calculate_performance_metrics()
            
            # Step 5: Create visualizations
            self.create_visualization_dashboard()
            
            # Step 6: Correlation analysis
            self.analyze_correlations()
            
            # Step 7: Portfolio strategies
            self.test_portfolio_strategies()
            
            # Step 8: Generate insights
            self.generate_insights()
            
            # Step 9: Export results
            self.export_results()
            
            print("\n🎉 Financial Portfolio Analysis Complete!")
            print("=" * 50)
            print("📊 Analysis Summary:")
            print("   • Stock price evolution and trends")
            print("   • Performance metrics and risk analysis")
            print("   • Correlation analysis between stocks")
            print("   • Portfolio construction strategies")
            print("   • Comprehensive visualizations")
            print("\n📁 Results exported to outputs/ directory")
            print("📈 Ready for portfolio decision-making!")
            
            return True
            
        except Exception as e:
            print(f"❌ Analysis pipeline failed: {e}")
            return False
        
        finally:
            # Always close database connection
            self.close_database()

def main():
    """Main function to run the financial analysis."""
    # Initialize analyzer
    analyzer = FinancialPortfolioAnalyzer()
    
    # Run complete analysis
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n✅ Analysis completed successfully!")
    else:
        print("\n❌ Analysis failed. Please check the error messages above.")

if __name__ == "__main__":
    main()

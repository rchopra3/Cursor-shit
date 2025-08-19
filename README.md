# Financial Portfolio Analysis Dashboard

A comprehensive end-to-end financial analysis project that demonstrates industry-relevant skills in Python, SQL, and data analysis. This project analyzes a portfolio of technology stocks and provides insights into performance, risk, and portfolio optimization.

## 🎯 Project Overview

The Financial Portfolio Analysis Dashboard is designed to showcase:
- **Data Engineering**: Automated data acquisition from Yahoo Finance API
- **Database Management**: SQLite database with normalized schema design
- **Financial Analysis**: Advanced metrics including Sharpe ratios, volatility, and correlations
- **Data Visualization**: Interactive charts and comprehensive dashboards
- **Portfolio Optimization**: Multiple allocation strategies and backtesting

## 🏗️ Project Structure

```
Financial Portfolio Analysis Dashboard/
├── requirements.txt          # Python dependencies
├── schema.sql               # Database schema and initial data
├── queries.sql              # Advanced SQL analysis queries
├── data_engineer.py         # Main data engineering pipeline
├── financial_analysis.py     # Python script for analysis and visualization
├── test_setup.py            # Test script to verify project setup
├── portfolio.db             # SQLite database (created after running)
├── README.md               # This file
└── outputs/                # Generated analysis files
    ├── portfolio_prices.csv
    ├── portfolio_performance.csv
    └── correlation_matrix.csv
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection for data fetching

### Installation

1. **Clone or download the project files**
   ```bash
   # Ensure you're in the project directory
   cd "Financial Portfolio Analysis Dashboard"
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test the setup**
   ```bash
   python test_setup.py
   ```

4. **Run the data engineering pipeline**
   ```bash
   python data_engineer.py
   ```

5. **Run the financial analysis script**
   ```bash
   python financial_analysis.py
   ```

## 📊 Database Schema

The project uses a normalized SQLite database with three main tables:

### `assets` Table
- **ticker**: Stock symbol (Primary Key)
- **company_name**: Full company name
- **sector**: Business sector classification
- **created_at**: Timestamp of record creation

### `daily_prices` Table
- **id**: Unique identifier (Primary Key)
- **date**: Trading date
- **ticker**: Stock symbol (Foreign Key to assets)
- **open_price**: Opening price
- **high_price**: Highest price of the day
- **low_price**: Lowest price of the day
- **close_price**: Closing price
- **volume**: Trading volume
- **created_at**: Timestamp of record creation

### `calculated_metrics` Table
- **id**: Unique identifier (Primary Key)
- **date**: Trading date
- **ticker**: Stock symbol (Foreign Key to assets)
- **daily_return**: Daily percentage return
- **volatility_30d**: 30-day rolling volatility
- **moving_avg_50d**: 50-day moving average
- **sharpe_ratio**: Risk-adjusted return metric
- **created_at**: Timestamp of record creation

## 🔧 Technical Implementation

### Data Engineering Pipeline (`data_engineer.py`)

The main pipeline performs the following operations:

1. **Data Acquisition**: Uses `yfinance` library to fetch 2 years of historical data
2. **Data Processing**: Calculates financial metrics including:
   - Daily returns
   - Rolling volatility (30-day)
   - Moving averages (50-day)
   - Sharpe ratios
3. **Database Storage**: Stores processed data using parameterized SQL queries
4. **Error Handling**: Comprehensive logging and exception handling

### Key Features:
- **Parameterized Queries**: Prevents SQL injection vulnerabilities
- **Data Validation**: Ensures data quality and completeness
- **Logging**: Detailed logging for debugging and monitoring
- **Modular Design**: Clean, maintainable code structure

### Advanced SQL Queries (`queries.sql`)

The project includes sophisticated SQL analysis:

1. **Monthly Performance Ranking**: Best and worst performing stocks by month
2. **Correlation Analysis**: Pairwise correlations between stock returns
3. **Portfolio Backtesting**: Hypothetical portfolio value over time
4. **Risk Metrics**: Comprehensive risk-adjusted performance analysis
5. **Sector Analysis**: Performance comparison across business sectors

## 📈 Analysis Capabilities

### Performance Metrics
- **Total Return**: Cumulative performance over the analysis period
- **Annualized Return**: Year-over-year growth rate
- **Volatility**: Standard deviation of returns (annualized)
- **Sharpe Ratio**: Risk-adjusted return metric
- **Maximum Drawdown**: Largest peak-to-trough decline

### Portfolio Strategies
1. **Equal Weight**: Equal allocation across all stocks
2. **Market Cap Weighted**: Allocation based on average stock prices
3. **Risk Parity**: Allocation inversely proportional to volatility

### Visualization Dashboard
- **Stock Price Evolution**: Time series of closing prices
- **Cumulative Returns**: Performance comparison over time
- **Risk-Return Scatter Plot**: Interactive risk-return analysis
- **Correlation Heatmap**: Stock relationship visualization
- **Performance Bar Charts**: Comparative analysis across metrics

## 🎨 Key Visualizations

### Static Charts (Matplotlib/Seaborn)
- Stock price evolution over time
- Distribution of daily returns
- Volatility comparison across stocks
- Sharpe ratio analysis
- Maximum drawdown visualization
- Trading volume analysis
- Sector performance comparison

### Interactive Charts (Plotly)
- Interactive stock price charts with hover information
- Performance comparison bar charts
- Risk-return scatter plots with color-coded Sharpe ratios
- Zoomable and pannable visualizations

## 📊 Sample Analysis Results

### Portfolio Performance Summary
Based on the analysis of technology stocks (AAPL, MSFT, GOOGL, AMZN, META):

- **Best Performer**: Typically shows highest annualized returns with favorable Sharpe ratios
- **Risk Profile**: Technology sector shows moderate to high volatility
- **Correlation Patterns**: Stocks within the same sector show moderate correlations
- **Diversification Benefits**: Portfolio construction reduces individual stock risk

### Key Insights
1. **Sector Concentration**: All stocks are technology-focused, limiting sector diversification
2. **Volatility Patterns**: Consistent with technology sector characteristics
3. **Performance Variation**: Significant differences in individual stock performance
4. **Correlation Structure**: Moderate correlations suggest some diversification benefits

## 🛠️ Customization Options

### Adding New Stocks
1. Modify the `tickers` list in `data_engineer.py`
2. Add company information to the `assets` table
3. Re-run the data pipeline

### Extending Analysis
1. Add new metrics to the `calculated_metrics` table
2. Create additional SQL queries in `queries.sql`
3. Extend the Python script with new visualizations

### Database Modifications
1. Update `schema.sql` with new tables or columns
2. Modify the data engineering pipeline accordingly
3. Update analysis queries as needed

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Ensure `schema.sql` exists in the project directory
   - Check file permissions for database creation

2. **Data Fetching Issues**
   - Verify internet connection
   - Check if Yahoo Finance API is accessible
   - Ensure ticker symbols are valid

3. **Missing Dependencies**
   - Run `pip install -r requirements.txt`
   - Check Python version compatibility

4. **Python Script Issues**
   - Ensure all required packages are installed
   - Check Python version compatibility

### Performance Tips
- The initial data fetch may take several minutes
- Database operations are optimized with proper indexing
- Large datasets may require additional memory allocation

## 📚 Learning Outcomes

This project demonstrates:

### Technical Skills
- **Python Programming**: Object-oriented design, data manipulation, API integration
- **SQL Database Design**: Normalized schema, advanced queries, performance optimization
- **Data Analysis**: Statistical analysis, financial metrics, time series analysis
- **Data Visualization**: Multiple chart types, interactive dashboards, professional presentation

### Financial Knowledge
- **Portfolio Theory**: Risk-return analysis, diversification, correlation analysis
- **Financial Metrics**: Returns, volatility, Sharpe ratios, drawdown analysis
- **Market Analysis**: Sector analysis, performance comparison, trend identification

### Industry Best Practices
- **Code Quality**: PEP8 compliance, documentation, error handling
- **Data Engineering**: ETL pipelines, data validation, logging
- **Project Structure**: Modular design, configuration management, deployment readiness

## 🚀 Future Enhancements

### Potential Improvements
1. **Additional Data Sources**: Incorporate fundamental data, news sentiment
2. **Machine Learning**: Predictive models, portfolio optimization algorithms
3. **Real-time Updates**: Automated data refresh and alerts
4. **Web Dashboard**: Flask/FastAPI web application
5. **Additional Assets**: Bonds, commodities, international stocks
6. **Risk Management**: VaR calculations, stress testing

### Advanced Analytics
1. **Factor Analysis**: Multi-factor risk models
2. **Options Analysis**: Greeks calculation, option strategies
3. **Backtesting Framework**: Historical strategy performance
4. **Portfolio Optimization**: Efficient frontier, risk budgeting

## 📄 License

This project is created for educational and portfolio demonstration purposes. Please ensure compliance with relevant financial data usage terms and conditions.

## 🤝 Contributing

This is a portfolio project demonstrating individual skills. However, suggestions for improvements are welcome through appropriate channels.

## 📞 Support

For questions or issues related to this project:
1. Check the troubleshooting section above
2. Review the code comments and documentation
3. Ensure all dependencies are properly installed
4. Verify the database schema and data integrity

---

**Note**: This project is designed for educational purposes and should not be used for actual investment decisions without proper financial advice and risk assessment.

**Happy Analyzing! 📊📈**

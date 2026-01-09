# 📊 Finance Dashboard

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A professional financial dashboard for real-time market analysis, backtesting strategies, and portfolio management. Built with Python and Streamlit.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Finance+Dashboard+Preview)

## 🎯 Project Overview

This project is developed as part of a quantitative finance course. It provides a comprehensive platform for:

- **Real-time Market Data**: Retrieve and display financial data from multiple sources
- **Single Asset Analysis (Quant A)**: Backtesting strategies for individual assets
- **Portfolio Analysis (Quant B)**: Multi-asset portfolio management and optimization
- **Automated Reporting**: Daily reports generated via cron jobs

## 🚀 Features

### Quant A - Single Asset Analysis
- ✅ Real-time price display for stocks, forex, crypto, and commodities
- ✅ Multiple backtesting strategies:
  - Buy and Hold
  - Moving Average Crossover
  - RSI Strategy
  - Momentum Strategy
- ✅ Performance metrics: Sharpe Ratio, Max Drawdown, Volatility, Win Rate
- ✅ Interactive parameter controls
- ✅ Price and strategy visualization on the same chart

### Quant B - Portfolio Analysis
- ✅ Multi-asset portfolio construction (3+ assets)
- ✅ Multiple allocation methods:
  - Equal Weight
  - Custom Weights
  - Risk Parity (Inverse Volatility)
- ✅ Portfolio metrics: Correlation matrix, diversification ratio, VaR
- ✅ Rebalancing simulation (Monthly, Quarterly, Yearly)
- ✅ Visual comparison between assets and portfolio

### Infrastructure
- ✅ Auto-refresh every 5 minutes
- ✅ Daily report generation via cron (8 PM)
- ✅ 24/7 deployment on Linux server
- ✅ Health check and auto-restart

## 📁 Project Structure

```
finance-dashboard/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── modules/                    # Core analysis modules
│   ├── __init__.py
│   ├── single_asset.py         # Quant A - Single asset analysis
│   └── portfolio.py            # Quant B - Portfolio analysis
│
├── utils/                      # Utility modules
│   ├── __init__.py
│   ├── data_fetcher.py         # Data retrieval (Yahoo Finance, etc.)
│   ├── metrics.py              # Financial metrics calculations
│   └── report_generator.py     # Daily report generation
│
├── scripts/                    # Deployment scripts
│   ├── start_app.sh            # Application startup script
│   ├── setup_cron.sh           # Cron job configuration
│   ├── health_check.sh         # Health monitoring script
│   └── generate_report.py      # Report generation script
│
├── reports/                    # Generated daily reports
│   └── daily_report_YYYY-MM-DD.txt
│
└── logs/                       # Application logs
    ├── app.log
    ├── cron_report.log
    └── health_check.log
```

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/AlexisAHG/finance-dashboard.git
cd finance-dashboard
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Access the dashboard**
Open your browser and go to `http://localhost:8501`

### Linux Server Deployment

1. **Connect to your server**
```bash
ssh user@your-server-ip
```

2. **Clone and setup**
```bash
git clone https://github.com/AlexisAHG/finance-dashboard.git
cd finance-dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **Start the application**
```bash
chmod +x scripts/*.sh
./scripts/start_app.sh start
```

4. **Configure cron jobs**
```bash
./scripts/setup_cron.sh
```

5. **Verify deployment**
```bash
./scripts/start_app.sh status
```

## 🔄 Cron Job Configuration

The following cron jobs are configured automatically:

| Schedule | Task | Description |
|----------|------|-------------|
| `0 20 * * *` | Daily Report | Generates financial report at 8 PM |
| `*/5 * * * *` | Health Check | Checks and restarts app if needed |
| `0 3 * * 0` | Log Cleanup | Removes logs older than 7 days |

To view current cron jobs:
```bash
crontab -l
```

To manually edit cron jobs:
```bash
crontab -e
```

## 📊 Data Sources

The dashboard supports multiple data sources:

| Source | Assets | API |
|--------|--------|-----|
| Yahoo Finance | Stocks, ETFs, Forex | yfinance |
| CoinGecko | Cryptocurrencies | REST API |
| Web Scraping | Various | BeautifulSoup |

### Supported Assets

- **Stocks**: AAPL, GOOGL, MSFT, TSLA, NVDA, META
- **French Stocks**: ENGI.PA, TTE.PA, AIR.PA, BNP.PA
- **Forex**: EURUSD, GBPUSD, USDJPY
- **Commodities**: Gold (GC=F), Silver (SI=F), Oil (CL=F)
- **Crypto**: BTC-USD, ETH-USD, SOL-USD

## 📈 Backtesting Strategies

### Buy and Hold
Simple strategy that buys at the start and holds until the end.

### Moving Average Crossover
- **Parameters**: Short MA (default: 20), Long MA (default: 50)
- **Logic**: Buy when short MA crosses above long MA, sell when it crosses below

### RSI Strategy
- **Parameters**: Period (default: 14), Oversold (default: 30), Overbought (default: 70)
- **Logic**: Buy when RSI < oversold, sell when RSI > overbought

### Momentum Strategy
- **Parameters**: Lookback period (default: 20)
- **Logic**: Buy when price is above N-day ago price, sell otherwise

## 📋 Daily Reports

Reports are automatically generated at 8 PM and include:

- Current prices for all tracked assets
- Daily price changes
- Open/High/Low/Close values
- Volume statistics
- Annualized volatility
- Maximum drawdown
- Monthly returns
- Top/Bottom performers

Reports are saved in the `reports/` directory as both `.txt` and `.json` files.

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
PORT=8501
API_KEY=your_api_key_here  # Optional: for premium data sources
```

### Customizing Tracked Assets

Edit the `DEFAULT_ASSETS` list in `utils/report_generator.py`:

```python
DEFAULT_ASSETS = [
    "AAPL", "GOOGL", "MSFT",  # Your preferred assets
]
```

## 🧪 Testing

Run the application locally to test all features:

```bash
# Test single asset analysis
streamlit run app.py

# Test report generation
python scripts/generate_report.py
```

## 🤝 Contributing

This project follows Git best practices:

1. Create a feature branch
```bash
git checkout -b feature/your-feature-name
```

2. Make commits with clear messages
```bash
git commit -m "Add: new RSI strategy implementation"
```

3. Push and create pull request
```bash
git push origin feature/your-feature-name
```

### Commit Message Convention

- `Add:` New feature
- `Fix:` Bug fix
- `Update:` Update existing feature
- `Docs:` Documentation changes
- `Refactor:` Code refactoring
- `Style:` Formatting changes

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Quant A** (Single Asset Analysis) - [GitHub Profile]
- **Quant B** (Portfolio Analysis) - [GitHub Profile]

## 🙏 Acknowledgments

- Course instructors for project guidance
- Yahoo Finance for data access
- Streamlit team for the amazing framework

---

**Note**: This dashboard is for educational purposes only. Always do your own research before making investment decisions.

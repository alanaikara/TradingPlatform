# Personal Trading Platform

A comprehensive Python-based trading platform built with Streamlit for managing trading strategies, backtesting, broker connections, and PnL analytics.

## 🔒 Security Setup (IMPORTANT)

### Before Using This Application:

1. **Copy the environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Add your broker credentials to .env:**
   ```
   DHAN_CLIENT_ID=your_actual_client_id
   DHAN_ACCESS_TOKEN=your_actual_access_token
   ```

3. **Never commit the .env file or trading_platform.db to version control!**
   - The .gitignore file is already configured to exclude these files
   - Your credentials are encrypted in the database for additional security

## 🚀 Quick Start

1. **Create and activate virtual environment:**
   ```bash
   python -m venv trading_env
   source trading_env/bin/activate  # On Windows: trading_env\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run Trading_app.py
   ```

## 📁 Project Structure

```
trading_platform/
├── Trading_app.py          # Main Streamlit application
├── dhan_connector.py       # Dhan broker API connector
├── requirements.txt        # Python dependencies
├── .env.example           # Template for environment variables
├── .gitignore            # Git ignore rules (includes sensitive files)
├── trading_platform.db   # SQLite database (auto-created, encrypted credentials)
└── README.md             # This file
```

## 🛡️ Security Features

- **Environment Variables**: Credentials loaded from .env file (not tracked by Git)
- **Encrypted Database**: All stored credentials are encrypted using cryptography library
- **Safe Defaults**: No hardcoded credentials in source code
- **Git Protection**: Comprehensive .gitignore to prevent credential leaks

## 📊 Features

- **Dashboard**: Real-time portfolio overview with live data
- **Strategy Management**: Create and manage trading strategies
- **Backtesting**: Test strategies with historical data simulation
- **Broker Integration**: Connect to Dhan broker with live API
- **PnL Analytics**: Track profit/loss and risk metrics
- **Trade History**: View and analyze trade records

## ⚠️ Important Notes

1. **Live Data**: Requires Dhan live data feed subscription
2. **API Credentials**: Never share your broker API credentials
3. **Database**: The SQLite database contains encrypted credentials
4. **Testing**: Use paper trading before live trading

## 🔧 Configuration

### Environment Variables (.env file):
```
DHAN_CLIENT_ID=your_client_id
DHAN_ACCESS_TOKEN=your_access_token
ENCRYPTION_KEY=optional_custom_encryption_key
DEBUG=False
```

### Database:
- SQLite database automatically created on first run
- Credentials are encrypted before storage
- Tables: strategies, trades, pnl_records, broker_credentials

## 🤝 Contributing

If you want to contribute to this project:

1. Fork the repository
2. Create a feature branch
3. **Never commit .env or .db files**
4. Submit a pull request

## 📄 License

This project is for personal use. Please ensure compliance with your broker's API terms of service.

---

**⚠️ SECURITY WARNING**: Always keep your API credentials secure and never commit them to version control!
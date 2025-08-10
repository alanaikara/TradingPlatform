# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based personal trading platform built with Streamlit that provides comprehensive trading strategy management, backtesting, broker connections, and PnL analytics. The application is designed for both stocks and commodities trading with support for multiple broker integrations.

## Architecture

The project follows a single-file Streamlit application architecture:

- **Main Application**: `Trading_app.py` - Contains the entire application with all pages and functionality
- **Database**: SQLite database (`trading_platform.db`) for persistent storage of trades, PnL records, and strategies
- **Virtual Environment**: `trading_env/` - Python virtual environment with all dependencies
- **Dependencies**: Listed in `requirements.txt` (minimal file structure)

### Core Components

1. **Data Models** (lines 22-49): Dataclass definitions for Strategy, Trade, and PnLRecord
2. **Database Layer** (lines 51-96): SQLite initialization and table creation
3. **Session State Management** (lines 98-104): Streamlit session state for in-memory data
4. **Multi-page Application** (lines 106-547): Six main pages with distinct functionality

### Application Pages

- **Dashboard** (lines 118-181): Portfolio overview, performance charts, recent activity
- **Strategy Management** (lines 182-251): Add/edit/delete trading strategies with parameters
- **Backtesting** (lines 253-341): Strategy backtesting engine with performance metrics
- **Broker Connections** (lines 343-390): Integration settings for multiple brokers
- **PnL Analytics** (lines 392-465): Profit/loss analysis and risk metrics
- **Trade History** (lines 467-540): Trade records with filtering and statistics

## Development Commands

### Running the Application
```bash
# Activate virtual environment
source trading_env/bin/activate  # On macOS/Linux
# or
trading_env\Scripts\activate     # On Windows

# Run the Streamlit application
streamlit run Trading_app.py
```

### Dependencies Management
```bash
# Install dependencies (if requirements.txt is populated)
pip install -r requirements.txt

# Generate/update requirements
pip freeze > requirements.txt
```

### Database Operations
The application automatically initializes the SQLite database on first run. Database file: `trading_platform.db`

## Key Technologies

- **Streamlit**: Web application framework
- **Plotly**: Interactive charts and visualizations  
- **Pandas/NumPy**: Data manipulation and analysis
- **SQLite**: Local database storage
- **Python Dataclasses**: Type-safe data models

## Data Flow

1. User interactions through Streamlit UI
2. Session state management for temporary data
3. SQLite database for persistent storage
4. Real-time chart updates via Plotly
5. Sample data generation for demonstration

## Development Notes

- Single-file architecture makes the codebase easy to understand but may need refactoring for scalability
- Database operations are currently basic - consider adding error handling and connection pooling
- Sample data is generated for demonstration - real broker API integration needed for production
- No authentication/authorization system currently implemented
- Virtual environment contains standard data science and web development packages
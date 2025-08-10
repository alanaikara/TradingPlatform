import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from typing import Dict, List, Optional
import sqlite3
import os
from cryptography.fernet import Fernet
import base64

# Load environment variables
def load_env_file():
    """Load .env file if it exists"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

# Load .env file
load_env_file()

# Security functions
def get_encryption_key():
    """Generate or retrieve encryption key for credentials"""
    key = os.getenv('ENCRYPTION_KEY')
    if not key:
        # Generate a key based on system info (not completely secure but better than plaintext)
        import platform
        import getpass
        base_string = f"{platform.node()}{getpass.getuser()}"
        # Create a 32-byte key from the base string
        key = base64.urlsafe_b64encode(base_string.encode().ljust(32)[:32])
    return key

def encrypt_credential(credential: str) -> str:
    """Encrypt credential for storage"""
    if not credential:
        return ""
    
    try:
        key = get_encryption_key()
        f = Fernet(key)
        encrypted = f.encrypt(credential.encode())
        return base64.b64encode(encrypted).decode()
    except Exception:
        # Fallback to base64 encoding if encryption fails
        return base64.b64encode(credential.encode()).decode()

def decrypt_credential(encrypted_credential: str) -> str:
    """Decrypt credential from storage"""
    if not encrypted_credential:
        return ""
    
    try:
        key = get_encryption_key()
        f = Fernet(key)
        encrypted_data = base64.b64decode(encrypted_credential.encode())
        decrypted = f.decrypt(encrypted_data)
        return decrypted.decode()
    except Exception:
        # Fallback to base64 decoding if decryption fails
        try:
            return base64.b64decode(encrypted_credential.encode()).decode()
        except Exception:
            return encrypted_credential  # Return as-is if all fails

# Configure page
st.set_page_config(
    page_title="Personal Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data Models
@dataclass
class Strategy:
    name: str
    description: str
    asset_type: str  # stocks, commodities_cash, commodities_futures
    parameters: Dict
    active: bool = True
    created_date: datetime = None

@dataclass
class Trade:
    strategy_name: str
    symbol: str
    side: str  # buy/sell
    quantity: float
    price: float
    timestamp: datetime
    broker: str
    asset_type: str

@dataclass
class PnLRecord:
    date: datetime
    strategy_name: str
    daily_pnl: float
    cumulative_pnl: float
    broker: str

# Database Functions
def init_database():
    """Initialize SQLite database for storing trades and PnL data"""
    conn = sqlite3.connect('trading_platform.db')
    
    # Create trades table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT,
            symbol TEXT,
            side TEXT,
            quantity REAL,
            price REAL,
            timestamp TEXT,
            broker TEXT,
            asset_type TEXT
        )
    ''')
    
    # Create pnl table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS pnl_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            strategy_name TEXT,
            daily_pnl REAL,
            cumulative_pnl REAL,
            broker TEXT
        )
    ''')
    
    # Create strategies table
    conn.execute('''
        CREATE TABLE IF NOT EXISTS strategies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            description TEXT,
            asset_type TEXT,
            parameters TEXT,
            active INTEGER,
            created_date TEXT
        )
    ''')
    
    # Create broker_credentials table for storing API credentials
    conn.execute('''
        CREATE TABLE IF NOT EXISTS broker_credentials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            broker_name TEXT UNIQUE,
            client_id TEXT,
            access_token TEXT,
            last_updated TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def save_broker_credentials(broker_name: str, client_id: str, access_token: str):
    """Save broker credentials to database with encryption"""
    conn = sqlite3.connect('trading_platform.db')
    from datetime import datetime
    
    # Encrypt credentials before saving
    encrypted_client_id = encrypt_credential(client_id)
    encrypted_access_token = encrypt_credential(access_token)
    
    conn.execute('''
        INSERT OR REPLACE INTO broker_credentials 
        (broker_name, client_id, access_token, last_updated)
        VALUES (?, ?, ?, ?)
    ''', (broker_name, encrypted_client_id, encrypted_access_token, datetime.now().isoformat()))
    
    conn.commit()
    conn.close()

def load_broker_credentials(broker_name: str):
    """Load broker credentials from environment variables or database"""
    # First, try to load from environment variables
    if broker_name == "Dhan":
        env_client_id = os.getenv('DHAN_CLIENT_ID')
        env_access_token = os.getenv('DHAN_ACCESS_TOKEN')
        if env_client_id and env_access_token:
            return env_client_id, env_access_token
    
    # Fallback to database
    conn = sqlite3.connect('trading_platform.db')
    cursor = conn.execute('''
        SELECT client_id, access_token FROM broker_credentials 
        WHERE broker_name = ?
    ''', (broker_name,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        # Decrypt credentials before returning
        decrypted_client_id = decrypt_credential(result[0])
        decrypted_access_token = decrypt_credential(result[1])
        return decrypted_client_id, decrypted_access_token
    return None, None

# Import Dhan broker connector
try:
    from dhan_connector import DhanBrokerManager, DhanBroker
    DHAN_AVAILABLE = True
except ImportError:
    DHAN_AVAILABLE = False
    st.warning("⚠️ Dhan connector not found. Create dhan_connector.py file for real broker integration.")

# Simple BrokerManager for future extensibility
class BrokerManager:
    def __init__(self):
        self.brokers = {}
    
    def add_broker(self, name, broker_instance):
        self.brokers[name] = broker_instance
    
    def get_connected_brokers(self):
        return [name for name, broker in self.brokers.items() if hasattr(broker, 'is_connected') and broker.is_connected]
    
    def get_all_positions(self):
        all_positions = {}
        for name, broker in self.brokers.items():
            if hasattr(broker, 'is_connected') and broker.is_connected:
                try:
                    positions = broker.get_positions()
                    if positions:
                        all_positions[name] = positions
                except Exception as e:
                    print(f"Error getting positions from {name}: {e}")
        return all_positions

# Other broker classes can be imported here when needed
# try:
#     from broker_connectors import AlpacaBroker, InteractiveBrokersBroker, ZerodhaBroker
#     OTHER_BROKERS_AVAILABLE = True
# except ImportError:
#     OTHER_BROKERS_AVAILABLE = False

BROKERS_AVAILABLE = True  # BrokerManager is always available now

# Initialize session state
if 'strategies' not in st.session_state:
    st.session_state.strategies = []
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'pnl_data' not in st.session_state:
    st.session_state.pnl_data = []
if 'dhan_manager' not in st.session_state and DHAN_AVAILABLE:
    st.session_state.dhan_manager = DhanBrokerManager()
if 'real_positions' not in st.session_state:
    st.session_state.real_positions = {}
if 'real_holdings' not in st.session_state:
    st.session_state.real_holdings = []
if 'broker_manager' not in st.session_state and BROKERS_AVAILABLE:
    st.session_state.broker_manager = BrokerManager()

# Sidebar Navigation
st.sidebar.title("Trading Platform")
page = st.sidebar.selectbox("Navigate", [
    "Dashboard", 
    "Strategy Management", 
    "Backtesting", 
    "Broker Connections",
    "PnL Analytics",
    "Trade History"
])

# Main Content Area
if page == "Dashboard":
    st.title("📈 Trading Dashboard")
    
    # Check if connected to Dhan for real data
    is_dhan_connected = (DHAN_AVAILABLE and 
                        hasattr(st.session_state, 'dhan_manager') and 
                        st.session_state.dhan_manager.is_connected())
    
    if is_dhan_connected:
        st.success("🔗 Connected to Dhan - Showing Real Data")
        dhan_broker = st.session_state.dhan_manager.get_broker()
        
        # Initialize variables
        funds_data = {}
        positions_data = []
        holdings_data = []
        error_messages = []
        
        # Get real account data with proper error handling
        try:
            funds_data = dhan_broker.get_funds()
        except Exception as e:
            error_messages.append(f"Funds API error: {str(e)}")
            st.error(f"❌ Error fetching funds: {str(e)}")
        
        try:
            positions_data = dhan_broker.get_positions()
        except Exception as e:
            error_messages.append(f"Positions API error: {str(e)}")
            st.error(f"❌ Error fetching positions: {str(e)}")
            
        try:
            holdings_data = dhan_broker.get_holdings()
        except Exception as e:
            error_messages.append(f"Holdings API error: {str(e)}")
            st.error(f"❌ Error fetching holdings: {str(e)}")
        
        # Calculate real portfolio metrics
        total_portfolio_value = 0
        total_pnl = 0
        available_balance = 0
        
        # Calculate from positions
        if positions_data:
            for position in positions_data:
                total_portfolio_value += position.market_value
                total_pnl += position.unrealized_pnl
        
        # Calculate from holdings  
        if holdings_data:
            for holding in holdings_data:
                total_portfolio_value += holding.market_value
                total_pnl += holding.pnl
        
        # Get available balance from funds data
        if isinstance(funds_data, list) and len(funds_data) > 0:
            # Sum available balance from all segments
            for fund in funds_data:
                if isinstance(fund, dict):
                    # Check if nested under 'data'
                    fund_item = fund.get('data', fund) if 'data' in fund else fund
                    
                    fund_balance = fund_item.get('availableBalance', 0)
                    if not fund_balance or fund_balance == 0:  # Handle None or 0
                        fund_balance = fund_item.get('availabelBalance', 0)  # Try misspelled version
                    
                    if fund_balance:
                        available_balance += float(fund_balance)
        elif isinstance(funds_data, dict):
            # Check if the data is nested under 'data' key
            if 'data' in funds_data and isinstance(funds_data['data'], dict):
                fund_data = funds_data['data']
            else:
                fund_data = funds_data
            
            # Check for both spellings, handling None values correctly
            fund_balance = fund_data.get('availableBalance', 0)
            if fund_balance == 0:  # Check if zero or None
                fund_balance = fund_data.get('availabelBalance', 0)  # Try misspelled version
                
            available_balance = float(fund_balance) if fund_balance else 0
        
        
        # Display real metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pnl_color = "normal" if total_pnl >= 0 else "inverse"
            st.metric(
                "Portfolio Value", 
                f"₹{total_portfolio_value:,.2f}", 
                f"₹{total_pnl:,.2f}",
                delta_color=pnl_color
            )
            
        with col2:
            st.metric("Available Balance", f"₹{available_balance:,.2f}")
            
        with col3:
            active_positions = len([p for p in positions_data if p.quantity != 0])
            st.metric("Active Positions", active_positions)
            
        with col4:
            total_holdings = len([h for h in holdings_data if h.quantity > 0])
            st.metric("Holdings", total_holdings)
        
        # Show detailed breakdowns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Current Positions")
            if positions_data:
                real_pos_data = []
                for pos in positions_data:
                    if pos.quantity != 0:
                        pnl_color = "🟢" if pos.unrealized_pnl >= 0 else "🔴"
                        real_pos_data.append({
                            'Symbol': pos.symbol,
                            'Qty': int(pos.quantity),
                            'Avg Cost': f"₹{pos.avg_cost:.2f}",
                            'Market Value': f"₹{pos.market_value:.2f}",
                            'P&L': f"{pnl_color} ₹{pos.unrealized_pnl:.2f}",
                            'Product': pos.product_type
                        })
                
                if real_pos_data:
                    st.dataframe(pd.DataFrame(real_pos_data), use_container_width=True)
                    
                    # Summary
                    total_position_value = sum(p.market_value for p in positions_data if p.quantity != 0)
                    total_position_pnl = sum(p.unrealized_pnl for p in positions_data if p.quantity != 0)
                    st.info(f"💰 Positions Total: ₹{total_position_value:,.2f} | P&L: ₹{total_position_pnl:,.2f}")
                else:
                    st.info("No active positions")
            else:
                st.info("No positions data available")
        
        with col2:
            st.subheader("🏦 Holdings")
            if holdings_data:
                real_hold_data = []
                for hold in holdings_data:
                    if hold.quantity > 0:
                        pnl_color = "🟢" if hold.pnl >= 0 else "🔴"
                        real_hold_data.append({
                            'Symbol': hold.symbol,
                            'Qty': int(hold.quantity),
                            'Avg Cost': f"₹{hold.avg_cost:.2f}",
                            'Current': f"₹{hold.current_price:.2f}",
                            'Market Value': f"₹{hold.market_value:.2f}",
                            'P&L': f"{pnl_color} ₹{hold.pnl:.2f}"
                        })
                
                if real_hold_data:
                    st.dataframe(pd.DataFrame(real_hold_data), use_container_width=True, hide_index=True)
                    
                    # Summary
                    total_holdings_value = sum(h.market_value for h in holdings_data if h.quantity > 0)
                    total_holdings_pnl = sum(h.pnl for h in holdings_data if h.quantity > 0)
                    st.info(f"🏦 Holdings Total: ₹{total_holdings_value:,.2f} | P&L: ₹{total_holdings_pnl:,.2f}")
                else:
                    st.info("No holdings found")
            else:
                st.info("No holdings data available")
        
        # Available Funds Section
        if funds_data:
            st.subheader("💰 Available Funds")
            
            if isinstance(funds_data, list):
                # Handle list of fund segments (each with nested structure)
                funds_display = []
                total_available = 0
                total_utilized = 0
                
                for fund in funds_data:
                    if isinstance(fund, dict):
                        # Extract nested data
                        fund_item = fund.get('data', fund) if 'data' in fund else fund
                        
                        segment_name = fund_item.get('segmentName', 'Trading Account')
                        
                        # Handle misspelled field
                        available = fund_item.get('availableBalance', 0)
                        if not available or available == 0:
                            available = fund_item.get('availabelBalance', 0)
                        available = float(available) if available else 0
                        
                        utilized = float(fund_item.get('utilizedAmount', 0))
                        sod_limit = float(fund_item.get('sodLimit', 0))
                        
                        total_available += available
                        total_utilized += utilized
                        
                        funds_display.append({
                            'Segment': segment_name,
                            'Available': f"₹{available:,.2f}",
                            'Utilized': f"₹{utilized:,.2f}",
                            'SOD Limit': f"₹{sod_limit:,.2f}"
                        })
                
                if funds_display:
                    st.dataframe(pd.DataFrame(funds_display), use_container_width=True)
                    
                    # Summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Available", f"₹{total_available:,.2f}")
                    with col2:
                        st.metric("Total Utilized", f"₹{total_utilized:,.2f}")
                    with col3:
                        utilization_pct = (total_utilized / (total_available + total_utilized) * 100) if (total_available + total_utilized) > 0 else 0
                        st.metric("Utilization %", f"{utilization_pct:.1f}%")
                        
            elif isinstance(funds_data, dict):
                # Single fund object - handle nested structure
                fund_data = funds_data.get('data', funds_data) if 'data' in funds_data else funds_data
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    available = fund_data.get('availableBalance', 0)
                    if available == 0:
                        available = fund_data.get('availabelBalance', 0)  # Try misspelled version
                    st.metric("Available Balance", f"₹{float(available):,.2f}")
                    
                with col2:
                    utilized = float(fund_data.get('utilizedAmount', 0))
                    st.metric("Utilized Amount", f"₹{utilized:,.2f}")
                    
                with col3:
                    sod_limit = float(fund_data.get('sodLimit', 0))
                    st.metric("SOD Limit", f"₹{sod_limit:,.2f}")
                
                # Additional fund details if available
                other_fields = {}
                for key, value in fund_data.items():
                    if key not in ['availableBalance', 'availabelBalance', 'utilizedAmount', 'sodLimit', 'dhanClientId'] and value is not None:
                        other_fields[key] = value
                
                if other_fields:
                    with st.expander("Additional Fund Details"):
                        st.json(other_fields)
        
        # Error Summary
        if error_messages:
            with st.expander("⚠️ API Errors", expanded=False):
                for error in error_messages:
                    st.error(error)
        
        # Debug Information
        with st.expander("🔍 Debug Information", expanded=False):
            st.write("**Portfolio Calculation Summary:**")
            st.write(f"- Total Portfolio Value: ₹{total_portfolio_value:,.2f}")
            st.write(f"- Total P&L: ₹{total_pnl:,.2f}")
            st.write(f"- Available Balance: ₹{available_balance:,.2f}")
            st.write(f"- Active Positions: {len([p for p in positions_data if p.quantity != 0])}")
            st.write(f"- Total Holdings: {len([h for h in holdings_data if h.quantity > 0])}")
            
            st.write("**Raw API Responses:**")
            st.write("Funds Data Type:", type(funds_data))
            st.json({"funds_sample": funds_data})
            
            if positions_data:
                st.write("Positions Sample:")
                st.json({"position_sample": vars(positions_data[0]) if positions_data else {}})
            
            if holdings_data:
                st.write("Holdings Sample:")
                st.json({"holding_sample": vars(holdings_data[0]) if holdings_data else {}})
    
    else:
        st.info("📌 Connect to Dhan broker to see real portfolio data")
        
        # Show demo data when not connected
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Portfolio Value", "₹12,54,300", "2.3%")
        with col2:
            st.metric("Daily PnL", "+₹12,470", "0.99%")
        with col3:
            st.metric("Active Strategies", "5", "1")
        with col4:
            st.metric("Demo Mode", "ON", "Connect Dhan")
    
    # Performance Chart (Demo data for now)
    st.subheader("📈 Portfolio Performance")
    
    # Generate sample data for demo
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    if is_dhan_connected and total_portfolio_value > 0:
        # If connected, show recent performance based on current portfolio value
        portfolio_values = []
        base_value = total_portfolio_value
        
        for i in range(len(dates)):
            # Simulate daily changes
            daily_change = np.random.normal(0.001, 0.02)  # 0.1% daily average with 2% volatility
            if i == 0:
                portfolio_values.append(base_value * 0.9)  # Start 10% lower
            else:
                portfolio_values.append(portfolio_values[-1] * (1 + daily_change))
        
        # Adjust to end at current value
        final_adjustment = total_portfolio_value / portfolio_values[-1]
        portfolio_values = [val * final_adjustment for val in portfolio_values]
        
        chart_title = "Portfolio Performance (Simulated Historical + Current)"
        chart_color = '#00cc96'
    else:
        # Demo data
        portfolio_values = 100000 + np.cumsum(np.random.normal(100, 500, len(dates)))
        chart_title = "Portfolio Performance (Demo Data)"
        chart_color = '#ff7f0e'
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=portfolio_values,
        mode='lines',
        name='Portfolio Value',
        line=dict(color=chart_color, width=2)
    ))
    
    fig.update_layout(
        title=chart_title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value (₹)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if not is_dhan_connected:
        # Recent Activity (Demo)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Recent Trades (Demo)")
            sample_trades = pd.DataFrame({
                'Symbol': ['RELIANCE', 'HDFC', 'TCS', 'INFY'],
                'Side': ['BUY', 'SELL', 'BUY', 'BUY'],
                'Quantity': [100, 50, 200, 10],
                'Price': ['₹2,450.25', '₹1,650.50', '₹3,420.75', '₹1,430.30'],
                'Strategy': ['Momentum', 'Mean Reversion', 'Index', 'Tech Trend']
            })
            st.dataframe(sample_trades, use_container_width=True)
        
        with col2:
            st.subheader("Strategy Performance (Demo)")
            strategy_perf = pd.DataFrame({
                'Strategy': ['Momentum', 'Mean Reversion', 'Index', 'Tech Trend'],
                'Returns (%)': [12.5, -2.3, 8.7, 15.2],
                'Trades': [45, 23, 12, 18],
                'Win Rate (%)': [65, 58, 75, 70]
            })
            st.dataframe(strategy_perf, use_container_width=True)

elif page == "Strategy Management":
    st.title("🎯 Strategy Management")
    
    # Add New Strategy
    st.subheader("Add New Strategy")
    
    col1, col2 = st.columns(2)
    
    with col1:
        strategy_name = st.text_input("Strategy Name")
        asset_type = st.selectbox("Asset Type", [
            "Stocks", 
            "Commodities (Cash)", 
            "Commodities (Futures)"
        ])
        
    with col2:
        description = st.text_area("Description")
        
    # Strategy Parameters
    st.subheader("Strategy Parameters")
    
    # Dynamic parameter inputs based on strategy type
    params = {}
    if asset_type == "Stocks":
        params['lookback_period'] = st.number_input("Lookback Period (days)", min_value=1, value=20)
        params['stop_loss'] = st.number_input("Stop Loss (%)", min_value=0.0, value=2.0)
        params['take_profit'] = st.number_input("Take Profit (%)", min_value=0.0, value=5.0)
        
    elif "Commodities" in asset_type:
        params['contract_size'] = st.number_input("Contract Size", min_value=1, value=1000)
        params['margin_requirement'] = st.number_input("Margin Requirement (%)", min_value=0.0, value=10.0)
        params['rollover_days'] = st.number_input("Rollover Days Before Expiry", min_value=1, value=5)
    
    if st.button("Add Strategy"):
        if strategy_name:
            new_strategy = Strategy(
                name=strategy_name,
                description=description,
                asset_type=asset_type,
                parameters=params,
                created_date=datetime.now()
            )
            st.session_state.strategies.append(new_strategy)
            st.success(f"Strategy '{strategy_name}' added successfully!")
        else:
            st.error("Please enter a strategy name")
    
    # Existing Strategies
    st.subheader("Existing Strategies")
    
    if st.session_state.strategies:
        for i, strategy in enumerate(st.session_state.strategies):
            with st.expander(f"{strategy.name} - {strategy.asset_type}"):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Description:** {strategy.description}")
                    st.write(f"**Parameters:** {strategy.parameters}")
                    
                with col2:
                    active = st.checkbox("Active", value=strategy.active, key=f"active_{i}")
                    strategy.active = active
                    
                with col3:
                    if st.button("Delete", key=f"delete_{i}"):
                        st.session_state.strategies.pop(i)
                        st.rerun()
    else:
        st.info("No strategies configured yet. Add your first strategy above!")

elif page == "Backtesting":
    st.title("🔄 Backtesting Engine")
    
    # Backtesting Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Backtest Configuration")
        
        if st.session_state.strategies:
            strategy_names = [s.name for s in st.session_state.strategies]
            selected_strategy = st.selectbox("Select Strategy", strategy_names)
            
            start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
            end_date = st.date_input("End Date", value=datetime.now())
            
            initial_capital = st.number_input("Initial Capital ($)", min_value=1000, value=100000)
            transaction_cost = st.number_input("Transaction Cost per Trade ($)", min_value=0.0, value=10.0)
            
            if st.button("Run Backtest"):
                # Simulate backtest results
                st.success("Backtest completed successfully!")
                
                # Generate sample backtest data
                days = (end_date - start_date).days
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                
                # Simulate returns
                daily_returns = np.random.normal(0.001, 0.02, len(dates))
                cumulative_returns = np.cumprod(1 + daily_returns)
                portfolio_values = initial_capital * cumulative_returns
                
                # Store results in session state
                st.session_state.backtest_results = {
                    'dates': dates,
                    'portfolio_values': portfolio_values,
                    'daily_returns': daily_returns,
                    'strategy': selected_strategy
                }
        else:
            st.warning("Please add at least one strategy before backtesting.")
    
    with col2:
        st.subheader("Backtest Metrics")
        
        if 'backtest_results' in st.session_state:
            results = st.session_state.backtest_results
            
            final_value = results['portfolio_values'][-1]
            total_return = (final_value - initial_capital) / initial_capital * 100
            
            # Calculate metrics
            daily_returns = results['daily_returns']
            sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            max_drawdown = np.min(np.minimum.accumulate(results['portfolio_values']) / np.maximum.accumulate(results['portfolio_values']) - 1) * 100
            
            col1_metrics, col2_metrics = st.columns(2)
            
            with col1_metrics:
                st.metric("Total Return", f"{total_return:.2f}%")
                st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
                
            with col2_metrics:
                st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                st.metric("Final Value", f"${final_value:,.2f}")
    
    # Backtest Results Chart
    if 'backtest_results' in st.session_state:
        st.subheader("Backtest Results")
        
        results = st.session_state.backtest_results
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results['dates'],
            y=results['portfolio_values'],
            mode='lines',
            name=f"{results['strategy']} Performance",
            line=dict(color='#636EFA', width=2)
        ))
        
        fig.update_layout(
            title=f"Backtest Results: {results['strategy']}",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

elif page == "Broker Connections":
    st.title("🔗 Broker Connections")
    
    if not DHAN_AVAILABLE:
        st.error("⚠️ Dhan connector not available. Please create dhan_connector.py file")
        st.code("pip install requests python-dotenv")
        st.stop()
    
    # Current: Dhan Only (Other brokers can be added later)
    st.info("🇮🇳 Currently supporting Dhan broker. More brokers coming soon!")
    
    # Dhan Broker Setup within BrokerManager
    st.subheader("🇮🇳 Dhan API Setup")
    st.info("📌 Get your credentials from: https://dhanhq.co/api/")
    
    # Load saved credentials
    saved_client_id, saved_access_token = load_broker_credentials("Dhan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dhan_client_id = st.text_input("Dhan Client ID", 
                                      value=saved_client_id or "",
                                      key="broker_dhan_client_id", 
                                      help="Your Dhan trading account client ID")
        
    with col2:
        dhan_access_token = st.text_input("Access Token", 
                                         value=saved_access_token or "",
                                         type="password", 
                                         key="broker_dhan_access_token",
                                         help="Generate access token from Dhan API portal")
    
    # Connection and management buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔌 Connect to Dhan", key="broker_connect_dhan"):
            if dhan_client_id and dhan_access_token:
                with st.spinner("Connecting to Dhan via BrokerManager..."):
                    try:
                        if st.session_state.dhan_manager.connect(dhan_client_id, dhan_access_token):
                            # Save credentials on successful connection
                            save_broker_credentials("Dhan", dhan_client_id, dhan_access_token)
                            
                            # Get broker instance after successful connection
                            dhan_broker = st.session_state.dhan_manager.get_broker()
                            # Add to BrokerManager
                            st.session_state.broker_manager.add_broker("Dhan", dhan_broker)
                            st.success("✅ Connected to Dhan successfully! Credentials saved.")
                            
                            # Show account info
                            account_info = dhan_broker.get_account_info()
                            if account_info:
                                # Debug: Show actual response structure
                                st.write("**Account Info Response:**")
                                st.json(account_info)
                                
                                # Extract relevant fields with different possible field names
                                balance_value = (account_info.get('availableBalance', 'N/A') or 
                                               account_info.get('availabelBalance', 'N/A'))
                                display_info = {
                                    "Client ID": account_info.get('dhanClientId') or account_info.get('clientId') or dhan_client_id,
                                    "Available Balance": balance_value,
                                    "SOD Limit": account_info.get('sodLimit', 'N/A'),
                                    "Status": "Connected via BrokerManager"
                                }
                                st.json(display_info)
                        else:
                            st.error("❌ Failed to connect to Dhan. Check your credentials.")
                    except Exception as e:
                        st.error(f"❌ Connection error: {str(e)}")
            else:
                st.warning("Please enter both Client ID and Access Token")
    
    with col2:
        if st.button("🔄 Update Credentials", key="broker_update_dhan"):
            if dhan_client_id and dhan_access_token:
                save_broker_credentials("Dhan", dhan_client_id, dhan_access_token)
                st.success("✅ Credentials updated successfully!")
                st.rerun()
            else:
                st.warning("Please enter both Client ID and Access Token")
    
    with col3:
        if st.button("📊 Get Positions", key="broker_dhan_positions"):
            if "Dhan" in st.session_state.broker_manager.brokers:
                dhan_broker = st.session_state.broker_manager.brokers["Dhan"]
                if st.session_state.dhan_manager.is_connected():
                    try:
                        positions = dhan_broker.get_positions()
                        if positions:
                            pos_data = []
                            for pos in positions:
                                if pos.quantity != 0:
                                    pos_data.append({
                                        "Symbol": pos.symbol,
                                        "Quantity": int(pos.quantity),
                                        "Avg Cost": f"₹{pos.avg_cost:.2f}",
                                        "Market Value": f"₹{pos.market_value:.2f}",
                                        "P&L": f"₹{pos.unrealized_pnl:.2f}",
                                        "Product": pos.product_type
                                    })
                            
                            if pos_data:
                                st.dataframe(pd.DataFrame(pos_data), use_container_width=True)
                            else:
                                st.info("No open positions found")
                        else:
                            st.info("No positions data available")
                    except Exception as e:
                        st.error(f"❌ Error fetching positions: {str(e)}")
                else:
                    st.warning("Not connected to Dhan")
            else:
                st.warning("Dhan not added to BrokerManager")
    
    with col4:
        if st.button("🏦 Get Holdings", key="broker_dhan_holdings"):
            if "Dhan" in st.session_state.broker_manager.brokers:
                dhan_broker = st.session_state.broker_manager.brokers["Dhan"]
                if st.session_state.dhan_manager.is_connected():
                    holdings = dhan_broker.get_holdings()
                    if holdings:
                        hold_data = []
                        for hold in holdings:
                            if hold.quantity > 0:
                                hold_data.append({
                                    "Symbol": hold.symbol,
                                    "Quantity": int(hold.quantity),
                                    "Avg Cost": f"₹{hold.avg_cost:.2f}",
                                    "Current Price": f"₹{hold.current_price:.2f}",
                                    "Market Value": f"₹{hold.market_value:.2f}",
                                    "P&L": f"₹{hold.pnl:.2f}"
                                })
                        
                        if hold_data:
                            st.dataframe(pd.DataFrame(hold_data), use_container_width=True)
                        else:
                            st.info("No holdings found")
                    else:
                        st.info("No holdings data available")
                else:
                    st.warning("Not connected to Dhan")
            else:
                st.warning("Dhan not added to BrokerManager")
    
    # Disconnect button in a separate row
    st.write("")
    if st.button("🔌 Disconnect from Dhan", key="broker_disconnect_dhan"):
        if "Dhan" in st.session_state.broker_manager.brokers:
            st.session_state.dhan_manager.disconnect()
            # Remove from BrokerManager
            del st.session_state.broker_manager.brokers["Dhan"]
            st.info("Disconnected Dhan from BrokerManager")
            st.rerun()
        else:
            st.warning("Dhan not connected")
    
    # Connection Status Summary
    st.subheader("📊 Connection Status")
    
    if hasattr(st.session_state, 'broker_manager'):
        connected_brokers = st.session_state.broker_manager.get_connected_brokers()
        
        if connected_brokers:
            st.success(f"✅ Connected brokers: {', '.join(connected_brokers)}")
            
            # Quick action buttons for connected brokers
            st.subheader("🚀 Quick Actions")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Refresh All Positions", key="refresh_all_positions"):
                    all_positions = st.session_state.broker_manager.get_all_positions()
                    st.session_state.real_positions = all_positions
                    
                    # Display combined positions
                    combined_positions = []
                    for broker_name, positions in all_positions.items():
                        for pos in positions:
                            if hasattr(pos, 'quantity') and pos.quantity != 0:
                                combined_positions.append({
                                    "Broker": broker_name,
                                    "Symbol": pos.symbol,
                                    "Quantity": int(pos.quantity),
                                    "Market Value": f"₹{pos.market_value:.2f}",
                                    "P&L": f"₹{pos.unrealized_pnl:.2f}"
                                })
                    
                    if combined_positions:
                        st.dataframe(pd.DataFrame(combined_positions), use_container_width=True)
                        st.success("Positions refreshed successfully!")
                    else:
                        st.info("No positions found across connected brokers")
            
            with col2:
                if st.button("📋 View All Broker Status", key="broker_status"):
                    st.write("**Connected Brokers:**")
                    for broker_name in connected_brokers:
                        broker = st.session_state.broker_manager.brokers[broker_name]
                        status = "🟢 Connected" if hasattr(broker, 'is_connected') and broker.is_connected else "🔴 Disconnected"
                        st.write(f"- **{broker_name}**: {status}")
        else:
            st.info("No brokers connected to BrokerManager")
            st.write("Connect Dhan above to get started!")
    else:
        st.error("BrokerManager not initialized")
    
    # Future brokers placeholder
    st.subheader("🔮 Coming Soon")
    st.info("Additional brokers (Zerodha, Interactive Brokers, Alpaca) will be added here in future updates!")

elif page == "PnL Analytics":
    st.title("📊 PnL Analytics")
    
    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", value=datetime.now() - timedelta(days=30))
    with col2:
        end_date = st.date_input("To", value=datetime.now())
    
    # Generate sample PnL data
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    strategies = ['Momentum', 'Mean Reversion', 'Index', 'Commodity Trend']
    
    pnl_data = []
    for strategy in strategies:
        for date in dates:
            daily_pnl = np.random.normal(100, 500)  # Random daily PnL
            pnl_data.append({
                'Date': date,
                'Strategy': strategy,
                'Daily_PnL': daily_pnl,
                'Cumulative_PnL': daily_pnl  # Simplified for demo
            })
    
    df_pnl = pd.DataFrame(pnl_data)
    
    # Calculate cumulative PnL properly
    for strategy in strategies:
        mask = df_pnl['Strategy'] == strategy
        df_pnl.loc[mask, 'Cumulative_PnL'] = df_pnl.loc[mask, 'Daily_PnL'].cumsum()
    
    # PnL Charts
    st.subheader("Daily PnL by Strategy")
    
    fig = px.line(df_pnl, x='Date', y='Cumulative_PnL', color='Strategy',
                  title='Cumulative PnL by Strategy')
    st.plotly_chart(fig, use_container_width=True)
    
    # Strategy Comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Strategy Performance Summary")
        
        summary = df_pnl.groupby('Strategy').agg({
            'Daily_PnL': ['sum', 'mean', 'std'],
            'Cumulative_PnL': 'last'
        }).round(2)
        
        summary.columns = ['Total PnL', 'Avg Daily PnL', 'Daily Volatility', 'Final PnL']
        st.dataframe(summary)
    
    with col2:
        st.subheader("Risk Metrics")
        
        risk_metrics = []
        for strategy in strategies:
            strategy_data = df_pnl[df_pnl['Strategy'] == strategy]['Daily_PnL']
            
            # Calculate risk metrics
            var_95 = np.percentile(strategy_data, 5)  # Value at Risk (95%)
            sharpe = strategy_data.mean() / strategy_data.std() if strategy_data.std() > 0 else 0
            max_drawdown = (strategy_data.cumsum().cummax() - strategy_data.cumsum()).max()
            
            risk_metrics.append({
                'Strategy': strategy,
                'VaR (95%)': round(var_95, 2),
                'Sharpe Ratio': round(sharpe, 3),
                'Max Drawdown': round(max_drawdown, 2)
            })
        
        risk_df = pd.DataFrame(risk_metrics)
        st.dataframe(risk_df, use_container_width=True)

elif page == "Trade History":
    st.title("📋 Trade History")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        strategy_filter = st.selectbox("Filter by Strategy", ["All"] + [s.name for s in st.session_state.strategies])
    
    with col2:
        asset_filter = st.selectbox("Filter by Asset Type", ["All", "Stocks", "Commodities (Cash)", "Commodities (Futures)"])
    
    with col3:
        broker_filter = st.selectbox("Filter by Broker", ["All", "Interactive Brokers", "Alpaca", "TD Ameritrade"])
    
    # Generate sample trade data
    sample_trades = []
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'GOLD', 'SILVER', 'CL=F', 'GC=F', 'ES=F']
    
    for i in range(50):  # Generate 50 sample trades
        trade = {
            'Date': datetime.now() - timedelta(days=np.random.randint(0, 90)),
            'Strategy': np.random.choice(['Momentum', 'Mean Reversion', 'Index', 'Commodity Trend']),
            'Symbol': np.random.choice(symbols),
            'Side': np.random.choice(['BUY', 'SELL']),
            'Quantity': np.random.randint(10, 1000),
            'Price': round(np.random.uniform(50, 500), 2),
            'Broker': np.random.choice(['Interactive Brokers', 'Alpaca', 'TD Ameritrade']),
            'PnL': round(np.random.normal(50, 200), 2)
        }
        sample_trades.append(trade)
    
    trades_df = pd.DataFrame(sample_trades)
    trades_df = trades_df.sort_values('Date', ascending=False)
    
    # Apply filters
    if strategy_filter != "All":
        trades_df = trades_df[trades_df['Strategy'] == strategy_filter]
    
    if broker_filter != "All":
        trades_df = trades_df[trades_df['Broker'] == broker_filter]
    
    # Display trades table
    st.subheader(f"Trade History ({len(trades_df)} trades)")
    
    # Format the dataframe for display
    display_df = trades_df.copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
    display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
    display_df['PnL'] = display_df['PnL'].apply(lambda x: f"${x:.2f}" if x >= 0 else f"-${abs(x):.2f}")
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Trade Statistics
    st.subheader("Trade Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trades = len(trades_df)
        st.metric("Total Trades", total_trades)
    
    with col2:
        winning_trades = len(trades_df[trades_df['PnL'] > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with col3:
        total_pnl = trades_df['PnL'].sum()
        st.metric("Total PnL", f"${total_pnl:.2f}")
    
    with col4:
        avg_pnl = trades_df['PnL'].mean()
        st.metric("Average PnL", f"${avg_pnl:.2f}")

# Initialize database
init_database()

# Footer
st.markdown("---")
st.markdown("**Personal Trading Platform** - Built with Streamlit")
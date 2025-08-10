# dhan_connector.py - Corrected Version
from dhanhq import dhanhq
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import pandas as pd
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Position:
    symbol: str
    quantity: float
    avg_cost: float
    market_value: float
    unrealized_pnl: float
    product_type: str = ""
    
@dataclass
class Order:
    order_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str
    status: str
    filled_qty: float = 0
    avg_fill_price: float = 0
    product_type: str = ""

@dataclass
class Holdings:
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    pnl: float

class DhanBroker:
    """Dhan API connector for Indian markets using dhanhq package"""
    
    def __init__(self, client_id: str, access_token: str):
        self.client_id = client_id
        self.access_token = access_token
        self.is_connected = False
        self.account_info = {}
        self.dhan_client = None
        
    def connect(self) -> bool:
        """Connect to Dhan API using dhanhq package"""
        try:
            self.dhan_client = dhanhq(self.client_id, self.access_token)
            
            # Test connection by getting fund limits
            fund_response = self.dhan_client.get_fund_limits()
            logger.info(f"Fund response type: {type(fund_response)}")
            logger.info(f"Fund response: {fund_response}")
            
            if fund_response is not None:
                self.account_info = self._parse_response(fund_response)
                self.is_connected = True
                logger.info(f"Connected to Dhan - Client ID: {self.client_id}")
                return True
            else:
                logger.error("Failed to connect to Dhan: No response from fund limits API")
                return False
                
        except Exception as e:
            logger.error(f"Dhan connection error: {e}")
            return False
    
    def _parse_response(self, response: Any) -> Union[Dict, List]:
        """Parse API response (handles both string JSON and dict/list)"""
        if isinstance(response, str):
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                return {"raw_response": response}
        elif isinstance(response, (dict, list)):
            return response
        else:
            logger.warning(f"Unexpected response type: {type(response)}")
            return {"raw_response": str(response)}
            
    def disconnect(self):
        """Disconnect from Dhan"""
        self.is_connected = False
        self.dhan_client = None
        logger.info("Disconnected from Dhan")
        
    def get_account_info(self) -> Dict:
        """Get Dhan account information"""
        if not self.is_connected or not self.dhan_client:
            return {}
            
        try:
            # Try get_fund_limits for account info
            response = self.dhan_client.get_fund_limits()
            parsed_response = self._parse_response(response)
            
            # Also try to get additional profile info if available
            try:
                profile_response = self.dhan_client.get_profile()
                if profile_response:
                    profile_data = self._parse_response(profile_response)
                    if isinstance(profile_data, dict):
                        parsed_response.update(profile_data)
            except Exception as e:
                logger.info(f"Profile API not available or failed: {e}")
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
        
    def get_funds(self) -> Union[Dict, List]:
        """Get account funds information"""
        if not self.is_connected or not self.dhan_client:
            return {}
            
        try:
            response = self.dhan_client.get_fund_limits()
            logger.info(f"Funds API response type: {type(response)}")
            logger.info(f"Funds API response: {response}")
            
            parsed_response = self._parse_response(response)
            
            # Handle different possible response structures
            if isinstance(parsed_response, list):
                # If it's a list of fund segments
                return parsed_response
            elif isinstance(parsed_response, dict):
                # If it's a single fund object, wrap in list for consistency
                return [parsed_response]
            else:
                logger.error(f"Unexpected funds response structure: {type(parsed_response)}")
                return []
                
        except Exception as e:
            logger.error(f"Error getting funds: {e}")
            return {}
        
    def get_positions(self) -> List[Position]:
        """Get current positions from Dhan"""
        if not self.is_connected or not self.dhan_client:
            return []
            
        try:
            positions_response = self.dhan_client.get_positions()
            logger.info(f"Positions API response type: {type(positions_response)}")
            logger.info(f"Positions API response: {positions_response}")
            
            positions_data = self._parse_response(positions_response)
            positions = []
            
            if isinstance(positions_data, list):
                for pos in positions_data:
                    if isinstance(pos, dict):
                        # Handle different possible field names
                        net_qty = self._safe_float(pos.get('netQty') or pos.get('quantity') or pos.get('qty'), 0)
                        
                        if net_qty != 0:  # Only include non-zero positions
                            ltp = self._safe_float(pos.get('ltp') or pos.get('lastPrice') or pos.get('currentPrice'), 0)
                            avg_price = self._safe_float(pos.get('avgPrice') or pos.get('avgCost') or pos.get('averagePrice'), 0)
                            
                            position = Position(
                                symbol=pos.get('tradingSymbol') or pos.get('symbol') or '',
                                quantity=net_qty,
                                avg_cost=avg_price,
                                market_value=abs(net_qty) * ltp,
                                unrealized_pnl=self._safe_float(pos.get('unrealizedPnl') or pos.get('pnl'), 0),
                                product_type=pos.get('productType') or pos.get('product') or ''
                            )
                            positions.append(position)
                            logger.info(f"Added position: {position.symbol} - Qty: {position.quantity}, Value: {position.market_value}")
            else:
                logger.warning(f"Positions data is not a list: {type(positions_data)}")
                    
            logger.info(f"Total positions parsed: {len(positions)}")
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {value} to float, using default {default}")
            return default
        
    def get_holdings(self) -> List[Holdings]:
        """Get holdings (delivery positions) from Dhan"""
        if not self.is_connected or not self.dhan_client:
            return []
            
        try:
            holdings_response = self.dhan_client.get_holdings()
            logger.info(f"Holdings API response type: {type(holdings_response)}")
            logger.info(f"Holdings API response: {holdings_response}")
            
            holdings_data = self._parse_response(holdings_response)
            holdings = []
            
            # Handle nested structure - check if data is under 'data' key
            if isinstance(holdings_data, dict) and 'data' in holdings_data:
                holdings_list = holdings_data['data']
            elif isinstance(holdings_data, list):
                holdings_list = holdings_data
            else:
                holdings_list = []
            
            logger.info(f"Holdings list type: {type(holdings_list)}, length: {len(holdings_list) if hasattr(holdings_list, '__len__') else 'N/A'}")
            
            if isinstance(holdings_list, list):
                for holding in holdings_list:
                    if isinstance(holding, dict):
                        # Handle different possible field names
                        total_qty = self._safe_float(holding.get('totalQty') or holding.get('quantity') or holding.get('qty'), 0)
                        
                        if total_qty > 0:  # Only include positive holdings
                            # Try multiple field names for current price (with live data feed)
                            ltp = self._safe_float(
                                holding.get('ltp') or 
                                holding.get('lastPrice') or 
                                holding.get('currentPrice') or
                                holding.get('close') or
                                holding.get('lastTradedPrice'), 0)
                            
                            # Try multiple field names for average cost
                            avg_cost = self._safe_float(
                                holding.get('avgCostPrice') or 
                                holding.get('avgPrice') or 
                                holding.get('averagePrice') or
                                holding.get('costPrice') or
                                holding.get('buyPrice'), 0)
                            
                            # With live data feed, try to get real-time quote if LTP not in holdings
                            symbol = holding.get('tradingSymbol') or holding.get('symbol') or ''
                            if ltp == 0 and symbol:
                                try:
                                    quote = self.get_market_quote(symbol)
                                    if isinstance(quote, dict):
                                        ltp = self._safe_float(
                                            quote.get('ltp') or 
                                            quote.get('lastPrice') or 
                                            quote.get('close'), 0)
                                        logger.info(f"Fetched live price for {symbol}: ₹{ltp:.2f}")
                                except Exception as e:
                                    logger.warning(f"Could not fetch live quote for {symbol}: {e}")
                            
                            # If still no LTP available, use avg_cost as fallback
                            if ltp == 0 and avg_cost > 0:
                                ltp = avg_cost
                                logger.info(f"Using avg_cost as current_price for {symbol}: ₹{avg_cost:.2f}")
                            
                            # Calculate market value
                            market_val = total_qty * ltp
                            
                            # Calculate P&L - try direct field first, then calculate
                            pnl = self._safe_float(
                                holding.get('pnl') or 
                                holding.get('unrealizedPnl') or 
                                holding.get('totalPnl') or
                                holding.get('dayPnl'), 0)
                            
                            if pnl == 0 and ltp > 0 and avg_cost > 0:
                                pnl = (ltp - avg_cost) * total_qty
                            
                            hold = Holdings(
                                symbol=holding.get('tradingSymbol') or holding.get('symbol') or '',
                                quantity=total_qty,
                                avg_cost=avg_cost,
                                current_price=ltp,
                                market_value=market_val,
                                pnl=pnl
                            )
                            holdings.append(hold)
                            logger.info(f"Added holding: {hold.symbol} - Qty: {hold.quantity}, Value: {hold.market_value}")
            else:
                logger.warning(f"Holdings data is not a list: {type(holdings_data)}")
                    
            logger.info(f"Total holdings parsed: {len(holdings)}")
            return holdings
            
        except Exception as e:
            logger.error(f"Error getting holdings: {e}")
            return []
        
    def place_order(self, 
                   symbol: str, 
                   side: str, 
                   quantity: int, 
                   order_type: str = "MARKET",
                   product_type: str = "MIS",
                   price: float = 0,
                   exchange: str = "NSE") -> Optional[str]:
        """Place order with Dhan using dhanhq package"""
        if not self.is_connected or not self.dhan_client:
            return None
            
        try:
            # Map our parameters to dhanhq format
            transaction_type = self.dhan_client.BUY if side.upper() == "BUY" else self.dhan_client.SELL
            exchange_segment = self.dhan_client.NSE if exchange == "NSE" else self.dhan_client.BSE
            
            # Map product type
            if product_type == "MIS":
                product = self.dhan_client.INTRA if hasattr(self.dhan_client, 'INTRA') else self.dhan_client.CNC
            elif product_type == "CNC":
                product = self.dhan_client.CNC
            else:
                product = self.dhan_client.NORMAL if hasattr(self.dhan_client, 'NORMAL') else self.dhan_client.CNC
            
            # Map order type
            if order_type.upper() == "MARKET":
                order_type_enum = self.dhan_client.MARKET
                order_price = 0
            elif order_type.upper() == "LIMIT":
                order_type_enum = self.dhan_client.LIMIT
                order_price = price
            else:
                order_type_enum = self.dhan_client.MARKET
                order_price = 0
            
            order_response = self.dhan_client.place_order(
                security_id=symbol,  # dhanhq should handle symbol to security_id mapping
                exchange_segment=exchange_segment,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=order_type_enum,
                product_type=product,
                price=order_price
            )
            
            logger.info(f"Order response: {order_response}")
            
            if order_response:
                parsed_response = self._parse_response(order_response)
                if isinstance(parsed_response, dict):
                    order_id = parsed_response.get('orderId') or parsed_response.get('data', {}).get('orderId')
                    if order_id:
                        logger.info(f"Order placed successfully: {order_id}")
                        return str(order_id)
                        
            logger.error("Order placement failed - no order ID in response")
            return None
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return None
        
    def get_orders(self) -> List[Order]:
        """Get order book from Dhan"""
        if not self.is_connected or not self.dhan_client:
            return []
            
        try:
            orders_response = self.dhan_client.get_order_list()
            logger.info(f"Orders API response: {orders_response}")
            
            orders_data = self._parse_response(orders_response)
            orders = []
            
            if isinstance(orders_data, list):
                for order_info in orders_data:
                    if isinstance(order_info, dict):
                        order = Order(
                            order_id=str(order_info.get('orderId') or order_info.get('id') or ''),
                            symbol=order_info.get('tradingSymbol') or order_info.get('symbol') or '',
                            side=order_info.get('transactionType') or order_info.get('side') or '',
                            quantity=self._safe_float(order_info.get('quantity'), 0),
                            order_type=order_info.get('orderType') or '',
                            status=order_info.get('orderStatus') or order_info.get('status') or '',
                            filled_qty=self._safe_float(order_info.get('filledQty') or order_info.get('filled_quantity'), 0),
                            avg_fill_price=self._safe_float(order_info.get('price') or order_info.get('avgPrice'), 0),
                            product_type=order_info.get('productType') or order_info.get('product') or ''
                        )
                        orders.append(order)
                        
            return orders
            
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
        
    def get_trade_book(self) -> List[Dict]:
        """Get trade book from Dhan"""
        if not self.is_connected or not self.dhan_client:
            return []
            
        try:
            response = self.dhan_client.get_trade_book()
            return self._parse_response(response) if response else []
        except Exception as e:
            logger.error(f"Error getting trade book: {e}")
            return []
        
    def get_market_quote(self, symbol: str, exchange: str = "NSE") -> Dict:
        """Get market quote for a symbol"""
        if not self.is_connected or not self.dhan_client:
            return {}
            
        try:
            exchange_segment = self.dhan_client.NSE if exchange == "NSE" else self.dhan_client.BSE
            
            # Try multiple methods to get quote data
            quote_data = None
            try:
                # Method 1: Try ohlc_data
                quote_data = self.dhan_client.ohlc_data(
                    exchange_segment=exchange_segment,
                    security_id=symbol
                )
            except Exception:
                # Method 2: Try ltp_data if available
                try:
                    quote_data = self.dhan_client.ltp_data(
                        exchange_segment=exchange_segment,
                        security_id=symbol
                    )
                except Exception:
                    pass
            
            if quote_data:
                parsed_quote = self._parse_response(quote_data)
                # Handle nested response structure
                if isinstance(parsed_quote, dict):
                    if 'data' in parsed_quote:
                        parsed_quote = parsed_quote['data']
                return parsed_quote
            return {}
            
        except Exception as e:
            logger.error(f"Error getting market quote for {symbol}: {e}")
            return {}
        
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if not self.is_connected or not self.dhan_client:
            return False
            
        try:
            response = self.dhan_client.cancel_order(order_id=order_id)
            if response:
                logger.info(f"Order cancelled: {order_id}")
                return True
            else:
                logger.error(f"Failed to cancel order: {order_id}")
                return False
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False

# Broker Manager for Dhan
class DhanBrokerManager:
    """Manages Dhan broker connection"""
    
    def __init__(self):
        self.dhan_broker: Optional[DhanBroker] = None
        
    def connect(self, client_id: str, access_token: str) -> bool:
        """Connect to Dhan broker"""
        self.dhan_broker = DhanBroker(client_id, access_token)
        return self.dhan_broker.connect()
        
    def disconnect(self):
        """Disconnect from Dhan broker"""
        if self.dhan_broker:
            self.dhan_broker.disconnect()
            self.dhan_broker = None
            
    def is_connected(self) -> bool:
        """Check if connected to Dhan"""
        return self.dhan_broker is not None and self.dhan_broker.is_connected
        
    def get_broker(self) -> Optional[DhanBroker]:
        """Get the Dhan broker instance"""
        return self.dhan_broker

if __name__ == "__main__":
    # Test the connector
    import os
    
    client_id = os.getenv("DHAN_CLIENT_ID", "YOUR_CLIENT_ID")
    access_token = os.getenv("DHAN_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")
    
    if client_id == "YOUR_CLIENT_ID" or access_token == "YOUR_ACCESS_TOKEN":
        print("Please set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN environment variables")
        exit(1)
    
    dhan = DhanBroker(client_id, access_token)
    
    if dhan.connect():
        print("✅ Connected to Dhan successfully!")
        
        # Test all methods
        print("\n📊 Account Info:")
        account_info = dhan.get_account_info()
        print(json.dumps(account_info, indent=2))
        
        print("\n💰 Funds:")
        funds = dhan.get_funds()
        print(json.dumps(funds, indent=2))
        
        print("\n📈 Positions:")
        positions = dhan.get_positions()
        for pos in positions:
            print(f"  {pos.symbol}: {pos.quantity} @ ₹{pos.avg_cost:.2f} = ₹{pos.market_value:.2f}")
        
        print("\n🏦 Holdings:")
        holdings = dhan.get_holdings()
        for hold in holdings:
            print(f"  {hold.symbol}: {hold.quantity} @ ₹{hold.avg_cost:.2f} = ₹{hold.market_value:.2f}")
        
    else:
        print("❌ Failed to connect to Dhan")
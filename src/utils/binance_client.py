"""
Binance API Client Module
"""
from binance.client import Client
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class BinanceAPI:
    def __init__(self):
        """Initialize Binance API client"""
        self.client = Client(
            os.getenv('BINANCE_API_KEY'),
            os.getenv('BINANCE_API_SECRET')
        )
    
    def get_account_balance(self):
        """Get account balance"""
        return self.client.get_account()
    
    def get_symbol_price(self, symbol):
        """Get current price for a symbol"""
        return self.client.get_symbol_ticker(symbol=symbol)
    
    def place_order(self, symbol, side, order_type, quantity):
        """Place an order"""
        return self.client.create_order(
            symbol=symbol,
            side=side,
            type=order_type,
            quantity=quantity
        )

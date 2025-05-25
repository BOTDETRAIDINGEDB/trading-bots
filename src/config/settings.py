"""
Global Settings Module
"""
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Trading Settings
TRADE_AMOUNT_USDT = float(os.getenv('TRADE_AMOUNT_USDT', 100))
MAX_TRADES = int(os.getenv('MAX_TRADES', 5))
STOP_LOSS_PERCENTAGE = float(os.getenv('STOP_LOSS_PERCENTAGE', 2))
TAKE_PROFIT_PERCENTAGE = float(os.getenv('TAKE_PROFIT_PERCENTAGE', 3))

# Environment
ENV = os.getenv('ENV', 'development')
IS_PRODUCTION = ENV == 'production'

# Trading Pairs
TRADING_PAIRS = {
    'XRP': 'XRPUSDT',
    'SOL': 'SOLUSDT'
}

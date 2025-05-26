#!/usr/bin/env python3
"""
Conector para la API de Binance
Maneja la conexión y operaciones con Binance
"""

import os
import logging
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class BinanceConnector:
    """Clase para manejar la conexión y operaciones con Binance"""
    
    def __init__(self, api_key=None, api_secret=None, testnet=False):
        """Inicializar el conector de Binance"""
        # Cargar credenciales desde variables de entorno si no se proporcionan
        if api_key is None or api_secret is None:
            load_dotenv()
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("API Key y API Secret son requeridos")
        
        # Inicializar cliente de Binance
        self.client = Client(api_key, api_secret, testnet=testnet)
        logger.info("Conector de Binance inicializado")
        
        # Información del exchange
        self.exchange_info = self.client.get_exchange_info()
        self.symbols_info = {s['symbol']: s for s in self.exchange_info['symbols']}
        
    def get_historical_klines(self, symbol, interval, lookback_days=30):
        """Obtener velas históricas"""
        try:
            # Calcular fecha de inicio
            start_time = datetime.now() - timedelta(days=lookback_days)
            start_str = start_time.strftime("%d %b, %Y")
            
            # Obtener datos históricos
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str
            )
            
            # Convertir a DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convertir tipos de datos
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
                
            df.set_index('timestamp', inplace=True)
            return df
            
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Error al obtener datos históricos: {e}")
            return None
    
    def get_account_balance(self, asset='USDT'):
        """Obtener balance de la cuenta"""
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    return float(balance['free'])
            return 0.0
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Error al obtener balance: {e}")
            return 0.0
    
    def get_symbol_price(self, symbol):
        """Obtener precio actual de un símbolo"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Error al obtener precio: {e}")
            return None
    
    def create_order(self, symbol, side, order_type, quantity=None, price=None, test=True):
        """Crear una orden"""
        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type
            }
            
            if quantity:
                params['quantity'] = quantity
                
            if price and order_type != 'MARKET':
                params['price'] = price
                
            if test:
                self.client.create_test_order(**params)
                logger.info(f"Orden de prueba creada: {params}")
                return {'status': 'TEST_SUCCESS', 'params': params}
            else:
                response = self.client.create_order(**params)
                logger.info(f"Orden creada: {response}")
                return response
                
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Error al crear orden: {e}")
            return {'status': 'ERROR', 'message': str(e)}
    
    def get_symbol_info(self, symbol):
        """Obtener información de un símbolo"""
        if symbol in self.symbols_info:
            return self.symbols_info[symbol]
        return None

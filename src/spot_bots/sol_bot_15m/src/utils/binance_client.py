# binance_client.py
import os
import logging
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

logger = logging.getLogger(__name__)

class BinanceAPI:
    """Cliente para interactuar con la API de Binance."""
    
    def __init__(self, api_key=None, api_secret=None):
        """
        Inicializa el cliente de Binance.
        
        Args:
            api_key (str, optional): API Key de Binance. Si no se proporciona, se busca en variables de entorno.
            api_secret (str, optional): API Secret de Binance. Si no se proporciona, se busca en variables de entorno.
        """
        self.api_key = api_key or os.getenv('BINANCE_API_KEY')
        self.api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("API Key y API Secret son requeridos. Proporciónalos como parámetros o configura las variables de entorno BINANCE_API_KEY y BINANCE_API_SECRET.")
        
        self.client = Client(self.api_key, self.api_secret)
        logger.info("Cliente de Binance inicializado correctamente.")
    
    def get_historical_klines(self, symbol, interval, start_time=None, end_time=None, limit=500):
        """
        Obtiene velas históricas de Binance.
        
        Args:
            symbol (str): Par de trading (ej. 'SOLUSDT').
            interval (str): Intervalo de tiempo (ej. '15m', '1h', '1d').
            start_time (int, optional): Tiempo de inicio en milisegundos.
            end_time (int, optional): Tiempo de fin en milisegundos.
            limit (int, optional): Número máximo de velas a obtener.
            
        Returns:
            list: Lista de velas en formato [timestamp, open, high, low, close, volume, ...].
        """
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                startTime=start_time,
                endTime=end_time,
                limit=limit
            )
            logger.info(f"Obtenidas {len(klines)} velas para {symbol} en intervalo {interval}.")
            return klines
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Error al obtener velas históricas: {str(e)}")
            return []
    
    def get_current_price(self, symbol):
        """
        Obtiene el precio actual de un símbolo.
        
        Args:
            symbol (str): Par de trading (ej. 'SOLUSDT').
            
        Returns:
            float: Precio actual o None si hay un error.
        """
        max_retries = 3
        retry_delay = 2  # segundos
        
        for attempt in range(max_retries):
            try:
                # Intentar obtener el precio usando get_symbol_ticker
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                price = float(ticker['price'])
                logger.debug(f"Precio actual de {symbol}: {price}")
                return price
            except (BinanceAPIException, BinanceRequestException) as e:
                logger.warning(f"Intento {attempt+1}/{max_retries}: Error al obtener precio con get_symbol_ticker: {str(e)}")
                
                # Si falla, intentar con get_ticker (método alternativo)
                try:
                    ticker_data = self.client.get_ticker(symbol=symbol)
                    if 'lastPrice' in ticker_data:
                        price = float(ticker_data['lastPrice'])
                        logger.debug(f"Precio actual de {symbol} (método alternativo): {price}")
                        return price
                except Exception as e2:
                    logger.warning(f"Intento {attempt+1}/{max_retries}: Error al obtener precio con get_ticker: {str(e2)}")
                
                # Si aún falla, intentar con get_recent_trades
                try:
                    trades = self.client.get_recent_trades(symbol=symbol, limit=1)
                    if trades and len(trades) > 0:
                        price = float(trades[0]['price'])
                        logger.debug(f"Precio actual de {symbol} (desde trades recientes): {price}")
                        return price
                except Exception as e3:
                    logger.warning(f"Intento {attempt+1}/{max_retries}: Error al obtener precio desde trades recientes: {str(e3)}")
                
                # Esperar antes de reintentar
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        # Si todos los intentos fallan
        logger.error(f"Error: No se pudo obtener el precio actual de {symbol} después de {max_retries} intentos")
        return None
    
    def place_market_buy_order(self, symbol, quantity, test=True):
        """
        Coloca una orden de compra a mercado.
        
        Args:
            symbol (str): Par de trading (ej. 'SOLUSDT').
            quantity (float): Cantidad a comprar.
            test (bool, optional): Si es True, usa la API de prueba.
            
        Returns:
            dict: Respuesta de la API o None si hay un error.
        """
        try:
            if test:
                order = self.client.create_test_order(
                    symbol=symbol,
                    side=Client.SIDE_BUY,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                logger.info(f"Orden de prueba de compra colocada para {symbol}, cantidad: {quantity}")
            else:
                order = self.client.create_order(
                    symbol=symbol,
                    side=Client.SIDE_BUY,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                logger.info(f"Orden de compra colocada para {symbol}, cantidad: {quantity}")
            return order
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Error al colocar orden de compra: {str(e)}")
            return None
    
    def place_market_sell_order(self, symbol, quantity, test=True):
        """
        Coloca una orden de venta a mercado.
        
        Args:
            symbol (str): Par de trading (ej. 'SOLUSDT').
            quantity (float): Cantidad a vender.
            test (bool, optional): Si es True, usa la API de prueba.
            
        Returns:
            dict: Respuesta de la API o None si hay un error.
        """
        try:
            if test:
                order = self.client.create_test_order(
                    symbol=symbol,
                    side=Client.SIDE_SELL,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                logger.info(f"Orden de prueba de venta colocada para {symbol}, cantidad: {quantity}")
            else:
                order = self.client.create_order(
                    symbol=symbol,
                    side=Client.SIDE_SELL,
                    type=Client.ORDER_TYPE_MARKET,
                    quantity=quantity
                )
                logger.info(f"Orden de venta colocada para {symbol}, cantidad: {quantity}")
            return order
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Error al colocar orden de venta: {str(e)}")
            return None
    
    def get_account_balance(self, asset):
        """
        Obtiene el balance de una moneda específica.
        
        Args:
            asset (str): Símbolo de la moneda (ej. 'SOL', 'USDT').
            
        Returns:
            float: Balance disponible o None si hay un error.
        """
        try:
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    free_balance = float(balance['free'])
                    logger.info(f"Balance disponible de {asset}: {free_balance}")
                    return free_balance
            logger.warning(f"No se encontró balance para {asset}")
            return 0.0
        except (BinanceAPIException, BinanceRequestException) as e:
            logger.error(f"Error al obtener balance: {str(e)}")
            return None

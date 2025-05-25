import os
import json
import time
import logging
import pandas as pd
import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv

# Configurar logging
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

class BinanceAPI:
    def __init__(self, api_key=None, api_secret=None, testnet=False, simulation=False, simulation_mode=False, initial_balance=None):
        self.simulation = simulation or simulation_mode
        self.simulation_balance = {'USDT': initial_balance or 1000.0}
        self.simulation_positions = {}
        self.simulation_orders = []
        self.simulation_state_file = 'simulation_state.json'
        
        if self.simulation:
            self.client = None
            self._load_simulation_state()
        else:
            api_key = api_key or os.getenv('BINANCE_API_KEY')
            api_secret = api_secret or os.getenv('BINANCE_API_SECRET')
            self.client = Client(api_key, api_secret, testnet=testnet)
    
    def _load_simulation_state(self, state_file=None):
        """Carga el estado de simulación desde un archivo JSON"""
        if state_file:
            self.simulation_state_file = state_file
        try:
            if os.path.exists(self.simulation_state_file):
                with open(self.simulation_state_file, 'r') as f:
                    state = json.load(f)
                    self.simulation_balance = state.get('balance', {'USDT': 1000.0})
                    self.simulation_positions = state.get('positions', {})
                    self.simulation_orders = state.get('orders', [])
                logger.info(f"Estado de simulación cargado: {self.simulation_balance}")
        except Exception as e:
            logger.error(f"Error al cargar estado de simulación: {e}")
    
    def load_simulation_state(self, state_file=None):
        """Método público para cargar el estado de simulación"""
        return self._load_simulation_state(state_file)
    
    def _save_simulation_state(self):
        """Guarda el estado de simulación en un archivo JSON"""
        try:
            state = {
                'balance': self.simulation_balance,
                'positions': self.simulation_positions,
                'orders': self.simulation_orders
            }
            with open(self.simulation_state_file, 'w') as f:
                json.dump(state, f, indent=4)
            logger.info(f"Estado de simulación guardado en {self.simulation_state_file}")
        except Exception as e:
            logger.error(f"Error al guardar estado de simulación: {e}")
   
    def save_simulation_state(self):
         """Método público para guardar el estado de simulación"""
         return self._save_simulation_state()    

    def get_account_balance(self, asset='USDT'):
        """Obtiene el balance de la cuenta para un activo específico"""
        if self.simulation:
            return self.simulation_balance
        
        try:
            account = self.client.get_account()
            balances = {balance['asset']: float(balance['free']) for balance in account['balances']}
            return balances
        except Exception as e:
            logger.error(f"Error al obtener balance: {e}")
            return {asset: 0.0}
    
    def get_symbol_ticker(self, symbol):
        """Obtiene el precio actual de un símbolo"""
        try:
            if self.simulation:
                # En simulación, obtenemos los datos históricos recientes
                klines = self.get_historical_klines(symbol, '1m', "1 minute ago")
                if klines:
                    return {'price': float(klines[-1][4])}  # Precio de cierre
                return {'price': 100.0}  # Valor predeterminado
            
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return {'price': float(ticker['price'])}
        except Exception as e:
            logger.error(f"Error al obtener precio para {symbol}: {e}")
            return {'price': 100.0}  # Valor predeterminado en caso de error
    
    def get_historical_klines(self, symbol, interval, start_str, end_str=None):
        """Obtiene datos históricos de velas para un símbolo"""
        try:
            if self.simulation:
                # En simulación, necesitamos un cliente temporal para obtener datos
                if not hasattr(self, '_temp_client'):
                    api_key = os.getenv('BINANCE_API_KEY')
                    api_secret = os.getenv('BINANCE_API_SECRET')
                    self._temp_client = Client(api_key, api_secret)
                
                # Usar el cliente temporal para obtener datos
                return self._temp_client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=500
                )
            
            # En modo normal, usar el cliente regular
            return self.client.get_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str,
                limit=500
            )
        except Exception as e:
            logger.error(f"Error al obtener datos históricos para {symbol}: {e}")
            return []
    
    def create_order(self, symbol, side, quantity, price=None, order_type='MARKET'):
        """Crea una orden en Binance o simula una orden"""
        try:
            if self.simulation:
                return self._simulate_order(symbol, side, quantity, price, order_type)
            
            if order_type == 'MARKET':
                order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type=order_type,
                    quantity=quantity
                )
            else:
                order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type=order_type,
                    timeInForce='GTC',
                    quantity=quantity,
                    price=price
                )
            return order
        except Exception as e:
            logger.error(f"Error al crear orden {side} para {symbol}: {e}")
            return None
    
    def place_order(self, symbol, side, order_type, quantity, price=None):
        """
        Método de compatibilidad que llama a create_order.
        Este método se agregó para resolver el error 'BinanceAPI' object has no attribute 'place_order'
        """
        return self.create_order(symbol=symbol, side=side, quantity=quantity, price=price, order_type=order_type)


    def _simulate_order(self, symbol, side, quantity, price=None, order_type='MARKET'):
        """Simula la ejecución de una orden"""
        try:
            # Obtener precio actual si no se proporciona
            if price is None or order_type == 'MARKET':
                ticker = self.get_symbol_ticker(symbol)
                price = float(ticker['price'])
            else:
                price = float(price)
            
            base_asset = symbol.replace('USDT', '')
            
            # Simular la ejecución de la orden
            if side == 'BUY':
                cost = price * quantity
                if self.simulation_balance.get('USDT', 0) >= cost:
                    self.simulation_balance['USDT'] = self.simulation_balance.get('USDT', 0) - cost
                    self.simulation_balance[base_asset] = self.simulation_balance.get(base_asset, 0) + quantity
                    logger.info(f"Orden ejecutada: {side} {symbol} - Precio: {price}, Cantidad: {quantity}")
                else:
                    logger.warning(f"Fondos insuficientes para {side} {symbol}")
                    return None
            elif side == 'SELL':
                if self.simulation_balance.get(base_asset, 0) >= quantity:
                    proceeds = price * quantity
                    self.simulation_balance[base_asset] = self.simulation_balance.get(base_asset, 0) - quantity
                    self.simulation_balance['USDT'] = self.simulation_balance.get('USDT', 0) + proceeds
                    logger.info(f"Orden ejecutada: {side} {symbol} - Precio: {price}, Cantidad: {quantity}")
                else:
                    logger.warning(f"Cantidad insuficiente de {base_asset} para vender")
                    return None
            
            # Registrar la orden con campos compatibles con Binance API
            order = {
                'symbol': symbol,
                'side': side,
                'price': price,
                'quantity': quantity,
                'executedQty': quantity,  # Cantidad ejecutada
                'cummulativeQuoteQty': price * quantity,  # Valor total en USDT
                'type': order_type,
                'status': 'FILLED',  # Estado de la orden
                'time': int(time.time() * 1000)
            }
            self.simulation_orders.append(order)
            
            # Guardar estado
            self._save_simulation_state()
            
            return order
        except Exception as e:
            logger.error(f"Error al simular orden: {e}")
            return None
    
    def get_current_atr(self, symbol, period=14, interval='30m'):
        """
        Calcula el ATR (Average True Range) actual para un símbolo.
        
        Args:
            symbol (str): Par de trading (ej. 'SOLUSDT')
            period (int): Período para el cálculo del ATR
            interval (str): Intervalo de tiempo para los datos
            
        Returns:
            float: Valor del ATR actual
        """
        try:
            # Obtener datos históricos
            klines = self.get_historical_klines(symbol, interval, f"{period * 2} intervals ago")
            if len(klines) < period + 1:
                logger.warning(f"No hay suficientes datos para calcular ATR para {symbol}")
                # Estimación basada en el precio actual (aproximadamente 2% del precio)
                current_price = float(klines[-1][4]) if klines else 100
                return current_price * 0.02
            
            # Calcular True Range para cada vela
            tr_values = []
            for i in range(1, len(klines)):
                high = float(klines[i][2])
                low = float(klines[i][3])
                close_prev = float(klines[i-1][4])
                
                # True Range es el máximo de:
                # 1. Alto actual - Bajo actual
                # 2. |Alto actual - Cierre previo|
                # 3. |Bajo actual - Cierre previo|
                tr = max(
                    high - low,
                    abs(high - close_prev),
                    abs(low - close_prev)
                )
                tr_values.append(tr)
            
            # ATR es el promedio de los True Range para el período
            atr = sum(tr_values[-period:]) / period
            return atr
        except Exception as e:
            logger.error(f"Error al calcular ATR para {symbol}: {e}")
            # Valor predeterminado en caso de error (2% del precio actual aproximadamente)
            try:
                current_price = float(self.get_symbol_ticker(symbol)['price'])
                return current_price * 0.02
            except:
                return 2  # Valor arbitrario si todo lo demás falla

# src/utils/binance_client.py
import os
import time
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import logging
import json

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot_xrp_30m.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BinanceAPI:
    def __init__(self, simulation_mode=True, initial_balance=1000):
        """
        Inicializa el cliente de Binance.
        
        Args:
            simulation_mode (bool): Si es True, se ejecutará en modo simulación
            initial_balance (float): Saldo inicial para el modo simulación (en USDT)
        """
        # Cargar variables de entorno
        load_dotenv()
        
        self.simulation_mode = simulation_mode
        
        # Obtener claves API (necesarias incluso en modo simulación para obtener datos)
        api_key = os.getenv('API_KEY')
        api_secret = os.getenv('API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("API_KEY y API_SECRET deben estar configurados en el archivo .env")
        
        # Inicializar cliente de Binance
        try:
            self.client = Client(api_key, api_secret)
            logger.info("Conexión exitosa con la API de Binance")
        except BinanceAPIException as e:
            logger.error(f"Error al conectar con Binance: {e}")
            raise
        
        # Configurar modo simulación si está activado
        if simulation_mode:
            logger.info(f"Iniciando en modo simulación con saldo inicial de {initial_balance} USDT")
            self.simulated_balance = {'USDT': initial_balance}
            self.simulated_positions = {}
            self.simulated_orders = []
    
    def get_historical_klines(self, symbol, interval, start_str, end_str=None):
        """
        Obtiene datos históricos de velas (klines) para un símbolo y intervalo específicos.
        
        Args:
            symbol (str): Par de trading (ej. 'XRPUSDT')
            interval (str): Intervalo de tiempo (ej. '1h', '4h', '1d')
            start_str (str): Fecha de inicio (ej. '1 day ago', '1 Jan, 2021')
            end_str (str, optional): Fecha de fin. Por defecto es None (hasta ahora).
            
        Returns:
            list: Lista de velas con formato [timestamp, open, high, low, close, volume, ...]
        """
        try:
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str
            )
            logger.info(f"Obtenidos {len(klines)} datos históricos para {symbol} en intervalo {interval}")
            return klines
        except BinanceAPIException as e:
            logger.error(f"Error al obtener datos históricos: {e}")
            return []
    
    def get_account_balance(self):
        """
        Obtiene el balance de la cuenta.
        
        Returns:
            dict: Diccionario con los balances de activos
        """
        if self.simulation_mode:
            logger.info(f"Balance simulado: {self.simulated_balance}")
            return self.simulated_balance
        
        try:
            account = self.client.get_account()
            balances = {asset['asset']: float(asset['free']) for asset in account['balances'] if float(asset['free']) > 0}
            logger.info(f"Balance de cuenta obtenido: {balances}")
            return balances
        except BinanceAPIException as e:
            logger.error(f"Error al obtener balance de cuenta: {e}")
            return {}
    
    def place_order(self, symbol, side, order_type, quantity):
        """
        Coloca una orden en Binance.
        
        Args:
            symbol (str): Par de trading (ej. 'XRPUSDT')
            side (str): 'BUY' o 'SELL'
            order_type (str): Tipo de orden (ej. 'MARKET', 'LIMIT')
            quantity (float): Cantidad a comprar/vender
            
        Returns:
            dict: Respuesta de la API con detalles de la orden
        """
        if self.simulation_mode:
            # Obtener el precio actual del símbolo
            klines = self.get_historical_klines(symbol, '1m', '1 minute ago')
            if not klines:
                logger.error(f"No se pudo obtener el precio actual para {symbol}")
                return None
            
            current_price = float(klines[-1][4])  # Precio de cierre de la última vela
            
            # Simular la ejecución de la orden
            order_id = f"sim_{int(time.time())}"
            executed_qty = quantity
            cummulative_quote_qty = quantity * current_price
            
            # Actualizar saldos simulados
            base_asset = symbol.replace('USDT', '')
            
            if side == 'BUY':
                # Verificar si hay suficiente USDT
                if self.simulated_balance.get('USDT', 0) < cummulative_quote_qty:
                    logger.warning(f"Saldo insuficiente para comprar {quantity} {base_asset}")
                    return None
                
                # Actualizar saldos
                self.simulated_balance['USDT'] = self.simulated_balance.get('USDT', 0) - cummulative_quote_qty
                self.simulated_balance[base_asset] = self.simulated_balance.get(base_asset, 0) + executed_qty
                
                # Registrar posición
                self.simulated_positions[symbol] = {
                    'quantity': executed_qty,
                    'entry_price': current_price,
                    'entry_time': datetime.now()
                }
            
            elif side == 'SELL':
                # Verificar si hay suficiente del activo
                if self.simulated_balance.get(base_asset, 0) < quantity:
                    logger.warning(f"Saldo insuficiente para vender {quantity} {base_asset}")
                    return None
                
                # Actualizar saldos
                self.simulated_balance[base_asset] = self.simulated_balance.get(base_asset, 0) - executed_qty
                self.simulated_balance['USDT'] = self.simulated_balance.get('USDT', 0) + cummulative_quote_qty
                
                # Eliminar posición
                if symbol in self.simulated_positions:
                    del self.simulated_positions[symbol]
            
            # Crear orden simulada
            order = {
                'symbol': symbol,
                'orderId': order_id,
                'side': side,
                'type': order_type,
                'executedQty': str(executed_qty),
                'cummulativeQuoteQty': str(cummulative_quote_qty),
                'status': 'FILLED',
                'price': str(current_price),
                'time': int(time.time() * 1000)
            }
            
            self.simulated_orders.append(order)
            logger.info(f"Orden ejecutada: {side} {symbol} - Precio: {current_price}, Cantidad: {executed_qty}")
            
            return order
        
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity
            )
            logger.info(f"Orden colocada: {order}")
            return order
        except BinanceAPIException as e:
            logger.error(f"Error al colocar orden: {e}")
            return None
    
    def get_current_atr(self, symbol, period=14, interval='30m'):
        """
        Calcula el ATR (Average True Range) actual para un símbolo.
        
        Args:
            symbol (str): Par de trading (ej. 'XRPUSDT')
            period (int): Período para el cálculo del ATR
            interval (str): Intervalo de tiempo para los datos
            
        Returns:
            float: Valor del ATR actual
        """
        try:
            # Obtener datos históricos
            klines = self.get_historical_klines(symbol, interval, f"{period * 3} intervals ago")
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
                prev_close = float(klines[i-1][4])
                
                # True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )
                tr_values.append(tr)
            
            # Calcular ATR (promedio de los últimos 'period' valores de TR)
            if len(tr_values) < period:
                # Si no hay suficientes datos, usar los disponibles
                atr = sum(tr_values) / len(tr_values)
            else:
                # Usar los últimos 'period' valores
                atr = sum(tr_values[-period:]) / period
            
            logger.info(f"ATR calculado para {symbol}: {atr:.6f}")
            return atr
        
        except Exception as e:
            logger.error(f"Error al calcular ATR para {symbol}: {e}")
            # Valor predeterminado basado en el precio actual (aproximadamente 2%)
            try:
                klines = self.get_historical_klines(symbol, '1m', '1 minute ago')
                current_price = float(klines[-1][4]) if klines else 100
                return current_price * 0.02
            except:
                return 2.0  # Valor predeterminado si todo falla
    
    def save_simulation_state(self, file_path='simulation_state_xrp_30m.json'):
        """
        Guarda el estado de la simulación en un archivo.
        
        Args:
            file_path (str): Ruta del archivo para guardar el estado
        """
        if not self.simulation_mode:
            return
        
        state = {
            'balance': self.simulated_balance,
            'positions': self.simulated_positions,
            'orders': self.simulated_orders,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(state, f, default=str)
        
        logger.info(f"Estado de simulación guardado en {file_path}")
    
    def load_simulation_state(self, file_path='simulation_state_xrp_30m.json'):
        """
        Carga el estado de la simulación desde un archivo.
        
        Args:
            file_path (str): Ruta del archivo para cargar el estado
            
        Returns:
            bool: True si se cargó correctamente, False en caso contrario
        """
        if not self.simulation_mode:
            return False
        
        if not os.path.exists(file_path):
            logger.info(f"No se encontró archivo de estado de simulación en {file_path}")
            return False
        
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            self.simulated_balance = state['balance']
            self.simulated_positions = state['positions']
            self.simulated_orders = state['orders']
            
            logger.info(f"Estado de simulación cargado desde {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error al cargar estado de simulación: {e}")
            return False

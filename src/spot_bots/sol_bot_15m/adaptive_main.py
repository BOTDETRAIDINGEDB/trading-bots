#!/usr/bin/env python3
"""
Bot de trading SOL con estrategia adaptativa
Implementa take profit dinámico, stop loss fijo y adaptación al capital disponible
"""

import os
import sys
import logging
import argparse
import time
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Ajustar el PYTHONPATH para encontrar los módulos correctamente
bot_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(bot_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Importar módulos del bot
from strategies.adaptive_strategy import AdaptiveStrategy
from utils.market_analyzer import MarketAnalyzer
from utils.capital_manager import CapitalManager
from utils.enhanced_telegram_notifier import EnhancedTelegramNotifier
from connectors.binance_connector import BinanceConnector

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("sol_bot_adaptive.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdaptiveBot:
    """Bot de trading SOL con estrategia adaptativa."""
    
    def __init__(self, symbol='SOLUSDT', interval='15m', lookback=90, 
                 balance=1000, risk=0.02, stop_loss=0.06, simulation=True, 
                 use_ml=True, retrain_interval=1440):
        """
        Inicializa el bot adaptativo.
        
        Args:
            symbol (str): Par de trading (ej. 'SOLUSDT').
            interval (str): Intervalo de tiempo ('1m', '5m', '15m', '1h', etc.).
            lookback (int): Días de datos históricos a considerar.
            balance (float): Balance inicial para simulación.
            risk (float): Riesgo por operación como porcentaje (0.02 = 2%).
            stop_loss (float): Stop loss como porcentaje (0.06 = 6%).
            simulation (bool): Si es True, opera en modo simulación.
            use_ml (bool): Si es True, utiliza el modelo de ML para las predicciones.
            retrain_interval (int): Intervalo de reentrenamiento del modelo en minutos.
        """
        self.symbol = symbol
        self.interval = interval
        self.lookback = lookback
        self.balance = balance
        self.risk = risk
        self.stop_loss = stop_loss
        self.simulation = simulation
        self.use_ml = use_ml
        self.retrain_interval = retrain_interval
        
        # Cargar variables de entorno
        load_dotenv()
        
        # Inicializar notificador mejorado de Telegram
        self.telegram = EnhancedTelegramNotifier()
        
        # Inicializar conector de Binance
        self.binance = BinanceConnector(simulation=simulation)
        
        # Inicializar analizador de mercado
        self.market_analyzer = MarketAnalyzer(symbol=symbol)
        
        # Inicializar estrategia adaptativa
        self.strategy = AdaptiveStrategy(
            symbol=symbol,
            risk_per_trade=risk,
            stop_loss_pct=stop_loss,
            take_profit_pct=0.04,  # Take profit inicial, se ajustará dinámicamente
            max_trades=3,
            use_ml=use_ml
        )
        
        # Nombre del archivo de estado
        self.state_file = f"sol_bot_{interval.replace('m', 'min')}_state.json"
        
        # Última actualización del modelo
        self.last_model_update = datetime.now()
        
        logger.info(f"Bot adaptativo inicializado para {symbol} con intervalo {interval}")
        
        # Enviar notificación de inicio
        self.notify_bot_start()
    
    def notify_bot_start(self):
        """Notifica el inicio del bot por Telegram."""
        # Obtener precio actual
        df = self.get_historical_data()
        current_price = df['close'].iloc[-1] if df is not None else 0
        
        config = {
            'symbol': self.symbol,
            'interval': self.interval,
            'stop_loss': self.stop_loss,
            'risk': self.risk,
            'simulation': self.simulation,
            'use_ml': self.use_ml,
            'balance': self.balance,
            'current_price': current_price
        }
        
        self.telegram.notify_bot_start(config)
    
    def load_state(self):
        """
        Carga el estado del bot desde el archivo de estado.
        
        Returns:
            bool: True si se cargó correctamente, False en caso contrario.
        """
        return self.strategy.load_state(self.state_file)
    
    def save_state(self):
        """
        Guarda el estado del bot en el archivo de estado.
        
        Returns:
            bool: True si se guardó correctamente, False en caso contrario.
        """
        return self.strategy.save_state(self.state_file)
    
    def get_historical_data(self):
        """
        Obtiene datos históricos para el par e intervalo configurados.
        
        Returns:
            pd.DataFrame: DataFrame con los datos históricos, o None si hubo un error.
        """
        try:
            # Calcular fecha de inicio
            end_time = datetime.now()
            start_time = end_time - timedelta(days=self.lookback)
            
            # Obtener datos históricos de Binance
            klines = self.binance.get_historical_klines(
                symbol=self.symbol,
                interval=self.interval,
                start_str=start_time.strftime("%Y-%m-%d"),
                end_str=end_time.strftime("%Y-%m-%d")
            )
            
            if not klines:
                logger.error("No se pudieron obtener datos históricos")
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convertir tipos de datos
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            logger.info(f"Datos históricos obtenidos: {len(df)} velas")
            return df
        except Exception as e:
            logger.error(f"Error al obtener datos históricos: {str(e)}")
            return None
    
    def calculate_indicators(self, df):
        """
        Calcula indicadores técnicos para los datos históricos.
        
        Args:
            df (pd.DataFrame): DataFrame con datos históricos.
            
        Returns:
            pd.DataFrame: DataFrame con indicadores calculados, o None si hubo un error.
        """
        try:
            # Calcular EMA de 8 y 21 períodos
            df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
            df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
            
            # Calcular RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Calcular MACD
            df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema12'] - df['ema26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Calcular Bollinger Bands
            df['sma20'] = df['close'].rolling(window=20).mean()
            df['std20'] = df['close'].rolling(window=20).std()
            df['upper_band'] = df['sma20'] + (df['std20'] * 2)
            df['lower_band'] = df['sma20'] - (df['std20'] * 2)
            
            # Calcular ATR
            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['close'].shift(1))
            df['tr3'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['tr'].rolling(14).mean()
            
            logger.info("Indicadores técnicos calculados")
            return df
        except Exception as e:
            logger.error(f"Error al calcular indicadores: {str(e)}")
            return None
    
    def generate_signal(self, df):
        """
        Genera señales de trading basadas en indicadores técnicos.
        
        Args:
            df (pd.DataFrame): DataFrame con indicadores calculados.
            
        Returns:
            int: Señal de trading (1: comprar, -1: vender, 0: mantener).
        """
        try:
            # Obtener última fila
            last_row = df.iloc[-1]
            
            # Inicializar señal
            signal = 0
            
            # Señal basada en cruce de EMAs
            if df['ema8'].iloc[-2] <= df['ema21'].iloc[-2] and last_row['ema8'] > last_row['ema21']:
                signal += 1  # Señal alcista
            elif df['ema8'].iloc[-2] >= df['ema21'].iloc[-2] and last_row['ema8'] < last_row['ema21']:
                signal -= 1  # Señal bajista
            
            # Señal basada en RSI
            if last_row['rsi'] < 30:
                signal += 1  # Sobrevendido
            elif last_row['rsi'] > 70:
                signal -= 1  # Sobrecomprado
            
            # Señal basada en MACD
            if df['macd_hist'].iloc[-2] <= 0 and last_row['macd_hist'] > 0:
                signal += 1  # Cruce alcista
            elif df['macd_hist'].iloc[-2] >= 0 and last_row['macd_hist'] < 0:
                signal -= 1  # Cruce bajista
            
            # Señal basada en Bollinger Bands
            if last_row['close'] < last_row['lower_band']:
                signal += 1  # Precio por debajo de banda inferior
            elif last_row['close'] > last_row['upper_band']:
                signal -= 1  # Precio por encima de banda superior
            
            # Normalizar señal
            if signal >= 2:
                return 1  # Señal de compra
            elif signal <= -2:
                return -1  # Señal de venta
            else:
                return 0  # Mantener
        except Exception as e:
            logger.error(f"Error al generar señal: {str(e)}")
            return 0
    
    def get_ml_prediction(self, df):
        """
        Obtiene predicción del modelo de ML.
        
        Args:
            df (pd.DataFrame): DataFrame con indicadores calculados.
            
        Returns:
            int: Predicción del modelo (1: subida, -1: bajada, 0: neutral).
        """
        if not self.use_ml or not self.strategy.ml_model:
            return None
        
        try:
            # Preparar datos para el modelo
            # (Asumimos que el modelo espera los mismos indicadores que calculamos)
            features = df.iloc[-1][['ema8', 'ema21', 'rsi', 'macd', 'macd_signal', 'atr']].values.reshape(1, -1)
            
            # Obtener predicción
            prediction = self.strategy.ml_model.predict(features)
            
            logger.info(f"Predicción del modelo ML: {prediction}")
            return prediction
        except Exception as e:
            logger.error(f"Error al obtener predicción del modelo ML: {str(e)}")
            return None
    
    def check_retrain_model(self):
        """
        Verifica si es necesario reentrenar el modelo de ML.
        
        Returns:
            bool: True si se reentrenó correctamente, False en caso contrario.
        """
        if not self.use_ml:
            return False
        
        # Verificar si ha pasado suficiente tiempo desde el último reentrenamiento
        time_since_update = (datetime.now() - self.last_model_update).total_seconds() / 60
        
        if time_since_update < self.retrain_interval:
            return False
        
        try:
            # Ejecutar script de actualización del modelo
            logger.info("Reentrenando modelo ML...")
            
            # En un entorno real, ejecutaríamos el script update_model.py
            # Aquí simulamos el reentrenamiento
            if self.strategy.ml_model:
                # Obtener datos históricos
                df = self.get_historical_data()
                if df is not None:
                    # Calcular indicadores
                    df = self.calculate_indicators(df)
                    if df is not None:
                        # Reentrenar modelo
                        self.strategy.ml_model.retrain(df)
                        self.last_model_update = datetime.now()
                        logger.info("Modelo ML reentrenado correctamente")
                        return True
            
            logger.error("No se pudo reentrenar el modelo ML")
            return False
        except Exception as e:
            logger.error(f"Error al reentrenar modelo ML: {str(e)}")
            return False
    
    def get_current_balance(self):
        """
        Obtiene el balance actual.
        
        Returns:
            float: Balance actual en USDT.
        """
        if self.simulation:
            # En simulación, usar el balance de la estrategia
            return self.strategy.current_balance
        else:
            # En modo real, obtener balance de Binance
            return self.binance.get_balance('USDT')
    
    def run_iteration(self):
        """
        Ejecuta una iteración del bot.
        
        Returns:
            bool: True si se ejecutó correctamente, False en caso contrario.
        """
        try:
            # Obtener datos históricos
            df = self.get_historical_data()
            if df is None:
                return False
            
            # Calcular indicadores
            df = self.calculate_indicators(df)
            if df is None:
                return False
            
            # Generar señal
            signal = self.generate_signal(df)
            
            # Obtener predicción del modelo ML
            ml_prediction = self.get_ml_prediction(df) if self.use_ml else None
            
            # Obtener precio actual
            current_price = df['close'].iloc[-1]
            current_time = df['close_time'].iloc[-1]
            
            # Obtener balance actual
            current_balance = self.get_current_balance()
            
            # Actualizar balance en la estrategia
            self.strategy.set_balance(current_balance)
            
            # Notificar actualización de mercado cada 4 horas (16 velas de 15 min)
            current_time = datetime.now()
            if not hasattr(self, 'last_market_update') or (current_time - self.last_market_update).total_seconds() >= 14400:  # 4 horas
                self.telegram.notify_market_update(self.strategy.market_conditions, current_price)
                self.last_market_update = current_time
            
            # Verificar si estamos en una posición
            if self.strategy.position != 0:
                # Verificar si debemos salir de la operación
                should_exit, exit_reason = self.strategy.should_exit_trade(current_price, current_time)
                
                if should_exit:
                    # Salir de la operación
                    closed_trade = self.strategy.exit_trade(current_price, current_time, exit_reason)
                    
                    if closed_trade:
                        # Notificar cierre de operación
                        self.telegram.notify_trade_exit(closed_trade, self.strategy.current_balance, self.strategy.performance_metrics)
                        
                        # Guardar estado
                        self.save_state()
            else:
                # Verificar si debemos entrar en una operación
                if self.strategy.should_enter_trade(signal, current_price, current_balance, ml_prediction):
                    # Entrar en la operación
                    trade = self.strategy.enter_trade(current_price, current_time)
                    
                    if trade:
                        # Notificar entrada en operación
                        self.telegram.notify_trade_entry(trade, current_price)
                        
                        # Guardar estado
                        self.save_state()
            
            # Verificar si es necesario reentrenar el modelo
            self.check_retrain_model()
            
            return True
        except Exception as e:
            logger.error(f"Error en iteración del bot: {str(e)}")
            return False
    

    

    
    def run(self):
        """
        Ejecuta el bot en un bucle continuo.
        
        Returns:
            bool: True si se ejecutó correctamente, False en caso contrario.
        """
        logger.info("Iniciando bot...")
        
        # Cargar estado si existe
        if os.path.exists(self.state_file):
            self.load_state()
            logger.info(f"Estado cargado desde {self.state_file}")
        else:
            # Establecer balance inicial
            self.strategy.set_balance(self.balance)
            logger.info(f"Balance inicial establecido: {self.balance} USDT")
        
        # Bucle principal
        try:
            while True:
                # Ejecutar iteración
                self.run_iteration()
                
                # Guardar estado
                self.save_state()
                
                # Esperar hasta la siguiente vela
                # Para intervalo de 15m, esperar 15 minutos
                interval_minutes = int(self.interval.replace('m', ''))
                logger.info(f"Esperando {interval_minutes} minutos hasta la siguiente iteración...")
                time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            logger.info("Bot detenido por el usuario")
            return True
        except Exception as e:
            logger.error(f"Error en el bucle principal: {str(e)}")
            return False

def parse_arguments():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description='Bot de trading SOL con estrategia adaptativa')
    parser.add_argument('--symbol', type=str, default='SOLUSDT', help='Par de trading')
    parser.add_argument('--interval', type=str, default='15m', help='Intervalo de tiempo')
    parser.add_argument('--lookback', type=int, default=90, help='Días de datos históricos')
    parser.add_argument('--balance', type=float, default=1000, help='Balance inicial para simulación')
    parser.add_argument('--risk', type=float, default=0.02, help='Riesgo por operación (0.02 = 2%%)')
    parser.add_argument('--stop-loss', type=float, default=0.06, help='Stop loss (0.06 = 6%%)')
    parser.add_argument('--simulation', action='store_true', help='Operar en modo simulación')
    parser.add_argument('--no-ml', action='store_true', help='No utilizar modelo de ML')
    parser.add_argument('--retrain-interval', type=int, default=1440, help='Intervalo de reentrenamiento en minutos')
    return parser.parse_args()

def main():
    """Función principal."""
    # Cargar variables de entorno
    load_dotenv()
    
    # Parsear argumentos
    args = parse_arguments()
    
    # Inicializar bot
    bot = AdaptiveBot(
        symbol=args.symbol,
        interval=args.interval,
        lookback=args.lookback,
        balance=args.balance,
        risk=args.risk,
        stop_loss=args.stop_loss,
        simulation=args.simulation,
        use_ml=not args.no_ml,
        retrain_interval=args.retrain_interval
    )
    
    # Ejecutar bot
    bot.run()

if __name__ == "__main__":
    main()

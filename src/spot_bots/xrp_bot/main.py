# main.py
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import logging.handlers
import json
import argparse
import sys

# Ajustar el PYTHONPATH para encontrar los módulos correctamente
bot_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(bot_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Ahora importar desde la ubicación correcta
from utils.binance_client import BinanceAPI
from data.processor import DataProcessor
from models.ml_model import TradingModel
from strategies.ml_strategy import MLTradingStrategy
from data.multi_timeframe_processor import MultiTimeframeProcessor

# Configurar logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Asegurarse de que no haya handlers previos
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
# Crear handler para archivo
file_handler = logging.handlers.RotatingFileHandler(
    "trading_bot_xrp_30m.log", 
    maxBytes=10485760,  # 10MB
    backupCount=5,
    mode='a'
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
# Crear handler para consola
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
# Añadir handlers al logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def get_interval_seconds(interval):
    """Convierte un intervalo de tiempo en segundos."""
    interval_map = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800}
    unit = interval[-1]
    value = int(interval[:-1])
    return value * interval_map.get(unit, 3600)

def save_bot_state(strategy, file_path='bot_state_xrp_30m.json'):
    """Guarda el estado actual del bot."""
    state = {
        'performance_metrics': strategy.get_performance_metrics(),
        'parameters': {
            'threshold': strategy.threshold,
            'risk_per_trade': strategy.risk_per_trade
        },
        'timestamp': datetime.now().isoformat()
    }
    with open(file_path, 'w') as f:
        json.dump(state, f, default=str)
    logger.info(f"Estado del bot guardado en {file_path}")

def load_bot_state(strategy, file_path='bot_state_xrp_30m.json'):
    """Carga el estado del bot desde un archivo."""
    if not os.path.exists(file_path):
        logger.info(f"No se encontró archivo de estado del bot en {file_path}")
        return False
    try:
        with open(file_path, 'r') as f:
            state = json.load(f)
        if 'parameters' in state:
            strategy.threshold = state['parameters']['threshold']
            strategy.risk_per_trade = state['parameters']['risk_per_trade']
        logger.info(f"Estado del bot cargado desde {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error al cargar estado del bot: {e}")
        return False

def train_model(symbols, interval, lookback_days, simulation_mode=True, initial_balance=1000):
    """Entrena el modelo con datos históricos."""
    logger.info(f"Iniciando entrenamiento del modelo con {len(symbols)} símbolos")
    
    # Inicializar cliente de Binance y procesador de datos
    binance_api = BinanceAPI(simulation_mode=simulation_mode, initial_balance=initial_balance)
    data_processor = DataProcessor()
    
    # Obtener datos históricos para cada símbolo
    all_X = []
    all_y = []
    
    for symbol in symbols:
        logger.info(f"Obteniendo datos históricos para {symbol}")
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%d %b, %Y")
        
        try:
            # Obtener datos de múltiples timeframes
            klines_1h = binance_api.get_historical_klines(symbol=symbol, interval=interval, start_str=start_date)
            klines_4h = binance_api.get_historical_klines(symbol=symbol, interval="4h", start_str=start_date)
            klines_1d = binance_api.get_historical_klines(symbol=symbol, interval="1d", start_str=start_date)
            
            logger.info(f"Obtenidos {len(klines_1h)} datos históricos para {symbol} en intervalo {interval}")
            logger.info(f"Obtenidos {len(klines_4h)} datos históricos para {symbol} en intervalo 4h")
            logger.info(f"Obtenidos {len(klines_1d)} datos históricos para {symbol} en intervalo 1d")

            if not klines_1h:
                logger.warning(f"No se pudieron obtener datos para {symbol}")
                continue

            # Procesar datos con múltiples timeframes
            multi_processor = MultiTimeframeProcessor()
            df_enriched = multi_processor.process_multiple_timeframes(klines_1h, klines_4h, klines_1d)

            # Preparar datos para el modelo
            X, y = data_processor.prepare_for_model(df_enriched)

            if len(X) > 0:
                all_X.append(X)
                all_y.append(y) 
            else:
                logger.warning(f"No se pudieron preparar datos para {symbol} (X está vacío)")
        except Exception as e:
            logger.error(f"Error al procesar datos para {symbol}: {e}")
            continue
    
    if not all_X:
        logger.error("No se pudieron obtener datos para ningún símbolo")
        return None
    
    # Combinar datos de todos los símbolos
    X_combined = np.vstack(all_X)
    y_combined = np.concatenate(all_y)
    
    # Inicializar y entrenar modelo
    input_shape = (X_combined.shape[1], X_combined.shape[2])
    model = TradingModel(input_shape)
    
    # Modificar la ruta del modelo para XRP
    model.model_path = "models/trading_model_xrp_30m.h5"
    
    # Intentar cargar modelo existente
    if not model.load_trained_model():
        logger.info("Entrenando nuevo modelo")
        model.train(X_combined, y_combined, epochs=50)
    
    return model

def run_trading_bot(symbols, interval, model, strategy, simulation_mode=True, initial_balance=1000):
    """Ejecuta el bot de trading en tiempo real."""
    logger.info(f"Iniciando bot de trading con {len(symbols)} símbolos")
    
    # Inicializar cliente de Binance y procesador de datos
    binance_api = BinanceAPI(simulation_mode=simulation_mode, initial_balance=initial_balance)
    data_processor = DataProcessor()
    
    # Cargar estado de simulación si está en modo simulación
    if simulation_mode:
        binance_api.load_simulation_state('simulation_state_xrp_30m.json')
    
    # Cargar estado del bot
    load_bot_state(strategy)
    
    # Contador de ciclos para reentrenamiento
    cycle_count = 0
    retrain_frequency = 24
    
    # Historial de datos para reentrenamiento
    training_data = {'X': [], 'y': [], 'trades': []}
    
    while True:
        try:
            # Incrementar contador de ciclos
            cycle_count += 1
            logger.info(f"Iniciando ciclo {cycle_count}")
            
            # Obtener balance de cuenta
            account_balance = binance_api.get_account_balance()
            logger.info(f"Balance actual: {account_balance}")
            
            # Obtener datos recientes para cada símbolo
            current_prices = {}
            all_recent_X = []
            recent_data_for_training = {'X': [], 'y': []}
            
            for symbol in symbols:
                start_date = (datetime.now() - timedelta(days=3)).strftime("%d %b, %Y")
                
                try:
                    klines = binance_api.get_historical_klines(symbol=symbol, interval=interval, start_str=start_date)
                    
                    if not klines:
                        logger.warning(f"No se pudieron obtener datos recientes para {symbol}")
                        continue
                    
                    # Procesar datos
                    df = data_processor.klines_to_dataframe(klines)
                    current_prices[symbol] = df['close'].iloc[-1]
                    logger.info(f"Precio actual de {symbol}: {current_prices[symbol]} USDT")
                    
                    df_with_indicators = data_processor.add_indicators(df)
                    X, y = data_processor.prepare_for_model(df_with_indicators)
                    
                    if len(X) > 0:
                        recent_data_for_training['X'].append(X)
                        recent_data_for_training['y'].append(y)
                        all_recent_X.append(X[-1:])
                    else:
                        logger.warning(f"No se pudieron preparar datos para {symbol} (X está vacío)")
                except Exception as e:
                    logger.error(f"Error al procesar datos para {symbol}: {e}")
                    continue
            
            if not all_recent_X:
                logger.error("No se pudieron obtener datos recientes para ningún símbolo")
                time.sleep(60)
                continue
            
            # Combinar datos recientes
            X_recent = np.vstack(all_recent_X)
            
            # Generar señales
            signals = strategy.generate_signals(X_recent, current_prices)
            
            # Añadir más logging para las señales generadas
            for signal in signals:
                logger.info(f"Señal generada para {signal['symbol']}: {signal.get('action', 'NONE')} (probabilidad: {signal.get('probability', 0):.4f}, umbral: {strategy.threshold})")
            
            # Ejecutar señales
            executed_orders = strategy.execute_signals(signals, binance_api, account_balance)
            
            # Añadir más logging para las órdenes ejecutadas
            if executed_orders:
                for order in executed_orders:
                    logger.info(f"Orden ejecutada: {order.get('side', 'UNKNOWN')} {order.get('symbol', 'UNKNOWN')} - Precio: {order.get('price', 0)}, Cantidad: {order.get('executedQty', 0)}")
                training_data['trades'].extend(executed_orders)
            else:
                logger.info("No se ejecutaron órdenes en este ciclo")
            
            # Añadir datos recientes al historial de entrenamiento
            if recent_data_for_training['X'] and recent_data_for_training['y']:
                for X_batch in recent_data_for_training['X']:
                    training_data['X'].append(X_batch)
                for y_batch in recent_data_for_training['y']:
                    training_data['y'].append(y_batch)
            
            # Reentrenar el modelo periódicamente
            if cycle_count % retrain_frequency == 0 and training_data['X'] and training_data['y']:
                try:
                    logger.info(f"Iniciando reentrenamiento del modelo en el ciclo {cycle_count}")
                    X_train = np.vstack(training_data['X'])
                    y_train = np.concatenate(training_data['y'])
                    logger.info(f"Reentrenando con {len(X_train)} muestras")
                    model.train(X_train, y_train, epochs=10, batch_size=32)
                    logger.info("Reentrenamiento completado")
                    
                    if training_data['trades']:
                        logger.info("Ajustando parámetros basados en rendimiento")
                        strategy.adjust_parameters_based_on_performance()
                        logger.info(f"Nuevos parámetros: umbral={strategy.threshold}, riesgo_por_operación={strategy.risk_per_trade}")
                    
                    training_data = {'X': [], 'y': [], 'trades': []}
                except Exception as e:
                    logger.error(f"Error al reentrenar el modelo: {e}")
            
            # Guardar estados
            save_bot_state(strategy)
            if simulation_mode:
                binance_api.save_simulation_state('simulation_state_xrp_30m.json')
            
            # Esperar hasta el próximo ciclo
            wait_time = get_interval_seconds(interval)
            logger.info(f"Esperando {wait_time} segundos hasta el próximo ciclo")
            time.sleep(wait_time)
            
        except Exception as e:
            logger.error(f"Error en el ciclo de trading: {e}")
            time.sleep(60)

def main():
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Bot de Trading con IA para Binance')
    parser.add_argument('--train', action='store_true', help='Entrenar modelo antes de ejecutar')
    parser.add_argument('--symbols', type=str, default='XRPUSDT', help='Símbolos separados por comas')
    parser.add_argument('--interval', type=str, default='30m', help='Intervalo de tiempo')
    parser.add_argument('--lookback', type=int, default=60, help='Días de datos históricos para entrenamiento')
    parser.add_argument('--simulation', action='store_true', help='Ejecutar en modo simulación (paper trading)')
    parser.add_argument('--initial_balance', type=float, default=1000, help='Saldo inicial para simulación (USDT)')
    args = parser.parse_args()
    
    # Convertir símbolos a lista
    symbols = args.symbols.split(',')
    
    # Entrenar o cargar modelo
    model = None
    if args.train:
        model = train_model(symbols, args.interval, args.lookback, 
                           simulation_mode=args.simulation, 
                           initial_balance=args.initial_balance)
        
        # Verificar que el modelo se haya entrenado correctamente
        if model is None:
            logger.error("No se pudo entrenar el modelo. Verifique los datos y parámetros.")
            return
    else:
        # Cargar modelo existente
        binance_api = BinanceAPI(simulation_mode=args.simulation, initial_balance=args.initial_balance)
        data_processor = DataProcessor()
        
        # Obtener una muestra de datos para determinar la forma de entrada
        sample_klines = binance_api.get_historical_klines(
            symbol=symbols[0],
            interval=args.interval,
            start_str='1 day ago'
        )
        
        if not sample_klines:
            logger.error("No se pudieron obtener datos de muestra")
            return
        
        sample_df = data_processor.klines_to_dataframe(sample_klines)
        sample_df_with_indicators = data_processor.add_indicators(sample_df)
        sample_X, _ = data_processor.prepare_for_model(sample_df_with_indicators)
        
        if len(sample_X) == 0:
            logger.error("No se pudieron preparar datos de muestra")
            return
        
        input_shape = (sample_X.shape[1], sample_X.shape[2])
        model = TradingModel(input_shape)
        
        # Modificar la ruta del modelo para XRP
        model.model_path = "models/trading_model_xrp_30m.h5"
        
        if not model.load_trained_model():
            logger.error("No se pudo cargar un modelo entrenado. Ejecute con --train primero.")
            return
    
    # Inicializar estrategia
    strategy = MLTradingStrategy(model)
    strategy.threshold = 0.50  # Ajustar el umbral para generar más señales
    
    # Ejecutar bot
    run_trading_bot(symbols, args.interval, model, strategy, 
                   simulation_mode=args.simulation, 
                   initial_balance=args.initial_balance)

if __name__ == "__main__":
    main()

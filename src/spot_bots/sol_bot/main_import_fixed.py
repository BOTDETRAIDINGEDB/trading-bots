#!/usr/bin/env python3
"""Bot de Trading para SOL.

Este script implementa un bot de trading automatizado para SOL/USDT.
Utiliza estrategias de machine learning para tomar decisiones de trading
y maneja la ejecución de órdenes a través de Binance.

Author: Edison
Date: 2025-05-25
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import argparse
from typing import Dict, Optional, Any, Union
from requests.exceptions import RequestException
from binance.exceptions import BinanceAPIException
# Importaciones corregidas para la estructura de la máquina virtual
from src.spot_bots.sol_bot.src.utils.binance_client import BinanceAPI
from src.spot_bots.sol_bot.src.data.processor import DataProcessor
from src.spot_bots.sol_bot.src.models.ml_model import TradingModel
from src.spot_bots.sol_bot.src.strategies.ml_strategy import MLTradingStrategy
from src.spot_bots.sol_bot.src.data.multi_timeframe_processor import MultiTimeframeProcessor
# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_interval_seconds(interval: str) -> int:
    """Convierte un intervalo de tiempo en segundos.
    
    Args:
        interval (str): Intervalo en formato '1m', '1h', '1d', etc.
        
    Returns:
        int: Número de segundos equivalente al intervalo
        
    Raises:
        ValueError: Si el formato del intervalo es inválido
    """
    try:
        interval_map = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800}
        unit = interval[-1]
        value = int(interval[:-1])
        if unit not in interval_map:
            raise ValueError(f"Unidad de tiempo inválida: {unit}")
        return value * interval_map[unit]
    except (IndexError, ValueError) as e:
        logger.error(f"Error al procesar intervalo {interval}: {e}")
        raise ValueError(f"Formato de intervalo inválido: {interval}. Use formato como '1m', '1h', '1d'")

def save_bot_state(strategy: MLTradingStrategy, file_path: str = 'bot_state.json') -> bool:
    """Guarda el estado actual del bot.
    
    Args:
        strategy (MLTradingStrategy): Estrategia de trading activa
        file_path (str): Ruta del archivo para guardar el estado
    
    Returns:
        bool: True si se guardó correctamente, False si hubo error
    """
    try:
        state = {
            'performance_metrics': strategy.get_performance_metrics(),
            'parameters': {
                'threshold': strategy.threshold,
                'risk_per_trade': strategy.risk_per_trade
            },
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'  # Control de versiones
        }
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(state, f, default=str, indent=4)
        logger.info(f"Estado del bot guardado en {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error al guardar estado del bot: {e}")
        return False

def load_bot_state(strategy: MLTradingStrategy, file_path: str = 'bot_state.json') -> bool:
    """Carga el estado del bot desde un archivo.
    
    Args:
        strategy (MLTradingStrategy): Estrategia de trading a actualizar
        file_path (str): Ruta del archivo con el estado guardado
    
    Returns:
        bool: True si se cargó correctamente, False si hubo error
    """
    try:
        if not os.path.exists(file_path):
            logger.info(f"No se encontró archivo de estado del bot en {file_path}")
            return False
            
        with open(file_path, 'r') as f:
            state = json.load(f)
            
        # Validar versión del estado
        if state.get('version', '0.0.0') < '1.0.0':
            logger.warning(f"Versión de estado antigua: {state.get('version')}")
            return False
            
        # Actualizar estrategia
        strategy.threshold = state['parameters']['threshold']
        strategy.risk_per_trade = state['parameters']['risk_per_trade']
        
        logger.info(f"Estado del bot cargado desde {file_path}")
        return True
        
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Error al cargar estado del bot: {e}")
        return False
    except Exception as e:
        logger.error(f"Error inesperado al cargar estado: {e}")
        return False

def train_model(
    symbols: list,
    interval: str,
    lookback_days: int,
    simulation_mode: bool = True,
    initial_balance: float = 1000.0
) -> Optional[TradingModel]:
    """Entrena el modelo de trading con datos históricos.
    
    Args:
        symbols (list): Lista de símbolos para entrenar (ej: ['SOLUSDT'])
        interval (str): Intervalo de tiempo para los datos
        lookback_days (int): Días de historia para entrenar
        simulation_mode (bool): Si es True, usa modo simulación
        initial_balance (float): Balance inicial para simulación
    
    Returns:
        Optional[TradingModel]: Modelo entrenado o None si hay error
    
    Raises:
        ValueError: Si los parámetros son inválidos
        RequestException: Si hay error de conexión
    """
    logger.info(f"Iniciando entrenamiento del modelo con {len(symbols)} símbolos")
    
    # Validar parámetros
    if not symbols or not isinstance(symbols, list):
        logger.error("Se requiere una lista válida de símbolos")
        return None
    if not interval or not isinstance(interval, str):
        logger.error("Se requiere un intervalo válido")
        return None
    if lookback_days <= 0:
        logger.error("lookback_days debe ser mayor a 0")
        return None
    
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
            
            if len(X) == 0 or len(y) == 0:
                logger.warning(f"No se pudieron preparar datos para {symbol}")
                continue
                
            all_X.append(X)
            all_y.append(y)
            
        except RequestException as e:
            logger.error(f"Error de conexión para {symbol}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error al procesar datos para {symbol}: {e}")
            continue
            
    # Verificar si tenemos datos suficientes
    if not all_X or not all_y:
        logger.error("No hay datos suficientes para entrenar el modelo")
        return None
        
    try:
        # Combinar datos de todos los símbolos
        X_combined = np.vstack(all_X)
        y_combined = np.concatenate(all_y)
        
        logger.info(f"Entrenando modelo con {len(X_combined)} muestras")
        
        # Crear y entrenar modelo
        input_shape = (X_combined.shape[1], X_combined.shape[2])
        model = TradingModel(input_shape)
        model.train(X_combined, y_combined, epochs=50, batch_size=32)
        
        # Guardar modelo
        model.save_trained_model()
        
        logger.info("Modelo entrenado y guardado exitosamente")
        return model
        
    except Exception as e:
        logger.error(f"Error al entrenar modelo: {e}")
        return None

def run_trading_bot(
    symbols: list,
    interval: str,
    model: TradingModel,
    strategy: MLTradingStrategy,
    simulation_mode: bool = True,
    initial_balance: float = 1000.0
) -> None:
    """Ejecuta el bot de trading en tiempo real.
    
    Args:
        symbols (list): Lista de símbolos para operar
        interval (str): Intervalo de tiempo para los datos
        model (TradingModel): Modelo entrenado para predicciones
        strategy (MLTradingStrategy): Estrategia de trading a utilizar
        simulation_mode (bool): Si es True, usa modo simulación
        initial_balance (float): Balance inicial para simulación
        
    Raises:
        ValueError: Si los parámetros son inválidos
    """
    # Validar parámetros
    if not symbols or not isinstance(symbols, list):
        raise ValueError("Se requiere una lista válida de símbolos")
    if not model or not isinstance(model, TradingModel):
        raise ValueError("Se requiere un modelo válido")
    if not strategy or not isinstance(strategy, MLTradingStrategy):
        raise ValueError("Se requiere una estrategia válida")
    if not interval:
        raise ValueError("Se requiere un intervalo válido")
        
    logger.info(f"Iniciando bot de trading con {len(symbols)} símbolos")
    
    # Inicializar variables
    binance_api = None
    data_processor = None
    cycle_count = 0
    retrain_frequency = 48  # Reentrenar cada 48 ciclos
    training_data = {'X': [], 'y': [], 'trades': []}
    
    try:
        # Inicializar componentes
        binance_api = BinanceAPI(simulation_mode=simulation_mode, initial_balance=initial_balance)
        data_processor = DataProcessor()
        
        # Cargar estados
        try:
            if simulation_mode:
                binance_api.load_simulation_state()
            load_bot_state(strategy)
        except Exception as e:
            logger.error(f"Error al cargar estados: {e}")
            raise
        
        # Bucle principal de trading
        while True:
            try:
                # Incrementar y registrar ciclo
                cycle_count += 1
                logger.info(f"Iniciando ciclo {cycle_count}")
                
                # Obtener balance de cuenta
                try:
                    account_balance = binance_api.get_account_balance()
                except (BinanceAPIException, Exception) as e:
                    logger.error(f"Error al obtener balance: {e}")
                    time.sleep(60)
                    continue
                
                # Preparar contenedores de datos
                current_prices = {}
                predictions = {}
                
                # Procesar cada símbolo
                for symbol in symbols:
                    try:
                        # Obtener datos recientes
                        recent_klines = binance_api.get_historical_klines(
                            symbol=symbol,
                            interval=interval,
                            start_str='2 days ago'
                        )
                        
                        if not recent_klines:
                            logger.warning(f"No hay datos recientes para {symbol}")
                            continue
                            
                        # Obtener precio actual
                        current_price = float(recent_klines[-1][4])  # Precio de cierre
                        current_prices[symbol] = current_price
                        
                        # Procesar datos para predicción
                        recent_df = data_processor.klines_to_dataframe(recent_klines)
                        recent_df = data_processor.add_indicators(recent_df)
                        X_recent, y_recent = data_processor.prepare_for_model(recent_df)
                        
                        if len(X_recent) == 0:
                            logger.warning(f"No se pudieron preparar datos para {symbol}")
                            continue
                            
                        # Hacer predicción
                        prediction = model.predict(X_recent[-1:])
                        predictions[symbol] = prediction[0]
                        
                        # Almacenar datos para reentrenamiento
                        training_data['X'].append(X_recent[-1:])
                        training_data['y'].append(y_recent[-1:])
                        
                        logger.info(f"{symbol}: Precio={current_price}, Predicción={prediction[0]}")
                        
                    except Exception as e:
                        logger.error(f"Error al procesar {symbol}: {e}")
                        continue
                
                # Ejecutar estrategia de trading
                if current_prices and predictions:
                    try:
                        trades = strategy.execute_strategy(
                            symbols=list(current_prices.keys()),
                            prices=current_prices,
                            predictions=predictions,
                            account_balance=account_balance,
                            api=binance_api
                        )
                        
                        if trades:
                            training_data['trades'].extend(trades)
                            logger.info(f"Ejecutadas {len(trades)} operaciones")
                        else:
                            logger.info("No se ejecutaron operaciones en este ciclo")
                            
                    except Exception as e:
                        logger.error(f"Error al ejecutar estrategia: {e}")
                
                # Reentrenar el modelo periódicamente
                if cycle_count % retrain_frequency == 0 and training_data['X'] and training_data['y']:
                    try:
                        X_train = np.vstack(training_data['X'])
                        y_train = np.concatenate(training_data['y'])
                        model.train(X_train, y_train, epochs=10, batch_size=32)
                        
                        if training_data['trades']:
                            strategy.adjust_parameters_based_on_performance()
                        
                        training_data = {'X': [], 'y': [], 'trades': []}
                        logger.info("Modelo reentrenado exitosamente")
                    except Exception as e:
                        logger.error(f"Error al reentrenar el modelo: {e}")
                
                # Guardar estados
                try:
                    save_bot_state(strategy)
                    if simulation_mode:
                        binance_api.save_simulation_state()
                except Exception as e:
                    logger.error(f"Error al guardar estados: {e}")
                
                # Esperar hasta el próximo ciclo
                try:
                    wait_time = get_interval_seconds(interval)
                    logger.info(f"Esperando {wait_time} segundos hasta el próximo ciclo")
                    time.sleep(wait_time)
                except Exception as e:
                    logger.error(f"Error al esperar ciclo: {e}")
                    time.sleep(60)
                
            except Exception as e:
                logger.error(f"Error en ciclo de trading: {e}")
                time.sleep(60)
                
    except Exception as e:
        logger.error(f"Error crítico en el bot: {e}")
        raise
    finally:
        if binance_api and simulation_mode:
            try:
                binance_api.save_simulation_state()
                logger.info("Estado de simulación guardado")
            except Exception as e:
                logger.error(f"Error al guardar estado final: {e}")

def main() -> None:
    """Función principal del bot de trading.
    
    Esta función maneja:
    1. Configuración inicial y validación de parámetros
    2. Entrenamiento o carga del modelo
    3. Ejecución del bot de trading
    """
    try:
        # Configurar argumentos de línea de comandos
        parser = argparse.ArgumentParser(description='Bot de Trading con IA para Binance')
        parser.add_argument('--train', action='store_true', help='Entrenar modelo antes de ejecutar')
        parser.add_argument('--symbols', type=str, default='SOLUSDT', help='Símbolos separados por comas')
        parser.add_argument('--interval', type=str, default='1h', help='Intervalo de tiempo')
        parser.add_argument('--lookback', type=int, default=90, help='Días de datos históricos para entrenamiento')
        parser.add_argument('--simulation', action='store_true', help='Ejecutar en modo simulación (paper trading)')
        parser.add_argument('--initial_balance', type=float, default=1000, help='Saldo inicial para simulación (USDT)')
        
        args = parser.parse_args()
        
        # Validar argumentos
        if args.lookback < 1:
            logger.error("--lookback debe ser mayor a 0")
            return
        if args.initial_balance <= 0:
            logger.error("--initial_balance debe ser mayor a 0")
            return
            
        # Validar intervalo
        try:
            interval_seconds = get_interval_seconds(args.interval)
            if interval_seconds <= 0:
                logger.error("Intervalo debe ser mayor a 0 segundos")
                return
        except (ValueError, TypeError) as e:
            logger.error(f"Intervalo inválido: {e}")
            return
    
        # Convertir y validar símbolos
        try:
            symbols = [s.strip().upper() for s in args.symbols.split(',')]
            if not symbols:
                logger.error("No se proporcionaron símbolos")
                return
        except Exception as e:
            logger.error(f"Error al procesar símbolos: {e}")
            return
            
        # Inicializar modelo
        model = None
        try:
            if args.train:
                logger.info("Iniciando entrenamiento del modelo...")
                model = train_model(
                    symbols=symbols,
                    interval=args.interval,
                    lookback_days=args.lookback,
                    simulation_mode=args.simulation,
                    initial_balance=args.initial_balance
                )
                
                if model is None:
                    logger.error("Fallo en entrenamiento del modelo")
                    return
                    
            else:
                logger.info("Cargando modelo existente...")
                try:
                    # Inicializar componentes
                    binance_api = BinanceAPI(
                        simulation_mode=args.simulation,
                        initial_balance=args.initial_balance
                    )
                    data_processor = DataProcessor()
                    
                    # Obtener datos de muestra
                    sample_klines = binance_api.get_historical_klines(
                        symbol=symbols[0],
                        interval=args.interval,
                        start_str='1 day ago'
                    )
                    
                    if not sample_klines:
                        logger.error("No hay datos de muestra disponibles")
                        return
                        
                    # Procesar datos de muestra
                    sample_df = data_processor.klines_to_dataframe(sample_klines)
                    sample_df = data_processor.add_indicators(sample_df)
                    sample_X, _ = data_processor.prepare_for_model(sample_df)
                    
                    if len(sample_X) == 0:
                        logger.error("No se pudieron preparar datos de muestra")
                        return
                        
                    # Crear y cargar modelo
                    input_shape = (sample_X.shape[1], sample_X.shape[2])
                    model = TradingModel(input_shape)
                    
                    if not model.load_trained_model():
                        logger.error("No se encontró modelo entrenado")
                        return
                        
                except Exception as e:
                    logger.error(f"Error al cargar modelo: {e}")
                    return
        except Exception as e:
            logger.error(f"Error al inicializar modelo: {e}")
            return
            
            # Inicializar estrategia y ejecutar bot
            try:
                strategy = MLTradingStrategy(model)
                logger.info("Iniciando bot de trading...")
                
                run_trading_bot(
                    symbols=symbols,
                    interval=args.interval,
                    model=model,
                    strategy=strategy,
                    simulation_mode=args.simulation,
                    initial_balance=args.initial_balance
                )
                
            except Exception as e:
                logger.error(f"Error al ejecutar bot: {e}")
                return
                
    except KeyboardInterrupt:
        logger.info("Bot detenido por el usuario")
    except Exception as e:
        logger.error(f"Error crítico: {e}")
        return
    finally:
        logger.info("Finalizando bot de trading")

if __name__ == "__main__":
    main()

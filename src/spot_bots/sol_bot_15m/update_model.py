#!/usr/bin/env python3
"""
Script para actualizar y reentrenar el modelo de ML del bot SOL
"""

import os
import sys
import logging
import argparse
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
from utils.binance_client import BinanceAPI
from data.processor import DataProcessor
from models.ml_model import MLModel

def setup_logging():
    """Configura el sistema de logging"""
    os.makedirs('logs', exist_ok=True)
    log_file = f"logs/update_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_environment():
    """Carga las variables de entorno"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        logger.warning("No se encontró el archivo .env. Usando variables de entorno del sistema.")

def get_historical_data(symbol, interval, days=30):
    """Obtiene datos históricos de Binance"""
    try:
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("API key o secret no configurados en las variables de entorno")
            
        client = BinanceAPI(api_key, api_secret)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        logger.info(f"Obteniendo datos históricos para {symbol} {interval} desde {start_time} hasta {end_time}")
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_time.strftime('%d %b %Y %H:%M:%S'),
            end_str=end_time.strftime('%d %b %Y %H:%M:%S')
        )
        
        logger.info(f"Se obtuvieron {len(klines)} velas")
        return klines
    except Exception as e:
        logger.error(f"Error al obtener datos históricos: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Actualizar y reentrenar el modelo de ML')
    parser.add_argument('--symbol', type=str, default='SOLUSDT', help='Símbolo del par de trading')
    parser.add_argument('--interval', type=str, default='15m', help='Intervalo de tiempo')
    parser.add_argument('--days', type=int, default=30, help='Número de días de datos históricos')
    args = parser.parse_args()
    
    load_environment()
    
    try:
        # Obtener datos históricos
        klines = get_historical_data(args.symbol, args.interval, args.days)
        if not klines:
            logger.error("No se obtuvieron datos históricos")
            return
        
        # Procesar datos
        processor = DataProcessor()
        df = processor.klines_to_dataframe(klines)
        logger.info(f"Datos convertidos a DataFrame: {df.shape[0]} filas, {df.shape[1]} columnas")
        
        # Calcular indicadores técnicos
        df = processor.calculate_indicators(df)
        logger.info("Indicadores técnicos calculados")
        
        # Generar señales
        df = processor.generate_signals(df)
        logger.info(f"Señales generadas. Señales de compra: {(df['signal'] == 1).sum()}, "
                   f"Señales de venta: {(df['signal'] == -1).sum()}")
        
        # Eliminar filas con NaN (pueden aparecer por los cálculos de indicadores)
        df = df.dropna()
        logger.info(f"Datos después de eliminar NaN: {df.shape[0]} filas")
        
        # Guardar datos de entrenamiento
        os.makedirs('training_data', exist_ok=True)
        training_file = f'training_data/{args.symbol.lower()}_training_data.csv'
        df.to_csv(training_file, index=False)
        logger.info(f"Datos de entrenamiento guardados en {training_file}")
        
        # Entrenar modelo
        model = MLModel()
        metrics = model.train(df)
        
        if metrics:
            logger.info(f"Modelo reentrenado exitosamente. Métricas: {metrics}")
        else:
            logger.error("No se pudo entrenar el modelo")
            
    except Exception as e:
        logger.error(f"Error en la actualización del modelo: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    logger = setup_logging()
    main()

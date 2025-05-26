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

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_update.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description='Actualizar modelo de ML para el bot SOL')
    parser.add_argument('--symbol', type=str, default='SOLUSDT', help='Símbolo a procesar')
    parser.add_argument('--interval', type=str, default='15m', help='Intervalo de tiempo')
    parser.add_argument('--lookback', type=int, default=90, help='Días de datos históricos a utilizar')
    parser.add_argument('--test-size', type=float, default=0.2, help='Proporción de datos para test')
    return parser.parse_args()

def main():
    """Función principal para actualizar el modelo."""
    # Cargar variables de entorno
    load_dotenv()
    
    # Parsear argumentos
    args = parse_arguments()
    
    logger.info(f"Iniciando actualización del modelo para {args.symbol} en intervalo {args.interval}")
    
    try:
        # Inicializar componentes
        binance_api = BinanceAPI()
        data_processor = DataProcessor()
        model = MLModel(model_path=f"{args.symbol.lower()}_model.pkl")
        
        # Obtener datos históricos
        end_time = datetime.now()
        start_time = end_time - timedelta(days=args.lookback)
        
        logger.info(f"Obteniendo datos históricos desde {start_time} hasta {end_time}")
        historical_data = binance_api.get_historical_klines(
            symbol=args.symbol,
            interval=args.interval,
            start_time=start_time,
            end_time=end_time
        )
        
        if not historical_data or len(historical_data) < 100:
            logger.error(f"No se pudieron obtener suficientes datos históricos. Obtenidos: {len(historical_data) if historical_data else 0}")
            return
        
        logger.info(f"Datos históricos obtenidos: {len(historical_data)} velas")
        
        # Procesar datos
        df = data_processor.process_historical_data(historical_data)
        
        # Añadir indicadores técnicos
        df = data_processor.add_technical_indicators(df)
        
        # Eliminar filas con NaN
        df = df.dropna()
        
        if df.empty:
            logger.error("DataFrame vacío después de procesar datos")
            return
        
        # Entrenar modelo
        logger.info("Entrenando modelo...")
        metrics = model.train(df, test_size=args.test_size)
        
        # Guardar modelo
        model.save_model()
        
        logger.info(f"Modelo actualizado y guardado. Métricas: {metrics}")
        
    except Exception as e:
        logger.error(f"Error al actualizar el modelo: {str(e)}")

if __name__ == "__main__":
    main()

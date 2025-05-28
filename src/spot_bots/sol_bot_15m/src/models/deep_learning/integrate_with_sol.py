#!/usr/bin/env python3
"""
Script de integración del sistema de aprendizaje profundo con el bot SOL.
Permite utilizar predicciones de LSTM/GRU para mejorar las decisiones de trading.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import json
import time
import pandas as pd
import numpy as np

# Ajustar el PYTHONPATH para encontrar los módulos correctamente
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Importar módulos propios
from models.deep_learning.deep_learning_integration import DeepLearningIntegration
from utils.enhanced_telegram_notifier import EnhancedTelegramNotifier

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deep_learning_integration.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description='Integración de aprendizaje profundo con bot SOL')
    
    # Argumentos generales
    parser.add_argument('--symbol', type=str, default='SOL/USDT', help='Par de trading')
    parser.add_argument('--timeframes', type=str, nargs='+', default=['5m', '15m', '1h', '4h'], 
                        help='Lista de timeframes a utilizar')
    parser.add_argument('--model-type', type=str, default='lstm', 
                        choices=['lstm', 'gru', 'bilstm', 'attention'],
                        help='Tipo de modelo a utilizar')
    
    # Modos de ejecución
    parser.add_argument('--predict', action='store_true', help='Realizar predicciones')
    parser.add_argument('--retrain', action='store_true', help='Reentrenar modelo')
    parser.add_argument('--backtest', action='store_true', help='Ejecutar backtesting')
    parser.add_argument('--integrate', action='store_true', help='Integrar con bot SOL')
    
    # Parámetros adicionales
    parser.add_argument('--risk-level', type=float, default=0.5, help='Nivel de riesgo (0-1)')
    parser.add_argument('--notify', action='store_true', help='Enviar notificaciones por Telegram')
    
    return parser.parse_args()

def load_market_data(symbol, timeframe='15m', limit=500):
    """Carga datos de mercado para pruebas."""
    try:
        # Importar ccxt para obtener datos
        import ccxt
        
        # Inicializar exchange
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot'
            }
        })
        
        # Obtener datos OHLCV
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        # Convertir a DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Datos de mercado cargados para {symbol} en timeframe {timeframe}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error al cargar datos de mercado: {str(e)}")
        return pd.DataFrame()

def send_telegram_notification(prediction, symbol):
    """Envía notificación de predicción por Telegram."""
    try:
        # Inicializar notificador
        telegram = EnhancedTelegramNotifier()
        
        # Crear mensaje
        direction = prediction.get('direction', 'NEUTRAL')
        confidence = prediction.get('confidence', 0) * 100 if 'confidence' in prediction else 0
        
        emoji = "🟢" if direction == "ALCISTA" else "🔴" if direction == "BAJISTA" else "⚪"
        
        message = f"{emoji} *Predicción IA - {symbol}*\n\n"
        message += f"*Dirección:* {direction}\n"
        
        if 'confidence' in prediction:
            message += f"*Confianza:* {confidence:.2f}%\n"
        elif 'predicted_change' in prediction:
            change = prediction.get('predicted_change', 0) * 100
            message += f"*Cambio Previsto:* {change:.2f}%\n"
        
        message += f"*Señal:* {prediction.get('signal', 0)}\n"
        message += f"\n*Generado:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Enviar mensaje
        telegram.send_message(message)
        
        logger.info("Notificación enviada por Telegram")
        
        return True
    
    except Exception as e:
        logger.error(f"Error al enviar notificación: {str(e)}")
        return False

def main():
    """Función principal."""
    # Parsear argumentos
    args = parse_arguments()
    
    # Inicializar integración
    dl_integration = DeepLearningIntegration(
        symbol=args.symbol,
        timeframes=args.timeframes,
        model_type=args.model_type
    )
    
    # Ejecutar según modo
    if args.retrain:
        logger.info("Reentrenando modelo...")
        success = dl_integration.retrain_model()
        if success:
            logger.info("Modelo reentrenado correctamente")
        else:
            logger.error("Error al reentrenar modelo")
    
    if args.predict:
        logger.info("Generando predicciones...")
        
        # Cargar datos de mercado
        market_data = load_market_data(args.symbol)
        
        if market_data.empty:
            logger.error("No se pudieron cargar datos de mercado")
            return
        
        # Obtener predicción
        prediction = dl_integration.get_prediction(market_data)
        
        if prediction:
            logger.info(f"Predicción: {prediction}")
            
            # Enviar notificación si se solicita
            if args.notify:
                send_telegram_notification(prediction, args.symbol)
        else:
            logger.error("No se pudo obtener predicción")
    
    if args.integrate:
        logger.info("Integrando con bot SOL...")
        
        # Cargar datos de mercado
        market_data = load_market_data(args.symbol)
        
        if market_data.empty:
            logger.error("No se pudieron cargar datos de mercado")
            return
        
        # Simular señal técnica (en un caso real, esto vendría del bot)
        technical_signal = 1  # Señal de compra
        
        # Mejorar decisión con ML
        enhanced_decision = dl_integration.enhance_trading_decision(
            technical_signal=technical_signal,
            market_data=market_data,
            risk_level=args.risk_level
        )
        
        if enhanced_decision:
            logger.info(f"Decisión mejorada: {enhanced_decision}")
            
            # Enviar notificación si se solicita
            if args.notify:
                # Crear predicción para notificación
                prediction = {
                    'direction': enhanced_decision.get('ml_direction', 'NEUTRAL'),
                    'confidence': enhanced_decision.get('ml_confidence', 0),
                    'signal': enhanced_decision.get('final_signal', 0)
                }
                send_telegram_notification(prediction, args.symbol)
        else:
            logger.error("No se pudo mejorar la decisión")
    
    if args.backtest:
        logger.info("Esta funcionalidad será implementada próximamente")
    
    # Si no se especificó ningún modo, mostrar ayuda
    if not (args.predict or args.retrain or args.integrate or args.backtest):
        logger.info("No se especificó ningún modo de ejecución. Use --predict, --retrain, --integrate o --backtest.")
        logger.info("Ejecute con --help para ver todas las opciones.")

if __name__ == "__main__":
    main()

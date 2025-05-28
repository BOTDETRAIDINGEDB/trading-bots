#!/usr/bin/env python3
"""
Script principal para entrenar y evaluar modelos de aprendizaje profundo.
Integra todos los componentes del sistema de aprendizaje profundo.
"""

import os
import sys
import logging
import argparse
from datetime import datetime
import json
import time

# Ajustar el PYTHONPATH para encontrar los módulos correctamente
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Importar módulos propios
from models.deep_learning.model_trainer import DeepLearningTrainer

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deep_learning_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description='Entrenador de modelos de aprendizaje profundo para trading')
    
    # Argumentos generales
    parser.add_argument('--symbol', type=str, default='SOL/USDT', help='Par de trading')
    parser.add_argument('--base-timeframe', type=str, default='15m', help='Timeframe base')
    parser.add_argument('--timeframes', type=str, nargs='+', default=['5m', '15m', '1h', '4h'], 
                        help='Lista de timeframes a utilizar')
    
    # Argumentos de modelo
    parser.add_argument('--model-type', type=str, default='lstm', 
                        choices=['lstm', 'gru', 'bilstm', 'attention'],
                        help='Tipo de modelo a utilizar')
    parser.add_argument('--sequence-length', type=int, default=60, 
                        help='Longitud de secuencia para LSTM/GRU')
    parser.add_argument('--prediction-horizon', type=int, default=3, 
                        help='Horizonte de predicción (cuántas velas adelante)')
    
    # Argumentos de datos
    parser.add_argument('--lookback-days', type=int, default=365, 
                        help='Días de histórico a utilizar')
    parser.add_argument('--force-update', action='store_true', 
                        help='Forzar actualización de datos desde el exchange')
    
    # Argumentos de salida
    parser.add_argument('--output-dir', type=str, default='models/deep_learning', 
                        help='Directorio para guardar modelos y resultados')
    parser.add_argument('--config-dir', type=str, default='config', 
                        help='Directorio de configuración')
    
    # Modos de ejecución
    parser.add_argument('--train', action='store_true', help='Entrenar modelo')
    parser.add_argument('--evaluate', action='store_true', help='Evaluar modelo')
    parser.add_argument('--predict', action='store_true', help='Realizar predicciones')
    parser.add_argument('--periods', type=int, default=3, help='Número de períodos a predecir')
    
    return parser.parse_args()

def save_results(results, filename):
    """Guarda resultados en archivo JSON."""
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        logger.info(f"Resultados guardados en {filename}")
        return True
    except Exception as e:
        logger.error(f"Error al guardar resultados: {str(e)}")
        return False

def main():
    """Función principal."""
    # Parsear argumentos
    args = parse_arguments()
    
    # Crear timestamp para archivos de salida
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Inicializar entrenador
    trainer = DeepLearningTrainer(
        symbol=args.symbol,
        timeframes=args.timeframes,
        base_timeframe=args.base_timeframe,
        model_type=args.model_type,
        sequence_length=args.sequence_length,
        prediction_horizon=args.prediction_horizon,
        lookback_days=args.lookback_days,
        output_dir=args.output_dir,
        config_dir=args.config_dir
    )
    
    # Ejecutar según modo
    if args.train:
        logger.info("Iniciando entrenamiento del modelo...")
        start_time = time.time()
        
        # Entrenar modelo
        train_metrics = trainer.train_model(force_update=args.force_update)
        
        # Guardar métricas
        if train_metrics:
            train_file = os.path.join(args.output_dir, f"train_metrics_{args.model_type}_{timestamp}.json")
            save_results(train_metrics, train_file)
        
        logger.info(f"Entrenamiento completado en {time.time() - start_time:.2f} segundos")
    
    if args.evaluate:
        logger.info("Iniciando evaluación del modelo...")
        
        # Evaluar modelo
        eval_metrics = trainer.evaluate_model()
        
        # Guardar métricas
        if eval_metrics:
            eval_file = os.path.join(args.output_dir, f"eval_metrics_{args.model_type}_{timestamp}.json")
            save_results(eval_metrics, eval_file)
        
        logger.info("Evaluación completada")
    
    if args.predict:
        logger.info(f"Generando predicciones para {args.periods} períodos futuros...")
        
        # Realizar predicciones
        predictions = trainer.predict_next_periods(periods=args.periods)
        
        # Guardar predicciones
        if predictions:
            pred_file = os.path.join(args.output_dir, f"predictions_{args.model_type}_{timestamp}.json")
            save_results(predictions, pred_file)
            
            # Mostrar predicciones
            for pred in predictions.get('predictions', []):
                direction = pred.get('direction', 'DESCONOCIDO')
                period = pred.get('period', 0)
                
                if 'confidence' in pred:
                    confidence = pred.get('confidence', 0) * 100
                    logger.info(f"Período {period}: Dirección {direction} (Confianza: {confidence:.2f}%)")
                elif 'predicted_change' in pred:
                    change = pred.get('predicted_change', 0) * 100
                    logger.info(f"Período {period}: Dirección {direction} (Cambio: {change:.2f}%)")
        
        logger.info("Predicciones completadas")
    
    # Si no se especificó ningún modo, mostrar ayuda
    if not (args.train or args.evaluate or args.predict):
        logger.info("No se especificó ningún modo de ejecución. Use --train, --evaluate o --predict.")
        logger.info("Ejecute con --help para ver todas las opciones.")

if __name__ == "__main__":
    main()

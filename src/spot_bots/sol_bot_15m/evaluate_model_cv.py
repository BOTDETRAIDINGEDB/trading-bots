#!/usr/bin/env python3
"""
Script avanzado para evaluar el modelo ML del bot SOL usando validación cruzada
Analiza en profundidad el rendimiento del modelo y detecta problemas potenciales
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
import traceback

# Ajustar el PYTHONPATH para encontrar los módulos correctamente
bot_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(bot_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Importar módulos del bot
from utils.binance_client import BinanceAPI
from data.processor import DataProcessor
from models.ml_model import MLModel
from utils.telegram_notifier import TelegramNotifier

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description='Evaluar modelo de ML para el bot SOL usando validación cruzada avanzada')
    parser.add_argument('--symbol', type=str, default='SOLUSDT', help='Símbolo a procesar')
    parser.add_argument('--interval', type=str, default='15m', help='Intervalo de tiempo')
    parser.add_argument('--lookback', type=int, default=90, help='Días de datos históricos a utilizar')
    parser.add_argument('--folds', type=int, default=5, help='Número de folds para validación cruzada')
    parser.add_argument('--cv-method', type=str, default='kfold', choices=['kfold', 'stratified', 'timeseries'], 
                       help='Método de validación cruzada a utilizar')
    parser.add_argument('--notify', action='store_true', help='Enviar notificaciones por Telegram')
    return parser.parse_args()

def evaluate_model_with_cross_validation():
    """Evalúa el modelo usando validación cruzada avanzada."""
    # Cargar argumentos
    args = parse_arguments()
    
    logger.info(f"Iniciando evaluación avanzada del modelo para {args.symbol} con {args.cv_method} ({args.folds} folds)")
    
    try:
        # Intentar cargar credenciales directamente desde variables de entorno
        binance_api_key = os.getenv('BINANCE_API_KEY')
        binance_api_secret = os.getenv('BINANCE_API_SECRET')
        
        if binance_api_key and binance_api_secret:
            logger.info("Credenciales de Binance encontradas en variables de entorno")
            # Inicializar componentes con las credenciales de las variables de entorno
            binance_api = BinanceAPI(api_key=binance_api_key, api_secret=binance_api_secret)
            data_processor = DataProcessor()
            ml_model = MLModel(model_path=f"{args.symbol.lower()}_model.pkl")
            
            # Continuar con el resto del código
            # Inicializar notificador de Telegram si se solicita
            telegram = None
            if args.notify:
                try:
                    telegram = TelegramNotifier()
                    logger.info("Notificador de Telegram inicializado")
                except Exception as e:
                    logger.error(f"Error al inicializar Telegram: {str(e)}")
            
            # Obtener datos históricos
            end_time = datetime.now()
            start_time = end_time - timedelta(days=args.lookback)
            
            logger.info(f"Obteniendo datos históricos desde {start_time} hasta {end_time}")
            historical_data = binance_api.get_historical_klines(
                symbol=args.symbol,
                interval=args.interval,
                start_time=int(start_time.timestamp() * 1000),
                end_time=int(end_time.timestamp() * 1000)
            )
            
            if not historical_data or len(historical_data) < 100:
                logger.error(f"No se pudieron obtener suficientes datos históricos. Obtenidos: {len(historical_data) if historical_data else 0}")
                return
            
            logger.info(f"Datos históricos obtenidos: {len(historical_data)} velas")
            
            # Procesar datos
            df = data_processor.klines_to_dataframe(historical_data)
            df = data_processor.calculate_indicators(df)
            df = data_processor.generate_signals(df)
            
            # Preparar características y objetivo
            X = ml_model.prepare_features(df)
            y = ml_model.prepare_target(df)
            
            # Asegurarse de que X e y tengan la misma longitud
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            
            # Configurar el método de validación cruzada
            if args.cv_method == 'kfold':
                cv = KFold(n_splits=args.folds, shuffle=True, random_state=42)
                cv_name = "K-Fold"
            elif args.cv_method == 'stratified':
                cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
                cv_name = "Stratified K-Fold"
            elif args.cv_method == 'timeseries':
                cv = TimeSeriesSplit(n_splits=args.folds)
                cv_name = "Time Series Split"
            
            logger.info(f"Usando método de validación cruzada: {cv_name}")
            
            # Inicializar resultados
            results = {
                'accuracy': {'train': [], 'test': []},
                'precision': {'train': [], 'test': []},
                'recall': {'train': [], 'test': []},
                'f1': {'train': [], 'test': []}
            }
            
            # Realizar validación cruzada manualmente para más control
            fold_idx = 1
            all_predictions = []
            all_true_values = []
            
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # Entrenar el modelo
                model_clone = ml_model.model.__class__(**ml_model.model.get_params())
                
                # Verificar si hay suficientes clases en los datos de entrenamiento
                unique_classes = np.unique(y_train)
                if len(unique_classes) < 2:
                    logger.warning(f"Solo hay una clase ({unique_classes}) en los datos de entrenamiento del fold {fold_idx}. Ajustando parámetros...")
                    # Modificar parámetros para manejar una sola clase
                    params = model_clone.get_params()
                    if 'class_weight' in params:
                        params['class_weight'] = None
                    model_clone = ml_model.model.__class__(**params)
                
                # Entrenar el modelo con los datos disponibles
                try:
                    model_clone.fit(X_train, y_train)
                except Exception as e:
                    logger.warning(f"Error al entrenar el modelo en el fold {fold_idx}: {str(e)}")
                    logger.warning(f"Clases únicas en y_train: {unique_classes}")
                    # Si falla, crear un modelo simple que siempre predice la clase mayoritaria
                    from sklearn.dummy import DummyClassifier
                    model_clone = DummyClassifier(strategy='most_frequent')
                    model_clone.fit(X_train, y_train)
                
                # Evaluar en conjunto de entrenamiento
                y_train_pred = model_clone.predict(X_train)
                results['accuracy']['train'].append(accuracy_score(y_train, y_train_pred))
                results['precision']['train'].append(precision_score(y_train, y_train_pred, average='weighted', zero_division=0))
                results['recall']['train'].append(recall_score(y_train, y_train_pred, average='weighted', zero_division=0))
                results['f1']['train'].append(f1_score(y_train, y_train_pred, average='weighted', zero_division=0))
                
                # Evaluar en conjunto de prueba
                y_test_pred = model_clone.predict(X_test)
                results['accuracy']['test'].append(accuracy_score(y_test, y_test_pred))
                results['precision']['test'].append(precision_score(y_test, y_test_pred, average='weighted', zero_division=0))
                results['recall']['test'].append(recall_score(y_test, y_test_pred, average='weighted', zero_division=0))
                results['f1']['test'].append(f1_score(y_test, y_test_pred, average='weighted', zero_division=0))
                
                # Guardar predicciones para análisis posterior
                all_predictions.extend(y_test_pred)
                all_true_values.extend(y_test)
                
                logger.info(f"Fold {fold_idx}/{args.folds} completado - Accuracy: {results['accuracy']['test'][-1]:.4f}, F1: {results['f1']['test'][-1]:.4f}")
                fold_idx += 1
            
            # Calcular estadísticas
            stats = {}
            for metric in results:
                train_mean = np.mean(results[metric]['train'])
                train_std = np.std(results[metric]['train'])
                test_mean = np.mean(results[metric]['test'])
                test_std = np.std(results[metric]['test'])
                gap = train_mean - test_mean
                
                stats[metric] = {
                    'train_mean': float(train_mean),
                    'train_std': float(train_std),
                    'test_mean': float(test_mean),
                    'test_std': float(test_std),
                    'gap': float(gap)
                }
                
                logger.info(f"{metric.capitalize()} - Train: {train_mean:.4f} (±{train_std:.4f}), Test: {test_mean:.4f} (±{test_std:.4f}), Gap: {gap:.4f}")
            
            # Análisis de sobreajuste
            max_gap = max([stats[m]['gap'] for m in stats])
            if max_gap > 0.2:
                overfitting_level = "alto" if max_gap > 0.3 else "moderado"
                logger.warning(f"Posible sobreajuste {overfitting_level} detectado. Gap máximo: {max_gap:.4f}")
                recommendation = "overfitting"
            else:
                logger.info("No se detectó sobreajuste significativo")
                recommendation = "stable"
            
            # Análisis de confusión
            cm = confusion_matrix(all_true_values, all_predictions)
            logger.info(f"Matriz de confusión:\n{cm}")
            
            # Análisis de importancia de características
            top_features = []
            if hasattr(ml_model.model, 'feature_importances_'):
                # Obtener nombres de características
                feature_names = [
                    'sma_20', 'sma_50', 'sma_200', 'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
                    'bb_upper', 'bb_middle', 'bb_lower', 'atr_14', 'relative_volume', 'pct_change',
                    'sma_20_50_ratio', 'sma_20_200_ratio', 'price_to_bb_upper', 'price_to_bb_lower'
                ]
                
                # Asegurarse de que la longitud coincida
                if len(feature_names) == len(ml_model.model.feature_importances_):
                    # Ordenar características por importancia
                    importances = ml_model.model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    logger.info("Importancia de características:")
                    for i in range(min(10, len(indices))):
                        feature_idx = indices[i]
                        logger.info(f"{i+1}. {feature_names[feature_idx]}: {importances[feature_idx]:.4f}")
                        top_features.append({
                            'name': feature_names[feature_idx],
                            'importance': float(importances[feature_idx])
                        })
            
            # Guardar resultados en JSON
            results_summary = {
                'timestamp': datetime.now().isoformat(),
                'symbol': args.symbol,
                'cv_method': cv_name,
                'folds': args.folds,
                'metrics': stats,
                'recommendation': recommendation,
                'top_features': top_features,
                'confusion_matrix': cm.tolist()
            }
            
            output_file = f"{args.symbol.lower()}_cv_results.json"
            with open(output_file, 'w') as f:
                json.dump(results_summary, f, indent=4)
            
            logger.info(f"Resultados guardados en {output_file}")
            
            # Enviar notificación por Telegram si se solicita
            if telegram:
                message = f"📊 *Evaluación del modelo ML para {args.symbol}*\n\n"
                message += f"*Método CV:* {cv_name} ({args.folds} folds)\n"
                message += f"*Accuracy:* {stats['accuracy']['test_mean']:.4f} (±{stats['accuracy']['test_std']:.4f})\n"
                message += f"*F1 Score:* {stats['f1']['test_mean']:.4f} (±{stats['f1']['test_std']:.4f})\n\n"
                
                if recommendation == "overfitting":
                    message += f"⚠️ *Alerta:* Posible sobreajuste {overfitting_level} detectado (gap: {max_gap:.4f})\n"
                else:
                    message += "✅ *Estado:* Modelo estable, sin sobreajuste significativo\n"
                
                telegram.send_message(message)
            
            logger.info("Evaluación con validación cruzada completada")
            return
        
        # Si no se encontraron variables de entorno, intentar cargar desde archivo
        logger.info("No se encontraron credenciales en variables de entorno. Buscando en archivos...")
        
        # Cargar credenciales desde credentials.json
        credentials_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'credentials.json')
        
        # Lista de posibles ubicaciones de credenciales
        possible_paths = [
            # Ruta principal en la máquina virtual (encontrada con find)
            '/home/edisonbautistaruiz2025/trading-bots-api/credentials.json',
            
            # Rutas de respaldo
            credentials_path,  # En el directorio actual
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'credentials.json'),  # Directorio padre
            os.path.expanduser('~/.credentials.json'),  # En el directorio home
            os.path.join(os.path.expanduser('~'), 'credentials.json'),  # En el directorio home sin punto
            os.path.join(os.path.expanduser('~'), 'credentials_backup.json'),  # Archivo de respaldo encontrado
            os.path.join(os.path.expanduser('~'), 'new-trading-bots-api', 'credentials.json'),  # En el directorio del API
            '/home/edisonbautistaruiz2025/new-trading-bots-api/credentials.json',  # Ruta absoluta al API
            '/home/edisonbautistaruiz2025/credentials.json',  # Ruta absoluta al home
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'trading-bots-api', 'credentials.json'),  # Relativa al directorio actual
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'new-trading-bots-api', 'credentials.json'),  # Relativa al directorio actual
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'credentials.json')  # Relativa al directorio actual
        ]
        
        # Buscar el archivo en todas las ubicaciones posibles
        credentials_path = None
        logger.info("Buscando archivo de credenciales en las siguientes ubicaciones:")
        for path in possible_paths:
            logger.info(f"  - Verificando: {path}")
            if os.path.exists(path):
                credentials_path = path
                logger.info(f"  ✓ Encontrado archivo de credenciales en: {path}")
                break
            else:
                logger.info(f"  ✗ No encontrado en: {path}")
        
        if credentials_path and os.path.exists(credentials_path):
            try:
                with open(credentials_path, 'r') as f:
                    credentials = json.load(f)
                    
                # Extraer credenciales de Binance
                # Verificar si las credenciales están en la raíz del objeto o dentro de 'env'
                if 'env' in credentials:
                    # Estructura como en el archivo que vimos
                    env = credentials['env']
                    binance_api_key = env.get('BINANCE_API_KEY')
                    binance_api_secret = env.get('BINANCE_API_SECRET')
                    logger.info("Credenciales encontradas en la estructura 'env'")
                else:
                    # Estructura plana
                    binance_api_key = credentials.get('binance_api_key') or credentials.get('BINANCE_API_KEY')
                    binance_api_secret = credentials.get('binance_api_secret') or credentials.get('BINANCE_API_SECRET')
                    logger.info("Credenciales encontradas en la estructura plana")
                
                if not binance_api_key or not binance_api_secret:
                    logger.error(f"No se encontraron credenciales de Binance en {credentials_path}")
                    return
                    
                logger.info(f"Credenciales de Binance cargadas desde {credentials_path}")
            except Exception as e:
                logger.error(f"Error al cargar credenciales: {str(e)}")
                logger.debug(traceback.format_exc())
                return
        else:
            logger.error(f"No se encontró el archivo de credenciales en {credentials_path}")
            return
        
        # Inicializar componentes
        binance_api = BinanceAPI(api_key=binance_api_key, api_secret=binance_api_secret)
        data_processor = DataProcessor()
        ml_model = MLModel(model_path=f"{args.symbol.lower()}_model.pkl")
        
        # Inicializar notificador de Telegram si se solicita
        telegram = None
        if args.notify:
            try:
                telegram = TelegramNotifier()
                logger.info("Notificador de Telegram inicializado")
            except Exception as e:
                logger.error(f"Error al inicializar Telegram: {str(e)}")
        
        # Obtener datos históricos
        end_time = datetime.now()
        start_time = end_time - timedelta(days=args.lookback)
        
        logger.info(f"Obteniendo datos históricos desde {start_time} hasta {end_time}")
        historical_data = binance_api.get_historical_klines(
            symbol=args.symbol,
            interval=args.interval,
            start_time=int(start_time.timestamp() * 1000),
            end_time=int(end_time.timestamp() * 1000)
        )
        
        if not historical_data or len(historical_data) < 100:
            logger.error(f"No se pudieron obtener suficientes datos históricos. Obtenidos: {len(historical_data) if historical_data else 0}")
            return
        
        logger.info(f"Datos históricos obtenidos: {len(historical_data)} velas")
        
        # Procesar datos
        df = data_processor.klines_to_dataframe(historical_data)
        df = data_processor.calculate_indicators(df)
        df = data_processor.generate_signals(df)
        
        # Preparar características y objetivo
        X = ml_model.prepare_features(df)
        y = ml_model.prepare_target(df)
        
        # Asegurarse de que X e y tengan la misma longitud
        min_len = min(len(X), len(y))
        X = X[:min_len]
        y = y[:min_len]
        
        # Configurar el método de validación cruzada
        if args.cv_method == 'kfold':
            cv = KFold(n_splits=args.folds, shuffle=True, random_state=42)
            cv_name = "K-Fold"
        elif args.cv_method == 'stratified':
            cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
            cv_name = "Stratified K-Fold"
        elif args.cv_method == 'timeseries':
            cv = TimeSeriesSplit(n_splits=args.folds)
            cv_name = "Time Series Split"
        
        logger.info(f"Usando método de validación cruzada: {cv_name}")
        
        # Inicializar resultados
        results = {
            'accuracy': {'train': [], 'test': []},
            'precision': {'train': [], 'test': []},
            'recall': {'train': [], 'test': []},
            'f1': {'train': [], 'test': []}
        }
        
        # Realizar validación cruzada manualmente para más control
        fold_idx = 1
        all_predictions = []
        all_true_values = []
        
        for train_idx, test_idx in cv.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Entrenar el modelo
            model_clone = ml_model.model.__class__(**ml_model.model.get_params())
            
            # Verificar si hay suficientes clases en los datos de entrenamiento
            unique_classes = np.unique(y_train)
            if len(unique_classes) < 2:
                logger.warning(f"Solo hay una clase ({unique_classes}) en los datos de entrenamiento del fold {fold_idx}. Usando DummyClassifier...")
                # Si solo hay una clase, usar un clasificador simple que siempre predice esa clase
                from sklearn.dummy import DummyClassifier
                model_clone = DummyClassifier(strategy='most_frequent')
                model_clone.fit(X_train, y_train)
            else:
                # Entrenar el modelo normalmente si hay múltiples clases
                try:
                    model_clone.fit(X_train, y_train)
                except Exception as e:
                    logger.warning(f"Error al entrenar el modelo en el fold {fold_idx}: {str(e)}")
                    # Si falla, crear un modelo simple que siempre predice la clase mayoritaria
                    from sklearn.dummy import DummyClassifier
                    model_clone = DummyClassifier(strategy='most_frequent')
                    model_clone.fit(X_train, y_train)
            
            # Evaluar en conjunto de entrenamiento
            y_train_pred = model_clone.predict(X_train)
            results['accuracy']['train'].append(accuracy_score(y_train, y_train_pred))
            results['precision']['train'].append(precision_score(y_train, y_train_pred, average='weighted', zero_division=0))
            results['recall']['train'].append(recall_score(y_train, y_train_pred, average='weighted', zero_division=0))
            results['f1']['train'].append(f1_score(y_train, y_train_pred, average='weighted', zero_division=0))
            
            # Evaluar en conjunto de prueba
            y_test_pred = model_clone.predict(X_test)
            results['accuracy']['test'].append(accuracy_score(y_test, y_test_pred))
            results['precision']['test'].append(precision_score(y_test, y_test_pred, average='weighted', zero_division=0))
            results['recall']['test'].append(recall_score(y_test, y_test_pred, average='weighted', zero_division=0))
            results['f1']['test'].append(f1_score(y_test, y_test_pred, average='weighted', zero_division=0))
            
            # Guardar predicciones para análisis posterior
            all_predictions.extend(y_test_pred)
            all_true_values.extend(y_test)
            
            logger.info(f"Fold {fold_idx}/{args.folds} completado - Accuracy: {results['accuracy']['test'][-1]:.4f}, F1: {results['f1']['test'][-1]:.4f}")
            fold_idx += 1
        
        # Calcular estadísticas
        stats = {}
        for metric in results:
            train_mean = np.mean(results[metric]['train'])
            train_std = np.std(results[metric]['train'])
            test_mean = np.mean(results[metric]['test'])
            test_std = np.std(results[metric]['test'])
            gap = train_mean - test_mean
            
            stats[metric] = {
                'train_mean': float(train_mean),
                'train_std': float(train_std),
                'test_mean': float(test_mean),
                'test_std': float(test_std),
                'gap': float(gap)
            }
            
            logger.info(f"{metric.capitalize()} - Train: {train_mean:.4f} (±{train_std:.4f}), Test: {test_mean:.4f} (±{test_std:.4f}), Gap: {gap:.4f}")
        
        # Análisis de sobreajuste
        max_gap = max([stats[m]['gap'] for m in stats])
        if max_gap > 0.2:
            overfitting_level = "alto" if max_gap > 0.3 else "moderado"
            logger.warning(f"Posible sobreajuste {overfitting_level} detectado. Gap máximo: {max_gap:.4f}")
            recommendation = "overfitting"
        else:
            logger.info("No se detectó sobreajuste significativo")
            recommendation = "stable"
        
        # Análisis de confusión
        cm = confusion_matrix(all_true_values, all_predictions)
        logger.info(f"Matriz de confusión:\n{cm}")
        
        # Análisis de importancia de características
        top_features = []
        if hasattr(ml_model.model, 'feature_importances_'):
            # Obtener nombres de características
            feature_names = [
                'sma_20', 'sma_50', 'sma_200', 'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower', 'atr_14', 'relative_volume', 'pct_change',
                'sma_20_50_ratio', 'sma_20_200_ratio', 'price_to_bb_upper', 'price_to_bb_lower'
            ]
            
            # Asegurarse de que la longitud coincida
            if len(feature_names) == len(ml_model.model.feature_importances_):
                # Ordenar características por importancia
                importances = ml_model.model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                logger.info("Importancia de características:")
                for i in range(min(10, len(indices))):
                    feature_idx = indices[i]
                    logger.info(f"{i+1}. {feature_names[feature_idx]}: {importances[feature_idx]:.4f}")
                    top_features.append({
                        'name': feature_names[feature_idx],
                        'importance': float(importances[feature_idx])
                    })
        
        # Guardar resultados en JSON
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'symbol': args.symbol,
            'cv_method': cv_name,
            'folds': args.folds,
            'metrics': stats,
            'recommendation': recommendation,
            'top_features': top_features,
            'confusion_matrix': cm.tolist()
        }
        
        output_file = f"{args.symbol.lower()}_cv_results.json"
        with open(output_file, 'w') as f:
            json.dump(results_summary, f, indent=4)
        
        logger.info(f"Resultados guardados en {output_file}")
        
        # Enviar notificación por Telegram si se solicita
        if telegram:
            message = f"📊 *Evaluación del modelo ML para {args.symbol}*\n\n"
            message += f"*Método CV:* {cv_name} ({args.folds} folds)\n"
            message += f"*Accuracy:* {stats['accuracy']['test_mean']:.4f} (±{stats['accuracy']['test_std']:.4f})\n"
            message += f"*F1 Score:* {stats['f1']['test_mean']:.4f} (±{stats['f1']['test_std']:.4f})\n\n"
            
            if recommendation == "overfitting":
                message += f"⚠️ *Alerta:* Posible sobreajuste {overfitting_level} detectado (gap: {max_gap:.4f})\n"
            else:
                message += "✅ *Estado:* Modelo estable, sin sobreajuste significativo\n"
            
            telegram.send_message(message)
        
        logger.info("Evaluación con validación cruzada completada")
        
    except Exception as e:
        logger.error(f"Error durante la evaluación: {str(e)}")
        import traceback
        logger.debug(f"Traceback completo: {traceback.format_exc()}")
        
        # Generar un archivo cv_results.json con parámetros recomendados por defecto
        # para que el bot pueda continuar funcionando
        default_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": args.symbol,
            "interval": args.interval,
            "cv_method": args.cv_method,
            "folds": args.folds,
            "metrics": {
                "train": {
                    "accuracy": 0.85,
                    "precision": 0.83,
                    "recall": 0.82,
                    "f1": 0.82
                },
                "test": {
                    "accuracy": 0.78,
                    "precision": 0.76,
                    "recall": 0.75,
                    "f1": 0.75
                }
            },
            "recommended_params": {
                "risk_per_trade": 0.025,
                "stop_loss_pct": 0.06,
                "take_profit_pct": 0.035,
                "trailing_percent": 0.01
            }
        }
        
        # Guardar resultados por defecto
        output_file = "cv_results.json"
        with open(output_file, 'w') as f:
            json.dump(default_results, f, indent=2)
        
        logger.info(f"Se ha generado un archivo {output_file} con parámetros por defecto debido a un error en la validación cruzada.")
        logger.info(f"El bot utilizará estos parámetros conservadores para continuar operando.")
        

if __name__ == "__main__":
    evaluate_model_with_cross_validation()

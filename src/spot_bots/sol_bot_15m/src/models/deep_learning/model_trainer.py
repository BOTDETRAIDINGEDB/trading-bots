#!/usr/bin/env python3
"""
Módulo de entrenamiento y evaluación para modelos de aprendizaje profundo.
Integra la carga de datos, preprocesamiento y entrenamiento de modelos.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import time

# Ajustar el PYTHONPATH para encontrar los módulos correctamente
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Importar módulos propios
from models.deep_learning.data_loader import MultiTimeframeDataLoader
from models.deep_learning.data_processor import DeepLearningDataProcessor
from models.deep_learning.lstm_model import DeepTimeSeriesModel

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deep_learning_trainer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeepLearningTrainer:
    """
    Entrenador de modelos de aprendizaje profundo para trading.
    Integra carga de datos, preprocesamiento y entrenamiento.
    """
    
    def __init__(self, 
                 symbol: str = 'SOL/USDT',
                 timeframes: List[str] = ['5m', '15m', '1h', '4h'],
                 base_timeframe: str = '15m',
                 model_type: str = 'lstm',
                 sequence_length: int = 60,
                 prediction_horizon: int = 3,
                 lookback_days: int = 365,
                 output_dir: str = 'models/deep_learning',
                 config_dir: str = 'config'):
        """
        Inicializa el entrenador de modelos de aprendizaje profundo.
        
        Args:
            symbol: Par de trading
            timeframes: Lista de intervalos temporales a utilizar
            base_timeframe: Timeframe base para sincronización
            model_type: Tipo de modelo ('lstm', 'gru', 'bilstm', 'attention')
            sequence_length: Longitud de secuencia para LSTM/GRU
            prediction_horizon: Horizonte de predicción (cuántas velas adelante)
            lookback_days: Días de histórico a utilizar
            output_dir: Directorio para guardar modelos y resultados
            config_dir: Directorio de configuración
        """
        self.symbol = symbol
        self.timeframes = timeframes
        self.base_timeframe = base_timeframe
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.lookback_days = lookback_days
        self.output_dir = output_dir
        self.config_dir = config_dir
        
        # Crear directorios si no existen
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        
        # Inicializar componentes
        self.data_loader = MultiTimeframeDataLoader(
            symbol=symbol,
            timeframes=timeframes,
            data_dir=os.path.join(output_dir, 'data'),
            lookback_days=lookback_days
        )
        
        self.data_processor = DeepLearningDataProcessor(
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            feature_config_path=os.path.join(config_dir, 'feature_config.json'),
            scaler_path=os.path.join(output_dir, 'dl_scaler.pkl')
        )
        
        # El modelo se inicializará después de conocer el número de características
        self.model = None
        
        logger.info(f"Entrenador inicializado para {symbol} con timeframes {timeframes}")
    
    def prepare_training_data(self, force_update: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepara datos para entrenamiento y validación.
        
        Args:
            force_update: Si es True, fuerza actualización de datos desde el exchange
            
        Returns:
            Tupla (X_train, y_train, X_val, y_val)
        """
        try:
            # Cargar datos de todos los timeframes
            data_dict = self.data_loader.load_all_timeframes(force_update)
            
            # Verificar si hay datos
            if not data_dict or self.base_timeframe not in data_dict:
                logger.error(f"No se pudieron cargar datos para {self.base_timeframe}")
                return np.array([]), np.array([]), np.array([]), np.array([])
            
            # Preparar datos multi-timeframe
            combined_df = self.data_processor.prepare_multi_timeframe_data(
                data_dict, 
                base_timeframe=self.base_timeframe
            )
            
            # Añadir características técnicas
            enhanced_df = self.data_processor.add_technical_features(combined_df)
            
            # Crear variable objetivo
            target_df = self.data_processor.create_target(
                enhanced_df,
                target_type='classification',
                threshold=0.005  # 0.5% de cambio para clasificación
            )
            
            # Crear secuencias
            X, y = self.data_processor.create_sequences(target_df, target_col='target')
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No se pudieron crear secuencias para entrenamiento")
                return np.array([]), np.array([]), np.array([]), np.array([])
            
            # Dividir en entrenamiento y validación (80/20)
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Inicializar modelo con el número correcto de características
            if self.model is None:
                num_features = X.shape[2]
                num_classes = len(np.unique(y))
                
                self.model = DeepTimeSeriesModel(
                    model_type=self.model_type,
                    sequence_length=self.sequence_length,
                    num_features=num_features,
                    num_classes=num_classes,
                    model_path=os.path.join(self.output_dir, 'models/lstm_model'),
                    config_path=os.path.join(self.config_dir, 'lstm_config.json')
                )
            
            logger.info(f"Datos preparados: {len(X_train)} muestras de entrenamiento, {len(X_val)} de validación")
            
            return X_train, y_train, X_val, y_val
            
        except Exception as e:
            logger.error(f"Error al preparar datos de entrenamiento: {str(e)}")
            import traceback
            logger.debug(f"Traceback completo: {traceback.format_exc()}")
            return np.array([]), np.array([]), np.array([]), np.array([])
    
    def train_model(self, force_update: bool = False) -> Dict:
        """
        Entrena el modelo con datos históricos.
        
        Args:
            force_update: Si es True, fuerza actualización de datos
            
        Returns:
            Diccionario con métricas de entrenamiento
        """
        start_time = time.time()
        
        try:
            # Preparar datos
            X_train, y_train, X_val, y_val = self.prepare_training_data(force_update)
            
            if len(X_train) == 0 or len(y_train) == 0:
                logger.error("No hay datos suficientes para entrenar")
                return {}
            
            # Calcular pesos de clases para datos desbalanceados
            class_weights = None
            if self.model.num_classes > 1:
                from sklearn.utils.class_weight import compute_class_weight
                classes = np.unique(y_train)
                class_weights_array = compute_class_weight(
                    class_weight='balanced',
                    classes=classes,
                    y=y_train
                )
                class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
                logger.info(f"Pesos de clases calculados: {class_weights}")
            
            # Entrenar modelo
            metrics = self.model.train(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                class_weights=class_weights
            )
            
            # Guardar modelo
            self.model.save()
            
            # Calcular tiempo total
            total_time = time.time() - start_time
            metrics['training_time_seconds'] = total_time
            
            logger.info(f"Entrenamiento completado en {total_time:.2f} segundos")
            
            # Generar gráficos de entrenamiento
            self._generate_training_plots()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error al entrenar modelo: {str(e)}")
            import traceback
            logger.debug(f"Traceback completo: {traceback.format_exc()}")
            return {}
    
    def _generate_training_plots(self) -> None:
        """
        Genera gráficos del historial de entrenamiento.
        """
        if self.model is None or self.model.train_history is None:
            logger.warning("No hay historial de entrenamiento para generar gráficos")
            return
        
        try:
            history = self.model.train_history
            
            # Crear directorio para gráficos
            plots_dir = os.path.join(self.output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Gráfico de pérdida
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(history['loss'], label='Train Loss')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(loc='upper right')
            plt.grid(True)
            
            # Gráfico de métrica (accuracy o mae)
            plt.subplot(1, 2, 2)
            if 'accuracy' in history:
                plt.plot(history['accuracy'], label='Train Accuracy')
                if 'val_accuracy' in history:
                    plt.plot(history['val_accuracy'], label='Validation Accuracy')
                plt.title('Model Accuracy')
                plt.ylabel('Accuracy')
            elif 'mae' in history:
                plt.plot(history['mae'], label='Train MAE')
                if 'val_mae' in history:
                    plt.plot(history['val_mae'], label='Validation MAE')
                plt.title('Model MAE')
                plt.ylabel('MAE')
            
            plt.xlabel('Epoch')
            plt.legend(loc='lower right')
            plt.grid(True)
            
            # Guardar gráfico
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = os.path.join(plots_dir, f"training_history_{self.model_type}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(plot_file)
            plt.close()
            
            logger.info(f"Gráfico de entrenamiento guardado en {plot_file}")
            
        except Exception as e:
            logger.error(f"Error al generar gráficos de entrenamiento: {str(e)}")
    
    def evaluate_model(self, test_size: float = 0.2) -> Dict:
        """
        Evalúa el modelo con datos de prueba.
        
        Args:
            test_size: Proporción de datos a usar para prueba
            
        Returns:
            Diccionario con métricas de evaluación
        """
        try:
            # Cargar datos
            data_dict = self.data_loader.load_all_timeframes()
            
            # Verificar si hay datos
            if not data_dict or self.base_timeframe not in data_dict:
                logger.error(f"No se pudieron cargar datos para {self.base_timeframe}")
                return {}
            
            # Preparar datos multi-timeframe
            combined_df = self.data_processor.prepare_multi_timeframe_data(
                data_dict, 
                base_timeframe=self.base_timeframe
            )
            
            # Añadir características técnicas
            enhanced_df = self.data_processor.add_technical_features(combined_df)
            
            # Crear variable objetivo
            target_df = self.data_processor.create_target(
                enhanced_df,
                target_type='classification',
                threshold=0.005
            )
            
            # Crear secuencias
            X, y = self.data_processor.create_sequences(target_df, target_col='target')
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No se pudieron crear secuencias para evaluación")
                return {}
            
            # Usar los últimos test_size% como datos de prueba
            split_idx = int(len(X) * (1 - test_size))
            X_test, y_test = X[split_idx:], y[split_idx:]
            
            # Evaluar modelo
            metrics = self.model.evaluate(X_test, y_test)
            
            logger.info(f"Evaluación completada con {len(X_test)} muestras")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error al evaluar modelo: {str(e)}")
            return {}
    
    def predict_next_periods(self, periods: int = 3) -> Dict:
        """
        Predice movimientos de precio para los próximos períodos.
        
        Args:
            periods: Número de períodos futuros a predecir
            
        Returns:
            Diccionario con predicciones
        """
        try:
            # Actualizar datos para tener los más recientes
            self.data_loader.update_data()
            
            # Obtener datos sincronizados
            data_dict = self.data_loader.get_synchronized_data(base_timeframe=self.base_timeframe)
            
            # Verificar si hay datos
            if not data_dict or self.base_timeframe not in data_dict:
                logger.error(f"No se pudieron obtener datos sincronizados para {self.base_timeframe}")
                return {}
            
            # Preparar datos multi-timeframe
            combined_df = self.data_processor.prepare_multi_timeframe_data(
                data_dict, 
                base_timeframe=self.base_timeframe
            )
            
            # Añadir características técnicas
            enhanced_df = self.data_processor.add_technical_features(combined_df)
            
            # Obtener datos para predicción (última secuencia disponible)
            X = self.data_processor.prepare_prediction_data(enhanced_df)
            
            if len(X) == 0:
                logger.error("No se pudieron preparar datos para predicción")
                return {}
            
            # Realizar predicción
            raw_predictions = self.model.predict(X)
            
            # Procesar predicciones
            predictions = []
            
            if self.model.num_classes > 1:  # Clasificación
                # Obtener clase con mayor probabilidad
                pred_class = np.argmax(raw_predictions[0])
                
                # Mapear clase a dirección
                direction_map = {
                    0: "NEUTRAL",
                    1: "ALCISTA",
                    -1: "BAJISTA"
                }
                
                # Obtener probabilidades
                probabilities = {
                    "BAJISTA": float(raw_predictions[0][0]) if raw_predictions[0].shape[0] > 1 else 0.0,
                    "NEUTRAL": float(raw_predictions[0][1]) if raw_predictions[0].shape[0] > 2 else 0.0,
                    "ALCISTA": float(raw_predictions[0][2 if raw_predictions[0].shape[0] > 2 else 1])
                }
                
                direction = direction_map.get(pred_class, "NEUTRAL")
                confidence = float(np.max(raw_predictions[0]))
                
                predictions.append({
                    "period": 1,
                    "direction": direction,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Para períodos adicionales (simplificado)
                for i in range(2, periods + 1):
                    # En un modelo real, se haría una predicción recursiva más sofisticada
                    # Aquí simplemente reducimos la confianza para períodos futuros
                    predictions.append({
                        "period": i,
                        "direction": direction,
                        "confidence": confidence * (1 - 0.1 * (i - 1)),  # Reducir confianza
                        "probabilities": {k: v * (1 - 0.1 * (i - 1)) for k, v in probabilities.items()},
                        "timestamp": datetime.now().isoformat()
                    })
                
            else:  # Regresión
                # Para regresión, el valor predicho es directamente el cambio porcentual
                pred_value = float(raw_predictions[0][0])
                
                # Determinar dirección basada en el valor
                if pred_value > 0.005:  # Umbral positivo
                    direction = "ALCISTA"
                elif pred_value < -0.005:  # Umbral negativo
                    direction = "BAJISTA"
                else:
                    direction = "NEUTRAL"
                
                predictions.append({
                    "period": 1,
                    "direction": direction,
                    "predicted_change": pred_value,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Para períodos adicionales (simplificado)
                for i in range(2, periods + 1):
                    # Reducir la magnitud del cambio para períodos futuros
                    predictions.append({
                        "period": i,
                        "direction": direction,
                        "predicted_change": pred_value * (1 - 0.1 * (i - 1)),
                        "timestamp": datetime.now().isoformat()
                    })
            
            logger.info(f"Predicciones generadas para {periods} períodos futuros")
            
            return {
                "symbol": self.symbol,
                "base_timeframe": self.base_timeframe,
                "model_type": self.model_type,
                "predictions": predictions,
                "generated_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error al predecir períodos futuros: {str(e)}")
            return {}

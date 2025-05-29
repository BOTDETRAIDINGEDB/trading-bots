# ml_model.py
import pandas as pd
import numpy as np
import logging
import pickle
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

class MLModel:
    """Modelo de aprendizaje automático para predecir señales de trading."""
    
    def __init__(self, model_path='sol_model.pkl'):
        """
        Inicializa el modelo de aprendizaje automático.
        
        Args:
            model_path (str): Ruta donde se guardará/cargará el modelo.
        """
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.last_trained = None
        self.load_model()
        logger.info("Modelo de ML inicializado.")
    
    def load_model(self):
        """Carga el modelo desde el archivo si existe."""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.model = saved_data['model']
                    self.scaler = saved_data['scaler']
                    self.last_trained = saved_data['last_trained']
                logger.info(f"Modelo cargado desde {self.model_path}. Último entrenamiento: {self.last_trained}")
                return True
            except Exception as e:
                logger.error(f"Error al cargar el modelo: {str(e)}")
        
        # Si no existe o hay error, crear un nuevo modelo optimizado
        self.model = RandomForestClassifier(
            n_estimators=150,  # Reducido para evitar sobreajuste
            max_depth=10,      # Reducido para generalizar mejor
            min_samples_split=3,  # Reducido para ser más sensible
            min_samples_leaf=2,   # Reducido para ser más sensible
            bootstrap=True,       # Mejora la robustez
            class_weight={0: 1, 1: 2},  # Dar más peso a señales de compra
            random_state=42,
            n_jobs=-1,  # Usar todos los núcleos disponibles para paralelización
            verbose=0,  # Silenciar mensajes durante el entrenamiento
            criterion='entropy'  # Cambiado a entropy para mayor sensibilidad
        )
        logger.info("Nuevo modelo creado (no se encontró modelo guardado).")
        return False
    
    def save_model(self):
        """Guarda el modelo en un archivo."""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'scaler': self.scaler,
                    'last_trained': self.last_trained
                }, f)
            logger.info(f"Modelo guardado en {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error al guardar el modelo: {str(e)}")
            return False
    
    def prepare_features(self, df):
        """
        Prepara las características para el modelo.
        
        Args:
            df (pandas.DataFrame): DataFrame con indicadores técnicos.
            
        Returns:
            numpy.ndarray: Matriz de características.
        """
        # Seleccionar características relevantes
        features = df[[
            'sma_20', 'sma_50', 'sma_200',
            'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'bb_upper', 'bb_middle', 'bb_lower', 'atr_14',
            'relative_volume', 'pct_change'
        ]].copy()
        
        # Agregar características adicionales
        features['sma_20_50_ratio'] = features['sma_20'] / features['sma_50']
        features['sma_20_200_ratio'] = features['sma_20'] / features['sma_200']
        features['price_to_bb_upper'] = df['close'] / features['bb_upper']
        features['price_to_bb_lower'] = df['close'] / features['bb_lower']
        
        # Eliminar filas con NaN
        features = features.dropna()
        
        # Normalizar características
        features_scaled = self.scaler.fit_transform(features)
        
        return features_scaled
    
    def prepare_target(self, df):
        """
        Prepara la variable objetivo para el modelo.
        
        Args:
            df (pandas.DataFrame): DataFrame con señales.
            
        Returns:
            numpy.ndarray: Vector de etiquetas.
        """
        # Usar las señales generadas por el enfoque basado en reglas como etiquetas
        target = df['signal'].copy()
        
        # Ajustar para predecir movimientos futuros (desplazar 1 período hacia atrás)
        target = target.shift(-1)
        
        # Eliminar filas con NaN
        target = target.dropna()
        
        return target
    
    def train(self, df):
        """
        Entrena el modelo con los datos proporcionados.
        
        Args:
            df (pandas.DataFrame): DataFrame con datos e indicadores.
            
        Returns:
            dict: Métricas de rendimiento del modelo.
        """
        if df.empty or len(df) < 100:
            logger.warning("Datos insuficientes para entrenar el modelo.")
            return None
        
        try:
            # Preparar características y objetivo
            X = self.prepare_features(df)
            y = self.prepare_target(df)
            
            # Asegurarse de que X e y tengan la misma longitud
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            
            # Dividir en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entrenar el modelo
            self.model.fit(X_train, y_train)
            
            # Evaluar el modelo
            y_pred = self.model.predict(X_test)
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
            
            self.last_trained = datetime.now()
            
            # Guardar el modelo
            self.save_model()
            
            logger.info(f"Modelo entrenado con éxito. Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error al entrenar el modelo: {str(e)}")
            return None
    
    def predict(self, df):
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            df (pandas.DataFrame): DataFrame con datos e indicadores.
            
        Returns:
            numpy.ndarray: Predicciones del modelo.
        """
        if self.model is None:
            logger.warning("No hay modelo entrenado para realizar predicciones.")
            return None
        
        try:
            # Verificar si el modelo está entrenado
            from sklearn.exceptions import NotFittedError
            
            # Preparar características
            X = self.prepare_features(df)
            
            try:
                # Intentar realizar predicciones
                predictions = self.model.predict(X)
                logger.info(f"Predicciones realizadas para {len(predictions)} muestras.")
                return predictions
            except (NotFittedError, ValueError) as fit_error:
                # Capturar específicamente errores de modelo no entrenado
                if "not fitted" in str(fit_error):
                    logger.warning("Modelo no entrenado. Iniciando entrenamiento automático...")
                    
                    # Generar etiquetas para entrenamiento
                    # Usar una estrategia simple basada en el movimiento del precio
                    target = np.zeros(len(df))
                    price_changes = df['close'].pct_change().shift(-1).fillna(0).values
                    
                    # Asignar 1 (compra) si el precio sube, -1 (venta) si baja
                    target[price_changes > 0.001] = 1  # Subida significativa
                    target[price_changes < -0.001] = -1  # Bajada significativa
                    
                    # Entrenar el modelo con los datos disponibles
                    self.model.fit(X, target)
                    self.last_trained = datetime.now()
                    
                    # Guardar el modelo entrenado
                    self.save_model()
                    
                    logger.info("Modelo entrenado automáticamente con éxito")
                    
                    # Realizar predicciones con el modelo recién entrenado
                    predictions = self.model.predict(X)
                    logger.info(f"Predicciones realizadas después del entrenamiento automático: {len(predictions)} muestras")
                    return predictions
                else:
                    # Otro tipo de error relacionado con el modelo
                    raise
            
        except Exception as e:
            logger.error(f"Error al realizar predicciones: {str(e)}")
            import traceback
            logger.debug(f"Traceback completo: {traceback.format_exc()}")
            return None
    
    def should_retrain(self, last_data_update):
        """
        Determina si el modelo debe ser reentrenado.
        
        Args:
            last_data_update (datetime): Timestamp de la última actualización de datos.
            
        Returns:
            bool: True si el modelo debe ser reentrenado, False en caso contrario.
        """
        # Si nunca se ha entrenado, entrenar
        if self.last_trained is None:
            return True
        
        # Si hay nuevos datos desde el último entrenamiento, entrenar
        if last_data_update > self.last_trained:
            # Calcular tiempo transcurrido desde el último entrenamiento en minutos
            elapsed_minutes = (datetime.now() - self.last_trained).total_seconds() / 60
            
            # Reentrenar cada 15 minutos
            if elapsed_minutes >= 20:
                logger.info(f"Han pasado {elapsed_minutes:.1f} minutos desde el último entrenamiento. Reentrenando...")
                return True
        
        return False

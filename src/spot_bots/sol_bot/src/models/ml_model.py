# src/models/ml_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import logging

logger = logging.getLogger(__name__)

class TradingModel:
    def __init__(self, input_shape):
        """
        Inicializa el modelo de trading.
        
        Args:
            input_shape (tuple): Forma de los datos de entrada (window_size, n_features)
        """
        self.model = self._build_model(input_shape)
        self.model_path = "models/trading_model.h5"
    
    def _build_model(self, input_shape):
        """
        Construye un modelo LSTM para predicción de precios.
        
        Args:
            input_shape (tuple): Forma de los datos de entrada
            
        Returns:
            tensorflow.keras.Model: Modelo compilado
        """
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info(f"Modelo construido con forma de entrada {input_shape}")
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """
        Entrena el modelo con los datos proporcionados.
        
        Args:
            X_train (numpy.ndarray): Datos de entrenamiento
            y_train (numpy.ndarray): Etiquetas de entrenamiento
            epochs (int): Número de épocas para entrenar
            batch_size (int): Tamaño del lote
            validation_split (float): Fracción de datos para validación
            
        Returns:
            tensorflow.keras.callbacks.History: Historial de entrenamiento
        """
        # Crear directorio para guardar el modelo si no existe
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Callback para guardar el mejor modelo
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            self.model_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Entrenar el modelo
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[checkpoint],
            verbose=1
        )
        
        logger.info(f"Modelo entrenado durante {epochs} épocas")
        return history
    
    def predict(self, X):
        """
        Realiza predicciones con el modelo.
        
        Args:
            X (numpy.ndarray): Datos para predecir
            
        Returns:
            numpy.ndarray: Predicciones (probabilidades)
        """
        predictions = self.model.predict(X)
        logger.info(f"Realizadas {len(predictions)} predicciones")
        return predictions
    
    def load_trained_model(self):
        """
        Carga un modelo entrenado previamente.
        
        Returns:
            bool: True si se cargó correctamente, False en caso contrario
        """
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            logger.info(f"Modelo cargado desde {self.model_path}")
            return True
        else:
            logger.warning(f"No se encontró un modelo guardado en {self.model_path}")
            return False

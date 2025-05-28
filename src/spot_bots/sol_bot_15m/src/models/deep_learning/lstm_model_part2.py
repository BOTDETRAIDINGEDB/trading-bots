#!/usr/bin/env python3
"""
Módulo que contiene la parte 2 de la implementación de DeepTimeSeriesModel.
Contiene métodos para construir diferentes arquitecturas de modelos.
"""

import os
import sys
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.layers import Bidirectional, TimeDistributed, Attention, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from typing import Dict, List, Tuple, Optional, Union, Any

# Configurar logging
logger = logging.getLogger(__name__)

def _build_bilstm_model(self) -> Model:
    """
    Construye un modelo LSTM bidireccional.
    
    Returns:
        Modelo Keras compilado
    """
    config = self.config.get("bilstm", {})
    units = config.get("units", [128, 64])
    dropout_rate = config.get("dropout", 0.3)
    recurrent_dropout = config.get("recurrent_dropout", 0.3)
    l1_reg = config.get("l1_reg", 0.0001)
    l2_reg = config.get("l2_reg", 0.0001)
    
    model = Sequential()
    
    # Primera capa BiLSTM
    model.add(Bidirectional(
        LSTM(units[0],
            return_sequences=len(units) > 1,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
        input_shape=(self.sequence_length, self.num_features)
    ))
    
    model.add(BatchNormalization())
    
    # Capas BiLSTM intermedias
    for i in range(1, len(units) - 1):
        model.add(Bidirectional(
            LSTM(units[i],
                return_sequences=True,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))
        ))
        model.add(BatchNormalization())
    
    # Última capa BiLSTM
    if len(units) > 1:
        model.add(Bidirectional(
            LSTM(units[-1],
                return_sequences=False,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout,
                kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))
        ))
        model.add(BatchNormalization())
    
    # Capa de salida
    model.add(Dense(self.num_classes, activation='softmax'))
    
    return model

def _build_attention_model(self) -> Model:
    """
    Construye un modelo LSTM con mecanismo de atención.
    
    Returns:
        Modelo Keras compilado
    """
    config = self.config.get("attention", {})
    lstm_units = config.get("lstm_units", 128)
    attention_units = config.get("attention_units", 64)
    dropout_rate = config.get("dropout", 0.3)
    l1_reg = config.get("l1_reg", 0.0001)
    l2_reg = config.get("l2_reg", 0.0001)
    
    # Entrada
    inputs = Input(shape=(self.sequence_length, self.num_features))
    
    # Capa LSTM que devuelve secuencias
    lstm_out = LSTM(lstm_units, 
                   return_sequences=True,
                   dropout=dropout_rate,
                   kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))(inputs)
    
    # Normalización
    lstm_out = BatchNormalization()(lstm_out)
    
    # Mecanismo de atención
    attention = Dense(attention_units, activation='tanh')(lstm_out)
    attention = Dense(1, activation='softmax')(attention)
    
    # Multiplicar LSTM por atención
    weighted = tf.multiply(lstm_out, attention)
    
    # Sumar a lo largo del eje temporal
    context = tf.reduce_sum(weighted, axis=1)
    
    # Capa de salida
    outputs = Dense(self.num_classes, activation='softmax')(context)
    
    # Crear modelo
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def _build_model(self) -> Model:
    """
    Construye el modelo según el tipo especificado.
    
    Returns:
        Modelo Keras compilado
    """
    # Intentar cargar modelo existente
    model_file = f"{self.model_path}_{self.model_type}.h5"
    if os.path.exists(model_file):
        try:
            model = load_model(model_file)
            logger.info(f"Modelo cargado desde {model_file}")
            return model
        except Exception as e:
            logger.error(f"Error al cargar modelo desde {model_file}: {str(e)}")
    
    # Construir nuevo modelo según el tipo
    if self.model_type == 'lstm':
        model = self._build_lstm_model()
    elif self.model_type == 'gru':
        model = self._build_gru_model()
    elif self.model_type == 'bilstm':
        model = self._build_bilstm_model()
    elif self.model_type == 'attention':
        model = self._build_attention_model()
    else:
        logger.warning(f"Tipo de modelo '{self.model_type}' no reconocido. Usando LSTM por defecto.")
        model = self._build_lstm_model()
    
    # Compilar modelo
    learning_rate = self.config.get(self.model_type, {}).get("learning_rate", 0.001)
    optimizer = Adam(learning_rate=learning_rate)
    
    if self.num_classes == 1:  # Regresión
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    else:  # Clasificación
        model.compile(optimizer=optimizer, 
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
    
    logger.info(f"Nuevo modelo {self.model_type} construido y compilado")
    model.summary(print_fn=logger.info)
    
    return model

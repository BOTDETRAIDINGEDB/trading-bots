import unittest
import pandas as pd
import numpy as np
import sys
import os
import logging
import pickle
from datetime import datetime, timedelta

# Configurar path para importar módulos del bot
bot_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(bot_dir, 'src', 'spot_bots', 'sol_bot_15m', 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from models.ml_model import MLModel

# Configurar logging para pruebas
logging.basicConfig(level=logging.ERROR)

class TestMLModel(unittest.TestCase):
    """Pruebas unitarias para el modelo de aprendizaje automático."""
    
    def setUp(self):
        """Configuración inicial para cada prueba."""
        # Usar un archivo temporal para el modelo de prueba
        self.test_model_path = os.path.join(os.path.dirname(__file__), 'test_model.pkl')
        self.ml_model = MLModel(model_path=self.test_model_path)
        
        # Crear datos de prueba
        dates = pd.date_range(start='2023-01-01', periods=500, freq='15min')
        self.test_df = pd.DataFrame({
            'open': np.random.normal(100, 5, 500),
            'high': np.random.normal(105, 5, 500),
            'low': np.random.normal(95, 5, 500),
            'close': np.random.normal(100, 5, 500),
            'volume': np.random.normal(1000, 200, 500),
            'rsi_14': np.random.normal(50, 15, 500),
            'sma_20': np.random.normal(100, 3, 500),
            'ema_50': np.random.normal(100, 2, 500),
            'macd': np.random.normal(0, 1, 500),
            'macd_signal': np.random.normal(0, 0.8, 500),
            'macd_hist': np.random.normal(0, 0.5, 500),
            'bb_upper': np.random.normal(110, 2, 500),
            'bb_middle': np.random.normal(100, 1, 500),
            'bb_lower': np.random.normal(90, 2, 500),
            'atr_14': np.random.normal(2, 0.5, 500),
            'relative_volume': np.random.normal(1, 0.3, 500),
            'signal': np.random.choice([-1, 0, 1], 500, p=[0.2, 0.6, 0.2])
        }, index=dates)
    
    def tearDown(self):
        """Limpieza después de cada prueba."""
        # Eliminar el archivo temporal del modelo si existe
        if os.path.exists(self.test_model_path):
            os.remove(self.test_model_path)
    
    def test_prepare_features(self):
        """Prueba la preparación de características para el modelo."""
        features = self.ml_model.prepare_features(self.test_df)
        
        # Verificar que las características se prepararon correctamente
        self.assertIsInstance(features, np.ndarray)
        # El número de filas debe ser igual al número de filas en el DataFrame
        self.assertEqual(features.shape[0], len(self.test_df))
        # Debe haber múltiples columnas de características
        self.assertGreater(features.shape[1], 1)
    
    def test_prepare_target(self):
        """Prueba la preparación de la variable objetivo para el modelo."""
        target = self.ml_model.prepare_target(self.test_df)
        
        # Verificar que la variable objetivo se preparó correctamente
        self.assertIsInstance(target, np.ndarray)
        # El número de elementos debe ser igual al número de filas en el DataFrame
        self.assertEqual(target.shape[0], len(self.test_df))
        # Los valores deben ser -1, 0 o 1
        self.assertTrue(np.all(np.isin(target, [-1, 0, 1])))
    
    def test_train_and_predict(self):
        """Prueba el entrenamiento y predicción del modelo."""
        # Entrenar el modelo
        metrics = self.ml_model.train(self.test_df)
        
        # Verificar que el entrenamiento devuelve métricas
        self.assertIsInstance(metrics, dict)
        self.assertTrue('accuracy' in metrics)
        self.assertTrue('precision' in metrics)
        self.assertTrue('recall' in metrics)
        self.assertTrue('f1' in metrics)
        
        # Verificar que el modelo se guardó correctamente
        self.assertTrue(os.path.exists(self.test_model_path))
        
        # Hacer predicciones
        predictions = self.ml_model.predict(self.test_df)
        
        # Verificar que las predicciones son correctas
        self.assertIsInstance(predictions, np.ndarray)
        self.assertEqual(predictions.shape[0], len(self.test_df))
        self.assertTrue(np.all(np.isin(predictions, [-1, 0, 1])))
    
    def test_save_and_load_model(self):
        """Prueba guardar y cargar el modelo."""
        # Entrenar y guardar el modelo
        self.ml_model.train(self.test_df)
        self.ml_model.save_model()
        
        # Crear una nueva instancia y cargar el modelo
        new_model = MLModel(model_path=self.test_model_path)
        
        # Verificar que el modelo se cargó correctamente
        self.assertIsNotNone(new_model.model)
        self.assertIsNotNone(new_model.scaler)
        self.assertIsNotNone(new_model.last_trained)
        
        # Verificar que ambos modelos hacen predicciones similares
        pred1 = self.ml_model.predict(self.test_df)
        pred2 = new_model.predict(self.test_df)
        
        # Las predicciones deberían ser idénticas
        np.testing.assert_array_equal(pred1, pred2)
    
    def test_should_retrain(self):
        """Prueba la lógica para determinar si el modelo debe ser reentrenado."""
        # Establecer una fecha de último entrenamiento
        self.ml_model.last_trained = datetime.now() - timedelta(hours=1)
        
        # No debería reentrenar si los datos son más antiguos que el último entrenamiento
        old_data_update = datetime.now() - timedelta(hours=2)
        self.assertFalse(self.ml_model.should_retrain(old_data_update))
        
        # Debería reentrenar si los datos son más recientes que el último entrenamiento
        new_data_update = datetime.now()
        self.assertTrue(self.ml_model.should_retrain(new_data_update))
        
        # Debería reentrenar si no hay fecha de último entrenamiento
        self.ml_model.last_trained = None
        self.assertTrue(self.ml_model.should_retrain(old_data_update))

if __name__ == '__main__':
    unittest.main()

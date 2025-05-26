import unittest
import pandas as pd
import numpy as np
import sys
import os
import logging

# Configurar path para importar módulos del bot
bot_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_dir = os.path.join(bot_dir, 'src', 'spot_bots', 'sol_bot_15m', 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data.processor import DataProcessor

# Configurar logging para pruebas
logging.basicConfig(level=logging.ERROR)

class TestDataProcessor(unittest.TestCase):
    """Pruebas unitarias para el procesador de datos."""
    
    def setUp(self):
        """Configuración inicial para cada prueba."""
        self.processor = DataProcessor()
        
        # Crear datos de prueba simulando velas de Binance
        self.test_klines = [
            [1622505600000, "35000.0", "36000.0", "34800.0", "35500.0", "100.5", 
             1622509199999, "3517500.0", 150, "60.3", "2110500.0", "0"],
            [1622509200000, "35500.0", "36200.0", "35200.0", "35800.0", "120.7", 
             1622512799999, "4321000.0", 180, "72.4", "2592000.0", "0"],
            [1622512800000, "35800.0", "37000.0", "35600.0", "36500.0", "150.2", 
             1622516399999, "5482500.0", 220, "90.1", "3286500.0", "0"]
        ]
    
    def test_klines_to_dataframe(self):
        """Prueba la conversión de velas a DataFrame."""
        df = self.processor.klines_to_dataframe(self.test_klines)
        
        # Verificar que el DataFrame se creó correctamente
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertTrue('open' in df.columns)
        self.assertTrue('high' in df.columns)
        self.assertTrue('low' in df.columns)
        self.assertTrue('close' in df.columns)
        self.assertTrue('volume' in df.columns)
        
        # Verificar la conversión de tipos
        self.assertTrue(pd.api.types.is_numeric_dtype(df['open']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['high']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['low']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['close']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['volume']))
    
    def test_calculate_indicators(self):
        """Prueba el cálculo de indicadores técnicos."""
        # Crear un DataFrame de prueba más grande para los indicadores
        dates = pd.date_range(start='2023-01-01', periods=100, freq='15min')
        df = pd.DataFrame({
            'open': np.random.normal(100, 5, 100),
            'high': np.random.normal(105, 5, 100),
            'low': np.random.normal(95, 5, 100),
            'close': np.random.normal(100, 5, 100),
            'volume': np.random.normal(1000, 200, 100)
        }, index=dates)
        
        # Calcular indicadores
        df_with_indicators = self.processor.calculate_indicators(df)
        
        # Verificar que se añadieron los indicadores
        self.assertTrue('rsi_14' in df_with_indicators.columns)
        self.assertTrue('sma_20' in df_with_indicators.columns)
        self.assertTrue('ema_50' in df_with_indicators.columns)
        self.assertTrue('macd' in df_with_indicators.columns)
        self.assertTrue('macd_signal' in df_with_indicators.columns)
        self.assertTrue('macd_hist' in df_with_indicators.columns)
    
    def test_generate_signals(self):
        """Prueba la generación de señales de trading."""
        # Crear un DataFrame con indicadores ya calculados
        dates = pd.date_range(start='2023-01-01', periods=300, freq='15min')
        
        # Crear datos con patrones que deberían generar señales
        df = pd.DataFrame({
            'open': np.random.normal(100, 5, 300),
            'high': np.random.normal(105, 5, 300),
            'low': np.random.normal(95, 5, 300),
            'close': np.random.normal(100, 5, 300),
            'volume': np.random.normal(1000, 200, 300),
            'rsi_14': np.random.normal(50, 15, 300),
            'sma_20': np.random.normal(100, 3, 300),
            'ema_50': np.random.normal(100, 2, 300),
            'macd': np.random.normal(0, 1, 300),
            'macd_signal': np.random.normal(0, 0.8, 300),
            'macd_hist': np.random.normal(0, 0.5, 300),
            'bb_upper': np.random.normal(110, 2, 300),
            'bb_middle': np.random.normal(100, 1, 300),
            'bb_lower': np.random.normal(90, 2, 300),
            'atr_14': np.random.normal(2, 0.5, 300),
            'relative_volume': np.random.normal(1, 0.3, 300)
        }, index=dates)
        
        # Crear cruces de medias móviles para generar señales
        df['sma_cross'] = np.where(df.index.dayofweek < 3, 1, -1)
        df['sma_cross_change'] = df['sma_cross'].diff().fillna(0)
        df['macd_cross'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        df['macd_cross_change'] = df['macd_cross'].diff().fillna(0)
        
        # Generar señales
        df_with_signals = self.processor.generate_signals(df)
        
        # Verificar que se añadieron las señales
        self.assertTrue('signal' in df_with_signals.columns)
        
        # Verificar que hay señales generadas (excepto en los primeros 200 períodos)
        signals_count = (df_with_signals.iloc[200:]['signal'] != 0).sum()
        self.assertGreater(signals_count, 0)
        
        # Verificar que los primeros 200 períodos tienen señal 0
        self.assertEqual((df_with_signals.iloc[:200]['signal'] != 0).sum(), 0)

if __name__ == '__main__':
    unittest.main()

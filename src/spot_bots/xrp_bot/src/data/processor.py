# src/data/processor.py
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        pass
    
    def klines_to_dataframe(self, klines):
        """
        Convierte datos de velas (klines) de Binance a un DataFrame de pandas.
        
        Args:
            klines (list): Lista de velas de Binance
            
        Returns:
            pandas.DataFrame: DataFrame con los datos procesados
        """
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        df = pd.DataFrame(klines, columns=columns)
        
        # Convertir tipos de datos
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'taker_buy_base_asset_volume', 
                          'taker_buy_quote_asset_volume']
        
        df[numeric_columns] = df[numeric_columns].astype(float)
        
        # Convertir timestamps a datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Establecer timestamp como índice
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"DataFrame creado con {len(df)} filas")
        return df
    
    def add_indicators(self, df):
        """
        Añade indicadores técnicos al DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame con datos OHLCV
            
        Returns:
            pandas.DataFrame: DataFrame con indicadores añadidos
        """
        # Copiar el DataFrame para no modificar el original
        df_with_indicators = df.copy()
        
        try:
            # Calcular medias móviles (optimizadas para velas de 30m)
            df_with_indicators['sma_20'] = df_with_indicators['close'].rolling(window=40).mean()  # 20 horas (40 períodos de 30m)
            df_with_indicators['sma_50'] = df_with_indicators['close'].rolling(window=100).mean()  # ~2 días (100 períodos de 30m)
            df_with_indicators['sma_200'] = df_with_indicators['close'].rolling(window=400).mean()  # ~8 días (400 períodos de 30m)
            
            # Calcular RSI (Relative Strength Index)
            delta = df_with_indicators['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=28).mean()  # 14 horas (28 períodos de 30m)
            avg_loss = loss.rolling(window=28).mean()
            
            rs = avg_gain / avg_loss
            df_with_indicators['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Calcular MACD (Moving Average Convergence Divergence)
            ema_12 = df_with_indicators['close'].ewm(span=24, adjust=False).mean()  # 12 horas (24 períodos de 30m)
            ema_26 = df_with_indicators['close'].ewm(span=52, adjust=False).mean()  # 26 horas (52 períodos de 30m)
            df_with_indicators['macd'] = ema_12 - ema_26
            df_with_indicators['macd_signal'] = df_with_indicators['macd'].ewm(span=18, adjust=False).mean()  # 9 horas (18 períodos de 30m)
            df_with_indicators['macd_histogram'] = df_with_indicators['macd'] - df_with_indicators['macd_signal']
            
            # Calcular Bandas de Bollinger
            df_with_indicators['bollinger_mid'] = df_with_indicators['close'].rolling(window=40).mean()  # 20 horas (40 períodos de 30m)
            df_with_indicators['bollinger_std'] = df_with_indicators['close'].rolling(window=40).std()
            df_with_indicators['bollinger_upper'] = df_with_indicators['bollinger_mid'] + 2 * df_with_indicators['bollinger_std']
            df_with_indicators['bollinger_lower'] = df_with_indicators['bollinger_mid'] - 2 * df_with_indicators['bollinger_std']
            
            # Calcular ATR (Average True Range)
            high_low = df_with_indicators['high'] - df_with_indicators['low']
            high_close = abs(df_with_indicators['high'] - df_with_indicators['close'].shift())
            low_close = abs(df_with_indicators['low'] - df_with_indicators['close'].shift())
            
            tr = pd.DataFrame(np.maximum(high_low, np.maximum(high_close, low_close)))
            tr.columns = ['TR']
            
            atr = tr.rolling(window=28).mean()  # 14 horas (28 períodos de 30m)
            df_with_indicators['atr_14'] = atr
            
            # Calcular ADX (Average Directional Index) completo
            plus_dm = df_with_indicators['high'].diff()
            minus_dm = df_with_indicators['low'].diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            minus_dm = abs(minus_dm)
            
            # Condición para +DM: si high(t) - high(t-1) > low(t-1) - low(t)
            condition = (df_with_indicators['high'].diff() > abs(df_with_indicators['low'].diff()))
            plus_dm[~condition] = 0
            
            # Condición para -DM: si low(t-1) - low(t) > high(t) - high(t-1)
            condition = (abs(df_with_indicators['low'].diff()) > df_with_indicators['high'].diff())
            minus_dm[~condition] = 0
            
            # Calcular +DI y -DI
            plus_di = 100 * (plus_dm.rolling(window=28).mean() / atr['TR'])  # 14 horas (28 períodos de 30m)
            minus_di = 100 * (minus_dm.rolling(window=28).mean() / atr['TR'])
            
            # Calcular DX y ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df_with_indicators['adx_14'] = dx.rolling(window=28).mean()  # 14 horas (28 períodos de 30m)
            df_with_indicators['plus_di_14'] = plus_di
            df_with_indicators['minus_di_14'] = minus_di
            
            # Calcular Stochastic Oscillator
            low_14 = df_with_indicators['low'].rolling(window=28).min()  # 14 horas (28 períodos de 30m)
            high_14 = df_with_indicators['high'].rolling(window=28).max()
            df_with_indicators['stoch_k'] = 100 * ((df_with_indicators['close'] - low_14) / (high_14 - low_14))
            df_with_indicators['stoch_d'] = df_with_indicators['stoch_k'].rolling(window=6).mean()  # 3 horas (6 períodos de 30m)
            
            # Calcular On-Balance Volume (OBV)
            obv = [0]
            for i in range(1, len(df_with_indicators)):
                if df_with_indicators['close'].iloc[i] > df_with_indicators['close'].iloc[i-1]:
                    obv.append(obv[-1] + df_with_indicators['volume'].iloc[i])
                elif df_with_indicators['close'].iloc[i] < df_with_indicators['close'].iloc[i-1]:
                    obv.append(obv[-1] - df_with_indicators['volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            df_with_indicators['obv'] = obv
            
            # Calcular Commodity Channel Index (CCI)
            typical_price = (df_with_indicators['high'] + df_with_indicators['low'] + df_with_indicators['close']) / 3
            mean_tp = typical_price.rolling(window=40).mean()  # 20 horas (40 períodos de 30m)
            mean_deviation = abs(typical_price - mean_tp).rolling(window=40).mean()
            df_with_indicators['cci_20'] = (typical_price - mean_tp) / (0.015 * mean_deviation)
            
            # Calcular Parabolic SAR (simplificado)
            # Para XRP, el Parabolic SAR puede ser útil para identificar cambios de tendencia
            df_with_indicators['parabolic_sar'] = df_with_indicators['close'].shift(1)
            
            # Calcular Williams %R
            # Este indicador es útil para identificar condiciones de sobrecompra/sobreventa
            high_14 = df_with_indicators['high'].rolling(window=28).max()  # 14 horas (28 períodos de 30m)
            low_14 = df_with_indicators['low'].rolling(window=28).min()
            df_with_indicators['williams_r'] = -100 * (high_14 - df_with_indicators['close']) / (high_14 - low_14)
            
            # Calcular Awesome Oscillator (AO)
            # Útil para identificar momentum
            median_price = (df_with_indicators['high'] + df_with_indicators['low']) / 2
            ao_fast = median_price.rolling(window=10).mean()  # 5 horas (10 períodos de 30m)
            ao_slow = median_price.rolling(window=68).mean()  # 34 horas (68 períodos de 30m)
            df_with_indicators['awesome_oscillator'] = ao_fast - ao_slow
            
            # Calcular Money Flow Index (MFI)
            # Útil para identificar divergencias de precio y volumen
            typical_price = (df_with_indicators['high'] + df_with_indicators['low'] + df_with_indicators['close']) / 3
            money_flow = typical_price * df_with_indicators['volume']
            
            positive_flow = [0]
            negative_flow = [0]
            
            for i in range(1, len(df_with_indicators)):
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    positive_flow.append(money_flow.iloc[i])
                    negative_flow.append(0)
                elif typical_price.iloc[i] < typical_price.iloc[i-1]:
                    positive_flow.append(0)
                    negative_flow.append(money_flow.iloc[i])
                else:
                    positive_flow.append(0)
                    negative_flow.append(0)
            
            positive_mf = pd.Series(positive_flow, index=df_with_indicators.index)
            negative_mf = pd.Series(negative_flow, index=df_with_indicators.index)
            
            positive_mf_sum = positive_mf.rolling(window=28).sum()  # 14 horas (28 períodos de 30m)
            negative_mf_sum = negative_mf.rolling(window=28).sum()
            
            mfi = 100 - (100 / (1 + positive_mf_sum / negative_mf_sum))
            df_with_indicators['mfi_14'] = mfi
            
            # Rellenar NaN en lugar de eliminar filas
            df_with_indicators.fillna(method='bfill', inplace=True)
            df_with_indicators.fillna(method='ffill', inplace=True)
            # Si aún quedan NaN después de rellenar hacia adelante y atrás, rellenar con 0
            df_with_indicators.fillna(0, inplace=True)
            
            logger.info(f"Indicadores técnicos añadidos al DataFrame, filas restantes: {len(df_with_indicators)}")
        except Exception as e:
            logger.error(f"Error al añadir indicadores: {e}")
        
        return df_with_indicators
    
    def prepare_for_model(self, df, window_size=10):
        """
        Prepara los datos para el modelo de aprendizaje automático.
        
        Args:
            df (pandas.DataFrame): DataFrame con datos e indicadores
            window_size (int): Tamaño de la ventana para características secuenciales
            
        Returns:
            tuple: (X, y) donde X son las características y y son las etiquetas
        """
        logger.info(f"Preparando datos para modelo, DataFrame de entrada: {len(df)} filas")
        
        if len(df) <= window_size:
            logger.warning(f"No hay suficientes datos para crear secuencias (necesita > {window_size} filas)")
            return np.array([]), np.array([])
        
        # Crear etiquetas: 1 si el precio sube en el siguiente período, 0 si baja
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Seleccionar características (solo las originales)
        features = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'sma_50', 'sma_200', 'rsi_14',
            'macd', 'macd_signal', 'macd_histogram',
            'bollinger_upper', 'bollinger_mid', 'bollinger_lower',
            'atr_14'
        ]
        
        # Verificar que todas las características existen en el DataFrame
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.error(f"Faltan características en el DataFrame: {missing_features}")
            # Si faltan características, usar solo las que están disponibles
            features = [f for f in features if f in df.columns]
            logger.info(f"Usando características disponibles: {features}")
        
        try:
            # Normalizar características
            df_normalized = df[features].copy()
            for column in df_normalized.columns:
                mean = df_normalized[column].mean()
                std = df_normalized[column].std()
                if std == 0:
                    logger.warning(f"Desviación estándar cero para {column}, no se normalizará")
                    continue
                df_normalized[column] = (df_normalized[column] - mean) / std
            
            # Crear secuencias
            X = []
            y = []
            
            logger.info(f"Creando secuencias con window_size={window_size}, datos disponibles: {len(df_normalized)}")
            
            for i in range(len(df_normalized) - window_size):
                X.append(df_normalized.iloc[i:i+window_size].values)
                y.append(df['target'].iloc[i+window_size])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Datos preparados para modelo: X shape {X.shape}, y shape {y.shape}")
            return X, y
        except Exception as e:
            logger.error(f"Error al preparar datos para modelo: {e}")
            return np.array([]), np.array([])

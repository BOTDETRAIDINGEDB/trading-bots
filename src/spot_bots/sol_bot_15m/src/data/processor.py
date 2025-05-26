# processor.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataProcessor:
    """Procesa datos de mercado y calcula indicadores técnicos."""
    
    def __init__(self):
        """Inicializa el procesador de datos."""
        logger.info("Procesador de datos inicializado.")
    
    def klines_to_dataframe(self, klines):
        """
        Convierte las velas de Binance a un DataFrame de pandas.
        
        Args:
            klines (list): Lista de velas de Binance.
            
        Returns:
            pandas.DataFrame: DataFrame con los datos procesados.
        """
        if not klines:
            logger.warning("No hay datos de velas para procesar.")
            return pd.DataFrame()
        
        # Crear DataFrame con las columnas adecuadas
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convertir tipos de datos
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'taker_buy_base_asset_volume', 
                          'taker_buy_quote_asset_volume']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])
        
        # Convertir timestamps a datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Establecer timestamp como índice
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Datos convertidos a DataFrame con {len(df)} filas.")
        return df
    
    def calculate_indicators(self, df):
        """
        Calcula indicadores técnicos para el DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame con datos de velas.
            
        Returns:
            pandas.DataFrame: DataFrame con indicadores añadidos.
        """
        if df.empty:
            logger.warning("DataFrame vacío, no se pueden calcular indicadores.")
            return df
        
        # Hacer una copia para evitar advertencias de SettingWithCopyWarning
        df = df.copy()
        
        # Calcular medias móviles
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Calcular bandas de Bollinger (20 períodos, 2 desviaciones estándar)
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Calcular RSI (14 períodos)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # Cálculo para periodos después de los primeros 14
        for i in range(14, len(df)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * 13 + gain.iloc[i]) / 14
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * 13 + loss.iloc[i]) / 14
        
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Calcular MACD
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Calcular ATR (14 períodos)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = true_range.rolling(window=14).mean()
        
        # Calcular variación porcentual
        df['pct_change'] = df['close'].pct_change() * 100
        
        # Calcular volumen relativo (comparado con la media de 20 períodos)
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['relative_volume'] = df['volume'] / df['volume_sma_20']
        
        logger.info("Indicadores técnicos calculados correctamente.")
        return df
    
    def generate_signals(self, df):
        """
        Genera señales de trading basadas en indicadores técnicos.
        
        Args:
            df (pandas.DataFrame): DataFrame con indicadores técnicos.
            
        Returns:
            pandas.DataFrame: DataFrame con señales añadidas.
        """
        if df.empty:
            logger.warning("DataFrame vacío, no se pueden generar señales.")
            return df
        
        # Hacer una copia para evitar advertencias
        df = df.copy()
        
        # Inicializar columna de señales (0: mantener, 1: comprar, -1: vender)
        df['signal'] = 0
        
        # Señal basada en cruce de medias móviles
        df['sma_cross'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        df['sma_cross_change'] = df['sma_cross'].diff()
        
        # Señal basada en RSI
        df['rsi_signal'] = np.where(df['rsi_14'] < 30, 1, np.where(df['rsi_14'] > 70, -1, 0))
        
        # Señal basada en MACD
        df['macd_cross'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        df['macd_cross_change'] = df['macd_cross'].diff()
        
        # Señal basada en Bandas de Bollinger
        df['bb_signal'] = np.where(df['close'] < df['bb_lower'], 1, 
                                 np.where(df['close'] > df['bb_upper'], -1, 0))
        
        # Combinar señales (estrategia personalizada para SOL)
        # Comprar cuando: 
        # 1. Hay cruce alcista de SMA Y
        # 2. RSI está saliendo de sobreventa O MACD cruza hacia arriba
        buy_condition = (
            (df['sma_cross_change'] == 2) | 
            ((df['rsi_signal'] == 1) & (df['rsi_14'].shift(1) < 30) & (df['rsi_14'] > 30)) |
            ((df['macd_cross_change'] == 2) & (df['relative_volume'] > 1.2))
        )
        
        # Vender cuando:
        # 1. Hay cruce bajista de SMA Y
        # 2. RSI está en sobreventa O MACD cruza hacia abajo
        sell_condition = (
            (df['sma_cross_change'] == -2) | 
            ((df['rsi_signal'] == -1) & (df['rsi_14'].shift(1) > 70) & (df['rsi_14'] < 70)) |
            ((df['macd_cross_change'] == -2) & (df['relative_volume'] > 1.2))
        )
        
        # Aplicar condiciones
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        # Eliminar señales en los primeros 200 períodos (insuficientes datos para indicadores)
        try:
            signal_col = df.columns.get_loc('signal')
            df.iloc[:200, signal_col] = 0
            logger.debug(f"Señales en los primeros 200 períodos establecidas a 0")
        except (KeyError, IndexError) as e:
            logger.error(f"Error al establecer señales iniciales a 0: {str(e)}")
            # Manejo de recuperación: si no existe la columna 'signal', la creamos
            if 'signal' not in df.columns:
                logger.warning("Columna 'signal' no encontrada. Creándola...")
                df['signal'] = 0
        
        # Contar señales generadas
        buy_signals = (df['signal'] == 1).sum()
        sell_signals = (df['signal'] == -1).sum()
        logger.info(f"Señales generadas: {buy_signals} de compra, {sell_signals} de venta.")
        
        return df

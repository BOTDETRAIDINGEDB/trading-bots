#!/usr/bin/env python3
"""
Analizador de mercado para el bot SOL
Proporciona funciones para analizar condiciones de mercado y velas de 15 minutos
"""

import logging
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    """Clase para analizar condiciones de mercado."""
    
    def __init__(self, symbol='SOLUSDT'):
        """
        Inicializa el analizador de mercado.
        
        Args:
            symbol (str): Símbolo de trading (ej. 'SOLUSDT').
        """
        self.symbol = symbol
        logger.info(f"Analizador de mercado inicializado para {symbol}")
    
    def get_klines(self, interval='15m', limit=100):
        """
        Obtiene velas (klines) de Binance.
        
        Args:
            interval (str): Intervalo de tiempo ('1m', '5m', '15m', '1h', etc.).
            limit (int): Número de velas a obtener.
            
        Returns:
            pd.DataFrame: DataFrame con las velas, o None si hubo un error.
        """
        try:
            # Obtener datos de mercado de Binance
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': self.symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params)
            if response.status_code != 200:
                logger.error(f"Error al obtener velas: {response.status_code} - {response.text}")
                return None
            
            # Procesar datos
            klines = response.json()
            
            # Convertir a DataFrame
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convertir tipos de datos
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            return df
        except Exception as e:
            logger.error(f"Error al obtener velas: {str(e)}")
            return None
    
    def calculate_volatility(self, df=None, period=14):
        """
        Calcula la volatilidad del mercado.
        
        Args:
            df (pd.DataFrame, optional): DataFrame con velas. Si es None, se obtienen.
            period (int): Período para calcular ATR.
            
        Returns:
            float: Volatilidad normalizada (0-1).
        """
        if df is None:
            df = self.get_klines(interval='15m', limit=100)
        
        if df is None or len(df) < period:
            return 0.5  # Valor por defecto
        
        try:
            # Calcular volatilidad como ATR normalizado
            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['close'].shift(1))
            df['tr3'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['tr'].rolling(period).mean()
            
            # Normalizar ATR como porcentaje del precio
            last_price = df['close'].iloc[-1]
            last_atr = df['atr'].iloc[-1]
            normalized_atr = last_atr / last_price
            
            # Convertir a escala 0-1 (0.5% ATR = 0.5, 2% ATR = 1)
            volatility = min(1, normalized_atr * 50)
            
            logger.info(f"Volatilidad calculada: {volatility:.4f}")
            return volatility
        except Exception as e:
            logger.error(f"Error al calcular volatilidad: {str(e)}")
            return 0.5  # Valor por defecto
    
    def calculate_trend_strength(self, df=None):
        """
        Calcula la fuerza de la tendencia.
        
        Args:
            df (pd.DataFrame, optional): DataFrame con velas. Si es None, se obtienen.
            
        Returns:
            float: Fuerza de tendencia (-1 a 1).
        """
        if df is None:
            df = self.get_klines(interval='15m', limit=100)
        
        if df is None or len(df) < 20:
            return 0  # Valor por defecto
        
        try:
            # Calcular EMA de 8 y 21 períodos
            df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
            df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
            
            # Calcular diferencia porcentual entre EMAs
            ema_diff = (df['ema8'].iloc[-1] - df['ema21'].iloc[-1]) / df['ema21'].iloc[-1]
            
            # Normalizar a escala -1 a 1
            trend_strength = np.tanh(ema_diff * 100)
            
            # Calcular dirección de la tendencia en las últimas 10 velas
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-10]) / df['close'].iloc[-10]
            trend_direction = np.sign(price_change)
            
            # Ajustar fuerza de tendencia con dirección
            adjusted_trend = trend_strength * trend_direction
            
            logger.info(f"Fuerza de tendencia calculada: {adjusted_trend:.4f}")
            return adjusted_trend
        except Exception as e:
            logger.error(f"Error al calcular fuerza de tendencia: {str(e)}")
            return 0  # Valor por defecto
    
    def analyze_market_conditions(self):
        """
        Analiza las condiciones actuales del mercado.
        
        Returns:
            dict: Condiciones del mercado.
        """
        # Obtener velas
        df = self.get_klines(interval='15m', limit=100)
        
        if df is None:
            return {
                'volatility': 0.5,
                'trend_strength': 0,
                'current_price': 0,
                'volume_change': 0,
                'rsi': 50,
                'timestamp': datetime.now().isoformat()
            }
        
        # Calcular métricas
        volatility = self.calculate_volatility(df)
        trend_strength = self.calculate_trend_strength(df)
        current_price = df['close'].iloc[-1]
        
        # Calcular cambio de volumen
        avg_volume_prev = df['volume'].iloc[-10:-5].mean()
        avg_volume_recent = df['volume'].iloc[-5:].mean()
        volume_change = (avg_volume_recent / avg_volume_prev) - 1 if avg_volume_prev > 0 else 0
        
        # Calcular RSI
        try:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
        except:
            rsi = 50  # Valor por defecto
        
        return {
            'volatility': volatility,
            'trend_strength': trend_strength,
            'current_price': current_price,
            'volume_change': volume_change,
            'rsi': rsi,
            'timestamp': datetime.now().isoformat()
        }
    
    def calculate_optimal_take_profit(self, base_tp=0.04, volatility=None, trend_strength=None):
        """
        Calcula el take profit óptimo basado en condiciones de mercado.
        
        Args:
            base_tp (float): Take profit base (0.04 = 4%).
            volatility (float, optional): Volatilidad del mercado (0-1).
            trend_strength (float, optional): Fuerza de tendencia (-1 a 1).
            
        Returns:
            float: Take profit óptimo.
        """
        # Si no se proporcionan, calcular
        if volatility is None or trend_strength is None:
            market_conditions = self.analyze_market_conditions()
            volatility = market_conditions['volatility']
            trend_strength = market_conditions['trend_strength']
        
        # Ajustar take profit según volatilidad (mayor volatilidad = mayor TP)
        volatility_adjustment = (volatility - 0.5) * 0.03
        
        # Ajustar take profit según tendencia (tendencia fuerte = mayor TP en dirección de la tendencia)
        trend_adjustment = trend_strength * 0.02
        
        # Calcular take profit adaptativo
        adaptive_tp = base_tp + volatility_adjustment + trend_adjustment
        
        # Limitar a rango razonable
        adaptive_tp = max(0.02, min(0.15, adaptive_tp))
        
        logger.info(f"Take profit óptimo calculado: {adaptive_tp:.4f} (base: {base_tp}, vol_adj: {volatility_adjustment:.4f}, trend_adj: {trend_adjustment:.4f})")
        
        return adaptive_tp

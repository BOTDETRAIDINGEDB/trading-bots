#!/usr/bin/env python3
"""
Módulo de carga y gestión de datos históricos multi-timeframe.
Permite obtener y sincronizar datos de diferentes intervalos temporales.
"""

import os
import sys
import logging
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
import time
import json
import pytz
from typing import Dict, List, Tuple, Optional, Union, Any

# Ajustar el PYTHONPATH para encontrar los módulos correctamente
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Importar utilidades propias
try:
    from utils.cloud_utils import ensure_directory_exists, safe_json_dump, safe_json_load, retry_with_backoff
except ImportError:
    # Definir funciones básicas si no se pueden importar las utilidades
    def ensure_directory_exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)
        return True
        
    def safe_json_dump(data, file_path, indent=4):
        try:
            directory = os.path.dirname(file_path)
            os.makedirs(directory, exist_ok=True)
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=indent)
            return True
        except Exception as e:
            print(f"Error al guardar JSON: {str(e)}")
            return False
            
    def safe_json_load(file_path, default=None):
        try:
            if not os.path.exists(file_path):
                return default
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error al cargar JSON: {str(e)}")
            return default
    
    # Decorador simple para reintentos
    def retry_with_backoff(max_attempts=3, delay=1):
        def decorator(func):
            def wrapper(*args, **kwargs):
                attempts = 0
                while attempts < max_attempts:
                    try:
                        return func(*args, **kwargs)
                    except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
                        attempts += 1
                        if attempts < max_attempts:
                            sleep_time = delay * (2 ** (attempts - 1))  # Backoff exponencial
                            print(f"Reintento {attempts}/{max_attempts} después de {sleep_time}s. Error: {str(e)}")
                            time.sleep(sleep_time)
                        else:
                            raise
            return wrapper
        return decorator

# Configurar logging
logger = logging.getLogger(__name__)

class MultiTimeframeDataLoader:
    """
    Cargador de datos históricos para múltiples timeframes.
    Gestiona la obtención, almacenamiento y sincronización de datos.
    """
    
    def __init__(self, 
                 symbol: str = 'SOL/USDT', 
                 timeframes: List[str] = ['5m', '15m', '1h', '4h'],
                 data_dir: str = 'data',
                 max_retries: int = 3,
                 retry_delay: int = 5,
                 lookback_days: int = 365,
                 exchange_id: str = 'binance',
                 use_testnet: bool = False,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None):
        """
        Inicializa el cargador de datos multi-timeframe.
        
        Args:
            symbol: Par de trading
            timeframes: Lista de intervalos temporales a utilizar
            data_dir: Directorio para almacenar datos
            max_retries: Número máximo de reintentos para peticiones
            retry_delay: Tiempo de espera entre reintentos (segundos)
            lookback_days: Días de histórico a obtener
            exchange_id: ID del exchange (binance, kucoin, etc.)
            use_testnet: Si es True, usa testnet en lugar de mainnet
            api_key: Clave API opcional para mayor límite de peticiones
            api_secret: Secreto API opcional para mayor límite de peticiones
        """
        self.symbol = symbol
        self.timeframes = timeframes
        self.data_dir = os.path.abspath(data_dir)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.lookback_days = lookback_days
        self.exchange_id = exchange_id
        self.use_testnet = use_testnet
        
        # Crear directorio de datos si no existe
        ensure_directory_exists(self.data_dir)
        
        # Verificar si hay credenciales en variables de entorno
        if api_key is None and 'EXCHANGE_API_KEY' in os.environ:
            api_key = os.environ.get('EXCHANGE_API_KEY')
            logger.info("Usando API key desde variables de entorno")
            
        if api_secret is None and 'EXCHANGE_API_SECRET' in os.environ:
            api_secret = os.environ.get('EXCHANGE_API_SECRET')
            logger.info("Usando API secret desde variables de entorno")
        
        # Inicializar exchange con manejo de errores
        try:
            # Verificar si el exchange es soportado
            if not hasattr(ccxt, exchange_id):
                available_exchanges = ', '.join(ccxt.exchanges)
                logger.warning(f"Exchange {exchange_id} no soportado. Exchanges disponibles: {available_exchanges}")
                logger.info(f"Usando binance como exchange por defecto")
                exchange_id = 'binance'
            
            # Configuración del exchange
            exchange_class = getattr(ccxt, exchange_id)
            exchange_config = {
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot'
                }
            }
            
            # Añadir credenciales si están disponibles
            if api_key and api_secret:
                exchange_config['apiKey'] = api_key
                exchange_config['secret'] = api_secret
                logger.info(f"Credenciales configuradas para {exchange_id}")
            
            # Configurar testnet si es necesario
            if use_testnet:
                if exchange_id == 'binance':
                    exchange_config['options']['defaultType'] = 'spot'
                    exchange_config['urls'] = {
                        'api': {
                            'public': 'https://testnet.binance.vision/api/v3',
                            'private': 'https://testnet.binance.vision/api/v3',
                        }
                    }
                    logger.info("Usando testnet de Binance")
                else:
                    logger.warning(f"Testnet no configurado para {exchange_id}, usando mainnet")
            
            # Crear instancia del exchange
            self.exchange = exchange_class(exchange_config)
            
            # Verificar conectividad
            self.exchange.load_markets()
            logger.info(f"Exchange {exchange_id} inicializado correctamente")
            
            # Verificar si el símbolo está disponible
            if self.symbol not in self.exchange.symbols:
                available_symbols = ', '.join(self.exchange.symbols[:10]) + '...'
                logger.warning(f"Símbolo {self.symbol} no disponible en {exchange_id}. Ejemplos disponibles: {available_symbols}")
                # No lanzar error, permitirá cargar datos locales si existen
            
        except Exception as e:
            logger.error(f"Error al inicializar exchange {exchange_id}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            logger.warning("Continuando sin exchange inicializado. Solo se podrán cargar datos locales.")
            self.exchange = None
        
        # Cache para datos
        self.data_cache = {}
        
        # Crear archivo de metadatos
        self._save_metadata()
        
        logger.info(f"Cargador de datos inicializado para {symbol} con timeframes {timeframes}")
        
    def _save_metadata(self):
        """Guarda metadatos del cargador de datos."""
        metadata = {
            'symbol': self.symbol,
            'timeframes': self.timeframes,
            'lookback_days': self.lookback_days,
            'exchange_id': self.exchange_id,
            'created_at': datetime.now().isoformat(),
            'timezone': str(datetime.now().astimezone().tzinfo)
        }
        
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        safe_json_dump(metadata, metadata_path)
        
    
    def _get_filename(self, timeframe: str) -> str:
        """
        Genera nombre de archivo para un timeframe específico.
        
        Args:
            timeframe: Intervalo temporal ('5m', '15m', etc.)
            
        Returns:
            Ruta al archivo de datos
        """
        symbol_clean = self.symbol.replace('/', '_')
        return os.path.join(self.data_dir, f"{symbol_clean}_{timeframe}_data.csv")
    
    @retry_with_backoff(stop_max_attempt_number=3, wait_exponential_multiplier=1000, wait_exponential_max=10000)
    def _fetch_historical_data(self, timeframe: str, since: Optional[datetime] = None, limit: int = 1000) -> pd.DataFrame:
        """
        Obtiene datos históricos del exchange con reintentos automáticos.
        
        Args:
            timeframe: Intervalo temporal ('1m', '5m', '15m', '1h', '4h', '1d')
            since: Fecha de inicio (opcional)
            limit: Límite de registros a obtener
            
        Returns:
            DataFrame con datos históricos
        """
        try:
            # Verificar si el exchange está inicializado
            if self.exchange is None:
                logger.error("No hay exchange inicializado para obtener datos")
                return pd.DataFrame()
            
            # Validar timeframe
            if timeframe not in self.exchange.timeframes:
                available_timeframes = ', '.join(self.exchange.timeframes)
                logger.error(f"Timeframe {timeframe} no soportado. Timeframes disponibles: {available_timeframes}")
                return pd.DataFrame()
            
            # Calcular timestamp de inicio si no se proporciona
            if since is None:
                since = datetime.now() - timedelta(days=self.lookback_days)
            
            # Asegurar que since está en UTC
            if since.tzinfo is None:
                since = since.replace(tzinfo=pytz.UTC)
            
            # Convertir datetime a timestamp en milisegundos
            since_timestamp = int(since.timestamp() * 1000)
            
            # Obtener datos OHLCV con manejo de límites de API
            all_ohlcv = []
            current_since = since_timestamp
            now_timestamp = int(datetime.now().timestamp() * 1000)
            
            # Obtener datos en bloques para evitar límites de API
            while current_since < now_timestamp:
                logger.debug(f"Obteniendo datos desde {datetime.fromtimestamp(current_since/1000)} para {timeframe}")
                
                # Obtener bloque de datos
                ohlcv_chunk = self.exchange.fetch_ohlcv(self.symbol, timeframe, since=current_since, limit=limit)
                
                if not ohlcv_chunk:
                    logger.warning(f"No se obtuvieron datos para {timeframe} desde {datetime.fromtimestamp(current_since/1000)}")
                    break
                
                all_ohlcv.extend(ohlcv_chunk)
                
                # Actualizar timestamp para siguiente bloque
                last_timestamp = ohlcv_chunk[-1][0]
                if last_timestamp <= current_since:
                    logger.warning("No se pueden obtener más datos históricos (timestamp no avanza)")
                    break
                
                current_since = last_timestamp + 1
                
                # Evitar exceder límites de API
                time.sleep(self.exchange.rateLimit / 1000)  # Convertir ms a segundos
                
                # Limitar número total de registros para evitar consumo excesivo de memoria
                if len(all_ohlcv) >= 10000:  # Límite arbitrario de 10,000 registros
                    logger.warning(f"Alcanzado límite de 10,000 registros para {timeframe}")
                    break
            
            # Verificar si se obtuvieron datos
            if not all_ohlcv:
                logger.error(f"No se obtuvieron datos para {self.symbol} en timeframe {timeframe}")
                return pd.DataFrame()
            
            # Convertir a DataFrame
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convertir tipos de datos
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Eliminar duplicados y valores nulos
            df.drop_duplicates(subset=['timestamp'], inplace=True)
            df.dropna(inplace=True)
            
            # Convertir timestamp a datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Establecer timestamp como índice
            df.set_index('timestamp', inplace=True)
            
            # Ordenar por índice
            df.sort_index(inplace=True)
            
            logger.info(f"Datos históricos obtenidos para {self.symbol} en timeframe {timeframe}: {len(df)} registros")
            
            # Guardar en caché
            self.data_cache[timeframe] = df
            
            return df
            
        except ccxt.NetworkError as e:
            logger.error(f"Error de red al obtener datos históricos: {str(e)}")
            raise  # Permitir reintento con el decorador
            
        except ccxt.ExchangeError as e:
            logger.error(f"Error del exchange al obtener datos históricos: {str(e)}")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error inesperado al obtener datos históricos: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
        return pd.DataFrame()  # DataFrame vacío
    
    def _load_cached_data(self, timeframe: str) -> pd.DataFrame:
        """
        Carga datos desde archivo si existen.
        
        Args:
            timeframe: Intervalo temporal ('5m', '15m', etc.)
            
        Returns:
            DataFrame con datos cargados o vacío si no existe
        """
        filename = self._get_filename(timeframe)
        
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                logger.info(f"Datos cargados desde {filename}: {len(df)} registros")
                return df
            except Exception as e:
                logger.error(f"Error al cargar datos desde {filename}: {str(e)}")
        
        logger.info(f"No se encontraron datos en caché para {timeframe}")
        return pd.DataFrame()  # DataFrame vacío
    
    def _save_data(self, timeframe: str, df: pd.DataFrame) -> bool:
        """
        Guarda datos en archivo CSV.
        
        Args:
            timeframe: Intervalo temporal ('5m', '15m', etc.)
            df: DataFrame a guardar
            
        Returns:
            True si se guardó correctamente, False en caso contrario
        """
        if df.empty:
            logger.warning(f"No hay datos para guardar en {timeframe}")
            return False
        
        try:
            filename = self._get_filename(timeframe)
            # Resetear índice para guardar la columna timestamp
            df_to_save = df.reset_index()
            df_to_save.to_csv(filename, index=False)
            logger.info(f"Datos guardados en {filename}: {len(df)} registros")
            return True
        except Exception as e:
            logger.error(f"Error al guardar datos en {filename}: {str(e)}")
            return False
    
    def load_all_timeframes(self, force_update: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Carga datos para todos los timeframes configurados.
        
        Args:
            force_update: Si es True, fuerza actualización desde el exchange
            
        Returns:
            Diccionario con DataFrames por timeframe
        """
        for tf in self.timeframes:
            self.load_timeframe(tf, force_update)
        
        return self.dataframes
    
    def load_timeframe(self, timeframe: str, force_update: bool = False) -> pd.DataFrame:
        """
        Carga datos para un timeframe específico.
        
        Args:
            timeframe: Intervalo temporal ('5m', '15m', etc.)
            force_update: Si es True, fuerza actualización desde el exchange
            
        Returns:
            DataFrame con datos del timeframe
        """
        # Verificar si el timeframe es válido
        if timeframe not in self.timeframes:
            logger.error(f"Timeframe {timeframe} no configurado")
            return pd.DataFrame()
        
        # Si no se fuerza actualización, intentar cargar desde caché
        if not force_update:
            df = self._load_cached_data(timeframe)
            if not df.empty:
                self.dataframes[timeframe] = df
                self.last_update[timeframe] = datetime.now()
                return df
        
        # Calcular desde cuándo obtener datos
        since = int((datetime.now() - timedelta(days=self.lookback_days)).timestamp() * 1000)
        
        # Obtener datos históricos
        df = self._fetch_historical_data(timeframe, since)
        
        if not df.empty:
            # Guardar en caché
            self._save_data(timeframe, df)
            
            # Actualizar en memoria
            self.dataframes[timeframe] = df
            self.last_update[timeframe] = datetime.now()
        
        return df
    
    def update_data(self) -> Dict[str, bool]:
        """
        Actualiza los datos de todos los timeframes con los últimos disponibles.
        
        Returns:
            Diccionario indicando éxito de actualización por timeframe
        """
        results = {}
        
        for tf in self.timeframes:
            # Cargar datos existentes
            df_existing = self.dataframes.get(tf, pd.DataFrame())
            
            if df_existing.empty:
                # Si no hay datos, cargar todo
                results[tf] = not self.load_timeframe(tf, True).empty
                continue
            
            # Obtener última fecha en datos existentes
            last_date = df_existing.index[-1]
            since = int(last_date.timestamp() * 1000)
            
            # Obtener nuevos datos
            df_new = self._fetch_historical_data(tf, since)
            
            if not df_new.empty:
                # Combinar datos existentes con nuevos
                df_combined = pd.concat([df_existing, df_new])
                # Eliminar duplicados
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                # Ordenar por índice
                df_combined.sort_index(inplace=True)
                
                # Actualizar en memoria y guardar
                self.dataframes[tf] = df_combined
                self._save_data(tf, df_combined)
                self.last_update[tf] = datetime.now()
                
                results[tf] = True
                logger.info(f"Datos actualizados para {tf}: {len(df_new)} nuevos registros")
            else:
                results[tf] = False
        
        return results
    
    def get_synchronized_data(self, base_timeframe: str = '15m') -> Dict[str, pd.DataFrame]:
        """
        Obtiene datos sincronizados para todos los timeframes.
        
        Args:
            base_timeframe: Timeframe base para sincronización
            
        Returns:
            Diccionario con DataFrames sincronizados por timeframe
        """
        if base_timeframe not in self.timeframes:
            logger.error(f"Timeframe base {base_timeframe} no configurado")
            return {}
        
        # Asegurarse de que todos los timeframes estén cargados
        self.load_all_timeframes()
        
        # Obtener fechas del timeframe base
        base_df = self.dataframes.get(base_timeframe, pd.DataFrame())
        if base_df.empty:
            logger.error(f"No hay datos para el timeframe base {base_timeframe}")
            return {}
        
        base_dates = base_df.index
        
        # Diccionario para almacenar datos sincronizados
        synced_data = {base_timeframe: base_df}
        
        # Sincronizar cada timeframe con el base
        for tf in self.timeframes:
            if tf == base_timeframe:
                continue
            
            df = self.dataframes.get(tf, pd.DataFrame())
            if df.empty:
                logger.warning(f"No hay datos para sincronizar en timeframe {tf}")
                continue
            
            # Para cada fecha en el timeframe base, encontrar el dato correspondiente
            # en el otro timeframe (el más reciente anterior a esa fecha)
            synced_df = pd.DataFrame(index=base_dates)
            
            # Reindexar usando el método asof (anterior o igual)
            for col in df.columns:
                synced_df[col] = df[col].reindex(index=synced_df.index, method='pad')
            
            synced_data[tf] = synced_df
        
        return synced_data
    
    def get_latest_data(self, timeframe: str, n_samples: int = 100) -> pd.DataFrame:
        """
        Obtiene los últimos N registros de un timeframe específico.
        
        Args:
            timeframe: Intervalo temporal ('5m', '15m', etc.)
            n_samples: Número de muestras a obtener
            
        Returns:
            DataFrame con los últimos N registros
        """
        if timeframe not in self.dataframes or self.dataframes[timeframe].empty:
            self.load_timeframe(timeframe)
        
        df = self.dataframes.get(timeframe, pd.DataFrame())
        
        if df.empty:
            logger.warning(f"No hay datos disponibles para {timeframe}")
            return pd.DataFrame()
        
        # Obtener los últimos n_samples
        return df.iloc[-n_samples:]

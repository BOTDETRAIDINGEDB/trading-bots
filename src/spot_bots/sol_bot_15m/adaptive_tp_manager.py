#!/usr/bin/env python3
"""
Gestor de Take Profit Adaptativo para el bot SOL
Ajusta dinámicamente el take profit para maximizar ganancias mientras mantiene un stop loss fijo
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

# Ajustar el PYTHONPATH para encontrar los módulos correctamente
bot_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(bot_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Importar módulos del bot
from utils.telegram_notifier import TelegramNotifier

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("adaptive_tp_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdaptiveTPManager:
    """Gestor de Take Profit Adaptativo para el bot SOL."""
    
    def __init__(self, state_file, config_file='adaptive_tp_config.json'):
        """
        Inicializa el gestor de take profit adaptativo.
        
        Args:
            state_file (str): Ruta al archivo de estado del bot.
            config_file (str): Ruta al archivo de configuración.
        """
        self.state_file = state_file
        self.config_file = config_file
        self.telegram = TelegramNotifier()
        
        # Cargar configuración
        self.config = self.load_config()
        if not self.config:
            # Configuración por defecto
            self.config = {
                'fixed_stop_loss_pct': 0.06,  # 6% del capital como máximo de pérdida
                'initial_take_profit_pct': 0.04,  # Take profit inicial
                'tp_adjustment_factor': 0.005,  # Cuánto ajustar el TP en cada iteración
                'volatility_factor': 0.5,  # Influencia de la volatilidad en el TP
                'trend_factor': 0.5,  # Influencia de la tendencia en el TP
                'min_take_profit_pct': 0.02,  # Mínimo TP
                'max_take_profit_pct': 0.15,  # Máximo TP
                'capital_tiers': {
                    'micro': {
                        'max_amount': 50,
                        'position_size_pct': 0.3,  # 30% del capital disponible
                        'risk_factor': 0.7  # Reduce el riesgo al 70%
                    },
                    'small': {
                        'max_amount': 200,
                        'position_size_pct': 0.4,
                        'risk_factor': 0.8
                    },
                    'medium': {
                        'max_amount': 1000,
                        'position_size_pct': 0.5,
                        'risk_factor': 0.9
                    },
                    'large': {
                        'max_amount': float('inf'),
                        'position_size_pct': 0.6,
                        'risk_factor': 1.0
                    }
                },
                'learning_mode': {
                    'enabled': True,
                    'min_trades': 20,  # Mínimo de operaciones para salir del modo aprendizaje
                    'min_win_rate': 55,  # Win rate mínimo para salir del modo aprendizaje
                    'position_size_multiplier': 0.5  # Reduce el tamaño de posición al 50% en modo aprendizaje
                },
                'last_update': None
            }
            self.save_config()
        
        logger.info(f"Gestor de Take Profit Adaptativo inicializado")
    
    def load_config(self):
        """
        Carga la configuración desde el archivo.
        
        Returns:
            dict: Configuración, o None si no se pudo cargar.
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de configuración no encontrado: {self.config_file}")
                return None
        except Exception as e:
            logger.error(f"Error al cargar la configuración: {str(e)}")
            return None
    
    def save_config(self):
        """
        Guarda la configuración en el archivo.
        
        Returns:
            bool: True si se guardó correctamente, False en caso contrario.
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
            logger.info(f"Configuración guardada en {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Error al guardar la configuración: {str(e)}")
            return False
    
    def load_state(self):
        """
        Carga el estado del bot desde el archivo de estado.
        
        Returns:
            dict: Estado del bot, o None si no se pudo cargar.
        """
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Archivo de estado no encontrado: {self.state_file}")
                return None
        except Exception as e:
            logger.error(f"Error al cargar el estado: {str(e)}")
            return None
    
    def save_state(self, state):
        """
        Guarda el estado del bot en el archivo de estado.
        
        Args:
            state (dict): Estado del bot.
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario.
        """
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=4)
            logger.info(f"Estado guardado en {self.state_file}")
            return True
        except Exception as e:
            logger.error(f"Error al guardar el estado: {str(e)}")
            return False
    
    def get_market_data(self, symbol='SOLUSDT', interval='15m', limit=100):
        """
        Obtiene datos de mercado de Binance.
        
        Args:
            symbol (str): Símbolo de trading.
            interval (str): Intervalo de tiempo.
            limit (int): Número de velas a obtener.
            
        Returns:
            pd.DataFrame: DataFrame con los datos de mercado, o None si hubo un error.
        """
        try:
            # Obtener datos de mercado de Binance
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params)
            if response.status_code != 200:
                logger.error(f"Error al obtener datos de mercado: {response.status_code} - {response.text}")
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
            logger.error(f"Error al obtener datos de mercado: {str(e)}")
            return None
    
    def calculate_volatility(self, df):
        """
        Calcula la volatilidad del mercado.
        
        Args:
            df (pd.DataFrame): DataFrame con datos de mercado.
            
        Returns:
            float: Volatilidad normalizada (0-1).
        """
        if df is None or len(df) < 10:
            return 0.5  # Valor por defecto
        
        try:
            # Calcular volatilidad como ATR normalizado
            df['tr1'] = abs(df['high'] - df['low'])
            df['tr2'] = abs(df['high'] - df['close'].shift(1))
            df['tr3'] = abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
            df['atr'] = df['tr'].rolling(14).mean()
            
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
    
    def calculate_trend_strength(self, df):
        """
        Calcula la fuerza de la tendencia.
        
        Args:
            df (pd.DataFrame): DataFrame con datos de mercado.
            
        Returns:
            float: Fuerza de tendencia (-1 a 1).
        """
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
    
    def determine_capital_tier(self, available_balance):
        """
        Determina el nivel de capital.
        
        Args:
            available_balance (float): Balance disponible.
            
        Returns:
            dict: Configuración del nivel de capital.
        """
        tiers = self.config['capital_tiers']
        
        for tier_name, tier_config in sorted(tiers.items(), key=lambda x: x[1]['max_amount']):
            if available_balance <= tier_config['max_amount']:
                logger.info(f"Nivel de capital determinado: {tier_name} (balance: {available_balance})")
                return tier_name, tier_config
        
        # Si no se encuentra ningún nivel (no debería ocurrir con la configuración actual)
        logger.warning(f"No se encontró nivel de capital para balance {available_balance}, usando 'large'")
        return 'large', tiers['large']
    
    def is_learning_mode(self, state):
        """
        Determina si el bot está en modo aprendizaje.
        
        Args:
            state (dict): Estado del bot.
            
        Returns:
            bool: True si está en modo aprendizaje, False en caso contrario.
        """
        if not self.config['learning_mode']['enabled']:
            return False
        
        metrics = state.get('performance_metrics', {})
        total_trades = metrics.get('total_trades', 0)
        win_rate = metrics.get('win_rate', 0)
        
        # Verificar condiciones para salir del modo aprendizaje
        if (total_trades >= self.config['learning_mode']['min_trades'] and 
            win_rate >= self.config['learning_mode']['min_win_rate']):
            logger.info(f"Saliendo del modo aprendizaje: trades={total_trades}, win_rate={win_rate}")
            return False
        
        logger.info(f"En modo aprendizaje: trades={total_trades}, win_rate={win_rate}")
        return True
    
    def calculate_adaptive_take_profit(self, state):
        """
        Calcula un take profit adaptativo basado en condiciones de mercado.
        
        Args:
            state (dict): Estado del bot.
            
        Returns:
            dict: Parámetros de trading actualizados.
        """
        # Obtener datos de mercado
        market_data = self.get_market_data(symbol='SOLUSDT', interval='15m', limit=100)
        
        # Calcular indicadores de mercado
        volatility = self.calculate_volatility(market_data)
        trend_strength = self.calculate_trend_strength(market_data)
        
        # Obtener take profit base
        base_tp = self.config['initial_take_profit_pct']
        
        # Ajustar take profit según volatilidad (mayor volatilidad = mayor TP)
        volatility_adjustment = (volatility - 0.5) * self.config['volatility_factor'] * self.config['tp_adjustment_factor']
        
        # Ajustar take profit según tendencia (tendencia fuerte = mayor TP en dirección de la tendencia)
        trend_adjustment = trend_strength * self.config['trend_factor'] * self.config['tp_adjustment_factor']
        
        # Calcular take profit adaptativo
        adaptive_tp = base_tp + volatility_adjustment + trend_adjustment
        
        # Limitar a rango configurado
        adaptive_tp = max(self.config['min_take_profit_pct'], min(self.config['max_take_profit_pct'], adaptive_tp))
        
        logger.info(f"Take profit adaptativo calculado: {adaptive_tp:.4f} (base: {base_tp}, vol_adj: {volatility_adjustment:.4f}, trend_adj: {trend_adjustment:.4f})")
        
        # Actualizar parámetros en el estado
        updated_params = {
            'take_profit_pct': adaptive_tp,
            'stop_loss_pct': self.config['fixed_stop_loss_pct'],
            'market_volatility': volatility,
            'market_trend': trend_strength,
            'last_tp_update': datetime.now().isoformat()
        }
        
        return updated_params
    
    def calculate_position_size(self, state, available_balance):
        """
        Calcula el tamaño de posición basado en el capital disponible.
        
        Args:
            state (dict): Estado del bot.
            available_balance (float): Balance disponible.
            
        Returns:
            float: Tamaño de posición como porcentaje del balance.
        """
        # Determinar nivel de capital
        tier_name, tier_config = self.determine_capital_tier(available_balance)
        
        # Obtener tamaño de posición base según nivel
        position_size_pct = tier_config['position_size_pct']
        
        # Aplicar factor de riesgo
        risk_factor = tier_config['risk_factor']
        
        # Verificar si está en modo aprendizaje
        if self.is_learning_mode(state):
            # Reducir tamaño de posición en modo aprendizaje
            learning_multiplier = self.config['learning_mode']['position_size_multiplier']
            position_size_pct *= learning_multiplier
            logger.info(f"Modo aprendizaje activo: tamaño de posición reducido a {position_size_pct:.2%}")
        
        # Calcular tamaño de posición final
        final_position_size = position_size_pct * risk_factor
        
        logger.info(f"Tamaño de posición calculado: {final_position_size:.2%} (nivel: {tier_name}, balance: {available_balance})")
        
        return final_position_size
    
    def update_trading_parameters(self):
        """
        Actualiza los parámetros de trading en el estado del bot.
        
        Returns:
            bool: True si se actualizó correctamente, False en caso contrario.
        """
        # Cargar estado actual
        state = self.load_state()
        if not state:
            logger.error("No se pudo cargar el estado del bot")
            return False
        
        # Obtener balance disponible
        available_balance = state.get('current_balance', 0)
        if available_balance <= 0:
            logger.error(f"Balance disponible inválido: {available_balance}")
            return False
        
        # Calcular take profit adaptativo
        tp_params = self.calculate_adaptive_take_profit(state)
        
        # Calcular tamaño de posición
        position_size_pct = self.calculate_position_size(state, available_balance)
        
        # Actualizar parámetros en el estado
        state.update(tp_params)
        state['risk_per_trade'] = position_size_pct
        
        # Guardar estado actualizado
        success = self.save_state(state)
        
        if success:
            # Actualizar timestamp de última actualización
            self.config['last_update'] = datetime.now().isoformat()
            self.save_config()
            
            # Notificar actualización
            self.notify_parameter_update(state, tp_params, position_size_pct)
        
        return success
    
    def notify_parameter_update(self, state, tp_params, position_size_pct):
        """
        Notifica la actualización de parámetros por Telegram.
        
        Args:
            state (dict): Estado del bot.
            tp_params (dict): Parámetros de take profit.
            position_size_pct (float): Tamaño de posición como porcentaje.
            
        Returns:
            bool: True si se envió correctamente, False en caso contrario.
        """
        learning_mode = "✅ Activo" if self.is_learning_mode(state) else "❌ Inactivo"
        tier_name, _ = self.determine_capital_tier(state.get('current_balance', 0))
        
        message = f"""🔄 *Parámetros de Trading Actualizados - Bot SOL*

💰 *Balance:* {state.get('current_balance', 0):.2f} USDT
📊 *Nivel de capital:* {tier_name.capitalize()}
🧠 *Modo aprendizaje:* {learning_mode}

📈 *Parámetros actualizados:*
- Take Profit: {tp_params['take_profit_pct'] * 100:.2f}%
- Stop Loss: {tp_params['stop_loss_pct'] * 100:.2f}% (fijo)
- Tamaño de posición: {position_size_pct * 100:.2f}% del balance

📉 *Condiciones de mercado:*
- Volatilidad: {tp_params['market_volatility'] * 100:.2f}%
- Tendencia: {tp_params['market_trend']:.4f}

⏱️ Actualizado: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        return self.telegram.send_message(message)
    
    def run(self):
        """
        Ejecuta el gestor de take profit adaptativo.
        
        Returns:
            bool: True si se ejecutó correctamente, False en caso contrario.
        """
        return self.update_trading_parameters()

def parse_arguments():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description='Gestor de Take Profit Adaptativo para el bot SOL')
    parser.add_argument('--state-file', type=str, default='sol_bot_15min_state.json', help='Archivo de estado del bot')
    parser.add_argument('--config-file', type=str, default='adaptive_tp_config.json', help='Archivo de configuración')
    parser.add_argument('--fixed-sl', type=float, help='Stop Loss fijo como porcentaje (ej: 6 para 6%%)')
    parser.add_argument('--min-tp', type=float, help='Take Profit mínimo como porcentaje')
    parser.add_argument('--max-tp', type=float, help='Take Profit máximo como porcentaje')
    parser.add_argument('--disable-learning', action='store_true', help='Desactivar modo aprendizaje')
    return parser.parse_args()

def main():
    """Función principal."""
    # Cargar variables de entorno
    load_dotenv()
    
    # Parsear argumentos
    args = parse_arguments()
    
    # Inicializar gestor
    tp_manager = AdaptiveTPManager(args.state_file, args.config_file)
    
    # Actualizar configuración si se especifican argumentos
    if args.fixed_sl:
        tp_manager.config['fixed_stop_loss_pct'] = args.fixed_sl / 100
    
    if args.min_tp:
        tp_manager.config['min_take_profit_pct'] = args.min_tp / 100
    
    if args.max_tp:
        tp_manager.config['max_take_profit_pct'] = args.max_tp / 100
    
    if args.disable_learning:
        tp_manager.config['learning_mode']['enabled'] = False
    
    tp_manager.save_config()
    
    # Ejecutar gestor
    success = tp_manager.run()
    
    if success:
        logger.info("Gestor de Take Profit Adaptativo ejecutado correctamente")
    else:
        logger.error("Error al ejecutar el Gestor de Take Profit Adaptativo")

if __name__ == "__main__":
    main()

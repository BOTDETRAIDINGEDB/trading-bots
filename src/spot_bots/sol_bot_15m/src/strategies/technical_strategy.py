# technical_strategy.py
import logging
import json
import os
import pandas as pd
from datetime import datetime

# Importar el modelo de ML
from models.ml_model import MLModel

logger = logging.getLogger(__name__)

class TechnicalStrategy:
    """Estrategia de trading basada en indicadores técnicos."""
    
    def __init__(self, symbol, risk_per_trade=0.02, stop_loss_pct=0.02, take_profit_pct=0.04, max_trades=3, use_ml=True):
        """
        Inicializa la estrategia de trading.
        
        Args:
            symbol (str): Par de trading (ej. 'SOLUSDT').
            risk_per_trade (float): Porcentaje del capital a arriesgar por operación (0.02 = 2%).
            stop_loss_pct (float): Porcentaje de stop loss (0.02 = 2%).
            take_profit_pct (float): Porcentaje de take profit (0.04 = 4%).
            max_trades (int): Número máximo de operaciones simultáneas.
            use_ml (bool): Si es True, utiliza el modelo de ML para las predicciones.
        """
        self.symbol = symbol
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_trades = max_trades
        self.use_ml = use_ml
        
        # Inicializar el modelo de ML
        if self.use_ml:
            model_path = f"{symbol.lower()}_model.pkl"
            self.ml_model = MLModel(model_path=model_path)
            logger.info(f"Modelo de ML inicializado para {symbol} en {model_path}")
        else:
            self.ml_model = None
        
        # Estado de la estrategia
        self.position = 0  # 0: sin posición, 1: long, -1: short
        self.entry_price = 0.0
        self.position_size = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_stop = 0.0
        self.trailing_active = False
        self.trailing_percent = 0.015  # 1.5% de trailing stop
        self.highest_price = 0.0  # Para tracking del trailing stop
        self.trades = []
        self.current_balance = 0.0
        self.initial_balance = 0.0
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'total_profit': 0.0
        }
        
        logger.info(f"Estrategia técnica inicializada para {symbol} con riesgo por operación de {risk_per_trade*100}%")
    
    def set_balance(self, balance):
        """
        Establece el balance actual y el balance inicial.
        
        Args:
            balance (float): Balance actual en USDT.
        """
        self.current_balance = balance
        if self.initial_balance == 0:
            self.initial_balance = balance
        logger.info(f"Balance establecido: {balance} USDT")
    
    def calculate_position_size(self, price):
        """
        Calcula el tamaño de la posición basado en el riesgo por operación.
        
        Args:
            price (float): Precio actual del activo.
            
        Returns:
            float: Tamaño de la posición en unidades del activo.
        """
        # Calcular el monto a arriesgar
        risk_amount = self.current_balance * self.risk_per_trade
        
        # Calcular el tamaño de la posición
        position_size = risk_amount / (price * self.stop_loss_pct)
        
        # Redondear a 4 decimales (ajustar según la precisión del par)
        position_size = round(position_size, 4)
        
        logger.info(f"Tamaño de posición calculado: {position_size} unidades a precio {price}")
        return position_size
    
    def should_enter_trade(self, signal, price, available_balance, ml_prediction=None):
        """
        Determina si se debe entrar en una operación.
        
        Args:
            signal (int): Señal de trading (1: comprar, -1: vender, 0: mantener).
            price (float): Precio actual del activo.
            available_balance (float): Balance disponible para operar.
            ml_prediction (int, optional): Predicción del modelo de ML si está disponible.
            
        Returns:
            bool: True si se debe entrar en la operación, False en caso contrario.
        """
        # Si ya estamos en una posición, no entrar en otra
        if self.position != 0:
            return False
        
        # Si estamos usando ML y tenemos una predicción, combinar con la señal técnica
        if self.use_ml and ml_prediction is not None:
            # Convertir ml_prediction a un valor simple si es un array
            ml_pred_value = 1 if hasattr(ml_prediction, 'any') and ml_prediction.any() == 1 else 0
            
            # Solo para logging
            if hasattr(ml_prediction, 'shape') and len(ml_prediction.shape) > 0:
                logger.info(f"Predicción ML es un array de forma {ml_prediction.shape}, usando valor simplificado: {ml_pred_value}")
            
            # MODIFICADO: Hacer más sensible - entrar si cualquiera de las señales es positiva
            if ml_pred_value == 1 or signal == 1:
                logger.info(f"Señal de entrada detectada: ML={ml_pred_value}, Técnica={signal}")
                return self._validate_trade_conditions(price, available_balance)
            else:
                return False
        else:
            # Comportamiento original basado solo en señales técnicas
            # MODIFICADO: Ser más sensible a señales técnicas
            if signal >= 0:  # Incluir señales neutrales (0) como potenciales entradas
                return self._validate_trade_conditions(price, available_balance)
            
            # Por ahora solo implementamos posiciones largas (compra)
            if signal != 1:
                return False
                
            return self._validate_trade_conditions(price, available_balance)
    
    def _validate_trade_conditions(self, price, available_balance):
        """
        Valida las condiciones adicionales para entrar en una operación.
        
        Args:
            price (float): Precio actual del activo.
            available_balance (float): Balance disponible para operar.
            
        Returns:
            bool: True si se cumplen las condiciones, False en caso contrario.
        """
        
        # Verificar si tenemos suficiente balance
        position_size = self.calculate_position_size(price)
        position_cost = position_size * price
        
        if position_cost > available_balance:
            logger.warning(f"Balance insuficiente para entrar en operación. Necesario: {position_cost}, Disponible: {available_balance}")
            return False
        
        # Verificar si ya tenemos el máximo de operaciones
        active_trades = sum(1 for trade in self.trades if trade['status'] == 'open')
        if active_trades >= self.max_trades:
            logger.warning(f"Máximo de operaciones alcanzado ({self.max_trades}). No se puede entrar en una nueva operación.")
            return False
        
        logger.info(f"Señal de entrada válida a precio {price}")
        return True
    
    def enter_trade(self, price, timestamp):
        """
        Entra en una operación.
        
        Args:
            price (float): Precio de entrada.
            timestamp (datetime): Timestamp de la entrada.
            
        Returns:
            dict: Información de la operación.
        """
        # Calcular tamaño de la posición
        self.position_size = self.calculate_position_size(price)
        
        # Establecer precio de entrada
        self.entry_price = price
        
        # Calcular stop loss y take profit
        self.stop_loss = price * (1 - self.stop_loss_pct)
        self.take_profit = price * (1 + self.take_profit_pct)
        
        # Resetear variables de trailing stop
        self.trailing_stop = 0.0
        self.trailing_active = False
        self.highest_price = price
        
        # Actualizar posición
        self.position = 1  # Long
        
        # Registrar la operación
        trade = {
            'id': len(self.trades) + 1,
            'type': 'long',
            'entry_price': price,
            'entry_time': timestamp,
            'position_size': self.position_size,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'exit_price': 0.0,
            'exit_time': None,
            'profit_loss': 0.0,
            'profit_loss_pct': 0.0,
            'status': 'open'
        }
        
        self.trades.append(trade)
        logger.info(f"Entrada en operación LONG a precio {price}, tamaño: {self.position_size}, SL: {self.stop_loss}, TP: {self.take_profit}")
        
        return trade
    
    def should_exit_trade(self, price, current_time):
        """
        Determina si se debe salir de una operación.
        
        Args:
            price (float): Precio actual del activo.
            current_time (datetime): Timestamp actual.
            
        Returns:
            bool: True si se debe salir de la operación, False en caso contrario.
        """
        # Si no estamos en una posición, no hay nada que salir
        if self.position == 0:
            return False
        
        # Actualizar el precio más alto alcanzado para el trailing stop
        if price > self.highest_price and self.position == 1:
            self.highest_price = price
            
            # Activar trailing stop cuando el precio supere el 50% del camino hacia el take profit
            halfway_to_tp = self.entry_price + ((self.take_profit - self.entry_price) * 0.5)
            
            if not self.trailing_active and price >= halfway_to_tp:
                self.trailing_active = True
                self.trailing_stop = price * (1 - self.trailing_percent)
                logger.info(f"Trailing stop activado a {self.trailing_stop} (precio actual: {price})")
            
            # Actualizar el trailing stop si ya está activo
            elif self.trailing_active:
                new_trailing_stop = price * (1 - self.trailing_percent)
                if new_trailing_stop > self.trailing_stop:
                    self.trailing_stop = new_trailing_stop
                    logger.info(f"Trailing stop actualizado a {self.trailing_stop} (precio actual: {price})")
        
        # Verificar trailing stop si está activo
        if self.trailing_active and price <= self.trailing_stop:
            logger.info(f"Trailing Stop alcanzado a precio {price} (TS: {self.trailing_stop})")
            return True
        
        # Verificar stop loss tradicional
        if price <= self.stop_loss:
            logger.info(f"Stop Loss alcanzado a precio {price} (SL: {self.stop_loss})")
            return True
        
        # Verificar take profit
        if price >= self.take_profit:
            logger.info(f"Take Profit alcanzado a precio {price} (TP: {self.take_profit})")
            return True
        
        # Verificar tiempo en la operación (salir después de 24 horas si no se ha alcanzado SL o TP)
        if self.trades and self.trades[-1]['status'] == 'open':
            entry_time = self.trades[-1]['entry_time']
            time_in_trade = (current_time - entry_time).total_seconds() / 3600  # Horas
            
            if time_in_trade >= 24:
                logger.info(f"Tiempo máximo en operación alcanzado: {time_in_trade} horas")
                return True
        
        return False
    
    def exit_trade(self, price, timestamp):
        """
        Sale de una operación.
        
        Args:
            price (float): Precio de salida.
            timestamp (datetime): Timestamp de la salida.
            
        Returns:
            dict: Detalles de la operación cerrada.
        """
        if not self.trades or self.trades[-1]['status'] != 'open':
            logger.warning("No hay operaciones abiertas para salir.")
            return None
        
        # Obtener la última operación
        trade = self.trades[-1]
        
        # Calcular profit/loss
        if self.position == 1:  # Long
            profit_loss = (price - trade['entry_price']) * trade['position_size']
            profit_loss_pct = (price / trade['entry_price'] - 1) * 100
        else:  # Short (no implementado aún)
            profit_loss = 0
            profit_loss_pct = 0
        
        # Actualizar la operación
        trade['exit_price'] = price
        trade['exit_time'] = timestamp
        trade['profit_loss'] = profit_loss
        trade['profit_loss_pct'] = profit_loss_pct
        trade['status'] = 'closed'
        
        # Actualizar el balance
        self.current_balance += profit_loss
        
        # Actualizar métricas de rendimiento
        self.update_performance_metrics(trade)
        
        # Resetear el estado de la posición
        self.position = 0
        self.entry_price = 0.0
        self.position_size = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        
        logger.info(f"Salida de operación a precio {price}, P/L: {profit_loss:.2f} USDT ({profit_loss_pct:.2f}%)")
        
        return trade
    
    def update_performance_metrics(self, trade):
        """
        Actualiza las métricas de rendimiento basadas en una operación cerrada.
        
        Args:
            trade (dict): Detalles de la operación cerrada.
        """
        metrics = self.performance_metrics
        
        # Incrementar contador de operaciones
        metrics['total_trades'] += 1
        
        # Actualizar contadores de operaciones ganadoras/perdedoras
        if trade['profit_loss'] > 0:
            metrics['winning_trades'] += 1
        else:
            metrics['losing_trades'] += 1
        
        # Actualizar profit/loss total
        metrics['total_profit'] += trade['profit_loss']
        
        # Calcular win rate
        if metrics['total_trades'] > 0:
            metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] * 100
        
        # Calcular profit factor y promedios
        total_profit = sum(t['profit_loss'] for t in self.trades if t['status'] == 'closed' and t['profit_loss'] > 0)
        total_loss = sum(abs(t['profit_loss']) for t in self.trades if t['status'] == 'closed' and t['profit_loss'] < 0)
        
        if metrics['winning_trades'] > 0:
            metrics['avg_profit'] = total_profit / metrics['winning_trades']
        
        if metrics['losing_trades'] > 0:
            metrics['avg_loss'] = total_loss / metrics['losing_trades']
        
        if total_loss > 0:
            metrics['profit_factor'] = total_profit / total_loss
        
        # Calcular drawdown
        peak_balance = self.initial_balance
        current_drawdown = 0
        
        for t in self.trades:
            if t['status'] == 'closed':
                balance_after_trade = peak_balance + t['profit_loss']
                if balance_after_trade > peak_balance:
                    peak_balance = balance_after_trade
                else:
                    drawdown = (peak_balance - balance_after_trade) / peak_balance * 100
                    if drawdown > current_drawdown:
                        current_drawdown = drawdown
        
        metrics['current_drawdown'] = current_drawdown
        metrics['max_drawdown'] = max(metrics['max_drawdown'], current_drawdown)
        
        logger.info(f"Métricas actualizadas: Win Rate: {metrics['win_rate']:.2f}%, P/L Total: {metrics['total_profit']:.2f} USDT")
    
    def get_performance_metrics(self):
        """
        Obtiene las métricas de rendimiento actuales.
        
        Returns:
            dict: Métricas de rendimiento.
        """
        return self.performance_metrics
    
    def should_retrain_model(self):
        """
        Determina si el modelo de ML debe ser reentrenado basado en el tiempo transcurrido
        desde el último entrenamiento y la cantidad de datos nuevos disponibles.
        
        Returns:
            bool: True si el modelo debe ser reentrenado, False en caso contrario.
        """
        # Si no hay tiempo de último entrenamiento, reentrenar
        if not hasattr(self, 'last_train_time'):
            return True
            
        # Verificar si ha pasado suficiente tiempo desde el último entrenamiento
        time_since_last_train = (datetime.now() - self.last_train_time).total_seconds() / 60  # en minutos
        
        # Por defecto, reentrenar cada 60 minutos (1 hora)
        retrain_interval = 60
        
        # Si hay suficientes datos nuevos o ha pasado suficiente tiempo, reentrenar
        return time_since_last_train >= retrain_interval
    
    def process_with_ml(self, df):
        """
        Procesa los datos con el modelo de ML.
        
        Args:
            df (DataFrame): DataFrame con los datos procesados.
            
        Returns:
            tuple: (predicción, métricas de reentrenamiento si se reentrena el modelo)
        """
        if not self.use_ml or self.ml_model is None:
            logger.warning("Modelo de ML no habilitado o no inicializado")
            return None, None
        
        # Guardar el DataFrame para uso futuro
        self.df = df
        
        # Verificar si es necesario reentrenar el modelo
        retrain_metrics = None
        if self.should_retrain_model():
            logger.info("Reentrenando modelo de ML con datos actualizados...")
            retrain_metrics = self.ml_model.train(df)
            self.last_train_time = datetime.now()
            logger.info(f"Modelo reentrenado. Métricas: {retrain_metrics}")
        
        # Realizar predicción
        prediction = self.ml_model.predict(df)
        logger.info(f"Predicción ML para última vela: {prediction}")
        
        return prediction, retrain_metrics
        
    def get_trend_strength(self):
        """
        Calcula la fuerza de la tendencia actual basada en los datos disponibles.
        
        Returns:
            float: Valor entre -1 y 1 que indica la fuerza y dirección de la tendencia.
                  Valores positivos indican tendencia alcista, negativos tendencia bajista.
        """
        if not hasattr(self, 'df') or self.df is None or self.df.empty:
            return 0.0
            
        try:
            # Usar las últimas 20 velas para calcular la tendencia
            df_tail = self.df.tail(20)
            if len(df_tail) < 5:  # Necesitamos al menos 5 velas para un cálculo significativo
                return 0.0
                
            # Calcular la tendencia basada en la diferencia porcentual entre el primer y último precio
            first_price = df_tail['close'].iloc[0]
            last_price = df_tail['close'].iloc[-1]
            
            if first_price <= 0:
                return 0.0
                
            # Calcular el cambio porcentual y normalizar a un rango de -1 a 1
            percent_change = (last_price - first_price) / first_price
            
            # Limitar a un rango de -1 a 1
            trend_strength = max(-1.0, min(1.0, percent_change * 5))  # Multiplicar por 5 para amplificar cambios pequeños
            
            return trend_strength
        except Exception as e:
            logger.error(f"Error al calcular la fuerza de la tendencia: {str(e)}")
            return 0.0
    
    def get_volatility(self):
        """
        Calcula la volatilidad del mercado basada en los datos disponibles.
        
        Returns:
            float: Valor entre 0 y 1 que indica el nivel de volatilidad del mercado.
        """
        if not hasattr(self, 'df') or self.df is None or self.df.empty:
            return 0.5
            
        try:
            # Usar las últimas 20 velas para calcular la volatilidad
            df_tail = self.df.tail(20)
            if len(df_tail) < 5:  # Necesitamos al menos 5 velas para un cálculo significativo
                return 0.5
                
            # Calcular la volatilidad como el rango porcentual entre el máximo y mínimo
            highest_high = df_tail['high'].max()
            lowest_low = df_tail['low'].min()
            
            if lowest_low <= 0:
                return 0.5
                
            # Calcular el rango porcentual
            range_percent = (highest_high - lowest_low) / lowest_low
            
            # Normalizar a un rango de 0 a 1
            volatility = min(1.0, range_percent)
            
            return volatility
        except Exception as e:
            logger.error(f"Error al calcular la volatilidad: {str(e)}")
            return 0.5
    
    def get_rsi(self):
        """
        Obtiene el valor actual del RSI (Relative Strength Index).
        
        Returns:
            float: Valor del RSI entre 0 y 100.
        """
        if not hasattr(self, 'df') or self.df is None or self.df.empty or 'rsi' not in self.df.columns:
            return 50.0
            
        try:
            # Obtener el último valor de RSI
            last_rsi = self.df['rsi'].iloc[-1]
            
            # Asegurarse de que esté en el rango correcto
            if pd.isna(last_rsi):
                return 50.0
                
            return float(last_rsi)
        except Exception as e:
            logger.error(f"Error al obtener el RSI: {str(e)}")
            return 50.0
    
    def get_volume_change(self):
        """
        Calcula el cambio porcentual del volumen actual respecto al promedio.
        
        Returns:
            float: Valor entre -1 y 1 que indica el cambio de volumen.
        """
        if not hasattr(self, 'df') or self.df is None or self.df.empty or 'volume' not in self.df.columns:
            return 0.0
            
        try:
            # Usar las últimas 20 velas para calcular el cambio de volumen
            df_tail = self.df.tail(20)
            if len(df_tail) < 5:  # Necesitamos al menos 5 velas para un cálculo significativo
                return 0.0
                
            # Calcular el promedio de volumen de las últimas 19 velas (excluyendo la última)
            avg_volume = df_tail['volume'].iloc[:-1].mean()
            last_volume = df_tail['volume'].iloc[-1]
            
            if avg_volume <= 0:
                return 0.0
                
            # Calcular el cambio porcentual
            volume_change = (last_volume - avg_volume) / avg_volume
            
            # Limitar a un rango de -1 a 1
            volume_change = max(-1.0, min(1.0, volume_change))
            
            return volume_change
        except Exception as e:
            logger.error(f"Error al calcular el cambio de volumen: {str(e)}")
            return 0.0
            logger.error(f"Error al procesar datos con ML: {str(e)}")
            return None, None
    
    def save_state(self, file_path='sol_bot_15m_state.json'):
        """
        Guarda el estado actual de la estrategia en un archivo JSON.
        
        Args:
            file_path (str): Ruta del archivo donde guardar el estado.
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario.
        """
        try:
            # Preparar datos para guardar
            state = {
                'symbol': self.symbol,
                'position': self.position,
                'entry_price': self.entry_price,
                'position_size': self.position_size,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'current_balance': self.current_balance,
                'initial_balance': self.initial_balance,
                'trades': self.trades,
                'performance_metrics': self.performance_metrics,
                'saved_at': datetime.now().isoformat(),
                'use_ml': self.use_ml
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=4)
            
            logger.info(f"Estado guardado en {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error al guardar estado: {str(e)}")
            return False
    
    def load_state(self, file_path='sol_bot_15m_state.json'):
        """
        Carga el estado de la estrategia desde un archivo JSON.
        
        Args:
            file_path (str): Ruta del archivo desde donde cargar el estado.
            
        Returns:
            bool: True si se cargó correctamente, False en caso contrario.
        """
        if not os.path.exists(file_path):
            logger.warning(f"Archivo de estado {file_path} no encontrado.")
            return False
        
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            self.symbol = state['symbol']
            self.position = state['position']
            self.entry_price = state['entry_price']
            self.position_size = state['position_size']
            self.stop_loss = state['stop_loss']
            self.take_profit = state['take_profit']
            self.current_balance = state['current_balance']
            self.initial_balance = state['initial_balance']
            self.performance_metrics = state['performance_metrics']
            self.trades = state['trades']
            
            logger.info(f"Estado cargado desde {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error al cargar estado: {str(e)}")
            return False

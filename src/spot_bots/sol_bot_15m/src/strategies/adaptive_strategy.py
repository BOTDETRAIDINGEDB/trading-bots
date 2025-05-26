#!/usr/bin/env python3
"""
Estrategia adaptativa para el bot SOL
Implementa take profit dinámico, stop loss fijo y adaptación al capital disponible
"""

import logging
import json
import os
from datetime import datetime
import sys

# Importar utilidades
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.market_analyzer import MarketAnalyzer
from utils.capital_manager import CapitalManager
from models.ml_model import MLModel

logger = logging.getLogger(__name__)

class AdaptiveStrategy:
    """Estrategia de trading adaptativa con take profit dinámico y stop loss fijo."""
    
    def __init__(self, symbol, risk_per_trade=0.02, stop_loss_pct=0.06, 
                 take_profit_pct=0.04, max_trades=3, use_ml=True):
        """
        Inicializa la estrategia adaptativa.
        
        Args:
            symbol (str): Par de trading (ej. 'SOLUSDT').
            risk_per_trade (float): Porcentaje del capital a arriesgar por operación (0.02 = 2%).
            stop_loss_pct (float): Porcentaje de stop loss (0.06 = 6%).
            take_profit_pct (float): Porcentaje de take profit inicial (0.04 = 4%).
            max_trades (int): Número máximo de operaciones simultáneas.
            use_ml (bool): Si es True, utiliza el modelo de ML para las predicciones.
        """
        self.symbol = symbol
        self.risk_per_trade = risk_per_trade
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_trades = max_trades
        self.use_ml = use_ml
        
        # Inicializar analizador de mercado
        self.market_analyzer = MarketAnalyzer(symbol=symbol)
        
        # Inicializar gestor de capital
        self.capital_manager = CapitalManager(max_risk_pct=stop_loss_pct)
        
        # Inicializar el modelo de ML
        if self.use_ml:
            model_path = f"{symbol.lower()}_model.pkl"
            self.ml_model = MLModel(model_path=model_path)
            logger.info(f"Modelo de ML inicializado para {symbol} en {model_path}")
        else:
            self.ml_model = None
        
        # Configuración de take profit adaptativo
        self.adaptive_tp_config = {
            'min_tp': 0.02,  # 2% mínimo
            'max_tp': 0.15,  # 15% máximo
            'volatility_factor': 0.5,  # Influencia de la volatilidad
            'trend_factor': 0.5,  # Influencia de la tendencia
        }
        
        # Estado de la estrategia
        self.position = 0  # 0: sin posición, 1: long, -1: short
        self.entry_price = 0.0
        self.position_size = 0.0
        self.position_amount = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_stop = 0.0
        self.trailing_active = False
        self.trailing_percent = 0.015  # 1.5% de trailing stop
        self.highest_price = 0.0  # Para tracking del trailing stop
        self.trades = []
        self.current_balance = 0.0
        self.initial_balance = 0.0
        self.market_conditions = {}
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
        
        logger.info(f"Estrategia adaptativa inicializada para {symbol} con stop loss fijo de {stop_loss_pct*100}%")
    
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
    
    def update_market_conditions(self):
        """
        Actualiza las condiciones del mercado.
        
        Returns:
            dict: Condiciones actualizadas del mercado.
        """
        self.market_conditions = self.market_analyzer.analyze_market_conditions()
        logger.info(f"Condiciones de mercado actualizadas: volatilidad={self.market_conditions['volatility']:.4f}, tendencia={self.market_conditions['trend_strength']:.4f}")
        return self.market_conditions
    
    def calculate_adaptive_take_profit(self):
        """
        Calcula el take profit adaptativo basado en condiciones de mercado.
        
        Returns:
            float: Take profit adaptativo.
        """
        # Asegurar que tenemos condiciones de mercado actualizadas
        if not self.market_conditions:
            self.update_market_conditions()
        
        # Calcular take profit adaptativo
        volatility = self.market_conditions['volatility']
        trend_strength = self.market_conditions['trend_strength']
        
        # Ajustar take profit según volatilidad (mayor volatilidad = mayor TP)
        volatility_adjustment = (volatility - 0.5) * self.adaptive_tp_config['volatility_factor'] * 0.04
        
        # Ajustar take profit según tendencia (tendencia fuerte = mayor TP en dirección de la tendencia)
        trend_adjustment = trend_strength * self.adaptive_tp_config['trend_factor'] * 0.03
        
        # Calcular take profit adaptativo
        adaptive_tp = self.take_profit_pct + volatility_adjustment + trend_adjustment
        
        # Limitar a rango configurado
        adaptive_tp = max(self.adaptive_tp_config['min_tp'], 
                          min(self.adaptive_tp_config['max_tp'], adaptive_tp))
        
        logger.info(f"Take profit adaptativo: {adaptive_tp:.4f} (base: {self.take_profit_pct}, vol_adj: {volatility_adjustment:.4f}, trend_adj: {trend_adjustment:.4f})")
        
        return adaptive_tp
    
    def calculate_position_size(self, price):
        """
        Calcula el tamaño de la posición basado en el capital disponible.
        
        Args:
            price (float): Precio actual del activo.
            
        Returns:
            float: Tamaño de la posición en unidades del activo.
        """
        position_size, position_amount = self.capital_manager.calculate_position_size(
            available_balance=self.current_balance,
            price=price,
            stop_loss_pct=self.stop_loss_pct,
            performance_metrics=self.performance_metrics
        )
        
        self.position_amount = position_amount
        
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
        
        # Actualizar condiciones de mercado
        self.update_market_conditions()
        
        # Si estamos usando ML y tenemos una predicción, combinar con la señal técnica
        if self.use_ml and ml_prediction is not None:
            # Solo entrar si ambas señales coinciden o si la predicción ML es muy fuerte
            if ml_prediction == 1 and (signal == 1 or signal == 0):
                logger.info(f"Señal de entrada confirmada por ML: {ml_prediction}")
                return self._validate_trade_conditions(price, available_balance)
            elif signal == 1 and ml_prediction == 0:
                # Si la señal técnica es fuerte pero ML es neutral, entrar con menor confianza
                logger.info("Señal técnica positiva con ML neutral, procediendo con precaución")
                return self._validate_trade_conditions(price, available_balance)
            else:
                return False
        else:
            # Comportamiento original basado solo en señales técnicas
            # Si no hay señal de entrada, no hacer nada
            if signal == 0:
                return False
            
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
        
        # Calcular stop loss fijo
        self.stop_loss = price * (1 - self.stop_loss_pct)
        
        # Calcular take profit adaptativo
        adaptive_tp_pct = self.calculate_adaptive_take_profit()
        self.take_profit = price * (1 + adaptive_tp_pct)
        
        # Resetear variables de trailing stop
        self.trailing_stop = 0.0
        self.trailing_active = False
        self.highest_price = price
        
        # Actualizar posición
        self.position = 1  # Long
        
        # Crear registro de la operación
        trade_id = f"{self.symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        trade = {
            'id': trade_id,
            'symbol': self.symbol,
            'type': 'long',
            'entry_price': price,
            'entry_time': timestamp.isoformat(),
            'position_size': self.position_size,
            'position_amount': self.position_amount,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'adaptive_tp_pct': adaptive_tp_pct,
            'market_conditions': self.market_conditions,
            'status': 'open',
            'exit_price': 0.0,
            'exit_time': None,
            'profit_loss': 0.0,
            'profit_loss_pct': 0.0
        }
        
        # Añadir a la lista de operaciones
        self.trades.append(trade)
        
        logger.info(f"Entrada en operación: {trade_id} a precio {price} con tamaño {self.position_size}")
        logger.info(f"Stop Loss: {self.stop_loss} ({self.stop_loss_pct*100}%), Take Profit: {self.take_profit} ({adaptive_tp_pct*100}%)")
        
        return trade
    
    def should_exit_trade(self, price, current_time):
        """
        Determina si se debe salir de una operación.
        
        Args:
            price (float): Precio actual del activo.
            current_time (datetime): Timestamp actual.
            
        Returns:
            tuple: (bool, str) - (salir, razón)
        """
        if self.position == 0:
            return False, "Sin posición"
        
        # Actualizar trailing stop si el precio sube
        if price > self.highest_price:
            self.highest_price = price
            
            # Calcular nuevo trailing stop
            if not self.trailing_active and price >= self.take_profit:
                # Activar trailing stop cuando se alcanza el take profit
                self.trailing_active = True
                self.trailing_stop = price * (1 - self.trailing_percent)
                logger.info(f"Trailing Stop activado a precio {price} (TS: {self.trailing_stop})")
            elif self.trailing_active:
                # Actualizar trailing stop
                new_trailing_stop = price * (1 - self.trailing_percent)
                if new_trailing_stop > self.trailing_stop:
                    self.trailing_stop = new_trailing_stop
                    logger.info(f"Trailing Stop actualizado: {self.trailing_stop}")
        
        # Verificar condiciones de salida
        
        # 1. Stop Loss (fijo al 6%)
        if price <= self.stop_loss:
            logger.info(f"Stop Loss alcanzado a precio {price} (SL: {self.stop_loss})")
            return True, "stop_loss"
        
        # 2. Take Profit
        if price >= self.take_profit and not self.trailing_active:
            logger.info(f"Take Profit alcanzado a precio {price} (TP: {self.take_profit})")
            return True, "take_profit"
        
        # 3. Trailing Stop
        if self.trailing_active and price <= self.trailing_stop:
            logger.info(f"Trailing Stop alcanzado a precio {price} (TS: {self.trailing_stop})")
            return True, "trailing_stop"
        
        # 4. Actualizar Take Profit dinámicamente si las condiciones cambian significativamente
        # Solo actualizar si no está en trailing stop y cada 15 minutos
        last_trade = next((t for t in self.trades if t['status'] == 'open'), None)
        if last_trade and not self.trailing_active:
            entry_time = datetime.fromisoformat(last_trade['entry_time'].replace('Z', '+00:00'))
            time_in_trade = (current_time - entry_time).total_seconds() / 60
            
            # Actualizar TP cada 15 minutos o múltiplo
            if time_in_trade > 0 and time_in_trade % 15 < 1:
                # Actualizar condiciones de mercado
                self.update_market_conditions()
                
                # Calcular nuevo TP adaptativo
                new_tp_pct = self.calculate_adaptive_take_profit()
                new_tp = self.entry_price * (1 + new_tp_pct)
                
                # Solo actualizar si el nuevo TP es significativamente diferente
                if abs(new_tp - self.take_profit) / self.take_profit > 0.05:
                    old_tp = self.take_profit
                    self.take_profit = new_tp
                    last_trade['take_profit'] = new_tp
                    last_trade['adaptive_tp_pct'] = new_tp_pct
                    logger.info(f"Take Profit actualizado: {old_tp} -> {new_tp} ({new_tp_pct*100}%)")
        
        return False, "mantener"
    
    def exit_trade(self, price, timestamp, reason):
        """
        Sale de una operación.
        
        Args:
            price (float): Precio de salida.
            timestamp (datetime): Timestamp de la salida.
            reason (str): Razón de la salida.
            
        Returns:
            dict: Información de la operación cerrada.
        """
        if self.position == 0:
            logger.warning("Intento de salir de una operación sin posición")
            return None
        
        # Calcular profit/loss
        profit_loss = (price - self.entry_price) * self.position_size * self.position
        profit_loss_pct = (price / self.entry_price - 1) * 100 * self.position
        
        # Actualizar balance
        self.current_balance += profit_loss
        
        # Buscar la operación abierta
        for trade in self.trades:
            if trade['status'] == 'open':
                # Actualizar operación
                trade['exit_price'] = price
                trade['exit_time'] = timestamp.isoformat()
                trade['profit_loss'] = profit_loss
                trade['profit_loss_pct'] = profit_loss_pct
                trade['status'] = 'closed'
                trade['exit_reason'] = reason
                
                # Actualizar métricas de rendimiento
                self.update_performance_metrics(trade)
                
                logger.info(f"Salida de operación: {trade['id']} a precio {price} con P/L: {profit_loss:.4f} USDT ({profit_loss_pct:.2f}%)")
                logger.info(f"Razón de salida: {reason}")
                logger.info(f"Nuevo balance: {self.current_balance:.4f} USDT")
                
                closed_trade = trade.copy()
                break
        
        # Resetear estado
        self.position = 0
        self.entry_price = 0.0
        self.position_size = 0.0
        self.position_amount = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_stop = 0.0
        self.trailing_active = False
        self.highest_price = 0.0
        
        return closed_trade
    
    def update_performance_metrics(self, trade):
        """
        Actualiza las métricas de rendimiento.
        
        Args:
            trade (dict): Operación cerrada.
        """
        # Incrementar contador de operaciones
        self.performance_metrics['total_trades'] += 1
        
        # Actualizar contadores de operaciones ganadoras/perdedoras
        if trade['profit_loss'] > 0:
            self.performance_metrics['winning_trades'] += 1
            self.performance_metrics['avg_profit'] = ((self.performance_metrics['avg_profit'] * 
                                                     (self.performance_metrics['winning_trades'] - 1) + 
                                                     trade['profit_loss_pct']) / 
                                                     self.performance_metrics['winning_trades'])
        else:
            self.performance_metrics['losing_trades'] += 1
            self.performance_metrics['avg_loss'] = ((self.performance_metrics['avg_loss'] * 
                                                   (self.performance_metrics['losing_trades'] - 1) + 
                                                   abs(trade['profit_loss_pct'])) / 
                                                   self.performance_metrics['losing_trades'])
        
        # Calcular win rate
        self.performance_metrics['win_rate'] = (self.performance_metrics['winning_trades'] / 
                                              self.performance_metrics['total_trades'] * 100)
        
        # Calcular profit factor
        total_profit = self.performance_metrics['winning_trades'] * self.performance_metrics['avg_profit']
        total_loss = self.performance_metrics['losing_trades'] * self.performance_metrics['avg_loss']
        self.performance_metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Actualizar profit total
        self.performance_metrics['total_profit'] = self.current_balance - self.initial_balance
        
        # Calcular drawdown
        drawdown_pct = (1 - self.current_balance / max(self.current_balance, self.initial_balance)) * 100
        self.performance_metrics['current_drawdown'] = drawdown_pct
        self.performance_metrics['max_drawdown'] = max(self.performance_metrics['max_drawdown'], drawdown_pct)
        
        logger.info(f"Métricas actualizadas: Win Rate: {self.performance_metrics['win_rate']:.2f}%, " +
                   f"Profit Factor: {self.performance_metrics['profit_factor']:.2f}, " +
                   f"Total P/L: {self.performance_metrics['total_profit']:.4f} USDT")
    
    def save_state(self, filename):
        """
        Guarda el estado de la estrategia en un archivo.
        
        Args:
            filename (str): Nombre del archivo.
            
        Returns:
            bool: True si se guardó correctamente, False en caso contrario.
        """
        state = {
            'symbol': self.symbol,
            'risk_per_trade': self.risk_per_trade,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'max_trades': self.max_trades,
            'use_ml': self.use_ml,
            'position': self.position,
            'entry_price': self.entry_price,
            'position_size': self.position_size,
            'position_amount': self.position_amount,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop': self.trailing_stop,
            'trailing_active': self.trailing_active,
            'trailing_percent': self.trailing_percent,
            'highest_price': self.highest_price,
            'trades': self.trades,
            'current_balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'performance_metrics': self.performance_metrics,
            'market_conditions': self.market_conditions,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(state, f, indent=4)
            logger.info(f"Estado guardado en {filename}")
            return True
        except Exception as e:
            logger.error(f"Error al guardar estado: {str(e)}")
            return False
    
    def load_state(self, filename):
        """
        Carga el estado de la estrategia desde un archivo.
        
        Args:
            filename (str): Nombre del archivo.
            
        Returns:
            bool: True si se cargó correctamente, False en caso contrario.
        """
        try:
            if not os.path.exists(filename):
                logger.warning(f"Archivo de estado no encontrado: {filename}")
                return False
            
            with open(filename, 'r') as f:
                state = json.load(f)
            
            # Restaurar estado
            self.symbol = state.get('symbol', self.symbol)
            self.risk_per_trade = state.get('risk_per_trade', self.risk_per_trade)
            self.stop_loss_pct = state.get('stop_loss_pct', self.stop_loss_pct)
            self.take_profit_pct = state.get('take_profit_pct', self.take_profit_pct)
            self.max_trades = state.get('max_trades', self.max_trades)
            self.use_ml = state.get('use_ml', self.use_ml)
            self.position = state.get('position', self.position)
            self.entry_price = state.get('entry_price', self.entry_price)
            self.position_size = state.get('position_size', self.position_size)
            self.position_amount = state.get('position_amount', self.position_amount)
            self.stop_loss = state.get('stop_loss', self.stop_loss)
            self.take_profit = state.get('take_profit', self.take_profit)
            self.trailing_stop = state.get('trailing_stop', self.trailing_stop)
            self.trailing_active = state.get('trailing_active', self.trailing_active)
            self.trailing_percent = state.get('trailing_percent', self.trailing_percent)
            self.highest_price = state.get('highest_price', self.highest_price)
            self.trades = state.get('trades', self.trades)
            self.current_balance = state.get('current_balance', self.current_balance)
            self.initial_balance = state.get('initial_balance', self.initial_balance)
            self.performance_metrics = state.get('performance_metrics', self.performance_metrics)
            self.market_conditions = state.get('market_conditions', self.market_conditions)
            
            logger.info(f"Estado cargado desde {filename}")
            return True
        except Exception as e:
            logger.error(f"Error al cargar estado: {str(e)}")
            return False

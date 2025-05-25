# src/strategies/ml_strategy.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MLTradingStrategy:
    def __init__(self, model, threshold=0.65, risk_per_trade=0.015, stop_loss_percent=0.06):
        """
        Inicializa la estrategia de trading basada en ML.
        
        Args:
            model: Modelo de ML entrenado
            threshold (float): Umbral de probabilidad para entrar en operaciones
            risk_per_trade (float): Porcentaje del capital a arriesgar por operación
            stop_loss_percent (float): Porcentaje de stop loss
        """
        self.model = model
        self.threshold = threshold
        self.risk_per_trade = risk_per_trade
        self.stop_loss_percent = stop_loss_percent
        self.positions = {}  # Para rastrear posiciones abiertas
        self.trade_history = []  # Para rastrear historial de operaciones
    
    def generate_signals(self, X_recent, current_prices):
        """
        Genera señales de trading basadas en predicciones del modelo.
        
        Args:
            X_recent (numpy.ndarray): Datos recientes para predicción
            current_prices (dict): Precios actuales de los símbolos
            
        Returns:
            list: Lista de señales de trading (dict)
        """
        predictions = self.model.predict(X_recent)
        signals = []
        
        for i, prob in enumerate(predictions):
            symbol = list(current_prices.keys())[i]
            price = current_prices[symbol]
            
            signal = {
                'symbol': symbol,
                'price': price,
                'timestamp': datetime.now(),
                'probability': float(prob[0]),
                'action': None
            }
            
            # Determinar acción basada en la probabilidad
            if prob[0] > self.threshold:
                signal['action'] = 'BUY'
            elif prob[0] < (1 - self.threshold):
                signal['action'] = 'SELL'
            else:
                signal['action'] = 'HOLD'
            
            signals.append(signal)
            logger.info(f"Señal generada para {symbol}: {signal['action']} (prob: {prob[0]:.4f})")
        
        return signals
    
    def calculate_position_size(self, capital, price, stop_loss_percent):
        """
        Calcula el tamaño de la posición basado en el riesgo.
        
        Args:
            capital (float): Capital disponible
            price (float): Precio actual
            stop_loss_percent (float): Porcentaje de stop loss
            
        Returns:
            float: Cantidad a comprar/vender
        """
        # Usar un porcentaje del capital disponible (máximo 90%)
        usable_capital = capital * 0.9
        
        # Calcular el riesgo basado en el capital disponible
        risk_amount = usable_capital * self.risk_per_trade
        stop_loss_amount = price * stop_loss_percent
        
        # Calcular el tamaño de la posición
        position_size = risk_amount / stop_loss_amount
        
        # Asegurarse de que el valor total no exceda el capital usable
        position_value = position_size * price
        if position_value > usable_capital:
            position_size = usable_capital / price
        
        # Redondear a 6 decimales para evitar errores de precisión
        return round(position_size, 6)
    
    def execute_signals(self, signals, api_client, account_balance):
        """
        Ejecuta las señales de trading.
        
        Args:
            signals (list): Lista de señales de trading
            api_client: Cliente de API para ejecutar órdenes
            account_balance (dict): Balance de la cuenta
            
        Returns:
            list: Lista de órdenes ejecutadas
        """
        executed_orders = []
        
        # Primero verificar stop loss y take profit para posiciones existentes
        sl_tp_orders = self.check_stop_loss_take_profit(api_client, {s['symbol']: s['price'] for s in signals})
        executed_orders.extend(sl_tp_orders)
        
        for signal in signals:
            symbol = signal['symbol']
            action = signal['action']
            price = signal['price']
            
            # Ignorar señales HOLD
            if action == 'HOLD':
                continue
            
            # Determinar si ya tenemos una posición abierta
            has_position = symbol in self.positions
            
            # Lógica para entrar o salir de posiciones
            if action == 'BUY' and not has_position:
                # Calcular tamaño de posición (usando el stop loss configurado)
                base_asset = symbol.replace('USDT', '')
                capital = account_balance.get('USDT', 0)
                
                if capital < 10:  # Establecer un mínimo de 10 USDT para operar
                    logger.warning(f"No hay suficiente USDT para comprar {symbol}, mínimo requerido: 10 USDT")
                    continue
                
                # Calcular un tamaño de posición adaptativo basado en el capital disponible
                position_size = self.calculate_position_size(capital, price, self.stop_loss_percent)
                
                # Ejecutar orden
                order = api_client.place_order(
                    symbol=symbol,
                    side='BUY',
                    order_type='MARKET',
                    quantity=position_size
                )
                
                if order:
                    # Obtener ATR actual para cálculos dinámicos
                    atr_value = getattr(api_client, 'get_current_atr', lambda x: price * 0.02)(symbol)
                    
                    # Calcular take profit dinámico basado en ATR
                    take_profit_price = price + (atr_value * 2.5)
                    
                    self.positions[symbol] = {
                        'entry_price': price,
                        'quantity': position_size,
                        'entry_time': datetime.now(),
                        'stop_loss_price': price * (1 - self.stop_loss_percent),
                        'take_profit_price': take_profit_price,
                        'highest_price': price,
                        'atr_at_entry': atr_value
                    }
                    executed_orders.append(order)
                    
                    logger.info(f"Posición abierta en {symbol} con stop loss a {price * (1 - self.stop_loss_percent):.4f} y take profit a {take_profit_price:.4f}")
                    
                    # Registrar operación en historial
                    self.trade_history.append({
                        'symbol': symbol,
                        'action': 'BUY',
                        'price': price,
                        'quantity': position_size,
                        'timestamp': datetime.now()
                    })
            
            elif action == 'SELL' and has_position:
                # Vender posición existente
                order = self._close_position(symbol, self.positions[symbol], api_client, price, 'SIGNAL')
                if order:
                    executed_orders.append(order)
        
        # Ajustar parámetros basándose en el rendimiento
        self.adjust_parameters_based_on_performance()
        
        return executed_orders
    
    def check_stop_loss_take_profit(self, api_client, current_prices):
        """
        Verifica si alguna posición ha alcanzado su nivel de stop loss o take profit.
        
        Args:
            api_client: Cliente de API para ejecutar órdenes
            current_prices (dict): Precios actuales de los símbolos
        
        Returns:
            list: Órdenes ejecutadas
        """
        executed_orders = []
        
        for symbol, position in list(self.positions.items()):
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            entry_price = position['entry_price']
            
            # Verificar stop loss (6% por debajo del precio de entrada)
            stop_loss_price = position.get('stop_loss_price', entry_price * (1 - self.stop_loss_percent))
            
            # Verificar take profit dinámico
            atr = position.get('atr_at_entry', entry_price * 0.02)  # Valor predeterminado si no hay ATR
            take_profit_price = position.get('take_profit_price', entry_price * 1.1)  # Valor predeterminado
            
            # Actualizar trailing take profit si el precio ha subido
            if current_price > position.get('highest_price', entry_price):
                position['highest_price'] = current_price
                # Si el precio ha subido más del 4%, activar trailing take profit
                if current_price > entry_price * 1.04:
                    new_take_profit = current_price - (atr * 1.5)
                    # Solo actualizar si el nuevo take profit es mayor que el anterior
                    if new_take_profit > position.get('take_profit_price', 0):
                        position['take_profit_price'] = new_take_profit
                        logger.info(f"Trailing take profit actualizado para {symbol}: {new_take_profit:.4f}")
            
            # Verificar si se ha alcanzado el stop loss
            if current_price <= stop_loss_price:
                logger.info(f"Stop loss activado para {symbol} a {current_price:.4f} (entrada: {entry_price:.4f})")
                order = self._close_position(symbol, position, api_client, current_price, 'STOP_LOSS')
                if order:
                    executed_orders.append(order)
            
            # Verificar si se ha alcanzado el take profit
            elif current_price >= take_profit_price:
                logger.info(f"Take profit activado para {symbol} a {current_price:.4f} (entrada: {entry_price:.4f})")
                order = self._close_position(symbol, position, api_client, current_price, 'TAKE_PROFIT')
                if order:
                    executed_orders.append(order)
        
        return executed_orders

    def _close_position(self, symbol, position, api_client, price, reason):
        """
        Cierra una posición y registra la operación.
        
        Args:
            symbol (str): Símbolo de la posición
            position (dict): Datos de la posición
            api_client: Cliente de API para ejecutar órdenes
            price (float): Precio actual
            reason (str): Razón del cierre (STOP_LOSS, TAKE_PROFIT, SIGNAL)
        
        Returns:
            dict: Orden ejecutada
        """
        order = api_client.place_order(
            symbol=symbol,
            side='SELL',
            order_type='MARKET',
            quantity=position['quantity']
        )
        
        if order:
            # Calcular P&L
            entry_price = position['entry_price']
            exit_price = price
            quantity = position['quantity']
            pnl = (exit_price - entry_price) * quantity
            
            logger.info(f"Cerrando posición en {symbol} por {reason}: P&L = {pnl:.2f} USDT")
            
            # Eliminar de posiciones activas
            del self.positions[symbol]
            
            # Registrar operación en historial
            self.trade_history.append({
                'symbol': symbol,
                'action': 'SELL',
                'price': price,
                'quantity': quantity,
                'pnl': pnl,
                'reason': reason,
                'timestamp': datetime.now()
            })
        
        return order
    
    def adjust_parameters_based_on_performance(self):
        """
        Ajusta los parámetros de la estrategia basándose en el rendimiento pasado.
        """
        metrics = self.get_performance_metrics()
        
        # Solo ajustar si tenemos suficientes operaciones
        if metrics['total_trades'] < 5:
            logger.info("No hay suficientes operaciones para ajustar parámetros")
            return
        
        # Ajustar umbral de confianza basado en la tasa de aciertos
        if metrics['win_rate'] < 0.4:  # Si la tasa de aciertos es baja
            # Aumentar el umbral para ser más selectivo
            new_threshold = min(self.threshold + 0.02, 0.50)
            logger.info(f"Ajustando umbral de confianza de {self.threshold} a {new_threshold} debido a baja tasa de aciertos")
            self.threshold = new_threshold
        elif metrics['win_rate'] > 0.7:  # Si la tasa de aciertos es alta
            # Podemos ser menos selectivos para aumentar el número de operaciones
            new_threshold = max(self.threshold - 0.01, 0.45)
            logger.info(f"Ajustando umbral de confianza de {self.threshold} a {new_threshold} debido a alta tasa de aciertos")
            self.threshold = new_threshold
        
        # Ajustar riesgo por operación basado en el profit factor
        if metrics['profit_factor'] < 1.0:  # Si estamos perdiendo dinero
            # Reducir el riesgo
            new_risk = max(self.risk_per_trade * 0.9, 0.005)
            logger.info(f"Reduciendo riesgo por operación de {self.risk_per_trade} a {new_risk} debido a bajo profit factor")
            self.risk_per_trade = new_risk
        elif metrics['profit_factor'] > 2.0:  # Si estamos ganando consistentemente
            # Podemos aumentar ligeramente el riesgo
            new_risk = min(self.risk_per_trade * 1.05, 0.03)
            logger.info(f"Aumentando riesgo por operación de {self.risk_per_trade} a {new_risk} debido a alto profit factor")
            self.risk_per_trade = new_risk
    
    def get_performance_metrics(self):
        """
        Calcula métricas de rendimiento basadas en el historial de operaciones.
        
        Returns:
            dict: Métricas de rendimiento
        """
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_pnl': 0
            }
        
        # Filtrar operaciones cerradas (con P&L)
        closed_trades = [trade for trade in self.trade_history if 'pnl' in trade]
        
        if not closed_trades:
            return {
                'total_trades': len(self.trade_history),
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'total_pnl': 0
            }
        
        # Calcular métricas
        total_trades = len(closed_trades)
        winning_trades = [trade for trade in closed_trades if trade['pnl'] > 0]
        losing_trades = [trade for trade in closed_trades if trade['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_profit = sum(trade['pnl'] for trade in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(trade['pnl'] for trade in losing_trades) / len(losing_trades) if losing_trades else 0
        
        total_profit = sum(trade['pnl'] for trade in winning_trades)
        total_loss = abs(sum(trade['pnl'] for trade in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        total_pnl = total_profit - total_loss
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_pnl': total_pnl
        }

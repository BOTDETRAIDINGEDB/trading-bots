#!/usr/bin/env python3
"""
Script para realizar backtesting del modelo de aprendizaje profundo.
Evalúa el rendimiento del modelo en datos históricos.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Ajustar el PYTHONPATH para encontrar los módulos correctamente
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Importar módulos propios
from models.deep_learning.data_loader import MultiTimeframeDataLoader
from models.deep_learning.data_processor import DeepLearningDataProcessor
from models.deep_learning.lstm_model import DeepTimeSeriesModel
from models.deep_learning.deep_learning_integration import DeepLearningIntegration

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("deep_learning_backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeepModelBacktester:
    """
    Backtester para modelos de aprendizaje profundo.
    Evalúa el rendimiento del modelo en datos históricos.
    """
    
    def __init__(self, 
                 symbol: str = 'SOL/USDT',
                 timeframes: List[str] = ['5m', '15m', '1h', '4h'],
                 base_timeframe: str = '15m',
                 model_type: str = 'lstm',
                 test_days: int = 30,
                 output_dir: str = 'backtest_results'):
        """
        Inicializa el backtester.
        
        Args:
            symbol: Par de trading
            timeframes: Lista de intervalos temporales a utilizar
            base_timeframe: Timeframe base para sincronización
            model_type: Tipo de modelo ('lstm', 'gru', 'bilstm', 'attention')
            test_days: Días de histórico para pruebas
            output_dir: Directorio para guardar resultados
        """
        self.symbol = symbol
        self.timeframes = timeframes
        self.base_timeframe = base_timeframe
        self.model_type = model_type
        self.test_days = test_days
        self.output_dir = output_dir
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Inicializar componentes
        self.data_loader = MultiTimeframeDataLoader(
            symbol=symbol,
            timeframes=timeframes,
            lookback_days=test_days + 10  # Añadir margen para secuencias
        )
        
        self.dl_integration = DeepLearningIntegration(
            symbol=symbol,
            timeframes=timeframes,
            base_timeframe=base_timeframe,
            model_type=model_type
        )
        
        # Resultados de backtesting
        self.results = {
            'trades': [],
            'metrics': {},
            'equity_curve': []
        }
        
        logger.info(f"Backtester inicializado para {symbol} con modelo {model_type}")
    
    def run_backtest(self, 
                    initial_balance: float = 1000.0,
                    position_size_pct: float = 0.1,
                    take_profit_pct: float = 0.03,
                    stop_loss_pct: float = 0.06,
                    use_ml_enhancement: bool = True,
                    risk_level: float = 0.5) -> Dict:
        """
        Ejecuta el backtesting en datos históricos.
        
        Args:
            initial_balance: Balance inicial en USDT
            position_size_pct: Porcentaje del balance a usar por operación
            take_profit_pct: Porcentaje de take profit
            stop_loss_pct: Porcentaje de stop loss
            use_ml_enhancement: Si es True, usa mejora de ML para decisiones
            risk_level: Nivel de riesgo (0-1) para integración de ML
            
        Returns:
            Diccionario con resultados de backtesting
        """
        try:
            # Cargar datos históricos
            logger.info("Cargando datos históricos...")
            data_dict = self.data_loader.load_all_timeframes(force_update=True)
            
            if not data_dict or self.base_timeframe not in data_dict:
                logger.error(f"No se pudieron cargar datos para {self.base_timeframe}")
                return {}
            
            # Obtener datos del timeframe base
            df = data_dict[self.base_timeframe]
            
            # Limitar a los días de prueba
            start_date = df.index[-1] - timedelta(days=self.test_days)
            test_df = df[df.index >= start_date].copy()
            
            logger.info(f"Datos de prueba: {len(test_df)} registros desde {start_date}")
            
            # Inicializar variables de trading
            balance = initial_balance
            position = None
            position_size = 0
            entry_price = 0
            trades = []
            equity_curve = [{'timestamp': test_df.index[0], 'equity': balance}]
            
            # Para cada vela en el período de prueba
            for i in range(60, len(test_df)):  # Empezar después de suficientes datos para secuencia
                current_time = test_df.index[i]
                current_price = test_df['close'].iloc[i]
                
                # Obtener datos hasta el momento actual (sin mirar al futuro)
                current_df = test_df.iloc[:i+1].copy()
                
                # Generar señal técnica (simplificada para el ejemplo)
                technical_signal = self._generate_technical_signal(current_df)
                
                # Mejorar decisión con ML si está habilitado
                if use_ml_enhancement:
                    decision = self.dl_integration.enhance_trading_decision(
                        technical_signal=technical_signal,
                        market_data=current_df,
                        risk_level=risk_level
                    )
                    
                    if decision:
                        signal = decision.get('final_signal', 0)
                    else:
                        signal = technical_signal
                else:
                    signal = technical_signal
                
                # Procesar señal
                if position is None:  # Sin posición abierta
                    if signal == 1:  # Señal de compra
                        # Calcular tamaño de posición
                        position_size = balance * position_size_pct
                        entry_price = current_price
                        position = 'long'
                        
                        logger.info(f"[{current_time}] COMPRA en {entry_price}, tamaño: {position_size:.2f} USDT")
                        
                        # Registrar operación
                        trades.append({
                            'entry_time': current_time,
                            'entry_price': entry_price,
                            'position': position,
                            'position_size': position_size,
                            'balance_before': balance
                        })
                
                elif position == 'long':  # Posición larga abierta
                    # Calcular take profit y stop loss
                    take_profit = entry_price * (1 + take_profit_pct)
                    stop_loss = entry_price * (1 - stop_loss_pct)
                    
                    # Verificar si se alcanzó take profit o stop loss
                    if current_price >= take_profit:  # Take profit
                        # Calcular ganancia
                        qty = position_size / entry_price
                        exit_value = qty * current_price
                        profit = exit_value - position_size
                        balance += profit
                        
                        logger.info(f"[{current_time}] TAKE PROFIT en {current_price}, ganancia: {profit:.2f} USDT")
                        
                        # Actualizar última operación
                        trades[-1].update({
                            'exit_time': current_time,
                            'exit_price': current_price,
                            'exit_reason': 'take_profit',
                            'profit': profit,
                            'profit_pct': (current_price / entry_price - 1) * 100,
                            'balance_after': balance
                        })
                        
                        # Resetear posición
                        position = None
                        position_size = 0
                        entry_price = 0
                        
                    elif current_price <= stop_loss:  # Stop loss
                        # Calcular pérdida
                        qty = position_size / entry_price
                        exit_value = qty * current_price
                        loss = exit_value - position_size
                        balance += loss
                        
                        logger.info(f"[{current_time}] STOP LOSS en {current_price}, pérdida: {loss:.2f} USDT")
                        
                        # Actualizar última operación
                        trades[-1].update({
                            'exit_time': current_time,
                            'exit_price': current_price,
                            'exit_reason': 'stop_loss',
                            'profit': loss,
                            'profit_pct': (current_price / entry_price - 1) * 100,
                            'balance_after': balance
                        })
                        
                        # Resetear posición
                        position = None
                        position_size = 0
                        entry_price = 0
                    
                    elif signal == -1:  # Señal de venta
                        # Calcular resultado
                        qty = position_size / entry_price
                        exit_value = qty * current_price
                        result = exit_value - position_size
                        balance += result
                        
                        logger.info(f"[{current_time}] VENTA en {current_price}, resultado: {result:.2f} USDT")
                        
                        # Actualizar última operación
                        trades[-1].update({
                            'exit_time': current_time,
                            'exit_price': current_price,
                            'exit_reason': 'signal',
                            'profit': result,
                            'profit_pct': (current_price / entry_price - 1) * 100,
                            'balance_after': balance
                        })
                        
                        # Resetear posición
                        position = None
                        position_size = 0
                        entry_price = 0
                
                # Actualizar curva de equity
                if i % 10 == 0 or position is None:  # Actualizar cada 10 velas o al cerrar posición
                    # Calcular equity actual (balance + valor de posición abierta)
                    if position == 'long':
                        qty = position_size / entry_price
                        position_value = qty * current_price
                        current_equity = balance + (position_value - position_size)
                    else:
                        current_equity = balance
                    
                    equity_curve.append({
                        'timestamp': current_time,
                        'equity': current_equity
                    })
            
            # Cerrar posición abierta al final del período
            if position == 'long':
                # Calcular resultado
                qty = position_size / entry_price
                exit_value = qty * current_price
                result = exit_value - position_size
                balance += result
                
                logger.info(f"[{test_df.index[-1]}] CIERRE FINAL en {current_price}, resultado: {result:.2f} USDT")
                
                # Actualizar última operación
                trades[-1].update({
                    'exit_time': test_df.index[-1],
                    'exit_price': current_price,
                    'exit_reason': 'end_of_period',
                    'profit': result,
                    'profit_pct': (current_price / entry_price - 1) * 100,
                    'balance_after': balance
                })
            
            # Calcular métricas
            metrics = self._calculate_metrics(trades, initial_balance, balance)
            
            # Guardar resultados
            self.results = {
                'trades': trades,
                'metrics': metrics,
                'equity_curve': equity_curve,
                'params': {
                    'initial_balance': initial_balance,
                    'position_size_pct': position_size_pct,
                    'take_profit_pct': take_profit_pct,
                    'stop_loss_pct': stop_loss_pct,
                    'use_ml_enhancement': use_ml_enhancement,
                    'risk_level': risk_level
                }
            }
            
            # Generar gráficos
            self._generate_backtest_plots()
            
            # Guardar resultados en JSON
            self._save_results()
            
            logger.info(f"Backtesting completado. Balance final: {balance:.2f} USDT")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error al ejecutar backtesting: {str(e)}")
            import traceback
            logger.debug(f"Traceback completo: {traceback.format_exc()}")
            return {}
    
    def _generate_technical_signal(self, df: pd.DataFrame) -> int:
        """
        Genera señal técnica simplificada para backtesting.
        
        Args:
            df: DataFrame con datos de mercado
            
        Returns:
            Señal (1: compra, -1: venta, 0: mantener)
        """
        try:
            # Calcular medias móviles
            df['ema9'] = df['close'].ewm(span=9, adjust=False).mean()
            df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
            
            # Obtener valores actuales
            current = df.iloc[-1]
            previous = df.iloc[-2]
            
            # Señal de cruce de medias móviles
            if previous['ema9'] <= previous['ema21'] and current['ema9'] > current['ema21']:
                return 1  # Señal de compra
            elif previous['ema9'] >= previous['ema21'] and current['ema9'] < current['ema21']:
                return -1  # Señal de venta
            
            return 0  # Sin señal
            
        except Exception as e:
            logger.error(f"Error al generar señal técnica: {str(e)}")
            return 0
    
    def _calculate_metrics(self, trades: List[Dict], initial_balance: float, final_balance: float) -> Dict:
        """
        Calcula métricas de rendimiento del backtesting.
        
        Args:
            trades: Lista de operaciones
            initial_balance: Balance inicial
            final_balance: Balance final
            
        Returns:
            Diccionario con métricas
        """
        # Filtrar operaciones cerradas
        closed_trades = [t for t in trades if 'exit_time' in t]
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'profitable_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_profit': 0,
                'max_drawdown': 0,
                'return_pct': 0
            }
        
        # Calcular métricas básicas
        total_trades = len(closed_trades)
        profitable_trades = sum(1 for t in closed_trades if t.get('profit', 0) > 0)
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # Calcular ganancias y pérdidas
        total_profit = sum(t.get('profit', 0) for t in closed_trades)
        winning_trades = [t for t in closed_trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in closed_trades if t.get('profit', 0) <= 0]
        
        total_gains = sum(t.get('profit', 0) for t in winning_trades)
        total_losses = abs(sum(t.get('profit', 0) for t in losing_trades))
        
        profit_factor = total_gains / total_losses if total_losses > 0 else float('inf')
        
        # Calcular drawdown
        equity_curve = []
        current_equity = initial_balance
        
        for trade in closed_trades:
            current_equity += trade.get('profit', 0)
            equity_curve.append(current_equity)
        
        max_equity = initial_balance
        max_drawdown = 0
        
        for equity in equity_curve:
            max_equity = max(max_equity, equity)
            drawdown = (max_equity - equity) / max_equity
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calcular retorno
        return_pct = (final_balance / initial_balance - 1) * 100
        
        return {
            'total_trades': total_trades,
            'profitable_trades': profitable_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_profit': total_profit,
            'max_drawdown': max_drawdown,
            'return_pct': return_pct,
            'initial_balance': initial_balance,
            'final_balance': final_balance
        }
    
    def _generate_backtest_plots(self):
        """
        Genera gráficos de resultados del backtesting.
        """
        try:
            if not self.results.get('trades'):
                logger.warning("No hay datos suficientes para generar gráficos")
                return
            
            # Crear directorio para gráficos
            plots_dir = os.path.join(self.output_dir, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            
            # Convertir curva de equity a DataFrame
            equity_df = pd.DataFrame(self.results['equity_curve'])
            equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
            equity_df.set_index('timestamp', inplace=True)
            
            # Gráfico de curva de equity
            plt.figure(figsize=(12, 6))
            plt.plot(equity_df.index, equity_df['equity'])
            plt.title(f'Curva de Equity - {self.symbol} ({self.model_type})')
            plt.xlabel('Fecha')
            plt.ylabel('Equity (USDT)')
            plt.grid(True)
            
            # Guardar gráfico
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = os.path.join(plots_dir, f"equity_curve_{self.model_type}_{timestamp}.png")
            plt.tight_layout()
            plt.savefig(plot_file)
            plt.close()
            
            logger.info(f"Gráfico de curva de equity guardado en {plot_file}")
            
            # Gráfico de distribución de ganancias/pérdidas
            profits = [t.get('profit', 0) for t in self.results['trades'] if 'exit_time' in t]
            
            if profits:
                plt.figure(figsize=(10, 6))
                plt.hist(profits, bins=20, alpha=0.7)
                plt.axvline(x=0, color='r', linestyle='--')
                plt.title(f'Distribución de Ganancias/Pérdidas - {self.symbol} ({self.model_type})')
                plt.xlabel('Ganancia/Pérdida (USDT)')
                plt.ylabel('Frecuencia')
                plt.grid(True)
                
                # Guardar gráfico
                plot_file = os.path.join(plots_dir, f"profit_distribution_{self.model_type}_{timestamp}.png")
                plt.tight_layout()
                plt.savefig(plot_file)
                plt.close()
                
                logger.info(f"Gráfico de distribución de ganancias guardado en {plot_file}")
            
        except Exception as e:
            logger.error(f"Error al generar gráficos de backtesting: {str(e)}")
    
    def _save_results(self):
        """
        Guarda resultados del backtesting en archivo JSON.
        """
        try:
            # Crear nombre de archivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_file = os.path.join(self.output_dir, f"backtest_results_{self.model_type}_{timestamp}.json")
            
            # Convertir timestamps a strings para serialización JSON
            results_copy = self.results.copy()
            
            for trade in results_copy['trades']:
                if 'entry_time' in trade:
                    trade['entry_time'] = str(trade['entry_time'])
                if 'exit_time' in trade:
                    trade['exit_time'] = str(trade['exit_time'])
            
            for point in results_copy['equity_curve']:
                if 'timestamp' in point:
                    point['timestamp'] = str(point['timestamp'])
            
            # Guardar en JSON
            with open(result_file, 'w') as f:
                json.dump(results_copy, f, indent=4)
            
            logger.info(f"Resultados guardados en {result_file}")
            
        except Exception as e:
            logger.error(f"Error al guardar resultados: {str(e)}")

def parse_arguments():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description='Backtesting de modelos de aprendizaje profundo')
    
    # Argumentos generales
    parser.add_argument('--symbol', type=str, default='SOL/USDT', help='Par de trading')
    parser.add_argument('--timeframes', type=str, nargs='+', default=['5m', '15m', '1h', '4h'], 
                        help='Lista de timeframes a utilizar')
    parser.add_argument('--model-type', type=str, default='lstm', 
                        choices=['lstm', 'gru', 'bilstm', 'attention'],
                        help='Tipo de modelo a utilizar')
    parser.add_argument('--test-days', type=int, default=30, help='Días de histórico para pruebas')
    
    # Parámetros de trading
    parser.add_argument('--initial-balance', type=float, default=1000.0, help='Balance inicial en USDT')
    parser.add_argument('--position-size', type=float, default=0.1, help='Porcentaje del balance por operación')
    parser.add_argument('--take-profit', type=float, default=0.03, help='Porcentaje de take profit')
    parser.add_argument('--stop-loss', type=float, default=0.06, help='Porcentaje de stop loss')
    parser.add_argument('--risk-level', type=float, default=0.5, help='Nivel de riesgo para ML (0-1)')
    
    # Opciones adicionales
    parser.add_argument('--no-ml', action='store_true', help='Desactivar mejora de ML')
    parser.add_argument('--output-dir', type=str, default='backtest_results', help='Directorio de salida')
    
    return parser.parse_args()

def main():
    """Función principal."""
    # Parsear argumentos
    args = parse_arguments()
    
    # Inicializar backtester
    backtester = DeepModelBacktester(
        symbol=args.symbol,
        timeframes=args.timeframes,
        model_type=args.model_type,
        test_days=args.test_days,
        output_dir=args.output_dir
    )
    
    # Ejecutar backtesting
    results = backtester.run_backtest(
        initial_balance=args.initial_balance,
        position_size_pct=args.position_size,
        take_profit_pct=args.take_profit,
        stop_loss_pct=args.stop_loss,
        use_ml_enhancement=not args.no_ml,
        risk_level=args.risk_level
    )
    
    # Mostrar resultados
    if results and 'metrics' in results:
        metrics = results['metrics']
        
        print("\n" + "="*50)
        print(f"RESULTADOS DE BACKTESTING - {args.symbol} ({args.model_type})")
        print("="*50)
        print(f"Balance Inicial: {metrics['initial_balance']:.2f} USDT")
        print(f"Balance Final: {metrics['final_balance']:.2f} USDT")
        print(f"Retorno: {metrics['return_pct']:.2f}%")
        print(f"Operaciones Totales: {metrics['total_trades']}")
        print(f"Operaciones Rentables: {metrics['profitable_trades']}")
        print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Máximo Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print("="*50)
        
        # Mostrar ruta a resultados
        print(f"\nResultados detallados guardados en: {args.output_dir}")
    else:
        print("\nNo se pudieron generar resultados de backtesting.")

if __name__ == "__main__":
    main()

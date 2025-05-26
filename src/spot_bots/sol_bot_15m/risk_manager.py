#!/usr/bin/env python3
"""
Gestor de riesgos para el bot SOL
Ajusta automáticamente los parámetros de riesgo basado en el rendimiento
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
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
        logging.FileHandler("risk_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RiskManager:
    """Gestor de riesgos para el bot SOL."""
    
    def __init__(self, state_file, config_file='risk_config.json'):
        """
        Inicializa el gestor de riesgos.
        
        Args:
            state_file (str): Ruta al archivo de estado del bot.
            config_file (str): Ruta al archivo de configuración de riesgos.
        """
        self.state_file = state_file
        self.config_file = config_file
        self.telegram = TelegramNotifier()
        
        # Cargar configuración de riesgos
        self.config = self.load_config()
        if not self.config:
            # Configuración por defecto
            self.config = {
                'risk_levels': {
                    'low': {
                        'risk_per_trade': 0.01,
                        'stop_loss_pct': 0.02,
                        'take_profit_pct': 0.04,
                        'max_trades': 2
                    },
                    'medium': {
                        'risk_per_trade': 0.02,
                        'stop_loss_pct': 0.025,
                        'take_profit_pct': 0.05,
                        'max_trades': 3
                    },
                    'high': {
                        'risk_per_trade': 0.03,
                        'stop_loss_pct': 0.03,
                        'take_profit_pct': 0.06,
                        'max_trades': 4
                    }
                },
                'thresholds': {
                    'win_rate_low': 40,
                    'win_rate_medium': 55,
                    'profit_factor_low': 1.0,
                    'profit_factor_medium': 1.5,
                    'drawdown_high': 15,
                    'drawdown_medium': 10
                },
                'current_level': 'medium',
                'auto_adjust': True,
                'last_adjustment': None
            }
            self.save_config()
        
        logger.info(f"Gestor de riesgos inicializado. Nivel actual: {self.config['current_level']}")
    
    def load_config(self):
        """
        Carga la configuración de riesgos desde el archivo.
        
        Returns:
            dict: Configuración de riesgos, o None si no se pudo cargar.
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
        Guarda la configuración de riesgos en el archivo.
        
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
    
    def analyze_performance(self, state):
        """
        Analiza el rendimiento del bot.
        
        Args:
            state (dict): Estado del bot.
            
        Returns:
            dict: Análisis del rendimiento.
        """
        # Obtener métricas de rendimiento
        metrics = state.get('performance_metrics', {})
        
        # Extraer métricas relevantes
        win_rate = metrics.get('win_rate', 0)
        profit_factor = metrics.get('profit_factor', 0)
        max_drawdown = metrics.get('max_drawdown', 0)
        total_trades = metrics.get('total_trades', 0)
        
        # Verificar si hay suficientes operaciones para un análisis significativo
        if total_trades < 10:
            logger.info(f"No hay suficientes operaciones para un análisis significativo: {total_trades}")
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'total_trades': total_trades,
                'recommended_level': None,
                'reason': "Insuficientes operaciones"
            }
        
        # Determinar nivel de riesgo recomendado
        thresholds = self.config['thresholds']
        
        if win_rate < thresholds['win_rate_low'] or profit_factor < thresholds['profit_factor_low'] or max_drawdown > thresholds['drawdown_high']:
            recommended_level = 'low'
            reason = f"Bajo rendimiento: WR={win_rate:.2f}%, PF={profit_factor:.2f}, DD={max_drawdown:.2f}%"
        elif win_rate > thresholds['win_rate_medium'] and profit_factor > thresholds['profit_factor_medium'] and max_drawdown < thresholds['drawdown_medium']:
            recommended_level = 'high'
            reason = f"Alto rendimiento: WR={win_rate:.2f}%, PF={profit_factor:.2f}, DD={max_drawdown:.2f}%"
        else:
            recommended_level = 'medium'
            reason = f"Rendimiento medio: WR={win_rate:.2f}%, PF={profit_factor:.2f}, DD={max_drawdown:.2f}%"
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'recommended_level': recommended_level,
            'reason': reason
        }
    
    def adjust_risk_parameters(self, state):
        """
        Ajusta los parámetros de riesgo basado en el rendimiento.
        
        Args:
            state (dict): Estado del bot.
            
        Returns:
            dict: Estado actualizado.
        """
        if not self.config['auto_adjust']:
            logger.info("Ajuste automático desactivado")
            return state
        
        # Verificar si ha pasado suficiente tiempo desde el último ajuste
        if self.config['last_adjustment']:
            last_adjustment = datetime.fromisoformat(self.config['last_adjustment'].replace('Z', '+00:00'))
            time_since_adjustment = (datetime.now() - last_adjustment).total_seconds() / 3600  # en horas
            
            if time_since_adjustment < 24:  # No ajustar más de una vez al día
                logger.info(f"Último ajuste hace {time_since_adjustment:.2f} horas, esperando 24 horas")
                return state
        
        # Analizar rendimiento
        analysis = self.analyze_performance(state)
        recommended_level = analysis['recommended_level']
        
        if not recommended_level:
            logger.info("No se puede recomendar un nivel de riesgo")
            return state
        
        # Si el nivel recomendado es diferente al actual, ajustar
        current_level = self.config['current_level']
        if recommended_level != current_level:
            logger.info(f"Ajustando nivel de riesgo: {current_level} -> {recommended_level}")
            
            # Actualizar nivel actual
            self.config['current_level'] = recommended_level
            self.config['last_adjustment'] = datetime.now().isoformat()
            
            # Guardar configuración
            self.save_config()
            
            # Actualizar parámetros en el estado
            risk_params = self.config['risk_levels'][recommended_level]
            state['risk_per_trade'] = risk_params['risk_per_trade']
            state['stop_loss_pct'] = risk_params['stop_loss_pct']
            state['take_profit_pct'] = risk_params['take_profit_pct']
            state['max_trades'] = risk_params['max_trades']
            
            # Notificar por Telegram
            self.notify_risk_adjustment(current_level, recommended_level, analysis['reason'])
        else:
            logger.info(f"Nivel de riesgo actual ({current_level}) es apropiado")
        
        return state
    
    def notify_risk_adjustment(self, old_level, new_level, reason):
        """
        Notifica un ajuste de riesgo por Telegram.
        
        Args:
            old_level (str): Nivel de riesgo anterior.
            new_level (str): Nuevo nivel de riesgo.
            reason (str): Razón del ajuste.
            
        Returns:
            bool: True si se envió correctamente, False en caso contrario.
        """
        message = f"""⚠️ *Ajuste de Riesgo - Bot SOL*

El nivel de riesgo ha sido ajustado:
- Anterior: {old_level.capitalize()}
- Nuevo: {new_level.capitalize()}

Razón: {reason}

Nuevos parámetros:
- Riesgo por operación: {self.config['risk_levels'][new_level]['risk_per_trade'] * 100}%
- Stop Loss: {self.config['risk_levels'][new_level]['stop_loss_pct'] * 100}%
- Take Profit: {self.config['risk_levels'][new_level]['take_profit_pct'] * 100}%
- Máx. operaciones: {self.config['risk_levels'][new_level]['max_trades']}

Ajustado automáticamente el {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        return self.telegram.send_message(message)
    
    def run(self):
        """
        Ejecuta el gestor de riesgos.
        
        Returns:
            bool: True si se ejecutó correctamente, False en caso contrario.
        """
        # Cargar estado
        state = self.load_state()
        if not state:
            logger.error("No se pudo cargar el estado del bot")
            return False
        
        # Ajustar parámetros de riesgo
        updated_state = self.adjust_risk_parameters(state)
        
        # Guardar estado actualizado
        return self.save_state(updated_state)

def parse_arguments():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description='Gestor de riesgos para el bot SOL')
    parser.add_argument('--state-file', type=str, default='sol_bot_15min_state.json', help='Archivo de estado del bot')
    parser.add_argument('--config-file', type=str, default='risk_config.json', help='Archivo de configuración de riesgos')
    parser.add_argument('--disable-auto', action='store_true', help='Desactivar ajuste automático')
    parser.add_argument('--set-level', type=str, choices=['low', 'medium', 'high'], help='Establecer nivel de riesgo manualmente')
    return parser.parse_args()

def main():
    """Función principal."""
    # Cargar variables de entorno
    load_dotenv()
    
    # Parsear argumentos
    args = parse_arguments()
    
    # Inicializar gestor de riesgos
    risk_manager = RiskManager(args.state_file, args.config_file)
    
    # Manejar configuración manual
    if args.disable_auto:
        risk_manager.config['auto_adjust'] = False
        risk_manager.save_config()
        logger.info("Ajuste automático desactivado")
    
    if args.set_level:
        risk_manager.config['current_level'] = args.set_level
        risk_manager.config['last_adjustment'] = datetime.now().isoformat()
        risk_manager.save_config()
        logger.info(f"Nivel de riesgo establecido manualmente: {args.set_level}")
        
        # Cargar estado y actualizar parámetros
        state = risk_manager.load_state()
        if state:
            risk_params = risk_manager.config['risk_levels'][args.set_level]
            state['risk_per_trade'] = risk_params['risk_per_trade']
            state['stop_loss_pct'] = risk_params['stop_loss_pct']
            state['take_profit_pct'] = risk_params['take_profit_pct']
            state['max_trades'] = risk_params['max_trades']
            risk_manager.save_state(state)
    
    # Ejecutar gestor de riesgos
    if not args.disable_auto and not args.set_level:
        success = risk_manager.run()
        if success:
            logger.info("Gestor de riesgos ejecutado correctamente")
        else:
            logger.error("Error al ejecutar el gestor de riesgos")

if __name__ == "__main__":
    main()

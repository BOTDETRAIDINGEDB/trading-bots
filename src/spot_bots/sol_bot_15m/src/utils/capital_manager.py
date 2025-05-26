#!/usr/bin/env python3
"""
Gestor de capital para el bot SOL
Adapta el tamaño de las posiciones según el capital disponible y el modo de aprendizaje
"""

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class CapitalManager:
    """Clase para gestionar el capital y tamaño de posiciones."""
    
    def __init__(self, max_risk_pct=0.06):
        """
        Inicializa el gestor de capital.
        
        Args:
            max_risk_pct (float): Riesgo máximo por operación como porcentaje (0.06 = 6%).
        """
        self.max_risk_pct = max_risk_pct
        
        # Configuración de niveles de capital
        self.capital_tiers = {
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
        }
        
        # Configuración del modo aprendizaje
        self.learning_mode = {
            'enabled': True,
            'min_trades': 20,  # Mínimo de operaciones para salir del modo aprendizaje
            'min_win_rate': 55,  # Win rate mínimo para salir del modo aprendizaje
            'position_size_multiplier': 0.5  # Reduce el tamaño de posición al 50% en modo aprendizaje
        }
        
        logger.info(f"Gestor de capital inicializado con riesgo máximo de {max_risk_pct*100}%")
    
    def determine_capital_tier(self, available_balance):
        """
        Determina el nivel de capital.
        
        Args:
            available_balance (float): Balance disponible.
            
        Returns:
            tuple: (nombre_nivel, configuración_nivel)
        """
        for tier_name, tier_config in sorted(self.capital_tiers.items(), key=lambda x: x[1]['max_amount']):
            if available_balance <= tier_config['max_amount']:
                logger.info(f"Nivel de capital determinado: {tier_name} (balance: {available_balance})")
                return tier_name, tier_config
        
        # Si no se encuentra ningún nivel (no debería ocurrir con la configuración actual)
        logger.warning(f"No se encontró nivel de capital para balance {available_balance}, usando 'large'")
        return 'large', self.capital_tiers['large']
    
    def is_learning_mode(self, performance_metrics):
        """
        Determina si el bot está en modo aprendizaje.
        
        Args:
            performance_metrics (dict): Métricas de rendimiento del bot.
            
        Returns:
            bool: True si está en modo aprendizaje, False en caso contrario.
        """
        if not self.learning_mode['enabled']:
            return False
        
        total_trades = performance_metrics.get('total_trades', 0)
        win_rate = performance_metrics.get('win_rate', 0)
        
        # Verificar condiciones para salir del modo aprendizaje
        if (total_trades >= self.learning_mode['min_trades'] and 
            win_rate >= self.learning_mode['min_win_rate']):
            logger.info(f"Saliendo del modo aprendizaje: trades={total_trades}, win_rate={win_rate}")
            return False
        
        logger.info(f"En modo aprendizaje: trades={total_trades}, win_rate={win_rate}")
        return True
    
    def calculate_position_size(self, available_balance, price, stop_loss_pct, performance_metrics=None):
        """
        Calcula el tamaño de posición óptimo.
        
        Args:
            available_balance (float): Balance disponible.
            price (float): Precio actual del activo.
            stop_loss_pct (float): Porcentaje de stop loss (0.06 = 6%).
            performance_metrics (dict, optional): Métricas de rendimiento para determinar modo aprendizaje.
            
        Returns:
            tuple: (tamaño_posición, monto_posición)
        """
        # Determinar nivel de capital
        tier_name, tier_config = self.determine_capital_tier(available_balance)
        
        # Obtener tamaño de posición base según nivel
        position_size_pct = tier_config['position_size_pct']
        
        # Aplicar factor de riesgo
        risk_factor = tier_config['risk_factor']
        
        # Verificar si está en modo aprendizaje
        if performance_metrics and self.is_learning_mode(performance_metrics):
            # Reducir tamaño de posición en modo aprendizaje
            learning_multiplier = self.learning_mode['position_size_multiplier']
            position_size_pct *= learning_multiplier
            logger.info(f"Modo aprendizaje activo: tamaño de posición reducido a {position_size_pct:.2%}")
        
        # Calcular monto de la posición
        position_amount = available_balance * position_size_pct * risk_factor
        
        # Asegurar que el riesgo no exceda el máximo permitido
        max_risk_amount = available_balance * self.max_risk_pct
        risk_amount = position_amount * stop_loss_pct
        
        if risk_amount > max_risk_amount:
            # Ajustar el monto de la posición para limitar el riesgo
            position_amount = max_risk_amount / stop_loss_pct
            logger.info(f"Posición ajustada para limitar riesgo al {self.max_risk_pct*100}% del capital")
        
        # Calcular tamaño de la posición en unidades del activo
        position_size = position_amount / price
        
        # Redondear a 4 decimales (ajustar según la precisión del par)
        position_size = round(position_size, 4)
        
        logger.info(f"Tamaño de posición calculado: {position_size} unidades ({position_amount:.2f} USDT) a precio {price}")
        
        return position_size, position_amount

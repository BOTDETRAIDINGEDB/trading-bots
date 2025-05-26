#!/usr/bin/env python3
"""
Monitor de rendimiento para el bot SOL
Genera informes detallados y alertas sobre el rendimiento del bot
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
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
        logging.FileHandler("performance_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    """Monitor de rendimiento para el bot SOL."""
    
    def __init__(self, state_file, report_dir="reports"):
        """
        Inicializa el monitor de rendimiento.
        
        Args:
            state_file (str): Ruta al archivo de estado del bot.
            report_dir (str): Directorio donde se guardarán los informes.
        """
        self.state_file = state_file
        self.report_dir = report_dir
        self.telegram = TelegramNotifier()
        
        # Crear directorio de informes si no existe
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        logger.info(f"Monitor de rendimiento inicializado. Archivo de estado: {state_file}")
    
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
    
    def analyze_trades(self, trades):
        """
        Analiza las operaciones del bot.
        
        Args:
            trades (list): Lista de operaciones.
            
        Returns:
            dict: Análisis de las operaciones.
        """
        if not trades:
            return {
                "total_trades": 0,
                "message": "No hay operaciones para analizar"
            }
        
        # Filtrar operaciones cerradas
        closed_trades = [t for t in trades if t.get('status') == 'closed']
        
        # Análisis básico
        total_trades = len(closed_trades)
        winning_trades = sum(1 for t in closed_trades if t.get('profit_loss', 0) > 0)
        losing_trades = sum(1 for t in closed_trades if t.get('profit_loss', 0) <= 0)
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calcular ganancias/pérdidas
        total_profit = sum(t.get('profit_loss', 0) for t in closed_trades if t.get('profit_loss', 0) > 0)
        total_loss = sum(abs(t.get('profit_loss', 0)) for t in closed_trades if t.get('profit_loss', 0) <= 0)
        net_profit = total_profit - total_loss
        
        # Calcular promedios
        avg_profit = (total_profit / winning_trades) if winning_trades > 0 else 0
        avg_loss = (total_loss / losing_trades) if losing_trades > 0 else 0
        
        # Calcular profit factor
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')
        
        # Calcular duración promedio de operaciones
        durations = []
        for trade in closed_trades:
            if trade.get('entry_time') and trade.get('exit_time'):
                entry_time = datetime.fromisoformat(trade['entry_time'].replace('Z', '+00:00'))
                exit_time = datetime.fromisoformat(trade['exit_time'].replace('Z', '+00:00'))
                duration = (exit_time - entry_time).total_seconds() / 3600  # en horas
                durations.append(duration)
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "net_profit": net_profit,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "avg_duration": avg_duration
        }
    
    def generate_report(self, analysis, period="daily"):
        """
        Genera un informe de rendimiento.
        
        Args:
            analysis (dict): Análisis de las operaciones.
            period (str): Periodo del informe ('daily', 'weekly', 'monthly').
            
        Returns:
            str: Ruta al archivo de informe generado.
        """
        # Crear nombre de archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.report_dir}/sol_bot_report_{period}_{timestamp}.html"
        
        # Generar HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Informe de Rendimiento - Bot SOL</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 20px; }}
                .metric {{ background-color: #e9ecef; padding: 10px; border-radius: 5px; }}
                .positive {{ color: green; }}
                .negative {{ color: red; }}
            </style>
        </head>
        <body>
            <h1>Informe de Rendimiento - Bot SOL</h1>
            <p>Periodo: {period.capitalize()}</p>
            <p>Generado: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            
            <div class="summary">
                <h2>Resumen</h2>
                <p>Operaciones totales: <strong>{analysis['total_trades']}</strong></p>
                <p>Operaciones ganadoras: <strong class="positive">{analysis['winning_trades']}</strong></p>
                <p>Operaciones perdedoras: <strong class="negative">{analysis['losing_trades']}</strong></p>
                <p>Tasa de éxito: <strong>{analysis['win_rate']:.2f}%</strong></p>
                <p>Beneficio neto: <strong class="{'positive' if analysis['net_profit'] >= 0 else 'negative'}">{analysis['net_profit']:.2f} USDT</strong></p>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <h3>Profit Factor</h3>
                    <p>{analysis['profit_factor']:.2f}</p>
                </div>
                <div class="metric">
                    <h3>Ganancia Promedio</h3>
                    <p class="positive">{analysis['avg_profit']:.2f} USDT</p>
                </div>
                <div class="metric">
                    <h3>Pérdida Promedio</h3>
                    <p class="negative">{analysis['avg_loss']:.2f} USDT</p>
                </div>
                <div class="metric">
                    <h3>Duración Promedio</h3>
                    <p>{analysis['avg_duration']:.2f} horas</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Guardar archivo
        with open(filename, 'w') as f:
            f.write(html)
        
        logger.info(f"Informe generado: {filename}")
        return filename
    
    def send_telegram_summary(self, analysis):
        """
        Envía un resumen del rendimiento por Telegram.
        
        Args:
            analysis (dict): Análisis de las operaciones.
            
        Returns:
            bool: True si se envió correctamente, False en caso contrario.
        """
        if analysis['total_trades'] == 0:
            message = "📊 *Resumen de Rendimiento - Bot SOL*\n\nNo hay operaciones para analizar."
            return self.telegram.send_message(message)
        
        message = f"""📊 *Resumen de Rendimiento - Bot SOL*

🔢 *Operaciones*:
- Total: {analysis['total_trades']}
- Ganadoras: {analysis['winning_trades']}
- Perdedoras: {analysis['losing_trades']}
- Tasa de éxito: {analysis['win_rate']:.2f}%

💰 *Resultados*:
- Beneficio neto: {analysis['net_profit']:.2f} USDT
- Profit Factor: {analysis['profit_factor']:.2f}
- Ganancia promedio: {analysis['avg_profit']:.2f} USDT
- Pérdida promedio: {analysis['avg_loss']:.2f} USDT

⏱️ *Duración promedio*: {analysis['avg_duration']:.2f} horas

Generado: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        return self.telegram.send_message(message)
    
    def run(self, send_telegram=True, generate_html=True):
        """
        Ejecuta el monitor de rendimiento.
        
        Args:
            send_telegram (bool): Si es True, envía un resumen por Telegram.
            generate_html (bool): Si es True, genera un informe HTML.
            
        Returns:
            dict: Análisis de las operaciones.
        """
        # Cargar estado
        state = self.load_state()
        if not state:
            logger.error("No se pudo cargar el estado del bot")
            return None
        
        # Obtener operaciones
        trades = state.get('trades', [])
        
        # Analizar operaciones
        analysis = self.analyze_trades(trades)
        
        # Enviar resumen por Telegram
        if send_telegram:
            self.send_telegram_summary(analysis)
        
        # Generar informe HTML
        if generate_html:
            self.generate_report(analysis)
        
        return analysis

def parse_arguments():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description='Monitor de rendimiento para el bot SOL')
    parser.add_argument('--state-file', type=str, default='sol_bot_15min_state.json', help='Archivo de estado del bot')
    parser.add_argument('--report-dir', type=str, default='reports', help='Directorio para guardar informes')
    parser.add_argument('--no-telegram', action='store_true', help='No enviar resumen por Telegram')
    parser.add_argument('--no-html', action='store_true', help='No generar informe HTML')
    return parser.parse_args()

def main():
    """Función principal."""
    # Cargar variables de entorno
    load_dotenv()
    
    # Parsear argumentos
    args = parse_arguments()
    
    # Inicializar monitor
    monitor = PerformanceMonitor(args.state_file, args.report_dir)
    
    # Ejecutar monitor
    analysis = monitor.run(
        send_telegram=not args.no_telegram,
        generate_html=not args.no_html
    )
    
    if analysis:
        logger.info(f"Análisis completado: {json.dumps(analysis, indent=2)}")
    else:
        logger.error("No se pudo completar el análisis")

if __name__ == "__main__":
    main()

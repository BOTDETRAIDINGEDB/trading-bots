#!/usr/bin/env python3
"""
Analizador de sentimiento del mercado para el bot SOL
Analiza noticias, redes sociales y datos de mercado para determinar el sentimiento
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv
from textblob import TextBlob
import tweepy

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
        logging.FileHandler("sentiment_analyzer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analizador de sentimiento del mercado para el bot SOL."""
    
    def __init__(self, state_file, config_file='sentiment_config.json'):
        """
        Inicializa el analizador de sentimiento.
        
        Args:
            state_file (str): Ruta al archivo de estado del bot.
            config_file (str): Ruta al archivo de configuración de sentimiento.
        """
        self.state_file = state_file
        self.config_file = config_file
        self.telegram = TelegramNotifier()
        
        # Cargar configuración de sentimiento
        self.config = self.load_config()
        if not self.config:
            # Configuración por defecto
            self.config = {
                'sources': {
                    'twitter': {
                        'enabled': True,
                        'weight': 0.3,
                        'keywords': ['Solana', 'SOL', '$SOL', 'SOL/USDT', 'Solana price']
                    },
                    'news': {
                        'enabled': True,
                        'weight': 0.4,
                        'keywords': ['Solana', 'SOL cryptocurrency', 'Solana blockchain']
                    },
                    'market_data': {
                        'enabled': True,
                        'weight': 0.3
                    }
                },
                'update_interval': 3600,  # segundos
                'influence_factor': 0.2,  # cuánto influye el sentimiento en las decisiones
                'last_update': None
            }
            self.save_config()
        
        # Inicializar APIs
        self.init_apis()
        
        logger.info("Analizador de sentimiento inicializado")
    
    def init_apis(self):
        """Inicializa las APIs necesarias."""
        # Twitter API (v2)
        twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        if twitter_bearer_token and self.config['sources']['twitter']['enabled']:
            try:
                self.twitter_client = tweepy.Client(bearer_token=twitter_bearer_token)
                logger.info("API de Twitter inicializada")
            except Exception as e:
                logger.error(f"Error al inicializar API de Twitter: {str(e)}")
                self.config['sources']['twitter']['enabled'] = False
        else:
            logger.warning("API de Twitter desactivada (token no disponible)")
            self.config['sources']['twitter']['enabled'] = False
        
        # News API
        self.news_api_key = os.getenv('NEWS_API_KEY')
        if not self.news_api_key:
            logger.warning("API de noticias desactivada (clave no disponible)")
            self.config['sources']['news']['enabled'] = False
    
    def load_config(self):
        """
        Carga la configuración de sentimiento desde el archivo.
        
        Returns:
            dict: Configuración de sentimiento, o None si no se pudo cargar.
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
        Guarda la configuración de sentimiento en el archivo.
        
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
    
    def analyze_twitter_sentiment(self):
        """
        Analiza el sentimiento en Twitter.
        
        Returns:
            float: Puntuación de sentimiento (-1 a 1), o None si hubo un error.
        """
        if not self.config['sources']['twitter']['enabled']:
            logger.info("Análisis de Twitter desactivado")
            return None
        
        try:
            # Obtener tweets relacionados con SOL
            keywords = ' OR '.join(self.config['sources']['twitter']['keywords'])
            query = f"{keywords} -is:retweet lang:en"
            
            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=100,
                tweet_fields=['created_at', 'public_metrics']
            )
            
            if not tweets.data:
                logger.warning("No se encontraron tweets")
                return 0
            
            # Analizar sentimiento de los tweets
            sentiments = []
            for tweet in tweets.data:
                blob = TextBlob(tweet.text)
                sentiment = blob.sentiment.polarity
                # Ponderación por métricas de engagement
                engagement = tweet.public_metrics['retweet_count'] + tweet.public_metrics['like_count']
                weight = 1 + min(engagement / 10, 5)  # Cap weight at 6x
                sentiments.append(sentiment * weight)
            
            # Calcular sentimiento promedio ponderado
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                logger.info(f"Sentimiento de Twitter: {avg_sentiment:.4f} (basado en {len(tweets.data)} tweets)")
                return avg_sentiment
            else:
                return 0
        except Exception as e:
            logger.error(f"Error al analizar sentimiento de Twitter: {str(e)}")
            return None
    
    def analyze_news_sentiment(self):
        """
        Analiza el sentimiento en noticias.
        
        Returns:
            float: Puntuación de sentimiento (-1 a 1), o None si hubo un error.
        """
        if not self.config['sources']['news']['enabled'] or not self.news_api_key:
            logger.info("Análisis de noticias desactivado")
            return None
        
        try:
            # Obtener noticias relacionadas con SOL
            keywords = ' OR '.join(self.config['sources']['news']['keywords'])
            url = f"https://newsapi.org/v2/everything?q={keywords}&language=en&sortBy=publishedAt&apiKey={self.news_api_key}"
            
            response = requests.get(url)
            if response.status_code != 200:
                logger.error(f"Error al obtener noticias: {response.status_code} - {response.text}")
                return None
            
            news_data = response.json()
            articles = news_data.get('articles', [])
            
            if not articles:
                logger.warning("No se encontraron noticias")
                return 0
            
            # Analizar sentimiento de las noticias
            sentiments = []
            for article in articles[:20]:  # Limitar a 20 artículos para eficiencia
                # Combinar título y descripción para análisis
                text = f"{article.get('title', '')} {article.get('description', '')}"
                blob = TextBlob(text)
                sentiment = blob.sentiment.polarity
                sentiments.append(sentiment)
            
            # Calcular sentimiento promedio
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                logger.info(f"Sentimiento de noticias: {avg_sentiment:.4f} (basado en {len(sentiments)} artículos)")
                return avg_sentiment
            else:
                return 0
        except Exception as e:
            logger.error(f"Error al analizar sentimiento de noticias: {str(e)}")
            return None
    
    def analyze_market_sentiment(self):
        """
        Analiza el sentimiento del mercado basado en datos técnicos.
        
        Returns:
            float: Puntuación de sentimiento (-1 a 1), o None si hubo un error.
        """
        if not self.config['sources']['market_data']['enabled']:
            logger.info("Análisis de datos de mercado desactivado")
            return None
        
        try:
            # Obtener datos de mercado de Binance
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': 'SOLUSDT',
                'interval': '1h',
                'limit': 24  # Últimas 24 horas
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
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Calcular indicadores de sentimiento
            # 1. Tendencia de precio (positiva/negativa)
            price_change = (df['close'].iloc[-1] - df['open'].iloc[0]) / df['open'].iloc[0]
            price_sentiment = np.tanh(price_change * 5)  # Normalizar a [-1, 1]
            
            # 2. Volatilidad (alta volatilidad = sentimiento negativo)
            volatility = df['high'].max() / df['low'].min() - 1
            volatility_sentiment = -np.tanh(volatility * 10)  # Normalizar a [-1, 1]
            
            # 3. Volumen (aumento de volumen = sentimiento positivo si precio sube)
            volume_change = df['volume'].iloc[-1] / df['volume'].iloc[0] - 1
            volume_sentiment = np.tanh(volume_change * 2) * np.sign(price_change)
            
            # Combinar indicadores
            market_sentiment = (price_sentiment * 0.5 + volatility_sentiment * 0.3 + volume_sentiment * 0.2)
            
            logger.info(f"Sentimiento de mercado: {market_sentiment:.4f}")
            return market_sentiment
        except Exception as e:
            logger.error(f"Error al analizar sentimiento de mercado: {str(e)}")
            return None
    
    def calculate_overall_sentiment(self):
        """
        Calcula el sentimiento general combinando todas las fuentes.
        
        Returns:
            dict: Resultado del análisis de sentimiento.
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'sources': {},
            'overall_score': 0,
            'sentiment_label': 'neutral',
            'confidence': 0
        }
        
        # Analizar cada fuente
        twitter_sentiment = self.analyze_twitter_sentiment()
        news_sentiment = self.analyze_news_sentiment()
        market_sentiment = self.analyze_market_sentiment()
        
        # Almacenar resultados individuales
        sources_data = {}
        weights_sum = 0
        weighted_scores = []
        
        if twitter_sentiment is not None:
            weight = self.config['sources']['twitter']['weight']
            sources_data['twitter'] = {
                'score': twitter_sentiment,
                'weight': weight
            }
            weighted_scores.append(twitter_sentiment * weight)
            weights_sum += weight
        
        if news_sentiment is not None:
            weight = self.config['sources']['news']['weight']
            sources_data['news'] = {
                'score': news_sentiment,
                'weight': weight
            }
            weighted_scores.append(news_sentiment * weight)
            weights_sum += weight
        
        if market_sentiment is not None:
            weight = self.config['sources']['market_data']['weight']
            sources_data['market_data'] = {
                'score': market_sentiment,
                'weight': weight
            }
            weighted_scores.append(market_sentiment * weight)
            weights_sum += weight
        
        results['sources'] = sources_data
        
        # Calcular sentimiento general ponderado
        if weights_sum > 0:
            overall_score = sum(weighted_scores) / weights_sum
            results['overall_score'] = overall_score
            
            # Determinar etiqueta de sentimiento
            if overall_score > 0.2:
                results['sentiment_label'] = 'bullish'
            elif overall_score < -0.2:
                results['sentiment_label'] = 'bearish'
            else:
                results['sentiment_label'] = 'neutral'
            
            # Calcular confianza (qué tan extremo es el sentimiento)
            results['confidence'] = abs(overall_score)
        
        logger.info(f"Sentimiento general: {results['overall_score']:.4f} ({results['sentiment_label']})")
        return results
    
    def update_state_with_sentiment(self, sentiment_results):
        """
        Actualiza el estado del bot con los resultados del análisis de sentimiento.
        
        Args:
            sentiment_results (dict): Resultados del análisis de sentimiento.
            
        Returns:
            bool: True si se actualizó correctamente, False en caso contrario.
        """
        # Cargar estado actual
        state = self.load_state()
        if not state:
            logger.error("No se pudo cargar el estado del bot")
            return False
        
        # Actualizar estado con sentimiento
        state['sentiment_analysis'] = sentiment_results
        
        # Ajustar parámetros de trading basados en el sentimiento
        influence_factor = self.config['influence_factor']
        sentiment_score = sentiment_results['overall_score']
        
        # Solo ajustar si la confianza es suficiente
        if sentiment_results['confidence'] > 0.3:
            # Ajustar take profit basado en sentimiento
            if 'take_profit_pct' in state:
                base_tp = state['take_profit_pct']
                if sentiment_score > 0:  # Bullish
                    # Aumentar take profit si sentimiento es positivo
                    adjusted_tp = base_tp * (1 + sentiment_score * influence_factor)
                    state['take_profit_pct'] = min(adjusted_tp, base_tp * 1.5)  # Limitar el aumento
                elif sentiment_score < 0:  # Bearish
                    # Reducir take profit si sentimiento es negativo
                    adjusted_tp = base_tp * (1 + sentiment_score * influence_factor)
                    state['take_profit_pct'] = max(adjusted_tp, base_tp * 0.7)  # Limitar la reducción
            
            # Ajustar stop loss basado en sentimiento
            if 'stop_loss_pct' in state:
                base_sl = state['stop_loss_pct']
                if sentiment_score < 0:  # Bearish
                    # Reducir stop loss si sentimiento es negativo (más cercano al precio)
                    adjusted_sl = base_sl * (1 - sentiment_score * influence_factor * 0.5)
                    state['stop_loss_pct'] = min(adjusted_sl, base_sl * 1.3)  # Limitar el aumento
                elif sentiment_score > 0:  # Bullish
                    # Aumentar stop loss si sentimiento es positivo (más lejos del precio)
                    adjusted_sl = base_sl * (1 - sentiment_score * influence_factor * 0.5)
                    state['stop_loss_pct'] = max(adjusted_sl, base_sl * 0.8)  # Limitar la reducción
        
        # Guardar estado actualizado
        return self.save_state(state)
    
    def notify_sentiment_update(self, sentiment_results):
        """
        Notifica los resultados del análisis de sentimiento por Telegram.
        
        Args:
            sentiment_results (dict): Resultados del análisis de sentimiento.
            
        Returns:
            bool: True si se envió correctamente, False en caso contrario.
        """
        sentiment_label = sentiment_results['sentiment_label'].capitalize()
        score = sentiment_results['overall_score']
        confidence = sentiment_results['confidence'] * 100
        
        # Determinar emoji basado en el sentimiento
        if sentiment_label == 'Bullish':
            emoji = '🐂📈'
        elif sentiment_label == 'Bearish':
            emoji = '🐻📉'
        else:
            emoji = '⚖️'
        
        message = f"""*{emoji} Análisis de Sentimiento - SOL {emoji}*

*Sentimiento actual:* {sentiment_label}
*Puntuación:* {score:.4f}
*Confianza:* {confidence:.2f}%

*Fuentes analizadas:*"""
        
        # Añadir detalles de cada fuente
        for source, data in sentiment_results['sources'].items():
            source_name = source.capitalize()
            source_score = data['score']
            source_weight = data['weight'] * 100
            
            message += f"\n- {source_name}: {source_score:.4f} (peso: {source_weight:.0f}%)"
        
        message += f"\n\n*Actualizado:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        return self.telegram.send_message(message)
    
    def run(self, notify=True):
        """
        Ejecuta el analizador de sentimiento.
        
        Args:
            notify (bool): Si es True, notifica los resultados por Telegram.
            
        Returns:
            dict: Resultados del análisis de sentimiento, o None si hubo un error.
        """
        # Verificar si es necesario actualizar (basado en intervalo)
        if self.config['last_update']:
            last_update = datetime.fromisoformat(self.config['last_update'].replace('Z', '+00:00'))
            time_since_update = (datetime.now() - last_update).total_seconds()
            
            if time_since_update < self.config['update_interval']:
                logger.info(f"Última actualización hace {time_since_update:.2f} segundos, esperando {self.config['update_interval']} segundos")
                
                # Cargar estado para obtener el último análisis
                state = self.load_state()
                if state and 'sentiment_analysis' in state:
                    return state['sentiment_analysis']
                return None
        
        # Calcular sentimiento general
        sentiment_results = self.calculate_overall_sentiment()
        
        # Actualizar estado con sentimiento
        self.update_state_with_sentiment(sentiment_results)
        
        # Actualizar timestamp de última actualización
        self.config['last_update'] = datetime.now().isoformat()
        self.save_config()
        
        # Notificar resultados
        if notify:
            self.notify_sentiment_update(sentiment_results)
        
        return sentiment_results

def parse_arguments():
    """Parsea los argumentos de la línea de comandos."""
    parser = argparse.ArgumentParser(description='Analizador de sentimiento para el bot SOL')
    parser.add_argument('--state-file', type=str, default='sol_bot_15min_state.json', help='Archivo de estado del bot')
    parser.add_argument('--config-file', type=str, default='sentiment_config.json', help='Archivo de configuración de sentimiento')
    parser.add_argument('--no-notify', action='store_true', help='No notificar resultados por Telegram')
    parser.add_argument('--update-interval', type=int, help='Intervalo de actualización en segundos')
    return parser.parse_args()

def main():
    """Función principal."""
    # Cargar variables de entorno
    load_dotenv()
    
    # Parsear argumentos
    args = parse_arguments()
    
    # Inicializar analizador de sentimiento
    sentiment_analyzer = SentimentAnalyzer(args.state_file, args.config_file)
    
    # Actualizar intervalo si se especifica
    if args.update_interval:
        sentiment_analyzer.config['update_interval'] = args.update_interval
        sentiment_analyzer.save_config()
    
    # Ejecutar analizador
    results = sentiment_analyzer.run(notify=not args.no_notify)
    
    if results:
        print(f"Sentimiento general: {results['overall_score']:.4f} ({results['sentiment_label']})")
        print(f"Confianza: {results['confidence'] * 100:.2f}%")
    else:
        print("No se pudo analizar el sentimiento")

if __name__ == "__main__":
    main()

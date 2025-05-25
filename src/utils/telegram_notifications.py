"""
Telegram Notifications Module
"""
import telegram
from dotenv import load_dotenv
import os
import asyncio

# Load environment variables
load_dotenv()

class TelegramNotifier:
    def __init__(self):
        """Initialize Telegram bot"""
        self.bot = telegram.Bot(token=os.getenv('TELEGRAM_BOT_TOKEN'))
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    async def send_message(self, message):
        """Send message via Telegram"""
        await self.bot.send_message(
            chat_id=self.chat_id,
            text=message
        )
    
    def notify(self, message):
        """Send notification (synchronous wrapper)"""
        asyncio.run(self.send_message(message))

#!/bin/bash
# Script para crear o actualizar el archivo .env del bot SOL

# Colores para mejor legibilidad
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Configuración del archivo .env para el bot SOL ===${NC}"

# Directorio del bot
BOT_DIR="/home/edisonbautistaruiz2025/new-trading-bots/src/spot_bots/sol_bot_15m"
ENV_FILE="${BOT_DIR}/.env"
BACKUP_FILE="${BOT_DIR}/.env.backup.$(date +%Y%m%d%H%M%S)"

# 1. Verificar que existe el directorio del bot
if [ ! -d "$BOT_DIR" ]; then
    echo -e "${RED}Error: El directorio del bot no existe: $BOT_DIR${NC}"
    exit 1
fi

# 2. Hacer copia de seguridad del archivo .env si existe
if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "$BACKUP_FILE"
    echo -e "${GREEN}Copia de seguridad creada: $BACKUP_FILE${NC}"
fi

# 3. Solicitar información necesaria
echo -e "${YELLOW}Ingresa tu API Key de Binance:${NC}"
read -r BINANCE_API_KEY

echo -e "${YELLOW}Ingresa tu API Secret de Binance:${NC}"
read -r BINANCE_API_SECRET

echo -e "${YELLOW}Ingresa tu Token de Bot de Telegram:${NC}"
read -r TELEGRAM_BOT_TOKEN

echo -e "${YELLOW}Ingresa tu Chat ID de Telegram:${NC}"
read -r TELEGRAM_CHAT_ID

echo -e "${YELLOW}Ingresa la URL de la API (o presiona Enter para usar el valor por defecto):${NC}"
read -r API_URL
API_URL=${API_URL:-"http://localhost:5000"}

echo -e "${YELLOW}Ingresa el ID del Bot (o presiona Enter para usar el valor por defecto):${NC}"
read -r BOT_ID
BOT_ID=${BOT_ID:-"sol_bot_15m"}

echo -e "${YELLOW}Ingresa la Clave Secreta de la API (o presiona Enter para usar el valor por defecto):${NC}"
read -r API_SECRET_KEY
API_SECRET_KEY=${API_SECRET_KEY:-"your_api_secret_key_here"}

# 4. Crear o actualizar el archivo .env
cat > "$ENV_FILE" << EOF
# Binance API Keys
BINANCE_API_KEY=$BINANCE_API_KEY
BINANCE_API_SECRET=$BINANCE_API_SECRET

# Telegram Notification Settings
TELEGRAM_BOT_TOKEN=$TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID=$TELEGRAM_CHAT_ID

# API Configuration
API_URL=$API_URL
BOT_ID=$BOT_ID
API_SECRET_KEY=$API_SECRET_KEY
EOF

echo -e "${GREEN}Archivo .env creado exitosamente en $ENV_FILE${NC}"

# 5. Verificar permisos
chmod 600 "$ENV_FILE"
echo -e "${GREEN}Permisos del archivo .env actualizados para mayor seguridad${NC}"

echo -e "${YELLOW}=== Configuración completada ===${NC}"
echo -e "Ahora puedes ejecutar el bot con el script start_sol_bot_15m.sh"
echo -e "Para verificar la configuración de Telegram, ejecuta: python3 check_telegram.py"

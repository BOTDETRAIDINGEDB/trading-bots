#!/bin/bash
# Script para actualizar el proyecto trading-bots en la máquina virtual

# Colores para mejor legibilidad
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Iniciando actualización del proyecto trading-bots ===${NC}"

# Directorio base
BASE_DIR="/home/edisonbautistaruiz2025"
NEW_PROJECT_DIR="${BASE_DIR}/new-trading-bots"
BACKUP_DIR="${BASE_DIR}/trading-bots-backup-$(date +%Y%m%d%H%M%S)"

# 1. Crear copia de seguridad del proyecto actual
echo -e "${YELLOW}1. Creando copia de seguridad del proyecto actual...${NC}"
if [ -d "$NEW_PROJECT_DIR" ]; then
    mkdir -p "$BACKUP_DIR"
    cp -r "$NEW_PROJECT_DIR" "$BACKUP_DIR"
    echo -e "${GREEN}   Copia de seguridad creada en $BACKUP_DIR${NC}"
else
    echo -e "${RED}   El directorio $NEW_PROJECT_DIR no existe. Omitiendo copia de seguridad.${NC}"
fi

# 2. Detener bots en ejecución
echo -e "${YELLOW}2. Deteniendo bots en ejecución...${NC}"
# Identificar sesiones de screen
SCREEN_SESSIONS=$(screen -ls | grep -oE "[0-9]+\.sol_bot_20m")
if [ -n "$SCREEN_SESSIONS" ]; then
    for session in $SCREEN_SESSIONS; do
        echo -e "   Deteniendo sesión: $session"
        screen -S "$session" -X quit
    done
    echo -e "${GREEN}   Bots detenidos correctamente${NC}"
else
    echo -e "${YELLOW}   No se encontraron sesiones de bot activas${NC}"
fi

# 3. Eliminar el proyecto anterior
echo -e "${YELLOW}3. Eliminando proyecto anterior...${NC}"
if [ -d "$NEW_PROJECT_DIR" ]; then
    rm -rf "$NEW_PROJECT_DIR"
    echo -e "${GREEN}   Proyecto anterior eliminado${NC}"
else
    echo -e "${YELLOW}   No existe directorio del proyecto anterior${NC}"
fi

# 4. Clonar el nuevo proyecto
echo -e "${YELLOW}4. Clonando el nuevo proyecto desde GitHub...${NC}"
git clone https://github.com/BOTDETRAIDINGEDB/trading-bots.git "$NEW_PROJECT_DIR"
if [ $? -eq 0 ]; then
    echo -e "${GREEN}   Proyecto clonado correctamente${NC}"
else
    echo -e "${RED}   Error al clonar el proyecto. Abortando.${NC}"
    exit 1
fi

# 5. Instalar dependencias
echo -e "${YELLOW}5. Instalando dependencias...${NC}"
cd "$NEW_PROJECT_DIR"
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}   Dependencias instaladas correctamente${NC}"
else
    echo -e "${YELLOW}   No se encontró requirements.txt o hubo errores en la instalación${NC}"
    # Instalar dependencias manualmente
    echo -e "${YELLOW}   Instalando dependencias manualmente...${NC}"
    pip install pandas numpy requests python-binance python-telegram-bot python-dotenv scikit-learn
    echo -e "${GREEN}   Dependencias manuales instaladas${NC}"
fi

# 6. Configurar permisos de ejecución para los scripts
echo -e "${YELLOW}6. Configurando permisos de ejecución...${NC}"
find "$NEW_PROJECT_DIR" -name "*.sh" -exec chmod +x {} \;
echo -e "${GREEN}   Permisos configurados${NC}"

# 7. Crear archivo .env de ejemplo si no existe
echo -e "${YELLOW}7. Verificando archivo .env...${NC}"
ENV_FILE="${NEW_PROJECT_DIR}/src/spot_bots/sol_bot_20m/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}   Creando archivo .env de ejemplo...${NC}"
    cat > "$ENV_FILE" << 'EOF'
# Configuración para el bot SOL con reentrenamiento

# Credenciales de Binance
BINANCE_API_KEY=tu_api_key_aqui
BINANCE_API_SECRET=tu_api_secret_aqui

# Configuración de Telegram
TELEGRAM_BOT_TOKEN=tu_token_de_bot_aqui
TELEGRAM_CHAT_ID=tu_chat_id_aqui

# Configuración de la API
API_URL=http://localhost:5000
BOT_ID=sol_bot_20m
API_SECRET_KEY=tu_clave_secreta_aqui
EOF
    echo -e "${YELLOW}   IMPORTANTE: Debes editar el archivo .env con tus credenciales${NC}"
    echo -e "${YELLOW}   Ubicación: $ENV_FILE${NC}"
else
    echo -e "${GREEN}   Archivo .env encontrado${NC}"
fi

# 8. Crear script de inicio para el bot SOL con reentrenamiento
echo -e "${YELLOW}8. Creando script de inicio para el bot SOL...${NC}"
START_SCRIPT="${NEW_PROJECT_DIR}/start_sol_bot_20m.sh"
cat > "$START_SCRIPT" << 'EOF'
#!/bin/bash
# Script para iniciar el bot SOL con reentrenamiento cada 20 minutos

# Cargar variables de entorno
source ~/.bashrc

# Directorio del bot
BOT_DIR="/home/edisonbautistaruiz2025/new-trading-bots/src/spot_bots/sol_bot_20m"

# Crear directorio de logs si no existe
mkdir -p "$BOT_DIR/logs"

# Iniciar el bot en una sesión de screen
cd "$BOT_DIR"
screen -dmS sol_bot_20m python3 main.py --use-ml --retrain-interval 20 --interval 20m --symbol SOLUSDT

echo "Bot SOL iniciado en sesión screen 'sol_bot_20m'"
echo "Para ver los logs: screen -r sol_bot_20m"
EOF

chmod +x "$START_SCRIPT"
echo -e "${GREEN}   Script de inicio creado en $START_SCRIPT${NC}"

# 9. Mostrar instrucciones finales
echo -e "${YELLOW}=== Actualización completada ===${NC}"
echo -e "${GREEN}Para iniciar el bot SOL con reentrenamiento:${NC}"
echo -e "   ${YELLOW}$START_SCRIPT${NC}"
echo -e "${GREEN}Para ver los logs del bot:${NC}"
echo -e "   ${YELLOW}screen -r sol_bot_20m${NC}"
echo -e "${GREEN}Para salir de la vista de logs (sin detener el bot):${NC}"
echo -e "   ${YELLOW}Presiona Ctrl+A y luego D${NC}"

# Recordatorio sobre el archivo .env
ENV_FILE="${NEW_PROJECT_DIR}/src/spot_bots/sol_bot_20m/.env"
echo -e "\n${YELLOW}IMPORTANTE: Asegúrate de configurar correctamente el archivo .env${NC}"
echo -e "${YELLOW}Ubicación: $ENV_FILE${NC}"
echo -e "${YELLOW}Debe contener:${NC}"
echo -e "   - BINANCE_API_KEY y BINANCE_API_SECRET\n   - TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_ID\n   - API_URL, BOT_ID y API_SECRET_KEY"

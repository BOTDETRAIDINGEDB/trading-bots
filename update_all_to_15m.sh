#!/bin/bash
# Script para actualizar todas las referencias de 20m a 15m en todos los archivos relevantes

# Colores para mejor legibilidad
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Actualizando todas las referencias de 20m a 15m ===${NC}"

# Directorio base
BASE_DIR="/home/edisonbautistaruiz2025"
TRADING_BOTS_DIR="${BASE_DIR}/new-trading-bots"
API_DIR="${BASE_DIR}/trading-bots-api"

# 1. Detener los bots y la API si están ejecutándose
echo -e "${YELLOW}1. Deteniendo servicios si están ejecutándose...${NC}"
screen -S sol_bot_20m -X quit 2>/dev/null || true
screen -S sol_bot_15m -X quit 2>/dev/null || true
API_SCREEN=$(screen -ls | grep -o "[0-9]*\.trading_api")
if [ -n "$API_SCREEN" ]; then
    echo -e "${YELLOW}Deteniendo sesión de API existente: $API_SCREEN${NC}"
    screen -S "$API_SCREEN" -X quit
fi
echo -e "${GREEN}   Servicios detenidos${NC}"

# 2. Actualizar el directorio del bot
echo -e "${YELLOW}2. Actualizando directorio del bot...${NC}"
OLD_BOT_DIR="${TRADING_BOTS_DIR}/src/spot_bots/sol_bot_20m"
NEW_BOT_DIR="${TRADING_BOTS_DIR}/src/spot_bots/sol_bot_15m"

if [ -d "$OLD_BOT_DIR" ]; then
    if [ -d "$NEW_BOT_DIR" ]; then
        echo -e "${YELLOW}   El directorio $NEW_BOT_DIR ya existe. Eliminándolo...${NC}"
        rm -rf "$NEW_BOT_DIR"
    fi
    echo -e "${YELLOW}   Renombrando $OLD_BOT_DIR a $NEW_BOT_DIR${NC}"
    mv "$OLD_BOT_DIR" "$NEW_BOT_DIR"
    echo -e "${GREEN}   Directorio actualizado${NC}"
else
    echo -e "${RED}   No se encontró el directorio del bot: $OLD_BOT_DIR${NC}"
    echo -e "${YELLOW}   Verificando si existe el directorio nuevo...${NC}"
    if [ -d "$NEW_BOT_DIR" ]; then
        echo -e "${GREEN}   El directorio $NEW_BOT_DIR ya existe. Continuando...${NC}"
    else
        echo -e "${RED}   No se encontró ningún directorio del bot. Abortando.${NC}"
        exit 1
    fi
fi

# 3. Actualizar referencias en archivos Python y shell scripts
echo -e "${YELLOW}3. Actualizando referencias en archivos...${NC}"

# Actualizar archivos en el directorio del bot
if [ -d "$NEW_BOT_DIR" ]; then
    echo -e "${YELLOW}   Actualizando archivos en $NEW_BOT_DIR${NC}"
    find "$NEW_BOT_DIR" -type f -name "*.py" -o -name "*.sh" | while read -r file; do
        echo "      Procesando $file"
        sed -i 's/20m/15m/g' "$file"
        sed -i 's/20min/15min/g' "$file"
        sed -i 's/20 min/15 min/g' "$file"
        sed -i 's/sol_bot_20m/sol_bot_15m/g' "$file"
    done
    echo -e "${GREEN}   Archivos del bot actualizados${NC}"
fi

# Actualizar scripts de inicio
START_SCRIPT_OLD="${TRADING_BOTS_DIR}/start_sol_bot_20m.sh"
START_SCRIPT_NEW="${TRADING_BOTS_DIR}/start_sol_bot_15m.sh"

if [ -f "$START_SCRIPT_OLD" ]; then
    echo -e "${YELLOW}   Renombrando script de inicio...${NC}"
    mv "$START_SCRIPT_OLD" "$START_SCRIPT_NEW"
    sed -i 's/20m/15m/g' "$START_SCRIPT_NEW"
    sed -i 's/20min/15min/g' "$START_SCRIPT_NEW"
    sed -i 's/20 min/15 min/g' "$START_SCRIPT_NEW"
    sed -i 's/sol_bot_20m/sol_bot_15m/g' "$START_SCRIPT_NEW"
    chmod +x "$START_SCRIPT_NEW"
    echo -e "${GREEN}   Script de inicio actualizado${NC}"
else
    if [ -f "$START_SCRIPT_NEW" ]; then
        echo -e "${GREEN}   El script de inicio $START_SCRIPT_NEW ya existe. Actualizando...${NC}"
        sed -i 's/20m/15m/g' "$START_SCRIPT_NEW"
        sed -i 's/20min/15min/g' "$START_SCRIPT_NEW"
        sed -i 's/20 min/15 min/g' "$START_SCRIPT_NEW"
        sed -i 's/sol_bot_20m/sol_bot_15m/g' "$START_SCRIPT_NEW"
        chmod +x "$START_SCRIPT_NEW"
    else
        echo -e "${RED}   No se encontró ningún script de inicio. Creando uno nuevo...${NC}"
        cat > "$START_SCRIPT_NEW" << 'EOF'
#!/bin/bash
# Script para iniciar el bot SOL con reentrenamiento cada 15 minutos

# Cargar variables de entorno
source ~/.bashrc

# Directorio del bot
BOT_DIR="/home/edisonbautistaruiz2025/new-trading-bots/src/spot_bots/sol_bot_15m"

# Crear directorio de logs si no existe
mkdir -p "$BOT_DIR/logs"

# Iniciar el bot en una sesión de screen con redirección de errores
cd "$BOT_DIR"
screen -dmS sol_bot_15m bash -c "python3 main.py --use-ml --retrain-interval 15 --interval 15m --symbol SOLUSDT 2> logs/error.log"

echo "Bot SOL iniciado en sesión screen 'sol_bot_15m'"
echo "Para ver los logs: screen -r sol_bot_15m"
echo "Para ver errores: cat $BOT_DIR/logs/error.log"
EOF
        chmod +x "$START_SCRIPT_NEW"
        echo -e "${GREEN}   Script de inicio creado${NC}"
    fi
fi

# 4. Actualizar configuración de la API
echo -e "${YELLOW}4. Actualizando configuración de la API...${NC}"
ENV_FILE="${API_DIR}/.env"
CONFIG_FILE="${API_DIR}/app/config/bot_config.py"

# Actualizar archivo .env
if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}   Actualizando archivo .env...${NC}"
    # Hacer copia de seguridad
    cp "$ENV_FILE" "${ENV_FILE}.backup.$(date +%Y%m%d%H%M%S)"
    
    # Actualizar referencias
    sed -i 's|/sol_bot_20m|/sol_bot_15m|g' "$ENV_FILE"
    sed -i 's|sol_bot_20min|sol_bot_15min|g' "$ENV_FILE"
    echo -e "${GREEN}   Archivo .env actualizado${NC}"
else
    echo -e "${RED}   No se encontró el archivo .env de la API${NC}"
fi

# Actualizar archivo de configuración de bots
if [ -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}   Actualizando archivo de configuración de bots...${NC}"
    # Hacer copia de seguridad
    cp "$CONFIG_FILE" "${CONFIG_FILE}.backup.$(date +%Y%m%d%H%M%S)"
    
    # Actualizar referencias
    sed -i 's|"sol_bot_20m"|"sol_bot_15m"|g' "$CONFIG_FILE"
    sed -i 's|"interval": "20m"|"interval": "15m"|g' "$CONFIG_FILE"
    echo -e "${GREEN}   Archivo de configuración de bots actualizado${NC}"
else
    echo -e "${RED}   No se encontró el archivo de configuración de bots${NC}"
fi

# 5. Reiniciar servicios
echo -e "${YELLOW}5. ¿Quieres reiniciar los servicios ahora? (s/n)${NC}"
read -r respuesta
if [[ "$respuesta" =~ ^[Ss]$ ]]; then
    echo -e "${YELLOW}   Iniciando bot SOL...${NC}"
    "$START_SCRIPT_NEW"
    
    echo -e "${YELLOW}   Iniciando API...${NC}"
    cd "$API_DIR"
    screen -dmS trading_api python app.py
    echo -e "${GREEN}   Servicios iniciados${NC}"
else
    echo -e "${YELLOW}   No se reiniciaron los servicios. Puedes hacerlo manualmente cuando lo desees.${NC}"
fi

echo -e "${GREEN}=== Actualización completada ===${NC}"
echo -e "${YELLOW}Para iniciar el bot SOL:${NC} $START_SCRIPT_NEW"
echo -e "${YELLOW}Para iniciar la API:${NC} cd $API_DIR && screen -dmS trading_api python app.py"
echo -e "${YELLOW}Para ver los logs del bot:${NC} screen -r sol_bot_15m"
echo -e "${YELLOW}Para ver los logs de la API:${NC} screen -r trading_api"
echo -e "${YELLOW}Para salir de la vista de logs (sin detener el servicio):${NC} Presiona Ctrl+A y luego D"

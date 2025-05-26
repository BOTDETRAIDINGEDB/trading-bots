#!/bin/bash
# Script para clonar de forma segura el código desde la máquina virtual a GitHub
# y luego a una carpeta local para análisis y mejora

# Colores para mejor legibilidad
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Clonación segura para análisis y mejora ===${NC}"

# Solicitar credenciales
echo -e "${YELLOW}Ingresa tu nombre de usuario de GitHub:${NC}"
read -r GITHUB_USER

echo -e "${YELLOW}Ingresa tu token de GitHub:${NC}"
read -r GITHUB_TOKEN

# Directorio base
BASE_DIR="/home/edisonbautistaruiz2025"
VM_TRADING_BOTS_DIR="${BASE_DIR}/new-trading-bots"
VM_API_DIR="${BASE_DIR}/trading-bots-api"
TEMP_DIR="${BASE_DIR}/github_temp"
LOCAL_DIR="${BASE_DIR}/analysis"

# 1. Crear directorios temporales
echo -e "${YELLOW}1. Creando directorios temporales...${NC}"
rm -rf "$TEMP_DIR" 2>/dev/null
mkdir -p "$TEMP_DIR/trading-bots"
mkdir -p "$TEMP_DIR/trading-bots-api"
mkdir -p "$LOCAL_DIR/trading-bots"
mkdir -p "$LOCAL_DIR/trading-bots-api"
echo -e "${GREEN}   Directorios temporales creados${NC}"

# 2. Copiar todos los archivos (incluyendo .env para análisis local)
echo -e "${YELLOW}2. Copiando archivos para análisis...${NC}"
cp -r "$VM_TRADING_BOTS_DIR/"* "$LOCAL_DIR/trading-bots/"
cp -r "$VM_API_DIR/"* "$LOCAL_DIR/trading-bots-api/"
echo -e "${GREEN}   Archivos copiados para análisis local${NC}"

# 3. Crear versiones seguras para GitHub (sin información sensible)
echo -e "${YELLOW}3. Creando versiones seguras para GitHub...${NC}"

# Copiar archivos del bot
cp -r "$VM_TRADING_BOTS_DIR/"* "$TEMP_DIR/trading-bots/"

# Copiar archivos de la API
cp -r "$VM_API_DIR/"* "$TEMP_DIR/trading-bots-api/"

# 4. Eliminar información sensible de las versiones para GitHub
echo -e "${YELLOW}4. Eliminando información sensible...${NC}"

# Eliminar archivos y directorios sensibles
find "$TEMP_DIR" -name ".env" -type f -delete
find "$TEMP_DIR" -name "venv" -type d -exec rm -rf {} + 2>/dev/null || true
find "$TEMP_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$TEMP_DIR" -name "*.pyc" -delete
find "$TEMP_DIR" -name "*.log" -delete
find "$TEMP_DIR" -name "*_state.json" -delete

# Crear versiones de ejemplo de los archivos .env
for dir in "$TEMP_DIR/trading-bots" "$TEMP_DIR/trading-bots-api"; do
    if [ -f "$dir/.env" ]; then
        cp "$dir/.env" "$dir/.env.example"
        
        # Reemplazar valores sensibles
        sed -i 's/\(BINANCE_API_KEY=\).*/\1your_binance_api_key_here/' "$dir/.env.example"
        sed -i 's/\(BINANCE_API_SECRET=\).*/\1your_binance_api_secret_here/' "$dir/.env.example"
        sed -i 's/\(TELEGRAM_BOT_TOKEN=\).*/\1your_telegram_token_here/' "$dir/.env.example"
        sed -i 's/\(TELEGRAM_CHAT_ID=\).*/\1your_chat_id_here/' "$dir/.env.example"
        sed -i 's/\(API_KEY=\).*/\1your_api_key_here/' "$dir/.env.example"
        sed -i 's/\(JWT_SECRET=\).*/\1your_jwt_secret_here/' "$dir/.env.example"
        sed -i 's/\(API_SECRET_KEY=\).*/\1your_api_secret_key_here/' "$dir/.env.example"
        
        # Eliminar el archivo .env original
        rm "$dir/.env"
    fi
done

echo -e "${GREEN}   Información sensible eliminada${NC}"

# 5. Inicializar repositorios Git
echo -e "${YELLOW}5. Inicializando repositorios Git...${NC}"

# Inicializar repositorio de bots
cd "$TEMP_DIR/trading-bots"
git init
git add .
git config user.email "edisonbautistaruiz2025@gmail.com"
git config user.name "$GITHUB_USER"
git commit -m "Versión completa del bot SOL con intervalo de 15m para análisis"

# Inicializar repositorio de API
cd "$TEMP_DIR/trading-bots-api"
git init
git add .
git config user.email "edisonbautistaruiz2025@gmail.com"
git config user.name "$GITHUB_USER"
git commit -m "Versión de la API para análisis y mejora"

echo -e "${GREEN}   Repositorios Git inicializados${NC}"

# 6. Crear repositorios en GitHub (si no existen)
echo -e "${YELLOW}6. Creando repositorios en GitHub para análisis...${NC}"

# Crear repositorio para bots
cd "$TEMP_DIR/trading-bots"
git remote add origin "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/trading-bots-analysis.git"
git push -f origin master

# Crear repositorio para API
cd "$TEMP_DIR/trading-bots-api"
git remote add origin "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/trading-bots-api-analysis.git"
git push -f origin master

echo -e "${GREEN}   Repositorios creados en GitHub${NC}"

# 7. Generar informe de análisis
echo -e "${YELLOW}7. Generando informe de análisis...${NC}"

REPORT_FILE="$LOCAL_DIR/analysis_report.md"

cat > "$REPORT_FILE" << 'EOF'
# Informe de Análisis del Bot de Trading

## Estructura del Proyecto

### Bot de Trading
```
trading-bots/
├── src/
│   ├── spot_bots/
│   │   ├── sol_bot_15m/
│   │   │   ├── main.py
│   │   │   ├── src/
│   │   │   │   ├── data/
│   │   │   │   ├── models/
│   │   │   │   ├── strategies/
│   │   │   │   └── utils/
```

### API
```
trading-bots-api/
├── app/
│   ├── config/
│   ├── controllers/
│   ├── models/
│   ├── routes/
│   └── utils/
├── app.py
```

## Problemas Identificados

1. **Conexión con la API**: El bot no puede conectarse a la API. Esto podría deberse a:
   - La API no está ejecutándose
   - La URL de la API en el archivo .env es incorrecta
   - El puerto de la API está bloqueado

2. **Notificaciones de Telegram**: Las notificaciones de Telegram están funcionando, pero podrían mejorarse.

3. **Intervalo de Trading**: El bot está configurado para usar un intervalo de 15 minutos, que es válido en Binance.

## Recomendaciones

1. **Conexión con la API**:
   - Verificar que la API esté ejecutándose: `ps aux | grep "python3 app.py"`
   - Actualizar la URL de la API en el archivo .env: `API_URL=http://<IP-CORRECTA>:5000`
   - Verificar que el puerto 5000 esté abierto: `netstat -tuln | grep 5000`

2. **Mejoras en el Código**:
   - Añadir más comentarios para facilitar el mantenimiento
   - Implementar manejo de errores más robusto
   - Mejorar el sistema de logs para facilitar el diagnóstico de problemas

3. **Seguridad**:
   - Asegurarse de que los archivos .env estén en .gitignore
   - Revisar permisos de archivos sensibles: `chmod 600 .env`
   - Considerar el uso de variables de entorno en lugar de archivos .env

## Próximos Pasos

1. Corregir la conexión con la API
2. Implementar las mejoras de código recomendadas
3. Realizar pruebas exhaustivas en modo simulación
4. Monitorear el rendimiento del bot
EOF

echo -e "${GREEN}   Informe de análisis generado en $REPORT_FILE${NC}"

# 8. Limpiar
echo -e "${YELLOW}8. Limpiando archivos temporales...${NC}"
rm -rf "$TEMP_DIR"
echo -e "${GREEN}   Archivos temporales eliminados${NC}"

echo -e "${GREEN}=== Proceso completado correctamente ===${NC}"
echo -e "${YELLOW}Los repositorios han sido creados en:${NC}"
echo -e "   https://github.com/${GITHUB_USER}/trading-bots-analysis"
echo -e "   https://github.com/${GITHUB_USER}/trading-bots-api-analysis"
echo -e "${YELLOW}Los archivos para análisis local están en:${NC}"
echo -e "   $LOCAL_DIR/trading-bots"
echo -e "   $LOCAL_DIR/trading-bots-api"
echo -e "${YELLOW}El informe de análisis está en:${NC}"
echo -e "   $REPORT_FILE"

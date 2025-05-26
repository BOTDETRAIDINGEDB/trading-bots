#!/bin/bash
# Script para clonar directamente el código a GitHub usando token
# Con protecciones adicionales para información sensible y reducción de falsos positivos

# Colores para mejor legibilidad
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Clonando código de forma segura a GitHub ===${NC}"

# Solicitar credenciales
echo -e "${YELLOW}Ingresa tu nombre de usuario de GitHub:${NC}"
read -r GITHUB_USER

echo -e "${YELLOW}Ingresa tu token de GitHub:${NC}"
read -r GITHUB_TOKEN

# Directorio base
BASE_DIR="/home/edisonbautistaruiz2025"
TRADING_BOTS_DIR="${BASE_DIR}/new-trading-bots"
API_DIR="${BASE_DIR}/trading-bots-api"
TEMP_DIR="${BASE_DIR}/github_temp"

# 1. Crear directorio temporal
echo -e "${YELLOW}1. Creando directorio temporal...${NC}"
rm -rf "$TEMP_DIR" 2>/dev/null
mkdir -p "$TEMP_DIR/trading-bots"
mkdir -p "$TEMP_DIR/trading-bots-api"
echo -e "${GREEN}   Directorio temporal creado${NC}"

# 2. Clonar repositorios existentes
echo -e "${YELLOW}2. Clonando repositorios existentes...${NC}"
cd "$TEMP_DIR/trading-bots"
git clone "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/BOTDETRAIDINGEDB/trading-bots.git" .
if [ $? -ne 0 ]; then
    echo -e "${RED}   Error al clonar el repositorio de bots. Inicializando nuevo repositorio...${NC}"
    git init
fi

cd "$TEMP_DIR/trading-bots-api"
git clone "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/BOTDETRAIDINGEDB/trading-bots-api.git" .
if [ $? -ne 0 ]; then
    echo -e "${RED}   Error al clonar el repositorio de API. Inicializando nuevo repositorio...${NC}"
    git init
fi
echo -e "${GREEN}   Repositorios clonados${NC}"

# 3. Limpiar repositorios (mantener solo .git)
echo -e "${YELLOW}3. Limpiando repositorios...${NC}"
cd "$TEMP_DIR/trading-bots"
find . -mindepth 1 -not -path "./.git*" -delete

cd "$TEMP_DIR/trading-bots-api"
find . -mindepth 1 -not -path "./.git*" -delete
echo -e "${GREEN}   Repositorios limpiados${NC}"

# 4. Copiar archivos del bot (excluyendo directorios y archivos sensibles)
echo -e "${YELLOW}4. Copiando archivos del bot (excluyendo directorios y archivos sensibles)...${NC}"
# Excluir directorios y archivos sensibles
mkdir -p "$TEMP_DIR/trading-bots/src"
mkdir -p "$TEMP_DIR/trading-bots/scripts"

# Copiar solo los directorios y archivos relevantes
cp -r "$TRADING_BOTS_DIR/src/spot_bots" "$TEMP_DIR/trading-bots/src/"
cp "$TRADING_BOTS_DIR"/*.sh "$TEMP_DIR/trading-bots/scripts/" 2>/dev/null || true
cp "$TRADING_BOTS_DIR"/*.py "$TEMP_DIR/trading-bots/" 2>/dev/null || true
cp "$TRADING_BOTS_DIR/README.md" "$TEMP_DIR/trading-bots/" 2>/dev/null || true

echo -e "${GREEN}   Archivos del bot copiados${NC}"

# 5. Copiar archivos de la API (excluyendo directorios y archivos sensibles)
echo -e "${YELLOW}5. Copiando archivos de la API (excluyendo directorios y archivos sensibles)...${NC}"
# Excluir directorios y archivos sensibles
mkdir -p "$TEMP_DIR/trading-bots-api/app"
mkdir -p "$TEMP_DIR/trading-bots-api/config"

# Copiar solo los directorios y archivos relevantes
cp -r "$API_DIR/app" "$TEMP_DIR/trading-bots-api/"
cp "$API_DIR/app.py" "$TEMP_DIR/trading-bots-api/" 2>/dev/null || true
cp "$API_DIR/update_api_config.sh" "$TEMP_DIR/trading-bots-api/" 2>/dev/null || true
cp "$API_DIR/README.md" "$TEMP_DIR/trading-bots-api/" 2>/dev/null || true
cp "$API_DIR/.env.example" "$TEMP_DIR/trading-bots-api/" 2>/dev/null || true

echo -e "${GREEN}   Archivos de la API copiados${NC}"

# 6. Eliminar información sensible
echo -e "${YELLOW}6. Eliminando información sensible...${NC}"

# Eliminar directorios que no deben subirse a GitHub
find "$TEMP_DIR" -name "venv" -type d -exec rm -rf {} + 2>/dev/null || true
find "$TEMP_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$TEMP_DIR" -name "*.pyc" -delete
find "$TEMP_DIR" -name "*.log" -delete
find "$TEMP_DIR" -name "*_state.json" -delete

# Crear versiones de ejemplo de los archivos .env
find "$TEMP_DIR" -name ".env" | while read -r env_file; do
    example_file="${env_file}.example"
    cp "$env_file" "$example_file"
    
    # Reemplazar valores sensibles
    sed -i 's/\(BINANCE_API_KEY=\).*/\1your_binance_api_key_here/' "$example_file"
    sed -i 's/\(BINANCE_API_SECRET=\).*/\1your_binance_api_secret_here/' "$example_file"
    sed -i 's/\(TELEGRAM_BOT_TOKEN=\).*/\1your_telegram_token_here/' "$example_file"
    sed -i 's/\(TELEGRAM_CHAT_ID=\).*/\1your_chat_id_here/' "$example_file"
    sed -i 's/\(API_KEY=\).*/\1your_api_key_here/' "$example_file"
    sed -i 's/\(JWT_SECRET=\).*/\1your_jwt_secret_here/' "$example_file"
    sed -i 's/\(API_SECRET_KEY=\).*/\1your_api_secret_key_here/' "$example_file"
    
    # Eliminar el archivo .env original
    rm "$env_file"
done

echo -e "${GREEN}   Información sensible eliminada${NC}"

# 7. Crear archivos README
echo -e "${YELLOW}7. Creando archivos README...${NC}"

# README para el repositorio de bots
cat > "$TEMP_DIR/trading-bots/README.md" << 'EOF'
# Trading Bots

Este repositorio contiene bots de trading para diferentes criptomonedas, incluyendo:

- Bot SOL con intervalo de 15 minutos y reentrenamiento de modelo ML
- Bot XRP con intervalo de 30 minutos

## Estructura del Proyecto

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
│   │   ├── xrp_bot/
│   │   │   ├── ...
├── scripts/
│   ├── update_vm.sh
│   ├── debug_bot.sh
│   ├── convert_to_15m.sh
│   └── fix_all_references.py
```

## Configuración

1. Crea un archivo `.env` en el directorio del bot basado en `.env.example`
2. Ejecuta el bot con:
   ```
   python main.py --use-ml --retrain-interval 15 --interval 15m --symbol SOLUSDT
   ```

## Scripts de Utilidad

- `update_vm.sh`: Actualiza el bot en la máquina virtual
- `debug_bot.sh`: Verifica la configuración del bot y diagnostica problemas
- `convert_to_15m.sh`: Convierte el bot de 20m a 15m
- `fix_all_references.py`: Corrige todas las referencias de 20m a 15m
EOF

# README para el repositorio de la API
cat > "$TEMP_DIR/trading-bots-api/README.md" << 'EOF'
# Trading Bots API

API para monitorear y controlar los bots de trading.

## Estructura del Proyecto

```
trading-bots-api/
├── app/
│   ├── config/
│   ├── controllers/
│   ├── models/
│   ├── routes/
│   └── utils/
├── app.py
├── update_api_config.sh
```

## Configuración

1. Crea un archivo `.env` basado en `.env.example`
2. Ejecuta la API con:
   ```
   python app.py
   ```

## Scripts de Utilidad

- `update_api_config.sh`: Actualiza la configuración de la API para incluir nuevos bots
EOF

echo -e "${GREEN}   Archivos README creados${NC}"

# 8. Verificación final de seguridad
echo -e "${YELLOW}8. Realizando verificación final de seguridad...${NC}"
echo -e "${YELLOW}   ¿Deseas continuar con la subida a GitHub? (s/n)${NC}"
read -r respuesta
if [[ "$respuesta" =~ ^[Ss]$ ]]; then
    echo -e "${GREEN}   Continuando con la subida a GitHub...${NC}"
else
    echo -e "${RED}   Operación cancelada por el usuario.${NC}"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# 9. Subir cambios a GitHub
echo -e "${YELLOW}9. Subiendo cambios a GitHub...${NC}"

# Subir cambios del bot
cd "$TEMP_DIR/trading-bots"
git add .
git config user.email "edisonbautistaruiz2025@gmail.com"
git config user.name "$GITHUB_USER"
git commit -m "Actualización completa del bot SOL con intervalo de 15m"
git push -f "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/BOTDETRAIDINGEDB/trading-bots.git" HEAD:main
if [ $? -eq 0 ]; then
    echo -e "${GREEN}   Cambios del bot subidos correctamente${NC}"
else
    echo -e "${RED}   Error al subir cambios del bot${NC}"
fi

# Subir cambios de la API
cd "$TEMP_DIR/trading-bots-api"
git add .
git config user.email "edisonbautistaruiz2025@gmail.com"
git config user.name "$GITHUB_USER"
git commit -m "Actualización de la API para soportar el bot SOL con intervalo de 15m"
git push -f "https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/BOTDETRAIDINGEDB/trading-bots-api.git" HEAD:main
if [ $? -eq 0 ]; then
    echo -e "${GREEN}   Cambios de la API subidos correctamente${NC}"
else
    echo -e "${RED}   Error al subir cambios de la API${NC}"
fi

# 10. Limpiar
echo -e "${YELLOW}10. Limpiando archivos temporales...${NC}"
rm -rf "$TEMP_DIR"
echo -e "${GREEN}   Archivos temporales eliminados${NC}"

echo -e "${GREEN}=== Clonación segura a GitHub completada ===${NC}"
echo -e "${YELLOW}Los repositorios han sido actualizados en:${NC}"
echo -e "   https://github.com/BOTDETRAIDINGEDB/trading-bots"
echo -e "   https://github.com/BOTDETRAIDINGEDB/trading-bots-api"

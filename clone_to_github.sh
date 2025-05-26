#!/bin/bash
# Script para clonar el código completo de la máquina virtual a GitHub

# Colores para mejor legibilidad
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Preparando código para subir a GitHub ===${NC}"

# Directorio base
BASE_DIR="/home/edisonbautistaruiz2025"
TRADING_BOTS_DIR="${BASE_DIR}/new-trading-bots"
API_DIR="${BASE_DIR}/trading-bots-api"
TEMP_DIR="${BASE_DIR}/github_temp"

# 1. Crear directorio temporal
echo -e "${YELLOW}1. Creando directorio temporal...${NC}"
rm -rf "$TEMP_DIR" 2>/dev/null
mkdir -p "$TEMP_DIR"
mkdir -p "$TEMP_DIR/trading-bots"
mkdir -p "$TEMP_DIR/trading-bots-api"
echo -e "${GREEN}   Directorio temporal creado${NC}"

# 2. Copiar archivos del bot
echo -e "${YELLOW}2. Copiando archivos del bot...${NC}"
cp -r "$TRADING_BOTS_DIR"/* "$TEMP_DIR/trading-bots/"
# Copiar archivos ocultos (excepto .git)
find "$TRADING_BOTS_DIR" -name ".*" -not -name ".git" -exec cp -r {} "$TEMP_DIR/trading-bots/" \; 2>/dev/null || true
echo -e "${GREEN}   Archivos del bot copiados${NC}"

# 3. Copiar archivos de la API
echo -e "${YELLOW}3. Copiando archivos de la API...${NC}"
cp -r "$API_DIR"/* "$TEMP_DIR/trading-bots-api/"
# Copiar archivos ocultos (excepto .git)
find "$API_DIR" -name ".*" -not -name ".git" -exec cp -r {} "$TEMP_DIR/trading-bots-api/" \; 2>/dev/null || true
echo -e "${GREEN}   Archivos de la API copiados${NC}"

# 4. Eliminar información sensible
echo -e "${YELLOW}4. Eliminando información sensible...${NC}"

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

# Eliminar archivos de log y estado
find "$TEMP_DIR" -name "*.log" -delete
find "$TEMP_DIR" -name "*_state.json" -delete
find "$TEMP_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$TEMP_DIR" -name "*.pyc" -delete

echo -e "${GREEN}   Información sensible eliminada${NC}"

# 5. Crear archivos README
echo -e "${YELLOW}5. Creando archivos README...${NC}"

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

# 6. Crear scripts para clonar a GitHub
echo -e "${YELLOW}6. Creando scripts para clonar a GitHub...${NC}"

# Script para el repositorio de bots
cat > "$TEMP_DIR/trading-bots/push_to_github.sh" << 'EOF'
#!/bin/bash
# Script para subir el código a GitHub

# Configurar Git
git init
git add .
git commit -m "Actualización completa del bot SOL con intervalo de 15m"

# Agregar el repositorio remoto (reemplaza la URL con tu repositorio)
git remote add origin https://github.com/BOTDETRAIDINGEDB/trading-bots.git

# Forzar push al repositorio remoto (cuidado: esto sobrescribirá el repositorio remoto)
git push -f origin master

echo "Código subido a GitHub correctamente"
EOF
chmod +x "$TEMP_DIR/trading-bots/push_to_github.sh"

# Script para el repositorio de la API
cat > "$TEMP_DIR/trading-bots-api/push_to_github.sh" << 'EOF'
#!/bin/bash
# Script para subir el código a GitHub

# Configurar Git
git init
git add .
git commit -m "Actualización de la API para soportar el bot SOL con intervalo de 15m"

# Agregar el repositorio remoto (reemplaza la URL con tu repositorio)
git remote add origin https://github.com/BOTDETRAIDINGEDB/trading-bots-api.git

# Forzar push al repositorio remoto (cuidado: esto sobrescribirá el repositorio remoto)
git push -f origin master

echo "Código subido a GitHub correctamente"
EOF
chmod +x "$TEMP_DIR/trading-bots-api/push_to_github.sh"

echo -e "${GREEN}   Scripts para clonar a GitHub creados${NC}"

# 7. Comprimir archivos
echo -e "${YELLOW}7. Comprimiendo archivos...${NC}"
cd "$BASE_DIR"
tar -czf trading-bots-github.tar.gz -C "$TEMP_DIR" .
echo -e "${GREEN}   Archivos comprimidos en $BASE_DIR/trading-bots-github.tar.gz${NC}"

# 8. Limpiar
echo -e "${YELLOW}8. Limpiando archivos temporales...${NC}"
rm -rf "$TEMP_DIR"
echo -e "${GREEN}   Archivos temporales eliminados${NC}"

echo -e "${GREEN}=== Preparación completada ===${NC}"
echo -e "${YELLOW}El archivo comprimido está en:${NC} $BASE_DIR/trading-bots-github.tar.gz"
echo -e "${YELLOW}Para descargar el archivo a tu máquina local:${NC}"
echo -e "   scp edisonbautistaruiz2025@iatraidingbots:~/trading-bots-github.tar.gz ."
echo -e "${YELLOW}Luego, descomprime el archivo y ejecuta los scripts push_to_github.sh en cada directorio.${NC}"

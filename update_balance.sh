#!/bin/bash
# Script para actualizar el balance en start_enhanced.sh sin afectar otros cambios

# Ruta al archivo
FILE="src/spot_bots/sol_bot_15m/start_enhanced.sh"

# Verificar si el archivo existe
if [ ! -f "$FILE" ]; then
    echo "Error: No se encontró el archivo $FILE"
    exit 1
fi

# Hacer backup del archivo original
cp "$FILE" "${FILE}.bak"
echo "Backup creado en ${FILE}.bak"

# Reemplazar la línea del balance
sed -i 's/BALANCE=${BALANCE:-[0-9]*}/BALANCE=${BALANCE:-100}/g' "$FILE"

# Verificar si el cambio se realizó
if grep -q "BALANCE=\${BALANCE:-100}" "$FILE"; then
    echo "Balance actualizado correctamente a 100 USDT"
else
    echo "Error: No se pudo actualizar el balance"
    # Restaurar backup
    cp "${FILE}.bak" "$FILE"
    echo "Archivo original restaurado desde backup"
    exit 1
fi

echo "Puedes reiniciar el bot con: cd src/spot_bots/sol_bot_15m && ./start_enhanced.sh"

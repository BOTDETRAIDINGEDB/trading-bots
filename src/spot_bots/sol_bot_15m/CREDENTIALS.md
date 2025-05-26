# Gestión de Credenciales - Bot SOL Adaptativo

## Importante: Seguridad de Credenciales

Este bot utiliza las credenciales almacenadas en el archivo `credentials.json` ubicado en el repositorio `trading-bots-api`.

### Ubicaciones del archivo credentials.json

El bot buscará el archivo `credentials.json` en las siguientes ubicaciones:

1. Servidor: `/home/edisonbautistaruiz2025/trading-bots-api/credentials.json`
2. Desarrollo local: `~/Documents/GitHub/trading-bots-api/credentials.json`

### Estructura del archivo credentials.json

El archivo debe contener la siguiente estructura:

```json
{
    "env": {
        "BINANCE_API_KEY": "tu_api_key_de_binance",
        "BINANCE_API_SECRET": "tu_api_secret_de_binance",
        "TELEGRAM_BOT_TOKEN": "tu_token_de_telegram",
        "TELEGRAM_CHAT_ID": "tu_chat_id_de_telegram",
        "API_KEY": "tu_api_key",
        "SOL_BOT_SIMULATION": "True"
    }
}
```

### ⚠️ ADVERTENCIA DE SEGURIDAD ⚠️

**NUNCA subas el archivo `credentials.json` a GitHub o cualquier otro repositorio público.**

Este archivo contiene información sensible que debe mantenerse privada.

### Cómo funciona

El bot utiliza el módulo `utils.credentials_loader` para cargar las credenciales desde `credentials.json` y establecerlas como variables de entorno, lo que permite que sean accesibles a través de `os.getenv()` en todo el código.

### Solución de problemas

Si el bot no puede encontrar o cargar las credenciales:

1. Verifica que el archivo `credentials.json` exista en una de las ubicaciones mencionadas
2. Comprueba que el formato del archivo sea JSON válido
3. Asegúrate de que todas las credenciales necesarias estén presentes

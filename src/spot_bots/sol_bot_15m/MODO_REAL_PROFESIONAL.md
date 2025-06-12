# Bot SOL - Modo Real Profesional

## 🚨 **SCRIPT PROFESIONAL PARA TRADING REAL**

Este documento describe el uso del nuevo script `start_cloud_real.sh`, diseñado específicamente para operar con dinero real siguiendo las más altas prácticas de seguridad y profesionalismo.

---

## ✅ **Características del Script Profesional**

### **🛡️ Validaciones de Seguridad**
- ✅ Verificación automática de dependencias (Python3, Screen, Curl)
- ✅ Validación de credenciales de Binance antes del inicio
- ✅ Confirmación obligatoria del usuario (debe escribir "SI" en mayúsculas)
- ✅ Verificación de instancias previas del bot
- ✅ Verificación de balance mínimo requerido (50 USDT)

### **🎯 Parámetros Optimizados para Real**
- **Riesgo conservador:** 1.5% por operación (vs 2% en simulación)
- **Reentrenamiento:** Cada 90 minutos (vs 15 en simulación)
- **Actualizaciones de estado:** Cada 4 horas
- **Balance mínimo:** 50 USDT para mayor seguridad

### **📊 Gestión Profesional de Logs**
- Logs con timestamp y códigos de colores
- Archivo único por ejecución: `sol_bot_real_YYYYMMDD_HHMMSS.log`
- Separación clara entre logs de real y simulación

---

## 🚀 **Cómo Usar el Script**

### **Paso 1: Preparación**
```bash
# Conectar a la máquina virtual
ssh edisonbautistaruiz2025@iatraidingbots

# Navegar al directorio del bot
cd ~/new-trading-bots/src/spot_bots/sol_bot_15m
```

### **Paso 2: Verificaciones Previas**
Antes de ejecutar, asegúrese de:
- ✅ Tener al menos 50 USDT en su cuenta de Binance
- ✅ Credenciales configuradas correctamente
- ✅ No tener otras instancias del bot ejecutándose

### **Paso 3: Ejecutar el Script**
```bash
./start_cloud_real.sh
```

**El script le mostrará:**
1. Banner profesional con advertencias
2. Validación de dependencias y credenciales
3. Configuración detallada del bot
4. **Solicitud de confirmación (debe escribir "SI")**
5. Inicio del bot en sesión screen

---

## 📋 **Proceso de Confirmación**

Cuando ejecute el script, verá:

```
⚠️⚠️⚠️  CONFIRMACIÓN REQUERIDA  ⚠️⚠️⚠️

Está a punto de iniciar el bot en MODO REAL con dinero real.
El bot realizará operaciones de compra y venta automáticamente.

¿Está seguro de que desea continuar? (escriba 'SI' en mayúsculas)
```

**⚠️ IMPORTANTE:** 
- Debe escribir exactamente `SI` (en mayúsculas)
- Cualquier otra respuesta cancelará la operación
- Esta confirmación es obligatoria por seguridad

---

## 🖥️ **Comandos de Monitoreo**

Una vez iniciado el bot, use estos comandos:

### **Ver el Bot en Tiempo Real**
```bash
screen -r sol_bot_real
```
*Para salir sin detener: `Ctrl+A`, luego `D`*

### **Ver Logs**
```bash
# Ver logs en tiempo real
tail -f logs/sol_bot_real_*.log

# Ver logs guardados
ls -la logs/sol_bot_real_*
```

### **Detener el Bot**
```bash
./stop.sh
```

### **Estado de Sesiones**
```bash
screen -ls
```

---

## ⚙️ **Configuración Técnica**

### **Parámetros del Bot Real**
```bash
SYMBOL="SOLUSDT"              # Par de trading
INTERVAL="15m"                # Velas de 15 minutos
RISK="0.015"                  # 1.5% riesgo por operación
RETRAIN_INTERVAL="90"         # Reentrenamiento cada 90 min
STATUS_INTERVAL="4"           # Estado cada 4 horas
MIN_BALANCE="50.0"           # Balance mínimo requerido
```

### **Archivos Importantes**
- **PID:** `sol_bot_real.pid`
- **Logs:** `logs/sol_bot_real_TIMESTAMP.log`
- **Credenciales:** `~/.binance_credentials`

---

## 🚨 **Advertencias y Mejores Prácticas**

### **⚠️ Antes de Iniciar**
1. **Verifique su balance** en Binance (mínimo 50 USDT)
2. **Revise las condiciones del mercado** 
3. **Asegúrese de entender los riesgos** del trading automatizado
4. **Configure alertas** para monitorear las operaciones

### **🛡️ Durante la Operación**
1. **Monitoree regularmente** el estado del bot
2. **Revise los logs** para detectar anomalías
3. **Mantenga un balance de reserva** para volatilidad
4. **No interfiera manualmente** mientras el bot opera

### **🔄 Mantenimiento**
1. **Actualice regularmente** desde GitHub
2. **Respalde los logs importantes**
3. **Revise las métricas de rendimiento** semanalmente
4. **Mantenga las credenciales actualizadas**

---

## 🆘 **Resolución de Problemas**

### **Error: "Dependencia faltante"**
```bash
# Instalar dependencias faltantes
sudo apt update
sudo apt install python3 screen curl
```

### **Error: "Credenciales no encontradas"**
```bash
# Verificar ubicación de credenciales
ls -la ~/.binance_credentials
cat ~/.binance_credentials
```

### **Error: "Bot ya ejecutándose"**
```bash
# Detener instancia previa
./stop.sh

# O limpiar sesiones manualmente
screen -ls
screen -S sol_bot_real -X quit
```

### **Error: "Balance insuficiente"**
- Deposite al menos 50 USDT en su cuenta de Binance
- Verifique que el balance esté disponible para trading spot

---

## 📞 **Contacto y Soporte**

- **Autor:** Edison Bautista
- **Versión del Script:** 1.0.0
- **Fecha:** 2025-06-12

**⚠️ Recuerde: Este bot opera con dinero real. Use bajo su propia responsabilidad y mantenga siempre el control sobre sus inversiones.**

# Integración de YOLO
Este repositorio muestra el funcionamiento de micropython por medio de una pequeña simulación en wokwi como primer, punto. En un segundo apartado muestra la implementación de YOLO, el cual tiene como funcionamiento la detección de distintos objetos

# Control de LEDs con Botones (MicroPython en ESP32)

##  Esquema de Conexiones (Pinout)

Basado en el diagrama de hardware, las conexiones a la placa ESP32 se han distribuido de la siguiente manera:

*   **Entradas (Botones):**
    *   🔴 Botón Rojo: Conectado al pin **GPIO 26**.
    *   🔵 Botón Azul: Conectado al pin **GPIO 27**.
*   **Salidas (LEDs):**
    *   🔴 LED Rojo: Conectado al pin **GPIO 4**.
    *   🟢 LED Verde: Conectado al pin **GPIO 2**.

![Esquema de Conexión de LEDs](imagenes/g.png)

## Explicación Código
El programa importa la clase Pin para interactuar físicamente con los pines de la ESP32. Se declaran los LEDs como salidas (Pin.OUT) y los botones como entradas (Pin.IN). Se habilita el modo Pin.PULL_UP, lo que obliga al microcontrolador a leer un "1 lógico" cuando el botón está suelto, y un "0 lógico" cuando el usuario lo presiona y cierra el circuito hacia tierra (GND). En el bucle while True, el código evalúa estas condiciones: si lee un 0, envía voltaje al LED para encenderlo (value(1)); si no, lo apaga (value(0)). El comando time.sleep(0.05) previene lecturas falsas causadas por el rebote mecánico del botón.

##  Código Fuente (`main.py`)

El siguiente código está escrito en MicroPython. Configura los pines de entrada con resistencias *Pull-Up* internas (ya que los botones cierran el circuito hacia Tierra) y evalúa constantemente si han sido presionados.

```python
from machine import Pin
import time

# 1. Configuración de los pines de salida (LEDs)
led_rojo = Pin(4, Pin.OUT)
led_verde = Pin(2, Pin.OUT)

# 2. Configuración de los pines de entrada (Botones con Pull-Up)
# Al usar PULL_UP, el pin leerá '1' normalmente y '0' al ser presionado.
boton_rojo = Pin(26, Pin.IN, Pin.PULL_UP)
boton_azul = Pin(27, Pin.IN, Pin.PULL_UP)

print("Iniciando control de LEDs...")

# 3. Bucle principal de control
while True:
    # Control del LED Rojo
    if boton_rojo.value() == 0:  # Si el botón rojo es presionado (conecta a GND)
        led_rojo.value(1)        # Encender LED rojo
    else:
        led_rojo.value(0)        # Apagar LED rojo

    # Control del LED Verde
    if boton_azul.value() == 0:  # Si el botón azul es presionado
        led_verde.value(1)       # Encender LED verde
    else:
        led_verde.value(0)       # Apagar LED verde

    # Pequeña pausa para evitar rebotes (debounce) y no saturar el procesador
    time.sleep(0.05)
```


# Detector de Objetos Específicos con YOLO-World

Este script en Python está diseñado para realizar detección de objetos en tiempo real utilizando una cámara web estándar. A diferencia de los modelos de detección convencionales, implementa YOLO-World, un modelo de arquitectura Zero-Shot que permite buscar e identificar elementos específicos basándose únicamente en descripciones de texto, sin necesidad de entrenar un modelo personalizado.

Este código está configurado específicamente para detectar un **carrito Hot Wheels** y una **caja de audífonos negra**.

##  Características Principales

*   **Búsqueda por Texto (Zero-Shot):** Las clases a detectar se definen dinámicamente mediante una lista de palabras en inglés (`"blue hot wheels toy car"`, `"black headphones box"`), reconfigurando la red neuronal al instante.
*   **Filtrado de Confianza Adaptativo:** Incluye un ajuste de sensibilidad (`conf=0.15`) diseñado para equilibrar la detección de objetos oscuros o translúcidos en entornos con condiciones de iluminación limitadas o contraluz.
*   **Diagnóstico de Clases:** Verifica por consola que el modelo haya sobreescrito correctamente las categorías por defecto antes de iniciar la inferencia en la cámara.

##  Requisitos de Instalación

Para ejecutar este proyecto, necesitas tener instalado Python y las siguientes dependencias:

```bash
pip install opencv-python ultralytics
```
## Explicación de código
El script funciona en tres etapas principales. Primero, carga los pesos del modelo YOLO-World y sustituye las clases por defecto por las descripciones personalizadas. Segundo, inicializa la cámara web a una resolución fija (640x480) para mantener un rendimiento óptimo. Finalmente, entra en un bucle infinito donde captura cada fotograma del video, lo pasa por la red neuronal con una confianza baja (15%) para captar objetos difíciles, y utiliza la librería OpenCV (cv2) para dibujar las cajas superpuestas y mostrar la ventana al usuario hasta que presiona la tecla 'q'.

## Codigo

```python
import cv2
from ultralytics import YOLO

# 1. Cargar el modelo YOLO-World (Zero-Shot Detection)
model = YOLO('yolov8s-world.pt')

# 2. Definir las clases personalizadas específicas
# Usamos descripciones precisas en inglés para ayudar al modelo con el contraste
clases = ["blue hot wheels toy car", "black headphones box"]
model.set_classes(clases)

# DEBUG: Verificar en terminal qué está buscando realmente el modelo
print("=========================================")
print("CLASES ACTIVAS EN EL MODELO:", model.names)
print("=========================================")

# 3. Inicializar la cámara web
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
    exit()

print("Cámara iniciada. Presiona 'q' para salir.")

# 4. Bucle principal de inferencia
while True:
    ret, frame = cap.read()
    if not ret: 
        print("Error al leer el frame de la cámara.")
        break

    # Realizar detección. 
    # conf=0.15: Umbral ajustado para compensar el material translúcido y cajas oscuras
    results = model(frame, conf=0.15, verbose=False)
    
    # Dibujar los recuadros delimitadores y etiquetas
    annotated_frame = results[0].plot()
    
    # Mostrar el resultado en pantalla
    cv2.imshow('Deteccion Especifica (YOLO-World)', annotated_frame)

    # Condición de salida (Tecla 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos de hardware y ventanas
cap.release()
cv2.destroyAllWindows()
```



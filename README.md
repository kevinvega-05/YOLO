# Detector de Objetos Específicos con YOLO-World

Este repositorio contiene un script en Python diseñado para realizar detección de objetos en tiempo real utilizando una cámara web estándar. A diferencia de los modelos de detección convencionales, este proyecto implementa **YOLO-World**, un modelo de arquitectura *Zero-Shot* que permite buscar e identificar elementos específicos basándose únicamente en descripciones de texto, sin necesidad de entrenar un modelo personalizado.

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




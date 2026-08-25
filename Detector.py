import cv2
from ultralytics import YOLO

# Cargar el modelo YOLO-World
model = YOLO('yolov8s-world.pt')

# Simplificación extrema de las clases
clases = ["car", "box", "toy"]
model.set_classes(clases)

# DEBUG: Verificar qué está buscando realmente el modelo
print("=========================================")
print("CLASES ACTIVAS EN EL MODELO:", model.names)
print("=========================================")

# Abrir cámara
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    # Bajamos la confianza al 10% para forzar detecciones iniciales
    results = model(frame, conf=0.10, verbose=False)
    
    annotated_frame = results[0].plot()
    cv2.imshow('Diagnostico YOLO-World', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
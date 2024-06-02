import cv2
from ultralytics import YOLO

# Cargar el modelo
model = YOLO('yolov8n.pt')

# Abrir el video
cap = cv2.VideoCapture('/home/seba/Desktop/Salmones_Videos/PAU.mkv')

# Verificar si se puede abrir el video
if not cap.isOpened():
    print("Error al abrir el archivo de video.")
    exit()

# Crear la ventana de visualización
cv2.namedWindow('YOLOv8 Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLOv8 Detection', 800, 600)  # Ajusta el tamaño según tu pantalla

# Procesar cada cuadro del video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar la detección en el cuadro
    #results = model(frame)

    # Dibujar los resultados sobre el cuadro
    #annotated_frame = results[0].plot()

    # Redimensionar el cuadro para mejorar el rendimiento
    resized_frame = cv2.resize(frame, (416, 416))

    # Realizar la detección en el cuadro
    results = model(resized_frame)

    # Dibujar los resultados sobre el cuadro
    annotated_frame2 = results[0].plot()

    # Mostrar el cuadro anotado
    cv2.imshow('YOLOv8 Detection', annotated_frame2)

    # Salir al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar el objeto de captura de video y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
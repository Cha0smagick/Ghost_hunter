import cv2

# Cargar clasificadores Haar Cascade para la detección de rostros y ojos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Variable para almacenar el fotograma anterior
prev_frame = None

while True:
    # Leer el fotograma de la cámara
    ret, frame = cap.read()

    # Comprobar si se pudo capturar el fotograma correctamente
    if not ret:
        break

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detección de rostros
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Dibujar rectángulos alrededor de los rostros detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Recortar la región del rostro para la detección de ojos
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detección de ojos en la región del rostro
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Dibujar rectángulos alrededor de los ojos detectados
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Mostrar el fotograma resultante
    cv2.imshow('Face and Eye Tracking', frame)

    # Seguimiento del movimiento de ojos o rostro
    if prev_frame is not None:
        # Convertir el fotograma anterior a escala de grises
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Calcular la diferencia entre el fotograma actual y el anterior
        frame_diff = cv2.absdiff(gray, prev_gray)

        # Aplicar un umbral para resaltar las diferencias
        _, frame_diff_thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        # Mostrar el resultado del seguimiento de movimiento
        cv2.imshow('Motion Tracking', frame_diff_thresh)

    # Actualizar el fotograma anterior
    prev_frame = frame.copy()

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos y cerrar las ventanas abiertas
cap.release()
cv2.destroyAllWindows()

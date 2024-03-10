import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture("testVideos/centro.MOV")

# Obtener la resolución del video
width = int(cap.get(3))
height = int(cap.get(4))

# Definir la nueva resolución deseada (puedes ajustar estos valores)
new_width = 800
new_height = 800

# Factor de escala para redimensionar el marco
scale_factor = min(new_width / width, new_height / height)

# Inicializar MediaPipe Pose
with mp_pose.Pose(static_image_mode=False) as pose:
    while True:
        # Leer un fotograma de la fuente de video
        ret, frame = cap.read()
        if not ret:
            break

        # Voltear el marco horizontalmente
        frame = cv2.flip(frame, 1)

        # Convertir el marco a RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar el marco usando MediaPipe Pose
        results = pose.process(frame_rgb)

        # Dibujar puntos de referencia en el marco si se detectan
        if results.pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(128, 0, 250), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )

        # Redimensionar el marco
        frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

        # Mostrar el marco
        cv2.imshow("Frame", frame)

        # Comprobar si se presiona la tecla 'Esc' para salir del bucle
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import keypointrcnn_resnet50_fpn
import tensorflow as tf
from yolov8 import YOLOv8

# Inicializa el módulo Pose de MediaPipe
mp_pose = mp.solutions.pose
pose_mp = mp_pose.Pose()

# Inicializa el modelo PyTorch para detección de keypoints
model = keypointrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Inicializa el modelo TensorFlow para detección de poses (PoseNet)
pose_net = tf.saved_model.load("archive/3.tflite")

# Inicializa el modelo YOLOv8
yolo_model = YOLOv8(weights='yolov8')

# Lee el video
video_path = "testVideos/tango.MP4"
cap = cv2.VideoCapture(video_path)

# Configuración de la ventana
window_name = "Detección de Poses con Distintas Tecnologías"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1280, 720)

while cap.isOpened():
    # Lee un fotograma del video
    success, frame = cap.read()
    if not success:
        break

    # Detección de poses con MediaPipe
    results_mp = pose_mp.process(frame)
    if results_mp.pose_landmarks:
        # Obtén las coordenadas de las articulaciones
        landmarks = results_mp.pose_landmarks.landmark

        # Imprime las coordenadas de algunas articulaciones como ejemplo
        for landmark, coord in enumerate(landmarks):
            x, y, z = coord.x, coord.y, coord.z  # coordenadas normalizadas (0-1)
            print(f'Articulación {landmark}: X={x}, Y={y}, Z={z}')

        # Puedes adaptar el bucle anterior para imprimir las coordenadas de las articulaciones específicas que desees

    # Detección de poses con PyTorch
    with torch.no_grad():
        # Preprocesa la imagen para adaptarse al modelo PyTorch
        input_tensor = transforms.ToTensor()(frame)
        input_batch = input_tensor.unsqueeze(0)
        output = model(input_batch)['keypoints'][0]
        # Aquí puedes acceder a las coordenadas de los keypoints y realizar análisis adicional

    # Detección de poses con TensorFlow (PoseNet)
    input_tensor = tf.convert_to_tensor(frame, dtype=tf.float32)
    input_tensor = tf.expand_dims(input_tensor, 0)
    pose_net_output = pose_net(input_tensor)
    # Aquí puedes acceder a las coordenadas de las articulaciones y realizar análisis adicional

    # Detección de poses con YOLOv8
    yolo_output = yolo_model.predict(frame)
    # Aquí debes adaptar el código para obtener las coordenadas de los keypoints de interés con YOLOv8

    # Muestra los resultados en la ventana
    frame = cv2.putText(frame, "MediaPipe", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow(window_name, frame)

    # Sale del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera los recursos
cap.release()
cv2.destroyAllWindows()
pose_mp.close()

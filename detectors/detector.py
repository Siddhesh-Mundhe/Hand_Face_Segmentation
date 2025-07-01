import cv2
import os
import mediapipe as mp

class YOLODetector:
    def __init__(self, conf_threshold=0.3):
        self.conf_threshold = conf_threshold

        # 🔍 Load OpenCV DNN face detection model
        model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        self.face_net = cv2.dnn.readNetFromCaffe(
            os.path.join(model_dir, "deploy.prototxt"),
            os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        )

        # ✋ MediaPipe hands initialization
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=conf_threshold
        )

    def detect_faces_and_hands(self, image):
        h, w, _ = image.shape
        bboxes = []

        # 🎯 FACE DETECTION (OpenCV DNN)
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                     mean=(104.0, 177.0, 123.0), swapRB=False)
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf_threshold:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = map(int, box)
                bboxes.append(("face", (x1, y1, x2, y2)))

        # ✋ HAND DETECTION (MediaPipe)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min = int(min(x_coords) * w)
                y_min = int(min(y_coords) * h)
                x_max = int(max(x_coords) * w)
                y_max = int(max(y_coords) * h)
                bboxes.append(("hand", (x_min, y_min, x_max, y_max)))

        return bboxes
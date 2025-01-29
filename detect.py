import cv2
import numpy as np
import mediapipe as mp
import joblib

# TODO: ADD a 'L OK' too

class Detection:
    def __init__(self):
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)

        # Load the custom Model
        self.model = joblib.load('gesture_model.pkl')
        
        # Custom labels for the model
        self.labels = ['R OK', 'R Pointer', 'L Pointer', 'Left Seek', 'Right Seek', 'Rock and Roll', 'Open Hand']

    def normalize_landmarks(self, landmarks):
        # Take the first landmark as the reference point (0, 0)
        base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
        normalized = np.array([[lm.x - base_x, lm.y - base_y, lm.z - base_z] for lm in landmarks])
        
        return normalized.flatten()
    
    def start(self):
        # Start capturing video from the camera
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Flip
            frame = cv2.flip(frame, 1)

            # Convert the frame to RGB as MediaPipe expects RGB images
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame to find hands
            result = self.hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    key = cv2.waitKey(1) & 0xFF
                    
                    if key == 27:
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

                    normalized_landmarks = self.normalize_landmarks(hand_landmarks.landmark)
                    
                    y_pred_new = self.model.predict_proba([normalized_landmarks.tolist()])
                    max_pred = y_pred_new[0][np.argmax(y_pred_new)]

                    if max_pred > 0.985 and np.argmax(y_pred_new) != 6:
                        print("Predictions for new data: ", self.labels[np.argmax(y_pred_new)])
                        print("Total Predictons: ", y_pred_new[0])
                    else:
                        print("No Custom Gesture Detected")

            cv2.imshow('Hand Gesture Data Collection', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break


        cap.release()
        cv2.destroyAllWindows()

detection = Detection()
detection.start()

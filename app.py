import cv2
import numpy as np
import mediapipe as mp
import joblib
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import multiprocessing
import time

# Custom labels for the model
labels = ['R OK', 'R Pointer', 'L Pointer', 'Left Seek', 'Right Seek', 'Rock and Roll', 'Open Hand']


class DetectionClass:
    def __init__(self):
        self.model = joblib.load('gesture_model.pkl')
        self.scaler = joblib.load('gesture_scaler.pkl')
        self.last_action = None
        
        # Queue to store frames from the main thread for processing
        self.frame_queue = multiprocessing.Queue(maxsize=1)
        
        # Initialize Spotify
        self._initialize_spotify()

        if not self._is_device_active():
            print("Connect Spotify to a Device first")
            exit()

    def _initialize_spotify(self):
        CLIENT_ID = "967fed1f6d9649aa82727bfb2d222fd8"
        CLIENT_SECRET = "26f8ffe6bbba438abdf5bf7b85021e60"
        REDIRECT_URI = "https://example.com"
        
        sp_oauth = SpotifyOAuth(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope="user-modify-playback-state user-read-playback-state"
        )
        
        token_info = sp_oauth.get_cached_token()
        
        if not token_info:
            token_info = sp_oauth.get_access_token()
        
        access_token = token_info['access_token']
        self.sp = spotipy.Spotify(auth=access_token)
        
        print("Spotify Authenticated Successfully!")

    def _is_device_active(self):
        devices = self.sp.devices()
        
        if devices['devices']:
            for device in devices['devices']:
                if device['is_active']:
                    return True

        return False

    def _toggle_loop(self):
        current_playback = self.sp.current_playback()
        if not current_playback:
            print("No active playback detected.")
            return

        current_repeat_state = current_playback.get("repeat_state", "off")
        if current_repeat_state == "track":
            self.sp.repeat(state="off")
            print("Looping disabled.")
        else:
            self.sp.repeat(state="track")
            print("Looping enabled.")

    def _normalize_landmarks(self, landmarks):
        base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
        normalized = np.array([[lm.x - base_x, lm.y - base_y, lm.z - base_z] for lm in landmarks])
        return normalized.flatten()
    

    def _scale_landmarks(self, normalized_landmarks):
        scaled_landmarks = self.scaler.transform([normalized_landmarks])
        return scaled_landmarks.flatten()


    def _process_frame(self, scaled_landmarks):
        y_pred_new = self.model.predict_proba([scaled_landmarks.tolist()])
        max_pred = y_pred_new[0][np.argmax(y_pred_new)]
        
        if max_pred > 0.985 and np.argmax(y_pred_new) != 6:
            print("Predictions for new data: ", labels[np.argmax(y_pred_new)])
            print("Total Predictions: ", y_pred_new[0])

            gesture = np.argmax(y_pred_new)
            if gesture != self.last_action:
                self.last_action = gesture

                if gesture == 0:
                    self.sp.start_playback()
                elif gesture == 1:
                    self.sp.pause_playback()
                elif gesture == 3:
                    self.sp.previous_track()
                elif gesture == 4:
                    self.sp.next_track()
                elif gesture == 5:
                    self._toggle_loop()

                time.sleep(3)

        else:
            print("No hands detected in the current frame.")

    def _capture_video(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)
        cap = cv2.VideoCapture(1)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb_frame)

            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
                if not self.frame_queue.full():
                    normalized_landmarks = self._normalize_landmarks(hand_landmarks.landmark)
                    scaled_landmarks = self._scale_landmarks(normalized_landmarks)
                    
                    self.frame_queue.put(scaled_landmarks)
            
            cv2.imshow('Gesture Detection', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                break
        
        cap.release()
        cv2.destroyAllWindows()

    def _processing_thread(self):
        while True:
            if not self.frame_queue.empty():
                scaled_landmarks = self.frame_queue.get()
                self._process_frame(scaled_landmarks)

    def start(self):
        processing_thread = multiprocessing.Process(target=self._processing_thread)
        processing_thread.daemon = True
        processing_thread.start()

        self._capture_video()

if __name__ == "__main__":
    detection = DetectionClass()
    detection.start()

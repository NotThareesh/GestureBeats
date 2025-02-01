import cv2
import numpy as np
import mediapipe as mp
import joblib
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import multiprocessing
import time
from dotenv import load_dotenv
import os

# All Custom Labels that the Model can recognize
labels = ['R OK', 'L OK', 'R Pointer', 'L Pointer', 'Left Seek', 'Right Seek', 'Rock and Roll', 'Open Hand']


class DetectionClass:
    def __init__(self):
        # Load all the models
        self.model = joblib.load('gesture_model.pkl')
        self.scaler = joblib.load('gesture_scaler.pkl')

        # Queue to store frames from the main process for detection processing
        self.frame_queue = multiprocessing.Queue(maxsize=1)

        # Initialize Spotify
        self._initialize_spotify()

        if not self._is_device_active():
            print("Connect Spotify to a Device first")
            exit()

        # Cooldowns when detecting gestures
        self.last_action_time = time.time()

    def _is_valid_action(self):
        return time.time() - self.last_action_time > 3

    def _initialize_spotify(self):
        load_dotenv()

        CLIENT_ID = os.getenv('CLIENT_ID')
        CLIENT_SECRET = os.getenv('CLIENT_SECRET')
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

    # Check if any playback device is active
    def _is_device_active(self):
        devices = self.sp.devices()

        if devices['devices']:
            for device in devices['devices']:
                if device['is_active']:
                    return True

        return False

    def _toggle_loop(self):
        current_playback = self.sp.current_playback()
        current_repeat_state = current_playback.get("repeat_state", "off")

        if current_repeat_state == "track":
            self.sp.repeat(state="off")
        else:
            self.sp.repeat(state="track")

    def _toggle_playback(self):
        current_playback = self.sp.current_playback()
        is_playing = current_playback.get("is_playing", False)

        if is_playing:
            self.sp.pause_playback()
        else:
            self.sp.start_playback()

    def _seek_timing(self):
        pass

    # Normalising all landmarks w.r.t the position of the first hand landmark
    def _normalize_landmarks(self, landmarks):
        base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
        normalized = np.array([[lm.x - base_x, lm.y - base_y, lm.z - base_z] for lm in landmarks])
        return normalized.flatten()

    # Using the custom scaler to scale all the values
    def _scale_landmarks(self, normalized_landmarks):
        scaled_landmarks = self.scaler.transform([normalized_landmarks])
        return scaled_landmarks.flatten()

    def _process_frame(self, scaled_landmarks):
        y_pred_new = self.model.predict_proba([scaled_landmarks.tolist()])
        max_pred = y_pred_new[0][np.argmax(y_pred_new)]

        # Accuracy Threshold is 96%
        if max_pred > 0.96 and np.argmax(y_pred_new) != 7:
            print("Predictions for new data: ", labels[np.argmax(y_pred_new)])
            print("Total Predictions: ", y_pred_new[0])

            gesture = np.argmax(y_pred_new)

            if self._is_valid_action():
                if gesture in (0, 1):
                    self._toggle_playback()
                elif gesture in (2, 3):
                    self._seek_timing()
                elif gesture == 4:
                    self.sp.previous_track()
                elif gesture == 5:
                    self.sp.next_track()
                elif gesture == 6:
                    self._toggle_loop()

                self.last_action_time = time.time()

    def _capture_video(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6)
        cap = cv2.VideoCapture(1)  # Change it to 0 or 1 depending upon your computer's camera input

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

                    h, w = frame.shape[:2]
                    landmarks = [(lm.x * w, lm.y * h) for lm in hand_landmarks.landmark]
                    x_min, y_min = np.min(landmarks, axis=0)
                    x_max, y_max = np.max(landmarks, axis=0)
                    x_min = max(0, int(x_min) - 20)
                    y_min = max(0, int(y_min) - 20)
                    x_max = min(w, int(x_max) + 20)
                    y_max = min(h, int(y_max) + 20)

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

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

    # Multiprocessing
    def start(self):
        processing_thread = multiprocessing.Process(target=self._processing_thread)
        processing_thread.daemon = True
        processing_thread.start()

        self._capture_video()


if __name__ == "__main__":
    detection = DetectionClass()
    detection.start()

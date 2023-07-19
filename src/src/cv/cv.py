import mediapipe as mp
import cv2
from PySide6 import QtCore
from araviq6 import ArrayWorker
import numpy as np
from numpy.linalg import inv

class KalmanFilter:
    """
    Simple Kalman filter
    """

    def __init__(self, X, F, Q, Z, H, R, P, B=np.array([0]), M=np.array([0])):
        """
        Initialise the filter
        Args:
            X: State estimate
            P: Estimate covariance
            F: State transition model
            B: Control matrix
            M: Control vector
            Q: Process noise covariance
            Z: Measurement of the state X
            H: Observation model
            R: Observation noise covariance
        """
        self.X = X
        self.P = P
        self.F = F
        self.B = B
        self.M = M
        self.Q = Q
        self.Z = Z
        self.H = H
        self.R = R

    def predict(self):
        """
        Predict the future state
        Args:
            self.X: State estimate
            self.P: Estimate covariance
            self.B: Control matrix
            self.M: Control vector
        Returns:
            updated self.X
        """
        # Project the state ahead
        self.X = self.F @ self.X + self.B @ self.M
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.X

    def correct(self, Z):
        """
        Update the Kalman Filter from a measurement
        Args:
            self.X: State estimate
            self.P: Estimate covariance
            Z: State measurement
        Returns:
            updated X
        """
        K = self.P @ self.H.T @ inv(self.H @ self.P @ self.H.T + self.R)
        self.X += K @ (Z - self.H @ self.X)
        self.P = self.P - K @ self.H @ self.P

        return self.X

class FingerTracker(ArrayWorker):
    sigFingerTrackingPos = QtCore.Signal(object)
    def __init__(self):
        super().__init__()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
                        static_image_mode = False,
                        max_num_hands = 1,
                        model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)


    def predict(self, frame):
        frame.flags.writeable = False
        results = self.hands.process(frame)
        frame.flags.writeable = True
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.sigFingerTrackingPos.emit((
                    True,
                    hand_landmarks.landmark[8].x,
                    hand_landmarks.landmark[8].y
                ))
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())
        else:
            self.sigFingerTrackingPos.emit((False, False))
        return frame

    def processArray(self, array: np.ndarray) -> np.ndarray:
        img = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        img = self.predict(img)
        return img
        
if __name__ == "__main__":
    f = FingerTracker()
# class FingerTracker(QtCore.QObject):
#     sigFingerTrackingRes = QtCore.Signal(object)
#     def __init__(self):
#         super().__init__()
#         self.VisionRunningMode = mp.tasks.vision.RunningMode
#         self.base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
#         self.options = vision.HandLandmarkerOptions(base_options=self.base_options,
#                                             running_mode=self.VisionRunningMode.LIVE_STREAM,
#                                             result_callback=self.res,
#                                             num_hands=1)
#         self.detector = vision.HandLandmarker.create_from_options(self.options)
#         self.i = 0
#     def res(self, res):
#         print(res)

#     def predict(self, frame):
#         mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
#         self.detector.detect_async(mp_image, self.i)
#         self.i += 1

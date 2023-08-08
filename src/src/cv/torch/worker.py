from src.src.cv.torch.gaze_estimator import GazeEstimator
from src.src.cv.torch.face_model_mediapipe import Camera, Face, FacePartsName, FaceModelMediaPipe
from src.src.cv.torch.visualizer import Visualizer
from araviq6 import ArrayWorker as Worker
import cv2
import numpy as np
from PySide6.QtCore import Signal
import mediapipe as mp
import time
class ArrayWorker(Worker):
    sigOriginalFrame = Signal(object)
    pausePreview = False
    scale_percent = 25
    

    def __init__(self):
        super().__init__()

    def scale(self, image):
        width = int(image.shape[1] * self.scale_percent / 100)
        height = int(image.shape[0] * self.scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return image

class GazeWorker(ArrayWorker):
    def __init__(self):
        super().__init__()
        self.gaze_estimator = GazeEstimator()
        self.face_model = FaceModelMediaPipe()
        self.visualizer = Visualizer(self.gaze_estimator.camera,self.face_model.NOSE_INDEX)
        
    def processArray(self, image: np.ndarray) -> np.ndarray:
        self.sigOriginalFrame.emit(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if self.pausePreview:
            return np.array([])
        
        image = self.scale(image)

        undistorted = cv2.undistort(
            image, self.gaze_estimator.camera.camera_matrix,
            self.gaze_estimator.camera.dist_coefficients)

        self.visualizer.set_image(image.copy())
        faces = self.gaze_estimator.detect_faces(undistorted)
        for face in faces:
            self.gaze_estimator.estimate_gaze(undistorted, face)
            self.visualizer.draw_bbox(face.bbox)
            self.visualizer.draw_model_axes(face, 0.1, lw=2)
            euler_angles = face.head_pose_rot.as_euler('XYZ', degrees=True)
            pitch, yaw, roll = face.change_coordinate_system(euler_angles)
            self.visualizer.draw_3d_line(
                face.center, face.center + 0.2 * face.gaze_vector)
            pitch, yaw = np.rad2deg(face.vector_to_angle(face.gaze_vector))
            # print(f'[face] pitch: {pitch:.2f}, yaw: {yaw:.2f}')

        # self.visualizer.image = self.visualizer.image[:, ::-1]
        img = cv2.cvtColor(self.visualizer.image, cv2.COLOR_BGR2RGB)
        return img
    
class FingerWorker(ArrayWorker):
    '''
    subclass of araviq6, utlizing multithreading to prevent forzen UI, used by video preview
    using QtMultiMedia API because it offers better compatibility when making playback commands
    araviq provides seamless transformation between numpy array(CV calculation) and QImage(display), see more in its documentation
    '''
    sigFingerTrackingPos = Signal(object)
    screen_transform = {
        'status': False
    }
    is_tracking_finger = False
    is_previewing_calibration = False
    M = np.array([])
    def __init__(self):
        super().__init__()
        self.mp_drawing = mp.solutions.drawing_utils
        # self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=5, circle_radius=8)
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
                        static_image_mode = False,
                        max_num_hands = 3,
                        model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)
        
        stateMatrix = np.zeros((4, 1), np.float32)  # [x, y, delta_x, delta_y]
        estimateCovariance = np.eye(stateMatrix.shape[0])
        transitionMatrix = np.array([[1, 0, 1, 0],[0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.001
        measurementStateMatrix = np.zeros((2, 1), np.float32)
        observationMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 0.03
        
        self.kalman = KalmanFilter(X=stateMatrix,
                            P=estimateCovariance,
                            F=transitionMatrix,
                            Q=processNoiseCov,
                            Z=measurementStateMatrix,
                            H=observationMatrix,
                            R=measurementNoiseCov)
        # self.SCREEN_RES = screen_res

    def enable_finger_tracking(self, e):
        self.is_tracking_finger = bool(e)

    def update_screen_show(self, e):
        self.is_previewing_calibration = e

    def update_screen_transform(self, e):
        '''
        用于手指追踪屏幕预览
        '''
        self.screen_transform = e
        if self.screen_transform['status'] is False:
            self.M = np.array([])
            return
        width, height = self.screen_transform['screen_dimension']
        pts = self.screen_transform['pts']
        # print(pts)
        dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        self.M = cv2.getPerspectiveTransform(pts, dst)
        self.M_inverse = cv2.invert(self.M)[1]

    def transform_and_scale(self, image):
        if self.screen_transform['status'] is True and self.M.size != 0:
            if self.is_previewing_calibration:
                image = cv2.warpPerspective(image, self.M, self.screen_transform['screen_dimension'])
            else:
                #在正常状态下画出屏幕边界
                #需要交换横纵坐标？
                pts = self.screen_transform['pts'].astype(int).tolist()
                for i in range(-4, 0):
                    cv2.line(image, pts[i], pts[i+1], (146, 136, 248), 6)
                # print()
                pass
        width = int(image.shape[1] * self.scale_percent / 100)
        height = int(image.shape[0] * self.scale_percent / 100)
        dim = (width, height)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return image

    def predict(self, frame):
        frame.flags.writeable = False
        results = self.hands.process(frame)
        frame.flags.writeable = True
        h, w, _ = frame.shape
        res = []
        if results.multi_hand_landmarks:
            #TODO: 卡尔曼滤波算法会在Apple Silicon上报错，未来添加
            for hand_landmarks in results.multi_hand_landmarks:
                if self.M.size != 0:
                    is_finger_in_screen = False
                    # e = np.int32((hand_landmarks.landmark[8].x * w, hand_landmarks.landmark[8].y *h))
                    e = np.float32((hand_landmarks.landmark[8].x * w, hand_landmarks.landmark[8].y *h))
                    x, y = np.squeeze(cv2.perspectiveTransform(e.reshape(-1, 1, 2), self.M))
                    # x = current_prediction[0][0]
                    # y = current_prediction[1][0]
                    # cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), 3)
                    if x > 0 and x < self.screen_transform['screen_dimension'][0]:
                        if y > 0 and y < self.screen_transform['screen_dimension'][1]:
                            is_finger_in_screen = True
                            self.paint_finger_screen_preview(frame, x, y)
                    res.append((is_finger_in_screen, x/4, y/4))
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(
                        color=(11, 102, 106),  # 设置关节的颜色
                        thickness=25  # 设置关节的粗细
                    ),
                    self.mp_drawing.DrawingSpec(
                        color=(151, 254, 237),  # 设置关节的颜色
                        thickness=3  # 设置关节的粗细
                    ))
            self.sigFingerTrackingPos.emit(res)
        else:
            pass
            # self.sigFingerTrackingPos.emit((False, False))
        return frame, res
    
    def paint_finger_screen_preview(self, frame, x, y):
        crds = np.array(
            [[x, 0],
            [x, self.screen_transform['screen_dimension'][1]],
            [0, y],
            [self.screen_transform['screen_dimension'][0], y]]
        )
        inverse_transformed_points = cv2.perspectiveTransform(crds.reshape(-1, 1, 2),
                                    self.M_inverse)
        p0,p1,p2,p3 = np.int32(inverse_transformed_points).squeeze().squeeze().tolist()
        cv2.line(frame, p0, p1, (234, 17, 121), 6)
        cv2.line(frame, p2, p3, (234, 17, 121), 6)

    def processArray(self, image: np.ndarray) -> np.ndarray:
        if len(image) == 0:
            pass
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.sigOriginalFrame.emit(image)
            if self.pausePreview:
                return np.array([])
            if self.is_tracking_finger:
                image, res = self.predict(image)
            image = self.transform_and_scale(image)
        return image
    
class EmptyWorker(ArrayWorker):
    def __init__(self):
        super().__init__()
        
    def processArray(self, array: np.ndarray) -> np.ndarray:
        # print(time.time())
        image = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
        self.sigOriginalFrame.emit(image)
        if self.pausePreview:
            return np.array([])
        #用于预览的图像一定要小，否则两个摄像头同时采集时会引发UI的卡顿
        image = self.scale(image)
        return image
    

    
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
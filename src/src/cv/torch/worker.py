from src.src.cv.torch.gaze_estimator import GazeEstimator
from src.src.cv.torch.face_model_mediapipe import Camera, Face, FacePartsName, FaceModelMediaPipe
from src.src.cv.torch.visualizer import Visualizer
from araviq6 import ArrayWorker as Worker
import cv2
import numpy as np
from PySide6.QtCore import Signal
import mediapipe as mp
import time
import cv2.aruco as aruco
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
        'status': False,
    }
    aruco_transform = {
        'status': False,
        'screen_dimension': [1500, 1500],
        'M': np.array([])

    }
    is_tracking_finger = False
    is_previewing_calibration = False
    M = np.array([])
    COLOR_FINGER_PREVIEW_CROSSHAIR = (255, 82, 162)
    COLOR_SCREEN_PREVIEW_RECT = (255, 176, 127)
    COLOR_ARUCO_RECT = (255, 0, 0)
    COLOR_ARUCO_SUCCESS = (0, 255, 0)
    COLOR_ARUCO_FAIL = (255, 0, 0)
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

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        parameters =  cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

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
            return
        if self.aruco_transform['status'] is True:
            '''
            如果已经对mark进行标定，那么用户选择的坐标应该是位于aruco_transform坐标系下
            这个坐标与aruco_transform坐标系直接相关，并不与世界坐标系直接相关
            因此，先根据这个坐标计算出从aruco_transform坐标系到屏幕坐标系的M
            原图上的每一点需要经过两次变换才能转换为屏幕上的坐标点
            '''
            width, height = self.screen_transform['screen_dimension'] #这里的宽高已经乘以了4
            pts = self.screen_transform['pts']
            dst = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
            self.M = cv2.getPerspectiveTransform(pts, dst)
        print(e)
        return

    def transform_and_scale(self, image):
        if self.is_previewing_calibration:
            #有Mark标记辅助的情况下
            if self.aruco_transform['status']:
                image = cv2.warpPerspective(image, self.aruco_transform['M'], self.aruco_transform['screen_dimension'])
                if self.M.size != 0 and self.screen_transform['status'] is True:
                    image = cv2.warpPerspective(image, self.M, self.screen_transform['screen_dimension'])
        elif self.M.size != 0 and self.screen_transform['status'] is True:
            # M_combined = np.dot(self.M, self.aruco_transform['M']) 
            M_inverse = cv2.invert(self.aruco_transform['M'])[1]
            transformed_point = cv2.perspectiveTransform(self.screen_transform['pts'].reshape(1, -1, 2), M_inverse)[0]
            image = self.draw_rect(image, transformed_point, color=self.COLOR_SCREEN_PREVIEW_RECT)
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
                    M = np.dot(self.M, self.aruco_transform['M'])
                    x, y = np.squeeze(cv2.perspectiveTransform(e.reshape(-1, 1, 2), M))
                    # x = current_prediction[0][0]
                    # y = current_prediction[1][0]
                    # cv2.circle(frame, (int(x), int(y)), 10, (255, 0, 0), 3)
                    if x > 0 and x < self.screen_transform['screen_dimension'][0]:
                        if y > 0 and y < self.screen_transform['screen_dimension'][1]:
                            is_finger_in_screen = True
                            print(x, y)
                            self.paint_finger_screen_preview(frame, x, y)
                    res.append((is_finger_in_screen, x/4, y/4))
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(
                        color=(11, 102, 106),  # 设置关节的颜色
                        thickness=8  # 设置关节的粗细
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
        M = np.dot(self.M, self.aruco_transform['M'])
        M_inversed = cv2.invert(M)[1]
        inverse_transformed_points = cv2.perspectiveTransform(crds.reshape(-1, 1, 2),
                                    M_inversed)
        p0,p1,p2,p3 = np.int32(inverse_transformed_points).squeeze().squeeze().tolist()
        cv2.line(frame, p0, p1, self.COLOR_FINGER_PREVIEW_CROSSHAIR, 6)
        cv2.line(frame, p2, p3, self.COLOR_FINGER_PREVIEW_CROSSHAIR, 6)

    def processArray(self, image: np.ndarray) -> np.ndarray:
        if len(image) == 0:
            return image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sigOriginalFrame.emit(image)
        if self.pausePreview:
            return np.array([])
        success, pts = self.update_aruco(image)
        if self.is_tracking_finger:
            image, res = self.predict(image)
        if success:
            try:
                image = self.paint_aruco(image, pts)
            except Exception as e:
                print(e)
        image = self.transform_and_scale(image)
        return image
    
    def sort_aruco_arrays(self, *e): 
        '''
        aruco检测算法得到的坐标顺序为规定的固定顺序
        将一个aruco标记的坐标顺序转换为实际在图像上位置的左上-右上-左下-右下
        '''
        res = []
        for array in e:
            x0, y0 = np.mean(array, axis=0)
            i = np.where(np.logical_and(array[:, 0] <= x0, array[:, 1] <= y0))[0][0] + 1
            array = np.vstack((array[-1], array, array[:3]))[i:i+4]
            res.append(array)
        return res
    
    def paint_aruco(self, image, pts):
        '''
        在检测到四个ARUCO标记时，计算出四个标记的位置顺序，获得标记屏幕范围的四边形参数
        '''
        pts = np.squeeze(np.array(pts), axis=1)
        aruco_centers = np.mean(pts, axis=1) #mark的中心点
        img_center = np.mean(aruco_centers, axis=0) #四边形的中心点

        #区分标记的相对位置，前提是mark都摆的比较正，不算太歪（不超过45度）
        tl = pts[(pts[:, :, 0] < img_center[0]) & (pts[:, :, 1] < img_center[1])]
        tr = pts[(pts[:, :, 0] > img_center[0]) & (pts[:, :, 1] < img_center[1])]
        bl = pts[(pts[:, :, 0] < img_center[0]) & (pts[:, :, 1] > img_center[1])]
        br = pts[(pts[:, :, 0] > img_center[0]) & (pts[:, :, 1] > img_center[1])]
        
        tl, tr, bl, br = self.sort_aruco_arrays(tl, tr, bl, br)
        
        #获取mark位置边界
        big_rect = np.array([
            tl[0], tr[1], br[2], bl[-1]
        ])

        
        if not self.is_previewing_calibration and not self.screen_transform['status']:
            image = self.draw_rect(image, big_rect, self.COLOR_ARUCO_RECT, 5)

        '''
        并没有使用比较复杂的solvePnP等方法推算空间位置，因为这会涉及到摄像头的参数矫正等问题
        1. 在用户尚未进行屏幕标定时(screen_transform['status'] is False)
            首先计算将mark边界四边形转换为1000*1000的正方形的矩阵M
            然后根据新的正方形里mark内边界与边长的比例关系（已知mark为正方形）计算mark矩形的长宽比
            最后根据新的确认的矩形尺寸重新计算转换矩阵M，将确认的M和矩形尺寸保存
        2. 用户已进行了屏幕标定(screen_transform['status'] is True)
            此时Mark的位置关系已经得到确认，不宜再进行更改
            因此直接计算将mark边界四边形转换为易保存尺寸的正方形的矩阵M
            保存新的M
        注意：并没有任何要求规定用户贴的Mark必须构成一个完美的矩形，因此这一步算出的矩形尺寸只是Estimate
            并不能少了用户标定这一步
        '''

        if self.screen_transform['status'] is False and self.is_previewing_calibration is False:
            width, height = 1000, 1000
            dst = np.array([[0,0],[width,0], [width,width], [0, width]]).astype(np.float32)
            M = cv2.getPerspectiveTransform(big_rect, dst)
            small_rect = np.array([
                tl[2], tr[-1], br[0], bl[1]
            ])
            #由mark构成的内部的小四边形转换到新的坐标系下，根据其畸变计算大致的长宽比
            small_rect = np.squeeze(cv2.perspectiveTransform(small_rect.reshape(-1, 1, 2), M) )
            # print(small_rect)
            h = np.sum(small_rect[:2, 1]) + np.sum(height - small_rect[2:, 1])
            w = np.sum(small_rect[[0,3], 0]) + np.sum(width - small_rect[[1,2], 0])
            #mark长度实际上对应的是所占的这一边的比例，数值越大这一边越短
            w_h_ratio = h / w
            #固定高度为1000
            self.aruco_transform['screen_dimension'] = [round(w_h_ratio*1000), 1000]

        width, height = self.aruco_transform['screen_dimension']
        dst = np.array([[0,0],[width,0], [width,height], [0, height]]).astype(np.float32)
        self.aruco_transform['M'] = cv2.getPerspectiveTransform(big_rect, dst)
        self.aruco_transform['status'] = True
        return image

    def draw_rect(self, image, rect, color=(255, 0, 0), thickness=5):
        rect = rect.astype(int).tolist()
        cv2.line(image, rect[0], rect[1], color, thickness)
        cv2.line(image, rect[1], rect[2], color, thickness)
        cv2.line(image, rect[2], rect[3], color, thickness)
        cv2.line(image, rect[0], rect[3], color, thickness)
        return image
    
    def update_aruco(self, image):
        '''
        检测画面中的aruco mark，根据第一个mark的坐标更新M和M_inverse
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 检测ArUco标记
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(gray)
        if not self.is_previewing_calibration:
            color = self.COLOR_ARUCO_SUCCESS if len(corners) == 4 else self.COLOR_ARUCO_FAIL
            for i in corners:
                image =self.draw_rect(image, i[0], color, 7)
        if ids is None or len(ids) != 4:
            return False, 0
        return True, corners
    
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
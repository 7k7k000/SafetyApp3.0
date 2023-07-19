from typing import List

import numpy as np
import torch
import timm

from src.src.cv.torch.face_model_mediapipe import Camera, Face, FacePartsName, FaceModelMediaPipe
from src.src.cv.torch.head_pose_estimation import HeadPoseNormalizer, LandmarkEstimator
import cv2
import torchvision.transforms as T


class GazeEstimator:
    EYE_KEYS = [FacePartsName.REYE, FacePartsName.LEYE]

    def __init__(self):

        self._face_model3d = FaceModelMediaPipe()

        self.camera = Camera('./src/src/cv/torch/sample_params.yaml')
        self._normalized_camera = Camera('./src/src/cv/torch/eth-xgaze.yaml')

        self._landmark_estimator = LandmarkEstimator()
        self._head_pose_normalizer = HeadPoseNormalizer(
            self.camera, self._normalized_camera, 0.6)
        self._gaze_estimation_model = self._load_model()
        size = tuple([224, 224])
        self._transform = T.Compose([
            T.Lambda(lambda x: cv2.resize(x, size)),
            T.Lambda(lambda x: x[:, :, ::-1].copy()),  # BGR -> RGB
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224,
                                                        0.225]),  # RGB
        ])

    def _load_model(self) -> torch.nn.Module:
        model = timm.create_model('resnet18', num_classes=2)
        checkpoint = torch.load('./src/src/cv/torch/eth-xgaze_resnet18.pth',
                                map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model.to('cpu')
        model.eval()
        return model

    def detect_faces(self, image: np.ndarray) -> List[Face]:
        return self._landmark_estimator.detect_faces(image)

    def estimate_gaze(self, image: np.ndarray, face: Face) -> None:
        self._face_model3d.estimate_head_pose(face, self.camera)
        self._face_model3d.compute_3d_pose(face)
        self._face_model3d.compute_face_eye_centers(face, 'ETH-XGaze')

        self._head_pose_normalizer.normalize(image, face)
        self._run_ethxgaze_model(face)

    @torch.no_grad()
    def _run_ethxgaze_model(self, face: Face) -> None:
        image = self._transform(face.normalized_image).unsqueeze(0)

        device = torch.device("cpu")
        image = image.to(device)
        prediction = self._gaze_estimation_model(image)
        prediction = prediction.cpu().numpy()

        face.normalized_gaze_angles = prediction[0]
        face.angle_to_vector()
        face.denormalize_gaze_vector()

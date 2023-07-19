from typing import List

import mediapipe
import numpy as np

from src.src.cv.torch.face_model_mediapipe import Face


class LandmarkEstimator:
    def __init__(self):
        self.detector = mediapipe.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            static_image_mode=False)

    def detect_faces(self, image: np.ndarray) -> List[Face]:
            return self._detect_faces_mediapipe(image)
        
    def _detect_faces_mediapipe(self, image: np.ndarray) -> List[Face]:
        h, w = image.shape[:2]
        predictions = self.detector.process(image[:, :, ::-1])
        detected = []
        if predictions.multi_face_landmarks:
            for prediction in predictions.multi_face_landmarks:
                pts = np.array([(pt.x * w, pt.y * h)
                                for pt in prediction.landmark],
                               dtype=np.float64)
                bbox = np.vstack([pts.min(axis=0), pts.max(axis=0)])
                bbox = np.round(bbox).astype(np.int32)
                detected.append(Face(bbox, pts))
        return detected

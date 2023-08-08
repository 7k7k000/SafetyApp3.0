import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import traceback
from src.src.cv.torch.face_model_mediapipe import Camera, FaceParts, FacePartsName


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


class HeadPoseNormalizer:
    def __init__(self, camera: Camera, normalized_camera: Camera,
                 normalized_distance: float):
        self.camera = camera
        self.normalized_camera = normalized_camera
        self.normalized_distance = normalized_distance

    def normalize(self, image: np.ndarray, eye_or_face: FaceParts) -> None:
        eye_or_face.normalizing_rot = self._compute_normalizing_rotation(
            eye_or_face.center, eye_or_face.head_pose_rot)
        self._normalize_image(image, eye_or_face)
        self._normalize_head_pose(eye_or_face)

    def _normalize_image(self, image: np.ndarray,
                         eye_or_face: FaceParts) -> None:
        # print(np.linalg.inv(self.camera.camera_matrix.copy()))
        print(0)
        camera_matrix_inv = np.array([[ 0.0015625,  0.,        -0.5,      ],
            [ 0.,         0.0015625, -0.375    ],
            [ 0.,         0.,         1.       ],])
        # camera_matrix_inv = np.linalg.inv(self.camera.camera_matrix)
        print(camera_matrix_inv)
        normalized_camera_matrix = self.normalized_camera.camera_matrix
        print(2)

        scale = self._get_scale_matrix(eye_or_face.distance)
        print(3)
        conversion_matrix = scale @ eye_or_face.normalizing_rot.as_matrix()
        print(4)

        projection_matrix = normalized_camera_matrix @ conversion_matrix @ camera_matrix_inv
        print(5)

        normalized_image = cv2.warpPerspective(
            image, projection_matrix,
            (self.normalized_camera.width, self.normalized_camera.height))
        print(6)
        normalized_image = image

        if eye_or_face.name in {FacePartsName.REYE, FacePartsName.LEYE}:
            normalized_image = cv2.cvtColor(normalized_image,
                                            cv2.COLOR_BGR2GRAY)
            normalized_image = cv2.equalizeHist(normalized_image)
        eye_or_face.normalized_image = normalized_image

    @staticmethod
    def _normalize_head_pose(eye_or_face: FaceParts) -> None:
        normalized_head_rot = eye_or_face.head_pose_rot * eye_or_face.normalizing_rot
        euler_angles2d = normalized_head_rot.as_euler('XYZ')[:2]
        eye_or_face.normalized_head_rot2d = euler_angles2d * np.array([1, -1])

    @staticmethod
    def _compute_normalizing_rotation(center: np.ndarray,
                                      head_rot: Rotation) -> Rotation:
        # See section 4.2 and Figure 9 of https://arxiv.org/abs/1711.09017
        z_axis = _normalize_vector(center.ravel())
        head_rot = head_rot.as_matrix()
        head_x_axis = head_rot[:, 0]
        y_axis = _normalize_vector(np.cross(z_axis, head_x_axis))
        x_axis = _normalize_vector(np.cross(y_axis, z_axis))
        return Rotation.from_matrix(np.vstack([x_axis, y_axis, z_axis]))

    def _get_scale_matrix(self, distance: float) -> np.ndarray:
        return np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, self.normalized_distance / distance],
        ],
                        dtype=float)

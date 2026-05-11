import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass

import config # <-- TAMBAHAN: Mengimpor file konfigurasi utama

# ─── Landmark index groups ────────────────────────────────────────────────────
LEFT_EYE_INDICES  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

NOSE_TIP   = 1
CHIN       = 152
LEFT_EYE_L = 263
RIGHT_EYE_R= 33
MOUTH_LEFT = 287
MOUTH_RIGHT= 57


@dataclass
class FaceResult:
    landmarks: list          # raw NormalizedLandmark list
    landmarks_px: np.ndarray # shape (N, 2) pixel coords
    bbox: tuple              # (x, y, w, h)
    image_shape: tuple       # (h, w)


class FaceMeshDetector:
    def __init__(
        self,
        max_faces: int = 1,
        refine_landmarks: bool = True,
        # --- PERBAIKAN: Mengambil nilai confidence dari config.py agar lebih sensitif di cahaya rendah ---
        min_detection_confidence: float = getattr(config, "MIN_DETECTION_CONFIDENCE", 0.5),
        min_tracking_confidence: float = getattr(config, "MIN_TRACKING_CONFIDENCE", 0.5),
    ):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> list[FaceResult]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        faces: list[FaceResult] = []
        if not results.multi_face_landmarks:
            return faces

        for face_landmarks in results.multi_face_landmarks:
            lm_px = np.array(
                [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark],
                dtype=np.int32,
            )
            xs, ys = lm_px[:, 0], lm_px[:, 1]
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            bbox = (x1, y1, x2 - x1, y2 - y1)

            faces.append(FaceResult(
                landmarks=face_landmarks.landmark,
                landmarks_px=lm_px,
                bbox=bbox,
                image_shape=(h, w),
            ))

        return faces

    # ------------------------------------------------------------------ #
    def draw(self, frame: np.ndarray, face: FaceResult) -> np.ndarray:
        annotated = frame.copy()
        # Reconstruct NormalizedLandmarkList for drawing
        lm_list = self.mp_face_mesh.FaceMesh  # just for typing hint
        fake_proto = type("FakeLM", (), {"landmark": face.landmarks})()
        self.mp_draw.draw_landmarks(
            annotated,
            fake_proto,
            self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_styles.get_default_face_mesh_tesselation_style(),
        )
        return annotated

    # ------------------------------------------------------------------ #
    def get_eye_points(self, face: FaceResult, eye: str) -> np.ndarray:
        indices = LEFT_EYE_INDICES if eye == "left" else RIGHT_EYE_INDICES
        return face.landmarks_px[indices]

    def get_head_pose_points(self, face: FaceResult) -> tuple:
        """
        Returns 6 2D points + 6 corresponding 3D model points for solvePnP.
        """
        lm = face.landmarks_px
        h, w = face.image_shape

        image_points = np.array([
            lm[NOSE_TIP],
            lm[CHIN],
            lm[LEFT_EYE_L],
            lm[RIGHT_EYE_R],
            lm[MOUTH_LEFT],
            lm[MOUTH_RIGHT],
        ], dtype=np.float64)

        model_points = np.array([
            [0.0,    0.0,    0.0],     # Nose tip
            [0.0,  -330.0, -65.0],     # Chin
            [-225.0, 170.0, -135.0],   # Left eye left corner
            [225.0,  170.0, -135.0],   # Right eye right corner
            [-150.0, -150.0, -125.0],  # Left mouth corner
            [150.0,  -150.0, -125.0],  # Right mouth corner
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        return image_points, model_points, camera_matrix, dist_coeffs

    def close(self):
        self.face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
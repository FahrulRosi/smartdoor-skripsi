import numpy as np
from facemesh.facemesh_detector import FaceResult, FaceMeshDetector

# --- PENGATURAN SENSITIVITAS KEDIPAN ---
EAR_THRESHOLD = 0.16   # Jika EAR di bawah ini, mata dianggap tertutup (sebelumnya 0.22)
CONSEC_FRAMES = 4      # Jumlah frame berturut-turut mata harus tertutup agar dihitung 1 kedipan (sebelumnya 2)


def _ear(eye_pts: np.ndarray) -> float:
    """Menghitung Eye Aspect Ratio (Soukupová & Čech, 2016)."""
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    # Menghitung rasio mata (EAR)
    return (A + B) / (2.0 * C + 1e-6)


class BlinkDetector:
    def __init__(
        self,
        ear_threshold: float = EAR_THRESHOLD,
        consec_frames: int = CONSEC_FRAMES,
        target_blinks: int = 1,
    ):
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.target_blinks = target_blinks

        self._frame_counter = 0
        self.blink_count = 0
        self.complete = False

    # ------------------------------------------------------------------ #
    def update(self, face: FaceResult, detector: FaceMeshDetector) -> dict:
        left_pts  = detector.get_eye_points(face, "left")
        right_pts = detector.get_eye_points(face, "right")

        left_ear  = _ear(left_pts)
        right_ear = _ear(right_pts)
        
        # Ambil rata-rata rasio mata kiri dan kanan
        avg_ear   = (left_ear + right_ear) / 2.0

        blinked_now = False
        
        # Jika nilai EAR di bawah threshold, berarti mata sedang tertutup
        if avg_ear < self.ear_threshold:
            self._frame_counter += 1
        else:
            # Jika mata kembali terbuka, cek apakah sebelumnya sempat tertutup 
            # selama minimal CONSEC_FRAMES (4 frame)
            if self._frame_counter >= self.consec_frames:
                self.blink_count += 1
                blinked_now = True
                
                # Cek apakah jumlah target kedipan sudah tercapai
                if self.blink_count >= self.target_blinks:
                    self.complete = True
            
            # Reset frame counter karena mata sudah terbuka
            self._frame_counter = 0

        return {
            "ear": round(avg_ear, 3),
            "blink_count": self.blink_count,
            "blinked_now": blinked_now,
            "complete": self.complete,
        }
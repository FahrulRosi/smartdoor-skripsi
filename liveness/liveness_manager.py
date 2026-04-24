from facemesh.facemesh_detector import FaceResult, FaceMeshDetector
from liveness.blink import BlinkDetector
from liveness.head_pose import HeadPoseEstimator
import config

class LivenessManager:
    def __init__(self):
        self.pose_estimator = HeadPoseEstimator()
        self._register_step = 0 # 0: Yaw, 1: Pitch, 2: Roll, 3: Blink, 4: Done
        self._register_blink = None

    def start_register(self):
        self._register_step = 0
        self._register_blink = BlinkDetector(target_blinks=config.REGISTER_BLINK_COUNT)

    def update_register(self, face: FaceResult, detector: FaceMeshDetector) -> dict:
        pose = self.pose_estimator.estimate(face, detector)
        
        if not pose["valid"]:
            return {"status": "pending", "step": "NO_FACE", "instruction": "Wajah tidak terdeteksi"}

        # Threshold toleransi dari config
        yaw_ok   = abs(pose["yaw"])   <= config.MAX_YAW
        pitch_ok = abs(pose["pitch"]) <= config.MAX_PITCH
        roll_ok  = abs(pose["roll"])  <= config.MAX_ROLL

        # Tahap 0: Validasi Yaw (Lurus ke depan)
        if self._register_step == 0:
            if yaw_ok:
                self._register_step = 1
            else:
                return {"status": "pending", "step": "YAW", "instruction": "Arahkan wajah lurus ke depan (Yaw)"}

        # Tahap 1: Validasi Pitch (Lurus/Tidak mendongak)
        if self._register_step == 1:
            if not yaw_ok: self._register_step = 0
            elif pitch_ok: self._register_step = 2
            else:
                return {"status": "pending", "step": "PITCH", "instruction": "Arahkan wajah lurus ke depan (Pitch)"}

        # Tahap 2: Validasi Roll (Tegak/Tidak miring)
        if self._register_step == 2:
            if not pitch_ok: self._register_step = 1
            elif roll_ok: self._register_step = 3
            else:
                return {"status": "pending", "step": "ROLL", "instruction": "Pastikan kepala tegak (Roll)"}

        # Tahap 3: Validasi Blink (Berkedip)
        if self._register_step == 3:
            if not (yaw_ok and pitch_ok and roll_ok):
                return {"status": "pending", "step": "HOLD", "instruction": "Tahan posisi wajah tetap lurus!"}
            
            blink_res = self._register_blink.update(face, detector)
            if blink_res["complete"]:
                self._register_step = 4
            else:
                needed = config.REGISTER_BLINK_COUNT - self._register_blink.blink_count
                return {"status": "pending", "step": "BLINK", "instruction": f"Silakan berkedip {needed} kali lagi"}

        if self._register_step == 4:
            return {"status": "complete", "step": "DONE", "instruction": "Liveness Selesai!"}

        return {"status": "pending", "step": "WAIT", "instruction": "Menunggu..."}
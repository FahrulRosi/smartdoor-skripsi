from facemesh.facemesh_detector import FaceResult, FaceMeshDetector
from liveness.blink import BlinkDetector
from liveness.head_pose import HeadPoseEstimator
import config

class LivenessManager:
    def __init__(self):
        self.pose_estimator = HeadPoseEstimator()
        self.reset_state()

    def reset_state(self):
        self._register_step = 0  
        self._register_blink = None
        # State: WAITING_EXTREME (sedang melakukan pose) atau WAITING_CENTER (jeda kembali lurus)
        self._pose_state = "WAITING_EXTREME" 

    def start_register(self):
        self.reset_state()
        self._register_blink = BlinkDetector(target_blinks=config.REGISTER_BLINK_COUNT)

    def update_register(self, face: FaceResult, detector: FaceMeshDetector) -> dict:
        pose = self.pose_estimator.estimate(face, detector)
        
        if not pose["valid"]:
            return {"status": "pending", "step": "WAIT", "instruction": "Wajah tidak terdeteksi", "progress": "Mencari wajah..."}

        yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]
        
        # Toleransi kembali ke tengah (diset ke 15.0 agar tidak terlalu kaku)
        is_center = abs(yaw) < 15.0 and abs(pitch) < 15.0 and abs(roll) < 15.0

        # ──── TAHAP 0: FACEMESH ────
        if self._register_step == 0:
            if is_center:
                self._register_step = 1
                self._pose_state = "WAITING_EXTREME"
                return {"status": "pending", "step": "FACEMESH", "instruction": "✅ Wajah Lurus", "progress": "Mulai Liveness..."}

            return {"status": "pending", "step": "FACEMESH", "instruction": "1. Tatap Lurus ke Kamera", "progress": "Arahkan wajah ke depan"}

        # ──── TAHAP 1: YAW (Toleh Kanan) ────
        elif self._register_step == 1:
            if self._pose_state == "WAITING_EXTREME":
                if yaw > config.CHALLENGE_YAW: 
                    self._pose_state = "WAITING_CENTER"
                return {"status": "pending", "step": "YAW", "instruction": "2a. Toleh ke KANAN", "progress": f"Target: >{config.CHALLENGE_YAW}° | Saat ini: {yaw:.1f}°"}
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._register_step = 2
                    self._pose_state = "WAITING_EXTREME"
                    return {"status": "pending", "step": "YAW", "instruction": "✅ Kanan Selesai", "progress": "Lanjut..."}
                return {"status": "pending", "step": "YAW", "instruction": "Tatap LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        # ──── TAHAP 2: YAW (Toleh Kiri) ────
        elif self._register_step == 2:
            if self._pose_state == "WAITING_EXTREME":
                if yaw < -config.CHALLENGE_YAW: 
                    self._pose_state = "WAITING_CENTER"
                return {"status": "pending", "step": "YAW", "instruction": "2b. Toleh ke KIRI", "progress": f"Target: <{-config.CHALLENGE_YAW}° | Saat ini: {yaw:.1f}°"}
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._register_step = 3
                    self._pose_state = "WAITING_EXTREME"
                    return {"status": "pending", "step": "YAW", "instruction": "✅ Kiri Selesai", "progress": "Lanjut..."}
                return {"status": "pending", "step": "YAW", "instruction": "Tatap LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        # ──── TAHAP 3: PITCH (Ngangguk Atas) ────
        elif self._register_step == 3:
            if self._pose_state == "WAITING_EXTREME":
                if pitch < -config.CHALLENGE_PITCH: 
                    self._pose_state = "WAITING_CENTER"
                return {"status": "pending", "step": "PITCH", "instruction": "3a. Dongak ke ATAS", "progress": f"Target: <{-config.CHALLENGE_PITCH}° | Saat ini: {pitch:.1f}°"}
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._register_step = 4
                    self._pose_state = "WAITING_EXTREME"
                    return {"status": "pending", "step": "PITCH", "instruction": "✅ Atas Selesai", "progress": "Lanjut..."}
                return {"status": "pending", "step": "PITCH", "instruction": "Tatap LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        # ──── TAHAP 4: PITCH (Ngangguk Bawah) ────
        elif self._register_step == 4:
            if self._pose_state == "WAITING_EXTREME":
                if pitch > config.CHALLENGE_PITCH: 
                    self._pose_state = "WAITING_CENTER"
                return {"status": "pending", "step": "PITCH", "instruction": "3b. Tunduk ke BAWAH", "progress": f"Target: >{config.CHALLENGE_PITCH}° | Saat ini: {pitch:.1f}°"}
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._register_step = 5
                    self._pose_state = "WAITING_EXTREME"
                    return {"status": "pending", "step": "PITCH", "instruction": "✅ Bawah Selesai", "progress": "Lanjut..."}
                return {"status": "pending", "step": "PITCH", "instruction": "Tatap LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        # ──── TAHAP 5: ROLL (Miring Kanan) ────
        elif self._register_step == 5:
            if self._pose_state == "WAITING_EXTREME":
                if roll < -config.CHALLENGE_ROLL: 
                    self._pose_state = "WAITING_CENTER"
                return {"status": "pending", "step": "ROLL", "instruction": "4a. Miring ke KANAN", "progress": f"Target: <{-config.CHALLENGE_ROLL}° | Saat ini: {roll:.1f}°"}
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._register_step = 6
                    self._pose_state = "WAITING_EXTREME"
                    return {"status": "pending", "step": "ROLL", "instruction": "✅ Miring Kanan Selesai", "progress": "Lanjut..."}
                return {"status": "pending", "step": "ROLL", "instruction": "Tatap LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        # ──── TAHAP 6: ROLL (Miring Kiri) ────
        elif self._register_step == 6:
            if self._pose_state == "WAITING_EXTREME":
                if roll > config.CHALLENGE_ROLL: 
                    self._pose_state = "WAITING_CENTER"
                return {"status": "pending", "step": "ROLL", "instruction": "4b. Miring ke KIRI", "progress": f"Target: >{config.CHALLENGE_ROLL}° | Saat ini: {roll:.1f}°"}
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._register_step = 7
                    self._pose_state = "WAITING_EXTREME"
                    return {"status": "pending", "step": "ROLL", "instruction": "✅ Miring Kiri Selesai", "progress": "Lanjut..."}
                return {"status": "pending", "step": "ROLL", "instruction": "Tatap LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        # ──── TAHAP 7: BLINK (Wajib Kedip) ────
        elif self._register_step == 7:
            blink_res = self._register_blink.update(face, detector)
            if blink_res["complete"]:
                self._register_step = 8
                return {"status": "pending", "step": "BLINK", "instruction": "✅ Blink Selesai", "progress": "Validasi Selesai..."}
            
            needed = max(0, config.REGISTER_BLINK_COUNT - self._register_blink.blink_count)
            return {"status": "pending", "step": "BLINK", "instruction": f"5. Kedipkan Mata ({needed}x)", "progress": "Lakukan Kedipan..."}

        # ──── TAHAP 8: SELESAI (Ekstraksi MobileFaceNet) ────
        elif self._register_step == 8:
            return {"status": "complete", "step": "DONE", "instruction": "Semua Liveness Berhasil!", "progress": "Mengekstrak MobileFaceNet..."}

        return {"status": "pending", "step": "WAIT", "instruction": "Menunggu...", "progress": "WAIT"}
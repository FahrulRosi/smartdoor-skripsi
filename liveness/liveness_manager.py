from facemesh.facemesh_detector import FaceResult, FaceMeshDetector
from liveness.blink import BlinkDetector
from liveness.head_pose import HeadPoseEstimator
import config

class LivenessManager:
    def __init__(self):
        self.pose_estimator = HeadPoseEstimator()
        self.reset_state()

    def reset_state(self):
        # 0: Stabil, 1: Yaw, 2: Pitch, 3: Roll, 4: Blink, 5: Done
        self._register_step = 0  
        self._register_blink = None
        
        # Tracker untuk memastikan pengguna menyelesaikan 2 arah pergerakan
        self._dir_1_done = False
        self._dir_2_done = False
        
        # Tracker untuk menahan frame agar benar-benar stabil di tahap awal
        self._step_frame_count = 0  
        self._required_frames = 3  # Butuh 10 frame beruntun (tanpa timer)

    def start_register(self):
        self.reset_state()
        self._register_blink = BlinkDetector(target_blinks=config.REGISTER_BLINK_COUNT)

    def update_register(self, face: FaceResult, detector: FaceMeshDetector) -> dict:
        pose = self.pose_estimator.estimate(face, detector)
        
        if not pose["valid"]:
            return {
                "status": "pending", "step": "WAIT",
                "instruction": "Wajah tidak terdeteksi",
                "progress": "Mencari wajah..."
            }

        yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]

        # ──── TAHAP 0: FACEMESH (Wajib Tatap Lurus & Stabil) ────
        if self._register_step == 0:
            if abs(yaw) < config.MAX_YAW and abs(pitch) < config.MAX_PITCH and abs(roll) < config.MAX_ROLL:
                self._step_frame_count += 1
            else:
                self._step_frame_count = 0 # Reset jika wajah bergerak, menuntut kestabilan penuh

            if self._step_frame_count >= self._required_frames:
                self._register_step = 1
                self._step_frame_count = 0
                return {"status": "pending", "step": "FACEMESH", "instruction": "✅ Wajah Stabil", "progress": "Lanjut Yaw..."}

            return {
                "status": "pending", "step": "FACEMESH",
                "instruction": "1. Tatap Lurus ke Kamera",
                "progress": f"Stabil: {self._step_frame_count}/{self._required_frames}",
                "yaw": f"{yaw:.1f}°"
            }

        # ──── TAHAP 1: YAW (Wajib Kanan & Kiri) ────
        elif self._register_step == 1:
            if yaw > config.CHALLENGE_YAW: self._dir_1_done = True
            if yaw < -config.CHALLENGE_YAW: self._dir_2_done = True

            if self._dir_1_done and self._dir_2_done:
                self._register_step = 2
                self._dir_1_done, self._dir_2_done = False, False 
                return {"status": "pending", "step": "YAW", "instruction": "✅ Yaw Selesai", "progress": "Lanjut Pitch..."}

            prog_text = f"Kanan {'✅' if self._dir_1_done else '❌'} | Kiri {'✅' if self._dir_2_done else '❌'}"
            return {"status": "pending", "step": "YAW", "instruction": "2. Toleh Kanan & Kiri", "progress": prog_text}

        # ──── TAHAP 2: PITCH (Wajib Bawah & Atas) ────
        elif self._register_step == 2:
            if pitch > config.CHALLENGE_PITCH: self._dir_1_done = True   # Tunduk
            if pitch < -config.CHALLENGE_PITCH: self._dir_2_done = True  # Angkat

            if self._dir_1_done and self._dir_2_done:
                self._register_step = 3
                self._dir_1_done, self._dir_2_done = False, False
                return {"status": "pending", "step": "PITCH", "instruction": "✅ Pitch Selesai", "progress": "Lanjut Roll..."}

            prog_text = f"Bawah {'✅' if self._dir_1_done else '❌'} | Atas {'✅' if self._dir_2_done else '❌'}"
            return {"status": "pending", "step": "PITCH", "instruction": "3. Tunduk & Angkat Kepala", "progress": prog_text}

        # ──── TAHAP 3: ROLL (Wajib Miring Kanan & Kiri) ────
        elif self._register_step == 3:
            if roll > config.CHALLENGE_ROLL: self._dir_1_done = True
            if roll < -config.CHALLENGE_ROLL: self._dir_2_done = True

            if self._dir_1_done and self._dir_2_done:
                self._register_step = 4
                return {"status": "pending", "step": "ROLL", "instruction": "✅ Roll Selesai", "progress": "Lanjut Blink..."}

            prog_text = f"Kanan {'✅' if self._dir_1_done else '❌'} | Kiri {'✅' if self._dir_2_done else '❌'}"
            return {"status": "pending", "step": "ROLL", "instruction": "4. Miring Kanan & Kiri", "progress": prog_text}

        # ──── TAHAP 4: BLINK (Wajib Kedip) ────
        elif self._register_step == 4:
            blink_res = self._register_blink.update(face, detector)
            if blink_res["complete"]:
                self._register_step = 5
                return {"status": "pending", "step": "BLINK", "instruction": "✅ Blink Selesai", "progress": "Validasi Selesai..."}
            
            needed = max(0, config.REGISTER_BLINK_COUNT - self._register_blink.blink_count)
            return {
                "status": "pending", "step": "BLINK", 
                "instruction": f"5. Kedipkan Mata ({needed}x)",
                "progress": f"Blink: {self._register_blink.blink_count}/{config.REGISTER_BLINK_COUNT}"
            }

        # ──── TAHAP 5: SELESAI (Ekstraksi MobileFaceNet) ────
        elif self._register_step == 5:
            return {
                "status": "complete", "step": "DONE",
                "instruction": "Semua Liveness Berhasil!",
                "progress": "Mengekstrak MobileFaceNet..."
            }

        return {"status": "pending", "step": "WAIT", "instruction": "Menunggu...", "progress": "WAIT"}
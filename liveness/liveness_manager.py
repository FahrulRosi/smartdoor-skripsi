from facemesh.facemesh_detector import FaceResult, FaceMeshDetector
from liveness.blink import BlinkDetector
from liveness.head_pose import HeadPoseEstimator
import config

class LivenessManager:
    def __init__(self):
        self.pose_estimator = HeadPoseEstimator()
        self._register_step = 0  # 0: FaceMesh, 1: Yaw, 2: Pitch, 3: Roll, 4: Blink, 5: Done
        self._register_blink = None
        
        # Tracker untuk memastikan pengguna melakukan gerakan ke DUA ARAH (Kiri & Kanan, Atas & Bawah)
        self._dir_1_done = False
        self._dir_2_done = False
        
        self._step_frame_count = 0  
        self._required_frames = 3  

    def start_register(self):
        self._register_step = 0
        self._step_frame_count = 0
        self._dir_1_done = False
        self._dir_2_done = False
        self._register_blink = BlinkDetector(target_blinks=config.REGISTER_BLINK_COUNT)

    def update_register(self, face: FaceResult, detector: FaceMeshDetector) -> dict:
        pose = self.pose_estimator.estimate(face, detector)
        
        if not pose["valid"]:
            return {
                "status": "pending",
                "step": "NO_FACE",
                "instruction": "Wajah tidak terdeteksi",
                "progress": "Menunggu wajah..."
            }

        yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]

        # ──── TAHAP 0: FACEMESH (Tatap Lurus di Awal Saja) ────
        if self._register_step == 0:
            if abs(yaw) < config.MAX_YAW and abs(pitch) < config.MAX_PITCH and abs(roll) < config.MAX_ROLL:
                self._step_frame_count += 1
            else:
                self._step_frame_count = max(0, self._step_frame_count - 1)

            if self._step_frame_count >= self._required_frames:
                self._register_step = 1
                self._step_frame_count = 0
                return {"status": "pending", "step": "FACEMESH", "instruction": "✅ Wajah terdeteksi", "progress": "Mulai Liveness..."}

            return {
                "status": "pending",
                "step": "FACEMESH",
                "instruction": "1. Tatap Lurus ke Kamera",
                "progress": f"Stabil: {self._step_frame_count}/{self._required_frames}",
                "yaw": f"{yaw:.1f}°"
            }

        # ──── TAHAP 1: YAW (Toleh Kanan DAN Kiri) ────
        elif self._register_step == 1:
            # Validasi langsung tanpa delay: Jika yaw melebihi batas positif (kanan) dan negatif (kiri)
            if yaw > config.CHALLENGE_YAW: self._dir_1_done = True
            if yaw < -config.CHALLENGE_YAW: self._dir_2_done = True

            # Jika KEDUA arah sudah dilakukan, LANGSUNG lompat ke Pitch
            if self._dir_1_done and self._dir_2_done:
                self._register_step = 2
                self._dir_1_done, self._dir_2_done = False, False # Reset untuk tahap berikutnya
                return {"status": "pending", "step": "YAW", "instruction": "✅ Yaw Selesai", "progress": "Lanjut Pitch..."}

            prog_text = ""
            prog_text += "Kanan ✅ " if self._dir_1_done else "Kanan ❌ "
            prog_text += "| Kiri ✅" if self._dir_2_done else "| Kiri ❌"

            return {
                "status": "pending",
                "step": "YAW",
                "instruction": "2. Tolehkan Kanan DAN Kiri",
                "progress": prog_text,
                "yaw": f"{yaw:.1f}°"
            }

        # ──── TAHAP 2: PITCH (Tunduk DAN Angkat) ────
        elif self._register_step == 2:
            if pitch > config.CHALLENGE_PITCH: self._dir_1_done = True   # Tunduk
            if pitch < -config.CHALLENGE_PITCH: self._dir_2_done = True  # Angkat

            # Jika KEDUA arah sudah dilakukan, LANGSUNG lompat ke Roll
            if self._dir_1_done and self._dir_2_done:
                self._register_step = 3
                self._dir_1_done, self._dir_2_done = False, False
                return {"status": "pending", "step": "PITCH", "instruction": "✅ Pitch Selesai", "progress": "Lanjut Roll..."}

            prog_text = ""
            prog_text += "Bawah ✅ " if self._dir_1_done else "Bawah ❌ "
            prog_text += "| Atas ✅" if self._dir_2_done else "| Atas ❌"

            return {
                "status": "pending",
                "step": "PITCH",
                "instruction": "3. Tunduk DAN Angkat Kepala",
                "progress": prog_text,
                "pitch": f"{pitch:.1f}°"
            }

        # ──── TAHAP 3: ROLL (Miring Kanan DAN Kiri) ────
        elif self._register_step == 3:
            if roll > config.CHALLENGE_ROLL: self._dir_1_done = True
            if roll < -config.CHALLENGE_ROLL: self._dir_2_done = True

            # Jika KEDUA arah sudah dilakukan, LANGSUNG lompat ke Blink
            if self._dir_1_done and self._dir_2_done:
                self._register_step = 4
                return {"status": "pending", "step": "ROLL", "instruction": "✅ Roll Selesai", "progress": "Lanjut Blink..."}

            prog_text = ""
            prog_text += "Miring Kanan ✅ " if self._dir_1_done else "Miring Kanan ❌ "
            prog_text += "| Miring Kiri ✅" if self._dir_2_done else "| Miring Kiri ❌"

            return {
                "status": "pending",
                "step": "ROLL",
                "instruction": "4. Miringkan Kanan DAN Kiri",
                "progress": prog_text,
                "roll": f"{roll:.1f}°"
            }

        # ──── TAHAP 4: BLINK (Berkedip) ────
        elif self._register_step == 4:
            blink_res = self._register_blink.update(face, detector)
            if blink_res["complete"]:
                self._register_step = 5
                return {"status": "pending", "step": "BLINK", "instruction": "✅ Blink Selesai", "progress": "Validasi..."}
            else:
                needed = max(0, config.REGISTER_BLINK_COUNT - self._register_blink.blink_count)
                return {
                    "status": "pending",
                    "step": "BLINK",
                    "instruction": f"5. Kedipkan Mata ({needed}x)",
                    "progress": f"Blink: {self._register_blink.blink_count}/{config.REGISTER_BLINK_COUNT}"
                }

        # ──── TAHAP 5: SELESAI (Lanjut ke Ekstraksi) ────
        elif self._register_step == 5:
            return {
                "status": "complete",
                "step": "DONE",
                "instruction": "Liveness Berhasil!",
                "progress": "Memproses MobileFaceNet..."
            }

        return {"status": "pending", "step": "WAIT", "instruction": "Menunggu...", "progress": "WAIT"}
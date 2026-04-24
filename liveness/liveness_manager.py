from facemesh.facemesh_detector import FaceResult, FaceMeshDetector
from liveness.blink import BlinkDetector
from liveness.head_pose import HeadPoseEstimator
import config

class LivenessManager:
    def __init__(self):
        self.pose_estimator = HeadPoseEstimator()
        self._register_step = 0  # 0: FaceMesh, 1: Yaw, 2: Pitch, 3: Roll, 4: Blink, 5: Done
        self._register_blink = None
        
        # ── FRAME COUNTERS untuk validasi stabil ──
        self._step_frame_count = 0  
        self._required_frames = 10  # Cukup 10 frame agar responsif
        self._last_valid_pose = None
        
        # [BARU] Counter untuk jeda antar tahap agar tidak ke-skip
        self._cooldown_frames = 0

    def start_register(self):
        self._register_step = 0
        self._step_frame_count = 0
        self._register_blink = BlinkDetector(target_blinks=config.REGISTER_BLINK_COUNT)
        self._last_valid_pose = None
        self._cooldown_frames = 0

    def update_register(self, face: FaceResult, detector: FaceMeshDetector) -> dict:
        pose = self.pose_estimator.estimate(face, detector)
        
        if not pose["valid"]:
            self._step_frame_count = 0  
            return {
                "status": "pending",
                "step": "NO_FACE",
                "instruction": "Wajah tidak terdeteksi",
                "progress": f"Frame: 0/{self._required_frames}"
            }

        yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]

        # ──── [BARU] JEDA TRANSISI & WAJIB KEMBALI LURUS ────
        if self._cooldown_frames > 0:
            # Tahap selanjutnya DITAHAN sampai wajah kembali melihat LURUS ke kamera
            if abs(yaw) < 15 and abs(pitch) < 15 and abs(roll) < 15:
                self._cooldown_frames -= 1
                instruction = "✅ Bagus! Tahan posisi lurus..."
            else:
                # Jika kepala masih miring/menoleh, cooldown berhenti
                instruction = "⚠️ Kembalikan wajah melihat LURUS ke depan"
            
            return {
                "status": "pending",
                "step": "WAIT",
                "instruction": instruction,
                "progress": "Persiapan transisi..."
            }

        # ──── TAHAP 0: FACEMESH (Lurus) ────
        if self._register_step == 0:
            if abs(yaw) < 15 and abs(pitch) < 15 and abs(roll) < 15:
                self._step_frame_count += 1
            else:
                self._step_frame_count = max(0, self._step_frame_count - 1)

            if self._step_frame_count >= self._required_frames:
                self._register_step = 1
                self._step_frame_count = 0
                self._cooldown_frames = 20  # Beri jeda 20 frame untuk bernapas
                return {
                    "status": "pending",
                    "step": "FACEMESH",
                    "instruction": "✅ Tahap Facemesh berhasil!",
                    "progress": "DONE"
                }

            return {
                "status": "pending",
                "step": "FACEMESH",
                "instruction": "1. Tatap kamera dengan lurus (jangan gerak)",
                "progress": f"Stabil: {self._step_frame_count}/{self._required_frames}",
                "yaw": f"{yaw:.1f}°", "pitch": f"{pitch:.1f}°", "roll": f"{roll:.1f}°"
            }

        # ──── TAHAP 1: YAW (Toleh Kiri/Kanan) ────
        elif self._register_step == 1:
            if abs(yaw) > 15:
                self._step_frame_count += 1
            else:
                self._step_frame_count = max(0, self._step_frame_count - 1) 

            if self._step_frame_count >= self._required_frames:
                self._register_step = 2
                self._step_frame_count = 0
                self._cooldown_frames = 30  # Jeda sebelum Pitch, wajib kembali lurus
                return {
                    "status": "pending",
                    "step": "YAW",
                    "instruction": "✅ Tahap Yaw berhasil!",
                    "progress": "DONE"
                }

            return {
                "status": "pending",
                "step": "YAW",
                "instruction": "2. Tolehkan kepala ke Kiri atau Kanan (min 15°)",
                "progress": f"Terdeteksi: {self._step_frame_count}/{self._required_frames}",
                "yaw": f"{yaw:.1f}° {'⏳' if abs(yaw) > 15 else '❌'}"
            }

        # ──── TAHAP 2: PITCH (Angkat/Tunduk) ────
        elif self._register_step == 2:
            if abs(pitch) > 15:
                self._step_frame_count += 1
            else:
                self._step_frame_count = max(0, self._step_frame_count - 1)

            if self._step_frame_count >= self._required_frames:
                self._register_step = 3
                self._step_frame_count = 0
                self._cooldown_frames = 30  # Jeda sebelum Roll, wajib kembali lurus
                return {
                    "status": "pending",
                    "step": "PITCH",
                    "instruction": "✅ Tahap Pitch berhasil!",
                    "progress": "DONE"
                }

            return {
                "status": "pending",
                "step": "PITCH",
                "instruction": "3. Angkat atau Tundukkan kepala (min 15°)",
                "progress": f"Terdeteksi: {self._step_frame_count}/{self._required_frames}",
                "pitch": f"{pitch:.1f}° {'⏳' if abs(pitch) > 15 else '❌'}"
            }

        # ──── TAHAP 3: ROLL (Miringkan Kepala) ────
        elif self._register_step == 3:
            if abs(roll) > 15:
                self._step_frame_count += 1
            else:
                self._step_frame_count = max(0, self._step_frame_count - 1)

            if self._step_frame_count >= self._required_frames:
                self._register_step = 4
                self._step_frame_count = 0
                self._cooldown_frames = 20  # Jeda sebelum Blink
                return {
                    "status": "pending",
                    "step": "ROLL",
                    "instruction": "✅ Tahap Roll berhasil!",
                    "progress": "DONE"
                }

            return {
                "status": "pending",
                "step": "ROLL",
                "instruction": "4. Miringkan kepala ke samping (min 15°)",
                "progress": f"Terdeteksi: {self._step_frame_count}/{self._required_frames}",
                "roll": f"{roll:.1f}° {'⏳' if abs(roll) > 15 else '❌'}"
            }

        # ──── TAHAP 4: BLINK (Berkedip) ────
        elif self._register_step == 4:
            blink_res = self._register_blink.update(face, detector)
            if blink_res["complete"]:
                self._register_step = 5
                self._step_frame_count = 0
                return {
                    "status": "pending",
                    "step": "BLINK",
                    "instruction": "✅ Tahap Blink berhasil! Lanjut ekstraksi",
                    "progress": "DONE"
                }
            else:
                needed = max(0, config.REGISTER_BLINK_COUNT - self._register_blink.blink_count)
                return {
                    "status": "pending",
                    "step": "BLINK",
                    "instruction": f"5. Berkedip {needed} kali lagi",
                    "progress": f"Blink: {self._register_blink.blink_count}/{config.REGISTER_BLINK_COUNT}"
                }

        # ──── TAHAP 5: SELESAI ────
        elif self._register_step == 5:
            return {
                "status": "complete",
                "step": "DONE",
                "instruction": "Liveness Berhasil! Mengambil data...",
                "progress": "COMPLETE"
            }

        return {
            "status": "pending",
            "step": "WAIT",
            "instruction": "Menunggu...",
            "progress": "WAIT"
        }
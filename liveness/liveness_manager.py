from facemesh.facemesh_detector import FaceResult, FaceMeshDetector
from liveness.blink import BlinkDetector
from liveness.head_pose import HeadPoseEstimator
import config

class LivenessManager:
    def __init__(self):
        self.pose_estimator = HeadPoseEstimator()
        self._register_step = 0  # 0: FaceMesh, 1: Yaw, 2: Pitch, 3: Roll, 4: Blink, 5: Done
        self._register_blink = None
        self._step_frame_count = 0  
        self._required_frames = 5  # Memastikan gerakan disengaja/stabil
        self._wait_for_center = False

    def start_register(self):
        self._register_step = 0
        self._step_frame_count = 0
        self._register_blink = BlinkDetector(target_blinks=config.REGISTER_BLINK_COUNT)
        self._wait_for_center = False

    def update_register(self, face: FaceResult, detector: FaceMeshDetector) -> dict:
        pose = self.pose_estimator.estimate(face, detector)
        
        if not pose["valid"]:
            self._step_frame_count = 0  
            return {
                "status": "pending",
                "step": "NO_FACE",
                "instruction": "Wajah tidak terdeteksi",
                "progress": "Validasi tertunda"
            }

        yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]

        # ──── VALIDASI WAJIB KEMBALI LURUS ANTAR TAHAP ────
        if self._wait_for_center:
            if abs(yaw) < config.MAX_YAW and abs(pitch) < config.MAX_PITCH and abs(roll) < config.MAX_ROLL:
                self._wait_for_center = False
                return {
                    "status": "pending",
                    "step": "WAIT",
                    "instruction": "✅ Bagus! Posisi lurus tervalidasi.",
                    "progress": "Melanjutkan..."
                }
            else:
                return {
                    "status": "pending",
                    "step": "WAIT",
                    "instruction": "⚠️ Kembalikan wajah melihat LURUS ke depan",
                    "progress": "Menunggu posisi lurus..."
                }

        # ──── TAHAP 0: FACEMESH (Lurus) ────
        if self._register_step == 0:
            if abs(yaw) < config.MAX_YAW and abs(pitch) < config.MAX_PITCH and abs(roll) < config.MAX_ROLL:
                self._step_frame_count += 1
            else:
                self._step_frame_count = max(0, self._step_frame_count - 1)

            if self._step_frame_count >= self._required_frames:
                self._register_step, self._step_frame_count, self._wait_for_center = 1, 0, True
                return {"status": "pending", "step": "FACEMESH", "instruction": "✅ Tahap Facemesh berhasil!", "progress": "DONE"}

            return {
                "status": "pending",
                "step": "FACEMESH",
                "instruction": "1. Tatap kamera dengan lurus (jangan gerak)",
                "progress": f"Stabil: {self._step_frame_count}/{self._required_frames}",
                "yaw": f"{yaw:.1f}°"
            }

        # ──── TAHAP 1: YAW (Toleh Kiri/Kanan) ────
        elif self._register_step == 1:
            if abs(yaw) > config.CHALLENGE_YAW:
                self._step_frame_count += 1
            else:
                self._step_frame_count = max(0, self._step_frame_count - 1) 

            if self._step_frame_count >= self._required_frames:
                self._register_step, self._step_frame_count, self._wait_for_center = 2, 0, True
                return {"status": "pending", "step": "YAW", "instruction": "✅ Tahap Yaw berhasil!", "progress": "DONE"}

            return {
                "status": "pending",
                "step": "YAW",
                "instruction": f"2. Tolehkan kepala ke Kiri/Kanan (min {config.CHALLENGE_YAW}°)",
                "progress": f"Terdeteksi: {self._step_frame_count}/{self._required_frames}",
                "yaw": f"{yaw:.1f}° {'✅' if abs(yaw) > config.CHALLENGE_YAW else '❌'}"
            }

        # ──── TAHAP 2: PITCH (Angkat/Tunduk) ────
        elif self._register_step == 2:
            if abs(pitch) > config.CHALLENGE_PITCH:
                self._step_frame_count += 1
            else:
                self._step_frame_count = max(0, self._step_frame_count - 1)

            if self._step_frame_count >= self._required_frames:
                self._register_step, self._step_frame_count, self._wait_for_center = 3, 0, True
                return {"status": "pending", "step": "PITCH", "instruction": "✅ Tahap Pitch berhasil!", "progress": "DONE"}

            return {
                "status": "pending",
                "step": "PITCH",
                "instruction": f"3. Angkat atau Tundukkan kepala (min {config.CHALLENGE_PITCH}°)",
                "progress": f"Terdeteksi: {self._step_frame_count}/{self._required_frames}",
                "pitch": f"{pitch:.1f}° {'✅' if abs(pitch) > config.CHALLENGE_PITCH else '❌'}"
            }

        # ──── TAHAP 3: ROLL (Miringkan Kepala) ────
        elif self._register_step == 3:
            if abs(roll) > config.CHALLENGE_ROLL:
                self._step_frame_count += 1
            else:
                self._step_frame_count = max(0, self._step_frame_count - 1)

            if self._step_frame_count >= self._required_frames:
                self._register_step, self._step_frame_count, self._wait_for_center = 4, 0, True
                return {"status": "pending", "step": "ROLL", "instruction": "✅ Tahap Roll berhasil!", "progress": "DONE"}

            return {
                "status": "pending",
                "step": "ROLL",
                "instruction": f"4. Miringkan kepala ke samping (min {config.CHALLENGE_ROLL}°)",
                "progress": f"Terdeteksi: {self._step_frame_count}/{self._required_frames}",
                "roll": f"{roll:.1f}° {'✅' if abs(roll) > config.CHALLENGE_ROLL else '❌'}"
            }

        # ──── TAHAP 4: BLINK (Berkedip) ────
        elif self._register_step == 4:
            blink_res = self._register_blink.update(face, detector)
            if blink_res["complete"]:
                self._register_step = 5
                self._step_frame_count = 0
                return {"status": "pending", "step": "BLINK", "instruction": "✅ Tahap Blink berhasil!", "progress": "DONE"}
            else:
                needed = max(0, config.REGISTER_BLINK_COUNT - self._register_blink.blink_count)
                return {
                    "status": "pending",
                    "step": "BLINK",
                    "instruction": f"5. Berkedip {needed} kali lagi",
                    "progress": f"Blink: {self._register_blink.blink_count}/{config.REGISTER_BLINK_COUNT}"
                }

        # ──── TAHAP 5: SELESAI (Masuk Tahap Ekstraksi) ────
        elif self._register_step == 5:
            return {
                "status": "complete",
                "step": "DONE",
                "instruction": "Liveness Berhasil! Memasuki mode Ekstraksi...",
                "progress": "COMPLETE"
            }

        return {"status": "pending", "step": "WAIT", "instruction": "Menunggu...", "progress": "WAIT"}
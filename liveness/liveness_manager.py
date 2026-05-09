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
        
        # --- PERBAIKAN ANTI-SKIP ---
        self._hold_frames = 0         # Penghitung frame (berapa lama pose ditahan)
        self._required_frames = 3     # Wajib tahan posisi minimal 3 frame berturut-turut
        # ---------------------------

    def start_register(self):
        self.reset_state()
        self._register_blink = BlinkDetector(target_blinks=getattr(config, 'REGISTER_BLINK_COUNT', 1))

    def update_register(self, face: FaceResult, detector: FaceMeshDetector) -> dict:
        pose = self.pose_estimator.estimate(face, detector)
        
        if not pose["valid"]:
            self._hold_frames = 0 # Reset frame jika wajah hilang
            return {"status": "pending", "step": "WAIT", "instruction": "Wajah tidak terdeteksi", "progress": "Mencari wajah..."}

        yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]
        
        # Toleransi kembali ke tengah (diset ke 15.0 agar tidak terlalu kaku)
        is_center = abs(yaw) < 15.0 and abs(pitch) < 15.0 and abs(roll) < 15.0

        # Mengambil Threshold dari config dengan aman
        thr_y = getattr(config, 'CHALLENGE_YAW', 20)
        thr_p = getattr(config, 'CHALLENGE_PITCH', 15)
        thr_r = getattr(config, 'CHALLENGE_ROLL', 15)

        # ──── TAHAP 0: FACEMESH ────
        if self._register_step == 0:
            if is_center:
                self._hold_frames += 1
                if self._hold_frames >= 10: # Khusus tahap awal, butuh tahan 10 frame lurus untuk kalibrasi
                    self._register_step = 1
                    self._pose_state = "WAITING_EXTREME"
                    self._hold_frames = 0
                    return {"status": "pending", "step": "FACEMESH", "instruction": "✅ Wajah Lurus", "progress": "Mulai Liveness..."}
                return {"status": "pending", "step": "FACEMESH", "instruction": "1. Tahan Posisi Lurus", "progress": f"Menahan... ({self._hold_frames}/10)"}
            else:
                self._hold_frames = 0 # Reset jika bergerak
            return {"status": "pending", "step": "FACEMESH", "instruction": "1. Tatap Lurus ke Kamera", "progress": "Arahkan wajah ke depan"}

        # ──── TAHAP 1: YAW (Toleh Kanan) ────
        elif self._register_step == 1:
            if self._pose_state == "WAITING_EXTREME":
                if yaw > thr_y: 
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames: # Wajib ditahan
                        self._pose_state = "WAITING_CENTER"
                        self._hold_frames = 0
                else:
                    self._hold_frames = 0
                return {"status": "pending", "step": "YAW", "instruction": "2a. Toleh ke KANAN", "progress": f"Target: >{thr_y}° | Saat ini: {yaw:.1f}°"}
            
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames:
                        self._register_step = 2
                        self._pose_state = "WAITING_EXTREME"
                        self._hold_frames = 0
                        return {"status": "pending", "step": "YAW", "instruction": "✅ Kanan Selesai", "progress": "Lanjut..."}
                else:
                    self._hold_frames = 0
                return {"status": "pending", "step": "YAW", "instruction": "Tahan LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        # ──── TAHAP 2: YAW (Toleh Kiri) ────
        elif self._register_step == 2:
            if self._pose_state == "WAITING_EXTREME":
                if yaw < -thr_y: 
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames:
                        self._pose_state = "WAITING_CENTER"
                        self._hold_frames = 0
                else:
                    self._hold_frames = 0
                return {"status": "pending", "step": "YAW", "instruction": "2b. Toleh ke KIRI", "progress": f"Target: <{-thr_y}° | Saat ini: {yaw:.1f}°"}
            
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames:
                        self._register_step = 3
                        self._pose_state = "WAITING_EXTREME"
                        self._hold_frames = 0
                        return {"status": "pending", "step": "YAW", "instruction": "✅ Kiri Selesai", "progress": "Lanjut..."}
                else:
                    self._hold_frames = 0
                return {"status": "pending", "step": "YAW", "instruction": "Tahan LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        # ──── TAHAP 3: PITCH (Ngangguk Atas) ────
        elif self._register_step == 3:
            if self._pose_state == "WAITING_EXTREME":
                if pitch < -thr_p: 
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames:
                        self._pose_state = "WAITING_CENTER"
                        self._hold_frames = 0
                else:
                    self._hold_frames = 0
                return {"status": "pending", "step": "PITCH", "instruction": "3a. Dongak ke ATAS", "progress": f"Target: <{-thr_p}° | Saat ini: {pitch:.1f}°"}
            
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames:
                        self._register_step = 4
                        self._pose_state = "WAITING_EXTREME"
                        self._hold_frames = 0
                        return {"status": "pending", "step": "PITCH", "instruction": "✅ Atas Selesai", "progress": "Lanjut..."}
                else:
                    self._hold_frames = 0
                return {"status": "pending", "step": "PITCH", "instruction": "Tahan LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        # ──── TAHAP 4: PITCH (Ngangguk Bawah) ────
        elif self._register_step == 4:
            if self._pose_state == "WAITING_EXTREME":
                if pitch > thr_p: 
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames:
                        self._pose_state = "WAITING_CENTER"
                        self._hold_frames = 0
                else:
                    self._hold_frames = 0
                return {"status": "pending", "step": "PITCH", "instruction": "3b. Tunduk ke BAWAH", "progress": f"Target: >{thr_p}° | Saat ini: {pitch:.1f}°"}
            
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames:
                        self._register_step = 5
                        self._pose_state = "WAITING_EXTREME"
                        self._hold_frames = 0
                        return {"status": "pending", "step": "PITCH", "instruction": "✅ Bawah Selesai", "progress": "Lanjut..."}
                else:
                    self._hold_frames = 0
                return {"status": "pending", "step": "PITCH", "instruction": "Tahan LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        # ──── TAHAP 5: ROLL (Miring Kanan) ────
        elif self._register_step == 5:
            if self._pose_state == "WAITING_EXTREME":
                if roll < -thr_r: 
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames:
                        self._pose_state = "WAITING_CENTER"
                        self._hold_frames = 0
                else:
                    self._hold_frames = 0
                return {"status": "pending", "step": "ROLL", "instruction": "4a. Miring ke KANAN", "progress": f"Target: <{-thr_r}° | Saat ini: {roll:.1f}°"}
            
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames:
                        self._register_step = 6
                        self._pose_state = "WAITING_EXTREME"
                        self._hold_frames = 0
                        return {"status": "pending", "step": "ROLL", "instruction": "✅ Miring Kanan Selesai", "progress": "Lanjut..."}
                else:
                    self._hold_frames = 0
                return {"status": "pending", "step": "ROLL", "instruction": "Tahan LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        # ──── TAHAP 6: ROLL (Miring Kiri) ────
        elif self._register_step == 6:
            if self._pose_state == "WAITING_EXTREME":
                if roll > thr_r: 
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames:
                        self._pose_state = "WAITING_CENTER"
                        self._hold_frames = 0
                else:
                    self._hold_frames = 0
                return {"status": "pending", "step": "ROLL", "instruction": "4b. Miring ke KIRI", "progress": f"Target: >{thr_r}° | Saat ini: {roll:.1f}°"}
            
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames:
                        self._register_step = 7
                        self._pose_state = "WAITING_EXTREME"
                        self._hold_frames = 0
                        return {"status": "pending", "step": "ROLL", "instruction": "✅ Miring Kiri Selesai", "progress": "Lanjut..."}
                else:
                    self._hold_frames = 0
                return {"status": "pending", "step": "ROLL", "instruction": "Tahan LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        # ──── TAHAP 7: BLINK (Wajib Kedip) ────
        elif self._register_step == 7:
            # Blink Detector sudah memiliki internal holding mechanisms
            blink_res = self._register_blink.update(face, detector)
            if blink_res["complete"]:
                self._register_step = 8
                return {"status": "pending", "step": "BLINK", "instruction": "✅ Blink Selesai", "progress": "Validasi Selesai..."}
            
            needed = max(0, getattr(config, 'REGISTER_BLINK_COUNT', 1) - self._register_blink.blink_count)
            return {"status": "pending", "step": "BLINK", "instruction": f"5. Kedipkan Mata ({needed}x)", "progress": "Lakukan Kedipan..."}

        # ──── TAHAP 8: SELESAI (Ekstraksi MobileFaceNet) ────
        elif self._register_step == 8:
            return {"status": "complete", "step": "DONE", "instruction": "Semua Liveness Berhasil!", "progress": "Mengekstrak MobileFaceNet..."}

        return {"status": "pending", "step": "WAIT", "instruction": "Menunggu...", "progress": "WAIT"}
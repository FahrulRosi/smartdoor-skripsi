import numpy as np
from facemesh.facemesh_detector import FaceResult, FaceMeshDetector
from liveness.head_pose import HeadPoseEstimator
import config

class LivenessManager:
    def __init__(self):
        self.pose_estimator = HeadPoseEstimator()
        self.reset_state()

    def reset_state(self):
        self._register_step = 0  
        self._pose_state = "WAITING_EXTREME" 
        self._hold_frames = 0         
        self._required_frames = 3     
        
        # Variabel khusus Kalibrasi Anti-Noise (Blink 2x)
        self._blink_state = 0         
        self._ear_history = []
        self._base_open_ear = 0.0
        self._blink_count = 0

    def start_register(self):
        self.reset_state()

    def update_register(self, face: FaceResult, detector: FaceMeshDetector) -> dict:
        pose = self.pose_estimator.estimate(face, detector)
        
        if not pose["valid"]:
            self._hold_frames = 0
            return {"status": "pending", "step": "WAIT", "instruction": "Wajah tidak terdeteksi", "progress": "Mencari wajah..."}

        yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]
        is_center = abs(yaw) < 15.0 and abs(pitch) < 15.0 and abs(roll) < 15.0

        thr_y = getattr(config, 'CHALLENGE_YAW', 25.0)
        thr_p = getattr(config, 'CHALLENGE_PITCH', 20.0)
        thr_r = getattr(config, 'CHALLENGE_ROLL', 25.0)

        # ──── TAHAPAN 0 - 6 (Wajah Lurus, Yaw, Pitch, Roll) ────
        if self._register_step == 0:
            if is_center:
                self._hold_frames += 1
                if self._hold_frames >= 10: 
                    self._register_step = 1; self._pose_state = "WAITING_EXTREME"; self._hold_frames = 0
                    return {"status": "pending", "step": "FACEMESH", "instruction": "✅ Wajah Lurus", "progress": "Mulai Liveness..."}
                return {"status": "pending", "step": "FACEMESH", "instruction": "1. Tahan Posisi Lurus", "progress": f"Menahan... ({self._hold_frames}/10)"}
            else: self._hold_frames = 0
            return {"status": "pending", "step": "FACEMESH", "instruction": "1. Tatap Lurus ke Kamera", "progress": "Arahkan wajah ke depan"}

        elif self._register_step == 1:
            if self._pose_state == "WAITING_EXTREME":
                if yaw > thr_y: 
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames: self._pose_state = "WAITING_CENTER"; self._hold_frames = 0
                else: self._hold_frames = 0
                return {"status": "pending", "step": "YAW", "instruction": "2a. Toleh ke KANAN", "progress": f"Target: >{thr_y}° | Saat ini: {yaw:.1f}°"}
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames: self._register_step = 2; self._pose_state = "WAITING_EXTREME"; self._hold_frames = 0
                else: self._hold_frames = 0
                return {"status": "pending", "step": "YAW", "instruction": "Tahan LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        elif self._register_step == 2:
            if self._pose_state == "WAITING_EXTREME":
                if yaw < -thr_y: 
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames: self._pose_state = "WAITING_CENTER"; self._hold_frames = 0
                else: self._hold_frames = 0
                return {"status": "pending", "step": "YAW", "instruction": "2b. Toleh ke KIRI", "progress": f"Target: <{-thr_y}° | Saat ini: {yaw:.1f}°"}
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames: self._register_step = 3; self._pose_state = "WAITING_EXTREME"; self._hold_frames = 0
                else: self._hold_frames = 0
                return {"status": "pending", "step": "YAW", "instruction": "Tahan LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        elif self._register_step == 3:
            if self._pose_state == "WAITING_EXTREME":
                if pitch < -thr_p: 
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames: self._pose_state = "WAITING_CENTER"; self._hold_frames = 0
                else: self._hold_frames = 0
                return {"status": "pending", "step": "PITCH", "instruction": "3a. Dongak ke ATAS", "progress": f"Target: <{-thr_p}° | Saat ini: {pitch:.1f}°"}
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames: self._register_step = 4; self._pose_state = "WAITING_EXTREME"; self._hold_frames = 0
                else: self._hold_frames = 0
                return {"status": "pending", "step": "PITCH", "instruction": "Tahan LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        elif self._register_step == 4:
            if self._pose_state == "WAITING_EXTREME":
                if pitch > thr_p: 
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames: self._pose_state = "WAITING_CENTER"; self._hold_frames = 0
                else: self._hold_frames = 0
                return {"status": "pending", "step": "PITCH", "instruction": "3b. Tunduk ke BAWAH", "progress": f"Target: >{thr_p}° | Saat ini: {pitch:.1f}°"}
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames: self._register_step = 5; self._pose_state = "WAITING_EXTREME"; self._hold_frames = 0
                else: self._hold_frames = 0
                return {"status": "pending", "step": "PITCH", "instruction": "Tahan LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        elif self._register_step == 5:
            if self._pose_state == "WAITING_EXTREME":
                if roll < -thr_r: 
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames: self._pose_state = "WAITING_CENTER"; self._hold_frames = 0
                else: self._hold_frames = 0
                return {"status": "pending", "step": "ROLL", "instruction": "4a. Miring ke KANAN", "progress": f"Target: <{-thr_r}° | Saat ini: {roll:.1f}°"}
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames: self._register_step = 6; self._pose_state = "WAITING_EXTREME"; self._hold_frames = 0
                else: self._hold_frames = 0
                return {"status": "pending", "step": "ROLL", "instruction": "Tahan LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        elif self._register_step == 6:
            if self._pose_state == "WAITING_EXTREME":
                if roll > thr_r: 
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames: self._pose_state = "WAITING_CENTER"; self._hold_frames = 0
                else: self._hold_frames = 0
                return {"status": "pending", "step": "ROLL", "instruction": "4b. Miring ke KIRI", "progress": f"Target: >{thr_r}° | Saat ini: {roll:.1f}°"}
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames: self._register_step = 7; self._pose_state = "WAITING_EXTREME"; self._hold_frames = 0
                else: self._hold_frames = 0
                return {"status": "pending", "step": "ROLL", "instruction": "Tahan LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        # ──── TAHAP 7: BLINK (KALIBRASI ADAPTIF & HITUNG KEDIPAN NATURAL) ────
        elif self._register_step == 7:
            ear_val = 1.0
            if face.landmarks and len(face.landmarks) >= 400:
                p = np.array([[face.landmarks[i].x, face.landmarks[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]])
                ear_val = ((np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2.0*np.linalg.norm(p[0]-p[3])+1e-6) + (np.linalg.norm(p[7]-p[11])+np.linalg.norm(p[8]-p[10]))/(2.0*np.linalg.norm(p[6]-p[9])+1e-6))/2.0
            
            # Smoothing (Rata-rata 5 frame agar tidak ada lompatan noise)
            self._ear_history.append(ear_val)
            if len(self._ear_history) > 5: self._ear_history.pop(0)
            smooth_ear = sum(self._ear_history) / len(self._ear_history)

            total_blinks = getattr(config, 'REGISTER_BLINK_COUNT', 2)

            # State 0: Kalibrasi Mata Terbuka (Mencari baseline pencahayaan aktual)
            if self._blink_state == 0:
                if is_center:
                    self._hold_frames += 1
                    if self._hold_frames >= 15: 
                        self._base_open_ear = smooth_ear  # Kunci baseline cahaya saat ini
                        self._blink_state = 1
                        self._hold_frames = 0
                        self._blink_count = 0
                else: self._hold_frames = 0
                return {"status": "pending", "step": "BLINK", "instruction": "Tatap Kamera (Kalibrasi Mata)", "progress": f"Kalibrasi Cahaya... ({self._hold_frames}/15)"}
            
            # State 1: Menunggu Mata Tertutup
            elif self._blink_state == 1:
                target_close = self._base_open_ear - 0.05 # Target kedipan dinamis
                
                if smooth_ear <= target_close:
                    self._hold_frames += 1
                    if self._hold_frames >= 2: # Tahan sebentar untuk memastikan ini bukan noise getaran
                        self._blink_state = 2
                        self._hold_frames = 0
                else: self._hold_frames = 0
                
                needed = total_blinks - self._blink_count
                return {"status": "pending", "step": "BLINK", "instruction": f"5. Kedipkan Mata ({needed}x)", "progress": f"Tutup Mata (Target: < {target_close:.2f})"}
            
            # State 2: Kembali Buka Mata (Satu siklus kedipan dihitung)
            elif self._blink_state == 2:
                target_open = self._base_open_ear - 0.02
                if smooth_ear >= target_open:
                    self._blink_count += 1
                    if self._blink_count >= total_blinks:
                        self._register_step = 8
                        self._blink_state = 0
                        self._hold_frames = 0
                        return {"status": "pending", "step": "BLINK", "instruction": f"✅ {total_blinks}x Kedipan Terekam", "progress": "Validasi Selesai..."}
                    else:
                        self._blink_state = 1 # Kembali tunggu tutup mata untuk kedipan selanjutnya
                        self._hold_frames = 0
                        
                needed = total_blinks - self._blink_count
                return {"status": "pending", "step": "BLINK", "instruction": "BUKA Mata Kembali", "progress": f"Target EAR: > {target_open:.2f}"}

        # ──── TAHAP 8: SELESAI ────
        elif self._register_step == 8:
            return {"status": "complete", "step": "DONE", "instruction": "Semua Liveness Berhasil!", "progress": "Mengekstrak MobileFaceNet..."}

        return {"status": "pending", "step": "WAIT", "instruction": "Menunggu...", "progress": "WAIT"}
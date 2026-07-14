import numpy as np
import random
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
        self._required_frames = 5     
        self._pose_history = []       # Buffer smoothing pose (rata-rata 3 frame)
        
        # Variabel khusus Kalibrasi Anti-Noise (Blink 2x)
        self._blink_state = 0         
        self._ear_history = []
        self._base_open_ear = 0.0
        self._blink_count = 0

        # Randomize one pose challenge from all options
        thr_y = getattr(config, 'CHALLENGE_YAW', 25.0)
        thr_p = getattr(config, 'CHALLENGE_PITCH', 20.0)
        thr_r = getattr(config, 'CHALLENGE_ROLL', 25.0)

        challenges_pool = [
            {"tag": "yaw_left", "axis": "yaw", "thr": thr_y, "target_dir": "left", "inst": "2. Toleh ke KIRI", "snap_key": "yaw_snapshots", "friendly": "Toleh Kiri"},
            {"tag": "yaw_right", "axis": "yaw", "thr": thr_y, "target_dir": "right", "inst": "2. Toleh ke KANAN", "snap_key": "yaw_snapshots", "friendly": "Toleh Kanan"},
            {"tag": "pitch_up", "axis": "pitch", "thr": thr_p, "target_dir": "up", "inst": "2. Dongak ke ATAS", "snap_key": "pitch_snapshots", "friendly": "Dongak Atas"},
            {"tag": "pitch_down", "axis": "pitch", "thr": thr_p, "target_dir": "down", "inst": "2. Tunduk ke BAWAH", "snap_key": "pitch_snapshots", "friendly": "Tunduk Bawah"},
            {"tag": "roll_left", "axis": "roll", "thr": thr_r, "target_dir": "left", "inst": "2. Miring ke KANAN", "snap_key": "roll_snapshots", "friendly": "Miring Kanan"},
            {"tag": "roll_right", "axis": "roll", "thr": thr_r, "target_dir": "right", "inst": "2. Miring ke KIRI", "snap_key": "roll_snapshots", "friendly": "Miring Kiri"}
        ]
        self.chosen_params = random.choice(challenges_pool)
        self.chosen_challenge = self.chosen_params["tag"]
        self._baseline_pose = None

    def start_register(self):
        self.reset_state()

    def update_register(self, face: FaceResult, detector: FaceMeshDetector) -> dict:
        pose = self.pose_estimator.estimate(face, detector)
        
        if not pose["valid"]:
            self._hold_frames = 0
            return {"status": "pending", "step": "WAIT", "instruction": "Wajah tidak terdeteksi", "progress": "Mencari wajah..."}

        yaw, pitch, roll = pose["yaw"], pose["pitch"], pose["roll"]
        
        # Smoothing pose: rata-rata 3 frame terakhir untuk meredam noise/jitter
        self._pose_history.append({"yaw": yaw, "pitch": pitch, "roll": roll})
        if len(self._pose_history) > 3:
            self._pose_history.pop(0)
        if len(self._pose_history) >= 2:
            yaw = sum(p["yaw"] for p in self._pose_history) / len(self._pose_history)
            pitch = sum(p["pitch"] for p in self._pose_history) / len(self._pose_history)
            roll = sum(p["roll"] for p in self._pose_history) / len(self._pose_history)
            # Update pose dict agar pose[axis] juga pakai nilai smoothed
            pose["yaw"], pose["pitch"], pose["roll"] = yaw, pitch, roll
        
        is_center = abs(yaw) < 15.0 and abs(pitch) < 15.0 and abs(roll) < 15.0

        # ──── TAHAPAN 0 (Wajah Lurus) ────
        if self._register_step == 0:
            if is_center:
                self._hold_frames += 1
                if self._hold_frames >= 10: 
                    self._baseline_pose = {"yaw": yaw, "pitch": pitch, "roll": roll}
                    self._register_step = 1; self._pose_state = "WAITING_EXTREME"; self._hold_frames = 0
                    return {"status": "pending", "step": "FACEMESH", "instruction": "✅ Wajah Lurus", "progress": "Mulai Liveness..."}
                return {"status": "pending", "step": "FACEMESH", "instruction": "1. Tahan Posisi Lurus", "progress": f"Menahan... ({self._hold_frames}/10)"}
            else: self._hold_frames = 0
            return {"status": "pending", "step": "FACEMESH", "instruction": "1. Tatap Lurus ke Kamera", "progress": "Arahkan wajah ke depan"}

        # ──── TAHAPAN 1: Pose Acak ────
        elif self._register_step == 1:
            axis = self.chosen_params["axis"]
            thr = self.chosen_params["thr"]
            
            baseline_val = self._baseline_pose.get(axis, 0.0) if self._baseline_pose else 0.0
            val = pose[axis] - baseline_val
            
            if self._pose_state == "WAITING_EXTREME":
                is_extreme = False
                if self.chosen_params["target_dir"] == "left" or self.chosen_params["target_dir"] == "up":
                    if val < -thr: is_extreme = True
                else:
                    if val > thr: is_extreme = True
                    
                if is_extreme:
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames:
                        self._pose_state = "WAITING_CENTER"
                        self._hold_frames = 0
                else:
                    self._hold_frames = max(0, self._hold_frames - 2)  # Gradual decay, bukan hard reset
                
                prog_sign = "<" if (self.chosen_params["target_dir"] in ("left", "up")) else ">"
                prog_val = -thr if (self.chosen_params["target_dir"] in ("left", "up")) else thr
                return {
                    "status": "pending", 
                    "step": "POSE", 
                    "instruction": self.chosen_params["inst"], 
                    "progress": f"Target: {prog_sign}{prog_val:.1f}° | Saat ini: {val:.1f}°"
                }
                
            elif self._pose_state == "WAITING_CENTER":
                if is_center:
                    self._hold_frames += 1
                    if self._hold_frames >= self._required_frames:
                        self._register_step = 2
                        self._pose_state = "WAITING_EXTREME"
                        self._hold_frames = 0
                else:
                    self._hold_frames = max(0, self._hold_frames - 2)  # Gradual decay, bukan hard reset
                return {"status": "pending", "step": "POSE", "instruction": "Tahan LURUS ke Depan", "progress": "Tunggu wajah lurus..."}

        # ──── TAHAPAN 2: BLINK (KALIBRASI ADAPTIF & HITUNG KEDIPAN NATURAL) ────
        elif self._register_step == 2:
            ear_val = 1.0
            if face.landmarks and len(face.landmarks) >= 400:
                p = np.array([[face.landmarks[i].x, face.landmarks[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]])
                ear_val = ((np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2.0*np.linalg.norm(p[0]-p[3])+1e-6) + (np.linalg.norm(p[7]-p[11])+np.linalg.norm(p[8]-p[10]))/(2.0*np.linalg.norm(p[6]-p[9])+1e-6))/2.0
            
            # Smoothing (Rata-rata 5 frame agar tidak ada lompatan noise)
            self._ear_history.append(ear_val)
            if len(self._ear_history) > 5: self._ear_history.pop(0)
            smooth_ear = sum(self._ear_history) / len(self._ear_history)

            total_blinks = getattr(config, 'REGISTER_BLINK_COUNT', 2)

            # State 0: Baseline Terbuka Instan (Menghapus penundaan kalibrasi 15-frame)
            if self._blink_state == 0:
                self._base_open_ear = smooth_ear if smooth_ear > 0.0 else 0.28
                self._blink_state = 1
                self._hold_frames = 0
                self._blink_count = 0
                target_close = self._base_open_ear - 0.05
                needed = total_blinks - self._blink_count
                return {"status": "pending", "step": "BLINK", "instruction": f"3. Kedipkan Mata ({needed}x)", "progress": f"Tutup Mata (Target: < {target_close:.2f})"}
            
            # State 1: Menunggu Mata Tertutup
            elif self._blink_state == 1:
                target_close = self._base_open_ear - 0.05 
                
                if smooth_ear <= target_close:
                    self._hold_frames += 1
                    if self._hold_frames >= 4: 
                        self._blink_state = 2
                        self._hold_frames = 0
                else: self._hold_frames = 0
                
                needed = total_blinks - self._blink_count
                return {"status": "pending", "step": "BLINK", "instruction": f"3. Kedipkan Mata ({needed}x)", "progress": f"Tutup Mata (Target: < {target_close:.2f})"}
            
            # State 2: Kembali Buka Mata (Satu siklus kedipan dihitung)
            elif self._blink_state == 2:
                target_open = self._base_open_ear - 0.02
                if smooth_ear >= target_open:
                    self._blink_count += 1
                    if self._blink_count >= total_blinks:
                        self._register_step = 3
                        self._blink_state = 0
                        self._hold_frames = 0
                        return {"status": "pending", "step": "BLINK", "instruction": f"✅ {total_blinks}x Kedipan Terekam", "progress": "Validasi Selesai..."}
                    else:
                        self._blink_state = 1 # Kembali tunggu tutup mata untuk kedipan selanjutnya
                        self._hold_frames = 0
                        
                needed = total_blinks - self._blink_count
                return {"status": "pending", "step": "BLINK", "instruction": "BUKA Mata Kembali", "progress": f"Target EAR: > {target_open:.2f}"}

        # ──── TAHAP 3: SELESAI ────
        elif self._register_step == 3:
            return {"status": "complete", "step": "DONE", "instruction": "Semua Liveness Berhasil!", "progress": "Mengekstrak MobileFaceNet..."}

        return {"status": "pending", "step": "WAIT", "instruction": "Menunggu...", "progress": "WAIT"}
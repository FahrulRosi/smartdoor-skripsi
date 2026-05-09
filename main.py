import cv2, random, json
import numpy as np
from datetime import datetime
from enum import Enum

import config
from camera.camera_stream       import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from recognition.mobilefacenet  import MobileFaceNet
from recognition.face_matcher   import FaceMatcher
from door.door_lock             import DoorLock
from liveness.head_pose         import HeadPoseEstimator
from liveness.blink             import BlinkDetector
from liveness.anti_spoofing     import SilentAntiSpoofing  
from database.face_db           import FaceDatabase

GPIO_AVAILABLE = True
try: import RPi.GPIO as GPIO
except ImportError: GPIO_AVAILABLE = False

class ValidationState(Enum):
    IDLE = 0; RECOGNIZING = 1; CHALLENGE = 2; UNMATCHED = 3; UNLOCKED = 4

class UIManager:
    @staticmethod
    def log(msg, level="INFO"): 
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {msg}")

    @staticmethod
    def draw_status(frame, bbox, status, color):
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.rectangle(frame, (x, y-35), (x + cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] + 15, y-5), color, -1)
        cv2.putText(frame, status, (x+8, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    @staticmethod
    def draw_dashboard(frame, is_locked):
        cv2.putText(frame, f"PINTU: {'TERKUNCI' if is_locked else 'TERBUKA'}", (10, config.FRAME_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, config.COLOR_RED if is_locked else config.COLOR_GREEN, 2)

class SmartDoorApp:
    CHALLENGES = {
        "BLINK": "Tantangan 2: Kedipkan Mata", "KANAN": "Tantangan 1: Toleh KANAN", "KIRI": "Tantangan 1: Toleh KIRI",
        "ATAS": "Tantangan 1: Dongak ATAS", "BAWAH": "Tantangan 1: Tunduk BAWAH", "MIRING_KANAN": "Tantangan 1: Miring KANAN", "MIRING_KIRI": "Tantangan 1: Miring KIRI"
    }

    def __init__(self):
        UIManager.log("Sistem Smart Door Lock diaktifkan", "SYSTEM")
        self.db = FaceDatabase()
        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        
        self.detector = FaceMeshDetector(
            min_detection_confidence=getattr(config, 'MIN_DETECTION_CONFIDENCE', 0.5), 
            min_tracking_confidence=getattr(config, 'MIN_TRACKING_CONFIDENCE', 0.5)
        )
        self.model = MobileFaceNet()
        self.pose_estimator = HeadPoseEstimator()
        self.anti_spoof = SilentAntiSpoofing()
        
        self.door = DoorLock(pin=getattr(config, 'LOCK_GPIO_PIN', 18), unlock_duration=getattr(config, 'UNLOCK_DURATION', 5))
        self.matcher = FaceMatcher(threshold=config.MATCH_THRESHOLD)
        
        self._reset_state()
        self._load_memory()

    def _load_memory(self):
        try:
            raw = self.db.load_all_faces()
            faces, profiles = {}, {}
            if raw:
                for k, v in raw.items():
                    if isinstance(v, dict):
                        emb = v.get('embedding', v.get('mobilefacenet_embedding'))
                        if emb is not None: faces[k] = np.array(emb, dtype=np.float32)
                        profiles[k] = v.get('liveness_config', v)
                    elif isinstance(v, (list, np.ndarray)):
                        faces[k] = np.array(v, dtype=np.float32)
            
            if hasattr(self.matcher, 'load_faces'): self.matcher.load_faces(faces)
            else: self.matcher.known_faces = faces
            self.user_profiles = profiles
            UIManager.log(f"Memori dimuat: {len(faces)} wajah siap dikenali.", "SUCCESS")
        except Exception as e: 
            UIManager.log(f"Gagal memuat memori: {e}", "WARNING")

    def _reset_state(self):
        self.state, self.last_name, self.match_score = ValidationState.IDLE, "", 0.0
        self.door.lock()
        self.challenge_sequence, self.current_step_idx, self.pose_hold_frames, self.fake_frames = [], 0, 0, 0
        self.blink_checker, self.reg_headpose = None, [0.0, 0.0, 0.0]
        self.print_counter = 0

    def _check_action_passed(self, action, face):
        """Mengecek aksi dan mengembalikan status lulus, nilai saat ini, dan nilai target untuk log terminal"""
        if action == "BLINK": 
            passed = self.blink_checker.update(face, self.detector).get("complete", False) if self.blink_checker else False
            ear = 0.0
            if face.landmarks and len(face.landmarks) >= 400:
                pts = np.array([[face.landmarks[i].x, face.landmarks[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]])
                ear = ((np.linalg.norm(pts[1]-pts[5]) + np.linalg.norm(pts[2]-pts[4])) / (2.0 * np.linalg.norm(pts[0]-pts[3]) + 1e-6) + 
                       (np.linalg.norm(pts[7]-pts[11]) + np.linalg.norm(pts[8]-pts[10])) / (2.0 * np.linalg.norm(pts[6]-pts[9]) + 1e-6)) / 2.0
            return passed, ear, getattr(config, 'BLINK_EAR_THRESHOLD', 0.21)
        
        p, ref = self.pose_estimator.estimate(face, self.detector), self.reg_headpose
        dy, dp, dr = p.get("yaw",0) - ref[0], p.get("pitch",0) - ref[1], p.get("roll",0) - ref[2]
        
        cfg = {
            "KANAN": (dy, getattr(config, 'CHALLENGE_YAW', 20), dy > getattr(config, 'CHALLENGE_YAW', 20)),
            "KIRI": (-dy, getattr(config, 'CHALLENGE_YAW', 20), dy < -getattr(config, 'CHALLENGE_YAW', 20)),
            "ATAS": (-dp, getattr(config, 'CHALLENGE_PITCH', 15), dp < -getattr(config, 'CHALLENGE_PITCH', 15)),
            "BAWAH": (dp, getattr(config, 'CHALLENGE_PITCH', 15), dp > getattr(config, 'CHALLENGE_PITCH', 15)),
            "MIRING_KANAN": (dr, getattr(config, 'CHALLENGE_ROLL', 15), dr > getattr(config, 'CHALLENGE_ROLL', 15)),
            "MIRING_KIRI": (-dr, getattr(config, 'CHALLENGE_ROLL', 15), dr < -getattr(config, 'CHALLENGE_ROLL', 15)),
        }.get(action, (0, 1, False))
        
        val, target, passed = cfg
        self.pose_hold_frames = (self.pose_hold_frames + 1) if passed else 0
        return self.pose_hold_frames >= 5, val, target

    def _process_face(self, frame, display, face):
        # 1. Cek Anti Spoofing
        spoof = self.anti_spoof.is_real(frame, face.bbox)
        if not spoof.get("real", True):
            self.fake_frames += 1
            UIManager.draw_status(display, face.bbox, "TERDETEKSI SPOOFING", config.COLOR_RED)
            if self.fake_frames >= 7: self._reset_state()
            return

        self.fake_frames = 0
        
        if self.state == ValidationState.IDLE: 
            self.state = ValidationState.RECOGNIZING
        
        # 2. Pengenalan Wajah
        if self.state == ValidationState.RECOGNIZING:
            emb = self.model.get_embedding(self.model.crop_face(frame, face.bbox))
            match = self.matcher.match(emb)
            
            if match.get("matched", False):
                self.last_name = match["name"]
                self.match_score = match.get("score", 0.0) # Menyimpan Skor Keberhasilan Pengenalan Wajah (Match Score)
                
                cfg = self.user_profiles.get(self.last_name, {})
                self.reg_headpose = cfg.get("headpose_vector", [0.0, 0.0, 0.0])
                ear = cfg.get("blink_closed", getattr(config, 'BLINK_EAR_THRESHOLD', 0.2))
                ear = ear.get("avg_ear", 0.2) if isinstance(ear, dict) else (0.2 if float(ear) == 0.0 else float(ear))
                
                self.blink_checker = BlinkDetector(ear_threshold=ear + 0.01, target_blinks=1)
                self.challenge_sequence = [random.choice([k for k in self.CHALLENGES.keys() if k != "BLINK"]), "BLINK"]
                self.state, self.current_step_idx = ValidationState.CHALLENGE, 0
                
                UIManager.log(f"Wajah dikenali: {self.last_name}. Memulai Tantangan Liveness...", "SUCCESS")
            else: 
                UIManager.draw_status(display, face.bbox, "TIDAK DIKENAL", config.COLOR_RED)

        # 3. Tantangan Liveness Aktif
        elif self.state == ValidationState.CHALLENGE:
            action = self.challenge_sequence[self.current_step_idx]
            
            # Tampilan UI seperti asli (Teks kuning di pojok kiri atas)
            UIManager.draw_status(display, face.bbox, f"User: {self.last_name}", config.COLOR_CYAN)
            cv2.putText(display, f"Tahap {self.current_step_idx+1}/{len(self.challenge_sequence)}: {self.CHALLENGES[action]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_YELLOW, 2)
            
            passed, current_val, target_val = self._check_action_passed(action, face)
            
            # Terminal Log: Menampilkan nilai Real-time (Keberhasilan Challenge)
            self.print_counter += 1
            if self.print_counter % 3 == 0:
                if action == "BLINK":
                    print(f"\r[{self.CHALLENGES[action]}] Nilai EAR Mata: {current_val:.2f} / Target: < {target_val:.2f}          ", end="", flush=True)
                else:
                    print(f"\r[{self.CHALLENGES[action]}] Sudut Kepala: {current_val:.1f}° / Target: {target_val}°          ", end="", flush=True)

            # Jika berhasil melewati challenge saat ini
            if passed:
                print() # Break baris agar tertulis permanen
                UIManager.log(f"✅ {self.CHALLENGES[action]} SELESAI -> Nilai Tercapai: {current_val:.2f}", "SUCCESS")
                
                self.current_step_idx += 1
                
                # Jika SEMUA tantangan selesai (Buka Pintu)
                if self.current_step_idx >= len(self.challenge_sequence):
                    self.state = ValidationState.UNLOCKED
                    self.door.unlock()
                    
                    # LOG TERMINAL: Menampilkan Keberhasilan Buka Pintu (Persentase Kecocokan Wajah)
                    persentase_wajah = self.match_score * 100
                    UIManager.log(f"🔓 AKSES DIBERIKAN: Pintu terbuka untuk '{self.last_name}'", "SUCCESS")
                    UIManager.log(f"📊 Persentase Keberhasilan (Akurasi Wajah): {persentase_wajah:.1f}%", "SUCCESS")
                    
                    if hasattr(self.db, 'push_access_log_async'): 
                        self.db.push_access_log_async(self.last_name, "UNLOCKED")

        elif self.state == ValidationState.UNLOCKED: 
            UIManager.draw_status(display, face.bbox, f"SELAMAT DATANG, {self.last_name}", config.COLOR_GREEN)

    def run(self):
        try:
            while True:
                ret, frame = self.cam.read()
                if not ret: continue
                display, faces = frame.copy(), self.detector.detect(frame)

                if not faces:
                    self._reset_state()
                    cv2.putText(display, "Menunggu Wajah...", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_YELLOW, 2)
                else: 
                    self._process_face(frame, display, faces[0])

                UIManager.draw_dashboard(display, self.door.locked)
                cv2.imshow("Smart Door Lock", display)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
        finally:
            if GPIO_AVAILABLE: GPIO.cleanup()
            self.cam.stop(); self.door.cleanup(); cv2.destroyAllWindows()

if __name__ == "__main__": 
    SmartDoorApp().run()
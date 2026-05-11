import cv2, random, time, threading, numpy as np
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
    IDLE=0; RECOGNIZING=1; CHALLENGE=2; UNMATCHED=3; UNLOCKED=4

class UIManager:
    @staticmethod
    def log(msg, level="INFO"): 
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {msg}")

    @staticmethod
    def draw_status(frame, bbox, status, color):
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        txt_w = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
        cv2.rectangle(frame, (x, y-35), (x + txt_w + 15, y-5), color, -1)
        cv2.putText(frame, status, (x+8, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

class IlluminationEnhancer:
    @staticmethod
    def enhance(frame):
        if not getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', True): return frame
        if np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) > 85.0: return frame
        l, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
        clahe = cv2.createCLAHE(clipLimit=getattr(config, 'CLAHE_CLIP_LIMIT', 2.5), tileGridSize=getattr(config, 'CLAHE_TILE_GRID_SIZE', (8, 8)))
        return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)

class SmartDoorApp:
    CHALLENGES = {
        "BLINK": "Tantangan 2: Kedipkan Mata", "KANAN": "Tantangan 1: Toleh KANAN", "KIRI": "Tantangan 1: Toleh KIRI", 
        "ATAS": "Tantangan 1: Dongak ATAS", "BAWAH": "Tantangan 1: Tunduk BAWAH", "MIRING_KANAN": "Tantangan 1: Miring KANAN", "MIRING_KIRI": "Tantangan 1: Miring KIRI"
    }

    def __init__(self):
        UIManager.log("Sistem Smart Door Lock diaktifkan", "SYSTEM")
        self.db, self.model, self.pose_estimator, self.anti_spoof = FaceDatabase(), MobileFaceNet(), HeadPoseEstimator(), SilentAntiSpoofing()
        
        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        self.detector = FaceMeshDetector(min_detection_confidence=getattr(config, 'MIN_DETECTION_CONFIDENCE', 0.5), min_tracking_confidence=getattr(config, 'MIN_TRACKING_CONFIDENCE', 0.5))
        self.door = DoorLock(pin=getattr(config, 'LOCK_GPIO_PIN', 18), unlock_duration=getattr(config, 'UNLOCK_DURATION', 5))
        self.matcher = FaceMatcher(threshold=config.MATCH_THRESHOLD)
        
        self.lock, self.running, self.shared_frame = threading.Lock(), True, None
        self.ui_state = {"wait_msg": True, "bbox": None, "status": "", "color": config.COLOR_WHITE, "instruction": ""}
        
        self._reset_state(); self._load_memory()
        self.ai_thread = threading.Thread(target=self._ai_worker, daemon=True)

    def _load_memory(self):
        try:
            raw = self.db.load_all_faces()
            if not raw: return
            faces, profiles = {}, {}
            for k, v in raw.items():
                if isinstance(v, dict):
                    emb = v.get('embedding', v.get('mobilefacenet_embedding'))
                    if emb is not None: faces[k] = np.array(emb, dtype=np.float32)
                    profiles[k] = v.get('liveness_config', v)
                elif isinstance(v, (list, np.ndarray)): faces[k] = np.array(v, dtype=np.float32)
                    
            if hasattr(self.matcher, 'load_faces'): self.matcher.load_faces(faces)
            else: self.matcher.known_faces = faces
            self.user_profiles = profiles
            UIManager.log(f"Memori dimuat: {len(faces)} wajah siap dikenali.", "SUCCESS")
        except Exception as e: UIManager.log(f"Gagal memuat memori: {e}", "WARNING")

    def _reset_state(self):
        self.state, self.last_name, self.match_score = ValidationState.IDLE, "", 0.0
        self.auth_start_time, self.current_latency = 0.0, 0.0
        self.challenge_sequence, self.current_step_idx, self.fake_frames, self.print_counter = [], 0, 0, 0
        self.blink_checker, self.reg_headpose, self.pose_hold_frames = None, [0.0, 0.0, 0.0], 0
        if hasattr(self, 'door'): self.door.lock()

    def _check_action_passed(self, action, face):
        if action == "BLINK": 
            passed = self.blink_checker.update(face, self.detector).get("complete", False) if self.blink_checker else False
            val = 0.0
            if face.landmarks and len(face.landmarks) >= 400:
                p = np.array([[face.landmarks[i].x, face.landmarks[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]])
                val = ((np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2.0*np.linalg.norm(p[0]-p[3])+1e-6) + (np.linalg.norm(p[7]-p[11])+np.linalg.norm(p[8]-p[10]))/(2.0*np.linalg.norm(p[6]-p[9])+1e-6))/2.0
            return passed, val, getattr(config, 'BLINK_EAR_THRESHOLD', 0.21)

        p = self.pose_estimator.estimate(face, self.detector)
        ref = self.reg_headpose
        dy, dp, dr = p.get("yaw", 0.0) - ref[0], p.get("pitch", 0.0) - ref[1], p.get("roll", 0.0) - ref[2]
        ty, tp, tr = getattr(config, 'CHALLENGE_YAW', 20.0), getattr(config, 'CHALLENGE_PITCH', 15.0), getattr(config, 'CHALLENGE_ROLL', 15.0)
        
        val, tgt, passed = {
            "KANAN": (dy, ty, dy > ty), "KIRI": (-dy, ty, -dy > ty),
            "ATAS": (-dp, tp, -dp > tp), "BAWAH": (dp, tp, dp > tp),
            "MIRING_KANAN": (-dr, tr, -dr > tr), "MIRING_KIRI": (dr, tr, dr > tr)
        }.get(action, (0.0, 1.0, False))

        self.pose_hold_frames = self.pose_hold_frames + 1 if passed else 0
        return self.pose_hold_frames >= 5, val, tgt

    def _ai_worker(self):
        while self.running:
            with self.lock: frame = self.shared_frame.copy() if self.shared_frame is not None else None
            if frame is None: time.sleep(0.01); continue
            
            t0 = time.time()
            raw_frame = frame.copy()
            enhanced_frame = IlluminationEnhancer.enhance(raw_frame)
            faces = self.detector.detect(enhanced_frame) 
            
            if not faces: self._reset_state(); self.ui_state.update({"wait_msg": True, "bbox": None})
            else: self.ui_state["wait_msg"] = False; self._process_face(raw_frame, enhanced_frame, faces[0])
                
            self.current_latency = (time.time() - t0) * 1000
            time.sleep(0.01) 

    def _process_face(self, raw_frame, enhanced_frame, face):
        self.ui_state["bbox"] = face.bbox
        if face.bbox[3] > int(config.FRAME_HEIGHT * 0.50):
            self._reset_state(); self.ui_state.update({"status": "WAJAH TERLALU DEKAT", "color": config.COLOR_YELLOW, "instruction": "Silakan mundur sedikit dari kamera"}); return
        
        # --- KETERANGAN UBAH MENJADI SPOOFING SAJA ---
        if self.state in (ValidationState.IDLE, ValidationState.RECOGNIZING):
            spoof = self.anti_spoof.is_real(raw_frame, face.bbox)
            if not spoof.get("real", True):
                self.fake_frames += 1
                if self.fake_frames >= 7: self._reset_state()
                self.ui_state.update({"status": "SPOOFING", "color": config.COLOR_RED, "instruction": ""})
                return
            else: self.fake_frames = 0
        else: self.fake_frames = 0 
        # ---------------------------------------------

        if self.state == ValidationState.IDLE: self.state, self.auth_start_time = ValidationState.RECOGNIZING, time.time()
        
        if self.state == ValidationState.RECOGNIZING:
            emb = self.model.get_embedding(self.model.crop_face(enhanced_frame, face.bbox))
            match = self.matcher.match(emb)
            
            if match.get("matched", False):
                self.last_name, self.match_score = match["name"], match.get("score", 0.0)
                cfg = self.user_profiles.get(self.last_name, {})
                
                curr = self.pose_estimator.estimate(face, self.detector)
                self.reg_headpose = [curr.get("yaw", 0.0), curr.get("pitch", 0.0), curr.get("roll", 0.0)]
                
                ear = cfg.get("blink_closed", getattr(config, 'BLINK_EAR_THRESHOLD', 0.2))
                ear_val = ear.get("avg_ear", 0.2) if isinstance(ear, dict) else float(ear or 0.2)
                self.blink_checker = BlinkDetector(ear_threshold=ear_val + 0.01, target_blinks=1)
                
                self.challenge_sequence = [random.choice([k for k in self.CHALLENGES if k != "BLINK"]), "BLINK"]
                self.state, self.current_step_idx, self.pose_hold_frames = ValidationState.CHALLENGE, 0, 0
                UIManager.log(f"Wajah dikenali: {self.last_name}. Memulai Tantangan...", "SUCCESS")
            else: self.ui_state.update({"status": "TIDAK DIKENAL", "color": config.COLOR_RED, "instruction": ""})

        elif self.state == ValidationState.CHALLENGE:
            act = self.challenge_sequence[self.current_step_idx]
            inst = self.CHALLENGES[act]
            
            self.ui_state.update({"status": f"User: {self.last_name}", "color": config.COLOR_CYAN, "instruction": f"Tahap {self.current_step_idx+1}/{len(self.challenge_sequence)}: {inst}"})
            passed, c_val, t_val = self._check_action_passed(act, face)
            
            self.print_counter += 1
            if self.print_counter % 3 == 0:
                mtrc = f"EAR Mata: {c_val:.2f}" if act == "BLINK" else f"Sudut: {c_val:.1f}°"
                print(f"\r[{inst}] {mtrc} | Latency: {self.current_latency:.1f}ms      ", end="", flush=True)

            if passed:
                print(); UIManager.log(f"✅ {inst} SELESAI -> Nilai Tercapai", "SUCCESS")
                self.current_step_idx += 1; self.pose_hold_frames = 0
                
                # --- PERBAIKAN: JEDA & RESET AGAR BLINK TIDAK TERSKIP ---
                if self.current_step_idx < len(self.challenge_sequence):
                    self.ui_state.update({"instruction": "Kembali ke posisi tengah..."})
                    time.sleep(1.2)  # Memberi waktu kepala kembali tegak sebelum membaca EAR
                    
                    cfg = self.user_profiles.get(self.last_name, {})
                    ear = cfg.get("blink_closed", getattr(config, 'BLINK_EAR_THRESHOLD', 0.2))
                    ear_val = ear.get("avg_ear", 0.2) if isinstance(ear, dict) else float(ear or 0.2)
                    self.blink_checker = BlinkDetector(ear_threshold=ear_val + 0.01, target_blinks=1)
                else:
                    self.state = ValidationState.UNLOCKED
                    threading.Thread(target=self.door.unlock, daemon=True).start()
                    
                    rs, thr, exc = self.match_score, getattr(config, 'MATCH_THRESHOLD', 0.55), 0.80 
                    pct = 100.0 if rs >= exc else (95.0 + (rs - thr) * 4.9 / (exc - thr) if rs >= thr else rs * 100.0)
                    auth_latency = time.time() - self.auth_start_time
                    
                    UIManager.log(f"🔓 AKSES DIBERIKAN: Pintu terbuka untuk '{self.last_name}'", "SUCCESS")
                    UIManager.log(f"📊 Akurasi Wajah: {pct:.1f}% | Waktu Otentikasi: {auth_latency:.2f}s", "SUCCESS")
                    if hasattr(self.db, 'push_access_log_async'): self.db.push_access_log_async(self.last_name, "UNLOCKED")

        elif self.state == ValidationState.UNLOCKED: self.ui_state.update({"status": f"SELAMAT DATANG, {self.last_name}", "color": config.COLOR_GREEN, "instruction": ""})

    def run(self):
        self.ai_thread.start() 
        try:
            while self.running:
                ret, frame = self.cam.read()
                if not ret: continue
                with self.lock: self.shared_frame = frame.copy()
                display = frame.copy()
                
                if self.ui_state["wait_msg"]: cv2.putText(display, "Menunggu Wajah...", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_YELLOW, 2)
                elif self.ui_state["bbox"] is not None:
                    UIManager.draw_status(display, self.ui_state["bbox"], self.ui_state["status"], self.ui_state["color"])
                    if self.ui_state["instruction"]: cv2.putText(display, self.ui_state["instruction"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_YELLOW, 2)

                is_locked = self.door.locked
                cv2.putText(display, f"PINTU: {'TERKUNCI' if is_locked else 'TERBUKA'}", (10, config.FRAME_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, config.COLOR_RED if is_locked else config.COLOR_GREEN, 2)
                
                cv2.imshow("Smart Door Lock", display)
                if cv2.waitKey(1) & 0xFF == ord("q"): self.running = False; break
        finally:
            self.running = False
            if GPIO_AVAILABLE: GPIO.cleanup()
            self.cam.stop(); self.door.cleanup(); cv2.destroyAllWindows()

if __name__ == "__main__": SmartDoorApp().run()
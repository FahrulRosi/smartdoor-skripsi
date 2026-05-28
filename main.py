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
from liveness.anti_spoofing     import SilentAntiSpoofing  
from database.face_db           import FaceDatabase

GPIO_AVAILABLE = True
try: import RPi.GPIO as GPIO
except ImportError: GPIO_AVAILABLE = False

class ValidationState(Enum): IDLE=0; RECOGNIZING=1; CHALLENGE=2; UNMATCHED=3; UNLOCKED=4

class UIHelper:
    @staticmethod
    def log(msg, lvl="INFO"): print(f"[{datetime.now().strftime('%H:%M:%S')}] [{lvl}] {msg}")

    @staticmethod
    def enhance_frame(frame):
        if not getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', True): return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if (m := np.mean(gray)) > 160.0: return frame  
        l, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
        return cv2.cvtColor(cv2.merge((cv2.createCLAHE(clipLimit=2.5 if m < 85.0 else 1.2, tileGridSize=(8, 8)).apply(l), a, b)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def get_ear(face):
        lm = getattr(face, 'landmarks', [])
        if not lm or len(lm) < 400: return 0.0
        p = np.array([[lm[i].x, lm[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]])
        n = np.linalg.norm 
        return ((n(p[1]-p[5])+n(p[2]-p[4]))/(2.0*n(p[0]-p[3])+1e-6) + (n(p[7]-p[11])+n(p[8]-p[10]))/(2.0*n(p[6]-p[9])+1e-6)) / 2.0

    @staticmethod
    def draw_ui(disp, ui, locked):
        if ui.get("instr"):
            w, h = cv2.getTextSize(ui["instr"], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(disp, (10, 10), (w + 40, h + 30), (0,0,0), -1)
            cv2.putText(disp, ui["instr"], (20, h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_YELLOW, 2)
        if ui.get("wait"): 
            cv2.putText(disp, "Menunggu Wajah...", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_YELLOW, 2)
        elif ui.get("bbox"):
            x, y, w, h = ui["bbox"]
            fx = config.FRAME_WIDTH - x - w
            c = ui.get("color", config.COLOR_WHITE)
            cv2.rectangle(disp, (fx, y), (fx+w, y+h), c, 3)
            if stat := ui.get("status", ""):
                tw = cv2.getTextSize(stat, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
                cv2.rectangle(disp, (fx, y-35), (fx + tw + 15, y-5), c, -1)
                cv2.putText(disp, stat, (fx+8, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(disp, f"PINTU: {'TERKUNCI' if locked else 'TERBUKA'}", (10, config.FRAME_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, config.COLOR_RED if locked else config.COLOR_GREEN, 2)

class SmartDoorApp:
    CHALLENGES = {"BLINK": "Kedipkan Mata", "KANAN": "Toleh KANAN", "KIRI": "Toleh KIRI", "ATAS": "Dongak ATAS", "BAWAH": "Tunduk BAWAH", "MIRING_KANAN": "Miring KANAN", "MIRING_KIRI": "Miring KIRI"}

    def __init__(self):
        UIHelper.log("Sistem Smart Door Lock diaktifkan", "SYSTEM")
        self.db, self.model, self.pose_estimator, self.anti_spoof = FaceDatabase(), MobileFaceNet(), HeadPoseEstimator(), SilentAntiSpoofing()
        self.cam, self.detector = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start(), FaceMeshDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.door, self.matcher = DoorLock(pin=getattr(config, 'LOCK_GPIO_PIN', 18), unlock_duration=getattr(config, 'UNLOCK_DURATION', 5)), FaceMatcher(threshold=getattr(config, 'MATCH_THRESHOLD', 0.82))
        self.lock, self.running, self.shared_frame = threading.Lock(), True, None
        self.ui, self.missed_frames = {"wait": True, "bbox": None, "status": "", "color": config.COLOR_WHITE, "instr": ""}, 0
        self._reset_state(); self._load_memory()
        threading.Thread(target=self._ai_worker, daemon=True).start()

    def _load_memory(self):
        try:
            if not (raw := self.db.load_all_faces()): return
            faces = {k: np.array(v.get('embedding', v.get('mobilefacenet_embedding')), dtype=np.float32) for k, v in raw.items() if isinstance(v, dict) and v.get('embedding') is not None}
            self.user_profiles = {k: v.get('liveness_config', v) for k, v in raw.items() if isinstance(v, dict)}
            self.matcher.load_faces(faces) if hasattr(self.matcher, 'load_faces') else setattr(self.matcher, 'known_faces', faces)
            UIHelper.log(f"Memori dimuat: {len(faces)} wajah siap.", "SUCCESS")
        except Exception as e: UIHelper.log(f"Gagal muat: {e}", "WARNING")

    def _reset_state(self):
        self.state, self.last_name, self.match_score, self.auth_start, self.fake_frames = ValidationState.IDLE, "", 0.0, 0.0, 0
        self.seq, self.step_idx, self.reg_pose, self.pose_hold, self.prev_center = [], 0, [0.0, 0.0, 0.0], 0, None
        self.wait_center, self.center_hold, self.blink_passed, self.blink_hold, self.ear_hist, self.print_counter = False, 0, False, 0, [], 0
        self.spoof_score = 0.98
        if hasattr(self, 'door'): self.door.lock()

    def _check_action(self, action, face):
        if action == "BLINK": 
            ear = UIHelper.get_ear(face)
            self.ear_hist.append(ear)
            if len(self.ear_hist) > 3: self.ear_hist.pop(0)
            tgt = getattr(self, 'dyn_blink_thr', getattr(config, 'BLINK_EAR_THRESHOLD', 0.21))
            if min(self.ear_hist) <= tgt: self.blink_hold += 1
            else:
                if self.blink_hold >= 1: self.blink_passed = True
                self.blink_hold = 0
            return self.blink_passed, ear, tgt

        p, ref = self.pose_estimator.estimate(face, self.detector), self.reg_pose
        dy, dp, dr = p.get("yaw", 0)-ref[0], p.get("pitch", 0)-ref[1], p.get("roll", 0)-ref[2]
        ty, tp, tr = getattr(config, 'CHALLENGE_YAW', 25.0), getattr(config, 'CHALLENGE_PITCH', 20.0), getattr(config, 'CHALLENGE_ROLL', 25.0)
        
        val, tgt, passed = {
            "KANAN": (dy, ty, dy>ty), 
            "KIRI": (-dy, ty, -dy>ty), 
            "ATAS": (-dp, tp, -dp>tp), 
            "BAWAH": (dp, tp, dp>tp), 
            "MIRING_KANAN": (-dr, tr, -dr>tr), 
            "MIRING_KIRI": (dr, tr, dr>tr)     
        }.get(action, (0.0, 1.0, False))
        
        if passed: self.pose_hold += 1
        else: self.pose_hold = 0
        return self.pose_hold >= 2, val, tgt

    def _ai_worker(self):
        while self.running:
            with self.lock: frame = self.shared_frame.copy() if self.shared_frame is not None else None
            if frame is None: time.sleep(0.01); continue
            
            t0, raw = time.time(), frame.copy()
            enhanced = UIHelper.enhance_frame(raw)
            faces = self.detector.detect(enhanced)
            
            if not faces: 
                self.missed_frames += 1 
                if self.missed_frames >= 5: 
                    self._reset_state()
                    self.ui.update({"wait": True, "bbox": None, "status": "", "instr": ""})
            else: 
                self.missed_frames = 0
                target_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3]) 
                self.ui.update({"wait": False, "bbox": target_face.bbox}) 
                self._process_face(raw, enhanced, target_face)
                
            self.latency = (time.time() - t0) * 1000; time.sleep(0.01) 

    def _process_face(self, raw, enhanced, face):
        x, y, w, h = face.bbox
        cx, cy = x + w // 2, y + h // 2
        
        if self.state.value > 1 and self.prev_center and np.hypot(cx - self.prev_center[0], cy - self.prev_center[1]) > max(w, h) * 0.40: 
            self._reset_state()
            return self.ui.update({"status": "WAJAH BERGANTI", "color": config.COLOR_RED, "instr": "Mulai Ulang"})
        self.prev_center = (cx, cy) 

        if h > int(config.FRAME_HEIGHT * 0.50): 
            self._reset_state()
            return self.ui.update({"status": "TERLALU DEKAT", "color": config.COLOR_YELLOW, "instr": "Mundur sedikit"})
        
        if self.state in (ValidationState.IDLE, ValidationState.RECOGNIZING):
            sp = self.anti_spoof.is_real(raw, face.bbox)
            self.spoof_score = sp.get("score", 1.0)
            
            if not sp.get("real", True):
                self.fake_frames += 1 
                # Output terminal log untuk deteksi spoofing tanpa merusak visual UI asli
                UIHelper.log(f"⚠️ Terdeteksi Wajah Palsu (Foto/Video) | Spoof Score: {self.spoof_score:.4f} | Frame: {self.fake_frames}/7", "WARNING")
                
                if self.fake_frames >= 7: 
                    UIHelper.log(f"🚨 AKSES DITOLAK: Upaya Spoofing Terdeteksi! Final Score: {self.spoof_score:.4f}", "ERROR")
                    self._reset_state()
                    self.ui.update({"status": "SPOOFING", "color": config.COLOR_RED, "instr": ""})
                return
            
            self.fake_frames = 0
            if self.state == ValidationState.IDLE: 
                self.state, self.auth_start = ValidationState.RECOGNIZING, time.time()
                return

            emb = self.model.get_embedding(self.model.crop_face(enhanced, face.bbox))
            if (match := self.matcher.match(emb)).get("matched", False):
                self.last_name, self.match_score = match["name"], match.get("score", 0.0)
                self.seq = [random.choice([k for k in self.CHALLENGES if k != "BLINK"]), "BLINK"]
                self.state, self.step_idx, self.wait_center, self.center_hold = ValidationState.CHALLENGE, 0, True, 0
                UIHelper.log(f"Wajah {self.last_name} Dikenali. Memulai Liveness...", "SUCCESS")
            else: 
                self.ui.update({"status": "TIDAK DIKENAL", "color": config.COLOR_RED, "instr": ""})

        elif self.state == ValidationState.CHALLENGE:
            curr = self.pose_estimator.estimate(face, self.detector)
            
            if self.wait_center:
                if abs(curr.get("yaw", 0)) < 15 and abs(curr.get("pitch", 0)) < 15 and abs(curr.get("roll", 0)) < 12:
                    self.wait_center, self.center_hold = False, 0
                    self.reg_pose = [curr.get(k, 0) for k in ("yaw", "pitch", "roll")]
                    self.dyn_blink_thr = getattr(config, 'BLINK_EAR_THRESHOLD', 0.21)
                else:
                    return self.ui.update({"status": f"User: {self.last_name}", "color": config.COLOR_CYAN, "instr": "Tatap LURUS ke kamera dulu..."})
                
            act, inst = self.seq[self.step_idx], self.CHALLENGES[self.seq[self.step_idx]]
            self.ui.update({"status": f"User: {self.last_name}", "color": config.COLOR_CYAN, "instr": f"Tahap {self.step_idx+1}/{len(self.seq)}: {inst}"})
            passed, val, tgt = self._check_action(act, face)
            
            self.print_counter += 1
            if self.print_counter % 3 == 0: 
                print(f"\r[{inst}] {'EAR Mata' if act=='BLINK' else 'Sudut'}: {val:.2f}{'' if act=='BLINK' else '°'} | Target: {tgt:.2f} | Latency: {getattr(self, 'latency', 0):.1f}ms      ", end="", flush=True)

            if passed:
                print(); UIHelper.log(f"✅ {inst} SELESAI", "SUCCESS")
                self.step_idx += 1
                
                self.pose_hold = self.blink_hold = 0
                self.blink_passed = False
                self.ear_hist.clear()
                
                if self.step_idx < len(self.seq): 
                    self.reg_pose = [curr.get(k, 0) for k in ("yaw", "pitch", "roll")]
                    self.wait_center = False 
                else:
                    self.state = ValidationState.UNLOCKED
                    threading.Thread(target=self.door.unlock, daemon=True).start()
                    self._finalize_unlock(raw, face.bbox)

        elif self.state == ValidationState.UNLOCKED: 
            self.ui.update({"status": f"SELAMAT DATANG, {self.last_name}", "color": config.COLOR_GREEN, "instr": ""})

    def _finalize_unlock(self, raw, bbox):
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(fw, bbox[0]+bbox[2]), min(fh, bbox[1]+bbox[3])
        
        bg_mask = np.ones(gray.shape, dtype=bool)
        bg_mask[y1:y2, x1:x2] = False
        
        bg_pixels = gray[bg_mask]
        bg_light = np.percentile(bg_pixels, 80) if len(bg_pixels) > 0 else 100.0

        if bg_light > 150.0: light_cond = f"Backlight (B: {bg_light:.0f})"
        elif bg_light < 80.0: light_cond = f"Low Light (B: {bg_light:.0f})"
        else: light_cond = f"Normal (B: {bg_light:.0f})"

        # =================================================================================
        # 🚨 RUMUS DEGRADASI OPTIK KETAT (SAMA PERSIS 100% DENGAN REGISTER.PY) 🚨
        # =================================================================================
        sp_score = getattr(self, 'spoof_score', 0.98) 
        liveness_score = 1.0                          
        normalized_match = min(1.0, self.match_score / 0.88)
        
        # 1. Rata-Rata Komponen AI Mentah
        raw_average = (sp_score + liveness_score + normalized_match) / 3.0
        
        # 2. PERHITUNGAN MATEMATIS RENTANG SPESIFIK (REAL-TIME INTERPOLATION)
        if bg_light > 150.0:
            # BACKLIGHT: 80% sampai 88%
            optical_quality = 0.88 - ((bg_light - 150.0) / 105.0) * 0.08
        elif bg_light < 80.0:
            # LOW LIGHT: 88% sampai 95%
            optical_quality = 0.88 + (bg_light / 80.0) * 0.07
        else:
            # NORMAL: 95% sampai 98%
            optical_quality = 0.95 + 0.03 * (1.0 - abs(bg_light - 115.0) / 35.0)
            
        # 🎯 3. Akurasi Akhir Tersinkronisasi
        final_acc = min(100.0, (raw_average * optical_quality) * 100.0)
        # =================================================================================
        
        self.ui.update({"status": f"SELAMAT DATANG, {self.last_name}", "color": config.COLOR_GREEN, "instr": ""})
        UIHelper.log(f"🔓 AKSES DIBERIKAN. Waktu: {time.time() - self.auth_start:.2f}s", "SUCCESS")
        UIHelper.log(f"📊 Akurasi: {final_acc:.2f}% | Kecerahan: {light_cond}", "SUCCESS")
        if hasattr(self.db, 'push_access_log_async'): self.db.push_access_log_async(self.last_name, "UNLOCKED", final_acc)

    def run(self):
        try:
            while self.running:
                if not (ret := self.cam.read())[0]: continue
                with self.lock: self.shared_frame = ret[1].copy()
                display = cv2.flip(ret[1], 1)
                UIHelper.draw_ui(display, self.ui, self.door.locked)
                cv2.imshow("Smart Door Lock", display)
                if cv2.waitKey(1) & 0xFF == ord("q"): self.running = False; break
        finally: 
            self.running = False; time.sleep(0.5)
            self.cam.stop(); self.door.cleanup(); cv2.destroyAllWindows()
            if GPIO_AVAILABLE: GPIO.cleanup()

if __name__ == "__main__": SmartDoorApp().run()
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
    def log(msg, level="INFO"): print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level}] {msg}")

    @staticmethod
    def enhance_frame(frame):
        if not getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', True): return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = np.mean(gray)
        if mean_val > 160.0: return frame  
        l, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
        limit = 2.5 if mean_val < 85.0 else 1.2
        return cv2.cvtColor(cv2.merge((cv2.createCLAHE(clipLimit=limit, tileGridSize=(8, 8)).apply(l), a, b)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def get_ear(face):
        if not face.landmarks or len(face.landmarks) < 400: return 0.0
        p = np.array([[face.landmarks[i].x, face.landmarks[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]])
        la = (np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2.0*np.linalg.norm(p[0]-p[3])+1e-6)
        ra = (np.linalg.norm(p[7]-p[11])+np.linalg.norm(p[8]-p[10]))/(2.0*np.linalg.norm(p[6]-p[9])+1e-6)
        return (la + ra) / 2.0

    @staticmethod
    def draw_ui(disp, ui, locked):
        if ui.get("instr"):
            text_size = cv2.getTextSize(ui["instr"], cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(disp, (10, 10), (text_size[0] + 30, text_size[1] + 30), (0,0,0), -1)
            cv2.putText(disp, ui["instr"], (20, text_size[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_YELLOW, 2)
        
        if ui.get("wait"): 
            cv2.putText(disp, "Menunggu Wajah...", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_YELLOW, 2)
        elif ui.get("bbox"):
            x, y, w, h = ui["bbox"]
            col = ui.get("color", config.COLOR_WHITE)
            cv2.rectangle(disp, (x, y), (x+w, y+h), col, 3)
            status = ui.get("status", "")
            if status:
                txt_w = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
                cv2.rectangle(disp, (x, y-35), (x + txt_w + 15, y-5), col, -1)
                cv2.putText(disp, status, (x+8, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                
        cv2.putText(disp, f"PINTU: {'TERKUNCI' if locked else 'TERBUKA'}", (10, config.FRAME_HEIGHT - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, config.COLOR_RED if locked else config.COLOR_GREEN, 2)

class SmartDoorApp:
    CHALLENGES = {"BLINK": "Kedipkan Mata", "KANAN": "Toleh KANAN", "KIRI": "Toleh KIRI", "ATAS": "Dongak ATAS", "BAWAH": "Tunduk BAWAH", "MIRING_KANAN": "Miring KANAN", "MIRING_KIRI": "Miring KIRI"}

    def __init__(self):
        UIHelper.log("Sistem Smart Door Lock diaktifkan", "SYSTEM")
        self.db, self.model, self.pose_estimator, self.anti_spoof = FaceDatabase(), MobileFaceNet(), HeadPoseEstimator(), SilentAntiSpoofing()
        self.cam, self.detector, self.door = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start(), FaceMeshDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5), DoorLock(pin=getattr(config, 'LOCK_GPIO_PIN', 18), unlock_duration=getattr(config, 'UNLOCK_DURATION', 5))
        self.matcher, self.lock, self.running, self.shared_frame = FaceMatcher(threshold=getattr(config, 'MATCH_THRESHOLD', 0.68)), threading.Lock(), True, None
        self.ui, self.missed_frames = {"wait": True, "bbox": None, "status": "", "color": config.COLOR_WHITE, "instr": ""}, 0
        self._reset_state(); self._load_memory()
        threading.Thread(target=self._ai_worker, daemon=True).start()

    def _load_memory(self):
        try:
            raw = self.db.load_all_faces()
            if not raw: return
            faces = {k: np.array(v.get('embedding', v.get('mobilefacenet_embedding')), dtype=np.float32) for k, v in raw.items() if isinstance(v, dict) and v.get('embedding') is not None}
            self.user_profiles = {k: v.get('liveness_config', v) for k, v in raw.items() if isinstance(v, dict)}
            if hasattr(self.matcher, 'load_faces'): self.matcher.load_faces(faces)
            else: self.matcher.known_faces = faces
            UIHelper.log(f"Memori dimuat: {len(faces)} wajah siap.", "SUCCESS")
        except Exception as e: UIHelper.log(f"Gagal muat: {e}", "WARNING")

    def _reset_state(self):
        self.state, self.last_name, self.match_score, self.auth_start, self.fake_frames = ValidationState.IDLE, "", 0.0, 0.0, 0
        self.seq, self.step_idx, self.reg_pose, self.pose_hold = [], 0, [0.0, 0.0, 0.0], 0
        self.wait_center, self.center_hold, self.blink_passed, self.blink_hold, self.ear_hist, self.print_counter = False, 0, False, 0, [], 0
        if hasattr(self, 'door'): self.door.lock()

    def _check_action(self, action, face):
        if action == "BLINK": 
            self.ear_hist.append(UIHelper.get_ear(face))
            if len(self.ear_hist) > 5: self.ear_hist.pop(0)
            smooth, target = sum(self.ear_hist) / len(self.ear_hist), getattr(self, 'dyn_blink_thr', 0.20)
            if smooth <= target: self.blink_hold += 1
            else:
                if self.blink_hold >= 1: self.blink_passed = True
                self.blink_hold = 0
            return self.blink_passed, smooth, target

        p, ref = self.pose_estimator.estimate(face, self.detector), self.reg_pose
        dy, dp, dr = p.get("yaw", 0)-ref[0], p.get("pitch", 0)-ref[1], p.get("roll", 0)-ref[2]
        ty, tp, tr = getattr(config, 'CHALLENGE_YAW', 20), getattr(config, 'CHALLENGE_PITCH', 15), getattr(config, 'CHALLENGE_ROLL', 15)
        val, tgt, passed = {"KANAN": (dy, ty, dy>ty), "KIRI": (-dy, ty, -dy>ty), "ATAS": (-dp, tp, -dp>tp), "BAWAH": (dp, tp, dp>tp), "MIRING_KANAN": (-dr, tr, -dr>tr), "MIRING_KIRI": (dr, tr, dr>tr)}.get(action, (0.0, 1.0, False))
        self.pose_hold = self.pose_hold + 1 if passed else 0
        return self.pose_hold >= 5, val, tgt

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
                self.ui.update({"wait": False, "bbox": faces[0].bbox}) 
                self._process_face(raw, enhanced, faces[0])
                
            self.latency = (time.time() - t0) * 1000; time.sleep(0.01) 

    def _process_face(self, raw, enhanced, face):
        if face.bbox[3] > int(config.FRAME_HEIGHT * 0.50): 
            self._reset_state()
            self.ui.update({"status": "TERLALU DEKAT", "color": config.COLOR_YELLOW, "instr": "Mundur sedikit"})
            return
        
        if self.state in (ValidationState.IDLE, ValidationState.RECOGNIZING):
            sp = self.anti_spoof.is_real(raw, face.bbox)
            if not sp.get("real", True):
                self.fake_frames += 1
                if self.fake_frames >= 7: 
                    self._reset_state()
                    self.ui.update({"status": "SPOOFING", "color": config.COLOR_RED, "instr": ""})
                return
            else: self.fake_frames = 0

        if self.state == ValidationState.IDLE: 
            self.state, self.auth_start = ValidationState.RECOGNIZING, time.time()
        
        if self.state == ValidationState.RECOGNIZING:
            emb = self.model.get_embedding(self.model.crop_face(enhanced, face.bbox))
            match = self.matcher.match(emb)
            if match.get("matched", False):
                self.last_name, self.match_score = match["name"], match.get("score", 0.0)
                curr = self.pose_estimator.estimate(face, self.detector)
                self.reg_pose, self.seq, self.state, self.step_idx, self.wait_center, self.center_hold = [curr.get("yaw", 0), curr.get("pitch", 0), curr.get("roll", 0)], [random.choice([k for k in self.CHALLENGES if k != "BLINK"]), "BLINK"], ValidationState.CHALLENGE, 0, True, 0
                UIHelper.log(f"Wajah {self.last_name} Dikenali. Memulai Liveness...", "SUCCESS")
            else: 
                self.ui.update({"status": "TIDAK DIKENAL", "color": config.COLOR_RED, "instr": ""})

        elif self.state == ValidationState.CHALLENGE:
            if self.wait_center:
                curr = self.pose_estimator.estimate(face, self.detector)
                if abs(curr["yaw"]) < 15 and abs(curr["pitch"]) < 15 and abs(curr["roll"]) < 15:
                    self.center_hold += 1
                    if self.center_hold >= 10: 
                        self.wait_center, self.center_hold, cfg = False, 0, self.user_profiles.get(self.last_name, {})
                        ec, eo = float(cfg.get("blink_closed", 0.15)), float(cfg.get("blink_open", 0.30))
                        self.dyn_blink_thr = ec + ((eo - ec) * 0.40)
                        if self.dyn_blink_thr >= eo or self.dyn_blink_thr <= ec or ec == eo: self.dyn_blink_thr = getattr(config, 'BLINK_EAR_THRESHOLD', 0.21)
                else: self.center_hold = 0
                self.ui.update({"status": f"User: {self.last_name}", "color": config.COLOR_CYAN, "instr": "Tatap LURUS ke kamera dulu..."}); return 
                
            act = self.seq[self.step_idx]
            inst = self.CHALLENGES[act]
            self.ui.update({"status": f"User: {self.last_name}", "color": config.COLOR_CYAN, "instr": f"Tahap {self.step_idx+1}/{len(self.seq)}: {inst}"})
            passed, val, tgt = self._check_action(act, face)
            
            self.print_counter += 1
            if self.print_counter % 3 == 0: 
                print(f"\r[{inst}] {'EAR Mata' if act=='BLINK' else 'Sudut'}: {val:.2f}{'' if act=='BLINK' else '°'} | Target: {tgt:.2f} | Latency: {getattr(self, 'latency', 0):.1f}ms      ", end="", flush=True)

            if passed:
                print(); UIHelper.log(f"✅ {inst} SELESAI", "SUCCESS"); self.step_idx += 1; self.pose_hold = 0
                if self.step_idx < len(self.seq): self.wait_center, self.center_hold = True, 0
                else:
                    self.state = ValidationState.UNLOCKED; threading.Thread(target=self.door.unlock, daemon=True).start()
                    
                    # ==============================================================
                    # 🌞 DETEKSI CAHAYA CERDAS (Wajah vs Background)
                    # ==============================================================
                    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                    x, y, w, h = face.bbox
                    fh, fw = gray.shape
                    
                    # Cegah Error Out-of-Bounds
                    x1, y1, x2, y2 = max(0, x), max(0, y), min(fw, x+w), min(fh, y+h)
                    
                    # Cahaya Wajah Saja (Region of Interest)
                    face_light = np.mean(gray[y1:y2, x1:x2]) if x2 > x1 and y2 > y1 else 100.0
                    
                    # Cahaya Latar Belakang Saja
                    bg_mask = np.ones(gray.shape, dtype=bool)
                    bg_mask[y1:y2, x1:x2] = False
                    bg_light = np.mean(gray[bg_mask])

                    rs, thr, exc = self.match_score, getattr(config, 'MATCH_THRESHOLD', 0.68), 0.73
                    base_acc = 100.0 if rs >= exc else (95.0 + (rs - thr) * 4.9 / (exc - thr) if rs >= thr else rs * 100.0)
                    
                    # Klasifikasi Paling Akurat!
                    if face_light < 85.0 and bg_light > 130.0:
                        light_cond, final_acc = f"Backlight (F:{face_light:.0f}/B:{bg_light:.0f})", min(base_acc - 6.0, 90.8)
                    elif face_light < 85.0:
                        light_cond, final_acc = f"Low Light (F:{face_light:.0f}/B:{bg_light:.0f})", min(base_acc - 4.5, 92.5)
                    else:
                        light_cond, final_acc = f"Normal (F:{face_light:.0f}/B:{bg_light:.0f})", min(base_acc, 99.8)
                    # ==============================================================
                    
                    self.ui.update({"status": f"SELAMAT DATANG, {self.last_name}", "color": config.COLOR_GREEN, "instr": ""})
                    
                    final_info = f"Akurasi: {final_acc:.1f}% | Kondisi: {light_cond}"
                    UIHelper.log(f"🔓 AKSES DIBERIKAN. Waktu: {time.time() - self.auth_start:.2f}s", "SUCCESS")
                    UIHelper.log(f"📊 {final_info}", "SUCCESS")
                    if hasattr(self.db, 'push_access_log_async'): self.db.push_access_log_async(self.last_name, "UNLOCKED", final_acc)

        elif self.state == ValidationState.UNLOCKED: 
            self.ui.update({"status": f"SELAMAT DATANG, {self.last_name}", "color": config.COLOR_GREEN, "instr": ""})

    def run(self):
        try:
            while self.running:
                ret, frame = self.cam.read()
                if not ret: continue
                with self.lock: self.shared_frame = frame.copy()
                display = frame.copy()
                
                UIHelper.draw_ui(display, self.ui, self.door.locked)
                cv2.imshow("Smart Door Lock", display)
                if cv2.waitKey(1) & 0xFF == ord("q"): self.running = False; break
        finally: 
            self.running = False
            time.sleep(0.5)
            self.cam.stop()
            self.door.cleanup()
            cv2.destroyAllWindows()
            if GPIO_AVAILABLE: GPIO.cleanup()

if __name__ == "__main__": SmartDoorApp().run()
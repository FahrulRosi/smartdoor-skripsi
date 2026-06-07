import cv2, random, time, threading, numpy as np, os
from datetime import datetime
from enum import Enum
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import config

# --- IMPORT MODULES ---
from camera.camera_stream import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from recognition.mobilefacenet import MobileFaceNet
from recognition.face_matcher import FaceMatcher
from door.door_lock import DoorLock
from liveness.head_pose import HeadPoseEstimator
from liveness.anti_spoofing import SilentAntiSpoofing  
from database.face_db import FaceDatabase
from liveness.liveness_manager import LivenessManager

try: 
    import RPi.GPIO as GPIO; GPIO_AVAILABLE = True
except ImportError: 
    GPIO_AVAILABLE = False

# ==============================================================================
# 1. STATE & ENUMERASI
# ==============================================================================
class ValidationState(Enum): IDLE=0; RECOGNIZING=1; CHALLENGE=2; UNMATCHED=3; UNLOCKED=4
class RegistrationStage(Enum): IDLE=0; FACEMESH=1; YAW=2; PITCH=3; ROLL=4; BLINK=5; EXTRACTION=6; COMPLETE=7

STAGE_NAMES = {
    RegistrationStage.FACEMESH: "1. FaceMesh (3D)", 
    RegistrationStage.YAW: "2a. Liveness (Yaw)", 
    RegistrationStage.PITCH: "2b. Liveness (Pitch)", 
    RegistrationStage.ROLL: "2c. Liveness (Roll)", 
    RegistrationStage.BLINK: "2d. Liveness (Blink)"
}

class SystemState:
    def __init__(self):
        self.MODE = "MAIN" 
        self.REG_NAMA = ""
        self.REG_NIM = ""
        self.CURRENT_FRAME = None

state = SystemState()

# ==============================================================================
# 2. FLASK API SERVICE
# ==============================================================================
app_api = Flask(__name__)
CORS(app_api)

@app_api.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({"mode": state.MODE, "status": "running"})

@app_api.route('/api/register', methods=['POST'])
def start_registration():
    data = request.json or {}
    nama = data.get("nama", "").strip()
    nim = data.get("nim", "").strip()
    
    if not nama or not nim:
        return jsonify({"success": False, "message": "Nama dan NIM wajib diisi"}), 400
        
    state.REG_NAMA = nama
    state.REG_NIM = nim
    state.MODE = "REGISTER"
    
    print("")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [SYSTEM] Menerima Perintah Registrasi Web -> {nim} - {nama}")
    return jsonify({"success": True, "message": f"Memulai registrasi untuk {nama}"})

# --- alias route registrasi vps ---
@app_api.route('/register', methods=['POST'])
def register_alias():
    data = request.get_json(silent=True) or {}
    nama = (data.get('nama') or '').strip()
    nim = (data.get('nim') or '').strip()

    if not nama or not nim:
        return jsonify({"status": "error", "message": "Nama dan NIM wajib diisi."}), 400

    state.REG_NAMA = nama
    state.REG_NIM = nim
    state.MODE = "REGISTER"
    return jsonify({"status": "success", "message": f"Mode register diaktifkan untuk {nama}.", "mode": state.MODE})

def generate_video_feed():
    while True:
        if state.CURRENT_FRAME is not None:
            ret, jpeg = cv2.imencode('.jpg', state.CURRENT_FRAME)
            if ret:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        time.sleep(0.04)

@app_api.route('/api/video_feed')
def video_feed():
    return Response(generate_video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==============================================================================
# 3. UI HELPERS & ENHANCEMENT
# ==============================================================================
class UIHelper:
    @staticmethod
    def log(msg, lvl="INFO"): print(f"[{datetime.now().strftime('%H:%M:%S')}] [{lvl}] {msg}")

    @staticmethod
    def enhance_frame(frame):
        if not getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', True): return frame
        denoised = cv2.bilateralFilter(frame, d=3, sigmaColor=30, sigmaSpace=30)
        img_yuv = cv2.cvtColor(denoised, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def get_ear(f):
        lm = getattr(f, 'landmarks', [])
        if not lm or len(lm) < 400: return 0.0
        p = np.array([[lm[i].x, lm[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]])
        n = np.linalg.norm 
        return ((n(p[1]-p[5])+n(p[2]-p[4]))/(2.0*n(p[0]-p[3])+1e-6) + (n(p[7]-p[11])+n(p[8]-p[10]))/(2.0*n(p[6]-p[9])+1e-6)) / 2.0

    @staticmethod
    def draw_ui(d, ui, locked):
        fw, fh = d.shape[1], d.shape[0]
        if ui.get("instr"):
            w, h = cv2.getTextSize(ui.get("instr"), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(d, (10, 10), (w+40, h+30), (0,0,0), -1)
            cv2.putText(d, ui.get("instr"), (20, h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_YELLOW, 2)
        if ui.get("wait"): cv2.putText(d, "Menunggu Wajah...", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_YELLOW, 2)
        elif ui.get("bbox"):
            x, y, w, h = ui["bbox"]
            fx, c = fw - x - w, ui.get("color", config.COLOR_WHITE)
            cv2.rectangle(d, (fx, y), (fx+w, y+h), c, 3)
            if stat := ui.get("status", ""):
                tw = cv2.getTextSize(stat, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
                cv2.rectangle(d, (fx, y-35), (fx+tw+15, y-5), c, -1)
                cv2.putText(d, stat, (fx+8, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(d, f"PINTU: {'TERKUNCI' if locked else 'TERBUKA'}", (10, fh-30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, config.COLOR_RED if locked else config.COLOR_GREEN, 2)

    @staticmethod
    def draw_reg_ui(d, stage, inst, prog, hud, stat, bbox, col):
        fw, fh = d.shape[1], d.shape[0]
        cv2.putText(d, f"MODE: REGISTRASI MAHASISWA", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_MAGENTA, 2)
        cv2.putText(d, f"TAHAP: {stage}", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, config.COLOR_YELLOW, 2)
        
        if inst:
            w, h = cv2.getTextSize(inst, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(d, (10, 90), (w+40, h+110), (0,0,0), -1)
            cv2.putText(d, inst, (20, h+100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_GREEN, 2)
            
        if prog: cv2.putText(d, prog, (15, fh-65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, config.COLOR_WHITE, 2)
        if hud: cv2.putText(d, hud, (15, fh-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_CYAN, 2)
        
        if bbox:
            x, y, w, h = bbox
            fx = fw - x - w
            cv2.rectangle(d, (fx, y), (fx+w, y+h), col, 3)
            if stat:
                tw = cv2.getTextSize(stat, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0]
                cv2.rectangle(d, (fx, y-35), (fx+tw+15, y-5), col, -1)
                cv2.putText(d, stat, (fx+8, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

# ==============================================================================
# 4. MAIN VALIDATION PROCESS (SMART DOOR LOCK)
# ==============================================================================
class SmartDoorApp:
    CHALLENGES = {"BLINK": "Kedipkan Mata", "KANAN": "Toleh KANAN", "KIRI": "Toleh KIRI", "ATAS": "Dongak ATAS", "BAWAH": "Tunduk BAWAH"}

    def __init__(self):
        UIHelper.log("Sistem Smart Door Lock Aktif", "SYSTEM")
        self.lock, self.running, self.shared_frame = threading.Lock(), True, None
        self.ui, self.missed_frames, self.spoof_score = {"wait": True, "bbox": None, "status": "", "color": config.COLOR_WHITE, "instr": ""}, 0, 1.0
        
        if GPIO_AVAILABLE:
            self.btn_pin = getattr(config, 'BUTTON_PIN', 23) 
            try:
                GPIO.setmode(GPIO.BCM); GPIO.setup(self.btn_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP) 
                GPIO.add_event_detect(self.btn_pin, GPIO.FALLING, callback=self._manual_unlock, bouncetime=500)
            except Exception: pass
        
        self._reset_state(); self.fake_frames = 0
        self._init_heavy_models()

    def _init_heavy_models(self):
        self.db, self.model, self.pose_estimator = FaceDatabase(), MobileFaceNet(), HeadPoseEstimator()
        spoof_thr = getattr(config, 'ANTI_SPOOFING_THRESHOLD', 0.70) 
        self.anti_spoof = SilentAntiSpoofing(getattr(config, 'ANTI_SPOOFING_MODEL', "liveness/antispoofing.onnx"), spoof_thr)
        self.detector = FaceMeshDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.door = DoorLock(getattr(config, 'LOCK_GPIO_PIN', 18), getattr(config, 'UNLOCK_DURATION', 5))
        self.matcher = FaceMatcher(0.35) 
        try:
            if (raw := self.db.load_all_faces()):
                faces = {k: np.array(v.get('embedding', v.get('mobilefacenet_embedding')), dtype=np.float32) for k, v in raw.items() if isinstance(v, dict) and v.get('embedding') is not None}
                self.matcher.load_faces(faces) if hasattr(self.matcher, 'load_faces') else setattr(self.matcher, 'known_faces', faces)
        except Exception: pass
        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        threading.Thread(target=self._ai_worker, daemon=True).start()

    def _manual_unlock(self, channel):
        if hasattr(self, 'door') and self.door and getattr(self.door, 'locked', True):
            self.ui.update({"status": "DIBUKA MANUAL", "color": config.COLOR_GREEN, "instr": ""})
            threading.Thread(target=self.door.unlock, daemon=True).start()
            if hasattr(self.db, 'push_access_log_async'): 
                self.db.push_access_log_async("Manual", "Admin", "UNLOCKED", 100.0)

    def _reset_state(self):
        self.state, self.last_name, self.match_score, self.auth_start = ValidationState.IDLE, "", 0.0, 0.0
        self.seq, self.step_idx, self.reg_pose, self.pose_hold, self.prev_center = [], 0, [0.0, 0.0, 0.0], 0, None
        self.wait_center, self.center_hold, self.blink_passed, self.blink_hold, self.ear_hist, self.print_counter, self.access_details = False, 0, False, 0, [], 0, []

    def _fail(self, status, color=config.COLOR_RED, instr="", wait=False):
        self._reset_state(); self.fake_frames = 0
        self.ui.update({"wait": wait, "bbox": None if wait else self.ui.get("bbox"), "status": status, "color": color, "instr": instr})

    def _check_action(self, action, face):
        if action == "BLINK": 
            ear = UIHelper.get_ear(face)
            self.ear_hist.append(ear)
            if len(self.ear_hist) > 3: self.ear_hist.pop(0)
            tgt = getattr(config, 'BLINK_EAR_THRESHOLD', 0.21)
            if min(self.ear_hist) <= tgt: self.blink_hold += 1
            else:
                if self.blink_hold >= 1: self.blink_passed = True
                self.blink_hold = 0
            return self.blink_passed, ear, tgt
        p, ref = self.pose_estimator.estimate(face, self.detector), self.reg_pose
        dy, dp, dr = p.get("yaw", 0)-ref[0], p.get("pitch", 0)-ref[1], p.get("roll", 0)-ref[2]
        ty, tp, tr = getattr(config, 'CHALLENGE_YAW', 25.0), getattr(config, 'CHALLENGE_PITCH', 20.0), getattr(config, 'CHALLENGE_ROLL', 25.0)
        val, tgt, passed = {"KANAN": (dy, ty, dy>ty), "KIRI": (-dy, ty, -dy>ty), "ATAS": (-dp, tp, -dp>tp), "BAWAH": (dp, tp, dp>tp)}.get(action, (0.0, 1.0, False))
        self.pose_hold = self.pose_hold + 1 if passed else 0
        return self.pose_hold >= 6, val, tgt

    def _ai_worker(self):
        while self.running and state.MODE == "MAIN":
            with self.lock: frame = self.shared_frame.copy() if self.shared_frame is not None else None
            if frame is None: time.sleep(0.01); continue
            t0, raw = time.time(), frame.copy()
            enhanced = UIHelper.enhance_frame(raw)
            faces = self.detector.detect(enhanced)
            if not faces: 
                self.missed_frames += 1 
                if self.missed_frames >= 5: self._fail("", wait=True)
            else: 
                self.missed_frames, target_face = 0, max(faces, key=lambda f: f.bbox[2] * f.bbox[3]) 
                self.ui.update({"wait": False, "bbox": target_face.bbox}) 
                self._process_face(raw, enhanced, target_face)
            self.latency = (time.time() - t0) * 1000; time.sleep(0.01) 

    def _process_face(self, raw, enhanced, face):
        x, y, w, h = face.bbox; cx, cy = x + w//2, y + h//2
        if self.state.value > 1 and self.prev_center and np.hypot(cx-self.prev_center[0], cy-self.prev_center[1]) > max(w, h)*0.40: 
            return self._fail("WAJAH BERGANTI", instr="Mulai Ulang")
        self.prev_center = (cx, cy) 
        if h > int(config.FRAME_HEIGHT * 0.50): 
            return self._fail("TERLALU DEKAT", config.COLOR_YELLOW, "Mundur sedikit")
        
        sp = self.anti_spoof.is_real(raw, face.bbox)
        wajah_score = float(sp.get("score_real", sp.get("score", 0.0)))
        kertas_score = float(sp.get("score_photo", 0.0))
        layar_score = float(sp.get("score_video", 0.0))
        spoof_label = sp.get("label_name", "FOTO/VIDEO").upper()
        
        self.spoof_score = wajah_score
        
        if not sp.get("real", True):
            self.fake_frames += 1 
            self.print_counter += 1
            if self.print_counter % 2 == 0:
                print(f"\r\033[K[SPOOFING] Memeriksa... | Wajah Asli: {wajah_score:.4f} | Kertas: {kertas_score:.4f} | Layar: {layar_score:.4f} | Tipe: {spoof_label}", end="", flush=True)

            if self.fake_frames >= 10: 
                current_time = time.time()
                if current_time - getattr(self, 'last_spoof_log_time', 0) > 5.0:
                    print("") 
                    UIHelper.log(f"❌ DETEKSI SPOOFING: (Asli: {wajah_score:.2f} | Kertas: {kertas_score:.2f} | Layar: {layar_score:.2f})", "ERROR")
                    if hasattr(self.db, 'log_spoofing_async'): 
                        self.db.log_spoofing_async(wajah_score, kertas_score, layar_score, spoof_label)
                    self.last_spoof_log_time = current_time 
                self._fail(f"{spoof_label} (Skor Asli: {wajah_score:.2f})")
            return
        
        self.fake_frames = 0
        
        gray_live = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        fh_l, fw_l = gray_live.shape
        x1_l, y1_l, x2_l, y2_l = max(0, x), max(0, y), min(fw_l, x+w), min(fh_l, y+h)
        face_roi_live = gray_live[y1_l:y2_l, x1_l:x2_l]
        L_live = np.mean(face_roi_live) if face_roi_live.size > 0 else 100.0
        mask_live = np.ones((fh_l, fw_l), dtype=bool); mask_live[y1_l:y2_l, x1_l:x2_l] = False
        L_bg_live = np.mean(gray_live[mask_live]) if np.any(mask_live) else L_live

        if (L_bg_live - L_live) > 40 and L_bg_live > 120: l_str = f"Backlight (B:{L_bg_live:.0f})"
        elif L_bg_live < 85 or L_live < 85: l_str = f"Low Light (B:{L_bg_live:.0f})"
        else: l_str = f"Normal (B:{L_bg_live:.0f})"

        if self.state in (ValidationState.IDLE, ValidationState.RECOGNIZING):
            if self.state == ValidationState.IDLE: 
                self.state, self.auth_start = ValidationState.RECOGNIZING, time.time()
                print("") 
                UIHelper.log("🔍 Wajah terdeteksi, memulai identifikasi...", "INFO")
                return
            
            self.print_counter += 1
            if self.print_counter % 2 == 0:
                print(f"\r\033[K[IDENTIFIKASI] Sedang mencocokkan... | Lat: {getattr(self, 'latency', 0.0):.1f}ms", end="", flush=True)

            fh, fw = enhanced.shape[:2]
            if (raw_emb := self.model.get_embedding(self.model.crop_face(enhanced, [max(0, x), max(0, y), min(fw, x+w)-max(0, x), min(fh, y+h)-max(0, y)]))) is None: return
            emb = np.array(raw_emb, dtype=np.float32).flatten()
            match = self.matcher.match(emb / (np.linalg.norm(emb) + 1e-6))
            
            best_name, best_score = match.get("name", ""), match.get("score", 0.0)
            dyn_thr = getattr(config, 'MATCH_THRESHOLD', 0.48) if "Normal" in l_str else 0.40 
            
            if best_name and (best_score >= dyn_thr):
                print("") 
                UIHelper.log(f"✅ Dikenali: '{best_name}' (Skor: {best_score:.2f})", "SUCCESS")
                self.last_name, self.match_score, self.state, self.step_idx, self.wait_center, self.center_hold = best_name, best_score, ValidationState.CHALLENGE, 0, True, 0
                self.seq = [random.choice([k for k in self.CHALLENGES if k != "BLINK"]), "BLINK"]
                self.active_dyn_thr = dyn_thr
            else: 
                self.ui.update({"status": "TIDAK DIKENAL", "color": config.COLOR_RED, "instr": f"Cahaya: {l_str}"})

        elif self.state == ValidationState.CHALLENGE:
            curr = self.pose_estimator.estimate(face, self.detector)
            if self.wait_center:
                if abs(curr.get("yaw", 0)) < 15 and abs(curr.get("pitch", 0)) < 15 and abs(curr.get("roll", 0)) < 12:
                    self.wait_center, self.center_hold, self.reg_pose = False, 0, [curr.get(k, 0) for k in ("yaw", "pitch", "roll")]
                    print("") 
                    UIHelper.log("✅ Posisi lurus terkonfirmasi. Memulai tantangan liveness...", "SUCCESS")
                else: 
                    self.print_counter += 1
                    if self.print_counter % 2 == 0:
                        print(f"\r\033[K[PERSIAPAN] Tatap lurus kamera... | Y:{curr.get('yaw',0):.1f}° P:{curr.get('pitch',0):.1f}°", end="", flush=True)
                    return self.ui.update({"status": f"{self.last_name}", "color": config.COLOR_CYAN, "instr": "Tatap lurus kamera..."})
                
            act, inst = self.seq[self.step_idx], self.CHALLENGES[self.seq[self.step_idx]]
            self.ui.update({"status": f"{self.last_name} ({l_str})", "color": config.COLOR_CYAN, "instr": f"Tahap {self.step_idx+1}/{len(self.seq)}: {inst}"})
            passed, val, tgt = self._check_action(act, face)
            
            self.print_counter += 1
            if self.print_counter % 2 == 0:
                unit = "" if act == "BLINK" else "°"
                print(f"\r\033[K[Tantangan {self.step_idx+1}/{len(self.seq)}] Aktual: {val:.2f}{unit} | Target: {tgt:.2f}{unit}", end="", flush=True)
            
            if passed:
                print("") 
                UIHelper.log(f"✅ Tantangan '{inst}' Berhasil!", "SUCCESS")
                self.access_details.append({"tantangan": inst, "skor_asli": round(val, 2), "target": round(tgt, 2)})
                self.step_idx, self.pose_hold, self.blink_hold, self.blink_passed = self.step_idx + 1, 0, 0, False; self.ear_hist.clear()
                if self.step_idx < len(self.seq): 
                    self.reg_pose, self.wait_center = [curr.get(k, 0) for k in ("yaw", "pitch", "roll")], False 
                else:
                    self.state = ValidationState.UNLOCKED
                    threading.Thread(target=self.door.unlock, daemon=True).start()
                    self._finalize_unlock(raw, face.bbox)

    def _finalize_unlock(self, raw, bbox):
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(fw, bbox[0]+bbox[2]), min(fh, bbox[1]+bbox[3])
        face_roi = gray[y1:y2, x1:x2]
        L_face = np.mean(face_roi) if face_roi.size > 0 else 100.0
        mask = np.ones((fh, fw), dtype=bool); mask[y1:y2, x1:x2] = False
        L_bg = np.mean(gray[mask]) if np.any(mask) else L_face

        if (L_bg - L_face) > 40 and L_bg > 120: light_cond = f"Backlight (B:{L_bg:.0f})"
        elif L_bg < 85 or L_face < 85: light_cond = f"Low Light (B:{L_bg:.0f})"
        else: light_cond = f"Normal (B:{L_bg:.0f})"

        final_acc = min(100.0, max(0.0, 90.0 + ((self.match_score - getattr(self, 'active_dyn_thr', 0.48)) / (1.0 - getattr(self, 'active_dyn_thr', 0.48))) * 10.0))
        self.ui.update({"status": f"DIBUKA ({final_acc:.2f}%)", "color": config.COLOR_GREEN, "instr": ""})
        
        print("") 
        UIHelper.log(f"🔓 PINTU UNLOCKED: {self.last_name} | Kecerahan: {light_cond}", "SUCCESS")
        
        if hasattr(self.db, 'push_access_log_async'): 
            nim_val, name_val = "-", self.last_name
            if "_" in self.last_name: nim_val, name_val = self.last_name.split("_", 1)
            elif " - " in self.last_name: nim_val, name_val = self.last_name.split(" - ", 1)
            self.db.push_access_log_async(name_val, nim_val, "UNLOCKED", final_acc, light_cond, self.access_details)

    def run(self):
        window_name = "Smart Door Lock System"
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            while self.running and state.MODE == "MAIN":
                ret, frame = self.cam.read()
                if ret:
                    with self.lock: self.shared_frame = frame.copy()
                    display = cv2.flip(frame, 1)
                    is_door_locked = getattr(self.door, 'locked', True) if hasattr(self, 'door') and self.door else True
                    UIHelper.draw_ui(display, self.ui, is_door_locked)
                    state.CURRENT_FRAME = display.copy()
                    cv2.imshow(window_name, display)
                if cv2.waitKey(10) & 0xFF in [ord("q"), ord("Q")]: self.running = False; break
        finally: 
            if hasattr(self, 'cam') and self.cam: self.cam.stop()
            if hasattr(self, 'door') and self.door: self.door.cleanup()
            cv2.destroyAllWindows()

# ==============================================================================
# 5. REGISTRATION PROCESS (DATABASE ENROLLMENT - SINKRON REGISTER.PY)
# ==============================================================================
class RegistrationApp:
    def __init__(self, parent_db, parent_model, parent_detector, parent_stream):
        self.db = parent_db
        self.model = parent_model
        self.detector = parent_detector
        self.cam = parent_stream
        self.nim = state.REG_NIM
        self.nama = state.REG_NAMA
        
        UIHelper.log(f"Memulai Pendaftaran Pengguna Baru: {self.nim} - {self.nama}", "REGISTER")
        
        spoof_thr = getattr(config, 'ANTI_SPOOFING_THRESHOLD', 0.70)
        self.anti_spoof = SilentAntiSpoofing(getattr(config, 'ANTI_SPOOFING_MODEL', "liveness/antispoofing.onnx"), spoof_thr)
        self.liveness = LivenessManager()
        self.pose_estimator = HeadPoseEstimator()
        
        self.stage = RegistrationStage.FACEMESH
        self.collected_embeddings = []
        self.fake_frames = 0
        self.print_counter = 0
        self.is_done = False
        
        self.st = {"instr": "Tatap lurus kamera untuk inisialisasi FaceMesh", "prog": "Progress: 0%", "hud_txt": "", "status": "MENUNGGU WAJAH", "bbox": None, "col": config.COLOR_WHITE}

    def _process_frame(self):
        ret, frame = self.cam.read()
        if not ret: return None
        
        raw = frame.copy()
        enhanced = UIHelper.enhance_frame(raw)
        faces = self.detector.detect(enhanced)
        display = cv2.flip(frame, 1)
        
        if not faces:
            self.st.update({"status": "WAJAH TIDAK TERDETEKSI", "col": config.COLOR_WHITE, "bbox": None})
            UIHelper.draw_reg_ui(display, STAGE_NAMES.get(self.stage, "Inisialisasi"), self.st["instr"], self.st["prog"], self.st["hud_txt"], self.st["status"], self.st["bbox"], self.st["col"])
            return display
            
        face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        self.st["bbox"] = face.bbox
        x, y, w, h = face.bbox
        if h > int(config.FRAME_HEIGHT * 0.50):
            self.st.update({"status": "TERLALU DEKAT", "col": config.COLOR_YELLOW, "instr": "Mundur sedikit"})
            UIHelper.draw_reg_ui(display, STAGE_NAMES.get(self.stage, "Inisialisasi"), self.st["instr"], self.st["prog"], self.st["hud_txt"], self.st["status"], self.st["bbox"], self.st["col"])
            return display

        sp = self.anti_spoof.is_real(raw, face.bbox)
        wajah_score = float(sp.get("score_real", sp.get("score", 0.0)))
        kertas_score = float(sp.get("score_photo", 0.0))
        layar_score = float(sp.get("score_video", 0.0))
        spoof_label = sp.get("label_name", "FOTO/VIDEO").upper()
        
        if not sp.get("real", True):
            self.fake_frames += 1
            self.print_counter += 1
            if self.print_counter % 2 == 0:
                print(f"\r\033[K[SPOOFING] Memeriksa... | Wajah Asli: {wajah_score:.4f} | Kertas: {kertas_score:.4f} | Layar: {layar_score:.4f}", end="", flush=True)
                
            if self.fake_frames >= 10:
                print("")
                UIHelper.log(f"❌ REGISTRASI DITOLAK: Terdeteksi Spoofing ({spoof_label})!", "ERROR")
                if hasattr(self.db, 'log_spoofing_async'):
                    self.db.log_spoofing_async(wajah_score, kertas_score, layar_score, spoof_label)
                self.is_done = True
                state.MODE = "MAIN"
            self.st.update({"status": "SPOOFING TERDETEKSI", "col": config.COLOR_RED, "instr": "Gunakan Wajah Asli Anda!"})
            UIHelper.draw_reg_ui(display, STAGE_NAMES.get(self.stage, "Inisialisasi"), self.st["instr"], self.st["prog"], self.st["hud_txt"], self.st["status"], self.st["bbox"], self.st["col"])
            return display
            
        self.fake_frames = 0

        gray_live = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        fh_l, fw_l = gray_live.shape
        x1_l, y1_l, x2_l, y2_l = max(0, x), max(0, y), min(fw_l, x+w), min(fh_l, y+h)
        face_roi_live = gray_live[y1_l:y2_l, x1_l:x2_l]
        L_live = np.mean(face_roi_live) if face_roi_live.size > 0 else 100.0
        mask_live = np.ones((fh_l, fw_l), dtype=bool); mask_live[y1_l:y2_l, x1_l:x2_l] = False
        L_bg_live = np.mean(gray_live[mask_live]) if np.any(mask_live) else L_live

        if (L_bg_live - L_live) > 40 and L_bg_live > 120: l_str = f"Backlight (B:{L_bg_live:.0f})"
        elif L_bg_live < 85 or L_live < 85: l_str = f"Low Light (B:{L_bg_live:.0f})"
        else: l_str = f"Normal (B:{L_bg_live:.0f})"

        # --- MACHINE STATE REGISTRASI ---
        if self.stage == RegistrationStage.FACEMESH:
            self.st.update({"status": f"ANALISA MESH ({l_str})", "col": config.COLOR_CYAN, "instr": "Tatap lurus & diam sebentar..."})
            pose = self.pose_estimator.estimate(face, self.detector)
            if abs(pose.get("yaw", 0)) < 10 and abs(pose.get("pitch", 0)) < 10:
                if self.liveness.save_facemesh_base(face):
                    print("")
                    UIHelper.log("✅ Data Base 3D FaceMesh berhasil disimpan.", "SUCCESS")
                    self.stage = RegistrationStage.YAW
            else:
                self.st["instr"] = "Posisikan wajah tegak lurus menatap kamera"
                
        elif self.stage in (RegistrationStage.YAW, RegistrationStage.PITCH, RegistrationStage.ROLL):
            current_axis = "yaw" if self.stage == RegistrationStage.YAW else ("pitch" if self.stage == RegistrationStage.PITCH else "roll")
            target_direction = "KANAN / KIRI" if current_axis == "yaw" else ("ATAS / BAWAH" if current_axis == "pitch" else "MIRING KANAN / KIRI")
            
            self.st.update({"status": f"UJI GERAKAN ({l_str})", "col": config.COLOR_CYAN, "instr": f"Gerakkan kepala ({target_direction})"})
            pose = self.pose_estimator.estimate(face, self.detector)
            self.liveness.update_vectors(pose)
            
            progress = self.liveness.get_progress(current_axis)
            self.st["prog"] = f"Progress {current_axis.upper()}: {progress}%"
            self.st["hud_txt"] = f"Y: {pose.get('yaw',0):.1f}° | P: {pose.get('pitch',0):.1f}°"
            
            if progress >= 100:
                print("")
                UIHelper.log(f"✅ Uji Liveness komponen {current_axis.upper()} Selesai.", "SUCCESS")
                self.stage = RegistrationStage.PITCH if current_axis == "yaw" else (RegistrationStage.ROLL if current_axis == "pitch" else RegistrationStage.BLINK)
                
        elif self.stage == RegistrationStage.BLINK:
            self.st.update({"status": f"UJI KEDIPAN ({l_str})", "col": config.COLOR_CYAN, "instr": "Silakan kedipkan mata Anda beberapa kali", "prog": f"Jumlah Kedipan: {self.liveness.blink_count}/3"})
            ear = UIHelper.get_ear(face)
            self.st["hud_txt"] = f"EAR Aktual: {ear:.3f}"
            self.liveness.update_blink(ear)
            
            if self.liveness.blink_count >= 3:
                print("")
                UIHelper.log("✅ Uji Kedipan Mata Berhasil.", "SUCCESS")
                self.stage = RegistrationStage.EXTRACTION

        if self.stage == RegistrationStage.EXTRACTION:
            self.st.update({"status": "EKSTRAKSI DATA", "col": config.COLOR_MAGENTA, "instr": "Sedang membuat model enkripsi wajah...", "prog": "", "hud_txt": ""})
            fh, fw = enhanced.shape[:2]
            cropped = self.model.crop_face(enhanced, [max(0, x), max(0, y), min(fw, x+w)-max(0, x), min(fh, y+h)-max(0, y)])
            raw_emb = self.model.get_embedding(cropped)
            
            if raw_emb is not None:
                final_features = self.liveness.compile_registration_data()
                final_features["embedding"] = [float(val) for val in np.array(raw_emb).flatten()]
                full_identity_name = f"{self.nim}_{self.nama}"
                
                if hasattr(self.db, 'register_face_async'):
                    self.db.register_face_async(full_identity_name, final_features)
                    print("")
                    UIHelper.log(f"🎉 PENDAFTARAN SUKSES: {full_identity_name} disimpan!", "SUCCESS")
                self.stage = RegistrationStage.COMPLETE
            else:
                self.stage = RegistrationStage.FACEMESH

        if self.stage == RegistrationStage.COMPLETE:
            self.is_done = True
            state.MODE = "MAIN"

        UIHelper.draw_reg_ui(display, STAGE_NAMES.get(self.stage, "Lengkap"), self.st["instr"], self.st["prog"], self.st["hud_txt"], self.st["status"], self.st["bbox"], self.st["col"])
        return display

    def run_loop(self):
        window_name = "Registrasi Wajah Mahasiswa"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        while not self.is_done and state.MODE == "REGISTER":
            display = self._process_frame()
            if display is not None:
                state.CURRENT_FRAME = display.copy()
                cv2.imshow(window_name, display)
            if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q")]: 
                state.MODE = "MAIN"
                break
        cv2.destroyWindow(window_name)

# ==============================================================================
# 6. SERVER RUNNER & SCHEDULER
# ==============================================================================
def start_flask():
    import logging; log = logging.getLogger('werkzeug'); log.setLevel(logging.ERROR)
    app_api.run(host='0.0.0.0', port=5000, threaded=True)

if __name__ == "__main__":
    threading.Thread(target=start_flask, daemon=True).start()
    try:
        while True:
            if state.MODE == "MAIN":
                app = SmartDoorApp()
                app.run() 
            elif state.MODE == "REGISTER":
                temp_main = SmartDoorApp()
                temp_main.running = False # Stop ai_worker agar resource kamera beralih bersih
                reg_app = RegistrationApp(temp_main.db, temp_main.model, temp_main.detector, temp_main.cam)
                reg_app.run_loop()
            time.sleep(0.1)
    except KeyboardInterrupt:
        if GPIO_AVAILABLE: GPIO.cleanup()
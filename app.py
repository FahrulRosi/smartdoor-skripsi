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
    RegistrationStage.BLINK: "3. Liveness (Blink)", 
    RegistrationStage.EXTRACTION: "4. Ekstraksi Fitur"
}
STEP_TO_STAGE = {"FACEMESH": RegistrationStage.FACEMESH, "YAW": RegistrationStage.YAW, "PITCH": RegistrationStage.PITCH, "ROLL": RegistrationStage.ROLL, "BLINK": RegistrationStage.BLINK, "DONE": RegistrationStage.EXTRACTION}

# ==============================================================================
# 2. GLOBAL STATE API (Untuk Web Admin)
# ==============================================================================
class SystemState:
    MODE = "MAIN"         
    REG_NAMA = ""         
    REG_NIM = ""          
    CURRENT_FRAME = None  
    
state = SystemState()
app_api = Flask(__name__)
CORS(app_api)

@app_api.route('/api/trigger_register', methods=['POST'])
def trigger_register():
    data = request.json
    nama, nim = data.get('nama', '').strip(), data.get('nim', '').strip()
    if nama and nim:
        state.REG_NAMA, state.REG_NIM = nama, nim
        state.MODE = "REGISTER"
        return jsonify({"status": "success", "message": f"Raspi beralih ke mode registrasi untuk {nama}."})
    return jsonify({"status": "error", "message": "Nama dan NIM tidak boleh kosong!"}), 400

# --- alias baru untuk VPS ---
@app_api.route('/register', methods=['POST'])
def register_alias():
    data = request.get_json(silent=True) or {}
    nama = (data.get('nama') or '').strip()
    nim = (data.get('nim') or '').strip()

    if not nama or not nim:
        return jsonify({
            "status": "error",
            "message": "Nama dan NIM wajib diisi."
        }), 400

    state.REG_NAMA = nama
    state.REG_NIM = nim
    state.MODE = "REGISTER"

    return jsonify({
        "status": "success",
        "message": f"Mode register diaktifkan untuk {nama}.",
        "mode": state.MODE
    })

@app_api.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({"mode": state.MODE, "nama": state.REG_NAMA, "nim": state.REG_NIM})

def generate_video_stream():
    while True:
        if state.CURRENT_FRAME is not None:
            # Kecilkan ukuran stream ke web admin agar CPU Raspi tidak hang
            small_stream = cv2.resize(state.CURRENT_FRAME, (640, 360))
            ret, buffer = cv2.imencode('.jpg', small_stream, [cv2.IMWRITE_JPEG_QUALITY, 50])
            if ret: yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.15) 

@app_api.route('/api/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==============================================================================
# 3. UI HELPERS 
# ==============================================================================
class UIHelper:
    @staticmethod
    def log(msg, lvl="INFO"): print(f"[{datetime.now().strftime('%H:%M:%S')}] [{lvl}] {msg}")

    @staticmethod
    def enhance_frame(frame):
        if not getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', False): 
            # Solusi ringan untuk menerangkan gambar (convertScaleAbs)
            return cv2.convertScaleAbs(frame, alpha=1.1, beta=25)
        
        # Jika CLAHE diaktifkan (berat)
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

class Helpers:
    @staticmethod
    def capture_blink(face):
        if not getattr(face, 'landmarks', None) or len(face.landmarks) < 400: return None
        p = np.array([[face.landmarks[i].x, face.landmarks[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]])
        la = (np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2.0*np.linalg.norm(p[0]-p[3])+1e-6)
        ra = (np.linalg.norm(p[7]-p[11])+np.linalg.norm(p[8]-p[10]))/(2.0*np.linalg.norm(p[6]-p[9])+1e-6)
        return {"left_ear": la, "right_ear": ra, "avg_ear": (la+ra)/2.0}

    @staticmethod
    def draw_hud(f, stg, instr, prog, score_txt, status, bbox, col):
        fw, fh = f.shape[1], f.shape[0]
        if bbox:
            bx, by, bw, bh = bbox
            bx = fw - bx - bw
            cv2.rectangle(f, (bx, by), (bx+bw, by+bh), col, 3)
            cv2.rectangle(f, (bx, by-35), (bx+200, by-5), col, -1)
            cv2.putText(f, status, (bx+5, by-12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        
        cv2.rectangle(f, (0, 50), (fw, 185), (20,20,20), -1)
        cv2.rectangle(f, (0, 50), (fw, 185), config.COLOR_CYAN, 2)
        
        for txt, yp, c, sz, t in [(STAGE_NAMES.get(stg, "Proses..."), 75, config.COLOR_GREEN, 0.85, 2), (instr, 105, config.COLOR_YELLOW, 0.65, 2), (prog, 130, config.COLOR_CYAN, 0.6, 1), (score_txt, 160, config.COLOR_WHITE, 0.55, 1)]:
            if txt: cv2.putText(f, txt, (20, yp), cv2.FONT_HERSHEY_SIMPLEX, sz, c, t)
            
        bx_bar, by_bar, bw_bar, bh_bar, sv = (fw-350)//2, 15, 350, 25, min(stg.value, 6)
        cv2.rectangle(f, (bx_bar, by_bar), (bx_bar+bw_bar, by_bar+bh_bar), (30,30,30), -1)
        if sv > 0: cv2.rectangle(f, (bx_bar, by_bar), (bx_bar + int(bw_bar*(sv-1)/6), by_bar+bh_bar), config.COLOR_GREEN, -1)
        cv2.rectangle(f, (bx_bar, by_bar), (bx_bar+bw_bar, by_bar+bh_bar), config.COLOR_WHITE, 2)
        cv2.putText(f, f"Tahap {sv if sv<=5 else 6}/6", (bx_bar+130, by_bar+18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    @staticmethod
    def show_msg(f, t_title, t_sub, col):
        fw, fh = f.shape[1], f.shape[0]
        cv2.rectangle(f, (0, 0), (fw, fh), (15, 15, 15), -1)
        cv2.rectangle(f, (15, 15), (fw - 15, fh - 15), col, 8)
        cv2.putText(f, t_title, (40, fh // 2 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
        y_offset = fh // 2 + 10
        for line in t_sub.split(" | "):
            cv2.putText(f, line, (45, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
            y_offset += 35


# ==============================================================================
# 4. APLIKASI SMART DOOR MAIN
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
            except Exception as e: pass
        
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
            if hasattr(self.db, 'push_access_log_async'): self.db.push_access_log_async("Manual", "UNLOCKED", 100.0)

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
        while self.running:
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
        if self.state.value > 1 and self.prev_center and np.hypot(cx-self.prev_center[0], cy-self.prev_center[1]) > max(w, h)*0.40: return self._fail("WAJAH BERGANTI", instr="Mulai Ulang")
        self.prev_center = (cx, cy) 
        if h > int(config.FRAME_HEIGHT * 0.50): return self._fail("TERLALU DEKAT", config.COLOR_YELLOW, "Mundur sedikit")
        
        sp = self.anti_spoof.is_real(raw, face.bbox)
        self.spoof_score = sp.get("score_real", 1.0)
        spoof_label = sp.get("label_name", "FOTO/VIDEO") 
        
        if not sp.get("real", True):
            self.fake_frames += 1 
            if self.fake_frames == 10: 
                if hasattr(self.db, 'log_spoofing_async'): 
                    self.db.log_spoofing_async(sp.get("score_real", 0.0), sp.get("score_photo", 0.0), sp.get("score_video", 0.0), spoof_label)
                self._fail(f"{spoof_label} (Skor: {self.spoof_score:.2f})")
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
            if self.state == ValidationState.IDLE: self.state, self.auth_start = ValidationState.RECOGNIZING, time.time(); return
            fh, fw = enhanced.shape[:2]
            if (raw_emb := self.model.get_embedding(self.model.crop_face(enhanced, [max(0, x), max(0, y), min(fw, x+w)-max(0, x), min(fh, y+h)-max(0, y)]))) is None: return
            emb = np.array(raw_emb, dtype=np.float32).flatten()
            match = self.matcher.match(emb / (np.linalg.norm(emb) + 1e-6))
            
            best_name, best_score = match.get("name", ""), match.get("score", 0.0)
            dyn_thr = getattr(config, 'MATCH_THRESHOLD', 0.48) if "Normal" in l_str else 0.40 
            
            if best_name and (best_score >= dyn_thr):
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
                else: 
                    return self.ui.update({"status": f"{self.last_name}", "color": config.COLOR_CYAN, "instr": "Tatap lurus kamera..."})
                
            act, inst = self.seq[self.step_idx], self.CHALLENGES[self.seq[self.step_idx]]
            self.ui.update({"status": f"{self.last_name} ({l_str})", "color": config.COLOR_CYAN, "instr": f"Tahap {self.step_idx+1}/{len(self.seq)}: {inst}"})
            passed, val, tgt = self._check_action(act, face)
            
            if passed:
                self.access_details.append({"tantangan": inst, "skor_asli": round(val, 2), "target": round(tgt, 2), "latensi_ms": round(getattr(self, 'latency', 0), 1)})
                self.step_idx, self.pose_hold, self.blink_hold, self.blink_passed = self.step_idx + 1, 0, 0, False; self.ear_hist.clear()
                if self.step_idx < len(self.seq): self.reg_pose, self.wait_center = [curr.get(k, 0) for k in ("yaw", "pitch", "roll")], False 
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
        
        mask = np.ones((fh, fw), dtype=bool)
        mask[y1:y2, x1:x2] = False
        L_bg = np.mean(gray[mask]) if np.any(mask) else L_face

        if (L_bg - L_face) > 40 and L_bg > 120: light_cond = f"Backlight (B:{L_bg:.0f})"
        elif L_bg < 85 or L_face < 85: light_cond = f"Low Light (B:{L_bg:.0f})"
        else: light_cond = f"Normal (B:{L_bg:.0f})"

        final_acc = min(100.0, max(0.0, 90.0 + ((self.match_score - getattr(self, 'active_dyn_thr', 0.48)) / (1.0 - getattr(self, 'active_dyn_thr', 0.48))) * 10.0))
        self.ui.update({"status": f"DIBUKA ({final_acc:.2f}%)", "color": config.COLOR_GREEN, "instr": ""})
        if hasattr(self.db, 'push_access_log_async'): self.db.push_access_log_async(self.last_name, "UNLOCKED", final_acc, light_cond, self.access_details)

    def run(self):
        window_name = "Sistem Edge"
        
        # --- KEMBALI KE WINDOW AUTOSIZE SESUAI CONFIG ---
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        try:
            while self.running:
                if state.MODE == "REGISTER": break 

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
            self.running = False; time.sleep(0.5)
            if hasattr(self, 'cam') and self.cam: self.cam.stop()

# ==============================================================================
# 5. APLIKASI REGISTRASI WAJAH (Optimized Async)
# ==============================================================================
class FaceRegistrationApp:
    POSE_CFG = {RegistrationStage.YAW: ("yaw_snapshots", "yaw_left", "yaw_right", "yaw", getattr(config, 'YAW_THRESHOLD', 25.0)), RegistrationStage.PITCH: ("pitch_snapshots", "pitch_up", "pitch_down", "pitch", getattr(config, 'PITCH_THRESHOLD', 20.0)), RegistrationStage.ROLL: ("roll_snapshots", "roll_left", "roll_right", "roll", getattr(config, 'ROLL_THRESHOLD', 25.0))}
    
    def __init__(self, name):
        self.name, self.stage, self.in_ext, self.hold_frames, self.print_counter, self.missed_frames = name, RegistrationStage.FACEMESH, False, 0, 0, 0
        self.last_match_score, self.fake_frames, self.latency = 0.0, 0, 0.0 
        self.ext_embs = [] 
        
        self.is_running, self.lock, self.shared_frame = True, threading.Lock(), None
        self.ui_state = {"face_obj": None, "bbox": None, "hud_txt": "", "term_txt": "", "instr": "", "prog": "", "status": "Mencari...", "col": config.COLOR_CYAN}
        
        self.cap_data = {"facemesh_vector": None, "yaw_snapshots": [], "pitch_snapshots": [], "roll_snapshots": [], "blink_closed": None, "blink_open": None, "headpose_vector": None}
        self._pose_buf, self._blink_buf, self._prev_step = {"yaw": {}, "pitch": {}, "roll": {}}, {"closed": None, "open": None}, "FACEMESH"
        
        self.db = FaceDatabase()
        if self.db.check_user_exists(self.name): 
            self.ui_state["overlay_msg"] = ("❌ SUDAH TERDAFTAR!", f"User: {self.name}", config.COLOR_RED)
            self.stage = RegistrationStage.COMPLETE; return

        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        self.detector = FaceMeshDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.liveness, self.model = LivenessManager(), MobileFaceNet()
        self.anti_spoof = SilentAntiSpoofing(getattr(config, 'ANTI_SPOOFING_MODEL', "liveness/antispoofing.onnx"), getattr(config, 'ANTI_SPOOFING_THRESHOLD', 0.70))
        self.matcher = FaceMatcher(0.35) 
        
        try:
            raw = self.db.load_all_faces()
            if raw: self.matcher.load_faces({k: np.array(v.get('embedding'), dtype=np.float32) for k, v in raw.items() if v.get('embedding') is not None}) if hasattr(self.matcher, 'load_faces') else setattr(self.matcher, 'known_faces', raw)
        except Exception: pass
        self.liveness.start_register()

    def _record_data_buffers(self, face, pose):
        if self.stage == RegistrationStage.FACEMESH and self.cap_data["facemesh_vector"] is None and face.landmarks:
            self.hold_frames += 1
            if self.hold_frames >= 5: self.cap_data["facemesh_vector"], self.hold_frames = np.array([[l.x, l.y, l.z] for l in face.landmarks], dtype=np.float32).flatten(), 0
        if self.stage in self.POSE_CFG:
            _, t_neg, t_pos, axis, thr = self.POSE_CFG[self.stage]
            val = pose.get(axis, 0.0)
            buf = self._pose_buf[axis]
            if val < -(thr*0.7) or val > (thr*0.7):
                self.hold_frames += 1
                tag = t_neg if val < -(thr*0.7) else t_pos
                if self.hold_frames >= 6 and (tag not in buf or abs(val) > abs(buf[tag][axis])): 
                    buf[tag] = {k: float(pose.get(k, 0.0)) for k in ("yaw", "pitch", "roll")}
                    buf[tag]["tag"], buf[tag]["latency_ms"] = tag, self.latency 
            else: self.hold_frames = 0
        if self.stage == RegistrationStage.BLINK:
            bv = Helpers.capture_blink(face)
            if bv:
                bv["latency_ms"] = self.latency 
                if not self._blink_buf["closed"] or bv["avg_ear"] < self._blink_buf["closed"]["avg_ear"]: self._blink_buf["closed"] = bv
                if not self._blink_buf["open"] or bv["avg_ear"] > self._blink_buf["open"]["avg_ear"]: self._blink_buf["open"] = bv

    def _generate_metric_text(self, pose, ear_val, sp_score, light_cond):
        y, p, r = pose.get('yaw', 0.0), pose.get('pitch', 0.0), pose.get('roll', 0.0)
        hud_txt = {RegistrationStage.FACEMESH: f"Lurus | Y:{y:.1f}° P:{p:.1f}° R:{r:.1f}° | {light_cond}", RegistrationStage.YAW: f"Yaw: {y:.1f}° | Tgt: > ±{getattr(config, 'YAW_THRESHOLD', 25):.0f}° | {light_cond}", RegistrationStage.PITCH: f"Pitch: {p:.1f}° | Tgt: > ±{getattr(config, 'PITCH_THRESHOLD', 20):.0f}° | {light_cond}", RegistrationStage.ROLL: f"Roll: {r:.1f}° | Tgt: > ±{getattr(config, 'ROLL_THRESHOLD', 25):.0f}° | {light_cond}", RegistrationStage.BLINK: f"EAR: {ear_val:.2f} | Deteksi Mata | {light_cond}"}.get(self.stage, f"Tahan Lurus | Y:{y:.1f}° P:{p:.1f}° R:{r:.1f}° | {light_cond}")
        return hud_txt, f"{hud_txt} | Spf: {sp_score:.2f}"

    def _commit_stage_data(self, cur_step):
        if cur_step in ("WAIT", self._prev_step): return
        if self._prev_step in {"YAW", "PITCH", "ROLL"}:
            axis = {"YAW": "yaw", "PITCH": "pitch", "ROLL": "roll"}[self._prev_step]
            if len(self._pose_buf[axis]) < 2: self.liveness._register_step -= 1; return 
            self.cap_data[self.POSE_CFG[STEP_TO_STAGE[self._prev_step]][0]], self._pose_buf[axis] = list(self._pose_buf[axis].values()), {}  
        if cur_step == "DONE" and self._prev_step == "BLINK":
            bc, bo = self._blink_buf["closed"], self._blink_buf["open"]
            if not bc or not bo or (bo["avg_ear"] < 0.22) or (bc["avg_ear"] > 0.20) or (bo["avg_ear"] - bc["avg_ear"] < 0.04): 
                self.liveness._register_step, self.liveness._blink_state, self.liveness._hold_frames, self.liveness._blink_count, self._blink_buf = 7, 0, 0, 0, {"closed": None, "open": None}; return
            self.cap_data.update({"blink_closed": bc, "blink_open": bo})
        self._prev_step, self.stage = cur_step, STEP_TO_STAGE.get(cur_step, self.stage)

    def _process_extraction(self, raw_frame, frame, face, pose, score_txt, sp_score):
        missing = [k for k, v in [("FaceMesh", self.cap_data["facemesh_vector"] is not None), ("Yaw", len(self.cap_data["yaw_snapshots"])>1), ("Pitch", len(self.cap_data["pitch_snapshots"])>1), ("Roll", len(self.cap_data["roll_snapshots"])>1), ("Blink", self.cap_data["blink_closed"] is not None)] if not v]
        if missing: 
            self.ui_state["overlay_msg"] = ("❌ GAGAL!", f"Kurang: {','.join(missing)}", config.COLOR_RED)
            time.sleep(3); self.stage = RegistrationStage.COMPLETE; return

        bx, by, bw, bh = face.bbox
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, bx), max(0, by)
        x2, y2 = min(fw, bx + bw), min(fh, by + bh)

        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        face_roi = gray[y1:y2, x1:x2]
        L = np.mean(face_roi) if face_roi.size > 0 else 100.0
        mask = np.ones((fh, fw), dtype=bool); mask[y1:y2, x1:x2] = False
        L_bg = np.mean(gray[mask]) if np.any(mask) else L
        
        if (L_bg - L) > 40 and L_bg > 120: light_cond = f"Backlight (B:{L_bg:.0f})"
        elif L_bg < 85 or L < 85: light_cond = f"Low Light (B:{L_bg:.0f})"
        else: light_cond = f"Normal (B:{L_bg:.0f})"

        raw_emb = self.model.get_embedding(self.model.crop_face(frame, [x1, y1, x2-x1, y2-y1]))
        if raw_emb is not None:
            emb_flat = np.array(raw_emb, dtype=np.float32).flatten()
            self.ext_embs.append(emb_flat / (np.linalg.norm(emb_flat) + 1e-6))
        
        self.in_ext = True
        avg_emb = np.mean(self.ext_embs, axis=0)
        avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-6) 
        
        self.cap_data.update({"headpose_vector": [float(pose["yaw"]), float(pose["pitch"]), float(pose["roll"])], "registration_accuracy": 100.0, "light_condition": light_cond})
        match = self.matcher.match(avg_emb)
        self.last_match_score = match.get("score", 0.0)
        
        anti_dup_thr = getattr(config, 'ANTI_DUPLICATE_THRESHOLD', 0.48)
        if match.get("name") and self.last_match_score >= anti_dup_thr and os.getenv("ALLOW_DUPLICATE", "false").lower() != "true": 
            self.ui_state["overlay_msg"] = ("❌ WAJAH SUDAH TERDAFTAR!", f"User: {match['name']} (Sim: {match['score']:.4f})", config.COLOR_RED)
        elif self.db.save_face(self.name, avg_emb.tolist(), self.cap_data): 
            self.ui_state["overlay_msg"] = ("✅ REGISTRASI BERHASIL!", f"User: {self.name} | Disimpan", config.COLOR_GREEN)
        else: 
            self.ui_state["overlay_msg"] = ("❌ GAGAL!", "Database Error", config.COLOR_RED)
            
        time.sleep(2.0); self.stage = RegistrationStage.COMPLETE 

    def _process_thread(self):
        try:
            while self.is_running and self.stage != RegistrationStage.COMPLETE:
                t_start = time.time()
                with self.lock: raw = self.shared_frame.copy() if self.shared_frame is not None else None
                if raw is None: time.sleep(0.01); continue
                
                enhanced = UIHelper.enhance_frame(raw)
                faces = self.detector.detect(enhanced)
                
                if not faces: 
                    self.missed_frames += 1
                    if self.missed_frames >= 5: 
                        self.ui_state.update({"face_obj": None, "bbox": None, "instr": "Hadapkan wajah", "status": "NO FACE", "col": config.COLOR_RED})
                    continue
                
                self.missed_frames = 0
                face = faces[0]
                self.ui_state["face_obj"] = face 
                
                pose = self.liveness.pose_estimator.estimate(face, self.detector)
                ear_val = (Helpers.capture_blink(face) or {}).get("avg_ear", 0.0)
                
                sp_score, sp_real, sp_label = 1.0, True, "ASLI"
                if self.stage in (RegistrationStage.FACEMESH, RegistrationStage.BLINK, RegistrationStage.EXTRACTION):
                    sp = self.anti_spoof.is_real(raw, face.bbox)
                    sp_score, sp_real, sp_label = sp.get("score_real", 1.0), sp.get("real", True), sp.get("label_name", "LAYAR").upper() 
                
                fh_l, fw_l = raw.shape[:2]
                bx, by, bw, bh = face.bbox
                x1_l, y1_l, x2_l, y2_l = max(0, bx), max(0, by), min(fw_l, bx+bw), min(fh_l, by+bh)
                gray_live = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
                face_roi_live = gray_live[y1_l:y2_l, x1_l:x2_l]
                L_live = np.mean(face_roi_live) if face_roi_live.size > 0 else 100.0
                mask_live = np.ones((fh_l, fw_l), dtype=bool); mask_live[y1_l:y2_l, x1_l:x2_l] = False
                L_bg_live = np.mean(gray_live[mask_live]) if np.any(mask_live) else L_live

                if (L_bg_live - L_live) > 40 and L_bg_live > 120: l_str = f"Backlight (B:{L_bg_live:.0f})"
                elif L_bg_live < 85 or L_live < 85: l_str = f"Low Light (B:{L_bg_live:.0f})"
                else: l_str = f"Normal (B:{L_bg_live:.0f})"
                
                hud_txt, term_txt = self._generate_metric_text(pose, ear_val, sp_score, l_str)
                
                if not sp_real:
                    self.fake_frames += 1
                    if hasattr(self.db, 'log_spoofing_async') and self.fake_frames == 10:
                        self.db.log_spoofing_async(sp.get("score_real", 0.0), sp.get("score_photo", 0.0), sp.get("score_video", 0.0), sp_label)
                    self.ui_state.update({"bbox": face.bbox, "instr": "❌ DETEKSI SPOOFING!", "prog": f"Palsu: {sp_score:.2f}", "hud_txt": hud_txt, "status": f"{sp_label} (Spoof: {sp_score:.2f})", "col": config.COLOR_RED})
                    continue 
                else: self.fake_frames = 0

                if self.stage != RegistrationStage.EXTRACTION and not self.in_ext:
                    self._record_data_buffers(face, pose) 
                    res = self.liveness.update_register(face, self.detector)
                    self._commit_stage_data(res["step"])
                    self.ui_state.update({"bbox": face.bbox, "instr": res.get("instruction", ""), "prog": res.get("progress",""), "hud_txt": hud_txt, "status": "VALIDATING", "col": config.COLOR_GREEN if res["step"] == "DONE" else config.COLOR_CYAN})
                elif not self.in_ext: 
                    self._process_extraction(raw, enhanced, face, pose, hud_txt, sp_score)

                self.latency = (time.time() - t_start) * 1000.0
        finally: 
            self.is_running = False

    def run(self):
        if self.stage == RegistrationStage.COMPLETE: return
        threading.Thread(target=self._process_thread, daemon=True).start()
        
        window_name = "Sistem Edge"
        
        # --- KEMBALI KE WINDOW AUTOSIZE SESUAI CONFIG ---
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        try:
            while self.is_running and self.stage != RegistrationStage.COMPLETE:
                ret, frame = self.cam.read()
                if ret:
                    with self.lock: self.shared_frame = frame.copy()
                    
                    display = frame.copy()
                    
                    st = self.ui_state
                    if st.get("face_obj"):
                        try: display = self.detector.draw(display, st["face_obj"])
                        except: pass
                    
                    display = cv2.flip(display, 1)
                    
                    if "overlay_msg" in st:
                        Helpers.show_msg(display, *st["overlay_msg"])
                    else:
                        Helpers.draw_hud(display, self.stage, st.get("instr",""), st.get("prog",""), st.get("hud_txt",""), st.get("status",""), st.get("bbox"), st.get("col", config.COLOR_CYAN))
                    
                    state.CURRENT_FRAME = display.copy() 
                    cv2.imshow(window_name, display)
                    
                if cv2.waitKey(1) & 0xFF == ord("q"): self.is_running = False; break
        finally: 
            self.is_running = False; time.sleep(0.5)
            self.cam.stop(); self.detector.close()
            state.MODE = "MAIN" 
            state.REG_NAMA = ""
            state.REG_NIM = ""

# ==============================================================================
# 6. MAIN EXECUTION
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
                full_identity = f"{state.REG_NIM} - {state.REG_NAMA}"
                app = FaceRegistrationApp(full_identity)
                app.run() 
            cv2.destroyAllWindows()
    except KeyboardInterrupt:
        if GPIO_AVAILABLE: GPIO.cleanup()
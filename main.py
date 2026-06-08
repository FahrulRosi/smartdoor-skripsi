import cv2, random, time, threading, sys, numpy as np
from datetime import datetime; from enum import Enum
import config
from camera.camera_stream import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from recognition.mobilefacenet import MobileFaceNet
from recognition.face_matcher import FaceMatcher
from door.door_lock import DoorLock
from liveness.head_pose import HeadPoseEstimator
from liveness.anti_spoofing import SilentAntiSpoofing  
from database.face_db import FaceDatabase

try: import RPi.GPIO as GPIO; GPIO_AVAILABLE = True
except ImportError: GPIO_AVAILABLE = False

class ValidationState(Enum): IDLE=0; RECOGNIZING=1; CHALLENGE=2; UNMATCHED=3; UNLOCKED=4

class UIHelper:
    @staticmethod
    def log(msg, lvl="INFO"): 
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [{lvl}] {msg}")

    @staticmethod
    def print_inline(msg):
        # Menggunakan ljust untuk memastikan teks sebelumnya tertimpa bersih
        sys.stdout.write(f"\r ⏳ {msg}".ljust(120))
        sys.stdout.flush()

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

class SmartDoorApp:
    CHALLENGES = {"BLINK": "Kedipkan Mata", "KANAN": "Toleh KANAN", "KIRI": "Toleh KIRI", "ATAS": "Dongak ATAS", "BAWAH": "Tunduk BAWAH"}

    def __init__(self):
        print("\n" + "="*50)
        UIHelper.log("SISTEM SMART DOOR LOCK AKTIF", "SYSTEM")
        print("="*50 + "\n")
        self.lock, self.running, self.shared_frame = threading.Lock(), True, None
        self.ui, self.missed_frames, self.spoof_score = {"wait": True, "bbox": None, "status": "", "color": config.COLOR_WHITE, "instr": ""}, 0, 1.0
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

    def _reset_state(self):
        self.state, self.last_name, self.match_score, self.auth_start = ValidationState.IDLE, "", 0.0, 0.0
        self.seq, self.step_idx, self.reg_pose, self.pose_hold, self.prev_center = [], 0, [0.0, 0.0, 0.0], 0, None
        self.current_stage_start, self.challenge_start_time = 0.0, 0.0
        self.wait_center, self.center_hold, self.blink_passed, self.blink_hold, self.ear_hist, self.print_counter, self.access_details = False, 0, False, 0, [], 0, []
        self.score_history = {}

    def _fail(self, status, color=config.COLOR_RED, instr="", wait=False):
        self._reset_state(); self.fake_frames = 0
        self.ui.update({"wait": wait, "bbox": None if wait else self.ui.get("bbox"), "status": status, "color": color, "instr": instr})

    def _check_action(self, action, face):
        if action == "BLINK": 
            ear = UIHelper.get_ear(face)
            self.ear_hist.append(ear)
            if len(self.ear_hist) > 3: self.ear_hist.pop(0)
            tgt_ear = getattr(config, 'BLINK_EAR_THRESHOLD', 0.21)
            
            if min(self.ear_hist) <= tgt_ear: self.blink_hold += 1
            else:
                if self.blink_hold >= 1: self.blink_passed = True
                self.blink_hold = 0
            
            # NORMALISASI KEDIPAN: Hanya menampilkan 0.0 (Belum) atau 1.0 (Sudah)
            val_display = 1.0 if self.blink_passed else 0.0
            return self.blink_passed, val_display, 1.0, False  

        p, ref = self.pose_estimator.estimate(face, self.detector), self.reg_pose
        dy, dp, dr = p.get("yaw", 0)-ref[0], p.get("pitch", 0)-ref[1], p.get("roll", 0)-ref[2]
        ty, tp, tr = getattr(config, 'CHALLENGE_YAW', 25.0), getattr(config, 'CHALLENGE_PITCH', 20.0), getattr(config, 'CHALLENGE_ROLL', 25.0)
        
        status_salah = False
        if action == "KANAN" and dy < -12.0: status_salah = True
        elif action == "KIRI" and dy > 12.0: status_salah = True
        elif action == "ATAS" and dp > 12.0: status_salah = True
        elif action == "BAWAH" and dp < -12.0: status_salah = True

        raw_val, tgt, passed = {"KANAN": (dy, ty, dy>ty), "KIRI": (-dy, ty, -dy>ty), "ATAS": (-dp, tp, -dp>tp), "BAWAH": (dp, tp, dp>tp)}.get(action, (0.0, 1.0, False))
        
        # NORMALISASI HEADPOSE: Mencegah nilai turun di bawah 0°
        val_display = max(0.0, float(raw_val))
        
        self.pose_hold = self.pose_hold + 1 if passed else 0
        return self.pose_hold >= 5, val_display, tgt, status_salah

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
        if self.state.value > 1 and self.prev_center and np.hypot(cx-self.prev_center[0], cy-self.prev_center[1]) > max(w, h)*0.40: 
            print("")
            UIHelper.log("⚠️ Peringatan: Wajah berganti di tengah proses! Reset ke awal.", "WARNING")
            return self._fail("WAJAH BERGANTI", instr="Mulai Ulang")
        self.prev_center = (cx, cy) 
        if h > int(config.FRAME_HEIGHT * 0.50): 
            return self._fail("TERLALU DEKAT", config.COLOR_YELLOW, "Mundur sedikit")
        
        sp = self.anti_spoof.is_real(raw, face.bbox)
        wajah_score = float(sp.get("score_real", sp.get("score", 0.0)))
        kertas_score = float(sp.get("score_photo", 0.0))
        layar_score = float(sp.get("score_video", 0.0))
        spoof_label = sp.get("label_name", "FOTO/VIDEO").upper()
        spoof_lat = float(sp.get("latency_ms", 0.0))
        
        self.spoof_score = wajah_score
        
        if not sp.get("real", True):
            self.fake_frames += 1 
            self.print_counter += 1
            if self.print_counter % 3 == 0: 
                UIHelper.print_inline(f"Deteksi Anti-Spoofing... Skor Liveness: {wajah_score:.2f} | Latensi AI: {spoof_lat:.1f}ms")

            if self.fake_frames >= 4: 
                print("") 
                UIHelper.log(f"❌ AKSES DITOLAK: Wajah Palsu Terdeteksi! (Skor: {wajah_score:.2f} | Tipe: {spoof_label} | Latensi: {spoof_lat:.1f}ms)", "ERROR")
                print("-" * 50)
                if hasattr(self.db, 'log_spoofing_async'): 
                    self.db.log_spoofing_async(wajah_score, kertas_score, layar_score, spoof_label, spoof_lat)
                return self._fail(f"PALSU ({wajah_score:.2f})", config.COLOR_RED, "Akses Ditolak")
            return
        
        self.fake_frames = 0
        
        gray_live = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        fh_l, fw_l = gray_live.shape
        x1_l, y1_l, x2_l, y2_l = max(0, x), max(0, y), min(fw_l, x+w), min(fh_l, y+h)
        
        mask_live = np.ones((fh_l, fw_l), dtype=bool)
        mask_live[y1_l:y2_l, x1_l:x2_l] = False
        L_bg_live = np.mean(gray_live[mask_live]) if np.any(mask_live) else 100.0

        if L_bg_live > 140: l_str = "Backlight"
        elif L_bg_live < 85: l_str = "Low Light"
        else: l_str = "Normal"

        if self.state in (ValidationState.IDLE, ValidationState.RECOGNIZING):
            if self.state == ValidationState.IDLE: 
                self.state = ValidationState.RECOGNIZING
                self.auth_start = time.time()
                self.current_stage_start = time.time()
                print(f"\n--- Sesi Autentikasi Baru Dimulai ---")
                UIHelper.log("Wajah terdeteksi, memulai identifikasi...", "INFO")
                return

            fh, fw = enhanced.shape[:2]
            if (raw_emb := self.model.get_embedding(self.model.crop_face(enhanced, [max(0, x), max(0, y), min(fw, x+w)-max(0, x), min(fh, y+h)-max(0, y)]))) is None: return
            emb = np.array(raw_emb, dtype=np.float32).flatten()
            match = self.matcher.match(emb / (np.linalg.norm(emb) + 1e-6))
            
            best_name, best_score = match.get("name", ""), match.get("score", 0.0)
            dyn_thr = getattr(config, 'MATCH_THRESHOLD', 0.48) if "Normal" in l_str else 0.40 
            
            if best_name:
                if best_name not in self.score_history:
                    self.score_history[best_name] = []
                self.score_history[best_name].append(best_score)
                if len(self.score_history[best_name]) > 5:
                    self.score_history[best_name].pop(0)
                smoothed_score = np.mean(self.score_history[best_name])
            else:
                smoothed_score = 0.0

            final_acc_live = min(100.0, max(0.0, 90.0 + ((smoothed_score - dyn_thr) / (1.0 - dyn_thr)) * 10.0)) if smoothed_score >= dyn_thr else 0.0
            
            self.print_counter += 1
            if self.print_counter % 3 == 0: 
                UIHelper.print_inline(f"Pencocokan... Nama: {best_name} | Skor: {smoothed_score:.4f} | Akurasi: {final_acc_live:.2f}% | Target: >= {dyn_thr}")
            
            if best_name and (smoothed_score >= dyn_thr):
                recog_duration = (time.time() - self.current_stage_start) * 1000
                self.access_details.append({"tahap": "RECOGNIZING", "latensi_ms": recog_duration})
                print("") 
                # MENAMPILKAN LATENSI VALIDASI DI SINI
                UIHelper.log(f"Wajah Berhasil Dikenali: {best_name} (Akurasi: {final_acc_live:.2f}% | Latensi Validasi: {recog_duration:.0f} ms)", "SUCCESS")
                
                self.last_name, self.match_score, self.state, self.step_idx, self.wait_center, self.center_hold = best_name, smoothed_score, ValidationState.CHALLENGE, 0, True, 0
                self.seq = [random.choice([k for k in self.CHALLENGES if k != "BLINK"]), "BLINK"]
                self.active_dyn_thr = dyn_thr
                
                # RESET WAKTU KE 0 SEBELUM MASUK TAHAP CHALLENGE
                self.current_stage_start = time.time()
                self.challenge_start_time = time.time() 
            else: 
                self.ui.update({"status": "TIDAK DIKENAL", "color": config.COLOR_RED, "instr": f"Cahaya: {l_str}"})

        elif self.state == ValidationState.CHALLENGE:
            curr = self.pose_estimator.estimate(face, self.detector)
            if self.wait_center:
                self.wait_center, self.center_hold, self.reg_pose = False, 0, [curr.get(k, 0) for k in ("yaw", "pitch", "roll")]
                self.challenge_start_time = time.time() # PASTIKAN WAKTU TER-RESET KE 0.0 SAAT POSISI DIKUNCI
                print("") 
                UIHelper.log("🎯 Posisi wajah dikunci. Memulai tantangan liveness...", "INFO")
                
            act, inst = self.seq[self.step_idx], self.CHALLENGES[self.seq[self.step_idx]]
            self.ui.update({"status": f"{self.last_name} ({l_str})", "color": config.COLOR_CYAN, "instr": f"Tahap {self.step_idx+1}/{len(self.seq)}: {inst}"})
            
            passed, val, tgt, status_salah = self._check_action(act, face)
            
            # PENGHITUNGAN WAKTU BERJALAN DARI 0.0
            waktu_berjalan = time.time() - self.challenge_start_time
            
            if status_salah:
                print("")
                UIHelper.log(f"❌ AKSES DITOLAK: Gerakan tidak sesuai target! Diminta '{inst}'.", "ERROR")
                print("-" * 50)
                return self._fail("GERAKAN SALAH", config.COLOR_RED, "Akses Ditolak", wait=True)

            if waktu_berjalan > 8.0:
                print("")
                UIHelper.log(f"❌ AKSES DITOLAK: Waktu habis! Anda tidak memenuhi target '{inst}'", "ERROR")
                print("-" * 50)
                return self._fail("WAKTU HABIS", config.COLOR_RED, "Mulai Ulang", wait=True)

            self.print_counter += 1
            if self.print_counter % 3 == 0:
                unit = "x" if act == "BLINK" else "°"
                # OUTPUT TERMINAL MENAMPILKAN SKOR AKTUAL & WAKTU YANG SELALU MULAI DARI 0
                UIHelper.print_inline(f"Tahap {self.step_idx+1}/{len(self.seq)} [{inst}] - Aktual: {val:.1f}{unit} | Target: {tgt:.1f}{unit} | Waktu: {waktu_berjalan:.1f}s / 8.0s")
            
            if passed:
                chal_duration = (time.time() - self.challenge_start_time) * 1000
                self.access_details.append({
                    "tantangan": inst, "skor_asli": val, "target": tgt, "latensi_ms": chal_duration
                })
                print("") 
                UIHelper.log(f"✅ Tantangan '{inst}' Lolos! ({chal_duration:.0f} ms)", "SUCCESS")
                
                self.step_idx, self.pose_hold, self.blink_hold, self.blink_passed = self.step_idx + 1, 0, 0, False; self.ear_hist.clear()
                
                if self.step_idx < len(self.seq): 
                    self.reg_pose, self.wait_center = [curr.get(k, 0) for k in ("yaw", "pitch", "roll")], False 
                    
                    # RESET SEMUA TIMER KE 0 LAGI UNTUK TANTANGAN BERIKUTNYA
                    self.current_stage_start = time.time()
                    self.challenge_start_time = time.time()
                else:
                    self.state = ValidationState.UNLOCKED
                    threading.Thread(target=self.door.unlock, daemon=True).start()
                    self._finalize_unlock(raw, face.bbox)

    def _finalize_unlock(self, raw, bbox):
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape
        x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(fw, bbox[0]+bbox[2]), min(fh, bbox[1]+bbox[3])
        
        mask = np.ones((fh, fw), dtype=bool)
        mask[y1:y2, x1:x2] = False
        L_bg = np.mean(gray[mask]) if np.any(mask) else 100.0

        if L_bg > 140: light_cond = "Backlight"
        elif L_bg < 85: light_cond = "Low Light"
        else: light_cond = "Normal"

        final_acc = min(100.0, max(0.0, 90.0 + ((self.match_score - getattr(self, 'active_dyn_thr', 0.48)) / (1.0 - getattr(self, 'active_dyn_thr', 0.48))) * 10.0))
        self.ui.update({"status": f"DIBUKA ({final_acc:.2f}%)", "color": config.COLOR_GREEN, "instr": ""})
        
        total_auth_time = (time.time() - self.auth_start) * 1000
        print("") 
        UIHelper.log(f"🔓 PINTU BERHASIL DIBUKA: {self.last_name} | Akurasi Akhir: {final_acc:.2f}%", "SUCCESS")
        UIHelper.log(f"⏱️ Total Durasi End-to-End: {total_auth_time:.0f} ms | Kondisi Cahaya: {light_cond}", "SYSTEM")
        print("="*50 + "\n")
        
        if hasattr(self.db, 'push_access_log_async'): 
            nim_val, name_val = "-", self.last_name
            if "_" in self.last_name:
                parts = self.last_name.split("_", 1)
                nim_val, name_val = parts[0], parts[1]
            elif " - " in self.last_name:
                parts = self.last_name.split(" - ", 1)
                nim_val, name_val = parts[0], parts[1]
            self.db.push_access_log_async(name_val, nim_val, "UNLOCKED", final_acc, light_cond, self.access_details, total_auth_time)

    def run(self):
        window_name = "Smart Door Lock"
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            while self.running:
                ret, frame = self.cam.read()
                if ret:
                    with self.lock: self.shared_frame = frame.copy()
                    display = cv2.flip(frame, 1)
                    is_door_locked = getattr(self.door, 'locked', True) if hasattr(self, 'door') and self.door else True
                    UIHelper.draw_ui(display, self.ui, is_door_locked)
                    cv2.imshow(window_name, display)
                if cv2.waitKey(10) & 0xFF in [ord("q"), ord("Q")]: self.running = False; break
        finally: 
            self.running = False
            if hasattr(self, 'cam') and self.cam: self.cam.stop()
            if hasattr(self, 'door') and self.door: self.door.cleanup()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SmartDoorApp()
    try: app.run()
    except KeyboardInterrupt: 
        if GPIO_AVAILABLE: GPIO.cleanup()
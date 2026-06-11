import cv2, os, time, threading, numpy as np
from enum import Enum
from datetime import datetime
import config
from camera.camera_stream       import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from liveness.liveness_manager  import LivenessManager
from recognition.mobilefacenet  import MobileFaceNet
from recognition.face_matcher   import FaceMatcher
from database.face_db           import FaceDatabase
from liveness.anti_spoofing     import SilentAntiSpoofing

GPIO_AVAILABLE = True
try: import RPi.GPIO as GPIO
except ImportError: GPIO_AVAILABLE = False

class RegistrationStage(Enum): IDLE=0; FACEMESH=1; YAW=2; PITCH=3; ROLL=4; BLINK=5; EXTRACTION=6; COMPLETE=7
STAGE_NAMES = {RegistrationStage.FACEMESH: "1. FaceMesh (3D)", RegistrationStage.YAW: "2a. Liveness (Yaw)", RegistrationStage.PITCH: "2b. Liveness (Pitch)", RegistrationStage.ROLL: "2c. Liveness (Roll)", RegistrationStage.BLINK: "3. Liveness (Blink)", RegistrationStage.EXTRACTION: "4. Ekstraksi Fitur"}
STEP_TO_STAGE = {"FACEMESH": RegistrationStage.FACEMESH, "YAW": RegistrationStage.YAW, "PITCH": RegistrationStage.PITCH, "ROLL": RegistrationStage.ROLL, "BLINK": RegistrationStage.BLINK, "DONE": RegistrationStage.EXTRACTION}

def _log(msg, level="INFO"): 
    lvl_str = f"[{level}]".ljust(10)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {lvl_str} {msg}")

class Helpers:
    @staticmethod
    def enhance_adaptive(frame, bbox=None, l_str="Normal"):
        if not getattr(config, 'ENABLE_CLAHE_ENHANCEMENT', True): return frame
        if not bbox: return frame
        
        w = bbox[2]
        if w > 180:    dist_cat = "DEKAT"
        elif w >= 100: dist_cat = "SEDANG"
        else:          dist_cat = "JAUH"
        
        matrix_clip = {
            "DEKAT":  {"Normal": 1.0, "Low Light": 1.3, "Backlight": 1.2},
            "SEDANG": {"Normal": 1.5, "Low Light": 2.0, "Backlight": 1.8},
            "JAUH":   {"Normal": 2.2, "Low Light": 2.5, "Backlight": 2.4}
        }
        clip_limit = matrix_clip[dist_cat].get(l_str, 1.5)
        
        d_val = 5 if dist_cat == "DEKAT" else 3
        denoised = cv2.bilateralFilter(frame, d=d_val, sigmaColor=30, sigmaSpace=30)
        img_yuv = cv2.cvtColor(denoised, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def clone_low_light(img):
        """Membuat kloning gambar redup secara digital"""
        low_light = cv2.convertScaleAbs(img, alpha=0.6, beta=-40)
        return cv2.GaussianBlur(low_light, (3, 3), 0)

    @staticmethod
    def clone_backlight(img):
        """Membuat kloning gambar backlight (silau dari belakang)"""
        backlight = cv2.convertScaleAbs(img, alpha=0.4, beta=-10)
        img_yuv = cv2.cvtColor(backlight, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    @staticmethod
    def capture_blink(face):
        if not getattr(face, 'landmarks', None) or len(face.landmarks) < 400: return None
        p = np.array([[face.landmarks[i].x, face.landmarks[i].y] for i in [33,160,158,133,153,144,362,385,387,263,373,380]])
        la = (np.linalg.norm(p[1]-p[5])+np.linalg.norm(p[2]-p[4]))/(2.0*np.linalg.norm(p[0]-p[3])+1e-6)
        ra = (np.linalg.norm(p[7]-p[11])+np.linalg.norm(p[8]-p[10]))/(2.0*np.linalg.norm(p[6]-p[9])+1e-6)
        return {"left_ear": la, "right_ear": ra, "avg_ear": (la+ra)/2.0}

    @staticmethod
    def get_light_condition(raw_frame, bbox):
        if not bbox: return "Normal"
        bx, by, bw, bh = bbox
        fh, fw = raw_frame.shape[:2]
        x1, y1, x2, y2 = max(0, bx), max(0, by), min(fw, bx + bw), min(fh, by + bh)
        gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
        
        L = np.mean(gray[y1:y2, x1:x2]) if gray[y1:y2, x1:x2].size > 0 else 100.0
        top_bg = gray[0:max(0, y1-10), max(0, x1-30):min(fw, x2+30)]
        L_bg_atas = np.mean(top_bg) if top_bg.size > 0 else L
        
        if (L_bg_atas - L) > 50 and L_bg_atas > 160 and L < 110: 
            return "Backlight"
        return "Low Light" if (L_bg_atas < 95 or L < 95) else "Normal"

    @staticmethod
    def is_image_quality_good(frame, bbox):
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(fw, x + w), min(fh, y + h)
        face_roi = frame[y1:y2, x1:x2]
        if face_roi.size == 0: return False, 0.0, 0.0
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return (50 < brightness < 200) and (blur_score > 80), brightness, blur_score

    @staticmethod
    def draw_hud(f, stg, instr, prog, score_txt, status, bbox, col):
        h, w = f.shape[:2]
        if bbox:
            bx, by, bw, bh = bbox
            bx = w - bx - bw
            cv2.rectangle(f, (bx, by), (bx+bw, by+bh), col, 2)
            lbl_h = 20
            lbl_y = max(lbl_h, by)
            cv2.rectangle(f, (bx, lbl_y - lbl_h), (bx + 110, lbl_y), col, -1)
            cv2.putText(f, status, (bx + 5, lbl_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
            
        cv2.rectangle(f, (0, 0), (w, 32), (25, 25, 25), -1)
        stage_txt = STAGE_NAMES.get(stg, "Proses...")
        cv2.putText(f, stage_txt, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, config.COLOR_GREEN, 1, cv2.LINE_AA)
        
        sv = min(stg.value, 6)
        bw_bar, bh_bar = 160, 14
        bx_bar, by_bar = w - bw_bar - 10, 9
        cv2.rectangle(f, (bx_bar, by_bar), (bx_bar+bw_bar, by_bar+bh_bar), (45, 45, 45), -1)
        if sv > 0:
            cv2.rectangle(f, (bx_bar, by_bar), (bx_bar + int(bw_bar*(sv-1)/6), by_bar+bh_bar), config.COLOR_GREEN, -1)
        cv2.rectangle(f, (bx_bar, by_bar), (bx_bar+bw_bar, by_bar+bh_bar), (255,255,255), 1)
        cv2.putText(f, f"Tahap {sv if sv<=5 else 6}/6", (bx_bar + 45, by_bar + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1, cv2.LINE_AA)

        cv2.rectangle(f, (0, h - 70), (w, h), (15, 15, 15), -1)
        cv2.line(f, (0, h - 70), (w, h - 70), config.COLOR_CYAN, 1)
        
        y_pos = h - 52
        if instr:
            cv2.putText(f, instr, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, config.COLOR_YELLOW, 1, cv2.LINE_AA)
            y_pos += 18
        if prog:
            cv2.putText(f, prog, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, config.COLOR_CYAN, 1, cv2.LINE_AA)
            y_pos += 16
        if score_txt:
            cv2.putText(f, score_txt, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.38, config.COLOR_WHITE, 1, cv2.LINE_AA)

    @staticmethod
    def show_msg(f, t_title, t_sub, col):
        h, w = f.shape[:2]
        cv2.rectangle(f, (0, 0), (w, h), (15, 15, 15), -1)
        cv2.rectangle(f, (10, 10), (w - 10, h - 10), col, 4)
        cv2.putText(f, t_title, (25, h // 2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2, cv2.LINE_AA)
        lines = t_sub.split(" | ")
        y_offset = h // 2 + 10
        for line in lines:
            cv2.putText(f, line, (25, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
            y_offset += 22

class FaceRegistrationApp:
    POSE_CFG = {RegistrationStage.YAW: ("yaw_snapshots", "yaw_left", "yaw_right", "yaw", getattr(config, 'YAW_THRESHOLD', 25.0)), RegistrationStage.PITCH: ("pitch_snapshots", "pitch_up", "pitch_down", "pitch", getattr(config, 'PITCH_THRESHOLD', 20.0)), RegistrationStage.ROLL: ("roll_snapshots", "roll_left", "roll_right", "roll", getattr(config, 'ROLL_THRESHOLD', 25.0))}
    
    def __init__(self, name):
        self.name, self.stage, self.in_ext, self.hold_frames, self.missed_frames = name, RegistrationStage.FACEMESH, False, 0, 0
        self.fake_frames = 0
        self.is_running, self.display_frame, self.frame_lock = True, None, threading.Lock()
        
        self.cap_data = {"facemesh_vector": None, "yaw_snapshots": [], "pitch_snapshots": [], "roll_snapshots": [], "blink_closed": None, "blink_open": None, "headpose_vector": None, "face_crops": []}
        self._pose_buf, self._blink_buf, self._prev_step = {"yaw": {}, "pitch": {}, "roll": {}}, {"closed": None, "open": None, "logged_closed": False, "logged_open": False}, "FACEMESH"
        
        self.db = FaceDatabase()
        check_nim = self.name.split(" - ")[0] if " - " in self.name else self.name
        if self.db.check_user_exists(check_nim): 
            _log(f"❌ User '{self.name}' sudah terdaftar!", "ERROR")
            self.stage = RegistrationStage.COMPLETE
            return

        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        self.detector = FaceMeshDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.liveness, self.model = LivenessManager(), MobileFaceNet()
        self.anti_spoof = SilentAntiSpoofing(getattr(config, 'ANTI_SPOOFING_MODEL', "liveness/antispoofing.onnx"), getattr(config, 'ANTI_SPOOFING_THRESHOLD', 0.70))
        self.matcher = FaceMatcher(0.35) 
        
        try:
            raw = self.db.load_all_faces()
            if raw:
                faces = {k: np.array(v.get('embedding', v.get('mobilefacenet_embedding')), dtype=np.float32) for k, v in raw.items() if isinstance(v, dict) and v.get('embedding') is not None}
                self.matcher.load_faces(faces) if hasattr(self.matcher, 'load_faces') else setattr(self.matcher, 'known_faces', faces)
        except Exception as e: pass
        
        self.liveness.start_register()
        self.action_start_time = None
        self._timer_started = False
        self.prev_instruction = None
        self.prev_tag = None
        self.individual_latencies = {}  
        _log(f"✅ Memulai Registrasi untuk {self.name}...", "SYSTEM")

    def _record_data_buffers(self, face, pose, enhanced):
        if not self.action_start_time: return
        
        if self.stage == RegistrationStage.FACEMESH and face.landmarks:
            if self.cap_data["facemesh_vector"] is None:
                self.hold_frames += 1
                
                if self.hold_frames % 3 == 0 and len(self.cap_data["face_crops"]) < 3:
                    crop = self.model.crop_face(enhanced, face.bbox)
                    if crop is not None and crop.size > 0:
                        idx = len(self.cap_data["face_crops"]) + 1
                        self.cap_data["face_crops"].append((f"Frontal_{idx}", crop))
                
                if self.hold_frames >= 9: 
                    lat_ms = round((time.time() - self.action_start_time) * 1000, 2)
                    self.cap_data["facemesh_vector"] = np.array([[l.x, l.y, l.z] for l in face.landmarks], dtype=np.float32).flatten()
                    
                    if not self.cap_data["face_crops"]:
                        crop = self.model.crop_face(enhanced, face.bbox)
                        if crop is not None and crop.size > 0: 
                            self.cap_data["face_crops"].append(("Frontal_Fallback", crop))

                    self.hold_frames = 0
                    self.individual_latencies["FaceMesh (3D)"] = lat_ms
                    _log(f"   -> [Wajah 3D Terekam]      | Latensi: {lat_ms:>8.2f} ms", "SUCCESS")
                    self.action_start_time = time.time() 

        if self.stage in self.POSE_CFG:
            _, t_neg, t_pos, axis, thr = self.POSE_CFG[self.stage]
            val = pose.get(axis, 0.0)
            buf = self._pose_buf[axis]
            if val < -thr or val > thr:
                tag = t_neg if val < -thr else t_pos
                if getattr(self, 'prev_tag', None) != tag:
                    self.hold_frames = 0
                    self.prev_tag = tag
                self.hold_frames += 1
                if self.hold_frames >= 6:
                    if tag not in buf:
                        lat_ms = round((time.time() - self.action_start_time) * 1000, 2)
                        buf[tag] = {k: float(pose.get(k, 0.0)) for k in ("yaw", "pitch", "roll")}
                        buf[tag]["tag"], buf[tag]["latency_ms"] = tag, lat_ms 
                        friendly_name = {"yaw_left": "Toleh Kiri", "yaw_right": "Toleh Kanan", "pitch_up": "Angguk Atas", "pitch_down": "Tunduk Bawah", "roll_left": "Miring Kiri", "roll_right": "Miring Kanan"}.get(tag, tag)
                        self.individual_latencies[friendly_name] = lat_ms
                        
                        _log(f"   -> [Berhasil {friendly_name:<12}] | Latensi: {lat_ms:>8.2f} ms", "SUCCESS")
                        self.action_start_time = time.time() 
                        self.hold_frames = 0
            else: 
                self.hold_frames = 0
                self.prev_tag = None

        if self.stage == RegistrationStage.BLINK:
            bv = Helpers.capture_blink(face)
            if bv:
                min_open, max_closed = getattr(config, 'MIN_BLINK_OPEN_EAR', 0.22), getattr(config, 'MAX_BLINK_CLOSED_EAR', 0.20)
                if bv["avg_ear"] <= max_closed and not self._blink_buf.get("logged_closed"):
                    lat_ms = round((time.time() - self.action_start_time) * 1000, 2)
                    bv["latency_ms"] = lat_ms
                    self._blink_buf["closed"], self._blink_buf["logged_closed"] = bv, True
                    self.individual_latencies["Mata Menutup"] = lat_ms
                    _log(f"   -> [Berhasil Kedip Tutup]  | Latensi: {lat_ms:>8.2f} ms", "SUCCESS")
                    self.action_start_time = time.time() 
                elif bv["avg_ear"] >= min_open and self._blink_buf.get("logged_closed") and not self._blink_buf.get("logged_open"):
                    lat_ms = round((time.time() - self.action_start_time) * 1000, 2)
                    bv["latency_ms"] = lat_ms
                    self._blink_buf["open"], self._blink_buf["logged_open"] = bv, True
                    self.individual_latencies["Mata Membuka"] = lat_ms
                    _log(f"   -> [Berhasil Kedip Buka]   | Latensi: {lat_ms:>8.2f} ms", "SUCCESS")
                    self.action_start_time = time.time() 
                if self._blink_buf.get("closed") and bv["avg_ear"] < self._blink_buf["closed"]["avg_ear"]: self._blink_buf["closed"].update(bv)
                if self._blink_buf.get("open") and bv["avg_ear"] > self._blink_buf["open"]["avg_ear"]: self._blink_buf["open"].update(bv)

    def _generate_metric_text(self, pose, ear_val, sp_score, light_cond):
        stg = self.stage
        hud_txt = {
            RegistrationStage.FACEMESH: f"Menganalisa 3D | Cahaya: {light_cond}",
            RegistrationStage.YAW: f"Aksi: Toleh Kanan/Kiri | Cahaya: {light_cond}",
            RegistrationStage.PITCH: f"Aksi: Angguk Atas/Bawah | Cahaya: {light_cond}",
            RegistrationStage.ROLL: f"Aksi: Miring Kanan/Kiri | Cahaya: {light_cond}",
            RegistrationStage.BLINK: f"Aksi: Kedipkan Mata | Cahaya: {light_cond}"
        }.get(stg, f"Tahan Lurus | {light_cond}")
        return hud_txt, f"{hud_txt} | Spf: {sp_score:.2f}"

    def _commit_stage_data(self, cur_step):
        if cur_step in ("WAIT", self._prev_step): return
        if self._prev_step in {"YAW", "PITCH", "ROLL"}:
            axis = {"YAW": "yaw", "PITCH": "pitch", "ROLL": "roll"}[self._prev_step]
            if len(self._pose_buf[axis]) < 2: self.liveness._register_step -= 1; return 
            self.cap_data[self.POSE_CFG[STEP_TO_STAGE[self._prev_step]][0]], self._pose_buf[axis] = list(self._pose_buf[axis].values()), {}  
            
        if cur_step == "DONE" and self._prev_step == "BLINK":
            bc, bo = self._blink_buf["closed"], self._blink_buf["open"]
            min_open, max_closed, min_delta = getattr(config, 'MIN_BLINK_OPEN_EAR', 0.22), getattr(config, 'MAX_BLINK_CLOSED_EAR', 0.20), getattr(config, 'MIN_BLINK_DELTA', 0.04)
            if not bc or not bo or (bo["avg_ear"] < min_open) or (bc["avg_ear"] > max_closed) or (bo["avg_ear"] - bc["avg_ear"] < min_delta): 
                self.liveness._register_step, self.liveness._blink_state, self.liveness._hold_frames, self.liveness._blink_count, self._blink_buf = 7, 0, 0, 0, {"closed": None, "open": None, "logged_closed": False, "logged_open": False}; return
            self.cap_data.update({"blink_closed": bc, "blink_open": bo})
        
        self._prev_step, self.stage = cur_step, STEP_TO_STAGE.get(cur_step, self.stage)
        
        if self.stage != RegistrationStage.COMPLETE:
            self.action_start_time = time.time()
            self.hold_frames = 0

    def _process_extraction(self, raw_frame, frame, face, display, pose, score_txt, sp_score, sp_label):
        missing = [k for k, v in [("FaceMesh", self.cap_data["facemesh_vector"] is not None), ("Yaw", len(self.cap_data["yaw_snapshots"])>1), ("Pitch", len(self.cap_data["pitch_snapshots"])>1), ("Roll", len(self.cap_data["roll_snapshots"])>1), ("Blink", self.cap_data["blink_closed"] is not None)] if not v]
        if missing: Helpers.show_msg(display, "❌ GAGAL!", f"Kurang: {','.join(missing)}", config.COLOR_RED); time.sleep(4); self.stage = RegistrationStage.COMPLETE; return

        quality_ok, brightness, blur_score = Helpers.is_image_quality_good(raw_frame, face.bbox)
        if not quality_ok:
            reason = "Cahaya Buruk" if not (50 < brightness < 200) else "Kamera Blur"
            Helpers.show_msg(display, "⚠️ KUALITAS GAMBAR BURUK", f"{reason} | Harap diam", config.COLOR_YELLOW)
            with self.frame_lock: self.display_frame = display.copy()
            time.sleep(1.5); return

        light_cond = Helpers.get_light_condition(raw_frame, face.bbox)
        latensi_respon_subjek = round(sum(self.individual_latencies.values()), 2)
        t_mfn_start = time.time()
        
        if not self.cap_data.get("face_crops"):
            crop_cadangan = self.model.crop_face(frame, face.bbox)
            if crop_cadangan is not None: self.cap_data["face_crops"].append(("Frontal Akhir", crop_cadangan))
            
        # -------------------------------------------------------------
        # MODIFIKASI SOLUSI A: KLONING DIGITAL MATRIKS 2D (3 VEKTOR)
        # -------------------------------------------------------------
        
        # 1. Ambil HANYA 1 gambar frontal asli terbaik (elemen pertama)
        _, crop_normal = self.cap_data["face_crops"][0]
        
        # 2. Buat Kloning Sintetis (Low Light & Backlight)
        crop_low_light = Helpers.clone_low_light(crop_normal)
        crop_backlight = Helpers.clone_backlight(crop_normal)
        
        # 3. Ekstraksi AI untuk ketiga kondisi
        emb_normal = self.model.get_embedding(crop_normal)
        emb_low_light = self.model.get_embedding(crop_low_light)
        emb_backlight = self.model.get_embedding(crop_backlight)
        
        if emb_normal is None or emb_low_light is None or emb_backlight is None:
            Helpers.show_msg(display, "❌ GAGAL!", "Ekstraksi AI Gagal", config.COLOR_RED)
            time.sleep(1.5); self.stage = RegistrationStage.COMPLETE; return
            
        # 4. Normalisasi Vektor Murni
        def norm_emb(e):
            e_arr = np.array(e).flatten()
            return (e_arr / (np.linalg.norm(e_arr) + 1e-6)).tolist()
            
        # 5. Gabungkan menjadi 1 Matriks (Array 2D berisi 3 Vektor)
        multi_master_embs = [norm_emb(emb_normal), norm_emb(emb_low_light), norm_emb(emb_backlight)]

        mfn_latency = round((time.time() - t_mfn_start) * 1000, 2)
        total_waktu_sistem = latensi_respon_subjek + mfn_latency
        
        nim_user, nama_user = self.name.split(" - ", 1) if " - " in self.name else ("0000", self.name)
        
        # Gunakan vektor normal untuk cek duplikat
        match = self.matcher.match(np.array(multi_master_embs[0], dtype=np.float32))
        anti_dup_thr = getattr(config, 'ANTI_DUPLICATE_THRESHOLD', 0.48)
        
        if match.get("name") and match.get("score", 0.0) >= anti_dup_thr and os.getenv("ALLOW_DUPLICATE", "false").lower() != "true": 
            Helpers.show_msg(display, "❌ WAJAH SUDAH TERDAFTAR!", f"User: {match['name']}", config.COLOR_RED)
        else:
            self.cap_data["reg_latency_ms"] = total_waktu_sistem
            self.cap_data["individual_latencies"] = self.individual_latencies
            self.cap_data.update({"headpose_vector": [float(pose["yaw"]), float(pose["pitch"]), float(pose["roll"])], "registration_accuracy": 100.0, "light_condition": light_cond})
            if "face_crops" in self.cap_data: del self.cap_data["face_crops"]
            
            # Simpan 3 Vektor Kloning ke Database
            success_master = self.db.save_face(nama_user, nim_user, multi_master_embs, self.cap_data)
            
            if success_master: 
                Helpers.show_msg(display, "✅ REGISTRASI BERHASIL!", f"User: {nama_user} | Multi-Vector", config.COLOR_GREEN)
                
                print("\n" + "="*65)
                print(" 🎉 REGISTRASI MULTI-VECTOR BERHASIL 🎉".center(65))
                print("="*65)
                print(f" 👤 Nama User             : {nama_user}")
                print(f" 💡 Kondisi Live          : {light_cond}")
                print(f" 🧠 Strategi              : Kloning Matriks 2D (Norm, Low, Back)")
                print("-" * 65)
                print(f" ⏱️ Latensi Respon Subjek : {latensi_respon_subjek:.2f} ms")
                print(f" ⏱️ Ekstraksi AI (3x)     : {mfn_latency:.2f} ms")
                print("="*65 + "\n")
            else:
                Helpers.show_msg(display, "❌ GAGAL!", "Database Error", config.COLOR_RED)
                
        with self.frame_lock: self.display_frame = display.copy()
        time.sleep(1.5); self.stage = RegistrationStage.COMPLETE 

    def _process_thread(self):
        try:
            bbox_memory = None
            while self.is_running and self.stage != RegistrationStage.COMPLETE:
                ret, frame = self.cam.read()
                if not ret: time.sleep(0.01); continue
                raw = frame.copy()
                display = raw.copy()
                
                faces = self.detector.detect(raw)
                
                if not faces: 
                    self.missed_frames += 1
                    if self.missed_frames >= 5: 
                        bbox_memory = None
                        display = cv2.flip(display, 1) 
                        Helpers.draw_hud(display, self.stage, "Hadapkan wajah", "", "", "NO FACE", None, config.COLOR_RED)
                        if self._timer_started: self._timer_started = False
                    elif bbox_memory: 
                        display = cv2.flip(display, 1) 
                        Helpers.draw_hud(display, self.stage, "Menganalisa...", "", "", "TRACKING", bbox_memory, config.COLOR_YELLOW)
                    with self.frame_lock: self.display_frame = display
                    continue
                
                self.missed_frames = 0
                face = faces[0]
                bbox_memory = face.bbox
                
                light_cond = Helpers.get_light_condition(raw, face.bbox)
                enhanced = Helpers.enhance_adaptive(raw, face.bbox, light_cond)
                
                if not self._timer_started:
                    self.action_start_time = time.time()
                    self._timer_started = True

                display = self.detector.draw(display, face)
                display = cv2.flip(display, 1)
                
                if face.bbox[3] > int(config.FRAME_HEIGHT * 0.50): 
                    Helpers.draw_hud(display, self.stage, "Wajah Terlalu Dekat!", "Mundur", "", "TOO CLOSE", face.bbox, config.COLOR_YELLOW)
                    with self.frame_lock: self.display_frame = display
                    continue
                
                pose = self.liveness.pose_estimator.estimate(face, self.detector)
                ear_val = (Helpers.capture_blink(face) or {}).get("avg_ear", 0.0)
                sp = self.anti_spoof.is_real(raw, face.bbox)
                sp_score, sp_real, sp_label = float(sp.get("score_real", sp.get("score", 1.0))), sp.get("real", True), sp.get("label_name", "FOTO").upper()
                
                hud_txt, term_txt = self._generate_metric_text(pose, ear_val, sp_score, light_cond)
                
                if not sp_real:
                    self.fake_frames += 1
                    if self.fake_frames >= 4:
                        Helpers.draw_hud(display, self.stage, "❌ DETEKSI SPOOFING!", f"Palsu: {sp_score:.2f}", hud_txt, f"{sp_label}", face.bbox, config.COLOR_RED)
                        with self.frame_lock: self.display_frame = display
                        continue 
                else: 
                    self.fake_frames = 0

                if self.stage != RegistrationStage.EXTRACTION and not self.in_ext:
                    self._record_data_buffers(face, pose, enhanced) 
                    res = self.liveness.update_register(face, self.detector)
                    if res["step"] != "WAIT" and res["step"] != self._prev_step:
                        if self.stage in self.POSE_CFG:
                            axis = self.POSE_CFG[self.stage][3]
                            if len(self._pose_buf[axis]) < 2: res["step"] = self._prev_step 
                                
                    self._commit_stage_data(res["step"])
                    instr = res.get("instruction", "")
                    if self.prev_instruction is None or instr != self.prev_instruction:
                        self.prev_instruction = instr
                        self.action_start_time = time.time()
                        self.hold_frames = 0
                    
                    hud_col = config.COLOR_GREEN if res["step"] == "DONE" else config.COLOR_CYAN
                    Helpers.draw_hud(display, self.stage, instr, res.get("progress",""), hud_txt, f"Real: {sp_score:.2f}", face.bbox, hud_col)
                elif not self.in_ext: 
                    self._process_extraction(raw, enhanced, face, display, pose, hud_txt, sp_score, sp_label)

                with self.frame_lock: self.display_frame = display
        finally: 
            self.is_running = False

    def run(self):
        if self.stage == RegistrationStage.COMPLETE: return
        threading.Thread(target=self._process_thread, daemon=True).start()
        try:
            cv2.namedWindow("Register", cv2.WINDOW_AUTOSIZE)
            while self.is_running and self.stage != RegistrationStage.COMPLETE:
                with self.frame_lock: frame = self.display_frame.copy() if self.display_frame is not None else None
                if frame is not None: cv2.imshow("Register", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"): self.is_running = False; break
        finally:
            self.is_running = False; time.sleep(0.5); self.cam.stop(); self.detector.close(); cv2.destroyAllWindows()

if __name__ == "__main__":
    print("\n" + "="*40)
    print("   SISTEM REGISTRASI WAJAH MULTI-VECTOR")
    print("="*40)
    nama_input = input("Masukan Nama : ").strip()
    if nama_input:
        nim_input = input("Masukan NIM  : ").strip()
        if nim_input: FaceRegistrationApp(f"{nim_input} - {nama_input}").run()
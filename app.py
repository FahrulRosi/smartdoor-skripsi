import cv2, time, threading, numpy as np, os
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# Import modul internal Anda (sesuaikan dengan struktur folder yang ada)
import config
from camera.camera_stream import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from recognition.mobilefacenet import MobileFaceNet
from recognition.face_matcher import FaceMatcher
from door.door_lock import DoorLock
from liveness.head_pose import HeadPoseEstimator
from liveness.anti_spoofing import SilentAntiSpoofing  
from database.face_db import FaceDatabase
from liveness.liveness_manager import LivenessManager

try: import RPi.GPIO as GPIO; GPIO_AVAILABLE = True
except ImportError: GPIO_AVAILABLE = False

# ==============================================================================
# 1. STATE MANAGER (Pengatur Mode Sistem & Berbagi Data dengan API)
# ==============================================================================
class SystemState:
    MODE = "MAIN"         # Bisa "MAIN" (Smart Door) atau "REGISTER"
    REG_NAMA = ""         # Menyimpan input Nama dari Web
    REG_NIM = ""          # Menyimpan input NIM dari Web
    REG_STATUS = "Standby"
    CURRENT_FRAME = None  # Buffer frame jika teman Anda butuh live view di Web

state = SystemState()

# Inisialisasi Flask API
app_api = Flask(__name__)
CORS(app_api)

# ==============================================================================
# 2. FLASK API (Dijalankan di Background Thread)
# ==============================================================================
# API ini akan ditembak oleh web teman Anda untuk memulai registrasi
@app_api.route('/api/trigger_register', methods=['POST'])
def trigger_register():
    data = request.json
    nama = data.get('nama', '').strip()
    nim = data.get('nim', '').strip()
    
    if nama and nim:
        state.REG_NAMA = nama
        state.REG_NIM = nim
        state.MODE = "REGISTER"  # <--- Ini Trigger Utamanya! Mengubah mode sistem.
        return jsonify({"status": "success", "message": f"Raspi beralih ke mode registrasi untuk {nama}."})
    return jsonify({"status": "error", "message": "Nama dan NIM tidak boleh kosong!"}), 400

# API untuk mengecek status saat ini (berguna untuk Web Admin)
@app_api.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "mode": state.MODE, 
        "reg_status": state.REG_STATUS,
        "nama": state.REG_NAMA,
        "nim": state.REG_NIM
    })

# API untuk streaming kamera ke Web Admin teman Anda
def generate_video_stream():
    while True:
        if state.CURRENT_FRAME is not None:
            ret, buffer = cv2.imencode('.jpg', state.CURRENT_FRAME, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.05)

@app_api.route('/api/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==============================================================================
# 3. KELAS SMART DOOR (Dari main.py)
# ==============================================================================
class SmartDoorApp:
    def __init__(self):
        self.running = True
        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        # Inisialisasi model (MobileFaceNet, Detector, DoorLock, dll) ada di sini
        
    def run(self):
        cv2.namedWindow("Layar Raspi", cv2.WINDOW_AUTOSIZE)
        print("[SYSTEM] Menjalankan Mode SMART DOOR...")
        
        while self.running:
            # PENTING: Jika ada perintah dari Web, keluar dari loop ini untuk ganti kamera!
            if state.MODE == "REGISTER":
                print("[SYSTEM] Menerima perintah pendaftaran. Menghentikan Smart Door sementara...")
                break
                
            ret, frame = self.cam.read()
            if ret:
                display = cv2.flip(frame, 1)
                
                # --- LOGIKA PENGENALAN WAJAH (MAIN.PY) ANDA MASUKKAN DI SINI ---
                cv2.putText(display, "MODE: SMART DOOR (STANDBY)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # ---------------------------------------------------------------
                
                state.CURRENT_FRAME = display.copy() # Bagikan ke Web
                cv2.imshow("Layar Raspi", display)
                
            if cv2.waitKey(10) & 0xFF in [ord("q"), ord("Q")]: 
                exit(0)
                
        # Matikan kamera sementara agar bisa dirilis untuk kelas Register
        self.running = False
        time.sleep(0.5)
        self.cam.stop()

# ==============================================================================
# 4. KELAS REGISTRASI WAJAH (Dari register.py)
# ==============================================================================
class FaceRegistrationApp:
    def __init__(self, nama, nim):
        self.name = nama
        self.nim = nim
        self.is_running = True
        self.cam = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
        # Inisialisasi LivenessManager, FaceMesh, dll ada di sini
        
    def run(self):
        cv2.namedWindow("Layar Raspi", cv2.WINDOW_AUTOSIZE)
        print(f"[SYSTEM] Memulai Pendaftaran Wajah: {self.name} ({self.nim})")
        
        start_time = time.time()
        
        while self.is_running:
            ret, frame = self.cam.read()
            if ret:
                display = cv2.flip(frame, 1)
                
                # --- LOGIKA LIVENESS & REGISTRASI (REGISTER.PY) MASUKKAN DI SINI ---
                cv2.rectangle(display, (0, 0), (config.FRAME_WIDTH, 70), (0, 0, 0), -1)
                cv2.putText(display, f"Registrasi: {self.name}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(display, f"NIM: {self.nim}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Dummy logika proses registrasi selesai (Ganti dengan Liveness aslinya)
                elapsed = time.time() - start_time
                if elapsed < 5:
                    status = "Tahap 1: Tatap Kamera"
                elif elapsed < 10:
                    status = "Tahap 2: Toleh Kanan/Kiri"
                else:
                    status = "Registrasi Berhasil!"
                    self.is_running = False # Stop Loop
                    
                state.REG_STATUS = status
                cv2.putText(display, status, (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # -------------------------------------------------------------------
                
                state.CURRENT_FRAME = display.copy() # Bagikan ke Web
                cv2.imshow("Layar Raspi", display)

            if cv2.waitKey(10) & 0xFF in [ord("q"), ord("Q")]: 
                exit(0)
            
            # Beri jeda 2 detik sebelum kembali ke main jika berhasil
            if not self.is_running:
                cv2.waitKey(2000)
                break
                
        # KEMBALI KE MODE SMART DOOR SETELAH SELESAI / GAGAL
        time.sleep(0.5)
        self.cam.stop()
        
        # Kembalikan state sistem ke semula
        state.MODE = "MAIN" 
        state.REG_NAMA = ""
        state.REG_NIM = ""
        state.REG_STATUS = "Standby"

# ==============================================================================
# 5. EXECUTION POINT
# ==============================================================================
def start_flask():
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR) # Agar terminal Raspi tidak penuh log flask
    # Menjalankan API di IP Lokal/Tailscale Port 5000
    app_api.run(host='0.0.0.0', port=5000, threaded=True)

if __name__ == "__main__":
    # 1. Jalankan API Listener untuk Teman Anda
    threading.Thread(target=start_flask, daemon=True).start()
    
    # 2. Main Event Loop untuk Layar Raspberry (GUI OpenCV)
    while True:
        if state.MODE == "MAIN":
            app = SmartDoorApp()
            app.run() # Akan berputar di main.py sampai dipanggil Web
            
        elif state.MODE == "REGISTER":
            app = FaceRegistrationApp(state.REG_NAMA, state.REG_NIM)
            app.run() # Akan berputar di register.py sampai registrasi selesai
            
        cv2.destroyAllWindows()
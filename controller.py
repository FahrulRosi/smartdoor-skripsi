import threading
import time
import cv2
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Mengimpor class dari file yang sudah ada tanpa perlu merubah isinya
from main import SmartDoorApp
from register import FaceRegistrationApp

# Inisialisasi API Server
app = FastAPI(title="Smart Door Lock Controller")

# Variabel global untuk mengontrol state antar thread
current_app = None
app_state = "MAIN"  # Status awal: "MAIN" (Smart Door Lock)
register_name = ""
is_transitioning = False  # Flag pengaman transisi hardware

# Schema data yang dikirim oleh backend
class RegisterRequest(BaseModel):
    name: str

@app.post("/api/trigger-register")
def trigger_register(req: RegisterRequest):
    global app_state, register_name, current_app, is_transitioning
    
    # Validasi State
    if app_state == "REGISTER":
        return {"status": "error", "message": "Sistem sudah dalam mode registrasi."}
    
    if is_transitioning:
        return {"status": "error", "message": "Kamera sedang dalam proses perpindahan. Harap tunggu."}
    
    # Kunci state agar tidak menerima request ganda
    is_transitioning = True
    register_name = req.name
    app_state = "REGISTER"
    
    # Hentikan loop kamera utama agar resource kamera terlepas dan daemon thread mati
    if current_app and hasattr(current_app, 'running'):
        current_app.running = False
        
    return {"status": "success", "message": f"Kamera dihentikan sementara. Beralih ke registrasi user: {register_name}"}

def cv2_app_runner():
    """
    Berjalan di Main Thread. Bertugas me-manage perpindahan aplikasi (Memory & Camera Release).
    """
    global current_app, app_state, register_name, is_transitioning
    
    while True:
        if app_state == "MAIN":
            print("\n[SYSTEM] Memulai Kamera Utama (Smart Door Lock)...")
            is_transitioning = False # Buka kunci API
            current_app = SmartDoorApp()
            
            # Program akan tertahan di sini mengeksekusi main loop UI
            current_app.run() 
            
            # Aplikasi utama dimatikan oleh API, jeda 1.5 detik untuk flush buffer kamera ke RAM
            time.sleep(1.5) 
            
        elif app_state == "REGISTER":
            print(f"\n[SYSTEM] Memulai Registrasi Wajah untuk: {register_name}...")
            is_transitioning = False # Buka kunci API
            current_app = FaceRegistrationApp(register_name)
            
            # Program akan tertahan di sini mengeksekusi registrasi sampai Selesai / Batal
            current_app.run() 
            
            print("\n[SYSTEM] Registrasi Selesai/Dihentikan. Mengembalikan ke Kamera Utama...")
            
            # Set up pengembalian ke UI Utama
            app_state = "MAIN"
            register_name = ""
            is_transitioning = True # Kunci API selama transisi balik
            
            # Jeda hardware flush (Penting untuk Raspberry Pi / Edge Device)
            time.sleep(1.5) 

def start_api_server():
    """
    Berjalan di Background Thread agar tidak nge-block loop UI OpenCV.
    """
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

if __name__ == "__main__":
    print("======================================================")
    print("  SISTEM PENGENDALI PINTU & REGISTRASI WAJAH")
    print("======================================================")
    
    # 1. Jalankan API Server di Background Thread
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    print("[INFO] Endpoint Web Server siap di POST http://0.0.0.0:8000/api/trigger-register")
    time.sleep(1.0) # Jeda agar pesan Uvicorn tidak bertabrakan dengan Print UI
    
    # 2. Jalankan eksekusi OpenCV App di Main Thread
    try:
        cv2_app_runner()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Mematikan seluruh sistem dari Terminal...")
        if current_app and hasattr(current_app, 'running'):
            current_app.running = False
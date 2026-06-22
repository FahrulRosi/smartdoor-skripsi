import threading
import time
import cv2
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from main import SmartDoorApp
from register import FaceRegistrationApp

app = FastAPI(title="Smart Door Lock Controller")

current_app = None
app_state = "MAIN"  
register_name = ""
is_transitioning = False  
class RegisterRequest(BaseModel):
    name: str

@app.post("/api/trigger-register")
def trigger_register(req: RegisterRequest):
    global app_state, register_name, current_app, is_transitioning

    if app_state == "REGISTER":
        return {"status": "error", "message": "Sistem sudah dalam mode registrasi."}
    
    if is_transitioning:
        return {"status": "error", "message": "Kamera sedang dalam proses perpindahan. Harap tunggu."}

    is_transitioning = True
    register_name = req.name
    app_state = "REGISTER"

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
            is_transitioning = False 
            current_app = SmartDoorApp()

            current_app.run() 

            time.sleep(1.5) 
            
        elif app_state == "REGISTER":
            print(f"\n[SYSTEM] Memulai Registrasi Wajah untuk: {register_name}...")
            is_transitioning = False 
            current_app = FaceRegistrationApp(register_name)
        
            current_app.run() 
            
            print("\n[SYSTEM] Registrasi Selesai/Dihentikan. Mengembalikan ke Kamera Utama...")

            app_state = "MAIN"
            register_name = ""
            is_transitioning = True 
            
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
    
    api_thread = threading.Thread(target=start_api_server, daemon=True)
    api_thread.start()
    print("[INFO] Endpoint Web Server siap di POST http://0.0.0.0:8000/api/trigger-register")
    time.sleep(1.0) 

    try:
        cv2_app_runner()
    except KeyboardInterrupt:
        print("\n[SYSTEM] Mematikan seluruh sistem dari Terminal...")
        if current_app and hasattr(current_app, 'running'):
            current_app.running = False
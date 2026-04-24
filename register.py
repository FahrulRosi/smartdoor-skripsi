import sys
import cv2
import config
from camera.camera_stream       import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from liveness.liveness_manager  import LivenessManager
from recognition.mobilefacenet  import MobileFaceNet
from database.face_db           import save_face
from liveness.anti_spoofing     import SilentAntiSpoofing

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

def run_register(name: str):
    # Setup IR-CUT (Pin 12)
    IR_CUT_PIN = 12
    if GPIO_AVAILABLE:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(IR_CUT_PIN, GPIO.OUT)
        GPIO.output(IR_CUT_PIN, GPIO.HIGH)

    cam      = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
    detector = FaceMeshDetector()
    liveness = LivenessManager()
    model    = MobileFaceNet()
    spoof_ai = SilentAntiSpoofing() # Pastikan file ini ada di folder liveness

    liveness.start_register()
    saved = False

    try:
        while True:
            ret, frame = cam.read()
            if not ret: continue
            display = frame.copy()
            faces = detector.detect(frame)

            if not faces:
                cv2.putText(display, "Mencari Wajah...", (10, 30), 1, 1, (0, 0, 255), 2)
            else:
                face = faces[0]
                # 1. Cek Anti-Spoofing Pasif (Cek Foto/Layar)
                is_real = spoof_ai.is_real(frame, face.bbox)
                
                if not is_real["real"]:
                    cv2.putText(display, "SPOOFING DETECTED! GUNAKAN WAJAH ASLI", (10, 30), 1, 1, (0, 0, 255), 2)
                else:
                    # 2. Jalankan Alur Bertahap (State Machine)
                    res = liveness.update_register(face, detector)
                    
                    cv2.putText(display, f"Langkah: {res['step']}", (10, 30), 1, 1, (0, 255, 0), 2)
                    cv2.putText(display, res["instruction"], (10, 60), 1, 1, (255, 255, 255), 2)

                    if res["status"] == "complete" and not saved:
                        # 3. Ekstraksi MobileFaceNet
                        crop = model.crop_face(frame, face.bbox)
                        emb  = model.get_embedding(crop)
                        save_face(name, emb)
                        saved = True
                        print(f"Berhasil meregistrasi {name}")
                        break

            cv2.imshow("Register", display)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    finally:
        if GPIO_AVAILABLE: GPIO.cleanup()
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1: run_register(sys.argv[1])
    else: print("Gunakan: python3 register.py <nama>")
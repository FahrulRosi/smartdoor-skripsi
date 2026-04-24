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

def _put(frame, text, y, color=config.COLOR_WHITE):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, color, config.LINE_TYPE)

def run_register(name):
    # Setup IR-CUT
    IR_CUT_PIN = 12
    if GPIO_AVAILABLE:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(IR_CUT_PIN, GPIO.OUT)
        GPIO.output(IR_CUT_PIN, GPIO.HIGH) # Aktifkan filter IR
        print("[IR-CUT] Filter Diaktifkan (Mode Siang)")

    cam      = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
    detector = FaceMeshDetector(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    liveness      = LivenessManager()
    model         = MobileFaceNet()
    anti_spoofing = SilentAntiSpoofing()

    liveness.start_register()
    embedding_saved = False

    try:
        while True:
            ret, frame = cam.read()
            if not ret: continue
            display = frame.copy()
            faces = detector.detect(frame)

            if not faces:
                _put(display, "Wajah tidak terdeteksi", 30, config.COLOR_RED)
                cv2.imshow("Register", display)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
                continue

            face = faces[0]
            cv2.rectangle(display, (face.bbox[0], face.bbox[1]), (face.bbox[0]+face.bbox[2], face.bbox[1]+face.bbox[3]), (255, 255, 0), 2)

            # 1. Cek Anti-Spoofing
            spoof_check = anti_spoofing.is_real(frame, face.bbox)
            if not spoof_check["real"]:
                _put(display, f"REGISTER: {name}", 30, config.COLOR_YELLOW)
                _put(display, "SPOOFING DETECTED! GUNAKAN WAJAH ASLI", 60, config.COLOR_RED)
            else:
                # 2. Alur Bertahap (State Machine)
                result = liveness.update_register(face, detector)
                _put(display, f"REGISTER: {name} (Real)", 30, config.COLOR_GREEN)
                _put(display, f"Langkah: {result.get('step', '')}", 60, config.COLOR_CYAN)
                _put(display, f"Instruksi: {result.get('instruction', '')}", 90, config.COLOR_WHITE)

                # 3. Selesai -> Ekstraksi MobileFaceNet
                if result["status"] == "complete" and not embedding_saved:
                    _put(display, "Valid! Mengekstraksi Wajah...", 130, config.COLOR_GREEN)
                    cv2.imshow("Register", display)
                    cv2.waitKey(500)
                    
                    face_crop = model.crop_face(frame, face.bbox)
                    embedding = model.get_embedding(face_crop)
                    save_face(name, embedding)
                    embedding_saved = True
                    _put(display, "Berhasil! Keluar dalam 3 detik...", 160, config.COLOR_GREEN)
                    cv2.imshow("Register", display)
                    cv2.waitKey(3000)
                    break

            cv2.imshow("Register", display)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    finally:
        if GPIO_AVAILABLE: GPIO.cleanup()
        cam.stop()
        detector.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Gunakan: python3 register.py <nama>")
    else:
        run_register(sys.argv[1].strip())
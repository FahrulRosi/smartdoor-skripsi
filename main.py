import sys
import cv2
import config
from camera.camera_stream       import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from recognition.mobilefacenet  import MobileFaceNet
from recognition.face_matcher   import FaceMatcher
from door.door_lock             import DoorLock
from liveness.anti_spoofing     import SilentAntiSpoofing

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

def _put(frame, text, y, color=config.COLOR_WHITE, scale=0.7):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

def run_unlock():
    IR_CUT_PIN = 12
    if GPIO_AVAILABLE:
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(IR_CUT_PIN, GPIO.OUT)
        GPIO.output(IR_CUT_PIN, GPIO.HIGH)

    cam      = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
    detector = FaceMeshDetector(min_detection_confidence=0.6, min_tracking_confidence=0.6)
    anti_spoofing = SilentAntiSpoofing()
    model         = MobileFaceNet()
    matcher       = FaceMatcher(threshold=config.MATCH_THRESHOLD)
    door          = DoorLock(pin=config.LOCK_GPIO_PIN, unlock_duration=config.UNLOCK_DURATION)

    hold_timer = 0

    try:
        while True:
            ret, frame = cam.read()
            if not ret: continue
            display = frame.copy()
            
            _put(display, f"Pintu: {door.status()}", 25, (0, 255, 0) if not door.locked else (0, 0, 255))

            if hold_timer > 0:
                hold_timer -= 1
                cv2.imshow("Smart Door Lock", display)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
                continue

            faces = detector.detect(frame)
            if not faces:
                _put(display, "Mencari Wajah...", 60, (0, 255, 255))
                cv2.imshow("Smart Door Lock", display)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
                continue

            face = faces[0]
            # 1. Silent Anti-Spoofing (Cek Foto/Video)
            liveness = anti_spoofing.is_real(frame, face.bbox)
            
            if not liveness["real"]:
                _put(display, "AKSES DITOLAK: FOTO DETECTED!", 90, (0, 0, 255))
                hold_timer = 60
            else:
                # 2. Face Recognition
                _put(display, "Liveness OK. Memindai...", 90, (0, 255, 0))
                face_crop = model.crop_face(frame, face.bbox)
                embedding = model.get_embedding(face_crop)
                match = matcher.match(embedding)

                if match["matched"]:
                    _put(display, f"Selamat Datang, {match['name']}!", 120, (0, 255, 0))
                    door.unlock()
                    hold_timer = 120
                else:
                    _put(display, "Wajah Tidak Dikenali", 120, (0, 0, 255))
                    hold_timer = 60

            cv2.imshow("Smart Door Lock", display)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    finally:
        if GPIO_AVAILABLE: GPIO.cleanup()
        cam.stop()
        detector.close()
        door.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_unlock()
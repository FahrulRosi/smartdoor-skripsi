import cv2
import config
from camera.camera_stream       import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from recognition.mobilefacenet  import MobileFaceNet
from recognition.face_matcher   import FaceMatcher
from door.door_lock             import DoorLock
from liveness.anti_spoofing     import SilentAntiSpoofing

def run_unlock():
    cam      = CameraStream(config.CAMERA_INDEX).start()
    detector = FaceMeshDetector()
    spoof_ai = SilentAntiSpoofing()
    model    = MobileFaceNet()
    matcher  = FaceMatcher(threshold=config.MATCH_THRESHOLD)
    door     = DoorLock(pin=config.LOCK_GPIO_PIN)

    try:
        while True:
            ret, frame = cam.read()
            if not ret: continue
            display = frame.copy()
            faces = detector.detect(frame)

            if faces:
                face = faces[0]
                # Anti-Spoofing + Recognition (Tanpa Challenge Gerak)
                if spoof_ai.is_real(frame, face.bbox)["real"]:
                    crop = model.crop_face(frame, face.bbox)
                    match = matcher.match(model.get_embedding(crop))
                    
                    if match["matched"]:
                        cv2.putText(display, f"Akses Diterima: {match['name']}", (10, 30), 1, 1, (0, 255, 0), 2)
                        door.unlock()
                    else:
                        cv2.putText(display, "Wajah Tidak Dikenali", (10, 30), 1, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(display, "FOTO DETECTED!", (10, 30), 1, 1, (0, 0, 255), 2)

            cv2.imshow("Unlock", display)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_unlock()
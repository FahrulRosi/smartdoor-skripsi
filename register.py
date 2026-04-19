import sys
import cv2

import config
from camera.camera_stream     import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from liveness.liveness_manager  import LivenessManager
from recognition.mobilefacenet  import MobileFaceNet
from database.face_db           import save_face


# ── HUD helpers ───────────────────────────────────────────────────────────────
def _put(frame, text: str, y: int, color=config.COLOR_WHITE):
    cv2.putText(frame, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                color, config.LINE_TYPE)


def _draw_bbox(frame, bbox, color):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


# ─────────────────────────────────────────────────────────────────────────────
def run_register(name: str):
    print(f"\n[REGISTER] Starting registration for: '{name}'")
    print("  Instructions: face the camera, blink twice.")
    print("  Press  Q  to quit.\n")

    cam      = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
    detector = FaceMeshDetector(
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    liveness = LivenessManager()
    model    = MobileFaceNet()

    liveness.start_register()
    embedding_saved = False

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            display = frame.copy()
            faces   = detector.detect(frame)

            # ── No face ──────────────────────────────────────────────── #
            if not faces:
                _put(display, "No face detected", 30, config.COLOR_RED)
                cv2.imshow("Register", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            face = faces[0]
            _draw_bbox(display, face.bbox, config.COLOR_CYAN)

            # ── Liveness update ──────────────────────────────────────── #
            state = liveness.update_register(face, detector)

            # HUD
            _put(display, f"REGISTER: {name}", 30, config.COLOR_YELLOW)
            _put(display, state.summary(),      60, config.COLOR_WHITE)
            _put(display,
                 f"Blink needed: {max(0, config.REGISTER_BLINK_COUNT - (2 if state.blink_ok else 0))}",
                 90, config.COLOR_WHITE)

            # ── Liveness passed ──────────────────────────────────────── #
            if state.all_passed and not embedding_saved:
                _put(display, "✅ Liveness OK! Capturing…", 130, config.COLOR_GREEN)
                cv2.imshow("Register", display)
                cv2.waitKey(500)

                # Crop face and extract embedding
                face_crop = model.crop_face(frame, face.bbox)
                embedding = model.get_embedding(face_crop)

                save_face(name, embedding)
                print(f"[REGISTER] ✅ '{name}' registered successfully!")
                embedding_saved = True

                _put(display, f"'{name}' saved! Press Q to exit.", 160, config.COLOR_GREEN)
                cv2.imshow("Register", display)
                cv2.waitKey(3000)
                break

            cv2.imshow("Register", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[REGISTER] Aborted by user.")
                break

    finally:
        cam.stop()
        detector.close()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:  python register.py <name>")
        sys.exit(1)
    run_register(sys.argv[1].strip())
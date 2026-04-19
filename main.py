import sys
import cv2

import config
from camera.camera_stream       import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from liveness.liveness_manager  import LivenessManager
from recognition.mobilefacenet  import MobileFaceNet
from recognition.face_matcher   import FaceMatcher
from door.door_lock             import DoorLock


# ── HUD helpers ───────────────────────────────────────────────────────────────
def _put(frame, text: str, y: int, color=config.COLOR_WHITE, scale: float | None = None):
    cv2.putText(frame, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale or config.FONT_SCALE,
                color, config.LINE_TYPE)


def _draw_bbox(frame, bbox, color, thickness: int = 2):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)


# ─── States ───────────────────────────────────────────────────────────────────
class UnlockState:
    IDLE       = "idle"
    CHALLENGE  = "challenge"
    RECOGNIZING= "recognizing"
    UNLOCKED   = "unlocked"
    DENIED     = "denied"


# ─────────────────────────────────────────────────────────────────────────────
def run_unlock():
    print("\n[UNLOCK] Smart door lock started. Press Q to quit.")

    cam      = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
    detector = FaceMeshDetector(
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    liveness = LivenessManager()
    model    = MobileFaceNet()
    matcher  = FaceMatcher(threshold=config.MATCH_THRESHOLD)
    door     = DoorLock(pin=config.LOCK_GPIO_PIN, unlock_duration=config.UNLOCK_DURATION)

    state      = UnlockState.IDLE
    hold_timer = 0    # frames to hold the result screen

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            display = frame.copy()
            faces   = detector.detect(frame)

            # ── Door status bar ──────────────────────────────────────── #
            status_color = config.COLOR_GREEN if not door.locked else config.COLOR_RED
            _put(display, f"Door: {door.status()}", 25, status_color, scale=0.8)

            # ── Hold UNLOCKED / DENIED screen ────────────────────────── #
            if hold_timer > 0:
                hold_timer -= 1
                cv2.imshow("Smart Door Lock", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if hold_timer == 0:
                    # Reset for next attempt
                    state = UnlockState.IDLE
                continue

            # ── No face → stay idle ───────────────────────────────────── #
            if not faces:
                _put(display, "Waiting for face…", 60, config.COLOR_YELLOW)
                state = UnlockState.IDLE
                cv2.imshow("Smart Door Lock", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            face = faces[0]
            _draw_bbox(display, face.bbox,
                       config.COLOR_GREEN if state == UnlockState.CHALLENGE
                       else config.COLOR_CYAN)

            # ══════════════════════════════════════════════════════════ #
            #  IDLE → start challenge
            # ══════════════════════════════════════════════════════════ #
            if state == UnlockState.IDLE:
                challenges = liveness.start_unlock()
                labels = ", ".join(c.value for c in challenges)
                print(f"[UNLOCK] Challenge sequence: {labels}")
                state = UnlockState.CHALLENGE

            # ══════════════════════════════════════════════════════════ #
            #  CHALLENGE
            # ══════════════════════════════════════════════════════════ #
            if state == UnlockState.CHALLENGE:
                result = liveness.update_unlock(face, detector)

                if result["status"] == "timeout":
                    _put(display, "⏰ Timeout! Try again.", 60, config.COLOR_RED)
                    print("[UNLOCK] Challenge timed out.")
                    state      = UnlockState.DENIED
                    hold_timer = 60
                    door.lock()

                elif result["status"] == "complete":
                    _put(display, "✅ Challenge passed!", 60, config.COLOR_GREEN)
                    print("[UNLOCK] All challenges passed. Running recognition…")
                    state = UnlockState.RECOGNIZING

                else:
                    _put(display, result.get("label",   ""), 60,  config.COLOR_YELLOW)
                    _put(display, result.get("progress",""), 90,  config.COLOR_WHITE)
                    _put(display,
                         f"Time left: {result.get('remaining', 0):.1f}s",
                         120, config.COLOR_WHITE)

            # ══════════════════════════════════════════════════════════ #
            #  RECOGNITION
            # ══════════════════════════════════════════════════════════ #
            if state == UnlockState.RECOGNIZING:
                face_crop = model.crop_face(frame, face.bbox)
                embedding = model.get_embedding(face_crop)
                match     = matcher.match(embedding)

                if match["matched"]:
                    name  = match["name"]
                    score = match["score"]
                    _put(display, f"✅ Welcome, {name}!", 60, config.COLOR_GREEN, scale=0.9)
                    _put(display, f"Score: {score:.4f}",  90, config.COLOR_GREEN)
                    print(f"[UNLOCK] 🔓 MATCH → {name}  (score={score:.4f})")
                    door.unlock()
                    state      = UnlockState.UNLOCKED
                    hold_timer = 90
                else:
                    _put(display, "❌ Face not recognized", 60, config.COLOR_RED, scale=0.9)
                    _put(display, match["reason"],           90, config.COLOR_RED)
                    print(f"[UNLOCK] ❌ DENY  ({match['reason']})")
                    door.lock()
                    state      = UnlockState.DENIED
                    hold_timer = 90

            cv2.imshow("Smart Door Lock", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cam.stop()
        detector.close()
        door.cleanup()
        cv2.destroyAllWindows()
        print("[UNLOCK] Session ended.")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "register":
        from register import run_register
        name = input("Enter name to register: ").strip()
        if not name:
            print("Name cannot be empty.")
            sys.exit(1)
        run_register(name)
    else:
        run_unlock()
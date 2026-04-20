import sys
import cv2

import config
from camera.camera_stream       import CameraStream
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
    print(f"\n[REGISTER] Memulai registrasi untuk: '{name}'")
    print("  Instruksi: Ikuti arahan di layar secara bertahap.")
    print("  Tekan  Q  untuk keluar.\n")

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
            
            # ── Langkah 1: MediaPipe FaceMesh ────────────────────────── #
            faces = detector.detect(frame)

            if not faces:
                _put(display, "Wajah tidak terdeteksi", 30, config.COLOR_RED)
                cv2.imshow("Register", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            face = faces[0]
            _draw_bbox(display, face.bbox, config.COLOR_CYAN)

            # ── Langkah 2 s/d 5: Liveness Update (Yaw -> Pitch -> Roll -> Blink) ── #
            result = liveness.update_register(face, detector)

            # HUD Informasi
            _put(display, f"REGISTER: {name}", 30, config.COLOR_YELLOW)
            _put(display, f"Langkah: {result.get('step', '')}", 60, config.COLOR_CYAN)
            _put(display, f"Instruksi: {result.get('instruction', '')}", 90, config.COLOR_WHITE)

            # ── Langkah 6: Validasi Liveness Selesai -> Ekstraksi MobileFaceNet ── #
            if result["status"] == "complete" and not embedding_saved:
                _put(display, "✅ Liveness OK! Mengekstraksi Wajah...", 130, config.COLOR_GREEN)
                cv2.imshow("Register", display)
                cv2.waitKey(500)  # Tahan frame sebentar agar user lihat status berhasil

                # Crop wajah dan masukkan ke MobileFaceNet
                face_crop = model.crop_face(frame, face.bbox)
                embedding = model.get_embedding(face_crop)

                # Simpan embedding ke .pkl
                save_face(name, embedding)
                print(f"[REGISTER] ✅ '{name}' berhasil didaftarkan!")
                embedding_saved = True

                _put(display, f"'{name}' tersimpan! Keluar dalam 3 detik...", 160, config.COLOR_GREEN)
                cv2.imshow("Register", display)
                cv2.waitKey(3000)
                break

            cv2.imshow("Register", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[REGISTER] Dibatalkan oleh pengguna.")
                break

    finally:
        cam.stop()
        detector.close()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Input nama dilakukan terlebih dahulu (Step awal)
    if len(sys.argv) < 2:
        print("Penggunaan:  python register.py <nama>")
        sys.exit(1)
    run_register(sys.argv[1].strip())
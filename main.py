import sys
import cv2
import time

import config
from camera.camera_stream       import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from recognition.mobilefacenet  import MobileFaceNet
from recognition.face_matcher   import FaceMatcher
from door.door_lock             import DoorLock
# Pastikan Anda sudah membuat file anti_spoofing.py di folder liveness
from liveness.anti_spoofing      import SilentAntiSpoofing 

# ── HUD Helpers (Tampilan Antarmuka) ──────────────────────────────────────────
def _put(frame, text: str, y: int, color=config.COLOR_WHITE, scale: float | None = None):
    """Menampilkan teks pada layar"""
    cv2.putText(frame, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale or config.FONT_SCALE,
                color, config.LINE_TYPE)

def _draw_bbox(frame, bbox, color, thickness: int = 2):
    """Menggambar kotak pada wajah yang terdeteksi"""
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

# ── State Machine ────────────────────────────────────────────────────────────
class UnlockState:
    IDLE        = "idle"
    VERIFYING   = "verifying"
    UNLOCKED    = "unlocked"
    DENIED      = "denied"

# ── Main Logic ───────────────────────────────────────────────────────────────
def run_unlock():
    print("\n[SYSTEM] Smart Door Lock Aktif (Face Recognition + Silent Anti-Spoofing).")
    print("[SYSTEM] Tekan 'Q' pada jendela kamera untuk keluar.")

    # Inisialisasi Komponen
    cam      = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
    detector = FaceMeshDetector(
        min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    
    # Model AI untuk Deteksi Foto/Video (Silent Anti-Spoofing)
    anti_spoofing = SilentAntiSpoofing(model_path="liveness/minifasnet.onnx", threshold=0.85)
    
    model    = MobileFaceNet()
    matcher  = FaceMatcher(threshold=config.MATCH_THRESHOLD)
    door     = DoorLock(pin=config.LOCK_GPIO_PIN, unlock_duration=config.UNLOCK_DURATION)

    state      = UnlockState.IDLE
    hold_timer = 0    # Durasi menahan tampilan status (dalam frame)

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                continue

            display = frame.copy()
            faces   = detector.detect(frame)

            # ── Status Bar Pintu ────────────────────────────────────── #
            status_color = config.COLOR_GREEN if not door.locked else config.COLOR_RED
            _put(display, f"Pintu: {door.status()}", 25, status_color, scale=0.8)

            # ── Logika Menahan Status (Success/Fail) ─────────────────── #
            if hold_timer > 0:
                hold_timer -= 1
                cv2.imshow("Smart Door Lock", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if hold_timer == 0:
                    state = UnlockState.IDLE
                continue

            # ── Cek Kehadiran Wajah ──────────────────────────────────── #
            if not faces:
                _put(display, "Menunggu wajah...", 60, config.COLOR_YELLOW)
                state = UnlockState.IDLE
                cv2.imshow("Smart Door Lock", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            face = faces[0]
            _draw_bbox(display, face.bbox, config.COLOR_CYAN)

            # ══════════════════════════════════════════════════════════ #
            #  PROSES VERIFIKASI (Anti-Spoofing -> Recognition)
            # ══════════════════════════════════════════════════════════ #
            
            # Langkah 1: Verifikasi Liveness Pasif (Cek apakah itu Foto/Video)
            _put(display, "Memverifikasi Liveness...", 60, config.COLOR_YELLOW)
            liveness = anti_spoofing.is_real(frame, face.bbox)

            if not liveness["real"]:
                # Terdeteksi Spoofing (Foto/Layar HP)
                _put(display, "❌ AKSES DITOLAK: FOTO TERDETEKSI!", 90, config.COLOR_RED)
                print(f"[SECURITY] Ancaman Spoofing Terdeteksi! Score: {liveness['score']}")
                door.lock()
                state = UnlockState.DENIED
                hold_timer = 90 # Tahan pesan error selama ~3 detik
            
            else:
                # Langkah 2: Jika Liveness OK, jalankan Face Recognition
                _put(display, "Liveness OK. Memindai Wajah...", 90, config.COLOR_GREEN)
                
                face_crop = model.crop_face(frame, face.bbox)
                embedding = model.get_embedding(face_crop)
                match     = matcher.match(embedding)

                if match["matched"]:
                    # Wajah Terdaftar
                    name  = match["name"]
                    score = match["score"]
                    _put(display, f"✅ Selamat Datang, {name}!", 120, config.COLOR_GREEN, scale=0.9)
                    _put(display, f"Skor Kemiripan: {score:.4f}", 150, config.COLOR_GREEN)
                    print(f"[UNLOCK] Akses Diberikan kepada: {name} (Skor: {score:.4f})")
                    
                    door.unlock()
                    state = UnlockState.UNLOCKED
                    hold_timer = 120 # Tahan pesan sukses lebih lama
                else:
                    # Wajah Tidak Terdaftar
                    _put(display, "❌ Wajah Tidak Dikenali", 120, config.COLOR_RED, scale=0.9)
                    _put(display, f"Alasan: {match['reason']}", 150, config.COLOR_RED)
                    print(f"[UNLOCK] Akses Ditolak: {match['reason']}")
                    
                    door.lock()
                    state = UnlockState.DENIED
                    hold_timer = 60

            cv2.imshow("Smart Door Lock", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Cleanup Resources
        cam.stop()
        detector.close()
        door.cleanup()
        cv2.destroyAllWindows()
        print("[SYSTEM] Sesi berakhir.")

# ── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Menangani argumen untuk mode registrasi
    if len(sys.argv) > 1 and sys.argv[1] == "register":
        from register import run_register
        name = input("Masukkan nama untuk registrasi baru: ").strip()
        if not name:
            print("Error: Nama tidak boleh kosong.")
            sys.exit(1)
        run_register(name)
    else:
        # Jalankan mode normal (Unlock)
        run_unlock()
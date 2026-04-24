import sys
import cv2

import config
from camera.camera_stream       import CameraStream
from facemesh.facemesh_detector import FaceMeshDetector
from liveness.liveness_manager  import LivenessManager
from recognition.mobilefacenet  import MobileFaceNet
from database.face_db           import save_face

# IMPORT TAMBAHAN: Masukkan fungsi Anti-Spoofing
from liveness.anti_spoofing     import SilentAntiSpoofing

def _put(frame, text: str, y: int, color=config.COLOR_WHITE):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, color, config.LINE_TYPE)

def _draw_bbox(frame, bbox, color):
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

def run_register(name: str):
    print(f"\n[REGISTER] Memulai registrasi untuk: '{name}'")
    
    cam      = CameraStream(config.CAMERA_INDEX, config.FRAME_WIDTH, config.FRAME_HEIGHT).start()
    detector = FaceMeshDetector(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    
    # INISIALISASI MODEL
    liveness      = LivenessManager()
    model         = MobileFaceNet()
    anti_spoofing = SilentAntiSpoofing(model_path="liveness/minifasnet.onnx", threshold=0.85)

    liveness.start_register()
    embedding_saved = False

    try:
        while True:
            ret, frame = cam.read()
            if not ret: continue

            display = frame.copy()
            faces   = detector.detect(frame)

            if not faces:
                _put(display, "Wajah tidak terdeteksi", 30, config.COLOR_RED)
                cv2.imshow("Register", display)
                if cv2.waitKey(1) & 0xFF == ord("q"): break
                continue

            face = faces[0]
            _draw_bbox(display, face.bbox, config.COLOR_CYAN)

            # ══════════════════════════════════════════════════════════ #
            # LOGIKA 1: CEK ANTI-SPOOFING (VIDEO/FOTO) TERLEBIH DAHULU
            # ══════════════════════════════════════════════════════════ #
            spoof_check = anti_spoofing.is_real(frame, face.bbox)
            
            if not spoof_check["real"]:
                # Jika terdeteksi foto/layar video, hentikan proses registrasi di frame ini
                _put(display, f"REGISTER: {name}", 30, config.COLOR_YELLOW)
                _put(display, "❌ REGISTRASI DITOLAK: FOTO/VIDEO TERDETEKSI!", 60, config.COLOR_RED)
                _put(display, "Gunakan wajah asli untuk mendaftar.", 90, config.COLOR_WHITE)
                
            else:
                # ══════════════════════════════════════════════════════════ #
                # LOGIKA 2: JIKA WAJAH ASLI, JALANKAN CHALLENGE GERAK
                # ══════════════════════════════════════════════════════════ #
                result = liveness.update_register(face, detector)

                _put(display, f"REGISTER: {name} (Asli)", 30, config.COLOR_GREEN)
                _put(display, f"Langkah: {result.get('step', '')}", 60, config.COLOR_CYAN)
                _put(display, f"Instruksi: {result.get('instruction', '')}", 90, config.COLOR_WHITE)

                # Jika Liveness Aktif selesai -> Ekstraksi -> Simpan
                if result["status"] == "complete" and not embedding_saved:
                    _put(display, "✅ Wajah Valid! Menyimpan data...", 130, config.COLOR_GREEN)
                    cv2.imshow("Register", display)
                    cv2.waitKey(500) 

                    face_crop = model.crop_face(frame, face.bbox)
                    embedding = model.get_embedding(face_crop)
                    save_face(name, embedding)
                    
                    print(f"[REGISTER] ✅ '{name}' berhasil didaftarkan!")
                    embedding_saved = True

                    _put(display, f"'{name}' tersimpan! Keluar dalam 3 detik...", 160, config.COLOR_GREEN)
                    cv2.imshow("Register", display)
                    cv2.waitKey(3000)
                    break

            cv2.imshow("Register", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cam.stop()
        detector.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Penggunaan:  python register.py <nama>")
        sys.exit(1)
    run_register(sys.argv[1].strip())
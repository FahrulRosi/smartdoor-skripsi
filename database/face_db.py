import os
import pickle
import numpy as np

# Path database
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(ROOT_DIR, "faces.pkl")


class FaceDB:
    def __init__(self, path=DB_PATH):
        self.path = path

    def load(self):
        print(f"[DEBUG] Sedang mencari database di: {self.path}")
        if not os.path.exists(self.path):
            print("[DEBUG] File faces.pkl BELUM ADA, akan dibuat baru.")
            return []
        try:
            with open(self.path, "rb") as f:
                data = pickle.load(f)
                print(f"[DEBUG] Berhasil memuat {len(data)} wajah.")
                return data
        except Exception as e:
            print(f"[FaceDB] Gagal memuat database: {e}")
            return []

    def save(self, data):
        try:
            with open(self.path, "wb") as f:
                pickle.dump(data, f)
            print(f"[FaceDB] Database wajah berhasil disimpan ke: {self.path}")
        except Exception as e:
            print(f"[FaceDB] Gagal menyimpan database: {e}")


# ─── SINGLETON ────────────────────────────────────────────────────────────────
_db = FaceDB()


def get_all_faces():
    return _db.load()


def save_face(name: str, embedding: np.ndarray, captured: dict | None = None):
    faces = _db.load()

    record = {
        "name":      name,
        "embedding": embedding,
        "registered_at": "unknown",
    }

    if captured is not None:
        record.update({
            "facemesh_vector": captured.get("facemesh_vector"),
            "yaw_snapshots":   captured.get("yaw_snapshots", []),
            "pitch_snapshots": captured.get("pitch_snapshots", []),
            "roll_snapshots":  captured.get("roll_snapshots", []),
            "blink_closed":    captured.get("blink_closed"),
            "blink_open":      captured.get("blink_open"),
            "mobilefacenet_embedding": captured.get("mobilefacenet_embedding", embedding),
        })

        _log_saved_summary(name, record)
    else:
        print(f"[FaceDB] WARNING: captured=None untuk {name}.")

    # Update atau tambah data baru
    updated = False
    for i, face in enumerate(faces):
        if face.get("name") == name:
            faces[i] = record
            updated = True
            print(f"[FaceDB] Data wajah '{name}' diperbarui.")
            break

    if not updated:
        faces.append(record)
        print(f"[FaceDB] Data wajah '{name}' ditambahkan sebagai entri baru.")

    _db.save(faces)


def _log_saved_summary(name: str, record: dict):
    """Versi AMAN dari summary (mencegah error NoneType)"""
    print("\n" + "─" * 75)
    print(f"  [FaceDB] RINGKASAN DATA TERSIMPAN UNTUK: {name}")
    print("─" * 75)

    fm = record.get("facemesh_vector")
    print(f"  Tahap 1 – FaceMesh vector   : {'shape=' + str(fm.shape) if fm is not None else 'TIDAK ADA'}")

    print(f"  Tahap 2 – YAW snapshots     : {len(record.get('yaw_snapshots', []))} snapshot(s)")
    print(f"  Tahap 2 – PITCH snapshots   : {len(record.get('pitch_snapshots', []))} snapshot(s)")
    print(f"  Tahap 2 – ROLL snapshots    : {len(record.get('roll_snapshots', []))} snapshot(s)")

    # === PERBAIKAN UTAMA: Penanganan aman untuk blink ===
    bc = record.get("blink_closed")
    bo = record.get("blink_open")

    blink_closed_ear = bc.get('avg_ear', 'TIDAK ADA') if bc is not None else 'TIDAK ADA / BELUM KEDIP'
    blink_open_ear   = bo.get('avg_ear', 'TIDAK ADA') if bo is not None else 'TIDAK ADA / BELUM KEDIP'

    print(f"  Tahap 3 – BLINK closed EAR  : {blink_closed_ear}")
    print(f"  Tahap 3 – BLINK open EAR    : {blink_open_ear}")

    emb = record.get("embedding")
    emb_norm = np.linalg.norm(emb) if emb is not None else 0
    print(f"  Tahap 4 – MobileFaceNet emb : shape={emb.shape if emb is not None else 'TIDAK ADA'}, "
          f"norm={emb_norm:.4f}")

    print("─" * 75 + "\n")
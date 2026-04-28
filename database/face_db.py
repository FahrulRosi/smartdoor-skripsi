import os
import pickle
import numpy as np

# PAKSA agar faces.pkl selalu berada di folder root project (smartdoor-skripsi)
# Naik 1 level dari folder 'database' ke folder utama
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(ROOT_DIR, "faces.pkl")


class FaceDB:
    def __init__(self, path=DB_PATH):
        self.path = path

    def load(self):
        print(f"[DEBUG] Sedang mencari database di: {self.path}")
        if not os.path.exists(self.path):
            print("[DEBUG] File faces.pkl BELUM ADA di lokasi tersebut!")
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


# ─── PUBLIC API ───────────────────────────────────────────────────────────────

def get_all_faces():
    """Mengambil semua data wajah yang terdaftar dari file pkl."""
    return _db.load()


def save_face(name: str, embedding: np.ndarray, captured: dict | None = None):
    """
    Menyimpan data lengkap registrasi ke dalam file pkl.

    Struktur record yang disimpan
    ─────────────────────────────
    {
        "name": str,

        # Tahap 1 – FaceMesh
        "facemesh_vector": np.ndarray | None,      # (1404,) float32

        # Tahap 2 – Pose
        "yaw_snapshots":   list[dict],             # [{tag, yaw, pitch, roll}, ...]
        "pitch_snapshots": list[dict],
        "roll_snapshots":  list[dict],

        # Tahap 3 – Blink
        "blink_closed": dict | None,               # {left_ear, right_ear, avg_ear, blink_detected}
        "blink_open":   dict | None,

        # Tahap 4 – MobileFaceNet (dipakai untuk recognition / matching)
        "embedding":    np.ndarray,                # (512,) float32   ← kunci utama matcher
    }

    Parameter
    ---------
    name      : Nama pengguna yang didaftarkan.
    embedding : Vektor MobileFaceNet (512-dim) – wajib ada.
    captured  : Dict hasil tangkapan tiap tahap dari register.py.
                Jika None, hanya embedding yang disimpan (fallback).
    """
    faces = _db.load()

    # Bangun record baru dari semua data yang dikumpulkan
    record = {
        "name":      name,
        "embedding": embedding,   # ← tetap di key "embedding" agar FaceMatcher tidak berubah
    }

    if captured is not None:
        # ── Tahap 1 – FaceMesh ───────────────────────────────────────────────
        record["facemesh_vector"] = captured.get("facemesh_vector")   # np.ndarray | None

        # ── Tahap 2 – Pose snapshots ─────────────────────────────────────────
        record["yaw_snapshots"]   = captured.get("yaw_snapshots",   [])
        record["pitch_snapshots"] = captured.get("pitch_snapshots", [])
        record["roll_snapshots"]  = captured.get("roll_snapshots",  [])

        # ── Tahap 3 – Blink ───────────────────────────────────────────────────
        record["blink_closed"] = captured.get("blink_closed")
        record["blink_open"]   = captured.get("blink_open")

        # Embedding MobileFaceNet juga disimpan di key khusus agar mudah diakses
        record["mobilefacenet_embedding"] = captured.get("mobilefacenet_embedding", embedding)

        # ── Log ringkasan apa yang berhasil disimpan ──────────────────────────
        _log_saved_summary(name, record)
    else:
        # Fallback: hanya embedding yang ada (tidak ada data tahap lain)
        print(f"[FaceDB] WARNING: captured=None untuk {name}. Hanya embedding yang disimpan.")

    # Update jika nama sudah ada, tambah jika baru
    updated = False
    for i, face in enumerate(faces):
        if face["name"] == name:
            faces[i] = record
            updated   = True
            print(f"[FaceDB] Data wajah '{name}' diperbarui.")
            break

    if not updated:
        faces.append(record)
        print(f"[FaceDB] Data wajah '{name}' ditambahkan sebagai entri baru.")

    _db.save(faces)


def _log_saved_summary(name: str, record: dict):
    """Cetak ringkasan data yang berhasil disimpan ke database."""
    fm  = record.get("facemesh_vector")
    emb = record.get("embedding")

    print("\n" + "─" * 60)
    print(f"  [FaceDB] RINGKASAN DATA TERSIMPAN UNTUK: {name}")
    print("─" * 60)
    print(f"  Tahap 1 – FaceMesh vector   : "
          f"{'shape=' + str(fm.shape) if fm is not None else 'TIDAK ADA'}")
    print(f"  Tahap 2 – YAW snapshots     : "
          f"{len(record.get('yaw_snapshots', []))} snapshot(s) "
          f"{[s['tag'] for s in record.get('yaw_snapshots', [])]}")
    print(f"  Tahap 2 – PITCH snapshots   : "
          f"{len(record.get('pitch_snapshots', []))} snapshot(s) "
          f"{[s['tag'] for s in record.get('pitch_snapshots', [])]}")
    print(f"  Tahap 2 – ROLL snapshots    : "
          f"{len(record.get('roll_snapshots', []))} snapshot(s) "
          f"{[s['tag'] for s in record.get('roll_snapshots', [])]}")

    bc = record.get("blink_closed")
    bo = record.get("blink_open")
    print(f"  Tahap 3 – BLINK closed EAR  : "
          f"{bc['avg_ear']:.4f if bc else 'TIDAK ADA'}")
    print(f"  Tahap 3 – BLINK open EAR    : "
          f"{bo['avg_ear']:.4f if bo else 'TIDAK ADA'}")
    print(f"  Tahap 4 – MobileFaceNet emb : "
          f"shape={emb.shape}, norm={np.linalg.norm(emb):.4f}")
    print("─" * 60 + "\n")
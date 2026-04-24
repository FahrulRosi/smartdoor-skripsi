import os
import pickle

# PAKSA agar faces.pkl selalu berada di folder root project (smartdoor-skripsi)
# Naik 1 level dari folder 'database' ke folder utama
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(ROOT_DIR, "faces.pkl")

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

# --- Helper functions ---
_db = FaceDB()

def get_all_faces():
    """Mengambil semua data wajah yang terdaftar dari file pkl"""
    return _db.load()

def save_face(name, embedding):
    """Menyimpan embedding wajah baru ke dalam file pkl"""
    faces = _db.load()
    
    updated = False
    for face in faces:
        if face["name"] == name:
            face["embedding"] = embedding
            updated = True
            break
            
    if not updated:
        faces.append({
            "name": name,
            "embedding": embedding
        })
        
    _db.save(faces)
# database/face_db.py
import pickle

class FaceDB:
    def __init__(self, path="faces.pkl"):
        self.path = path

    def load(self):
        try:
            return pickle.load(open(self.path, "rb"))
        except:
            return []

    def save(self, data):
        pickle.dump(data, open(self.path, "wb"))

# --- Tambahkan helper functions di bawah ini ---
_db = FaceDB()

def get_all_faces():
    """Mengambil semua data wajah yang terdaftar dari file pkl"""
    return _db.load()

def save_face(name, embedding):
    """Menyimpan embedding wajah baru ke dalam file pkl"""
    faces = _db.load()
    faces.append({
        "name": name,
        "embedding": embedding
    })
    _db.save(faces)
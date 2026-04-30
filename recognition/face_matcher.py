import numpy as np
# 1. Sesuaikan import dengan struktur Firebase yang baru
from database.face_db import FaceDatabase
import config 

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Menghitung kemiripan Cosine antara dua vektor embedding."""
    # Flatten array untuk mencegah error dimensi matriks
    a = a.flatten()
    b = b.flatten()
    
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    
    # Hitung similarity dan cegah hasil float di luar batas -1.0 hingga 1.0
    similarity = float(np.dot(a, b))
    return max(-1.0, min(1.0, similarity))

class FaceMatcher:
    def __init__(self, threshold: float = 0.55):
        """Inisialisasi Face Matcher dengan Threshold Cosine Similarity"""
        self.threshold = threshold
        
        # 2. Inisialisasi Database Firebase di dalam Class Matcher
        # GANTI URL DI BAWAH dengan URL Firebase Realtime Database Anda
        # (Lebih baik jika URL ini disimpan di config.py)
        db_url = "https://smart-door-lock-feb6b-default-rtdb.asia-southeast1.firebasedatabase.app"
        credentials_path = 'serviceAccount.json'  # Pastikan file ini ada di folder yang sama
        self.db = FaceDatabase(db_url, credentials_path)
        
        # 3. Muat semua wajah saat sistem (matcher) dinyalakan
        # Ini menghindari pengambilan data (download) dari Firebase setiap frame kamera
        self.known_faces = self.db.load_all_faces()

    def update_database(self):
        """
        Fungsi opsional: Panggil ini jika Anda ingin sistem menyegarkan 
        (refresh) database tanpa harus merestart program utama.
        """
        self.known_faces = self.db.load_all_faces()

    def match(self, embedding: np.ndarray) -> dict:
        """
        Membandingkan `embedding` wajah dari kamera dengan setiap wajah 
        yang tersimpan di memori (hasil unduhan dari Firebase).
        Mengembalikan data wajah terbaik (jika similarity >= threshold)[cite: 6].
        """
        # Cek dari dictionary self.known_faces, BUKAN get_all_faces() lagi
        if not self.known_faces:
            return {"matched": False, "name": "No_DB", "score": 0.0, "reason": "Belum ada wajah di DB"}

        best_name  = "Unknown"
        best_score = -1.0

        # Iterasi dictionary (Key: Nama, Value: Embedding Array)
        for name, db_embedding in self.known_faces.items():
            score = cosine_similarity(embedding, db_embedding)
            if score > best_score:
                best_score = score
                best_name  = name

        matched = best_score >= self.threshold
        
        return {
            "matched": matched,
            # Tetap kembalikan nama terdekat agar saat tidak match, 
            # main.py bisa menampilkan "Wajah terdekat: Nama (Score)" di layar HUD[cite: 6]
            "name":    best_name, 
            "score":   round(best_score, 4),
            "reason":  "OK" if matched else f"Score {best_score:.4f} < threshold {self.threshold}",
        }
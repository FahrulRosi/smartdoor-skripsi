import numpy as np
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
        
        # Wadah memori untuk menyimpan data wajah.
        # DIKOSONGKAN saat inisialisasi. Data akan disuntikkan dari luar (main.py / register.py)
        self.known_faces = {}

    def load_faces(self, faces_dict: dict):
        """
        Fungsi untuk memuat data wajah dari luar (yang sudah di-download oleh FaceDatabase)
        ke dalam memori FaceMatcher ini.
        """
        if faces_dict:
            self.known_faces = faces_dict

    def match(self, embedding: np.ndarray) -> dict:
        """
        Membandingkan `embedding` wajah dari kamera dengan setiap wajah 
        yang tersimpan di memori (self.known_faces).
        Mengembalikan data wajah terbaik (jika similarity >= threshold).
        """
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
            # main.py bisa menampilkan "Wajah terdekat: Nama (Score)" di layar HUD untuk debugging
            "name":    best_name, 
            "score":   round(best_score, 4),
            "reason":  "OK" if matched else f"Score {best_score:.4f} < threshold {self.threshold}",
        }
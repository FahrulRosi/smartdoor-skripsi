import numpy as np
import config 

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Menghitung kemiripan Cosine antara dua vektor embedding dengan proteksi ketat."""
    # Flatten array untuk mencegah error dimensi matriks
    a = a.flatten()
    b = b.flatten()
    
    # PROTEKSI 1: Tolak jika panjang dimensi vektor tidak sama atau kosong
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    
    if a_norm == 0 or b_norm == 0:
        return 0.0
        
    a = a / (a_norm + 1e-8)
    b = b / (b_norm + 1e-8)
    
    # Hitung similarity dan cegah hasil float di luar batas -1.0 hingga 1.0
    similarity = float(np.dot(a, b))
    return max(-1.0, min(1.0, similarity))

class FaceMatcher:
    def __init__(self, threshold: float = None):
        """Inisialisasi Face Matcher dengan Threshold Cosine Similarity yang ketat"""
        # Mengambil nilai MATCH_THRESHOLD dari config (0.68) jika tidak diberikan spesifik
        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = getattr(config, 'MATCH_THRESHOLD', 0.68)
            
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
        Mengembalikan data wajah terbaik (jika similarity >= threshold ketat).
        """
        if not self.known_faces:
            return {"matched": False, "name": "No_DB", "score": 0.0, "reason": "Belum ada wajah di DB"}

        best_name  = "Unknown"
        best_score = -1.0

        # Iterasi dictionary (Key: Nama, Value: Embedding Array)
        for name, db_embedding in self.known_faces.items():
            # PROTEKSI 2: Pastikan data dari database selalu terkonversi menjadi numpy array float32
            db_emb = np.array(db_embedding, dtype=np.float32)
            score = cosine_similarity(embedding, db_emb)
            
            if score > best_score:
                best_score = score
                best_name  = name

        # PROTEKSI 3: Memaksa batas minimal kelulusan mutlak di angka aman 0.65
        # (Untuk mencegah penerimaan keliru / FAR jika ada parameter threshold rendah)
        strict_threshold = max(self.threshold, 0.65)
        matched = best_score >= strict_threshold
        
        return {
            "matched": matched,
            # Tetap kembalikan nama terdekat agar saat tidak match, 
            # main.py bisa menampilkan "Wajah terdekat: Nama (Score)" di layar HUD untuk debugging
            "name":    best_name, 
            "score":   round(best_score, 4),
            "reason":  "OK" if matched else f"Score {best_score:.4f} < threshold {strict_threshold}",
        }
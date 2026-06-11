import numpy as np
import config 

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Menghitung kemiripan Cosine antara dua vektor embedding dengan proteksi ketat."""
    a = a.flatten()
    b = b.flatten()
    
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    
    if a_norm == 0 or b_norm == 0:
        return 0.0
        
    a = a / (a_norm + 1e-8)
    b = b / (b_norm + 1e-8)
    
    similarity = float(np.dot(a, b))
    return max(-1.0, min(1.0, similarity))

class FaceMatcher:
    def __init__(self, threshold: float = None):
        """Inisialisasi Face Matcher dengan Threshold Cosine Similarity yang ketat"""
        if threshold is not None:
            self.threshold = threshold
        else:
            self.threshold = getattr(config, 'MATCH_THRESHOLD', 0.70)
            
        self.known_faces = {}

    def load_faces(self, faces_dict: dict):
        """Memuat data wajah dari luar ke dalam memori"""
        if faces_dict:
            self.known_faces = faces_dict

    def match(self, embedding: np.ndarray) -> dict:
        """
        Mendukung perbandingan dengan Matriks 2D (Solusi A - Multi Vector)
        maupun data 1D (versi lama).
        """
        if not self.known_faces:
            return {"matched": False, "name": "No_DB", "score": 0.0, "reason": "Belum ada wajah di DB"}

        best_name  = "Unknown"
        best_score = -1.0

        for name, db_embedding in self.known_faces.items():
            # PENYESUAIAN SOLUSI A: Cek apakah data di DB berupa Matriks 2D (Multi-Vector) atau 1D (Lama)
            if isinstance(db_embedding, list) and len(db_embedding) > 0 and isinstance(db_embedding[0], list):
                # Jika Matriks 2D (Data baru dari register kloning)
                db_embs = [np.array(e, dtype=np.float32) for e in db_embedding]
            else:
                # Jika 1D (Data register versi lama)
                db_embs = [np.array(db_embedding, dtype=np.float32)]
            
            # Looping untuk membandingkan wajah dengan ke-3 vektor kloning, ambil skor murni tertinggi
            for db_emb in db_embs:
                score = cosine_similarity(embedding, db_emb)
                if score > best_score:
                    best_score = score
                    best_name  = name

        strict_threshold = max(self.threshold, 0.65)
        matched = best_score >= strict_threshold
        
        return {
            "matched": matched,
            "name":    best_name, 
            "score":   round(best_score, 4),
            "reason":  "OK" if matched else f"Score {best_score:.4f} < threshold {strict_threshold}",
        }
import numpy as np
from database.face_db import get_all_faces

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    # Flatten array untuk mencegah error dimensi matriks, misal (1, 512) dengan (512,)
    a = a.flatten()
    b = b.flatten()
    
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    
    # Hitung similarity dan cegah hasil float di luar batas -1.0 hingga 1.0
    similarity = float(np.dot(a, b))
    return max(-1.0, min(1.0, similarity))

class FaceMatcher:
    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold

    def match(self, embedding: np.ndarray) -> dict:
        """
        Compare `embedding` against every stored face.
        Returns the best match (if similarity >= threshold).
        """
        faces = get_all_faces()
        if not faces:
            return {"matched": False, "name": "No_DB", "score": 0.0, "reason": "Belum ada wajah di DB"}

        best_name  = "Unknown"
        best_score = -1.0

        for face in faces:
            score = cosine_similarity(embedding, face["embedding"])
            if score > best_score:
                best_score = score
                best_name  = face["name"]

        matched = best_score >= self.threshold
        
        return {
            "matched": matched,
            # Tetap kembalikan nama terdekat agar saat tidak match, 
            # main.py bisa menampilkan "Wajah terdekat: Nama (Score)" di layar HUD
            "name":    best_name, 
            "score":   round(best_score, 4),
            "reason":  "OK" if matched else f"Score {best_score:.4f} < threshold {self.threshold}",
        }
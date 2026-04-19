import numpy as np
from database.face_db import get_all_faces


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


class FaceMatcher:
    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold

    def match(self, embedding: np.ndarray) -> dict:
        """
        Compare `embedding` against every stored face.
        Returns the best match (if similarity >= threshold), else None.
        """
        faces = get_all_faces()
        if not faces:
            return {"matched": False, "name": None, "score": 0.0, "reason": "No registered faces"}

        best_name  = None
        best_score = -1.0

        for face in faces:
            score = cosine_similarity(embedding, face["embedding"])
            if score > best_score:
                best_score = score
                best_name  = face["name"]

        matched = best_score >= self.threshold
        return {
            "matched": matched,
            "name":    best_name if matched else None,
            "score":   round(best_score, 4),
            "reason":  "OK" if matched else f"Score {best_score:.4f} < threshold {self.threshold}",
        }
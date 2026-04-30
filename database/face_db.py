import os
import time
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
import traceback

class FaceDatabase:
    def __init__(self, db_url="https://smart-door-lock-feb6b-default-rtdb.asia-southeast1.firebasedatabase.app", credentials_path="../serviceAccount.json"):
        """Inisialisasi koneksi ke Firebase Realtime Database"""
        # Cek agar tidak inisialisasi ulang jika dipanggil berkali-kali
        if not firebase_admin._apps:
            # Pastikan file serviceAccountKey.json ada di folder yang sama
            cred = credentials.Certificate(credentials_path)
            firebase_admin.initialize_app(cred, {
                'databaseURL': db_url
            })
        
        # Referensi utama ke tabel/node 'registered_users'
        self.ref_users = db.reference('registered_users')

    def save_face(self, name, embedding, liveness_data):
        """Menyimpan data wajah dan parameter liveness ke Firebase"""
        try:
            # 1. Ekstraksi Blink secara ekstrim aman
            # Ambil data blink dari dictionary yang dikirim register.py
            blink_c = liveness_data.get("blink_closed", {})
            blink_o = liveness_data.get("blink_open", {})
            
            # Ekstrak nilai angkanya (jika dia dictionary, ambil avg_ear. Jika tidak, jadikan 0.0)
            ear_c = blink_c.get("avg_ear", 0.0) if isinstance(blink_c, dict) else 0.0
            ear_o = blink_o.get("avg_ear", 0.0) if isinstance(blink_o, dict) else 0.0

            # 2. Ekstraksi Facemesh & Embedding secara ekstrim aman
            fm_vector = liveness_data.get("facemesh_vector", [])
            fm_list = fm_vector.tolist() if isinstance(fm_vector, np.ndarray) else list(fm_vector)
            emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)

            # 3. Susun data JSON
            data_to_save = {
                "name": name,
                "embedding": emb_list,
                "liveness_config": {
                    "facemesh_vector": fm_list,
                    "blink_closed": float(ear_c),
                    "blink_open": float(ear_o)
                },
                "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            
            # 4. Kirim ke Firebase
            self.ref_users.child(name).set(data_to_save)
            print(f"[Firebase] Data wajah untuk '{name}' berhasil diunggah ke Cloud.")
            return True
            
        except Exception as e:
            print(f"[Firebase ERROR] Gagal menyimpan data: {e}")
            traceback.print_exc() # Ini akan mencetak baris ke-berapa yang error jika masih gagal
            return False

    def check_user_exists(self, name):
        """Mengecek apakah nama sudah terdaftar di Firebase"""
        user = self.ref_users.child(name).get()
        return user is not None

    def load_all_faces(self):
        """Mengambil semua data wajah dari Firebase untuk proses Matching"""
        print("[Firebase] Mengunduh data wajah dari cloud...")
        faces = self.ref_users.get()
        embeddings = {}
        
        if faces:
            for name, data in faces.items():
                if "embedding" in data:
                    # Kembalikan ke format numpy array untuk kalkulasi jarak di face_matcher
                    embeddings[name] = np.array(data["embedding"])
            print(f"[Firebase] Berhasil memuat {len(embeddings)} wajah.")
        else:
            print("[Firebase] Database masih kosong.")
            
        return embeddings
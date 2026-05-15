import os, json, time, threading, traceback, numpy as np
from supabase import create_client, Client
import config

# ==============================================================================
# 1. DATA TRANSFORMER
# ==============================================================================
class DataTransformer:
    @staticmethod
    def prepare_payload(name, embedding, liveness_data):
        blink_c = liveness_data.get("blink_closed", {})
        blink_o = liveness_data.get("blink_open", {})
        ear_c = blink_c.get("avg_ear", 0.0) if isinstance(blink_c, dict) else 0.0
        ear_o = blink_o.get("avg_ear", 0.0) if isinstance(blink_o, dict) else 0.0

        fm_vector = liveness_data.get("facemesh_vector", [])
        fm_list = fm_vector.tolist() if isinstance(fm_vector, np.ndarray) else list(fm_vector)
        emb_list = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
        hp_vec = liveness_data.get("headpose_vector", [0.0, 0.0, 0.0])

        yaw_left = yaw_right = pitch_up = pitch_down = roll_left = roll_right = 0.0
        for snap in liveness_data.get("yaw_snapshots", []):
            if snap.get("tag") == "yaw_left": yaw_left = snap.get("yaw", 0.0)
            elif snap.get("tag") == "yaw_right": yaw_right = snap.get("yaw", 0.0)
        for snap in liveness_data.get("pitch_snapshots", []):
            if snap.get("tag") == "pitch_up": pitch_up = snap.get("pitch", 0.0)
            elif snap.get("tag") == "pitch_down": pitch_down = snap.get("pitch", 0.0)
        for snap in liveness_data.get("roll_snapshots", []):
            if snap.get("tag") == "roll_left": roll_left = snap.get("roll", 0.0)
            elif snap.get("tag") == "roll_right": roll_right = snap.get("roll", 0.0)

        return {
            "name": name,
            "embedding": emb_list,
            "liveness_config": {
                "facemesh_vector": fm_list,
                "blink_closed": float(ear_c),
                "blink_open": float(ear_o),
                "headpose_vector": hp_vec,  # KUNCI KRUSIAL: Agar sinkron langsung dibaca oleh main.py
                "headpose": {
                    "neutral_vector": hp_vec,
                    "yaw_left": float(yaw_left),
                    "yaw_right": float(yaw_right),
                    "pitch_up": float(pitch_up),
                    "pitch_down": float(pitch_down),
                    "roll_left": float(roll_left),
                    "roll_right": float(roll_right)
                }
            },
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

# ==============================================================================
# 2. LOCAL STORAGE
# ==============================================================================
class LocalStorage:
    def __init__(self, path="local_faces.json"):
        self.path, self.lock = path, threading.Lock()
        if not os.path.exists(self.path): self._write({})

    def _read(self):
        try:
            with open(self.path, 'r') as f: return json.load(f)
        except Exception: return {}

    def _write(self, data):
        with open(self.path, 'w') as f: json.dump(data, f, indent=4)

    def save_user(self, name, payload):
        with self.lock:
            d = self._read()
            d[name] = payload
            self._write(d)

    def load_all(self):
        with self.lock: return self._read()

    def exists(self, name):
        with self.lock: return name in self._read()

    def overwrite_all(self, data_dict):
        with self.lock: self._write(data_dict)

# ==============================================================================
# 3. CLOUD STORAGE
# ==============================================================================
class CloudStorage:
    def __init__(self):
        self.url = getattr(config, "SUPABASE_URL", "")
        self.key = getattr(config, "SUPABASE_KEY", "")
        self.is_connected = False
        
        if self.url and self.key:
            try:
                self.client: Client = create_client(self.url, self.key)
                self.is_connected = True
                print("[Supabase] Terhubung ke Cloud Database PostgreSQL.")
            except Exception as e: print(f"[Supabase WARNING] Gagal inisialisasi: {e}")

    def get_user_id(self, username):
        if not self.is_connected: return None
        try:
            res = self.client.table("users_profile").select("id").eq("username", username).execute()
            return res.data[0]["id"] if res.data else None
        except Exception: return None

    def create_dummy_user(self, username):
        try:
            payload = {
                "username": username,
                "email": f"{username.lower().replace(' ', '')}@auto.local",
                "password_hash": "auto_generated_by_ai"
            }
            res = self.client.table("users_profile").insert(payload).execute()
            return res.data[0]["id"] if res.data else None
        except Exception as e:
            print(f"[Supabase ERROR] Gagal auto-create profil web: {e}"); return None

    def sync_register(self, name, payload):
        if not self.is_connected: return
        try:
            user_id = self.get_user_id(name)
            if not user_id:
                print(f"[Supabase] Username '{name}' belum ada. Membuatkan profil Web otomatis...")
                user_id = self.create_dummy_user(name)
                if not user_id: return
            
            db_payload = {
                "user_id": user_id,
                "username": name,
                "face_embedding": payload["embedding"],
                "liveness_config": payload["liveness_config"],
                "registered_at": payload["registered_at"]
            }
            self.client.table("register").upsert(db_payload).execute()
            print(f"[Supabase] Vektor wajah '{name}' berhasil disinkronkan ke Cloud!")
        except Exception as e: print(f"[Cloud Error] Gagal sinkronisasi vektor: {e}")

    def push_access_log(self, user_name, status, score):
        if not self.is_connected: return
        try:
            user_id = self.get_user_id(user_name)
            if not user_id: return
            log_data = {
                "user_id": user_id, "status": status,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            if score is not None: log_data["liveness_score"] = float(score)
            self.client.table("access_logs").insert(log_data).execute()
        except Exception: pass

    def fetch_all_registered_users(self):
        if not self.is_connected: return None
        try: return self.client.table("register").select("*").execute().data
        except Exception: return None

# ==============================================================================
# 4. MAIN FACADE
# ==============================================================================
class FaceDatabase:
    def __init__(self, local_db_path="local_faces.json"):
        self.local = LocalStorage(local_db_path)
        self.cloud = CloudStorage()

    def check_user_exists(self, name):
        return self.local.exists(name)

    def save_face(self, name, embedding, liveness_data):
        try:
            payload = DataTransformer.prepare_payload(name, embedding, liveness_data)
            self.local.save_user(name, payload)
            if self.cloud.is_connected:
                t = threading.Thread(target=self.cloud.sync_register, args=(name, payload))
                t.daemon = True; t.start()
            return True
        except Exception: traceback.print_exc(); return False

    def load_all_faces(self, silent=False):
        faces = self.local.load_all()

        if not faces and self.cloud.is_connected:
            if not silent: print("[Hybrid Sync] Memori lokal kosong. Menarik data vektor dari Supabase...")
            cloud_data = self.cloud.fetch_all_registered_users()
            if cloud_data:
                faces = {
                    row["username"]: {
                        "name": row["username"],
                        "embedding": row["face_embedding"],
                        "liveness_config": row["liveness_config"],
                        "registered_at": row["registered_at"]
                    } for row in cloud_data
                }
                self.local.overwrite_all(faces)
                if not silent: print(f"[Hybrid Sync] Berhasil memulihkan {len(faces)} identitas wajah.")

        # --- PERBAIKAN MUTLAK: MENGEMBALIKAN FACES SECARA UTUH ---
        # Dengan mengembalikan dictionary aslinya, logika pemrosesan memori 
        # di main.py dan register.py dapat mengakses 'embedding' sekaligus 
        # menarik keluar 'liveness_config' unik milik setiap user secara akurat.
        return faces

    def push_access_log_async(self, user_name, status="UNLOCKED", score=None):
        if self.cloud.is_connected:
            t = threading.Thread(target=self.cloud.push_access_log, args=(user_name, status, score))
            t.daemon = True; t.start()
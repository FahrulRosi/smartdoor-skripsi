import os, json, time, threading, traceback, numpy as np
from supabase import create_client, Client
import config

# ==============================================================================
# 1. DATA TRANSFORMER
# ==============================================================================
class DataTransformer:
    @staticmethod
    def prepare_payload(name, embedding, liveness_data):
        blink_c = liveness_data.get("blink_closed") or {}
        blink_o = liveness_data.get("blink_open") or {}
        ear_c = float(blink_c.get("avg_ear", 0.0))
        ear_o = float(blink_o.get("avg_ear", 0.0))
        ear_c_lat = float(blink_c.get("latency", 0.0))
        ear_o_lat = float(blink_o.get("latency", 0.0))

        # Konversi aman numpy array ke list python standard
        emb_list = [float(x) for x in embedding]
        
        # Ekstrak facemesh dan headpose vector untuk liveness_config
        fm_vector = liveness_data.get("facemesh_vector", [])
        fm_list = fm_vector.tolist() if isinstance(fm_vector, np.ndarray) else list(fm_vector)
        hp_vec = liveness_data.get("headpose_vector", [0.0, 0.0, 0.0])

        yl = yr = pu = pd = rl = rr = 0.0
        lat_yl = lat_yr = lat_pu = lat_pd = lat_rl = lat_rr = 0.0

        for snap in liveness_data.get("yaw_snapshots", []):
            if snap.get("tag") == "yaw_left": 
                yl, lat_yl = float(snap.get("yaw", 0.0)), float(snap.get("latency", 0.0))
            elif snap.get("tag") == "yaw_right": 
                yr, lat_yr = float(snap.get("yaw", 0.0)), float(snap.get("latency", 0.0))
                
        for snap in liveness_data.get("pitch_snapshots", []):
            if snap.get("tag") == "pitch_up": 
                pu, lat_pu = float(snap.get("pitch", 0.0)), float(snap.get("latency", 0.0))
            elif snap.get("tag") == "pitch_down": 
                pd, lat_pd = float(snap.get("pitch", 0.0)), float(snap.get("latency", 0.0))
                
        for snap in liveness_data.get("roll_snapshots", []):
            if snap.get("tag") == "roll_left": 
                rl, lat_rl = float(snap.get("roll", 0.0)), float(snap.get("latency", 0.0))
            elif snap.get("tag") == "roll_right": 
                rr, lat_rr = float(snap.get("roll", 0.0)), float(snap.get("latency", 0.0))

        return {
            "name": str(name), 
            "embedding": emb_list,
            # 🚨 MENGEMBALIKAN LIVENESS_CONFIG KE DALAM PAYLOAD
            "liveness_config": {
                "facemesh_vector": fm_list, 
                "blink_closed": float(ear_c), 
                "blink_open": float(ear_o),
                "headpose_vector": hp_vec,
                "headpose": {
                    "neutral_vector": hp_vec, 
                    "yaw_left": float(yl), "yaw_right": float(yr),
                    "pitch_up": float(pu), "pitch_down": float(pd), 
                    "roll_left": float(rl), "roll_right": float(rr)
                }
            },
            # DATA TERPISAH (DENGAN LATENSI)
            "yaw_score": f"L:{yl:.1f}° ({lat_yl:.1f}ms) R:{yr:.1f}° ({lat_yr:.1f}ms)",
            "pitch_score": f"U:{pu:.1f}° ({lat_pu:.1f}ms) D:{pd:.1f}° ({lat_pd:.1f}ms)",
            "roll_score": f"L:{rl:.1f}° ({lat_rl:.1f}ms) R:{rr:.1f}° ({lat_rr:.1f}ms)",
            "blink_score": f"Buka:{ear_o:.2f} ({ear_o_lat:.1f}ms) Kedip:{ear_c:.2f} ({ear_c_lat:.1f}ms)",
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

# ==============================================================================
# 2. LOCAL STORAGE (JSON)
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
# 3. CLOUD STORAGE (SUPABASE)
# ==============================================================================
class CloudStorage:
    def __init__(self):
        self.url, self.key, self.is_connected = getattr(config, "SUPABASE_URL", ""), getattr(config, "SUPABASE_KEY", ""), False
        if self.url and self.key:
            try:
                self.client: Client = create_client(self.url, self.key)
                self.is_connected = True
                print("[Supabase] Terhubung ke Cloud Database PostgreSQL.")
            except Exception as e: print(f"[Supabase WARNING] Gagal inisialisasi: {e}")

    def sync_register(self, name, payload):
        if not self.is_connected: return
        try:
            db_payload = {
                "name": payload["name"], 
                "embedding": payload["embedding"], 
                "liveness_config": payload.get("liveness_config", {}), # 🚨 Liveness Config Dikirim Lagi
                "yaw_score": payload.get("yaw_score", ""),
                "pitch_score": payload.get("pitch_score", ""),
                "roll_score": payload.get("roll_score", ""),
                "blink_score": payload.get("blink_score", ""),
                "created_at": payload["registered_at"]
            }
            self.client.table("registered_faces").upsert(db_payload, on_conflict="name").execute()
            print(f"[Supabase] ✅ Data wajah '{name}' berhasil disimpan ke Cloud!")
        except Exception as e: 
            print(f"[Cloud Error] Gagal sinkronisasi registered_faces untuk '{name}': {e}")

    def push_register_log(self, name, status, accuracy, light_cond, message, payload=None):
        if not self.is_connected: return
        try:
            data = {
                "name": name, "status": status, "accuracy": float(accuracy),
                "light_condition": light_cond, "message": message
            }
            if payload:
                data.update({
                    "yaw_score": payload.get("yaw_score", ""),
                    "pitch_score": payload.get("pitch_score", ""),
                    "roll_score": payload.get("roll_score", ""),
                    "blink_score": payload.get("blink_score", "")
                })

            self.client.table("register_logs").insert(data).execute()
            print(f"[Supabase] 📝 Log Registrasi '{name}' ({status}) tersimpan ke Cloud!")
        except Exception as e: 
            print(f"[Cloud Error] Gagal menyimpan register_logs untuk '{name}': {e}")

    def push_access_log(self, user_name, status, accuracy, light_cond, access_details=None):
        if not self.is_connected: return
        try:
            headpose_str, blink_str = "", ""
            if access_details:
                for d in access_details:
                    t = d.get("tantangan", "")
                    info = f"{t}: {float(d.get('skor_asli', 0)):.2f} (Tgt: {float(d.get('target', 0)):.2f}, Lat: {float(d.get('latensi_ms', 0)):.0f}ms)"
                    
                    tl = t.lower()
                    if any(x in tl for x in ["toleh", "dongak", "tunduk", "miring"]): headpose_str += info + " | "
                    elif "kedip" in tl or "mata" in tl: blink_str += info + " | "

            data = {
                "name": user_name, "status": status, 
                "headpose_score": headpose_str.strip(" | "), "blink_score": blink_str.strip(" | "),
                "accuracy": float(accuracy), "light_condition": light_cond
            }
            self.client.table("access_logs").insert(data).execute()
            print(f"[Supabase] 📝 Log Akses Pintu '{user_name}' tersimpan ke Cloud!")
        except Exception as e: 
            print(f"[Cloud Error] Gagal merekam log akses pintu: {e}")

    def push_spoofing_log(self, score, message):
        if not self.is_connected: return
        try:
            self.client.table("spoofing_logs").insert({"spoof_score": float(score), "message": message}).execute()
            print(f"[Supabase] 🚨 Peringatan Spoofing (Skor: {score:.2f}) tersimpan ke Cloud!")
        except Exception as e: 
            print(f"[Cloud Error] Gagal menyimpan spoofing log: {e}")

    def fetch_all_registered_users(self):
        if not self.is_connected: return None
        container = []
        def target_worker():
            try: 
                res = self.client.table("registered_faces").select("*").execute()
                container.append(res.data)
            except Exception as e:
                print(f"[Supabase Error] Gagal mengambil data: {e}")

        t = threading.Thread(target=target_worker, daemon=True)
        t.start()
        t.join(timeout=5.0)
        return container[0] if container else None

# ==============================================================================
# 4. MAIN FACADE
# ==============================================================================
class FaceDatabase:
    def __init__(self, local_db_path="local_faces.json"):
        self.local = LocalStorage(local_db_path)
        self.cloud = CloudStorage()

    def check_user_exists(self, name): return self.local.exists(name)

    def save_face(self, name, embedding, cap_data):
        try:
            payload = DataTransformer.prepare_payload(name, embedding, cap_data)
            self.local.save_user(name, payload)
            
            acc = float(cap_data.get("registration_accuracy", 0.0))
            lc = cap_data.get("light_condition", "N/A")
            
            if self.cloud.is_connected:
                def cloud_upload_sequence():
                    self.cloud.sync_register(name, payload)
                    self.cloud.push_register_log(name, "SUCCESS", acc, lc, "Berhasil didaftarkan", payload)

                print(f"\n[Supabase] ⏳ Sedang menyinkronkan data '{name}' ke Cloud database...")
                upload_thread = threading.Thread(target=cloud_upload_sequence)
                upload_thread.start()
                upload_thread.join(timeout=6.0)
                
                if upload_thread.is_alive():
                    print("[Supabase WARNING] Unggah data memakan waktu terlalu lama. Dilanjutkan di background.")
            return True
        except Exception: 
            traceback.print_exc()
            if self.cloud.is_connected:
                threading.Thread(target=self.cloud.push_register_log, args=(name, "FAILED", 0.0, "N/A", "Sistem error saat menyimpan ke memori"), daemon=True).start()
            return False

    def load_all_faces(self, silent=False):
        faces = self.local.load_all()
        if not faces and self.cloud.is_connected:
            if not silent: print("[Hybrid Sync] Memori lokal kosong. Menarik data vektor dari Supabase...")
            if cloud_data := self.cloud.fetch_all_registered_users():
                faces = {
                    r["name"]: {
                        "name": r["name"], 
                        "embedding": r["embedding"], 
                        "liveness_config": r.get("liveness_config", {}), # 🚨 Liveness Config Ditarik Kembali
                        "yaw_score": r.get("yaw_score", ""),
                        "pitch_score": r.get("pitch_score", ""),
                        "roll_score": r.get("roll_score", ""),
                        "blink_score": r.get("blink_score", ""),
                        "registered_at": r.get("created_at", "")
                    } for r in cloud_data
                }
                self.local.overwrite_all(faces)
                if not silent: print(f"[Hybrid Sync] Berhasil memulihkan {len(faces)} identitas wajah.")
        return faces

    def log_register_async(self, name, status, accuracy, message, light_cond="N/A", cap_data=None):
        if self.cloud.is_connected:
            payload = None
            if cap_data:
                try:
                    payload = DataTransformer.prepare_payload(name, np.zeros(128), cap_data)
                    light_cond = cap_data.get("light_condition", light_cond)
                except Exception: pass
            threading.Thread(target=self.cloud.push_register_log, args=(name, status, accuracy, light_cond, message, payload), daemon=True).start()

    def push_access_log_async(self, user_name, status, accuracy, light_cond="N/A", access_details=None):
        if self.cloud.is_connected:
            threading.Thread(target=self.cloud.push_access_log, args=(user_name, status, accuracy, light_cond, access_details), daemon=True).start()

    def log_spoofing_async(self, score, message):
        if self.cloud.is_connected:
            threading.Thread(target=self.cloud.push_spoofing_log, args=(score, message), daemon=True).start()
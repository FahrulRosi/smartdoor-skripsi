import os, json, time, threading, traceback, sqlite3, numpy as np, uuid
from datetime import datetime
from contextlib import closing 
import requests
from supabase import create_client, Client
import config
from cryptography.fernet import Fernet

# ==============================================================================
# 0. ENCRYPTION HELPER
# ==============================================================================
class EncryptionHelper:
    _fernet = None

    @classmethod
    def get_fernet(cls):
        if cls._fernet is None:
            key = getattr(config, 'ENCRYPTION_KEY', None)
            if not key:
                key = b'VsreqR9RZ65RprbhhhJA5yi4TTt5fzDislDDEJPYy6c='
            if isinstance(key, str):
                key = key.encode()
            cls._fernet = Fernet(key)
        return cls._fernet

    @classmethod
    def encrypt(cls, plaintext: str) -> str:
        if not plaintext or plaintext == "-":
            return plaintext
        try:
            f = cls.get_fernet()
            return f.encrypt(plaintext.encode()).decode()
        except Exception as e:
            print(f"[Encryption Error] Gagal enkripsi: {e}")
            return plaintext

    @classmethod
    def decrypt(cls, ciphertext: str) -> str:
        if not ciphertext or ciphertext == "-":
            return ciphertext
        try:
            f = cls.get_fernet()
            return f.decrypt(ciphertext.encode()).decode()
        except Exception as e:
            return ciphertext

# ==============================================================================
# 1. DATA TRANSFORMER
# ==============================================================================
class DataTransformer:
    @staticmethod
    def prepare_payload(name, user_id, embedding, liveness_data):
        blink_c = liveness_data.get("blink_closed") or {}
        blink_o = liveness_data.get("blink_open") or {}
        
        if len(embedding) > 0 and isinstance(embedding[0], list):
            emb_list = [[float(val) for val in vec] for vec in embedding]
        else:
            emb_list = [float(x) for x in embedding]
        
        yl = yr = pu = pd = rl = rr = 0.0
        yl_lat = yr_lat = pu_lat = pd_lat = rl_lat = rr_lat = 0.0

        for snap in liveness_data.get("yaw_snapshots", []):
            lat = float(snap.get("latensi_ms") or snap.get("latency_ms", 0.0))
            if snap.get("tag") == "yaw_left": yl, yl_lat = float(snap.get("yaw", 0.0)), lat
            elif snap.get("tag") == "yaw_right": yr, yr_lat = float(snap.get("yaw", 0.0)), lat
                
        for snap in liveness_data.get("pitch_snapshots", []):
            lat = float(snap.get("latensi_ms") or snap.get("latency_ms", 0.0))
            if snap.get("tag") == "pitch_up": pu, pu_lat = float(snap.get("pitch", 0.0)), lat
            elif snap.get("tag") == "pitch_down": pd, pd_lat = float(snap.get("pitch", 0.0)), lat
                
        for snap in liveness_data.get("roll_snapshots", []):
            lat = float(snap.get("latensi_ms") or snap.get("latency_ms", 0.0))
            if snap.get("tag") == "roll_left": rl, rl_lat = float(snap.get("roll", 0.0)), lat
            elif snap.get("tag") == "roll_right": rr, rr_lat = float(snap.get("roll", 0.0)), lat

        pose_l = []
        if yl_lat > 0: pose_l.append(f"Yaw Kiri ({yl_lat:.1f}ms)")
        if yr_lat > 0: pose_l.append(f"Yaw Kanan ({yr_lat:.1f}ms)")
        if pu_lat > 0: pose_l.append(f"Pitch Atas ({pu_lat:.1f}ms)")
        if pd_lat > 0: pose_l.append(f"Pitch Bawah ({pd_lat:.1f}ms)")
        if rl_lat > 0: pose_l.append(f"Roll Kiri ({rl_lat:.1f}ms)")
        if rr_lat > 0: pose_l.append(f"Roll Kanan ({rr_lat:.1f}ms)")
        
        blink_l = []
        bo_lat = float(blink_o.get("latensi_ms") or blink_o.get("latency_ms", 0.0))
        bc_lat = float(blink_c.get("latensi_ms") or blink_c.get("latency_ms", 0.0))
        if bo_lat > 0: blink_l.append(f"Mata Membuka ({bo_lat:.1f}ms)")
        if bc_lat > 0: blink_l.append(f"Mata Menutup ({bc_lat:.1f}ms)")

        return {
            "name": str(name), 
            "id": str(user_id),
            "embedding": emb_list,
            "pose_data": ", ".join(pose_l) if pose_l else "-",
            "blink_data": ", ".join(blink_l) if blink_l else "-",
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

# ==============================================================================
# 2. MAIN FACADE
# ==============================================================================
class FaceDatabase:
    def __init__(self, local_db_path="local_faces.db"):
        self.db_path = local_db_path
        db_exists = os.path.exists(self.db_path)
        self.is_new_db = not db_exists

        journal_path = self.db_path + "-journal"
        if os.path.exists(journal_path):
            try: os.remove(journal_path)
            except Exception: pass

        self.db_lock = threading.Lock()
        self.sync_trigger = threading.Event() 
        self._init_sqlite()

        self.url, self.key = getattr(config, "SUPABASE_URL", ""), getattr(config, "SUPABASE_KEY", "")
        self.is_connected = False
        
        if self.url and self.key:
            try:
                self.client: Client = create_client(self.url, self.key)
                self.is_connected = True
                print("[Supabase] Terhubung ke Cloud Database PostgreSQL.")
            except Exception as e: 
                print(f"[Supabase WARNING] Gagal inisialisasi awal: {e}")

        threading.Thread(target=self._https_sync_worker, daemon=True).start()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
        conn.execute("PRAGMA foreign_keys = ON;") 
        conn.execute("PRAGMA journal_mode = MEMORY;") 
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_sqlite(self):
        with self.db_lock, closing(self._get_connection()) as conn:
            try: conn.execute("VACUUM;")
            except Exception: pass
            c = conn.cursor()
            
            # Skema migration: Drop tabel lama jika kolom lama masih ada
            c.execute("PRAGMA table_info(registered_faces)")
            cols = [col[1] for col in c.fetchall()]
            if cols and "liveness_config" in cols:
                c.execute("DROP TABLE IF EXISTS register_logs")
                c.execute("DROP TABLE IF EXISTS registered_faces")

            c.execute('''CREATE TABLE IF NOT EXISTS registered_faces (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, embedding TEXT NOT NULL,
                reg_latency_ms REAL, created_at TEXT, is_synced INTEGER DEFAULT 0)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS register_logs (
                id TEXT PRIMARY KEY, name TEXT, 
                user_id TEXT REFERENCES registered_faces(id) ON DELETE CASCADE, 
                status TEXT NOT NULL, pose_data TEXT, blink_data TEXT,
                light_condition TEXT, reg_latency_ms REAL, created_at TEXT, is_synced INTEGER DEFAULT 0)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS access_logs (
                id TEXT PRIMARY KEY, name TEXT, 
                user_id TEXT REFERENCES registered_faces(id) ON DELETE SET NULL, 
                status TEXT NOT NULL, face_val_latency_ms REAL, headpose_data TEXT, blink_data TEXT, accuracy REAL, light_condition TEXT,
                auth_latency_ms REAL, created_at TEXT, is_synced INTEGER DEFAULT 0)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS spoofing_logs (
                id TEXT PRIMARY KEY, spoof_score REAL NOT NULL, spoof_type TEXT NOT NULL,
                spoof_latency_ms REAL, created_at TEXT, is_synced INTEGER DEFAULT 0)''')

            c.execute('''CREATE TABLE IF NOT EXISTS sync_deletes (
                id INTEGER PRIMARY KEY AUTOINCREMENT, table_name TEXT, record_id TEXT)''')

            triggers = [("registered_faces", "id"), ("register_logs", "id"), ("access_logs", "id"), ("spoofing_logs", "id")]
            for tbl, col in triggers:
                c.execute(f'''CREATE TRIGGER IF NOT EXISTS trg_del_{tbl} AFTER DELETE ON {tbl} FOR EACH ROW BEGIN INSERT INTO sync_deletes (table_name, record_id) VALUES ('{tbl}', OLD.{col}); END;''')
            conn.commit()

    def _is_online(self):
        try:
            requests.head("https://1.1.1.1", timeout=3)
            return True
        except requests.exceptions.RequestException:
            return False

    def check_user_exists(self, user_id):
        with self.db_lock, closing(self._get_connection()) as conn:
            c = conn.cursor()
            c.execute("SELECT 1 FROM registered_faces WHERE id = ?", (user_id,))
            return c.fetchone() is not None

    def save_face(self, name, user_id, embedding, cap_data):
        try:
            pure_name = name.rsplit('_', 1)[0] if "_" in name else name
            p = DataTransformer.prepare_payload(pure_name, user_id, embedding, cap_data)
            
            encrypted_emb = EncryptionHelper.encrypt(json.dumps(p["embedding"]))
            
            lc = cap_data.get("light_condition", "Normal")
            created_at = p.get("registered_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            reg_lat = float(cap_data.get("reg_latency_ms", 0.0))
            
            log_id = str(uuid.uuid4())

            with self.db_lock, closing(self._get_connection()) as conn:
                c = conn.cursor()
                
                c.execute("""INSERT INTO registered_faces 
                    (id, name, embedding, reg_latency_ms, created_at, is_synced)
                    VALUES (?, ?, ?, ?, ?, 0)
                    ON CONFLICT(id) DO UPDATE SET 
                    name=excluded.name, embedding=excluded.embedding,
                    reg_latency_ms=excluded.reg_latency_ms, created_at=excluded.created_at, is_synced=0""",
                    (user_id, pure_name, encrypted_emb, 
                     reg_lat, created_at))
                
                c.execute("""INSERT INTO register_logs (id, name, user_id, status, pose_data, blink_data, light_condition, reg_latency_ms, created_at, is_synced)
                             VALUES (?, ?, ?, 'SUCCESS', ?, ?, ?, ?, ?, 0)""",
                             (log_id, pure_name, user_id, p["pose_data"], p["blink_data"], lc, reg_lat, created_at))
                conn.commit()
            
            print(f"\n[Database] ✅ Master Wajah '{pure_name} ({user_id})' berhasil disimpan. Memulai sinkronisasi Cloud...")
            self.sync_trigger.set() 
            return True
        except Exception as e: 
            print(f"❌ [Database Error] Gagal save_face ke SQLite lokal: {e}")
            return False

    def load_all_faces(self, silent=False):
        faces = {}
        with self.db_lock, closing(self._get_connection()) as conn:
            c = conn.cursor()
            c.execute("SELECT id, name, embedding, reg_latency_ms, created_at FROM registered_faces")
            for row in c.fetchall():
                decrypted_emb = EncryptionHelper.decrypt(row[2])
                label_id = f"{row[0]} - {row[1]}"
                faces[label_id] = {
                    "id": row[0], "name": row[1], "embedding": json.loads(decrypted_emb),
                    "reg_latency_ms": row[3], "registered_at": row[4]
                }
        self.sync_trigger.set() 
        return faces

    def _pull_logs_from_supabase(self, cursor, table_name, columns):
        res = self.client.table(table_name).select("*").execute()
        if res.data is not None: 
            remote_ids = [str(r["id"]) for r in res.data if "id" in r]
            
            cursor.execute(f"SELECT id FROM {table_name} WHERE is_synced = 1")
            for row in cursor.fetchall():
                local_id = str(row[0])
                if local_id not in remote_ids:
                    cursor.execute(f"DELETE FROM {table_name} WHERE id = ?", (local_id,))
                    cursor.execute("DELETE FROM sync_deletes WHERE table_name = ? AND record_id = ?", (table_name, local_id))

            for r in res.data:
                remote_id = str(r["id"])
                cursor.execute(f"SELECT 1 FROM {table_name} WHERE id = ?", (remote_id,))
                if not cursor.fetchone():
                    if "user_id" in r and r["user_id"] is not None:
                        cursor.execute("SELECT 1 FROM registered_faces WHERE id = ?", (r["user_id"],))
                        if not cursor.fetchone(): continue

                    vals = []
                    for c in columns:
                        v = r.get(c)
                        if v is None:
                            num_fields = ["accuracy", "spoof_score", "reg_latency_ms", "auth_latency_ms", "spoof_latency_ms", "face_val_latency_ms"]
                            v = 0.0 if c in num_fields else "-"
                        if c == "created_at" and v != "-":
                            v = str(v).replace("+00:00", "Z")
                        vals.append(v)
                    vals.append(1) 

                    placeholders = ", ".join(["?"] * len(vals))
                    cols_str = ", ".join(columns) + ", is_synced"
                    cursor.execute(f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders})", vals)

    def log_register_async(self, name, user_id, status, message=None, light_cond="N/A", cap_data=None):
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        log_id = str(uuid.uuid4())
        def _task():
            pure_name = name.rsplit('_', 1)[0] if "_" in name else name
            pose_val = "-"
            b = "-"
            local_light_cond = light_cond
            reg_lat = 0.0
            if cap_data:
                try: 
                    local_light_cond = cap_data.get("light_condition", light_cond)
                    reg_lat = float(cap_data.get("reg_latency_ms", 0.0))
                    p_dummy = DataTransformer.prepare_payload(pure_name, user_id, [[0.0]*128], cap_data)
                    pose_val, b = p_dummy.get("pose_data", "-"), p_dummy.get("blink_data", "-")
                except Exception: pass

            try:
                with self.db_lock, closing(self._get_connection()) as conn:
                    if status == "SUCCESS":
                        conn.cursor().execute("SELECT 1 FROM registered_faces WHERE id = ?", (user_id,))
                        if not conn.cursor().fetchone(): return
                    
                    conn.execute("""INSERT INTO register_logs (id, name, user_id, status, pose_data, blink_data, light_condition, reg_latency_ms, created_at, is_synced)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""", (log_id, pure_name, user_id, status, pose_val, b, local_light_cond, reg_lat, created_at))
                    conn.commit()
                self.sync_trigger.set() 
            except Exception: pass
        threading.Thread(target=_task, daemon=True).start()

    def push_access_log_async(self, user_name, user_id, status, accuracy, light_cond="N/A", access_details=None, auth_latency_ms=0.0, face_val_latency_ms=0.0):
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        log_id = str(uuid.uuid4())
        def _task():
            headpose_str, blink_str = "-" , "-"
            if access_details:
                headpose_str, blink_str = "", ""
                for d in access_details:
                    info = f"Berhasil {d.get('tantangan', '')} ({float(d.get('latensi_ms', 0)):.0f} ms)"
                    tl = str(d.get("tantangan", "")).lower()
                    if any(x in tl for x in ["toleh", "dongak", "tunduk", "miring"]): headpose_str += info + " | "
                    elif "kedip" in tl or "mata" in tl: blink_str += info + " | "

            clean_name = user_name.rsplit('_', 1)[0] if "_" in user_name else user_name

            try:
                with self.db_lock, closing(self._get_connection()) as conn:
                    target_user_id = user_id
                    if target_user_id:
                        c_check = conn.cursor()
                        c_check.execute("SELECT 1 FROM registered_faces WHERE id = ?", (target_user_id,))
                        if not c_check.fetchone(): target_user_id = None
                    
                    conn.execute("""INSERT INTO access_logs (id, name, user_id, status, face_val_latency_ms, headpose_data, blink_data, accuracy, light_condition, auth_latency_ms, created_at, is_synced)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""", (log_id, clean_name, target_user_id, status, float(face_val_latency_ms), headpose_str.strip(" | ") or "-", blink_str.strip(" | ") or "-", float(accuracy), light_cond, float(auth_latency_ms), created_at))
                    conn.commit()
                self.sync_trigger.set() 
            except Exception: pass
        threading.Thread(target=_task, daemon=True).start()

    def log_spoofing_async(self, score_real, score_photo, score_video, spoof_label, spoof_latency_ms=0.0):
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        log_id = str(uuid.uuid4())
        def _task():
            try:
                with self.db_lock, closing(self._get_connection()) as conn:
                    conn.execute("INSERT INTO spoofing_logs (id, spoof_score, spoof_type, spoof_latency_ms, created_at, is_synced) VALUES (?, ?, ?, ?, ?, 0)", (log_id, float(score_real), str(spoof_label), float(spoof_latency_ms), created_at))
                    conn.commit()
                self.sync_trigger.set()
            except Exception: pass
        threading.Thread(target=_task, daemon=True).start()

    def _https_sync_worker(self):
        time.sleep(3) 
        while True:
            self.sync_trigger.wait(timeout=3.0) 
            self.sync_trigger.clear() 
            
            if not self.is_connected or not self._is_online(): continue 
            
            # Deteksi jika file database dihapus secara fisik saat program sedang berjalan
            if not os.path.exists(self.db_path) and not getattr(self, 'is_new_db', False):
                self.is_new_db = True

            if getattr(self, 'is_new_db', False):
                print("\n[Sync] ⚠️ Database lokal baru dideteksi (local_faces.db dihapus/kosong). Membersihkan data Cloud Supabase...")
                try:
                    dummy_uuid = "00000000-0000-0000-0000-000000000000"
                    self.client.table("register_logs").delete().neq("id", dummy_uuid).execute()
                    self.client.table("access_logs").delete().neq("id", dummy_uuid).execute()
                    self.client.table("spoofing_logs").delete().neq("id", dummy_uuid).execute()
                    self.client.table("registered_faces").delete().neq("id", dummy_uuid).execute()
                    print("[Sync] ✅ Semua data di Cloud Supabase berhasil dibersihkan agar sinkron.")
                    self.is_new_db = False
                except Exception as e:
                    print(f"❌ [Sync Error] Gagal membersihkan data Cloud Supabase: {e}")

            try:
                with self.db_lock, closing(self._get_connection()) as conn:
                    c = conn.cursor()
                    
                    try:
                        c.execute("SELECT id, table_name, record_id FROM sync_deletes")
                        for row in c.fetchall():
                            del_id, tbl, rec_id = row
                            try:
                                self.client.table(tbl).delete().eq('id', rec_id).execute()
                                c.execute("DELETE FROM sync_deletes WHERE id = ?", (del_id,))
                            except Exception: pass
                    except Exception: pass
                    
                    # --- MULAI UKUR LATENSI CLOUD SINKRONISASI ---
                    start_sync_time = time.time()
                    synced_count = 0

                    # 1. Sync Registered Faces
                    try:
                        c.execute("SELECT id, name, embedding, reg_latency_ms, created_at FROM registered_faces WHERE is_synced = 0")
                        for r in c.fetchall():
                            payload = {"id": r[0], "name": r[1], "embedding": r[2], "reg_latency_ms": float(r[3] or 0.0), "created_at": r[4]}
                            try:
                                self.client.table("registered_faces").upsert(payload, on_conflict="id").execute()
                                c.execute("UPDATE registered_faces SET is_synced = 1 WHERE id = ?", (r[0],))
                                synced_count += 1
                            except Exception as e:
                                err = str(e).lower()
                                print(f"❌ [Sync Error] PUSH registered_faces ({r[0]}): {e}")
                                if "could not find" in err or "column" in err:
                                    payload.pop("reg_latency_ms", None)
                                    try: 
                                        self.client.table("registered_faces").upsert(payload, on_conflict="id").execute()
                                        c.execute("UPDATE registered_faces SET is_synced = 1 WHERE id = ?", (r[0],))
                                        synced_count += 1
                                    except Exception: pass
                    except Exception as err_db: print(f"❌ Query Error (registered_faces): {err_db}")
                    
                    # 2. Sync Register Logs (UPDATED to UPSERT)
                    try:
                        c.execute("SELECT id, name, user_id, status, pose_data, blink_data, light_condition, reg_latency_ms, created_at FROM register_logs WHERE is_synced = 0")
                        for r in c.fetchall():
                            payload = {"id": r[0], "name": r[1], "user_id": r[2], "status": r[3], "pose_data": r[4] or "-", "blink_data": r[5] or "-", "light_condition": r[6] or "-", "reg_latency_ms": float(r[7] or 0.0), "created_at": r[8]}
                            try:
                                self.client.table("register_logs").upsert(payload, on_conflict="id").execute()
                                c.execute("UPDATE register_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                                synced_count += 1
                            except Exception as e:
                                err = str(e).lower()
                                print(f"❌ [Sync Error] PUSH register_logs (ID {r[0]}): {e}")
                                if "could not find" in err or "column" in err:
                                    payload.pop("light_condition", None)
                                    payload.pop("reg_latency_ms", None)
                                    try:
                                        self.client.table("register_logs").upsert(payload, on_conflict="id").execute()
                                        c.execute("UPDATE register_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                                        synced_count += 1
                                    except Exception: 
                                        c.execute("UPDATE register_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                                elif "foreign key" in err or "23503" in err:
                                    c.execute("UPDATE register_logs SET is_synced = 1 WHERE id = ?", (r[0],)) 
                    except Exception as err_db: print(f"❌ Query Error (register_logs): {err_db}")
                        
                    # 3. Sync Access Logs (UPDATED to UPSERT)
                    try:
                        c.execute("SELECT id, name, user_id, status, face_val_latency_ms, headpose_data, blink_data, accuracy, light_condition, auth_latency_ms, created_at FROM access_logs WHERE is_synced = 0")
                        for r in c.fetchall():
                            payload = {"id": r[0], "name": r[1], "user_id": r[2], "status": r[3], "face_val_latency_ms": float(r[4] or 0.0), "headpose_data": r[5] or "-", "blink_data": r[6] or "-", "accuracy": float(r[7] or 0.0), "light_condition": r[8] or "-", "auth_latency_ms": float(r[9] or 0.0), "created_at": r[10]}
                            try:
                                self.client.table("access_logs").upsert(payload, on_conflict="id").execute()
                                c.execute("UPDATE access_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                                synced_count += 1
                            except Exception as e:
                                err = str(e).lower()
                                print(f"❌ [Sync Error] PUSH access_logs (ID {r[0]}): {e}")
                                if "could not find" in err or "column" in err:
                                    payload.pop("light_condition", None)
                                    payload.pop("auth_latency_ms", None)
                                    payload.pop("face_val_latency_ms", None)
                                    try:
                                        self.client.table("access_logs").upsert(payload, on_conflict="id").execute()
                                        c.execute("UPDATE access_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                                        synced_count += 1
                                    except Exception: 
                                        c.execute("UPDATE access_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                                elif "foreign key" in err or "23503" in err:
                                    c.execute("UPDATE access_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                    except Exception as err_db: print(f"❌ Query Error (access_logs): {err_db}")
                        
                    # 4. Sync Spoofing Logs (UPDATED to UPSERT)
                    try:
                        c.execute("SELECT id, spoof_score, spoof_type, spoof_latency_ms, created_at FROM spoofing_logs WHERE is_synced = 0")
                        for r in c.fetchall():
                            payload = {"id": r[0], "spoof_score": float(r[1] or 0.0), "spoof_type": str(r[2] or "-"), "spoof_latency_ms": float(r[3] or 0.0), "created_at": r[4]}
                            try:
                                self.client.table("spoofing_logs").upsert(payload, on_conflict="id").execute()
                                c.execute("UPDATE spoofing_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                                synced_count += 1
                            except Exception as e:
                                err = str(e).lower()
                                print(f"❌ [Sync Error] PUSH spoofing_logs (ID {r[0]}): {e}")
                                if "could not find" in err or "column" in err:
                                    payload.pop("spoof_latency_ms", None)
                                    try:
                                        self.client.table("spoofing_logs").upsert(payload, on_conflict="id").execute()
                                        c.execute("UPDATE spoofing_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                                        synced_count += 1
                                    except Exception:
                                        c.execute("UPDATE spoofing_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                    except Exception as err_db: print(f"❌ Query Error (spoofing_logs): {err_db}")
                    
                    # --- CETAK LATENSI UNGGAL REALTIME KE CLOUD ---
                    if synced_count > 0: 
                        latency_cloud_ms = (time.time() - start_sync_time) * 1000
                        print(f"\n[Background Sync] ☁️ Berhasil UPLOAD {synced_count} baris data ke Supabase! | Latensi Cloud: {latency_cloud_ms:.0f} ms")

                    # --- PULL DARI CLOUD ---
                    try:
                        res = self.client.table("registered_faces").select("*").execute()
                        if res.data is not None:
                            remote_ids = [str(r["id"]) for r in res.data]
                            c.execute("SELECT id FROM registered_faces WHERE is_synced = 1")
                            for row in c.fetchall():
                                if row[0] not in remote_ids:
                                    c.execute("DELETE FROM registered_faces WHERE id = ?", (row[0],))
                                    c.execute("DELETE FROM sync_deletes WHERE table_name = 'registered_faces' AND record_id = ?", (row[0],))
                            
                            for r in res.data:
                                remote_cr = str(r.get("created_at", ""))
                                c.execute("""INSERT INTO registered_faces 
                                     (id, name, embedding, reg_latency_ms, created_at, is_synced)
                                     VALUES (?, ?, ?, ?, ?, 1)
                                     ON CONFLICT(id) DO UPDATE SET 
                                     name=excluded.name, embedding=excluded.embedding,
                                     reg_latency_ms=excluded.reg_latency_ms, is_synced=1""",
                                     (r.get("id", "-"), r.get("name", "-"), r.get("embedding", "-"), 
                                      float(r.get("reg_latency_ms", 0.0)), remote_cr))
                    except Exception: pass

                    try: self._pull_logs_from_supabase(c, "register_logs", ["id", "name", "user_id", "status", "pose_data", "blink_data", "light_condition", "reg_latency_ms", "created_at"])
                    except Exception: pass
                    
                    try: self._pull_logs_from_supabase(c, "access_logs", ["id", "name", "user_id", "status", "face_val_latency_ms", "headpose_data", "blink_data", "accuracy", "light_condition", "auth_latency_ms", "created_at"])
                    except Exception: pass
                    
                    try: self._pull_logs_from_supabase(c, "spoofing_logs", ["id", "spoof_score", "spoof_type", "spoof_latency_ms", "created_at"])
                    except Exception: pass
                        
                    conn.commit()
            except Exception as e: 
                print(f"❌ [Fatal Sync Error] Koneksi ke database terganggu: {e}")
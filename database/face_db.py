import os, json, time, threading, traceback, sqlite3, numpy as np
from datetime import datetime
from contextlib import closing 
import requests
from supabase import create_client, Client
import config

# ==============================================================================
# 1. DATA TRANSFORMER
# ==============================================================================
class DataTransformer:
    @staticmethod
    def prepare_payload(name, nim, embedding, liveness_data):
        blink_c = liveness_data.get("blink_closed") or {}
        blink_o = liveness_data.get("blink_open") or {}
        ear_c = float(blink_c.get("avg_ear", 0.0))
        ear_o = float(blink_o.get("avg_ear", 0.0))
        
        # PROTEKSI MATRIKS 2D: Cek apakah ini Multi-Vector (List di dalam List)
        if len(embedding) > 0 and isinstance(embedding[0], list):
            emb_list = [[float(val) for val in vec] for vec in embedding]
        else:
            emb_list = [float(x) for x in embedding]
        
        fm_vector = liveness_data.get("facemesh_vector", [])
        if isinstance(fm_vector, np.ndarray): fm_list = fm_vector.tolist()
        elif fm_vector is not None: fm_list = list(fm_vector)
        else: fm_list = []
            
        hp_vec = liveness_data.get("headpose_vector", [0.0, 0.0, 0.0])

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

        yaw_c, yaw_l = [], []
        if yl_lat > 0: yaw_c.append("Berhasil Yaw Kiri"); yaw_l.append(f"Berhasil Yaw Kiri ({yl_lat:.1f}ms)")
        if yr_lat > 0: yaw_c.append("Berhasil Yaw Kanan"); yaw_l.append(f"Berhasil Yaw Kanan ({yr_lat:.1f}ms)")
        
        pitch_c, pitch_l = [], []
        if pu_lat > 0: pitch_c.append("Berhasil Pitch Atas"); pitch_l.append(f"Berhasil Pitch Atas ({pu_lat:.1f}ms)")
        if pd_lat > 0: pitch_c.append("Berhasil Pitch Bawah"); pitch_l.append(f"Berhasil Pitch Bawah ({pd_lat:.1f}ms)")
        
        roll_c, roll_l = [], []
        if rl_lat > 0: roll_c.append("Berhasil Roll Kiri"); roll_l.append(f"Berhasil Roll Kiri ({rl_lat:.1f}ms)")
        if rr_lat > 0: roll_c.append("Berhasil Roll Kanan"); roll_l.append(f"Berhasil Roll Kanan ({rr_lat:.1f}ms)")
        
        blink_c_list, blink_l = [], []
        bo_lat = float(blink_o.get("latensi_ms") or blink_o.get("latency_ms", 0.0))
        bc_lat = float(blink_c.get("latensi_ms") or blink_c.get("latency_ms", 0.0))
        if bo_lat > 0: blink_c_list.append("Berhasil Mata Membuka"); blink_l.append(f"Berhasil Mata Membuka ({bo_lat:.1f}ms)")
        if bc_lat > 0: blink_c_list.append("Berhasil Mata Menutup"); blink_l.append(f"Berhasil Mata Menutup ({bc_lat:.1f}ms)")

        return {
            "name": str(name), 
            "nim": str(nim),
            "embedding": emb_list,
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
            "yaw_score_clean": ", ".join(yaw_c) if yaw_c else "-",
            "pitch_score_clean": ", ".join(pitch_c) if pitch_c else "-",
            "roll_score_clean": ", ".join(roll_c) if roll_c else "-",
            "blink_score_clean": ", ".join(blink_c_list) if blink_c_list else "-",
            "yaw_score_log": ", ".join(yaw_l) if yaw_l else "-",
            "pitch_score_log": ", ".join(pitch_l) if pitch_l else "-",
            "roll_score_log": ", ".join(roll_l) if roll_l else "-",
            "blink_score_log": ", ".join(blink_l) if blink_l else "-",
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

# ==============================================================================
# 2. MAIN FACADE
# ==============================================================================
class FaceDatabase:
    def __init__(self, local_db_path="local_faces.db"):
        self.db_path = local_db_path
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
            c = conn.cursor()
            
            c.execute('''CREATE TABLE IF NOT EXISTS registered_faces (
                nim TEXT PRIMARY KEY, name TEXT NOT NULL, embedding TEXT NOT NULL, liveness_config TEXT,
                yaw_score TEXT, pitch_score TEXT, roll_score TEXT, blink_score TEXT,
                reg_latency_ms REAL, created_at TEXT, is_synced INTEGER DEFAULT 0)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS register_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, 
                nim TEXT REFERENCES registered_faces(nim) ON DELETE CASCADE, 
                status TEXT NOT NULL, yaw_score TEXT, pitch_score TEXT, roll_score TEXT, blink_score TEXT,
                light_condition TEXT, reg_latency_ms REAL, created_at TEXT, is_synced INTEGER DEFAULT 0)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, 
                nim TEXT REFERENCES registered_faces(nim) ON DELETE SET NULL, 
                status TEXT NOT NULL, face_val_latency_ms REAL, headpose_score TEXT, blink_score TEXT, accuracy REAL, light_condition TEXT,
                auth_latency_ms REAL, created_at TEXT, is_synced INTEGER DEFAULT 0)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS spoofing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, spoof_score REAL NOT NULL, spoof_type TEXT NOT NULL,
                spoof_latency_ms REAL, created_at TEXT, is_synced INTEGER DEFAULT 0)''')

            c.execute('''CREATE TABLE IF NOT EXISTS sync_deletes (
                id INTEGER PRIMARY KEY AUTOINCREMENT, table_name TEXT, record_id TEXT)''')

            triggers = [("registered_faces", "nim"), ("register_logs", "created_at"), ("access_logs", "created_at"), ("spoofing_logs", "created_at")]
            for tbl, col in triggers:
                c.execute(f'''CREATE TRIGGER IF NOT EXISTS trg_del_{tbl} AFTER DELETE ON {tbl} FOR EACH ROW BEGIN INSERT INTO sync_deletes (table_name, record_id) VALUES ('{tbl}', OLD.{col}); END;''')
            conn.commit()

    def _is_online(self):
        try:
            requests.head("https://1.1.1.1", timeout=2)
            return True
        except requests.ConnectionError:
            return False

    def check_user_exists(self, nim):
        with self.db_lock, closing(self._get_connection()) as conn:
            c = conn.cursor()
            c.execute("SELECT 1 FROM registered_faces WHERE nim = ?", (nim,))
            return c.fetchone() is not None

    def save_face(self, name, nim, embedding, cap_data):
        try:
            pure_name = name.rsplit('_', 1)[0] if "_" in name else name
            p = DataTransformer.prepare_payload(pure_name, nim, embedding, cap_data)
            lc = cap_data.get("light_condition", "Normal")
            created_at = p.get("registered_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            reg_lat = float(cap_data.get("reg_latency_ms", 0.0))

            with self.db_lock, closing(self._get_connection()) as conn:
                c = conn.cursor()
                
                c.execute("""INSERT INTO registered_faces 
                    (nim, name, embedding, liveness_config, yaw_score, pitch_score, roll_score, blink_score, reg_latency_ms, created_at, is_synced)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(nim) DO UPDATE SET 
                    name=excluded.name, embedding=excluded.embedding, liveness_config=excluded.liveness_config,
                    yaw_score=excluded.yaw_score, pitch_score=excluded.pitch_score, roll_score=excluded.roll_score,
                    blink_score=excluded.blink_score, reg_latency_ms=excluded.reg_latency_ms, created_at=excluded.created_at, is_synced=0""",
                    (nim, pure_name, json.dumps(p["embedding"]), json.dumps(p.get("liveness_config", {})), 
                     p["yaw_score_clean"] or "-", p["pitch_score_clean"] or "-", p["roll_score_clean"] or "-", p["blink_score_clean"] or "-", reg_lat, created_at))
                
                c.execute("""INSERT INTO register_logs (name, nim, status, yaw_score, pitch_score, roll_score, blink_score, light_condition, reg_latency_ms, created_at, is_synced)
                             VALUES (?, ?, 'SUCCESS', ?, ?, ?, ?, ?, ?, ?, 0)""",
                             (pure_name, nim, p["yaw_score_log"] or "-", p["pitch_score_log"] or "-", p["roll_score_log"] or "-", p["blink_score_log"] or "-", lc, reg_lat, created_at))
                conn.commit()
            
            print(f"\n[Database] ✅ Master Wajah '{pure_name} ({nim})' berhasil disimpan. Memulai sinkronisasi Cloud...")
            self.sync_trigger.set() 
            return True
        except Exception as e: 
            print(f"❌ [Database Error] Gagal save_face ke SQLite lokal: {e}")
            return False

    def load_all_faces(self, silent=False):
        faces = {}
        with self.db_lock, closing(self._get_connection()) as conn:
            c = conn.cursor()
            c.execute("SELECT nim, name, embedding, liveness_config, yaw_score, pitch_score, roll_score, blink_score, reg_latency_ms, created_at FROM registered_faces")
            for row in c.fetchall():
                label_id = f"{row[0]} - {row[1]}"
                faces[label_id] = {
                    "nim": row[0], "name": row[1], "embedding": json.loads(row[2]), "liveness_config": json.loads(row[3]),
                    "yaw_score": row[4], "pitch_score": row[5], "roll_score": row[6], "blink_score": row[7],
                    "reg_latency_ms": row[8], "registered_at": row[9]
                }
        self.sync_trigger.set() 
        return faces

    def _pull_logs_from_supabase(self, cursor, table_name, columns):
        res = self.client.table(table_name).select("*").execute()
        if res.data is not None: 
            remote_times = [str(r["created_at"]).replace("+00:00", "Z") for r in res.data]
            
            cursor.execute(f"SELECT id, created_at FROM {table_name} WHERE is_synced = 1")
            for row in cursor.fetchall():
                local_id, local_time = row[0], str(row[1])
                if local_time not in remote_times:
                    cursor.execute(f"DELETE FROM {table_name} WHERE id = ?", (local_id,))
                    cursor.execute("DELETE FROM sync_deletes WHERE table_name = ? AND record_id = ?", (table_name, local_time))

            for r in res.data:
                remote_created_at = str(r["created_at"]).replace("+00:00", "Z")
                cursor.execute(f"SELECT 1 FROM {table_name} WHERE created_at = ?", (remote_created_at,))
                if not cursor.fetchone():
                    if "nim" in r and r["nim"] is not None:
                        cursor.execute("SELECT 1 FROM registered_faces WHERE nim = ?", (r["nim"],))
                        if not cursor.fetchone(): continue

                    vals = []
                    for c in columns:
                        if c == "created_at": vals.append(remote_created_at)
                        else:
                            v = r.get(c)
                            if v is None:
                                num_fields = ["accuracy", "spoof_score", "reg_latency_ms", "auth_latency_ms", "spoof_latency_ms", "face_val_latency_ms"]
                                v = 0.0 if c in num_fields else "-"
                            vals.append(v)
                    vals.append(1) 

                    placeholders = ", ".join(["?"] * len(vals))
                    cols_str = ", ".join(columns) + ", is_synced"
                    cursor.execute(f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders})", vals)

    def log_register_async(self, name, nim, status, message=None, light_cond="N/A", cap_data=None):
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        def _task():
            pure_name = name.rsplit('_', 1)[0] if "_" in name else name
            y = p = r = b = "-"
            local_light_cond = light_cond
            reg_lat = 0.0
            if cap_data:
                try: 
                    local_light_cond = cap_data.get("light_condition", light_cond)
                    reg_lat = float(cap_data.get("reg_latency_ms", 0.0))
                    p_dummy = DataTransformer.prepare_payload(pure_name, nim, [[0.0]*128], cap_data)
                    y, p, r, b = p_dummy.get("yaw_score_log", "-"), p_dummy.get("pitch_score_log", "-"), p_dummy.get("roll_score_log", "-"), p_dummy.get("blink_score_log", "-")
                except Exception: pass
            try:
                with self.db_lock, closing(self._get_connection()) as conn:
                    if status == "SUCCESS":
                        conn.cursor().execute("SELECT 1 FROM registered_faces WHERE nim = ?", (nim,))
                        if not conn.cursor().fetchone(): return
                    
                    conn.execute("""INSERT INTO register_logs (name, nim, status, yaw_score, pitch_score, roll_score, blink_score, light_condition, reg_latency_ms, created_at, is_synced)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""", (pure_name, nim, status, y, p, r, b, local_light_cond, reg_lat, created_at))
                    conn.commit()
                self.sync_trigger.set() 
            except Exception: pass
        threading.Thread(target=_task, daemon=True).start()

    def push_access_log_async(self, user_name, nim, status, accuracy, light_cond="N/A", access_details=None, auth_latency_ms=0.0, face_val_latency_ms=0.0):
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        def _task():
            headpose_str, blink_str = "-" , "-"
            if access_details:
                headpose_str, blink_str = "", ""
                for d in access_details:
                    info = f"Berhasil {d.get('tantangan', '')} ({float(d.get('latensi_ms', 0)):.0f} ms)"
                    tl = str(d.get("tantangan", "")).lower()
                    if any(x in tl for x in ["toleh", "dongak", "tunduk", "miring"]): headpose_str += info + " | "
                    elif "kedip" in tl or "mata" in tl: blink_str += info + " | "

            try:
                with self.db_lock, closing(self._get_connection()) as conn:
                    clean_name = user_name.rsplit('_', 1)[0] if "_" in user_name else user_name
                    target_nim = nim
                    if target_nim:
                        c_check = conn.cursor()
                        c_check.execute("SELECT 1 FROM registered_faces WHERE nim = ?", (target_nim,))
                        if not c_check.fetchone(): target_nim = None
                    
                    conn.execute("""INSERT INTO access_logs (name, nim, status, face_val_latency_ms, headpose_score, blink_score, accuracy, light_condition, auth_latency_ms, created_at, is_synced)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""", (clean_name, target_nim, status, float(face_val_latency_ms), headpose_str.strip(" | ") or "-", blink_str.strip(" | ") or "-", float(accuracy), light_cond, float(auth_latency_ms), created_at))
                    conn.commit()
                self.sync_trigger.set() 
            except Exception: pass
        threading.Thread(target=_task, daemon=True).start()

    def log_spoofing_async(self, score_real, score_photo, score_video, spoof_label, spoof_latency_ms=0.0):
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        def _task():
            try:
                with self.db_lock, closing(self._get_connection()) as conn:
                    conn.execute("INSERT INTO spoofing_logs (spoof_score, spoof_type, spoof_latency_ms, created_at, is_synced) VALUES (?, ?, ?, ?, 0)", (float(score_real), str(spoof_label), float(spoof_latency_ms), created_at))
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
            
            try:
                with self.db_lock, closing(self._get_connection()) as conn:
                    c = conn.cursor()
                    
                    # 1. Hapus sinkronisasi
                    try:
                        c.execute("SELECT id, table_name, record_id FROM sync_deletes")
                        for row in c.fetchall():
                            del_id, tbl, rec_id = row
                            try:
                                if tbl == 'registered_faces': self.client.table(tbl).delete().eq('nim', rec_id).execute()
                                else: self.client.table(tbl).delete().eq('created_at', rec_id).execute()
                                c.execute("DELETE FROM sync_deletes WHERE id = ?", (del_id,))
                            except Exception: pass
                    except Exception: pass
                    
                    synced_count = 0

                    # 2. Sync Registered Faces
                    try:
                        c.execute("SELECT nim, name, embedding, liveness_config, yaw_score, pitch_score, roll_score, blink_score, reg_latency_ms, created_at FROM registered_faces WHERE is_synced = 0")
                        for r in c.fetchall():
                            payload = {"nim": r[0], "name": r[1], "embedding": json.loads(r[2]), "liveness_config": json.loads(r[3]), "yaw_score": r[4] or "-", "pitch_score": r[5] or "-", "roll_score": r[6] or "-", "blink_score": r[7] or "-", "reg_latency_ms": float(r[8] or 0.0), "created_at": r[9]}
                            try:
                                self.client.table("registered_faces").upsert(payload, on_conflict="nim").execute()
                                c.execute("UPDATE registered_faces SET is_synced = 1 WHERE nim = ?", (r[0],))
                                synced_count += 1
                            except Exception as e:
                                err = str(e).lower()
                                print(f"❌ [Sync Error] PUSH registered_faces ({r[0]}): {e}")
                                # AUTO-HEAL: Buang kolom yang tidak ada di Supabase
                                if "could not find" in err or "column" in err:
                                    payload.pop("reg_latency_ms", None)
                                    try: 
                                        self.client.table("registered_faces").upsert(payload, on_conflict="nim").execute()
                                        c.execute("UPDATE registered_faces SET is_synced = 1 WHERE nim = ?", (r[0],))
                                        synced_count += 1
                                    except Exception: pass
                    except Exception as err_db: print(f"❌ Query Error (registered_faces): {err_db}")
                    
                    # 3. Sync Register Logs (TERPERBAIKI: Evaluasi try per baris data)
                    try:
                        c.execute("SELECT id, name, nim, status, yaw_score, pitch_score, roll_score, blink_score, light_condition, reg_latency_ms, created_at FROM register_logs WHERE is_synced = 0")
                        for r in c.fetchall():
                            payload = {"name": r[1], "nim": r[2], "status": r[3], "yaw_score": r[4] or "-", "pitch_score": r[5] or "-", "roll_score": r[6] or "-", "blink_score": r[7] or "-", "light_condition": r[8] or "-", "reg_latency_ms": float(r[9] or 0.0), "created_at": r[10]}
                            try:
                                self.client.table("register_logs").insert(payload).execute()
                                c.execute("UPDATE register_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                                synced_count += 1
                            except Exception as e:
                                err = str(e).lower()
                                print(f"❌ [Sync Error] PUSH register_logs (ID {r[0]}): {e}")
                                # AUTO-HEAL: Buang kolom yang belum ada di Supabase
                                if "could not find" in err or "column" in err:
                                    payload.pop("light_condition", None)
                                    payload.pop("reg_latency_ms", None)
                                    try:
                                        self.client.table("register_logs").insert(payload).execute()
                                        c.execute("UPDATE register_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                                        synced_count += 1
                                    except Exception: 
                                        c.execute("UPDATE register_logs SET is_synced = 1 WHERE id = ?", (r[0],)) # Bypass permanently
                                elif "foreign key" in err or "23503" in err:
                                    # Bypass FK mismatch agar tidak block antrian selamanya
                                    c.execute("UPDATE register_logs SET is_synced = 1 WHERE id = ?", (r[0],)) 
                    except Exception as err_db: print(f"❌ Query Error (register_logs): {err_db}")
                        
                    # 4. Sync Access Logs
                    try:
                        c.execute("SELECT id, name, nim, status, face_val_latency_ms, headpose_score, blink_score, accuracy, light_condition, auth_latency_ms, created_at FROM access_logs WHERE is_synced = 0")
                        for r in c.fetchall():
                            payload = {"name": r[1], "nim": r[2], "status": r[3], "face_val_latency_ms": float(r[4] or 0.0), "headpose_score": r[5] or "-", "blink_score": r[6] or "-", "accuracy": float(r[7] or 0.0), "light_condition": r[8] or "-", "auth_latency_ms": float(r[9] or 0.0), "created_at": r[10]}
                            try:
                                self.client.table("access_logs").insert(payload).execute()
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
                                        self.client.table("access_logs").insert(payload).execute()
                                        c.execute("UPDATE access_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                                        synced_count += 1
                                    except Exception: 
                                        c.execute("UPDATE access_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                                elif "foreign key" in err or "23503" in err:
                                    c.execute("UPDATE access_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                    except Exception as err_db: print(f"❌ Query Error (access_logs): {err_db}")
                        
                    # 5. Sync Spoofing Logs
                    try:
                        c.execute("SELECT id, spoof_score, spoof_type, spoof_latency_ms, created_at FROM spoofing_logs WHERE is_synced = 0")
                        for r in c.fetchall():
                            payload = {"spoof_score": float(r[1] or 0.0), "spoof_type": str(r[2] or "-"), "spoof_latency_ms": float(r[3] or 0.0), "created_at": r[4]}
                            try:
                                self.client.table("spoofing_logs").insert(payload).execute()
                                c.execute("UPDATE spoofing_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                                synced_count += 1
                            except Exception as e:
                                err = str(e).lower()
                                print(f"❌ [Sync Error] PUSH spoofing_logs (ID {r[0]}): {e}")
                                if "could not find" in err or "column" in err:
                                    payload.pop("spoof_latency_ms", None)
                                    try:
                                        self.client.table("spoofing_logs").insert(payload).execute()
                                        c.execute("UPDATE spoofing_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                                        synced_count += 1
                                    except Exception:
                                        c.execute("UPDATE spoofing_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                    except Exception as err_db: print(f"❌ Query Error (spoofing_logs): {err_db}")
                    
                    if synced_count > 0: print(f"\n[Background Sync] ☁️ Berhasil UPLOAD {synced_count} baris data ke Supabase!")

                    # --- PULL DARI CLOUD ---
                    try:
                        res = self.client.table("registered_faces").select("*").execute()
                        if res.data is not None:
                            remote_nims = [str(r["nim"]) for r in res.data]
                            c.execute("SELECT nim FROM registered_faces WHERE is_synced = 1")
                            for row in c.fetchall():
                                if row[0] not in remote_nims:
                                    c.execute("DELETE FROM registered_faces WHERE nim = ?", (row[0],))
                                    c.execute("DELETE FROM sync_deletes WHERE table_name = 'registered_faces' AND record_id = ?", (row[0],))
                            
                            for r in res.data:
                                remote_cr = str(r.get("created_at", "")).replace("+00:00", "Z")
                                c.execute("""INSERT INTO registered_faces 
                                    (nim, name, embedding, liveness_config, yaw_score, pitch_score, roll_score, blink_score, reg_latency_ms, created_at, is_synced)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                                    ON CONFLICT(nim) DO UPDATE SET 
                                    name=excluded.name, embedding=excluded.embedding, liveness_config=excluded.liveness_config,
                                    yaw_score=excluded.yaw_score, pitch_score=excluded.pitch_score, roll_score=excluded.roll_score,
                                    blink_score=excluded.blink_score, reg_latency_ms=excluded.reg_latency_ms, is_synced=1""",
                                    (r.get("nim", "-"), r.get("name", "-"), json.dumps(r["embedding"]), json.dumps(r.get("liveness_config", {})), 
                                     r.get("yaw_score", "-"), r.get("pitch_score", "-"), r.get("roll_score", "-"), r.get("blink_score", "-"), float(r.get("reg_latency_ms", 0.0)), remote_cr))
                    except Exception: pass

                    try: self._pull_logs_from_supabase(c, "register_logs", ["name", "nim", "status", "yaw_score", "pitch_score", "roll_score", "blink_score", "light_condition", "reg_latency_ms", "created_at"])
                    except Exception: pass
                    
                    try: self._pull_logs_from_supabase(c, "access_logs", ["name", "nim", "status", "face_val_latency_ms", "headpose_score", "blink_score", "accuracy", "light_condition", "auth_latency_ms", "created_at"])
                    except Exception: pass
                    
                    try: self._pull_logs_from_supabase(c, "spoofing_logs", ["spoof_score", "spoof_type", "spoof_latency_ms", "created_at"])
                    except Exception: pass
                        
                    conn.commit()
            except Exception as e: 
                print(f"❌ [Fatal Sync Error] Koneksi ke database terganggu: {e}")
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
        bc_lat = float(blink_c.get("latensi_ms") or blink_c.get("latency_ms", 0.0))
        bo_lat = float(blink_o.get("latensi_ms") or blink_o.get("latency_ms", 0.0))

        emb_list = [float(x) for x in embedding]
        
        fm_vector = liveness_data.get("facemesh_vector", [])
        fm_list = fm_vector.tolist() if isinstance(fm_vector, np.ndarray) else list(fm_vector)
        hp_vec = liveness_data.get("headpose_vector", [0.0, 0.0, 0.0])

        yl = yr = pu = pd = rl = rr = 0.0
        yl_lat = yr_lat = pu_lat = pd_lat = rl_lat = rr_lat = 0.0

        for snap in liveness_data.get("yaw_snapshots", []):
            lat = float(snap.get("latensi_ms") or snap.get("latency_ms", 0.0))
            if snap.get("tag") == "yaw_left": 
                yl, yl_lat = float(snap.get("yaw", 0.0)), lat
            elif snap.get("tag") == "yaw_right": 
                yr, yr_lat = float(snap.get("yaw", 0.0)), lat
                
        for snap in liveness_data.get("pitch_snapshots", []):
            lat = float(snap.get("latensi_ms") or snap.get("latency_ms", 0.0))
            if snap.get("tag") == "pitch_up": 
                pu, pu_lat = float(snap.get("pitch", 0.0)), lat
            elif snap.get("tag") == "pitch_down": 
                pd, pd_lat = float(snap.get("pitch", 0.0)), lat
                
        for snap in liveness_data.get("roll_snapshots", []):
            lat = float(snap.get("latensi_ms") or snap.get("latency_ms", 0.0))
            if snap.get("tag") == "roll_left": 
                rl, rl_lat = float(snap.get("roll", 0.0)), lat
            elif snap.get("tag") == "roll_right": 
                rr, rr_lat = float(snap.get("roll", 0.0)), lat

        yaw_score_clean = f"L:{yl:.1f}° R:{yr:.1f}°"
        pitch_score_clean = f"U:{pu:.1f}° D:{pd:.1f}°"
        roll_score_clean = f"L:{rl:.1f}° R:{rr:.1f}°"
        blink_score_clean = f"Buka:{ear_o:.2f} Kedip:{ear_c:.2f}"

        yaw_score_log = f"L:{yl:.1f}° ({yl_lat:.1f}ms) R:{yr:.1f}° ({yr_lat:.1f}ms)"
        pitch_score_log = f"U:{pu:.1f}° ({pu_lat:.1f}ms) D:{pd:.1f}° ({pd_lat:.1f}ms)"
        roll_score_log = f"L:{rl:.1f}° ({rl_lat:.1f}ms) R:{rr:.1f}° ({rr_lat:.1f}ms)"
        blink_score_log = f"Buka:{ear_o:.2f} ({bo_lat:.1f}ms) Kedip:{ear_c:.2f} ({bc_lat:.1f}ms)"

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
            "yaw_score_clean": yaw_score_clean,
            "pitch_score_clean": pitch_score_clean,
            "roll_score_clean": roll_score_clean,
            "blink_score_clean": blink_score_clean,
            "yaw_score_log": yaw_score_log,
            "pitch_score_log": pitch_score_log,
            "roll_score_log": roll_score_log,
            "blink_score_log": blink_score_log,
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }

# ==============================================================================
# 2. MAIN FACADE (HYBRID SQLITE + HTTPS SUPABASE - REAL-TIME SYNC)
# ==============================================================================
class FaceDatabase:
    def __init__(self, local_db_path="local_faces.db"):
        self.db_path = local_db_path
        
        journal_path = self.db_path + "-journal"
        if os.path.exists(journal_path):
            try:
                os.remove(journal_path)
                print(f"[Database] ✅ File sisa '{journal_path}' berhasil dihapus otomatis.")
            except Exception as e:
                print(f"[Database WARNING] Gagal menghapus '{journal_path}': {e}")

        self.db_lock = threading.Lock()
        self.sync_trigger = threading.Event() 
        self._init_sqlite()

        self.url, self.key = getattr(config, "SUPABASE_URL", ""), getattr(config, "SUPABASE_KEY", "")
        self.is_connected = False
        
        if self.url and self.key:
            try:
                self.client: Client = create_client(self.url, self.key)
                self.is_connected = True
                print("[Supabase] Terhubung ke Cloud Database PostgreSQL (Mode Edge Offline-First).")
            except Exception as e: 
                print(f"[Supabase WARNING] Gagal inisialisasi awal: {e}")

        threading.Thread(target=self._https_sync_worker, daemon=True).start()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10.0)
        conn.execute("PRAGMA journal_mode = MEMORY;") 
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_sqlite(self):
        with self.db_lock, closing(self._get_connection()) as conn:
            c = conn.cursor()
            
            c.execute('''CREATE TABLE IF NOT EXISTS registered_faces (
                nim TEXT PRIMARY KEY, name TEXT, embedding TEXT, liveness_config TEXT,
                yaw_score TEXT, pitch_score TEXT, roll_score TEXT, blink_score TEXT,
                created_at TEXT, is_synced INTEGER DEFAULT 0)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS register_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, nim TEXT, status TEXT,
                yaw_score TEXT, pitch_score TEXT, roll_score TEXT, blink_score TEXT,
                light_condition TEXT, created_at TEXT, is_synced INTEGER DEFAULT 0)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, nim TEXT, status TEXT,
                headpose_score TEXT, blink_score TEXT, accuracy REAL, light_condition TEXT,
                created_at TEXT, is_synced INTEGER DEFAULT 0)''')
            
            # [PERBAIKAN BESAR] Menyesuaikan nama kolom sesuai dengan SQL Supabase Anda
            c.execute('''CREATE TABLE IF NOT EXISTS spoofing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, spoof_score REAL, spoof_type TEXT,
                created_at TEXT, is_synced INTEGER DEFAULT 0)''')

            c.execute('''CREATE TABLE IF NOT EXISTS sync_deletes (
                id INTEGER PRIMARY KEY AUTOINCREMENT, table_name TEXT, record_id TEXT)''')

            triggers = [
                ("registered_faces", "nim"),
                ("register_logs", "created_at"),
                ("access_logs", "created_at"),
                ("spoofing_logs", "created_at")
            ]
            for tbl, col in triggers:
                c.execute(f'''CREATE TRIGGER IF NOT EXISTS trg_del_{tbl} 
                              AFTER DELETE ON {tbl} 
                              FOR EACH ROW BEGIN 
                                  INSERT INTO sync_deletes (table_name, record_id) VALUES ('{tbl}', OLD.{col}); 
                              END;''')
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
            p = DataTransformer.prepare_payload(name, nim, embedding, cap_data)
            lc = cap_data.get("light_condition", "N/A")
            created_at = p.get("registered_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

            with self.db_lock, closing(self._get_connection()) as conn:
                c = conn.cursor()
                c.execute("""INSERT INTO registered_faces 
                    (nim, name, embedding, liveness_config, yaw_score, pitch_score, roll_score, blink_score, created_at, is_synced)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(nim) DO UPDATE SET 
                    name=excluded.name, embedding=excluded.embedding, liveness_config=excluded.liveness_config,
                    yaw_score=excluded.yaw_score, pitch_score=excluded.pitch_score, roll_score=excluded.roll_score,
                    blink_score=excluded.blink_score, created_at=excluded.created_at, is_synced=0""",
                    (nim, name, json.dumps(p["embedding"]), json.dumps(p.get("liveness_config", {})), 
                     p["yaw_score_clean"] or "-", p["pitch_score_clean"] or "-", p["roll_score_clean"] or "-", p["blink_score_clean"] or "-", created_at))
                
                c.execute("""INSERT INTO register_logs 
                    (name, nim, status, yaw_score, pitch_score, roll_score, blink_score, light_condition, created_at, is_synced)
                    VALUES (?, ?, 'SUCCESS', ?, ?, ?, ?, ?, ?, 0)""",
                    (name, nim, p["yaw_score_log"] or "-", p["pitch_score_log"] or "-", p["roll_score_log"] or "-", p["blink_score_log"] or "-", lc, created_at))
                conn.commit()
            
            print(f"\n[Database] ✅ Wajah '{name} ({nim})' tersimpan instan ke SQLite LOKAL. Memicu sinkronisasi ke Supabase...")
            self.sync_trigger.set() 
            return True
        except Exception: 
            traceback.print_exc()
            self.log_register_async(name, nim, "FAILED", light_cond="N/A", cap_data=cap_data)
            return False

    def load_all_faces(self, silent=False):
        faces = {}
        with self.db_lock, closing(self._get_connection()) as conn:
            c = conn.cursor()
            c.execute("SELECT nim, name, embedding, liveness_config, yaw_score, pitch_score, roll_score, blink_score, created_at FROM registered_faces")
            for row in c.fetchall():
                label_id = f"{row[0]}_{row[1]}" 
                faces[label_id] = {
                    "nim": row[0], "name": row[1], "embedding": json.loads(row[2]), "liveness_config": json.loads(row[3]),
                    "yaw_score": row[4], "pitch_score": row[5], "roll_score": row[6], "blink_score": row[7],
                    "registered_at": row[8]
                }
        self.sync_trigger.set() 
        return faces

    def _pull_logs_from_supabase(self, cursor, table_name, columns):
        res = self.client.table(table_name).select("*").execute()
        if res.data is not None: 
            remote_times = [str(r["created_at"]) for r in res.data]
            
            cursor.execute(f"SELECT id, created_at FROM {table_name} WHERE is_synced = 1")
            for row in cursor.fetchall():
                local_id, local_time = row[0], str(row[1])
                if local_time not in remote_times:
                    cursor.execute(f"DELETE FROM {table_name} WHERE id = ?", (local_id,))
                    cursor.execute("DELETE FROM sync_deletes WHERE table_name = ? AND record_id = ?", (table_name, local_time))

            for r in res.data:
                cursor.execute(f"SELECT 1 FROM {table_name} WHERE created_at = ?", (r["created_at"],))
                if not cursor.fetchone():
                    placeholders = ", ".join(["?"] * (len(columns) + 1))
                    cols_str = ", ".join(columns) + ", is_synced"
                    # [PERBAIKAN] Cek numerik untuk akurasi dan spoof_score
                    vals = [r.get(c, (0.0 if c in ["accuracy", "spoof_score"] else "-")) for c in columns] + [1]
                    cursor.execute(f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders})", vals)

    # --- FUNGSI LOGGING BACKGROUND (ANTI-LAG) ---

    def log_register_async(self, name, nim, status, message=None, light_cond="N/A", cap_data=None):
        def _task():
            y = p = r = b = "-"
            local_light_cond = light_cond
            if cap_data:
                try: 
                    local_light_cond = cap_data.get("light_condition", light_cond)
                    p_dummy = DataTransformer.prepare_payload(name, nim, [0.0]*128, cap_data)
                    y = p_dummy.get("yaw_score_log", "-")
                    p = p_dummy.get("pitch_score_log", "-")
                    r = p_dummy.get("roll_score_log", "-")
                    b = p_dummy.get("blink_score_log", "-")
                except Exception: pass
                
            created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            try:
                with self.db_lock, closing(self._get_connection()) as conn:
                    conn.execute("""INSERT INTO register_logs 
                        (name, nim, status, yaw_score, pitch_score, roll_score, blink_score, light_condition, created_at, is_synced)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                        (name, nim, status, y or "-", p or "-", r or "-", b or "-", local_light_cond, created_at))
                    conn.commit()
                self.sync_trigger.set() 
            except Exception as e:
                print(f"[DB Error] log_register_async: {e}")
                
        threading.Thread(target=_task, daemon=True).start()

    def push_access_log_async(self, user_name, nim, status, accuracy, light_cond="N/A", access_details=None):
        def _task():
            headpose_str, blink_str = "-", "-"
            if access_details:
                headpose_str, blink_str = "", ""
                for d in access_details:
                    info = f"{d.get('tantangan', '')}: {float(d.get('skor_asli', 0)):.2f} (Tgt: {float(d.get('target', 0)):.2f}, Lat: {float(d.get('latensi_ms', 0)):.0f}ms)"
                    tl = str(d.get("tantangan", "")).lower()
                    if any(x in tl for x in ["toleh", "dongak", "tunduk", "miring"]): headpose_str += info + " | "
                    elif "kedip" in tl or "mata" in tl: blink_str += info + " | "

            created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            try:
                with self.db_lock, closing(self._get_connection()) as conn:
                    conn.execute("""INSERT INTO access_logs 
                        (name, nim, status, headpose_score, blink_score, accuracy, light_condition, created_at, is_synced)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                        (user_name, nim, status, headpose_str.strip(" | ") or "-", blink_str.strip(" | ") or "-", float(accuracy), light_cond, created_at))
                    conn.commit()
                self.sync_trigger.set() 
            except Exception as e:
                print(f"[DB Error] push_access_log_async: {e}")
                
        threading.Thread(target=_task, daemon=True).start()

    def log_spoofing_async(self, score_real, score_photo, score_video, spoof_label):
        """
        [PERBAIKAN JENIUS] Tetap menerima 4 argumen dari main.py/register.py agar tidak error,
        tetapi program otomatis mengekstrak `score_real` dan `spoof_label` agar sesuai dengan SQL Supabase Anda.
        """
        def _task():
            created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            try:
                with self.db_lock, closing(self._get_connection()) as conn:
                    conn.execute("INSERT INTO spoofing_logs (spoof_score, spoof_type, created_at, is_synced) VALUES (?, ?, ?, 0)", 
                                 (float(score_real), str(spoof_label), created_at))
                    conn.commit()
                self.sync_trigger.set()
            except Exception as e:
                print(f"[DB Error] log_spoofing_async: {e}")
                
        threading.Thread(target=_task, daemon=True).start()

    # ==============================================================================
    # 3. BACKGROUND WORKER: LIVE 2-WAY SYNC YANG TAHAN ERROR (ANTI CRASH)
    # ==============================================================================
    def _https_sync_worker(self):
        time.sleep(3) 
        
        while True:
            self.sync_trigger.wait(timeout=3.0) 
            self.sync_trigger.clear() 
            
            if not self.is_connected or not self._is_online(): continue 
            
            try:
                with self.db_lock, closing(self._get_connection()) as conn:
                    c = conn.cursor()
                    
                    try:
                        c.execute("SELECT id, table_name, record_id FROM sync_deletes")
                        for row in c.fetchall():
                            del_id, tbl, rec_id = row
                            try:
                                if tbl == 'registered_faces': self.client.table(tbl).delete().eq('nim', rec_id).execute()
                                else: self.client.table(tbl).delete().eq('created_at', rec_id).execute()
                                
                                c.execute("DELETE FROM sync_deletes WHERE id = ?", (del_id,))
                            except Exception as e: pass
                    except Exception as e: pass
                    
                    synced_count = 0
                    
                    try:
                        c.execute("SELECT nim, name, embedding, liveness_config, yaw_score, pitch_score, roll_score, blink_score, created_at FROM registered_faces WHERE is_synced = 0")
                        for r in c.fetchall():
                            payload = {"nim": r[0], "name": r[1], "embedding": json.loads(r[2]), "liveness_config": json.loads(r[3]), "yaw_score": r[4] or "-", "pitch_score": r[5] or "-", "roll_score": r[6] or "-", "blink_score": r[7] or "-", "created_at": r[8]}
                            self.client.table("registered_faces").upsert(payload, on_conflict="nim").execute()
                            c.execute("UPDATE registered_faces SET is_synced = 1 WHERE nim = ?", (r[0],))
                            synced_count += 1
                    except Exception as e: pass
                    
                    try:
                        c.execute("SELECT id, name, nim, status, yaw_score, pitch_score, roll_score, blink_score, light_condition, created_at FROM register_logs WHERE is_synced = 0")
                        for r in c.fetchall():
                            payload = {"name": r[1], "nim": r[2], "status": r[3], "yaw_score": r[4] or "-", "pitch_score": r[5] or "-", "roll_score": r[6] or "-", "blink_score": r[7] or "-", "light_condition": r[8] or "-", "created_at": r[9]}
                            self.client.table("register_logs").insert(payload).execute()
                            c.execute("UPDATE register_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                            synced_count += 1
                    except Exception as e: pass
                        
                    try:
                        c.execute("SELECT id, name, nim, status, headpose_score, blink_score, accuracy, light_condition, created_at FROM access_logs WHERE is_synced = 0")
                        for r in c.fetchall():
                            payload = {"name": r[1], "nim": r[2], "status": r[3], "headpose_score": r[4] or "-", "blink_score": r[5] or "-", "accuracy": float(r[6] or 0.0), "light_condition": r[7] or "-", "created_at": r[8]}
                            self.client.table("access_logs").insert(payload).execute()
                            c.execute("UPDATE access_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                            synced_count += 1
                    except Exception as e: pass
                        
                    # [PERBAIKAN] Push payload spoofing_logs disesuaikan dengan 2 field: spoof_score dan spoof_type
                    try:
                        c.execute("SELECT id, spoof_score, spoof_type, created_at FROM spoofing_logs WHERE is_synced = 0")
                        for r in c.fetchall():
                            payload = {
                                "spoof_score": float(r[1] or 0.0), 
                                "spoof_type": str(r[2] or "-"), 
                                "created_at": r[3]
                            }
                            self.client.table("spoofing_logs").insert(payload).execute()
                            c.execute("UPDATE spoofing_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                            synced_count += 1
                    except Exception as e: print(f"[Sync Error] Gagal push spoofing_logs: {e}")
                    
                    if synced_count > 0: print(f"\n[Background Sync] ☁️ Berhasil UPLOAD {synced_count} baris data ke Supabase!")

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
                                c.execute("""INSERT INTO registered_faces 
                                    (nim, name, embedding, liveness_config, yaw_score, pitch_score, roll_score, blink_score, created_at, is_synced)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                                    ON CONFLICT(nim) DO UPDATE SET 
                                    name=excluded.name, embedding=excluded.embedding, liveness_config=excluded.liveness_config,
                                    yaw_score=excluded.yaw_score, pitch_score=excluded.pitch_score, roll_score=excluded.roll_score,
                                    blink_score=excluded.blink_score, is_synced=1""",
                                    (r.get("nim", "-"), r.get("name", "-"), json.dumps(r["embedding"]), json.dumps(r.get("liveness_config", {})), 
                                     r.get("yaw_score", "-"), r.get("pitch_score", "-"), r.get("roll_score", "-"), r.get("blink_score", "-"), r.get("created_at", "")))
                    except Exception as e: pass

                    try: self._pull_logs_from_supabase(c, "register_logs", ["name", "nim", "status", "yaw_score", "pitch_score", "roll_score", "blink_score", "light_condition", "created_at"])
                    except Exception as e: pass
                    
                    try: self._pull_logs_from_supabase(c, "access_logs", ["name", "nim", "status", "headpose_score", "blink_score", "accuracy", "light_condition", "created_at"])
                    except Exception as e: pass
                    
                    # [PERBAIKAN] Pull data dari Supabase menggunakan 2 field saja
                    try: self._pull_logs_from_supabase(c, "spoofing_logs", ["spoof_score", "spoof_type", "created_at"])
                    except Exception as e: print(f"[Sync Error] Gagal pull spoofing_logs: {e}")
                        
                    conn.commit()
            except Exception as e: 
                print(f"[Fatal Sync Error] Koneksi ke database lokal terganggu: {e}")
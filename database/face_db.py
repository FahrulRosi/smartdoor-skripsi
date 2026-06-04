import os, json, time, threading, traceback, sqlite3, numpy as np
from datetime import datetime
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

        # ✅ FORMAT 1: TANPA LATENSI (Untuk tabel registered_faces)
        yaw_score_clean = f"L:{yl:.1f}° R:{yr:.1f}°"
        pitch_score_clean = f"U:{pu:.1f}° D:{pd:.1f}°"
        roll_score_clean = f"L:{rl:.1f}° R:{rr:.1f}°"
        blink_score_clean = f"Buka:{ear_o:.2f} Kedip:{ear_c:.2f}"

        # ✅ FORMAT 2: DENGAN LATENSI LENGKAP (Untuk tabel register_logs)
        yaw_score_log = f"L:{yl:.1f}° ({yl_lat:.1f}ms) R:{yr:.1f}° ({yr_lat:.1f}ms)"
        pitch_score_log = f"U:{pu:.1f}° ({pu_lat:.1f}ms) D:{pd:.1f}° ({pd_lat:.1f}ms)"
        roll_score_log = f"L:{rl:.1f}° ({rl_lat:.1f}ms) R:{rr:.1f}° ({rr_lat:.1f}ms)"
        blink_score_log = f"Buka:{ear_o:.2f} ({bo_lat:.1f}ms) Kedip:{ear_c:.2f} ({bc_lat:.1f}ms)"

        return {
            "name": str(name), 
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
        self.db_lock = threading.Lock()
        self.sync_trigger = threading.Event() 
        self._init_sqlite()

        self.url, self.key = getattr(config, "SUPABASE_URL", ""), getattr(config, "SUPABASE_KEY", "")
        self.is_connected = False
        if self.url and self.key:
            try:
                self.client: Client = create_client(self.url, self.key)
                self.is_connected = True
                print("[Supabase] Terhubung ke Cloud Database PostgreSQL (Mode Hybrid SQLite).")
            except Exception as e: 
                print(f"[Supabase WARNING] Gagal inisialisasi: {e}")

        threading.Thread(target=self._https_sync_worker, daemon=True).start()

    def _init_sqlite(self):
        with self.db_lock, sqlite3.connect(self.db_path, check_same_thread=False) as conn:
            c = conn.cursor()
            # Tabel master dikembalikan kolom skornya (tapi nanti diisi tanpa latensi)
            c.execute('''CREATE TABLE IF NOT EXISTS registered_faces (
                name TEXT PRIMARY KEY, embedding TEXT, liveness_config TEXT,
                yaw_score TEXT, pitch_score TEXT, roll_score TEXT, blink_score TEXT,
                created_at TEXT, is_synced INTEGER DEFAULT 0)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS register_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, status TEXT,
                yaw_score TEXT, pitch_score TEXT, roll_score TEXT, blink_score TEXT,
                accuracy REAL, light_condition TEXT, created_at TEXT, is_synced INTEGER DEFAULT 0)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS access_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, status TEXT,
                headpose_score TEXT, blink_score TEXT, accuracy REAL, light_condition TEXT,
                created_at TEXT, is_synced INTEGER DEFAULT 0)''')
            
            c.execute('''CREATE TABLE IF NOT EXISTS spoofing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, spoof_score REAL, message TEXT,
                created_at TEXT, is_synced INTEGER DEFAULT 0)''')
            conn.commit()

    def check_user_exists(self, name):
        with self.db_lock, sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT 1 FROM registered_faces WHERE name = ?", (name,))
            return c.fetchone() is not None

    def save_face(self, name, embedding, cap_data):
        try:
            p = DataTransformer.prepare_payload(name, embedding, cap_data)
            acc = float(cap_data.get("registration_accuracy", 0.0))
            lc = cap_data.get("light_condition", "N/A")
            created_at = p.get("registered_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

            with self.db_lock, sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                # 1. Simpan ke master data menggunakan format *_clean (Tanpa Latensi)
                c.execute("""INSERT INTO registered_faces 
                    (name, embedding, liveness_config, yaw_score, pitch_score, roll_score, blink_score, created_at, is_synced)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(name) DO UPDATE SET 
                    embedding=excluded.embedding, liveness_config=excluded.liveness_config,
                    yaw_score=excluded.yaw_score, pitch_score=excluded.pitch_score, roll_score=excluded.roll_score,
                    blink_score=excluded.blink_score, created_at=excluded.created_at, is_synced=0""",
                    (name, json.dumps(p["embedding"]), json.dumps(p.get("liveness_config", {})), 
                     p["yaw_score_clean"] or "-", p["pitch_score_clean"] or "-", p["roll_score_clean"] or "-", p["blink_score_clean"] or "-", created_at))
                
                # 2. Simpan ke log pendaftaran menggunakan format *_log (Dengan Latensi)
                c.execute("""INSERT INTO register_logs 
                    (name, status, yaw_score, pitch_score, roll_score, blink_score, accuracy, light_condition, created_at, is_synced)
                    VALUES (?, 'SUCCESS', ?, ?, ?, ?, ?, ?, ?, 0)""",
                    (name, p["yaw_score_log"] or "-", p["pitch_score_log"] or "-", p["roll_score_log"] or "-", p["blink_score_log"] or "-", acc, lc, created_at))
                conn.commit()
            
            print(f"\n[Database] ✅ Wajah '{name}' tersimpan instan ke SQLite. Sinkronisasi Cloud dijadwalkan...")
            self.sync_trigger.set() 
            return True
        except Exception: 
            traceback.print_exc()
            self.log_register_async(name, "FAILED", 0.0, light_cond="N/A", cap_data=cap_data)
            return False

    def load_all_faces(self, silent=False):
        faces = {}
        with self.db_lock, sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT name, embedding, liveness_config, yaw_score, pitch_score, roll_score, blink_score, created_at FROM registered_faces")
            for row in c.fetchall():
                faces[row[0]] = {
                    "name": row[0], "embedding": json.loads(row[1]), "liveness_config": json.loads(row[2]),
                    "yaw_score": row[3], "pitch_score": row[4], "roll_score": row[5], "blink_score": row[6],
                    "registered_at": row[7]
                }
        
        if not faces and self.is_connected:
            if not silent: print("[Hybrid Sync] SQLite lokal kosong. Mengunduh data dari Cloud Supabase...")
            try:
                res = self.client.table("registered_faces").select("*").execute()
                if res.data:
                    with self.db_lock, sqlite3.connect(self.db_path) as conn:
                        c = conn.cursor()
                        for r in res.data:
                            c.execute("""INSERT OR IGNORE INTO registered_faces 
                                (name, embedding, liveness_config, yaw_score, pitch_score, roll_score, blink_score, created_at, is_synced)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                                (r["name"], json.dumps(r["embedding"]), json.dumps(r.get("liveness_config", {})), 
                                 r.get("yaw_score", "-"), r.get("pitch_score", "-"), r.get("roll_score", "-"), r.get("blink_score", "-"), r.get("created_at", "")))
                            faces[r["name"]] = {"name": r["name"], "embedding": r["embedding"], "liveness_config": r.get("liveness_config", {}), "registered_at": r.get("created_at", "")}
                        conn.commit()
                    if not silent: print(f"[Hybrid Sync] Berhasil memulihkan {len(faces)} identitas wajah.")
            except Exception as e: print(f"[Cloud Error] Gagal unduh data master: {e}")
            
        return faces

    def log_register_async(self, name, status, accuracy, message=None, light_cond="N/A", cap_data=None):
        y = p = r = b = "-"
        if cap_data:
            try: 
                light_cond = cap_data.get("light_condition", light_cond)
                p_dummy = DataTransformer.prepare_payload(name, [0.0]*128, cap_data)
                y = p_dummy.get("yaw_score_log", "-")
                p = p_dummy.get("pitch_score_log", "-")
                r = p_dummy.get("roll_score_log", "-")
                b = p_dummy.get("blink_score_log", "-")
            except Exception: pass
            
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self.db_lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""INSERT INTO register_logs 
                (name, status, yaw_score, pitch_score, roll_score, blink_score, accuracy, light_condition, created_at, is_synced)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (name, status, y or "-", p or "-", r or "-", b or "-", float(accuracy), light_cond, created_at))
            conn.commit()
        
        self.sync_trigger.set() 

    def push_access_log_async(self, user_name, status, accuracy, light_cond="N/A", access_details=None):
        headpose_str, blink_str = "-", "-"
        if access_details:
            headpose_str, blink_str = "", ""
            for d in access_details:
                info = f"{d.get('tantangan', '')}: {float(d.get('skor_asli', 0)):.2f} (Tgt: {float(d.get('target', 0)):.2f}, Lat: {float(d.get('latensi_ms', 0)):.0f}ms)"
                tl = str(d.get("tantangan", "")).lower()
                if any(x in tl for x in ["toleh", "dongak", "tunduk", "miring"]): headpose_str += info + " | "
                elif "kedip" in tl or "mata" in tl: blink_str += info + " | "

        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self.db_lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("""INSERT INTO access_logs 
                (name, status, headpose_score, blink_score, accuracy, light_condition, created_at, is_synced)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)""",
                (user_name, status, headpose_str.strip(" | ") or "-", blink_str.strip(" | ") or "-", float(accuracy), light_cond, created_at))
            conn.commit()
            
        self.sync_trigger.set() 

    def log_spoofing_async(self, score, message):
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with self.db_lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO spoofing_logs (spoof_score, message, created_at, is_synced) VALUES (?, ?, ?, 0)", 
                         (float(score), message or "-", created_at))
            conn.commit()
            
        self.sync_trigger.set() 

    # ==============================================================================
    # 3. BACKGROUND WORKER: REAL-TIME EVENT DRIVEN SYNC VIA HTTPS
    # ==============================================================================
    def _https_sync_worker(self):
        time.sleep(3) 
        
        while True:
            self.sync_trigger.wait(timeout=10.0)
            self.sync_trigger.clear() 
            
            if self.is_connected:
                try:
                    with self.db_lock, sqlite3.connect(self.db_path) as conn:
                        c = conn.cursor()
                        synced_count = 0
                        
                        # A. Upload Registered Faces Baru
                        c.execute("SELECT name, embedding, liveness_config, yaw_score, pitch_score, roll_score, blink_score, created_at FROM registered_faces WHERE is_synced = 0")
                        for r in c.fetchall():
                            payload = {
                                "name": r[0], 
                                "embedding": json.loads(r[1]), 
                                "liveness_config": json.loads(r[2]), 
                                "yaw_score": r[3] or "-",
                                "pitch_score": r[4] or "-",
                                "roll_score": r[5] or "-",
                                "blink_score": r[6] or "-",
                                "created_at": r[7]
                            }
                            self.client.table("registered_faces").upsert(payload, on_conflict="name").execute()
                            conn.execute("UPDATE registered_faces SET is_synced = 1 WHERE name = ?", (r[0],))
                            synced_count += 1
                        
                        # B. Upload Register Logs
                        c.execute("SELECT id, name, status, yaw_score, pitch_score, roll_score, blink_score, accuracy, light_condition, created_at FROM register_logs WHERE is_synced = 0")
                        for r in c.fetchall():
                            payload = {
                                "name": r[1], "status": r[2],
                                "yaw_score": r[3] or "-",
                                "pitch_score": r[4] or "-",
                                "roll_score": r[5] or "-",
                                "blink_score": r[6] or "-",
                                "accuracy": float(r[7] or 0.0),
                                "light_condition": r[8] or "-",
                                "created_at": r[9]
                            }
                            self.client.table("register_logs").insert(payload).execute()
                            conn.execute("UPDATE register_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                            synced_count += 1
                            
                        # ... (Bagian Access Logs dan Spoofing Logs tidak berubah, sama seperti kode sebelumnya) ...
                        c.execute("SELECT id, name, status, headpose_score, blink_score, accuracy, light_condition, created_at FROM access_logs WHERE is_synced = 0")
                        for r in c.fetchall():
                            payload = {"name": r[1], "status": r[2], "headpose_score": r[3] or "-", "blink_score": r[4] or "-", "accuracy": float(r[5] or 0.0), "light_condition": r[6] or "-", "created_at": r[7]}
                            self.client.table("access_logs").insert(payload).execute()
                            conn.execute("UPDATE access_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                            synced_count += 1
                            
                        c.execute("SELECT id, spoof_score, message, created_at FROM spoofing_logs WHERE is_synced = 0")
                        for r in c.fetchall():
                            payload = {"spoof_score": float(r[1] or 0.0), "message": r[2] or "-", "created_at": r[3]}
                            self.client.table("spoofing_logs").insert(payload).execute()
                            conn.execute("UPDATE spoofing_logs SET is_synced = 1 WHERE id = ?", (r[0],))
                            synced_count += 1
                            
                        conn.commit()
                        
                        if synced_count > 0:
                            print(f"\n[HTTPS Worker] ☁️ Berhasil melakukan PUSH {synced_count} baris data SQLite ke Supabase secara Real-Time!")
                            
                except Exception as e:
                    pass
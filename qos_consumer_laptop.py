"""
================================================================================
QoS TESTING - CONSUMER (Di Jalankan di Laptop)
================================================================================
Fungsi:
1. Merepresentasikan Aplikasi Pemantau (Dasbor) di Laptop.
2. Melakukan polling ke Supabase untuk mendapatkan data baru yang diunggah oleh Raspberry Pi.
3. Mendeteksi Waktu Publish (T_Publish) dari payload dan membandingkannya dengan
   Waktu Terima Lokal (T_Consume) untuk menghitung Latensi End-to-End secara murni.

Cara Pakai:
1. Pastikan jam di Laptop TERSINKRON dengan jam Raspberry Pi via server NTP!
   Windows: Settings > Time & Language > Date & Time > Klik "Sync Now"
2. Jalankan: python qos_consumer_laptop.py
3. Setelah tertulis "Menunggu data masuk...", jalankan skrip publisher di Raspberry Pi.
================================================================================
"""

import os, sys, time, sqlite3, statistics, csv
from datetime import datetime
from contextlib import closing

# Pastikan path project tersedia
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from supabase import create_client

# ==============================================================================
# KONFIGURASI CONSUMER
# ==============================================================================
SUB_CONFIG = {
    "db_path": "qos_consumer_laptop_test.db",
    "polling_interval_sec": 1.0,     # Seberapa sering Laptop mengecek Supabase
    "inactivity_timeout_sec": 20.0,  # Berhenti jika tidak ada data baru selama 20 detik
    "cleanup_after_test": True
}

def cprint(text, color_code="\033[0m"):
    print(f"{color_code}{text}\033[0m")

class LaptopConsumerTester:
    def __init__(self):
        self.supabase_url = getattr(config, "SUPABASE_URL", "")
        self.supabase_key = getattr(config, "SUPABASE_KEY", "")
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL atau SUPABASE_KEY tidak ditemukan di config.py")
        
        cprint("[INIT] Menyiapkan SQLite Lokal Sementara...", "\033[93m")
        self._init_sqlite()
        
        cprint("[INIT] Menghubungkan ke Supabase Cloud...", "\033[93m")
        self.client = create_client(self.supabase_url, self.supabase_key)
        cprint("✅ Siap Menerima Data dari Raspberry Pi!\n", "\033[92m")
        
        self.local_ids = {"registered_faces": set(), "register_logs": set(), "access_logs": set(), "spoofing_logs": set()}
        self.metrics = {"registered_faces": [], "register_logs": [], "access_logs": [], "spoofing_logs": []}

    def _get_connection(self):
        conn = sqlite3.connect(SUB_CONFIG["db_path"], check_same_thread=False)
        return conn

    def _init_sqlite(self):
        with closing(self._get_connection()) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS registered_faces (id TEXT PRIMARY KEY)''')
            c.execute('''CREATE TABLE IF NOT EXISTS register_logs (id TEXT PRIMARY KEY)''')
            c.execute('''CREATE TABLE IF NOT EXISTS access_logs (id TEXT PRIMARY KEY)''')
            c.execute('''CREATE TABLE IF NOT EXISTS spoofing_logs (id TEXT PRIMARY KEY)''')
            conn.commit()

    def _extract_t_pub(self, record, table_name):
        """Mengekstrak T_Publish dari payload string yang dibuat oleh Raspberry Pi"""
        try:
            if table_name == "spoofing_logs":
                tag = record.get("spoof_type", "")
            else:
                tag = record.get("name", "")
            
            if "QoSTest_" in tag:
                parts = tag.split("_")
                return float(parts[1])  # Extract time.time() float
        except:
            pass
        return None

    def run(self):
        cprint("="*70, "\033[96m")
        cprint(" 📡 MEMULAI QOS CONSUMER (SUPABASE -> LAPTOP)", "\033[1m\033[96m")
        cprint("    Pastikan Anda sudah menyinkronkan jam dengan Raspberry Pi (NTP)!", "\033[91m")
        cprint("="*70, "\033[96m")
        
        tables = ["registered_faces", "register_logs", "access_logs", "spoofing_logs"]
        last_receive_time = time.time()
        total_received = 0
        
        cprint("\n⏳ Menunggu data masuk dari Raspberry Pi... (Jalankan publisher di Pi sekarang)\n", "\033[93m")
        
        try:
            with closing(self._get_connection()) as conn:
                c = conn.cursor()
                while True:
                    current_time = time.time()
                    
                    if total_received > 0 and (current_time - last_receive_time) > SUB_CONFIG["inactivity_timeout_sec"]:
                        cprint("\n🛑 Timeout tercapai. Tidak ada data baru yang masuk. Mengakhiri pengujian...", "\033[93m")
                        break
                        
                    new_data_found = False
                    
                    for table in tables:
                        try:
                            # Pull dari Supabase
                            res = self.client.table(table).select("*").like("name" if table != "spoofing_logs" else "spoof_type", "%QoSTest_%").execute()
                            if res.data:
                                for record in res.data:
                                    rec_id = record["id"]
                                    if rec_id not in self.local_ids[table]:
                                        t_consume = time.time()
                                        t_pub = self._extract_t_pub(record, table)
                                        
                                        if t_pub is not None:
                                            # End-to-End Latency = Waktu Terima di Laptop - Waktu Tulis di Pi
                                            latency_ms = (t_consume - t_pub) * 1000
                                            self.metrics[table].append(latency_ms)
                                            
                                            c.execute(f"INSERT INTO {table} (id) VALUES (?)", (rec_id,))
                                            self.local_ids[table].add(rec_id)
                                            
                                            total_received += 1
                                            new_data_found = True
                                            last_receive_time = time.time()
                                            
                                            print(f"   📥 [PULL] {table} | ID: {rec_id[:8]} | End-to-End Latency: {latency_ms:.1f}ms")
                                conn.commit()
                        except Exception as e:
                            cprint(f"   ⚠️ Error polling {table}: {e}", "\033[91m")
                    
                    if not new_data_found:
                        time.sleep(SUB_CONFIG["polling_interval_sec"])

        except KeyboardInterrupt:
            cprint("\n🛑 Dihentikan manual oleh pengguna.", "\033[93m")
            
        self._print_report()
        self._cleanup()

    def _print_report(self):
        cprint("\n" + "=" * 70, "\033[96m")
        cprint("          HASIL PENGUKURAN QOS (RASPBERRY PI -> LAPTOP)", "\033[1m\033[96m")
        cprint("=" * 70, "\033[96m")
        
        all_lats = []
        
        for table, latencies in self.metrics.items():
            count = len(latencies)
            if count == 0: continue
            
            cprint(f"\n  📊 Tabel: {table} ({count} record ditarik)", "\033[1m")
            print(f"  {'─' * 50}")
            print(f"  │ Rata-rata Latency : {statistics.mean(latencies):.2f} ms")
            print(f"  │ Minimum Latency   : {min(latencies):.2f} ms")
            print(f"  │ Maximum Latency   : {max(latencies):.2f} ms")
            if count > 1:
                print(f"  │ Jitter (Stdev)    : {statistics.stdev(latencies):.2f} ms")
            
            all_lats.extend(latencies)
            
        if all_lats:
            cprint(f"\n{'=' * 70}", "\033[96m")
            cprint(f"  KESELURUHAN ({len(all_lats)} Record)", "\033[1m\033[92m")
            cprint(f"{'=' * 70}", "\033[96m")
            print(f"  Total Rata-rata Latency End-to-End : {statistics.mean(all_lats):.2f} ms")
            if len(all_lats) > 1:
                print(f"  Total Jitter (Standar Deviasi)     : {statistics.stdev(all_lats):.2f} ms")
            
            self._export_csv(all_lats)
        else:
            cprint("\n❌ Tidak ada data uji yang diterima.", "\033[91m")

    def _export_csv(self, all_lats):
        filename = f"qos_endtoend_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["=" * 80])
            writer.writerow(["LAPORAN QoS (RASPBERRY PI -> SUPABASE -> LAPTOP) - Smart Door Lock"])
            writer.writerow(["=" * 80])
            writer.writerow([])
            
            writer.writerow(["HASIL PER TABEL"])
            writer.writerow(["Tabel", "Jumlah Record", "Avg Latency (ms)", "Min Latency (ms)", "Max Latency (ms)", "Jitter (ms)"])
            
            for table, latencies in self.metrics.items():
                if latencies:
                    writer.writerow([
                        table, len(latencies),
                        round(statistics.mean(latencies), 2),
                        round(min(latencies), 2), round(max(latencies), 2),
                        round(statistics.stdev(latencies), 2) if len(latencies)>1 else 0
                    ])
            
            writer.writerow([])
            writer.writerow(["DETAIL LATENSI PER RECORD"])
            writer.writerow(["Tabel", "Latency End-to-End (ms)"])
            for table, latencies in self.metrics.items():
                for lat in latencies:
                    writer.writerow([table, round(lat, 2)])
                    
        cprint(f"\n📄 Hasil CSV disimpan ke: {filepath}\n", "\033[92m")

    def _cleanup(self):
        if not SUB_CONFIG["cleanup_after_test"]: return
        cprint("🧹 Membersihkan data uji dari Supabase...", "\033[93m")
        tables = ["registered_faces", "register_logs", "access_logs", "spoofing_logs"]
        try:
            for table in tables:
                ids = list(self.local_ids[table])
                for i in range(0, len(ids), 10):
                    batch_ids = ids[i:i+10]
                    self.client.table(table).delete().in_("id", batch_ids).execute()
        except Exception as e:
            pass
            
        if os.path.exists(SUB_CONFIG["db_path"]):
            try: os.remove(SUB_CONFIG["db_path"])
            except: pass
        cprint("✅ Pembersihan selesai.\n", "\033[92m")

if __name__ == "__main__":
    LaptopConsumerTester().run()

"""
================================================================================
QoS TESTING - PUBLISHER (Di Jalankan di Raspberry Pi)
================================================================================
Fungsi:
1. Menyomulasikan perangkat keras di pintu (Smart Door Lock) yang sedang beroperasi.
2. Menggunakan class `FaceDatabase` bawaan sistem Anda untuk menyimpan log registrasi,
   akses, dan spoofing ke SQLite lokal.
3. Background worker di `face_db.py` secara alamiah akan mengunggahnya ke Supabase.
4. Skrip menyisipkan Timestamp lokal ke dalam nama pengguna/label untuk dihitung
   oleh Laptop (Consumer).

Syarat: Sinkronkan jam Raspberry Pi Anda dengan NTP!
        sudo systemctl stop systemd-timesyncd
        sudo ntpd -gq
        sudo systemctl start systemd-timesyncd
================================================================================
"""

import os, sys, time, uuid, json
import numpy as np
from datetime import datetime

# Pastikan path project tersedia
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database.face_db import FaceDatabase

PUB_CONFIG = {
    "iterations_per_table": 10,
    "delay_between_records": 1.0, # Jeda waktu (detik) antar pengiriman untuk memberi napas pada _https_sync_worker
}

def cprint(text, color_code="\033[0m"):
    print(f"{color_code}{text}\033[0m")

class PiPublisherTester:
    def __init__(self):
        cprint("[INIT] Menginisialisasi FaceDatabase (SQLite + Supabase Sync Worker)...", "\033[93m")
        # Menggunakan database lokal spesifik agar tidak merusak data asli
        self.db = FaceDatabase(local_db_path="qos_test_local.db")
        cprint("✅ FaceDatabase siap dan Worker Sinkronisasi berjalan!\n", "\033[92m")

    def _simulate_save_face(self):
        t_pub = time.time()
        user_id = str(uuid.uuid4())
        name = f"QoSTest_{t_pub}_{user_id[:4]}"
        
        embedding = [[float(np.random.randn()) for _ in range(128)]]
        cap_data = {
            "reg_latency_ms": 1500.0,
            "light_condition": "Normal",
            "facemesh_vector": [0.1]*30,
            "blink_closed": {"avg_ear": 0.15},
            "blink_open": {"avg_ear": 0.30},
            "headpose_vector": [0.0, 0.0, 0.0]
        }
        
        # Ini akan insert ke SQLite dan is_synced=0. Background worker akan ambil alih
        self.db.save_face(name, user_id, embedding, cap_data)
        return t_pub

    def _simulate_register_log(self):
        t_pub = time.time()
        user_id = str(uuid.uuid4())
        name = f"QoSTest_{t_pub}_{user_id[:4]}"
        # Kita menggunakan log_register_async bawaan
        self.db.log_register_async(name, user_id, "SUCCESS", light_cond="Normal")
        return t_pub

    def _simulate_access_log(self):
        t_pub = time.time()
        user_id = str(uuid.uuid4())
        name = f"QoSTest_{t_pub}_{user_id[:4]}"
        # Kita menggunakan push_access_log_async bawaan
        self.db.push_access_log_async(
            user_name=name, 
            user_id=None, # None agar tidak kena constraint Foreign Key jika wajah aslinya tdk ada
            status="UNLOCKED", 
            accuracy=99.9, 
            light_cond="Normal", 
            access_details=[{"tantangan": "Kedip", "latensi_ms": 500}], 
            auth_latency_ms=2500.0, 
            face_val_latency_ms=150.0
        )
        return t_pub

    def _simulate_spoofing_log(self):
        t_pub = time.time()
        # Masukkan ke dalam label karena spoofing_log tidak punya field name
        spoof_label = f"QoSTest_{t_pub}"
        # Kita menggunakan log_spoofing_async bawaan
        self.db.log_spoofing_async(
            score_real=0.01, 
            score_photo=0.99, 
            score_video=0.0, 
            spoof_label=spoof_label, 
            spoof_latency_ms=100.0
        )
        return t_pub

    def run(self):
        cprint("="*70, "\033[96m")
        cprint(" 🚀 MEMULAI QOS PUBLISHER (RASPBERRY PI -> SQLITE -> SUPABASE)", "\033[1m\033[96m")
        cprint("="*70, "\033[96m")
        
        operations = [
            ("registered_faces", self._simulate_save_face),
            ("register_logs", self._simulate_register_log),
            ("access_logs", self._simulate_access_log),
            ("spoofing_logs", self._simulate_spoofing_log)
        ]
        
        n_iters = PUB_CONFIG["iterations_per_table"]
        
        for table_name, action_func in operations:
            cprint(f"\n📡 Menyimulasikan {n_iters} aksi untuk tabel: {table_name}", "\033[93m")
            
            for i in range(n_iters):
                t_action = action_func()
                print(f"   ✅ [{i+1}/{n_iters}] Aksi dipicu. (Menulis ke SQLite lokal...) Waktu: {t_action:.2f}")
                
                # Memberikan napas agar background worker face_db.py sempat upload ke Supabase
                time.sleep(PUB_CONFIG["delay_between_records"])
                
        cprint("\n" + "="*70, "\033[96m")
        cprint(" 🎉 PUBLISHER SELESAI MENGUBAH DATA", "\033[1m\033[92m")
        cprint(" Penting: Jangan tutup skrip ini langsung. Tunggu sekitar 10 detik agar", "\033[93m")
        cprint(" background worker `_https_sync_worker` selesai mengunggah antrian terakhir.", "\033[93m")
        cprint("="*70 + "\n", "\033[96m")
        
        # Tahan agar worker thread punya waktu upload ke cloud
        try:
            for i in range(15, 0, -1):
                sys.stdout.write(f"\rMenunggu worker selesai upload... {i} detik")
                sys.stdout.flush()
                time.sleep(1)
        except KeyboardInterrupt:
            pass
            
        print("\nSkrip selesai.")

if __name__ == "__main__":
    PiPublisherTester().run()
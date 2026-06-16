import time
import csv
import statistics
import uuid
import os
from datetime import datetime
from supabase import create_client, Client

# ==========================================
# KONFIGURASI SUPABASE
# ==========================================
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://gwwxebdmavlmxxcdrlge.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "sb_publishable_tUsmI6J1BubOlRf9YhtNPQ_F9dWIyOv")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def run_qos_test(iterations=30, delay_between_requests=1.0):
    print(f"\n{'='*50}")
    print(f"[SYSTEM] MEMULAI UJI QoS SINKRONISASI SUPABASE (INSERT + CLEANUP)")
    print(f"{'='*50}")
    print(f"Total Iterasi : {iterations}")
    print(f"Jeda/Iterasi  : {delay_between_requests} detik\n")

    latencies = []
    jitters = []
    csv_data_rows = []

    for i in range(iterations):
        # 1. Siapkan data dummy dengan UUID unik
        dummy_id = str(uuid.uuid4())
        dummy_log = {
            "id": dummy_id,
            "user_name": "QoS_Test_User",
            "status": "UNLOCKED",
            "accuracy": 99.5,
            "light_condition": "Normal",
            "total_latency_ms": 1200,
            "timestamp": datetime.now().isoformat()
        }

        # 2. Mulai catat waktu
        start_time = time.time()

        try:
            # 3. Eksekusi sinkronisasi (Insert)
            # Pastikan "access_logs" sesuai dengan nama tabel Anda
            supabase.table("access_logs").insert(dummy_log).execute()
            
            # 4. Hentikan waktu SEGERA setelah insert berhasil
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # 5. Cleanup: Hapus record yang baru saja dimasukkan
            # Asumsi kolom primary key di tabel Anda bernama 'id'
            supabase.table("access_logs").delete().eq("id", dummy_id).execute()
            
        except Exception as e:
            print(f"❌ Iterasi {i+1} Gagal: {e}")
            continue

        latencies.append(latency_ms)

        # 6. Hitung Jitter
        if len(latencies) == 1:
            jitter_ms = 0.0
        else:
            jitter_ms = abs(latencies[-1] - latencies[-2])
        
        jitters.append(jitter_ms)

        # Simpan baris data untuk CSV
        csv_data_rows.append([i + 1, round(latency_ms, 2), round(jitter_ms, 2)])
        
        print(f"Iterasi {i+1:02d}/{iterations} | Latency Insert: {latency_ms:6.2f} ms | Jitter: {jitter_ms:6.2f} ms | Status: Inserted & Deleted")
        time.sleep(delay_between_requests)

    # ==========================================
    # KALKULASI STATISTIK
    # ==========================================
    if not latencies:
        print("\n❌ Tidak ada data yang berhasil dikirim. Uji dibatalkan.")
        return

    mean_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    std_dev_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    mean_jitter = statistics.mean(jitters)

    print(f"\n{'='*50}")
    print("[HASIL STATISTIK QoS]")
    print(f"Mean Latency   : {mean_latency:.2f} ms")
    print(f"Median Latency : {median_latency:.2f} ms")
    print(f"Std Dev Latency: {std_dev_latency:.2f} ms")
    print(f"Mean Jitter    : {mean_jitter:.2f} ms")
    print(f"{'='*50}\n")

    # ==========================================
    # EKSPOR KE CSV
    # ==========================================
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"qos_metrics_{timestamp_str}.csv"

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow(["Iterasi", "Latency (ms)", "Jitter (ms)"])
        writer.writerows(csv_data_rows)
        
        writer.writerow([]) 
        writer.writerow(["Statistik Total", "Nilai (ms)"])
        writer.writerow(["Mean Latency", round(mean_latency, 2)])
        writer.writerow(["Median Latency", round(median_latency, 2)])
        writer.writerow(["Standar Deviasi Latency", round(std_dev_latency, 2)])
        writer.writerow(["Mean Jitter", round(mean_jitter, 2)])

    print(f"✅ File CSV berhasil disimpan di: {csv_filename}\n")

if __name__ == "__main__":
    run_qos_test(iterations=50, delay_between_requests=1.0)
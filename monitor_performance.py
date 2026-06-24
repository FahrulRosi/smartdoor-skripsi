import os
import sys
import time
import csv
from datetime import datetime

# Mencoba mengimpor psutil. Jika belum ada, beri panduan instalasi.
try:
    import psutil
except ImportError:
    print("\n[ERROR] Library 'psutil' belum terinstal.")
    print("Silakan instal terlebih dahulu menggunakan perintah:")
    print("  pip install psutil --break-system-packages  (untuk Raspberry Pi OS Bookworm)")
    print("  atau")
    print("  sudo apt install python3-psutil\n")
    sys.exit(1)

# Mencoba mengimpor matplotlib untuk visualisasi grafik.
HAS_MATPLOTLIB = False
try:
    import matplotlib
    matplotlib.use('Agg')  # Mencegah crash GUI jika dijalankan headless / via SSH di Raspi
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    pass

# Konfigurasi Monitoring
POLLING_INTERVAL = 0.5  # Detik (seberapa cepat data diperbarui)
CSV_FILE_PREFIX = "performance_log"

def get_cpu_temp():
    """Membaca temperatur CPU Raspberry Pi (hanya bekerja di Linux/Raspberry Pi)"""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = float(f.read().strip()) / 1000.0
            return temp
    except FileNotFoundError:
        # Jika dijalankan di Windows/Mac untuk testing, kembalikan 0.0 atau dummy
        return 0.0

def find_target_process(keywords):
    """Mencari process yang sedang berjalan berdasarkan nama file Python"""
    for proc in psutil.process_iter(['pid', 'cmdline']):
        try:
            cmdline = proc.info['cmdline']
            if cmdline:
                # Gabungkan semua argumen baris perintah menjadi satu string
                cmd_str = " ".join(cmdline)
                for kw in keywords:
                    if kw in cmd_str and "monitor_performance.py" not in cmd_str:
                        return proc.info['pid'], kw, proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None, None, None

def print_summary(data_log, duration):
    if not data_log:
        print("\n[INFO] Tidak ada data performa yang direkam.")
        return

    cpu_vals = [d["proc_cpu"] for d in data_log]
    mem_vals = [d["proc_ram_mb"] for d in data_log]
    temp_vals = [d["cpu_temp"] for d in data_log]

    print("\n" + "=" * 55)
    print("         RINGKASAN PENGUJIAN PERFORMA HARDWARE")
    print("=" * 55)
    print(f"Durasi Monitoring   : {duration:.2f} detik")
    print(f"Jumlah Sampel Data  : {len(data_log)}")
    print("-" * 55)
    print(f"CPU Usage Process   : Min: {min(cpu_vals):.1f}% | Max: {max(cpu_vals):.1f}% | Avg: {sum(cpu_vals)/len(cpu_vals):.2f}%")
    print(f"RAM Usage Process   : Min: {min(mem_vals):.1f} MB | Max: {max(mem_vals):.1f} MB | Avg: {sum(mem_vals)/len(mem_vals):.2f} MB")
    
    # Suhu CPU (Tampilkan hanya jika terdeteksi/di Raspi)
    valid_temps = [t for t in temp_vals if t > 0]
    if valid_temps:
        print(f"Suhu CPU Raspi      : Min: {min(valid_temps):.1f}°C | Max: {max(valid_temps):.1f}°C | Avg: {sum(valid_temps)/len(valid_temps):.2f}°C")
    else:
        print("Suhu CPU Raspi      : Tidak terdeteksi (Bukan Raspberry Pi/OS Linux)")
    print("=" * 55)

def save_to_csv(data_log, target_name):
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{CSV_FILE_PREFIX}_{target_name}_{timestamp_str}.csv"
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    
    fields = ["timestamp", "elapsed_sec", "proc_cpu", "sys_cpu", "proc_ram_mb", "sys_ram_percent", "cpu_temp"]
    
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for data in data_log:
            writer.writerow(data)
            
    print(f"📊 Laporan CSV berhasil disimpan ke: {filepath}")
    return filepath

def generate_plot(data_log, target_name):
    if not HAS_MATPLOTLIB:
        print("💡 Petunjuk: Instal 'matplotlib' (pip install matplotlib) untuk membuat grafik otomatis.")
        return
        
    try:
        elapsed = [d["elapsed_sec"] for d in data_log]
        proc_cpu = [d["proc_cpu"] for d in data_log]
        proc_ram = [d["proc_ram_mb"] for d in data_log]
        cpu_temp = [d["cpu_temp"] for d in data_log]
        
        fig, host = plt.subplots(figsize=(10, 6))
        fig.subplots_adjust(right=0.75)
        
        par1 = host.twinx()
        par2 = host.twinx()
        
        # Geser posisi sumbu y ketiga ke kanan
        par2.spines["right"].set_position(("axes", 1.2))
        
        p1, = host.plot(elapsed, proc_cpu, "r-", label="CPU Process (%)", linewidth=1.5)
        p2, = par1.plot(elapsed, proc_ram, "g-", label="RAM Process (MB)", linewidth=1.5)
        
        plots = [p1, p2]
        
        # Plot suhu jika valid
        has_temp = any(t > 0 for t in cpu_temp)
        if has_temp:
            p3, = par2.plot(elapsed, cpu_temp, "b-", label="Suhu CPU (°C)", linewidth=1.5)
            plots.append(p3)
            par2.set_ylabel("Suhu CPU (°C)")
            par2.yaxis.label.set_color(p3.get_color())
        
        host.set_xlabel("Waktu (detik)")
        host.set_ylabel("CPU Usage (%)")
        par1.set_ylabel("RAM Usage (MB)")
        
        host.yaxis.label.set_color(p1.get_color())
        par1.yaxis.label.set_color(p2.get_color())
        
        host.legend(plots, [l.get_label() for l in plots], loc="upper left")
        plt.title(f"Grafik Performa Hardware - {target_name.upper()}\n(Smart Door Lock System)")
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"performance_plot_{target_name}_{timestamp_str}.png"
        plot_filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), plot_filename)
        
        plt.savefig(plot_filepath, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"📈 Grafik performa berhasil disimpan ke: {plot_filepath}")
    except Exception as e:
        print(f"⚠️ Gagal membuat grafik: {e}")

def main():
    print("=" * 60)
    print("      RASPBERRY PI HARDWARE PERFORMANCE MONITOR")
    print("=" * 60)
    
    targets = ["main.py", "register.py", "qos_publisher_pi.py"]
    print(f"Mencari proses aktif dari: {', '.join(targets)} ...")
    
    # Tunggu dan cari proses sampai ada yang aktif
    pid, kw, proc = None, None, None
    try:
        while True:
            pid, kw, proc = find_target_process(targets)
            if pid:
                break
            sys.stdout.write("\r⏳ Menunggu aplikasi dijalankan (Jalankan 'main.py' atau 'register.py' di terminal lain)...")
            sys.stdout.flush()
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring dibatalkan.")
        return

    print(f"\n\n🚀 Terdeteksi proses target!")
    print(f"   • File Python : {kw}")
    print(f"   • PID         : {pid}")
    print(f"   • Mulai merekam penggunaan CPU, RAM, dan Suhu...")
    print("-" * 60)
    print(f"{'Waktu (s)':<10} | {'CPU Proc (%)':<14} | {'RAM Proc (MB)':<14} | {'Suhu CPU (°C)':<14}")
    print("-" * 60)
    
    data_log = []
    start_time = time.time()
    
    # Inisialisasi awal untuk pembacaan CPU
    try:
        proc.cpu_percent(interval=None)
    except Exception:
        pass
        
    try:
        while proc.is_running():
            t_now = time.time()
            elapsed_sec = t_now - start_time
            
            # Ambil metrik performa
            try:
                # Untuk proses multi-thread, nilai cpu_percent bisa > 100% (misal 4 core = maks 400%).
                # Kita bagi dengan jumlah core agar skalanya sesuai dengan kapasitas sistem total (0% - 100%).
                num_cores = psutil.cpu_count() or 1
                proc_cpu = proc.cpu_percent(interval=None) / num_cores
                
                # Mengatasi nilai pertama kali ambil yang seringkali 0.0
                if elapsed_sec < 1.0 and proc_cpu == 0.0:
                    time.sleep(0.1)
                    proc_cpu = proc.cpu_percent(interval=None) / num_cores
                
                # RAM Usage (RSS) dalam MegaBytes (MB)
                proc_ram_mb = proc.memory_info().rss / (1024 * 1024)
                
                # System Metrics
                sys_cpu = psutil.cpu_percent(interval=None)
                sys_ram_percent = psutil.virtual_memory().percent
                cpu_temp = get_cpu_temp()
                
                data_point = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                    "elapsed_sec": round(elapsed_sec, 2),
                    "proc_cpu": round(proc_cpu, 1),
                    "sys_cpu": round(sys_cpu, 1),
                    "proc_ram_mb": round(proc_ram_mb, 2),
                    "sys_ram_percent": round(sys_ram_percent, 1),
                    "cpu_temp": round(cpu_temp, 1)
                }
                
                data_log.append(data_point)
                
                # Tampilkan ke layar real-time
                temp_str = f"{cpu_temp:.1f} C" if cpu_temp > 0 else "N/A"
                print(f"{elapsed_sec:<10.1f} | {proc_cpu:<14.1f} | {proc_ram_mb:<14.2f} | {temp_str:<14}")
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print(f"\n⏹️ Proses {kw} telah berhenti.")
                break
                
            time.sleep(POLLING_INTERVAL)
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Monitoring dihentikan manual via keyboard.")
        
    duration = time.time() - start_time
    
    # Cetak laporan & Simpan hasil
    print_summary(data_log, duration)
    if data_log:
        csv_path = save_to_csv(data_log, kw.replace(".py", ""))
        generate_plot(data_log, kw.replace(".py", ""))

if __name__ == "__main__":
    main()
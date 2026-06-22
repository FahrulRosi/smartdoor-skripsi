from onnxruntime.quantization import quantize_dynamic, QuantType

def compress_model(input_path, output_path):
    print(f"Memulai kuantisasi untuk: {input_path}")
    try:
        quantize_dynamic(
            model_input=input_path,
            model_output=output_path,
            weight_type=QuantType.QUInt8
        )
        print(f"Sukses! Model diringankan ke: {output_path}\n")
    except Exception as e:
        print(f"Gagal mengkuantisasi {input_path}: {e}")

if __name__ == "__main__":
    # Kuantisasi model Anti-Spoofing
    compress_model("liveness/antispoofing.onnx", "liveness/antispoofing_int8.onnx")
    
    # Kuantisasi model MobileFaceNet
    compress_model("recognition/mobilefacenet.onnx", "recognition/mobilefacenet_int8.onnx")
    
    print("Semua proses optimasi model selesai. Pastikan config.py Anda mengarah ke file _int8.onnx.")
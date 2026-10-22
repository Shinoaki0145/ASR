# ASR

Project thử nghiệm nhận dạng tiếng Việt và hội thoại ngắt lời.

| Thư mục | Nội dung |
|---|---|
| [Sub-ASR](Sub-ASR/README.md) | Ứng dụng CLI/Marimo, notebook hội thoại, runtime, helper ASR/audio và môi trường uv |
| `Compare_Zipformer_Gipformer/` | Notebook và kết quả so sánh Zipformer/Gipformer |
| `Test_Silero_VAD/` | Notebook thử Silero VAD |
| `voice/` | Audio mẫu |
| `tests/` | Test benchmark và notebook VAD |
| `docs/` | Tài liệu thiết kế và kế hoạch |

Notebook hội thoại: [Hoi_Thoai_Ngat_Loi_Testing.ipynb](Sub-ASR/Hoi_Thoai_Ngat_Loi_Testing.ipynb). Cần repo đầy đủ để chạy; trên Colab clone repo rồi `%cd` vào gốc repo. Cấu hình notebook dùng `.env` ở gốc, sao chép từ `.env.example`.

Hai script `gipformer_benchmark.py` và `quality_benchmark.py` ở gốc là source cho benchmark.

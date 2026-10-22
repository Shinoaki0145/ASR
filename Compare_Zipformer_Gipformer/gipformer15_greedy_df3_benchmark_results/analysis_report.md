# Gipformer greedy so với beam 4

Đã tính lại từ 922 dòng per-clip, ghép đủ 461 cặp. Manifest SHA256 giống nhau. Đã đọc đủ 922 waveform raw/DF3: sample rate và giá trị mẫu giống hệt giữa hai lần chạy.

## WER (tổng lỗi / tổng token tham chiếu)

| Cấu hình | Clean raw | Clean DF3 | Noisy raw | Noisy DF3 |
|---|---:|---:|---:|---:|
| Gipformer greedy | 7,41% | 7,96% | 7,67% | 8,40% |
| Gipformer beam 4 | 7,41% | 7,41% | 7,62% | 8,27% |
| Zipformer greedy | 8,52% | 8,15% | 8,78% | 9,16% |

Greedy noisy raw: 870/11340 lỗi, DF3: 953/11340. DF3 tăng 83 lỗi (+0,732 điểm phần trăm); 30 clip tốt hơn, 69 tệ hơn, 321 bằng số lỗi. Beam 4 giảm 6 lỗi raw và 15 lỗi DF3 so với greedy trên 420 clip nhiễu. CER noisy raw greedy 5,098%, beam 5,121%; vì vậy beam không tốt hơn ở mọi chỉ số.

## Latency

Cùng clip vi_000_keyboard_events_snr05 dài 11,34 giây; median 20 lượt sau 5 warm-up, Tesla T4.

| Cấu hình | ASR raw | ASR DF3 |
|---|---:|---:|
| Gipformer greedy | 99,47 ms | 99,58 ms |
| Gipformer beam 4 | 391,38 ms | 396,13 ms |
| Zipformer greedy | 91,87 ms | 94,29 ms |

Greedy nhanh hơn beam 4 khoảng 3,93 lần ở raw. Tổng ước tính DF3 + resample + greedy ASR = 72,47 + 15,23 + 99,58 = 187,28 ms; đây là tổng thời gian các bước đo riêng, không phải phép đo end-to-end trực tiếp.

## Nhiễu và DF3

| Loại nhiễu | WER raw | WER DF3 |
|---|---:|---:|
| Keyboard continuous | 8,02% | 9,69% |
| Keyboard events | 7,96% | 8,40% |
| Knock | 7,53% | 7,65% |
| Doorslam | 7,47% | 7,59% |
| Keys | 7,35% | 8,33% |
| Pageturn | 7,72% | 9,32% |
| Phone | 7,65% | 7,84% |

Noise-only: greedy và beam đều sinh từ trên 21/21 mẫu ở cả raw và DF3. Greedy raw: 16 “Ừ”, 5 “ĐÂY”; DF3: 21 “Ừ”. Chuyển greedy không giải quyết hiện tượng nhận dạng khi không có lời nói.

## Diễn giải

Trong tập thử này, greedy là lựa chọn đáng ưu tiên nếu cần giảm latency: WER tăng ít so với beam 4 nhưng ASR nhanh hơn gần 4 lần. DF3 chưa giúp cải thiện nhận dạng tổng thể ở cả hai decoder; cần đánh giá thêm trước khi bật mặc định cho ASR.

Chỉ có 20 câu nói gốc; 420 clip nhiễu là các biến thể có tương quan, không phải 420 câu độc lập. Latency được đo ở các lần chạy khác nhau, không phải thử nghiệm xen kẽ hai decoder. So sánh với Zipformer vẫn khác model và runtime (PyTorch/ONNX), dù đều dùng greedy. Các kết quả này không chứng minh lợi thế trên toàn bộ dữ liệu thực tế.

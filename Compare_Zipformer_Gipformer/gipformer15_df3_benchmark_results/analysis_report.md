# Phân tích Gipformer 1.5 68M + DeepFilterNet3

Đã kiểm tra 922 dòng per-clip, ghép đủ 461 cặp raw/DF3 và tính lại WER/CER từ số lỗi từng clip. Các tỷ lệ trong quality_summary.csv khớp.

Kết luận trong tập thử: Gipformer + modified beam search (beam=4) có WER thấp hơn cấu hình Zipformer greedy trước đây, nhưng DF3 chưa cải thiện ASR tổng thể. Gipformer hiện sinh chữ trên toàn bộ mẫu noise-only.

## Cấu hình và khả năng đối chiếu

- Model: g-group-ai-lab/gipformer1.5-68M-rnnt, 68,625,511 parameters, version 1.5.
- Runtime: 2.8.0+cu128, CUDA 12.8, Tesla T4.
- Decoder: modified_beam_search, beam=4; threads intra/inter = 2/1.
- CUDA checks đều True cho model parameters, Fbank features, encoder output.
- Manifest SHA256 trùng giữa hai model: a52dfb2432a17898b5436afe32d74655533c2b7be87f186b92817648ca84bc09.
- Đã đọc và đối chiếu toàn bộ 461 raw + 461 DF3 waveforms giữa hai thư mục: sample rate, shape và giá trị mẫu đều giống nhau. SHA256 file WAV có thể khác do header; sai khác lớn nhất của waveform bằng 0.
- So sánh dưới đây là hai cấu hình đầy đủ: Gipformer PyTorch + beam=4 và Zipformer ONNX + greedy. Không tách riêng ảnh hưởng model, decoder hay runtime.

## Chất lượng nhận dạng

| Tập | Số cặp | WER raw | WER DF3 | CER raw | CER DF3 |
|---|---:|---:|---:|---:|---:|
| Câu gốc | 20 | 7.41% | 7.41% | 4.83% | 4.94% |
| Có nhiễu | 420 | 7.62% | 8.27% | 5.12% | 5.60% |

Trên 420 bản trộn: raw 864/11.340 lỗi token, DF3 938/11.340; tăng 74 lỗi, tương đương +0,653 điểm phần trăm WER. 31 bản giảm lỗi, 62 bản tăng, 327 bản giữ nguyên số lỗi. Câu gốc raw/DF3 đều có 40/540 lỗi token; từng câu có thể thay đổi dù tổng bằng nhau.

### Theo nhóm nhiễu — gộp ba SNR

| Nhiễu | Kiểu | WER raw | WER DF3 | ΔWER (điểm %) |
|---|---|---:|---:|---:|
| keyboard | continuous | 7.90% | 9.57% | +1.67 |
| keyboard | events | 7.90% | 8.33% | +0.43 |
| knock | events | 7.47% | 7.41% | -0.06 |
| doorslam | events | 7.47% | 7.53% | +0.06 |
| keys | events | 7.41% | 8.15% | +0.74 |
| pageturn | events | 7.59% | 9.07% | +1.48 |
| phone | events | 7.59% | 7.84% | +0.25 |

### Theo SNR

| Nhiễu | Kiểu | SNR | WER raw | WER DF3 | ΔWER (điểm %) |
|---|---|---:|---:|---:|---:|
| keyboard | continuous | 0 | 8.70% | 10.74% | +2.04 |
| keyboard | continuous | 5 | 7.59% | 9.26% | +1.67 |
| keyboard | continuous | 10 | 7.41% | 8.70% | +1.30 |
| keyboard | events | 0 | 8.52% | 9.44% | +0.93 |
| keyboard | events | 5 | 7.41% | 7.78% | +0.37 |
| keyboard | events | 10 | 7.78% | 7.78% | +0.00 |
| knock | events | 0 | 7.59% | 7.41% | -0.19 |
| knock | events | 5 | 7.59% | 7.41% | -0.19 |
| knock | events | 10 | 7.22% | 7.41% | +0.19 |
| doorslam | events | 0 | 7.59% | 7.96% | +0.37 |
| doorslam | events | 5 | 7.41% | 7.22% | -0.19 |
| doorslam | events | 10 | 7.41% | 7.41% | +0.00 |
| keys | events | 0 | 7.59% | 8.15% | +0.56 |
| keys | events | 5 | 7.22% | 7.96% | +0.74 |
| keys | events | 10 | 7.41% | 8.33% | +0.93 |
| pageturn | events | 0 | 7.96% | 9.81% | +1.85 |
| pageturn | events | 5 | 7.41% | 9.07% | +1.67 |
| pageturn | events | 10 | 7.41% | 8.33% | +0.93 |
| phone | events | 0 | 7.59% | 8.15% | +0.56 |
| phone | events | 5 | 7.78% | 8.15% | +0.37 |
| phone | events | 10 | 7.41% | 7.22% | -0.19 |

## Các mẫu nên nghe

### vi_018_pageturn_events_snr00

Reference: Cơn bão được đặt tên thứ mười của mùa Bão Atlantic, Bão cận nhiệt đới Jerry, đã hình thành tại Biển Atlantic hôm nay.

Raw (4/24 lỗi): CƠN BÃO ĐƯỢC ĐẶT TÊN THỚI MỜI CỦA MÙA BÃO ATLANTIC BÃO CẢNH NHIỆT ĐỚI CHERI ĐÃ HÌNH THÀNH TẠI BIỂN ATLANTIC HÔM NAY

DF3 (9/24 lỗi): CƠN BÃO ĐƯỢC ĐẶT TÊN THỚI MỜI CỦA MÙA BÃO ALANTIC BÃO CẢNH NHIỆT ĐỚI ERRY ĐÃ HỌC HÀNH TẠI BIỂN ANH NĂM NAY

### vi_007_keyboard_continuous_snr00

Reference: Hồ sơ, theo vụ rò rỉ, sẽ nhắc tới xung đột biên giới, điều Palestine muốn dựa trên đường biên giới trước cuộc Chiến tranh Trung Đông 1967.

Raw (8/28 lỗi): HỒ SƠ THEO VỤ GÒ RỈ SẼ NHẮC TỚI XUNG ĐỘT BIÊN GIỚI ĐIỀU BALETINA MUỐN DỰA TRÊN ĐƯỜNG BIÊN GIỚI TRƯỚC CUỘC CHIẾN TRANH TRUNG ĐÔNG MỘT NGHÌN CHÍN TRĂM SÁU BẢY

DF3 (12/28 lỗi): HỒ SƠ THEO VỤ VỪA RỈ SẼ NHẮC TỚI XUNG ĐỘT BIÊN GIỚI ĐỀU BALET ĐI NGANG NẾU DỰA TRÊN ĐƯỜNG BIÊN GIỚI TRƯỚC CUỘC CHIẾN TRANH TRUNG ĐÔNG MỘT NGHÌN CHÍN TRĂM SÁU BẢY

### vi_019_phone_events_snr10

Reference: Sốc tái hòa nhập xảy ra sớm hơn so với sốc văn hóa (ít giai đoạn trăng mật hơn), lâu hơn, và có thể còn nghiêm trọng hơn.

Raw (2/28 lỗi): SỐC TÁI HÒA NHẬP XẢY RA SỚM HƠN SO VỚI SỐC VĂN HÓA ÍT GIAI ĐOẠN CHĂN MẶT HƠN LÂU HƠN VÀ CÓ THỂ CÒN NGHIÊM TRỌNG HƠN

DF3 (0/28 lỗi): SỐC TÁI HÒA NHẬP XẢY RA SỚM HƠN SO VỚI SỐC VĂN HÓA ÍT GIAI ĐOẠN TRĂNG MẬT HƠN LÂU HƠN VÀ CÓ THỂ CÒN NGHIÊM TRỌNG HƠN

### vi_013_keyboard_events_snr00

Reference: Tuy nhiên, sau khi thua gôn của đội trưởng, Ấn Độ chỉ thực hiện được 36 lần chạy và thua 7 gôn trước khi kết thúc hiệp thi đấu.

Raw (7/29 lỗi): TUY NHIÊN SAU KHI THU GOLD CỦA ĐỘI TRƯỞNG ẤN ĐỘ CHỈ THỰC HIỆN ĐƯỢC BA MƯƠI SÁU LẦN CHẠY VÀ THUA BẢY GOLD TRƯỚC KHI KẾT THÚC HIỆP THI ĐẤU

DF3 (5/29 lỗi): TUY NHIÊN SAU KHI THUA GÔN CỦA ĐỘI TRƯỞNG ẤN ĐỘ CHỈ THỰC HIỆN ĐƯỢC BA MƯƠI SÁU LẦN CHẠY VÀ THUA BẢY GOLD TRƯỚC KHI KẾT THÚC HIỆP THI ĐẤU

Riêng vi_018 (Atlantic/Jerry) đóng góp +46/74 lỗi tăng thêm qua 21 điều kiện. vi_007 đóng góp +12. Các bản trộn chỉ xuất phát từ 20 câu gốc, không phải 420 câu nói độc lập; chưa đủ để kết luận DF3 luôn làm giảm chất lượng hoặc xếp hạng model nói chung.

## Noise-only: điểm cần chú ý nhất

Gipformer sinh chữ ở 21/21 mẫu raw và 21/21 mẫu DF3. Raw có 3 “ĐÂY” (keyboard liên tục) và 18 “Ừ”; sau DF3 cả 21 mẫu đều là “Ừ”. Đây là các chữ thực, không phải ký hiệu bị bộ chuẩn hóa bỏ đi.

| Cấu hình | Noise-only raw sinh chữ | Noise-only DF3 sinh chữ |
|---|---:|---:|
| Gipformer PyTorch + modified beam=4 | 21/21 | 21/21 |
| Zipformer ONNX + greedy | 3/21 | 4/21 |

Chỉ có 3 mẫu mỗi loại. Kết quả mô tả hành vi của pipeline hiện tại, chưa xác định nguyên nhân nằm ở decoder, model hay frontend. Không nên suy ra mọi cấu hình Gipformer đều có cùng hành vi.

## So sánh với Zipformer

| Cấu hình | WER câu gốc | WER có nhiễu raw | WER có nhiễu DF3 |
|---|---:|---:|---:|
| Gipformer + beam=4 | 7.41% | 7.62% | 8.27% |
| Zipformer + greedy | 8.52% | 8.78% | 9.16% |

Trên bản trộn raw: Gipformer ít lỗi token hơn ở 135 cặp, nhiều hơn ở 35 cặp và bằng ở 250 cặp. Sau DF3: ít hơn ở 125, nhiều hơn ở 58 và bằng ở 237 cặp. Các cặp lặp cùng câu gốc nên không dùng số đếm này như các quan sát độc lập.

## Latency trên mẫu vi_000_keyboard_events_snr05 (11,34 giây)

| Cấu hình | Median raw | P95 raw | Median sau DF3, chỉ ASR |
|---|---:|---:|---:|
| Gipformer + beam=4 | 391.38 ms | 410.21 ms | 396.13 ms |
| Zipformer + greedy | 91.87 ms | 95.35 ms | 94.29 ms |

Gipformer raw RTF=0.03451; median raw chậm khoảng 4.26 lần cấu hình Zipformer trong hai lần chạy này. Không quy nguyên nhân chỉ cho kiến trúc model vì decoder/runtime khác nhau.

Ước lượng cho mẫu benchmark: DF3 74.84 ms + resample 16.88 ms + ASR 396.13 ms = 487.85 ms. Đây là tổng đo ở worker warm riêng, không phải latency end-to-end trực tiếp.

Memory Torch: peak allocated 316.48 MiB, reserved 342.00 MiB. Không so trực tiếp với PID VRAM NVML của Zipformer vì hai cách đo khác nhau.

## Giới hạn và phép thử tiếp theo

- Cùng 20 câu FLEURS gốc 16 kHz, tên riêng nước ngoài và số chưa chuẩn hóa thành chữ vẫn tạo nhiều lỗi.
- Kết quả tốt hơn trên speech đi cùng hành vi noise-only kém hơn ở cấu hình hiện tại; cần đánh giá hai mục tiêu riêng.
- Phép thử hữu ích tiếp theo là Gipformer greedy trên đúng cùng audio, lưu trong OUTPUT_ROOT khác. Điều này giúp khảo sát ảnh hưởng decoder; không thay đổi dataset/DF3/chuẩn hóa cùng lúc.
- Muốn xác định DF3 làm thay đổi âm lời nói ra sao, nghe các cặp xấu/tốt ở trên; không thể kết luận méo tiếng chỉ từ WER.
- Báo cáo này không sửa code, cấu hình decoder hoặc kết quả đầu vào.

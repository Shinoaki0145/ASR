# Phân tích Zipformer 30M + DeepFilterNet3

Đã đối chiếu 922 dòng per-clip thành 461 cặp raw/DF3. Tất cả WER/CER trong quality_summary.csv khớp phép tính lại từ số lỗi từng clip.

**Kết luận trong tập thử này:** DF3 giảm mức năng lượng ở các mẫu noise-only, nhưng chưa cải thiện nhận dạng tổng thể trên các bản trộn. Kết quả phụ thuộc mạnh vào một số câu; không khái quát thành DF3 luôn làm giảm chất lượng ASR.

| Tập | Số cặp | WER raw | WER DF3 | CER raw | CER DF3 |
|---|---:|---:|---:|---:|---:|
| Câu gốc | 20 | 8.52% | 8.15% | 5.05% | 4.94% |
| Có nhiễu | 420 | 8.78% | 9.16% | 5.30% | 5.87% |

Trên 420 bản trộn: raw 996/11.340 lỗi token, DF3 1.039/11.340; tăng 43 lỗi, tương đương +0,379 điểm phần trăm WER. 38 bản giảm số lỗi, 52 bản tăng, 330 bản giữ nguyên số lỗi. Giữ nguyên số lỗi không đồng nghĩa transcript giống nhau.

## Theo loại nhiễu — gộp ba SNR

| Nhiễu | Kiểu | Số cặp | WER raw | WER DF3 | ΔWER (điểm %) |
|---|---|---:|---:|---:|---:|
| keyboard | continuous | 60 | 9.01% | 9.57% | +0.56 |
| keyboard | events | 60 | 9.01% | 9.51% | +0.49 |
| knock | events | 60 | 8.52% | 8.70% | +0.19 |
| doorslam | events | 60 | 8.70% | 8.77% | +0.06 |
| keys | events | 60 | 8.52% | 8.52% | +0.00 |
| pageturn | events | 60 | 9.01% | 9.57% | +0.56 |
| phone | events | 60 | 8.70% | 9.51% | +0.80 |

## Chi tiết theo SNR

| Nhiễu | Kiểu | SNR dB | WER raw | WER DF3 | ΔWER (điểm %) |
|---|---|---:|---:|---:|---:|
| keyboard | continuous | 0 | 9.26% | 10.19% | +0.93 |
| keyboard | continuous | 5 | 9.26% | 9.63% | +0.37 |
| keyboard | continuous | 10 | 8.52% | 8.89% | +0.37 |
| keyboard | events | 0 | 8.89% | 9.81% | +0.93 |
| keyboard | events | 5 | 8.89% | 9.81% | +0.93 |
| keyboard | events | 10 | 9.26% | 8.89% | -0.37 |
| knock | events | 0 | 8.52% | 8.70% | +0.19 |
| knock | events | 5 | 8.52% | 8.70% | +0.19 |
| knock | events | 10 | 8.52% | 8.70% | +0.19 |
| doorslam | events | 0 | 8.70% | 8.89% | +0.19 |
| doorslam | events | 5 | 8.70% | 8.70% | +0.00 |
| doorslam | events | 10 | 8.70% | 8.70% | +0.00 |
| keys | events | 0 | 8.52% | 8.70% | +0.19 |
| keys | events | 5 | 8.52% | 8.52% | +0.00 |
| keys | events | 10 | 8.52% | 8.33% | -0.19 |
| pageturn | events | 0 | 9.26% | 10.19% | +0.93 |
| pageturn | events | 5 | 8.89% | 9.44% | +0.56 |
| pageturn | events | 10 | 8.89% | 9.07% | +0.19 |
| phone | events | 0 | 8.70% | 10.19% | +1.48 |
| phone | events | 5 | 8.70% | 9.07% | +0.37 |
| phone | events | 10 | 8.70% | 9.26% | +0.56 |

## Các trường hợp nên nghe đối chiếu

### vi_018_keyboard_events_snr00

Reference: Cơn bão được đặt tên thứ mười của mùa Bão Atlantic, Bão cận nhiệt đới Jerry, đã hình thành tại Biển Atlantic hôm nay.

Raw (6/24 lỗi): CƠN BÃO ĐƯỢC ĐẶT TÊN THỐI MỜI CỦA MÙA BÃO LÊN BẢO CẢNH NHIỆT ĐỚI CHERRY ĐÃ HÌNH THÀNH TẠI BIỂN ATLANTIC HÔM NAY

DF3 (11/24 lỗi): CƠN BÃO ĐƯỢC ĐẶT TÊN THỐI MỜI CỦA MÙA BÃO ÁNH DƯƠNG TÍCH BẢO VỆ NHIỆT ĐỚI CHERY ĐÃ HÌNH THÀNH TẠI BIỂN NAM RANG

### vi_018_phone_events_snr00

Reference: Cơn bão được đặt tên thứ mười của mùa Bão Atlantic, Bão cận nhiệt đới Jerry, đã hình thành tại Biển Atlantic hôm nay.

Raw (5/24 lỗi): CƠN BÃO ĐƯỢC ĐẶT TÊN THỐI MỜI CỦA MÙA BÃO ATLANTIC BẢO CẢNH NHIỆT ĐỚI CHERRY ĐÃ HÌNH THÀNH TẠI BIỂN ATLANTIC HÔM NAY

DF3 (9/24 lỗi): CƠN BÃO ĐƯỢC ĐẶT TÊN THỐI MỜI CỦA MÙA BÃO ÁCHAEL BẢO CẢNH NHIỆT ĐỚI CHERY ĐÃ THÀNH TÌNH TẠI BIỂN OLYMPI HÔM NAY

### vi_005_keyboard_events_snr10

Reference: Mọi danh từ, kế bên từ Sie dành cho bạn, luôn bắt đầu bằng một chữ viết hoa, ngay cả khi ở giữa câu.

Raw (4/24 lỗi): MỌI DANH TỪ KẾ BÊN TỪ SI NHÌN CHO BẠN LUÔN BẮT ĐẦU BẰNG MỘT CHỮ VIẾT HOA NGAY CÁO SI Ở GIỮA CÂU

DF3 (1/24 lỗi): MỌI DANH TỪ KẾ BÊN TỪ XƯA DÀNH CHO BẠN LUÔN BẮT ĐẦU BẰNG MỘT CHỮ VIẾT HOA NGAY CẢ KHI Ở GIỮA CÂU

### vi_016_keyboard_events_snr05

Reference: Cuối cùng, có nhiều loài mèo nhỏ (gồm cả mèo nuôi trong nhà) ăn những con mồi nhỏ có số lượng lớn như côn trùng, các loài gặm nhấm, thằn lằn và chim.

Raw (4/33 lỗi): CUỐI CÙNG CÓ NHIỀU LOẠI MÈO NHỎ GỒM CẢ MÈO NUÔI TRONG NHÀ ĂN NHỮNG CON MỒI NHỎ CÓ SỐ LƯỢNG LỚN NHƯ CÔN TRÙNG CÁC LOẠI GẪM NHẮM THẰN LẰN VÀ CHIM

DF3 (2/33 lỗi): CUỐI CÙNG CÓ NHIỀU LOẠI MÈO NHỎ GỒM CẢ MÈO NUÔI TRONG NHÀ ĂN NHỮNG CON MỒI NHỎ CÓ SỐ LƯỢNG LỚN NHƯ CÔN TRÙNG CÁC LOẠI GẶM NHẤM THẰN LẰN VÀ CHIM

## Độ nhạy theo câu gốc

Riêng vi_018 (câu có Atlantic/Jerry) đóng góp +38 trong tổng +43 lỗi token sau DF3 trên 21 bản trộn của câu này. Nếu tạm bỏ câu đó để kiểm tra độ nhạy, WER raw là 8.21%, DF3 là 8.26%. Đây là phân tích độ nhạy, không phải lý do loại câu khỏi kết quả chính.

420 bản trộn chỉ bắt nguồn từ 20 câu nói; các bản trộn không phải 420 câu độc lập. Không đủ cơ sở khẳng định khác biệt có ý nghĩa thống kê hoặc đại diện triển khai thực tế.

382/420 bản trộn raw có cùng số lỗi token với câu gốc raw; 33 bản tăng lỗi và 5 bản giảm. Nhiễu trong cấu hình này nhìn chung chưa làm Zipformer suy giảm nhiều.

## Noise-only và sinh chữ

| Nhóm | Raw sinh chữ | DF3 sinh chữ |
|---|---:|---:|
| keyboard/continuous | 0/3 | 0/3 |
| keyboard/events | 0/3 | 1/3 |
| knock/events | 0/3 | 0/3 |
| doorslam/events | 0/3 | 0/3 |
| keys/events | 0/3 | 0/3 |
| pageturn/events | 1/3 | 0/3 |
| phone/events | 2/3 | 3/3 |

Tổng raw 3/21 mẫu sinh chữ; DF3 4/21 mẫu. Các chữ phát sinh đều là “THÌ”. Có mẫu phone sau DF3 giảm RMS rất mạnh nhưng ASR vẫn sinh chữ; giảm năng lượng nhiễu chưa đồng nghĩa ngăn ASR sinh chữ ngoài ý muốn.

Nhiều transcript thô là ký hiệu `<`; bộ chuẩn hóa hiện coi đây là không có chữ. Tỷ lệ nonempty_transcript_rate được tính sau chuẩn hóa, không phải kiểm tra chuỗi thô có rỗng hay không.

RMS attenuation ở 21 mẫu noise-only: min 10.45 dB, median 26.90 dB, max 59.22 dB. Đây là RMS toàn waveform 48 kHz trước/sau enhancement, không đo mức bảo toàn lời nói.

## Latency và giới hạn phép thử

Trên 420 bản trộn, median enhancement khoảng 73.78 ms; resample 14.95 ms. Median của tổng enhancement + resample + ASR theo từng clip khoảng 186.79 ms. Tổng này chỉ là ước lượng từ worker warm riêng, không phải latency end-to-end đo trực tiếp.

Speech FLEURS gốc 16 kHz, được resample lên 48 kHz cho DF3; không đánh giá đầy đủ lợi ích trên speech fullband thu ở 48 kHz. Mỗi nhánh chỉ một lần ASR/clip trong quality run; dùng benchmark lặp riêng để đánh giá tốc độ ổn định.

Cách chấm chưa đọc số thành chữ: ví dụ reference “1967” và output “một nghìn chín trăm sáu bảy” vẫn bị tính khác. Tên nước/tên riêng tiếng nước ngoài cũng chiếm nhiều lỗi. Cần giữ cùng chuẩn hóa cho mọi nhánh và mọi model khi so sánh.

Không sửa cấu hình/model hoặc loại mẫu trong lần phân tích này. Muốn đánh giá tiếp, nên tăng số câu độc lập, nghe các cặp có delta lớn và kiểm tra riêng noise-only; chưa có cơ sở từ tập này để bật DF3 mặc định chỉ nhằm giảm WER.

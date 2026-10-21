# Runtime notebook hội thoại

Đầu ra cho người dùng: `../Hoi_Thoai_Ngat_Loi_Test.ipynb`. Notebook nhúng bản sao runtime, test, các helper cần thiết của Sub-ASR, lockfile và audio ghép 3 giây; chạy được khi không có repo bên cạnh.

## Cập nhật notebook

Sửa `voice.py`/`test_voice.py`, chạy kiểm thử rồi đóng gói lại:

```bash
uv run --with numpy --with httpx python -m unittest discover -s notebook_runtime -p test_voice.py -v
python3 notebook_runtime/build_notebook.py
```

Lệnh trên chạy từ thư mục gốc ASR. Phần dependency/model thật trong notebook dùng đúng `Sub-ASR/pyproject.toml`, `uv.lock` và setup đã ghim revision. Không tự thay bản k2/kaldifeat để phù hợp kernel; dùng Python 3.12 riêng qua uv.

## Các quyết định điều chỉnh kế hoạch

- Theo yêu cầu mới của người dùng, triển khai thành notebook độc lập và runtime nhúng; không sửa CLI/Marimo gốc. Thiết kế state và asyncio vẫn theo kế hoạch. Chi phí: khi muốn dùng trong CLI chính thức cần ghép các giao diện và chạy lại test CLI, không coi việc tạo notebook là đã triển khai CLI.
- Worker là process dành riêng cho môi trường Python 3.12 và CUDA; bên trong vẫn một event loop điều phối, một ASR native đang chạy, không message broker. Chi phí: khởi động process/model mỗi phiên; độ trễ khởi động không phải độ trễ ngắt lời.
- Trên Colab mặc định hiển thị phản hồi và mô phỏng thời gian phát; không giả vờ eSpeak trên server phát tiếng tới browser. Phát tiếng thật và microphone là tùy chọn local. Chi phí: chế độ Colab không kiểm định được ngắt loa/microphone thật.
- Event JSON stdout là dữ liệu UI để người dùng xem transcript, yêu cầu và câu bot; không phải telemetry ghi đĩa. Notebook cảnh báo output có nội dung hội thoại và nên xóa trước khi chia sẻ. Chi phí: lưu notebook kèm output có thể lưu nội dung đó; runtime không tự ghi log hoặc API key ra tệp.
- Lỗi ASR/CUDA không dự kiến là fatal, không âm thầm tiếp tục bằng transcript cũ; cleanup phải luôn dừng session rồi mới đợi native ASR và cuối cùng đóng HTTP client. Chi phí: người dùng phải sửa lỗi model/môi trường rồi chạy phiên mới.
- Phản hồi mô phỏng không được xem là đã phát cho người dùng hoặc đưa vào lịch sử như câu đã nghe. Chi phí: câu trả lời ngắn dựa vào câu hỏi bot chưa xác nhận đã phát có thể bị hỏi rõ hơn.

## Rà soát và bằng chứng

Đã có lượt review độc lập về generation, endpoint, single-flight ASR, schema và shutdown. Các lỗi đã sửa có test tái hiện: TTS cũ/mới chồng nhau; transcript hiện tại lặp trong resolver context; cleanup bỏ qua HTTP client khi có lỗi; action sai kiểu.

Bộ test offline ban đầu có 11 test: state/schema, JSON/timeout/context hợp lệ và không hợp lệ, stale resolver, không chặn capture, không chồng playback, ASR single-flight và kết quả cũ, hai vùng nói cách nhau khoảng nghỉ, kill TTS, shutdown lỗi.

Notebook được thực thi trong thư mục tạm không có repo, với bản runtime/audio được nhúng. Demo và test offline được kiểm tra bằng kernel thật; GPU, checkpoint ASR, API LLM thật, microphone, âm thanh eSpeak và p95 ngắt tiếng chưa được kiểm định tại đây. Mục tiêu đo ở máy đích vẫn theo tài liệu kế hoạch; không tuyên bố đạt realtime chỉ từ demo.

Bộ đánh giá 5 cách nói cho mỗi action, 30 lần ngắt microphone và AEC/loa ngoài vẫn cần thực hiện trên máy đích.

## Cấu hình .env

File `.env` ở thư mục gốc ASR được notebook đọc bằng `dotenv_values` tại `ENV_PATH`. Trên Colab upload file vào `/content/.env`. Giá trị trong file ưu tiên hơn biến môi trường; chạy lại cell cấu hình sau mỗi lần sửa. Bản mẫu `.env.example` không có secret. `.gitignore` loại `.env` khỏi Git; bundle notebook không chứa file này. Worker nhận cấu hình qua environment, không qua tham số dòng lệnh.

## Cập nhật chốt đoạn theo im lặng (21/10/2026)

Theo yêu cầu đã được người dùng đồng ý: bỏ semantic endpoint khỏi đường chốt audio. `SEGMENT_SILENCE_SECONDS=1.0` (trong .env hoặc mặc định) điều khiển khoảng lặng. Giới hạn 30s và pre-roll giữ nguyên. Chốt audio không đồng nghĩa chốt yêu cầu; resolver vẫn dùng lịch sử để bổ sung/sửa/thay/hủy.

Audio được đóng băng ngay khi đủ khoảng lặng, kể cả ASR đang bận; hàng đợi tối đa 8 đoạn, một native ASR chạy tại một thời điểm. Transcript trả muộn của generation cũ được giữ theo thứ tự làm ngữ cảnh nhưng không bật phản hồi cũ. ASR/LLM chậm vẫn có thể khiến bot chưa kịp nói trước đoạn tiếp; hệ thống không được giả vờ đã ngắt tiếng.

Kiểm chứng mới: 14 test offline pass, gồm ba test tái hiện rồi sửa cho ranh giới 1 giây không gọi endpoint LLM, ASR chậm không nhập hai đoạn và nghỉ dưới 1 giây vẫn cùng đoạn. Notebook độc lập đã chạy lại trong thư mục sạch.

Thử file MP3 mẫu với **Silero VAD CPU thật**, ASR/LLM/phát tiếng là fixture: có hai speech_started, hai segment_closed tại audio clock 2,976s và 8,640s (silence=1,024s do frame 32ms), một cancel_requested. Chưa xác nhận Gipformer CUDA + LLM thật với code mới trên Colab.

Đã cập nhật cả notebook gốc và Testing. Output của lần Testing cũ giữ ở `testing_before_silence_fix.ipynb`; bản Testing cập nhật xóa output để không nhầm số liệu cũ với code mới. Người dùng cần nạp notebook mới và chạy lại từ cell bootstrap để runtime được bung lại.

## Cập nhật nghĩa của clarify và chẩn đoán resolver

Log người dùng: đoạn đầu đã rõ ý định đặt vé nhưng LLM dùng clarify vì thiếu ngày; request vẫn null. Đoạn ngày đi sau đó rơi vào fallback. Không suy đoán nguyên nhân fallback cũ khi chưa có diagnostic.

Prompt mới bắt buộc lưu yêu cầu đã hiểu, kể cả thiếu ngày/ga đi; reply thu thập thông tin thiếu. clarify dành cho ý nghĩa/quan hệ chưa rõ. Nếu request chưa có nhưng pending_text/history chứa yêu cầu cũ, resolver phải xét nó và dùng new để khôi phục yêu cầu, không phát append sai trạng thái.

Lỗi kỹ thuật được truyền bằng ResolveError và hiển thị resolve_failed (reason, HTTP status); không in secret/raw HTTP body. Câu hỏi dự phòng vẫn giữ state/pending_text, nhưng không bị nhầm với clarify do model trả hợp lệ.

16 test offline pass. Thử đúng hai transcript của người dùng với Gemini API thật: new → append, request cuối đặt vé tàu đi Đà Nẵng vào ngày 30 tháng 10. Hai lần gọi mất 2,564s và 2,267s khi timeout chẩn đoán 10s; một lần thử timeout cũ 2s thất bại. Vì vậy timeout mặc định mới 5s và đọc RESOLVE_TIMEOUT_SECONDS từ .env. Tăng timeout không chặn capture và không trì hoãn cancellation khi có giọng nói mới. Không kết luận lỗi fallback cũ nhất định là timeout.

Với khoảng nghỉ chỉ 3s, nếu ASR + LLM chưa xong trước đoạn sau, không bảo đảm bot đã nói để bị ngắt: transcript cũ vẫn được giữ làm ngữ cảnh. GPU/ASR và tiếng phát vật lý chưa được chạy lại cùng phiên API này.

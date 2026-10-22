# Kết quả kiểm định Web-ASR

Ngày: 22/10/2026. Môi trường local Linux x86_64, Python 3.12.3, Gradio 6.30.0, soxr 1.1.0, Chromium do Playwright cài. Các kiểm tra dưới đây dùng fixture, không gọi API LLM và không nạp checkpoint GPU.

## Đã kiểm tra

- 16 unit test Web-ASR pass: chuẩn hóa/resample stereo 48 kHz → mono 16 kHz, flush đúng một lần; config/secret repr; audio + hai lần Enter gom một nhóm; ASR lỗi vẫn giữ text; history khi tắt loa; stale ack; toggle loa không đổi generation/không replay; giới hạn phiên/mic; packet đến đảo thứ tự; audio overflow vẫn giữ control/text; native ASR được serialize; race hủy TTS đúng lúc process thoát; notebook sạch output và compile được.
- Regression `Sub-ASR/test_voice.py`: 16 test của runtime cũ chạy pass sau các hook tương thích. Notebook Testing người dùng đã sửa không bị ghi đè.
- Browser end-to-end: `uv run --extra dev python tests/browser/test_browser.py`. Chromium thu file WAV qua fake microphone bằng MediaDevices + AudioWorklet thật, phát WAV qua HTMLAudio thật. Assert capture/playback có khoảng chồng nhau; barge-in dừng player và mic tiếp tục; text ghép nhóm audio; tắt/bật loa không replay; browser ack stopped/ended đúng; tab thứ hai bị từ chối mic và tắt track; chặn autoplay bằng Promise rejection rồi bấm phát thủ công; play cũ khác audio_epoch bị bỏ.
- Bài browser cuối không dùng cờ bỏ qua autoplay. Nhánh chặn autoplay được ép bằng fixture để kiểm tra fallback; chưa xác minh mọi chính sách autoplay của Safari/mobile.
- Notebook đã thực thi các cell thiết lập/uv/khởi chạy/xem log ở fake mode local, HTTP `/config` trả 200; chạy lại cell launch dừng process trước. Cell dừng đã dọn server thử.
- Screenshot giao diện đã xem kiểm tra: hai cột, ô nhập/output, nút mic/loa và sidebar log. Screenshot nằm trong `.cache/`, không đưa vào Git.
- Ruff kiểm tra lỗi F và format source Python mới; uv.lock khóa cả dependency CPU/dev và CUDA extra. `.venv`, `.env`, cache, log được Git bỏ qua.

## Điều chỉnh so với kế hoạch ban đầu

1. Không dùng `gr.Audio(recording=...)` làm công tắc mic. Probe browser cho thấy cập nhật thuộc tính không khởi tạo getUserMedia từ trạng thái ban đầu. Dùng AudioWorklet do ứng dụng sở hữu; audio PCM được gửi qua event Gradio, không có frontend/server/WebSocket riêng. Vì vậy không còn phụ thuộc contract chunk/cumulative/stop_recording của Gradio Audio.
2. Timer 0,2s gom PCM thành packet có sequence và recording_id. Server giữ thứ tự packet, nhận mỗi sample một lần. Resample có state; mic-off gửi đuôi trước khi chốt. Tick chờ ASR khi mic bật không tự thêm khoảng im lặng vào audio clock.
3. WAV trả bằng data URL qua Gradio; không mở endpoint file cho audio và không allowlist repo. Payload audio được bỏ khỏi command sau ack started/blocked để tránh gửi lại file mỗi tick. Client giữ playback hiện tại phục vụ thao tác phát thủ công.
4. Control/text dùng FIFO không bị quota audio chiếm hết; quota 96 frame chỉ áp dụng audio. Audio overflow có lỗi rõ và buộc browser dừng mic. Giới hạn hai phiên/một mic, polling, TTL đang là hằng số MVP thay vì thêm nhiều biến cấu hình.
5. Text bot được lưu với nhãn “đã hiển thị text; chưa xác nhận đã đọc/nghe”. Audio `completed` chỉ sau ended; blocked có nút phát thủ công và hết deadline vẫn là lỗi. Hiện dùng deadline tổng thời lượng WAV +10s; chưa tách deadline started/stop 2s hoặc thu thập p95.
6. Không commit/push hay mở share link từ môi trường local. Các lệnh chạy nhầm test từ repo root lúc đầu bị lỗi vì test cũ tham chiếu đường dẫn trước khi repo được tổ chức lại; đã chạy đúng bộ Web-ASR và regression Sub-ASR từ thư mục tương ứng, không sửa test root ngoài phạm vi.

## Chưa kiểm định tại đây

- Colab T4/CUDA, tải checkpoint/model và eSpeak tiếng Việt thật trong process web.
- Share URL/TLS/login, microphone/loa vật lý, chất lượng ASR/LLM thật, tiếng vọng và trải nghiệm trên điện thoại/Safari.
- Độ trễ ngắt tiếng vật lý, p95, chạy nhiều giờ, mất mạng rồi khôi phục. Packet mất/gián đoạn dài yêu cầu dừng mic/tải lại; chưa có cơ chế nối lại audio đáng tin cậy.
- Browser đóng tab dùng unload để dọn sớm; heartbeat 120s là dự phòng khi unload không tới server. Chưa kiểm thử thu hồi runtime Colab trong lúc ASR native đang chạy.

## Checklist thử trên Colab

1. Chọn T4, `FAKE=False`, thiết lập secret/config, chạy notebook đến cell URL.
2. Mở tab riêng, đăng nhập, cho phép mic, đeo tai nghe.
3. Nói “đặt vé tàu đi Đà Nẵng”, chờ bot nói rồi nói “ngày 30 tháng 10”. Xem stop command và browser ack, request cuối giữ điểm đến/ngày.
4. Trong lúc nói gửi text “2 vé”; kiểm tra một transcript cuối của nhóm có audio và text, bản nháp không bị ASR đè.
5. Tắt/bật loa lúc sinh text, TTS và playback; mic/text tiếp tục, bật lại không phát câu cũ.
6. Thử từ chối quyền mic, tab thứ hai và phát thủ công khi autoplay blocked. Đối chiếu sidebar, không suy ra đã nghe từ text hiển thị.

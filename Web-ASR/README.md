# Web-ASR

Ứng dụng Gradio dùng một notebook Colab để chạy backend và giao diện. Người dùng mở link web, nhập text hoặc bật mic, xem phản hồi LLM và nghe bot. Hai nút mic/loa độc lập; sidebar bên phải hiện request, generation, nhóm input và 200 sự kiện debug.

## Chạy Colab

1. Đưa **repo đầy đủ** lên Colab/Drive hoặc đặt `REPO_URL` trong [Run_Web_ASR_Colab.ipynb](Run_Web_ASR_Colab.ipynb). Cần cả `Web-ASR` và `Sub-ASR`.
2. Chọn runtime T4 GPU. Notebook mặc định `FAKE=False`, `SHARE=True`.
3. Điền `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` trong `.env` gốc repo, file upload `/content/.env` hoặc Colab Secrets. Notebook tự đọc file trước khi hỏi thông tin còn thiếu. Có thể xem [.env.example](.env.example); `ASR_ENV_PATH` đổi đường dẫn config. Google AI Studio dùng `https://generativelanguage.googleapis.com/v1beta/openai`; tên model lấy từ tài khoản của bạn.
4. Chạy các cell lần lượt **đến cell xem URL**. Nhập mật khẩu web khi được hỏi. Cell cuối để dừng server, không chạy ngay nếu còn muốn sử dụng.
5. Mở URL Gradio trong tab riêng, đăng nhập và cho phép microphone. Dùng tai nghe. Nút loa chỉ bật/tắt phát tiếng bot.

Notebook dùng `uv sync --locked --extra cuda`, Python 3.12 trong `.venv`, model và CUDA pins của Sub-ASR. Model ASR nạp một lần ở process server. `espeak-ng` tạo tiếng Việt WAV, gửi qua Gradio đến browser; không dùng loa Colab. Không cần notebook frontend thứ hai.

## Chạy local / kiểm tra không GPU

Từ thư mục `Web-ASR`:

```bash
uv sync --locked
WEB_FAKE=true uv run --locked python app.py
```

Mở `http://127.0.0.1:7860`. Fake dùng transcript cố định, nối text theo quy tắc đơn giản và phát tone; **không đánh giá chất lượng ASR/LLM**. Chạy thật cần Linux x86_64, CUDA tương thích, `espeak-ng`, tải model bằng setup như notebook rồi bỏ `WEB_FAKE=true`.

```bash
uv run --locked python -m unittest discover -s tests -v
uv sync --locked --extra dev
uv run --extra dev playwright install chromium
uv run --extra dev python tests/browser/test_browser.py
```

## Hành vi hội thoại

- Chốt đoạn sau 1 giây im lặng (`SEGMENT_SILENCE_SECONDS`); resolver timeout mặc định 5 giây (`RESOLVE_TIMEOUT_SECONDS`). Không dùng LLM để dò hết câu.
- Khi bắt đầu nói, hủy phản hồi cũ và gửi lệnh dừng browser player. ASR vẫn chạy tuần tự, kết quả đúng đoạn được giữ.
- Text gửi khi đang có nhóm audio chưa hoàn tất được thêm vào nhóm đó theo thứ tự. Chờ audio/ASR rồi gọi resolver một lần; nguồn text được đánh dấu để LLM phân biệt bổ sung/sửa/thay/hủy. Text gõ chưa gửi luôn được giữ nguyên.
- Tắt loa dừng âm thanh/TTS, vẫn sinh và hiển thị text, giữ mic/ngữ cảnh. Bật lại chỉ áp dụng cho phản hồi tiếp theo.
- Log `audio_command stop` là yêu cầu dừng; `browser_ack stopped` là browser xác nhận. Chỉ `ended` mới tính phát xong. Text đã hiển thị không đồng nghĩa người dùng đã nghe.

## Cấu hình và giới hạn MVP

`WEB_SHARE=true` hoặc `WEB_HOST=0.0.0.0` yêu cầu `WEB_USERNAME`, `WEB_PASSWORD`. Key LLM chỉ dùng ở backend. `.env`, `.venv`, log/cache được gitignore; `uv.lock` được lưu để tái tạo môi trường. Mặc định bind localhost, port 7860; `WEB_PORT` đổi port.

Hai session UI, một mic hoạt động cùng lúc; session mất heartbeat được dọn sau 120s. Buffer audio tối đa khoảng 3s, log 200 dòng. AudioWorklet của browser thu mono và gửi PCM qua Gradio, resample có state về 16 kHz. Cần browser có microphone/AudioWorklet, HTTPS hoặc localhost. Không dùng native `gr.Audio.recording`: thử nghiệm cho thấy đổi thuộc tính này không khởi tạo microphone reliably ở phiên bản pinned.

Vòng đời share link phụ thuộc Colab/Gradio, không phù hợp chạy 24/7. Chưa có AEC riêng cho loa ngoài, TTS streaming, lưu hội thoại lâu dài hoặc đặt vé thật. Việc chạy GPU/loa vật lý trên Colab cần kiểm tra tại thiết bị người dùng; xem [kết quả kiểm định](docs/validation.md).

- [Thiết kế ban đầu](docs/design.md)
- [Kế hoạch và phạm vi](docs/plan.md)

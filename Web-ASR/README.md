# Web-ASR

## Điền form bằng giọng nói (mới)

Đẩy source lên GitHub với hai thư mục `Web-ASR` và `Sub-ASR` ở gốc repo.
Mở `Run_Form_ASR_Colab.ipynb`, đặt `REPO_URL` (mặc định repo ASR hiện tại)
và `REPO_BRANCH` nếu cần chọn nhánh, rồi chọn T4 GPU và chạy đến cell URL.
Notebook tự clone khi chưa có source, hoặc tái sử dụng checkout sẵn có như
Web-ASR. Nếu checkout cũ thiếu `form_app.py`, cập nhật code rồi chạy lại;
notebook không tự ghi đè checkout. Không cần upload ZIP, LLM key hoặc cài espeak-ng.
Mở link web và cấp quyền mic; không có bước đăng nhập.

Luồng: browser PCM → Silero VAD → Gipformer → trường đã chọn. Form mẫu gồm
họ tên, điện thoại và địa chỉ. Nghỉ mặc định 0,7 giây để chốt một đoạn;
`SEGMENT_SILENCE_SECONDS` đổi ngưỡng. Các đoạn nối vào trường; điện thoại đọc
từng chữ số (giữ số 0 đầu), không hỗ trợ cách đọc nhóm như “chín mươi”.
Chọn trường mới không chuyển đích của đoạn đã bắt đầu trước đó.

Ba trường hiển thị là kết quả nhận dạng. Muốn sửa hoặc xóa: chọn trường, nhập
toàn bộ giá trị vào ô sửa rồi bấm **Lưu sửa tay**. Kết quả ASR cũ không ghi đè
bản sửa đã lưu. Bản gõ chưa bấm lưu chưa thuộc form; đổi trường sẽ nạp lại giá
trị của trường mới vào ô sửa. Tắt mic và chờ ASR xong trước khi **Tải JSON**.
Yêu cầu họ tên, địa chỉ và điện thoại định dạng `0` + 9 số hoặc `+84` + 9 số;
đây là kiểm tra định dạng, không xác thực chủ thuê bao. Không tự gửi form.

Chạy demo local từ `Web-ASR` (PowerShell):

```powershell
$env:WEB_FAKE='true'
.\.venv\Scripts\python -X utf8 form_app.py
```

Hoặc trên Linux: `WEB_FAKE=true uv run --locked python form_app.py`.
Demo dùng transcript cố định, không đánh giá chất lượng model thật.
Chạy model thật vẫn cần Linux/CUDA và model setup như notebook. Link public
không yêu cầu `WEB_USERNAME`/`WEB_PASSWORD`, kể cả khi `.env` còn chứa hai biến này.
Ai có link public đều có thể mở và dùng ứng dụng.
JSON được tải bằng browser, không tạo file form trên server. Dữ liệu và
transcript trong RAM được xóa khi phiên đóng/hết hạn; cần tải trước khi đóng
tab hoặc dừng Colab. Hai phiên độc lập, một mic hoạt động cùng lúc.

Kiểm tra: `python -X utf8 -m unittest discover -s tests -v`.
Kiểm tra trình duyệt: `python -X utf8 tests/browser/test_form_browser.py`.
Chưa benchmark GPU thật hoặc độ chính xác tên riêng/số điện thoại.

Phần dưới là tài liệu luồng hội thoại cũ (`app.py`).

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

# Kế hoạch triển khai Web-ASR bằng Gradio trên một notebook Colab

> **Dành cho tác nhân triển khai:** dùng `superpowers:executing-plans` để thực hiện tuần tự; chỉ dùng `superpowers:subagent-driven-development` nếu người dùng chọn cách đó. Các bước dùng ô đánh dấu để theo dõi. Code MVP đã được triển khai. Kết quả thực thi và các điều chỉnh so với kế hoạch được ghi tại [validation.md](validation.md); checkbox dưới đây giữ nguyên làm checklist nghiệm thu chi tiết, không biểu thị tất cả hạng mục đã được kiểm định trên Colab.

**Mục tiêu:** Người dùng mở link Gradio, nhập text hoặc bật mic, nhận text và tiếng bot; ngắt lời được, đồng thời quan sát log/trạng thái ở cột bên phải.

**Kiến trúc:** Một notebook khởi chạy một process Python 3.12 trên Colab. Gradio nhận chunk từ browser và phục vụ UI; vòng lặp asyncio nền dùng runtime `Sub-ASR/voice.py` để chốt audio và quản lý ngữ cảnh. Browser audio player có command/ack gắn generation để dừng tiếng cũ an toàn.

**Công nghệ:** Gradio 6.30.0, Python 3.12, asyncio, NumPy, soxr 1.1.0, Silero/Gipformer và CUDA pins hiện có, chat-completions LLM, eSpeak tổng hợp WAV; unittest và kiểm thử browser.

**Đặc tả:** [Thiết kế, giao diện và tiêu chí nghiệm thu](design.md).

## Ràng buộc chung

- Code trong `Web-ASR`; notebook `Run_Web_ASR_Colab.ipynb` chỉ cài đặt/khởi chạy/theo dõi/dừng process, không chứa business logic hoặc bundle Base64.
- Cần repo đầy đủ và `Sub-ASR` hiện có; không sao chép runtime thành một fork.
- Python `>=3.12,<3.13`; cùng pin torch/torchaudio/k2/kaldifeat của Sub-ASR; khóa dependency bằng uv.lock mới.
- Audio mono float32, 16 kHz, 512 sample/frame; chốt sau 1.0s im lặng, resolver timeout 5.0s; không gọi LLM endpoint để chốt đoạn.
- Đọc `SEGMENT_SILENCE_SECONDS=1.0` và `RESOLVE_TIMEOUT_SECONDS=5.0` từ .env; giữ các biến LLM của notebook hiện có.
- Cấu hình `WEB_MIC_CHUNK_SECONDS=0.25`, `WEB_POLL_SECONDS=0.2`; một native ASR, tối đa 2 session UI và 1 mic hoạt động.
- Queue audio tối đa 3s, queue đoạn đã chốt tối đa 8, debug tối đa 200 dòng, heartbeat expiry 120s.
- Không phát tiếng ra loa Colab; browser ack mới xác nhận đã phát xong. Không coi clear output, task.cancel hoặc hết timer là tiếng vật lý đã dừng.
- Mic tiếp tục khi bot nói; tắt mic chốt phần đang nói. Text gửi trong lúc nói bổ sung vào cùng input_group_id; giữ buffer audio và typed_additions, chờ ASR/hết lượt trước khi phản hồi.
- Nút loa độc lập: mặc định bật, tắt chỉ dừng/hủy audio và TTS, không hủy text/LLM/mic. Bật lại không phát audio cũ; audio_epoch chống play trả muộn.
- Key chỉ ở server. Không public repo hoặc tệp cấu hình qua đường dẫn audio. Bản đầu kiểm định với tai nghe, chưa có AEC riêng.

## Trọng tâm review

1. Lệnh play cũ/ack cũ đến sau lần ngắt: không được phát hoặc commit history sai generation.
2. Callback/ASR bận hoặc Enter trùng segment_closed: mic/stop/poll không chặn; không mất audio/text, không submit group hai lần hoặc dùng generation cũ.
3. Tắt mic khi có đuôi audio, sau đó bật lại: không nhân đôi transcript, không nối audio cũ vào lượt mới.
4. Tắt/bật loa giữa TTS, autoplay bị chặn, tab mất mạng hoặc đóng: không báo completed giả, dọn audio và tài nguyên đúng phiên.
5. Hai tab dùng chung model: history/VAD state/ack được cách ly; không gọi ASR native đồng thời hoặc lộ secret trong debug.

## Tệp dự kiến

```text
Web-ASR/
├── Run_Web_ASR_Colab.ipynb
├── app.py
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── README.md
├── web_asr/
│   ├── __init__.py
│   ├── config.py
│   ├── backend.py
│   ├── audio.py
│   ├── ui.py
│   └── browser_audio.js
├── tests/
│   ├── test_config.py
│   ├── test_audio.py
│   ├── test_backend.py
│   ├── test_ui.py
│   ├── test_notebook.py
│   └── browser/                 # fixture browser và bài kiểm tra cần browser thật
└── docs/
    ├── design.md
    ├── plan.md
    └── validation.md
```

Chỉ sửa `Sub-ASR/voice.py` ở các hook dùng chung cần thiết và thêm regression test tại `Sub-ASR/test_voice.py`. Giữ notebook Testing/CLI/Marimo hiện tại chạy được. Không di chuyển model/checkpoint vào Git.

Mọi lệnh test Web-ASR bên dưới chạy từ `Web-ASR/`.

## Nhiệm vụ 1 — Chứng minh đường mic/phát/dừng của Gradio

**Tệp:** tạo fixture kiểm tra browser ở `tests/browser/`; phần đã chứng minh chuyển thành `web_asr/ui.py`, `web_asr/browser_audio.js`. Tạo pyproject/lock cho Gradio + soxr và dependency offline tối thiểu; thêm CUDA extra đúng pin ở nhiệm vụ 2.

**Giao diện:**
- `build_ui(controller) -> gr.Blocks`.
- `browser_audio.apply(command) -> ack | null` và `browser_audio.poll_ack() -> ack | null` trong JS.
- Speaker toggle dừng player ngay, giữ mic, tăng audio_epoch; ack muted không phải completed.
- Mic toggle trả cấu hình `gr.Audio(recording=...)`; nhận chunk `(sample_rate, ndarray)` và stop_recording bằng API công khai.

- [ ] Tạo trang thử Gradio không model với tone WAV dài 5s; nút mic thật và player HTMLAudio có ID do ứng dụng sở hữu.
- [ ] Viết bài browser: bật mic → nhận chunk → phát tone trong lúc mic tiếp tục → gửi stop → nhận stopped/ended đúng playback_id; command generation cũ bị bỏ. Ban đầu test phải thất bại khi player/hook chưa có.
- [ ] Viết test loa off trong lúc phát/TTS/chưa nhận play rồi on: không phát lại file cũ, mic vẫn có chunk; test không cấp quyền mic, autoplay blocked và bật/tắt hai lần; không được giả trạng thái nghe/phát thành công.
- [ ] Đo contract của Gradio 6.30: callback stream trả chunk hay cumulative, stop_recording có toàn bản ghi hay đuôi, thứ tự callback cuối. Ghi đúng vào `docs/validation.md`, không đoán từ tên event. Chốt adapter chỉ nhận mỗi sample đúng một lần.
- [ ] Cài hook JS chỉ cho player do mình sở hữu; chạy trong Gradio, không mở frontend/server thứ hai. Ack được chuyển qua component/event bridge công khai; một Timer ngắn lấy trạng thái, không query CSS class nội bộ.
- [ ] Chạy test browser trên local và mở share link trong Colab để kiểm tra quyền mic/TLS/autoplay thực tế. Nếu chỉ pass fake hoặc headless, ghi chưa xác minh mic/loa thật.
- [ ] Nếu không thể chứng minh nghe và phát đồng thời với API pinned này, dừng phần tích hợp model và báo rõ giới hạn; không đổi thành nhấn thu rồi thả để gửi mà vẫn gọi là ngắt lời realtime.
- [ ] Review/commit phần đã kiểm chứng, lưu bằng chứng trong validation.

## Nhiệm vụ 2 — Cấu hình và model thường trực

**Tệp:** `web_asr/config.py`, `backend.py`, pyproject/uv.lock, .env.example, .gitignore, `tests/test_config.py`, `tests/test_backend.py`.

**Giao diện:**
- `Settings.load(env_path: Path | None = None) -> Settings`; tìm root repo từ `__file__`, ưu tiên ASR_ENV_PATH, mặc định `.env` gốc repo; không phụ thuộc cwd.
- `ModelPool(settings)`, `load() -> None`, `recognize(wave: np.ndarray) -> tuple[str, float]`, `new_vad() -> Callable[[np.ndarray], float]`, `close() -> None`.
- FakeModelPool cùng giao diện cho test/demo; không yêu cầu CUDA khi chỉ chạy fake.

- [ ] Viết test từ cwd bất kỳ vẫn tìm được config/source đúng; không in API key/password; thiếu model/base URL hoặc cấu hình số không hợp lệ báo rõ.
- [ ] Viết test gọi load nhiều lần chỉ tải một lần; hai recognize đồng thời có peak native calls=1; hai VAD instance không chung trạng thái.
- [ ] Chạy `uv run --locked python -m unittest discover -s tests -p 'test_config.py' -v` và `test_backend.py`; xác nhận fail trước khi cài phần tối thiểu.
- [ ] Dùng adapter Gipformer/setup/TurnConfig hiện có qua một điểm import được kiểm soát, không sao chép code. Resolve model paths từ `Sub-ASR/config.toml` và thư mục chứa file đó.
- [ ] Thêm CUDA extra cùng các pin/index trong Sub-ASR; `uv lock`, `uv sync --locked`. Lệnh CPU test không kéo extra CUDA.
- [ ] Test lại pass; notebook Testing cũ và test runtime Sub-ASR không bị đổi hành vi. Commit theo nhiệm vụ.

## Nhiệm vụ 3 — Browser audio thành frame và trạng thái phiên

**Tệp:** `web_asr/audio.py`, `backend.py`, `tests/test_audio.py`, `tests/test_backend.py`.

**Giao diện:**
- `AudioIngress.push(rate: int, samples: np.ndarray) -> list[np.ndarray]`; chuẩn hóa int16/float, stereo→mono, resample có state; mỗi frame đủ 512 float32 sample.
- `AudioIngress.finish() -> list[np.ndarray]`: flush đuôi một lần, pad frame cuối, không lặp khi gọi lần hai. `reset()` dùng khi bắt đầu recording mới.
- `Controller.open_session() -> str`, `set_mic(session_id: str, enabled: bool) -> None`, `push_audio(session_id: str, rate: int, samples: np.ndarray) -> None`, `send_text(session_id: str, text: str) -> None`, `set_speaker(session_id: str, enabled: bool) -> None`, `close_session(session_id: str) -> None`.
- Mỗi lượt kết hợp có input_group_id, input_revision, generation hiện tại, audio_turn_id, spoken_text và typed_additions có timestamp. AudioRunner cần hook lấy transcript đúng group trước khi Session submit; giữ mặc định cũ cho notebook Testing.
- Controller giữ một asyncio loop nền; callback Gradio enqueue nhanh, không await HTTP/ASR/TTS. Chỉ loop đó thay đổi Session/AudioRunner.

- [ ] Viết test 44,1/48 kHz chia chunk không đều → frame có sample count liên tục, không tăng/giảm tốc giọng nói; đầu vào NaN/Inf/rate sai bị từ chối. So sánh resample streaming và batch theo dung sai phù hợp soxr.
- [ ] Viết test stop có phần đuôi, stop gọi hai lần và start mới; mỗi sample/transcript chỉ xử lý một lần theo contract đã đo ở nhiệm vụ 1.
- [ ] Viết test queue quá 3s báo overload rõ; hai session không lẫn history, request, VAD hoặc event. Mic thứ hai bị từ chối khi slot đã có người dùng.
- [ ] Viết test ASR/HTTP fake bị giữ bằng Event nhưng push_audio/stop/text vẫn trả ngay và không làm UI đợi cùng hàng đợi.
- [ ] Chạy test để thấy fail; cài ingress bằng soxr stream + buffer, controller tái dùng AudioRunner, tối đa 8 đoạn chờ. Không dùng `microphone()` của server để lấy mic người dùng.
- [ ] Tắt mic: ngừng nhận recording mới, xử lý phần đuôi và thêm silence để chốt phần đang nói; không kết thúc/hủy state hội thoại. Text submit khi đang nói/ASR của đoạn hiện tại chưa xong: giữ audio, append typed_additions vào group, vô hiệu hóa phản hồi cũ nhưng không bỏ ASR đúng group. Chốt group sau im lặng/tắt mic và dùng generation mới nhất. Nếu không có group chưa commit thì text xử lý ngay. Test các ranh giới này.
- [ ] Viết test nói “đặt vé Đà Nẵng” + gõ “ngày 30/10, 2 vé” trước khi ASR xong: chỉ một lần resolve cho group, request có cả ba dữ kiện, mic không reset. Hai Enter giữ thứ tự; nghỉ dưới 1s không chốt sớm.
- [ ] Viết test Enter ngay trước/sau segment_closed hoặc ASR final, ASR trả muộn sau Enter, audio rỗng/lỗi nhưng text còn, mic bật chưa nói. Không có double-submit/mất text; ASR generation cũ vẫn được điền vào group audio đúng, nhưng LLM/TTS revision cũ không commit.
- [ ] Test text có chỉ dẫn thay/hủy: không ép action append chỉ vì đến đồng thời với audio; lưu nguồn nguyên văn để resolver xét đúng ý định.
- [ ] Chạy `uv run --locked python -m unittest discover -s tests -p 'test_audio.py' -v` và `test_backend.py` đến pass, rồi commit.

## Nhiệm vụ 4 — TTS browser và vòng đời ngắt lời

**Tệp:** `backend.py`, `browser_audio.js`, `tests/test_backend.py`, `tests/browser/`; sửa hook tương thích trong `../Sub-ASR/voice.py` và test tương ứng.

**Giao diện:**
- `BrowserSpeaker(text: str) -> Awaitable[PlaybackResult]` được tiêm vào Session; PlaybackResult phân biệt completed/muted/interrupted/blocked/error. Chỉ completed khi ack ended; adapter local cũ trả None vẫn giữ hành vi tương thích.
- `Controller.read_updates(session_id: str, cursor: int) -> dict`: snapshot text/trạng thái/events mới, cursor mới và playback command. Đọc lại cùng cursor không tạo command_id/playback_id mới.
- `Controller.ack_playback(session_id: str, ack: dict) -> None`; chỉ nhận command/playback/generation/audio_epoch đang có của phiên đó.
- Thêm `speaker_delivery='browser'` vào Session (default giữ hành vi cũ). Tách ghi nhận text phản hồi và trạng thái audio: text hoàn chỉnh vẫn vào history khi loa tắt; chỉ đánh dấu đã nghe trọn khi delivery speaker/browser có status completed. Text hiển thị không phải xác nhận đã đọc/nghe.

- [ ] Viết test câu đầu new dù thiếu ngày, câu sau append ngày, update/replace/cancel/clarify và lỗi resolver; giữ các regression test đang có trong Sub-ASR.
- [ ] Viết test chặn synthesis, reply hoặc playback rồi phát speech_started: capture tiếp tục; command stop được ưu tiên trước play mới; TTS subprocess terminate/kill có giới hạn.
- [ ] Viết test set_speaker(false) khi LLM đang sinh vẫn tiếp tục text, không tạo TTS mới; trong lúc synthesis/playback thì dừng audio, giải phóng waiter và giữ request/mic. Bật lại không replay; generation hội thoại không đổi do speaker toggle, audio_epoch phải đổi.
- [ ] Viết test text history khi muted và audio status riêng; câu bot bị ngắt/muted không được coi là đã nghe trọn.
- [ ] Viết test play/ack cũ đến sau ngắt hoặc tắt/bật loa không phát và không thêm assistant vào history. Ack sai session không được chấp nhận; chỉ hết timer không được mark completed.
- [ ] Viết test blocked/error/no ack/mất tab: cleanup task, audio và mic slot; không báo đã phát trọn. Deadline chờ ended = thời lượng WAV + 10s; start/stop ack chờ tối đa 2s, lỗi thì không giả thành công.
- [ ] Cài tổng hợp eSpeak ra WAV/bytes; cấm gọi loa Colab. Phục vụ audio qua cơ chế file Gradio với allowlist đúng thư mục temp của phiên, tên ngẫu nhiên. Xóa khi hết vòng đời, không dùng tên người dùng làm path.
- [ ] Chạy unit + browser test: nghe thật rồi ngắt trong lúc mic vẫn thu, không chỉ assert output bị clear. Chạy lại `uv run --locked python -m unittest -v test_voice` từ Sub-ASR.
- [ ] Commit khi các test pass; nếu chưa đo trên thiết bị thật, đánh dấu rõ mục đó chưa kiểm định.

## Nhiệm vụ 5 — Giao diện và bảng debug

**Tệp:** `web_asr/ui.py`, `app.py`, `tests/test_ui.py`, `tests/browser/`.

**Giao diện:** `build_ui(controller, settings) -> gr.Blocks`; `app.py` chỉ load settings/models, tạo controller/UI và launch.

- [ ] Test giao diện có ô text Enter để gửi, output read-only cập nhật dần, một nút toggle mic, một nút toggle loa độc lập, player và cột debug bên phải; không thêm dashboard hoặc framework frontend khác.
- [ ] Bind audio stream khoảng 0,25s; chỉ enqueue dữ liệu, event ingestion được tuần tự hóa, không để cơ chế latest-only làm mất chunk. Không chung concurrency group với tác vụ trả lời dài.
- [ ] Bind Timer 0,2s đọc snapshot/event và command, đồng bộ JS player + ack qua event riêng. Không đưa toàn bộ history vào gr.State; state phía client chỉ chứa opaque session_id/cursor/mic flag/speaker flag/audio_epoch cần thiết.
- [ ] Textbox giữ bản nháp người dùng; transcript mic chỉ cập nhật debug, không đè text đang gõ và không tự submit. Enter trong lúc nói gắn text vào group hiện tại; UI hiện “đã thêm text, đang chờ hết lượt/ASR”. Khi generation mới, không append chunk phản hồi cũ trở lại.
- [ ] Debug có trạng thái mic/loa/model/ASR/LLM/TTS, input_group_id/input_revision, nguồn audio/text, transcript tạm/cuối, active_request, action, generation, sự kiện hủy, browser ack, reason/status lỗi. Tối đa 200 dòng và phân biệt requested/observed stop; không hiện API key/password.
- [ ] Test reload, đóng tab, mất kết nối, hai tab, bộ đệm đầy và browser không hỗ trợ/không cho mic. Session expiry 120s không heartbeat; đóng rõ ràng dọn ngay.
- [ ] Test fake UI qua Gradio config/API và browser; sau đó smoke test Colab real. Review/commit phần UI.

## Nhiệm vụ 6 — Notebook khởi chạy và kiểm định tích hợp

**Tệp:** `Run_Web_ASR_Colab.ipynb`, README.md, `tests/test_notebook.py`, `docs/validation.md`.

Notebook theo thứ tự:

1. Hướng dẫn bật T4 GPU, dùng tai nghe và mở share link trong tab riêng để cấp quyền mic.
2. Tìm repo đầy đủ; nếu chưa có thì clone URL/branch người dùng cấu hình. Không tự reset source local.
3. Chọn `.env`/Colab Secrets, kiểm tra các biến cần thiết nhưng không in secret. Không sửa `.env` của Sub-ASR.
4. Kiểm tra NVIDIA GPU; cài uv, `uv sync --locked --extra cuda` trong Web-ASR và chạy setup pinned của Sub-ASR với đường dẫn config tuyệt đối.
5. Khởi chạy `app.py` bằng `uv run --locked --extra cuda python -u app.py`; Gradio 6.30 cùng process worker Python 3.12, không cài CUDA vào kernel Colab.
6. Hiện URL share và trạng thái server. Không nạp model lần nữa theo mỗi kết nối hoặc mỗi request.
7. Cell dừng server: graceful shutdown, chờ cleanup rồi terminate/kill có giới hạn. Chạy lại không tạo hai server hoặc mất dấu process cũ.

- [ ] Viết test notebook schema/syntax, đường dẫn từ repo root/Web-ASR, không có Base64 bundle hoặc secret literal, khởi chạy đúng một server; fake mode chạy không GPU.
- [ ] Chạy `uv run --locked python -m unittest discover -s tests -v`; xác nhận test notebook fail trước khi tạo launcher rồi pass khi hoàn thiện.
- [ ] Chạy notebook từ môi trường sạch với fake model/LLM để chứng minh launcher và UI; không gọi đó là kiểm định CUDA.
- [ ] Trên Colab, thử text + mic + TTS thật và hai browser/tab. Ghi version GPU/model/Gradio, kết quả từng tiêu chí và các mục chưa kiểm định vào validation.
- [ ] Thử vừa nói Đà Nẵng vừa Enter “ngày 30/10, 2 vé”; kiểm tra không đè draft/mất audio và chỉ trả lời khi group đủ dữ liệu. Tắt/bật loa trong lúc nghe/LLM/TTS/phát: mic và text vẫn hoạt động, audio cũ không replay.
- [ ] Dùng các câu Đà Nẵng → ngày 30/10, sửa Huế, thay tác vụ khách sạn, hủy, Hà Nội mơ hồ; xác nhận text và âm thanh browser thuộc đúng generation.
- [ ] Đo thời gian speech_started → stop command → browser stopped ack, cùng đo âm thanh thật nếu đánh giá p95; không tự báo đạt 150ms. Kiểm tra CPU/GPU load và queue khi sử dụng liên tục.
- [ ] Cập nhật README: cách chạy một notebook, link tạm theo vòng đời Colab, giới hạn một mic, chính sách Colab, cách chỉnh config và tái sử dụng backend trên server riêng sau này.

## Thứ tự và điều kiện hoàn thành

Thực hiện **1 → 2 → 3 → 4 → 5 → 6**. Ưu tiên tuần tự vì audio/cancellation/browser ack dùng chung giao diện. Đây là một MVP, không tách thành dự án frontend/backend hay hai notebook.

Hoàn thành phần code khi test offline/browser và notebook fake pass. Chỉ xác nhận luồng mic/TTS/ngắt lời trên Colab sau khi đã nghe và thử trên browser thật. Nếu runtime bị Colab thu hồi hoặc share không khả dụng, ghi đúng trở ngại; không thêm cơ chế né chính sách.

Các mục chưa làm trong MVP: nhiều phiên mic đồng thời, AEC riêng cho loa ngoài, TTS streaming theo câu nhỏ, mobile native, giao dịch đặt vé thật, vận hành 24/7 và cơ sở dữ liệu hội thoại.

# Sub-ASR — trợ lý hội thoại tiếng Việt

Ứng dụng Python chạy ở terminal, **không dùng DF3**:

```text
Micro 16 kHz → Silero VAD (CPU) → bộ đệm lượt nói
                                      ↓
                             Gipformer 1.5 greedy (CUDA)
                                      ↓
                                 Transcript tạm
                                      ↓
                 LLM đánh giá kết thúc lượt + thời gian im lặng
                                      ↓
                      Transcript cuối → LLM → eSpeak NG
```

## Môi trường uv

Đã có `.venv` cho phần điều phối, HTTP, audio và tests; `uv.lock` khóa cả bộ CUDA tùy chọn. Linux x86_64, Python 3.12. Chạy trong thư mục này:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv sync --locked
uv run python -m sub_asr --demo
uv run python -m unittest discover -s tests -v
```

`--demo` là mô phỏng logic có kết quả model giả lập, không gọi Silero, Gipformer hoặc LLM.

## Cài để chạy model và micro thật

Cần GPU NVIDIA và driver tương thích CUDA 12.8. Bản này dùng đúng bộ PyTorch/k2/kaldifeat của benchmark đã chạy; không có fallback Gipformer CPU.

Trên Ubuntu/Debian, cài thư viện thu âm và bộ phát tiếng Việt:

```bash
sudo apt-get install libportaudio2 espeak-ng
uv sync --locked --extra cuda
uv run --extra cuda python -m sub_asr.setup
```

Lệnh setup tải source Gipformer, Icefall và ba file model vào `vendor/`, `models/`. Revision được cố định trong `sub_asr/setup.py`. Các lần chạy model sau **giữ `--extra cuda`** để uv không gỡ bộ thư viện tùy chọn.

Có thể dùng model/source đã tải trước bằng cách sửa các đường dẫn `[asr]` trong `config.toml`. Đường dẫn tương đối tính từ thư mục chứa file cấu hình.

## Cấu hình LLM

```bash
cp .env.example .env
```

Điền `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` theo máy chủ của bạn. Máy chủ cần hỗ trợ:

- `POST {LLM_BASE_URL}/chat/completions` với `messages`.
- Trả JSON thông thường cho `stream=false`; nội dung JSON `{"complete": true/false}` cho câu hỏi kết thúc lượt.
- Trả SSE `data: {"choices":[{"delta":{"content":"..."}}]}` cho `stream=true`.

Không tích hợp riêng với một nhà cung cấp. Có thể dùng máy chủ local hoặc remote cùng giao thức. `TURN_MODEL` tùy chọn nếu muốn model riêng cho nhận định kết thúc lượt; bỏ trống để dùng chung `LLM_MODEL`. Cần model hiểu tiếng Việt; khả năng chốt lượt phải đo trên hội thoại thực tế, không được bảo đảm chỉ từ việc hỗ trợ tiếng Việt.

Audio được xử lý local. Transcript và lịch sử được gửi đến máy chủ LLM đã cấu hình. Ứng dụng không tự lưu bản thu hoặc transcript ra file; text hiện trên terminal.

## Chạy

```bash
uv run --extra cuda python -m sub_asr --list-devices
uv run --extra cuda python -m sub_asr --device 0
```

Dùng tai nghe: bản này chưa có AEC. Nếu dùng loa ngoài, tiếng trợ lý lọt vào micro có thể bị coi là lời người dùng.

Các chế độ khác:

```bash
# In text phản hồi, không phát tiếng
uv run --extra cuda python -m sub_asr --no-tts

# Chỉ nhận dạng/chốt lượt; không sinh phản hồi hoặc phát tiếng
uv run --extra cuda python -m sub_asr --transcribe-only

# Mô phỏng luồng audio từ file, đọc theo thời gian thực
uv run --extra cuda python -m sub_asr --file /path/to/audio_16k.wav --no-tts
```

`--transcribe-only` vẫn dùng nhận định ngữ nghĩa nếu đã cấu hình LLM. Nếu chưa cấu hình, chương trình báo rõ và chỉ dùng `max_wait` giây im lặng. File phải 16 kHz; stereo sẽ được lấy trung bình thành mono. Thêm khoảng im lặng ở EOF để cho phép chốt lượt. Chuyển file khi cần:

```bash
ffmpeg -i input.wav -ac 1 -ar 16000 audio_16k.wav
```

Dừng bằng Ctrl+C.

## Giữ ngữ cảnh và kết thúc lượt

- VAD phải thấy tiếng nói liên tục khoảng `start_speech` trước khi mở lượt/ngắt trợ lý, nhằm giảm kích hoạt bởi tiếng động ngắn. Đây chưa phải bộ chống báo nhầm hoàn chỉnh.
- Giữ phần audio `pre_roll` trước khi mở lượt để tránh mất đầu từ.
- Khoảng nghỉ ngắn không reset buffer. Gipformer giải mã lại audio tích lũy theo `partial_interval`; transcript mới **thay thế** text tạm cũ, không nối máy móc.
- LLM chỉ được hỏi khi có khoảng nghỉ và transcript đã bao phủ phần tiếng nói mới nhất. Kết quả cũ không được dùng nếu người dùng đã nói tiếp.
- Nếu người dùng nói lại trước khi chốt, tiếp tục cùng lượt và bỏ nhận định kết thúc cũ.
- Sau khi chốt, text được thêm vào lịch sử trước khi LLM trả lời. Khi đang sinh/phát câu trả lời mà có lời người dùng, hủy tác vụ và dừng TTS.
- Lịch sử tồn tại trong RAM, mặc định giữ khoảng 20 message. Không có trí nhớ dài hạn. Với phản hồi bị ngắt lúc TTS đang đọc, lịch sử đánh dấu phản hồi bị ngắt, chưa biết chính xác người dùng đã nghe đến từ nào.

Giá trị ban đầu trong `config.toml` (đơn vị giây, chưa hiệu chỉnh trên người dùng thật):

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `start_speech` | 0.16 | Xác nhận bắt đầu nói/ngắt lời |
| `pre_roll` | 0.32 | Audio giữ trước khi bắt đầu |
| `min_silence` | 0.8 | Chờ tối thiểu khi LLM cho rằng đã nói xong |
| `incomplete_silence` | 2.5 | Chờ lâu hơn khi LLM cho rằng còn dang dở |
| `max_wait` | 4.0 | Chờ dự phòng khi chưa biết/lỗi/ngữ nghĩa quá chậm |
| `partial_interval` | 1.0 | Khoảng giữa các yêu cầu giải mã lại lúc đang nói |
| `max_turn` | 30.0 | Giới hạn buffer; chốt bắt buộc, có thể tách câu dài |

Các khoảng trên tính từ **audio**, không tính từ lúc HTTP/ASR trả kết quả. Độ trễ thực tế còn phụ thuộc ASR và network. Nếu nhận định ngữ nghĩa lỗi/timeout/JSON không đúng, chương trình log fallback và chờ `max_wait`, không coi lỗi là “đã nói xong”. Không chốt khi chưa nhận dạng hết phần giọng nói hiện tại.

## Giới hạn của bản đầu

- Gipformer ở đây là recipe giải mã clip đã dùng trong benchmark, **không phải streaming encoder giữ state**. Giải mã lại cả lượt tốn thêm tính toán theo độ dài audio; chưa bảo đảm real-time ở mọi máy. Capture dùng queue có giới hạn và báo lỗi khi tràn, không âm thầm bỏ khung.
- Bản terminal hỗ trợ micro; giao diện Marimo hỗ trợ upload file. Chưa có server phục vụ nhiều người dùng, AEC hoặc deployment.
- LLM stream text ra terminal; eSpeak NG bắt đầu đọc sau khi toàn bộ câu trả lời được tạo. Chưa có TTS theo từng câu/token. Giọng eSpeak là lựa chọn chạy local cơ bản, không phải giọng tự nhiên chất lượng cao.
- VAD có thể bỏ sót giọng nhỏ hoặc nhận nhầm tiếng động. Không có cam kết loại hết hallucination noise-only. Các ngưỡng phải đo lại trên micro, môi trường và người nói thực tế.
- Semantic endpoint dựa trên text không biết chắc ý định người nói. Cần bổ sung test hội thoại có khoảng nghỉ được gán nhãn, đo tỷ lệ cắt ngang và thời gian phản hồi.

## Mã nguồn

- `sub_asr/turns.py`: trạng thái lượt nói và audio buffer.
- `sub_asr/asr.py`: adapter Gipformer CUDA độc lập với báo cáo benchmark.
- `sub_asr/app.py`: điều phối bất đồng bộ, chống kết quả cũ và ngắt lời.
- `sub_asr/services.py`: HTTP LLM, quyết định ngữ nghĩa và TTS có thể hủy.
- `sub_asr/audio.py`: micro và mô phỏng file.
- `sub_asr/setup.py`: tải source/model cố định revision.

Tham khảo upstream: [Gipformer](https://github.com/ggroup-ai-lab/gipformer), [Icefall](https://github.com/k2-fsa/icefall), [Silero VAD](https://github.com/snakers4/silero-vad). Không sửa các notebook/benchmark cũ trong thư mục cha.

## Upload audio trên Marimo / Molab

Đã có giao diện `molab_app.py`, không cần quyền micro hoặc PortAudio. Mỗi file được coi là **một lượt nói**: VAD kiểm tra có lời nói, sau đó Gipformer greedy nhận dạng **nguyên file**, giữ cả ngữ cảnh trước và sau khoảng nghỉ. Đây là xử lý file hoàn chỉnh, không mô phỏng chốt lượt realtime. File chỉ có nhiễu mà VAD không nhận ra lời nói sẽ trả transcript rỗng.

### Mở local

```bash
uv sync --locked --extra ui --extra cuda
uv run --extra ui --extra cuda marimo edit molab_app.py
```

Trong giao diện, bấm **Cài môi trường CUDA và tải model**, upload file rồi bấm **Nhận dạng**. Có thể bỏ bước cài nếu đã setup trước.

### Mirror từ GitHub lên Molab

Đưa code trong `Sub-ASR/` lên repo `https://github.com/Shinoaki0145/ASR`, nhánh `main`. Chỉ đưa source/config mẫu/uv.lock lên GitHub; `.env`, `.venv`, model và vendor đã được bỏ qua bởi `.gitignore`.

1. Trên Molab, mở menu tạo notebook và chọn mirror/sync từ GitHub.
2. Dán URL notebook: `https://github.com/Shinoaki0145/ASR/blob/main/Sub-ASR/molab_app.py`.
3. Bật GPU, bấm **Cài môi trường CUDA và tải model**.
4. Upload audio vào giao diện, bấm **Nhận dạng**. Không cần upload thêm source.

Link mở nhanh sau khi đã push: [Open in Molab](https://molab.marimo.io/github/Shinoaki0145/ASR/blob/main/Sub-ASR/molab_app.py).

Notebook dùng source local nếu đã có đủ module; khi mirror chỉ có notebook, nó tự clone repo vào thư mục làm việc `.sub_asr_source`. Bấm cài môi trường lại sẽ cập nhật checkout được quản lý này từ `main`. Không tự reset hay ghi đè source local của bạn. Khi thay đổi source, push GitHub trước rồi cập nhật Molab. Nếu fork repo hoặc đổi nhánh, sửa `repo`/`branch` trong hàm `resolve_project` của notebook.

Notebook dùng uv project Python 3.12 riêng; không ép đổi Torch/Python của kernel Marimo. Các lỗi cài/CUDA hiện trong log cell. Khả năng wheel k2/kaldifeat chạy trên GPU thực tế của Molab vẫn cần xác minh.

Bật **Gửi transcript cho trợ lý** nếu đã cấu hình LLM trong biến môi trường Molab hoặc `.env` ở thư mục project runtime. Trên Molab, `127.0.0.1` chỉ máy chủ cloud, không phải máy cá nhân; dùng endpoint mà Molab truy cập được.

Bản upload trả **text** cho phản hồi trợ lý, chưa phát TTS trong browser. Các file upload có phản hồi LLM được giữ trong lịch sử phiên; dùng **Xóa lịch sử trợ lý** để bắt đầu hội thoại khác. Mỗi lần bấm nhận dạng tạo worker mới và tải lại model; số ASR ms hiển thị không gồm chi phí khởi động/tải model.

Hỗ trợ WAV/FLAC/OGG/MP3 nếu libsndfile đọc được. M4A/WebM và codec khác dùng FFmpeg nếu có trên máy chủ. Không đổi tên file để giả định dạng; nội dung được giải mã theo codec. Tối đa **25 MB và 60 giây/file**; quá giới hạn sẽ báo lỗi, không cắt ngắn âm thầm. Audio được đổi mono 16 kHz bằng resampling, không peak-normalize. Nếu codec không hỗ trợ, chuyển sang WAV trước.

Tài liệu API: [upload file](https://docs.marimo.io/api/inputs/file/), [GitHub mirror trên Molab](https://docs.marimo.io/guides/molab/#mirror-notebooks-from-github).

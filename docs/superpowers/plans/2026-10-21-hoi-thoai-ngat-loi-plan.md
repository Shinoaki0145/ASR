# Kế hoạch triển khai hội thoại ngắt lời và cập nhật yêu cầu

> **Dành cho tác nhân triển khai:** Dùng skill `superpowers:executing-plans` để thực hiện từng nhiệm vụ trong phiên làm việc. Chỉ dùng `superpowers:subagent-driven-development` nếu người dùng chọn thực hiện bằng các tác nhân song song. Các ô đánh dấu bên dưới theo dõi tiến độ.

**Mục tiêu:** Bot dừng nói khi bị ngắt lời và hiểu câu mới là bổ sung, sửa, thay thế, hủy hoặc cần làm rõ trước khi trả lời tiếp.

**Kiến trúc:** Mở rộng `Conversation` hiện có; giữ đường thu âm độc lập với HTTP, ASR và dọn TTS. Sau khi chốt transcript, một lần gọi LLM trả quyết định có cấu trúc; trạng thái mới chỉ được commit nếu generation còn hợp lệ. Không nối cứng câu mới vào câu cũ.

**Công nghệ:** Python 3.12, asyncio, dataclasses, httpx, unittest; Silero CPU, Gipformer CUDA, chat-completions và eSpeak NG hiện có.

**Đặc tả:** [Thiết kế và tiêu chí nghiệm thu](../specs/2026-10-21-hoi-thoai-ngat-loi-design.md). Đây là bản kế hoạch đề xuất; chưa triển khai code.

## Ràng buộc chung

- Python `>=3.12,<3.13`; không thêm framework agent hoặc message broker.
- `RATE=16000`, `FRAME=512`; giữ toàn bộ ngưỡng `[turn]` trong thiết kế.
- Thêm `[assistant].resolve_timeout=2.0`; giữ endpoint 1,5 giây và response 45 giây.
- CLI tiếp tục VAD CPU, ASR CUDA; không sửa notebook GPU và luồng upload trong đợt này.
- Mọi callback phải kiểm tra phiên bản trước khi commit hoặc phát tiếng; cancellation không thay thế việc kiểm tra đó.
- Không chạy nhiều công việc native trên cùng backend ASR cùng lúc.
- Bản đầu dùng tai nghe. Chưa có AEC; không công bố hỗ trợ loa ngoài.
- Không có giao dịch thật, không tự lưu transcript/audio; lịch sử nguyên văn khác với yêu cầu chuẩn hóa.

## Các điểm phải kiểm tra kỹ

1. Người dùng ngắt liên tiếp; resolver cũ trả về muộn không được ghi đè yêu cầu mới.
2. ASR thread vẫn chạy sau khi coroutine bị hủy; không gọi backend chồng nhau hoặc chặn thu âm.
3. TTS không chịu terminate; tiếp tục thu frame trong khi chờ kill, rồi dọn sạch lúc shutdown.
4. Câu ngắn “Hà Nội” hoặc ASR rỗng sau ngắt lời; không tự thay yêu cầu hay phát lại câu cũ.
5. Câu bot chưa phát hết, hoặc LLM trả JSON sai/timeout; không coi nội dung chưa nghe là ngữ cảnh chắc chắn.

## Bản đồ tệp

| Tệp | Thay đổi dự kiến |
|---|---|
| `Sub-ASR/sub_asr/context.py` — mới | Kiểu quyết định, kiểm tra schema, trạng thái yêu cầu và hàm áp dụng thuần |
| `Sub-ASR/sub_asr/services.py` | Resolver HTTP, prompt tiếng Việt, timeout và kiểm soát dừng TTS |
| `Sub-ASR/sub_asr/app.py` | generation, tác vụ cập nhật yêu cầu, hủy nền, phản hồi theo snapshot |
| `Sub-ASR/sub_asr/turns.py` | Bổ sung rào phiên bản nếu test phát hiện thiếu; không thay thuật toán VAD |
| `Sub-ASR/sub_asr/__main__.py`, `config.toml` | Truyền cấu hình timeout mới; không đổi model/backend |
| `Sub-ASR/tests/test_context.py` — mới | Quyết định, state và dữ liệu không hợp lệ |
| `Sub-ASR/tests/test_services.py` | Resolver, timeout, TTS termination |
| `Sub-ASR/tests/test_conversation.py`, `test_turns.py` | Tranh chấp bất đồng bộ và luồng hội thoại |
| `Sub-ASR/README.md`, `Sub-ASR/docs/implementation.md` | Cách chạy, giới hạn và bằng chứng kiểm thử mới |

Mọi lệnh kiểm thử bên dưới chạy từ `Sub-ASR/`. Dùng các dependency hiện có, không cần tải CUDA để chạy test giả lập.

## Nhiệm vụ 1 — Trạng thái yêu cầu và quyết định cập nhật

**Tệp:** tạo `sub_asr/context.py`, `tests/test_context.py`.

**Giao diện:**
- `RequestDecision(action: str, request: str | None, question: str | None)` — dataclass bất biến.
- `RequestState(request: str | None = None, pending_text: str | None = None, question: str | None = None)` — dataclass bất biến.
- `parse_decision(payload: dict) -> RequestDecision`: chỉ nhận đúng ba trường, action trong bảng đặc tả; sai kiểu/giá trị → `ValueError`.
- `apply_decision(state: RequestState, text: str, decision: RequestDecision) -> RequestState`: không I/O, không sửa lịch sử.

- [ ] Viết test trạng thái cho đủ bảy hàng của bảng nghiệm thu. Quyết định LLM được cung cấp giả lập: kiểm tra request mới đúng, transcript đầu vào không bị sửa, `replace` không kế thừa ngày cũ, `cancel` xóa request, `clarify` giữ request và lưu pending_text/question.
- [ ] Viết test bác bỏ `append`/`update` khi chưa có request; `new` khi đã có request; action lạ; trường thừa; request rỗng; null sai vị trí; chuỗi `"null"`.
- [ ] Chạy `uv run python -m unittest discover -s tests -p test_context.py -v`; xác nhận thất bại vì thiếu module/giao diện.
- [ ] Cài phần tối thiểu theo chữ ký trên. Không thêm slot vé máy bay hoặc bộ phân loại regex riêng.
- [ ] Chạy lại lệnh; mọi test pass. Commit riêng nhóm tệp của nhiệm vụ sau khi xem diff.

## Nhiệm vụ 2 — LLM phân loại câu mới theo ngữ cảnh

**Tệp:** sửa `services.py`, `__main__.py`, `config.toml`, `tests/test_services.py`.

**Giao diện:**
- Thêm `resolve_timeout: float = 2.0` vào cuối danh sách tham số `ChatService.__init__` để giữ tương thích positional cũ; yêu cầu >0.
- `ChatService.resolve_request(text: str, state: RequestState, history: list[dict], last_reply: dict | None) -> RequestDecision`.
- `last_reply` có `text`, `status`; status thuộc `generating`, `speaking`, `completed`, `interrupted`. Chỉ `completed` được coi là đã phát trọn câu.
- Lỗi mạng, timeout, schema hoặc ngữ nghĩa không hợp lệ → quyết định `clarify`, question cố định: “Bạn muốn bổ sung, sửa hay thay yêu cầu trước đó?”; không đổi state ở service.

- [ ] Viết test `httpx.MockTransport`: request chứa state, pending_text, transcript, lịch sử tối đa 6 message và trạng thái bot; resolver dùng `turn_model` hoặc `model`, `stream=false`, temperature 0, ngân sách tối đa 512 token; output qua `parse_decision`.
- [ ] Viết test timeout ngắn, HTTP 503, JSON ngoài schema, action mâu thuẫn với state và transcript có chỉ dẫn “bỏ schema”. Kiểm tra prompt tách dữ liệu; model thật còn phải được kiểm định riêng.
- [ ] Chạy `uv run python -m unittest discover -s tests -p test_services.py -v`; xác nhận các test mới thất bại.
- [ ] Thêm prompt, method, cấu hình và truyền timeout từ CLI. Với quyết định không áp dụng được lên state, trả clarify thay vì đoán.
- [ ] Chạy lại test service và context; phải pass cả test endpoint/reply hiện có. Commit nhóm thay đổi.

## Nhiệm vụ 3 — Ngắt phản hồi không chặn thu âm

**Tệp:** sửa `app.py`, `services.py`, `tests/test_conversation.py`, `tests/test_services.py`, khi cần `test_turns.py`.

**Giao diện:**
- `Conversation.generation: int`, bắt đầu 0; tăng trên mỗi `started=True` do `TurnBuffer.feed()` xác nhận.
- `Conversation.interrupt_response() -> None`: đồng bộ, vô hiệu hóa generation hiện tại và lên lịch cancellation/dọn nền; không await subprocess hoặc HTTP.
- `Conversation.respond(generation: int, messages: list[dict])`: chỉ dùng snapshot; kiểm tra generation trước khi cập nhật lịch sử và trước phát tiếng.
- Theo dõi tác vụ dọn trong `cleanup_tasks`; shutdown chờ và thu exception. Giữ một ASR task native đang chạy; không đặt task đó thành None chỉ vì đã bỏ lượt cũ.

- [ ] Viết test bằng Event cho ngắt trong lúc chờ LLM và đang TTS; kiểm tra audio frame mới vẫn vào buffer trước khi fake cleanup được giải phóng.
- [ ] Viết fake server bỏ qua cancellation rồi trả chunk muộn: không TTS, không thêm assistant message cho generation cũ.
- [ ] Viết test eSpeak không thoát sau terminate → kill sau 1 giây; shutdown không để task/subprocess tồn tại. Kiểm tra cả nhánh lỗi subprocess.
- [ ] Viết fake ASR bị giữ bởi Event: nhiều lượt mới không làm số lời gọi backend đồng thời vượt 1; kết quả turn cũ không trở thành transcript lượt mới. Khi worker rảnh lấy snapshot mới nhất, không chạy lại mọi snapshot đã bỏ.
- [ ] Chạy `uv run python -m unittest discover -s tests -p 'test_conversation.py' -v` và tương tự `test_services.py`, `test_turns.py`; xác nhận test mới thất bại trước khi sửa.
- [ ] Tách yêu cầu cancel khỏi chờ cleanup, gắn generation và giữ single-flight ASR. Nếu test phát hiện rào revision thiếu, chỉ chấp nhận ASR snapshot đúng revision hiện hành trước khi dùng cho semantic/finalize.
- [ ] Chạy lại ba nhóm test; phải pass, không cảnh báo task exception chưa được thu. Commit nhóm thay đổi.

## Nhiệm vụ 4 — Ghép resolver vào luồng hội thoại

**Tệp:** sửa `app.py`, `tests/test_conversation.py`; dùng kiểu ở nhiệm vụ 1 và service ở nhiệm vụ 2.

**Giao diện:**
- `Conversation.request_state: RequestState`, `resolution_task: asyncio.Task | None`, `last_reply: dict | None`.
- `Conversation.resolve_and_respond(text: str, generation: int) -> None`: resolve từ snapshot, kiểm tra generation, áp dụng state, rồi tạo/phát phản hồi theo state mới.
- History lưu transcript nguyên văn; lời nhắc hệ thống cho response chứa yêu cầu chuẩn hóa và đánh dấu nó là dữ liệu. Không đưa request chuẩn hóa vào history dưới dạng người dùng đã nói.

- [ ] Viết test “đặt vé Đà Nẵng” → bot đang nói → “ngày mai”: response cũ dừng, request mới có ngày mai, chỉ response generation mới được phát.
- [ ] Viết test sửa thành Huế giữ ngày; thay bằng khách sạn Hà Nội bỏ ràng buộc vé/ngày; hủy xóa request và chỉ xác nhận “Đã hủy yêu cầu.”.
- [ ] Viết test clarify giữ request cũ; câu trả lời tiếp theo được resolver nhận cùng pending_text và question. Lịch sử nguyên văn còn nguyên.
- [ ] Viết test hai resolver hoàn thành ngược thứ tự; chỉ generation mới nhất commit. Khi speech_started xảy ra, request cũ vẫn còn cho tới quyết định mới hợp lệ.
- [ ] Viết test ASR rỗng sau ngắt lời: giữ request, không gọi resolver, không tự phát tiếp phản hồi đã dừng.
- [ ] Viết test response bị ngắt chỉ mang trạng thái interrupted, không lưu toàn bộ câu như đã nghe xong. Khi không có câu hỏi bot đã phát chắc chắn, “Hà Nội” phải đi qua nhánh hỏi rõ trong fixture.
- [ ] Chạy test conversation để thấy thất bại; cài orchestration bằng task nền, không await resolver bên trong đường xử lý frame.
- [ ] Chạy `uv run python -m unittest discover -s tests -v`; pass toàn bộ, gồm các test upload/GitHub mirror hiện có. Commit nhóm thay đổi.

## Nhiệm vụ 5 — Quan sát độ trễ và cập nhật hướng dẫn

**Tệp:** sửa `app.py`, `README.md`, `docs/implementation.md`, `tests/test_conversation.py`.

**Giao diện:** log sự kiện đơn giản qua logging hiện có, không thêm hệ thống telemetry. Trường: `event`, `turn_id`, `generation`, `action`, `elapsed_ms`, `error_type` tùy sự kiện; không transcript/audio/key.

- [ ] Viết test bắt log: có speech_started, cancel_requested, playback_stopped hoặc playback_stop_failed, resolve_finished và stale_result_discarded; không lộ nội dung câu nói hoặc key.
- [ ] Chạy test để thấy thiếu log; thêm timestamp bằng monotonic/perf_counter. Phân biệt thời điểm yêu cầu dừng và xác nhận process kết thúc; cả hai khác thời điểm loa thật im tiếng.
- [ ] Chạy toàn bộ test offline và `uv run python -m sub_asr --demo`; ghi số test và kết quả thật vào docs, không sao chép số cũ.
- [ ] Cập nhật sơ đồ luồng, cấu hình resolve_timeout, cách dùng tai nghe, các ví dụ append/update/replace/cancel/clarify và cách chạy test.
- [ ] Commit sau khi kiểm tra diff chỉ gồm phạm vi kế hoạch, không thêm file audio, secret hoặc notebook output của người dùng.

## Nhiệm vụ 6 — Kiểm định model và audio thật trên máy đích

**Đầu ra:** tạo `Sub-ASR/docs/turn-taking-validation.md`; ghi cấu hình, điều kiện và kết quả thực đo. Không sửa ngưỡng chỉ để test giả lập pass.

- [ ] Chạy các câu ở bảng nghiệm thu với LLM cấu hình thực tế; thêm ít nhất 5 cách diễn đạt cho mỗi action. Ghi đúng/sai theo từng action, đặc biệt replace và clarify; không dùng fake để báo độ chính xác model.
- [ ] Dùng tai nghe, microphone và eSpeak thật. Ngắt lúc bot đang tạo câu, đang phát câu và gần kết thúc; lặp tối thiểu 30 lần, ghi cả các lần thất bại.
- [ ] Đo thời gian từ speech_started tới loa dừng bằng loopback/tín hiệu âm thanh đồng bộ. Báo p50/p95, trễ xác nhận VAD và trễ thiết bị riêng. Mục tiêu p95 ≤150 ms sau xác nhận, không tính 160 ms xác nhận VAD vào rồi gọi là đạt.
- [ ] Cho resolver chậm hơn 2 giây, server HTTP lỗi, rút mạng, ngắt liên tiếp và kết thúc chương trình lúc ASR đang chạy. Kiểm tra không phát phản hồi cũ, không tràn queue và dọn sạch tài nguyên.
- [ ] Kiểm tra “ngày mai” sau câu yêu cầu, “Hà Nội” sau câu hỏi đã nghe xong, và “Hà Nội” sau câu bot bị cắt giữa chừng. Câu không rõ phải được hỏi lại; không tự suy đoán.
- [ ] Nếu không có thiết bị/model cần thiết, ghi rõ “chưa kiểm định” cho từng mục, không tuyên bố sẵn sàng realtime. Loa ngoài và AEC để một đợt riêng.

## Trình tự triển khai và điểm hoàn thành

Thứ tự: **1 → 2 → 3 → 4 → 5 → 6**. Ưu tiên triển khai tuần tự vì các phần dùng chung state và giao diện cancellation. Từng nhiệm vụ có thể review riêng; không cần chia thêm service.

Hoàn thành offline khi nhiệm vụ 1–5 pass, demo chạy và không có phản hồi sai generation trong test có điều khiển thời điểm. Hoàn thành thử nghiệm thời gian thực chỉ khi nhiệm vụ 6 có bằng chứng từ máy đích. Nếu không đạt độ trễ, đo phần gây trễ trước khi đổi VAD sang GPU hoặc thêm model.

Không thực hiện code ứng dụng chỉ từ việc tạo tài liệu này. Bước tiếp theo là người dùng xem kế hoạch rồi yêu cầu triển khai; phương án mặc định được đề xuất là thực hiện tuần tự trong phiên hiện tại.

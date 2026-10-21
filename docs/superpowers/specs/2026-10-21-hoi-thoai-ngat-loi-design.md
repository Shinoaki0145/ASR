# Thiết kế hội thoại tiếng Việt: ngắt lời và cập nhật yêu cầu

Ngày: 21/10/2026. Trạng thái: đề xuất để triển khai; chưa thay đổi chương trình.

## 1. Mục tiêu và phạm vi

Khi người dùng nói trong lúc bot đang tạo hoặc phát câu trả lời, bot phải dừng phản hồi cũ, tiếp tục nghe và hiểu câu mới theo ngữ cảnh. Câu mới có thể bổ sung, sửa một phần, thay toàn bộ, hủy yêu cầu hoặc cần hỏi lại. Không mặc định nối hai transcript.

Triển khai trong ứng dụng microphone `Sub-ASR`. Notebook `Silero_VAD_Test.ipynb` vẫn phục vụ thử VAD trên file; `molab_app.py` vẫn là giao diện upload từng lượt. Hai giao diện đó không chứng minh được ngắt lời thời gian thực.

Không thực hiện đặt vé hay giao dịch thật. Chưa thêm công cụ thực thi, bộ nhớ dài hạn, dịch vụ hàng đợi, TTS streaming hoặc AEC trong đợt này. Bản đầu dùng tai nghe; hỗ trợ loa ngoài cần giai đoạn AEC và kiểm thử riêng.

## 2. Nền tảng đang có

- `turns.py`: gom frame 512 sample, 16 kHz; pre-roll, xác nhận bắt đầu nói, revision và chờ im lặng.
- `app.py`: điều phối ASR nền, semantic endpoint, LLM trả lời và ngắt tác vụ phản hồi.
- `services.py`: `ChatService.end_of_turn()`, `reply()` và `speak()` hủy được subprocess eSpeak.
- `audio.py`: hàng đợi microphone có giới hạn; báo lỗi nếu tràn.
- Chưa có trạng thái yêu cầu đã chuẩn hóa và quyết định cập nhật yêu cầu rõ ràng.
- `consume()` hiện chờ `cancel()` hoàn tất; việc dọn TTS có thể chờ tới 1 giây trước khi kill. Phải tách dọn tài nguyên khỏi đường xử lý frame.
- ASR dùng `asyncio.to_thread`; hủy coroutine không đảm bảo dừng được công việc native đang chạy.
- Notebook đã chạy VAD CUDA, nhưng CLI hiện chạy VAD CPU và ASR CUDA. Đây là hai cấu hình khác nhau.

## 3. Quyết định thiết kế

### 3.1. Tách ba loại nhận định

1. **Bắt đầu nói:** VAD + thời gian xác nhận; dừng tiếng bot ngay, không chờ LLM.
2. **Kết thúc lượt nói:** dùng `end_of_turn()` và ngưỡng im lặng hiện có.
3. **Ý nghĩa câu mới:** sau transcript cuối, gọi `resolve_request()` để cập nhật yêu cầu trước khi sinh câu trả lời.

LLM đọc yêu cầu hiện tại, transcript mới, lịch sử gần đây và trạng thái phản hồi bot. Câu bot đã phát xong hữu ích để hiểu câu trả lời ngắn; câu mới chỉ được tạo nhưng chưa phát không được coi là người dùng đã nghe.

### 3.2. Trạng thái nhỏ, không xây hệ thống slot cho mọi lĩnh vực

Giữ `active_request: str | None` là diễn đạt chuẩn hóa của yêu cầu hiện tại. Lịch sử vẫn giữ nguyên các câu người dùng nói; không sửa transcript cũ thành câu ghép do LLM sinh.

Quyết định có ba trường: `action`, `request`, `question`.

| action | Ý nghĩa | request | question |
|---|---|---|---|
| `new` | Chưa có yêu cầu trước, hoặc bắt đầu sau hủy | Câu yêu cầu mới đầy đủ | null |
| `append` | Bổ sung thông tin | Câu yêu cầu sau bổ sung | null |
| `update` | Sửa một phần yêu cầu | Câu yêu cầu sau sửa | null |
| `replace` | Thay toàn bộ tác vụ | Chỉ yêu cầu mới, không kế thừa ràng buộc cũ | null |
| `cancel` | Hủy tác vụ hiện tại | null | null |
| `clarify` | Chưa đủ rõ để cập nhật an toàn | null | Câu hỏi ngắn bằng tiếng Việt |

`append`/`update` cần có yêu cầu hiện tại. `replace` chỉ dùng khi có yêu cầu cũ; nếu chưa có thì dùng `new`. Giá trị null là JSON null, không phải chuỗi. Không dùng điểm confidence tự báo của LLM như xác suất đã được hiệu chuẩn.

Khi `clarify`, giữ nguyên yêu cầu cũ, lưu transcript đang cần làm rõ để câu trả lời tiếp theo có đủ ngữ cảnh. Khi `cancel`, xóa yêu cầu và chỉ xác nhận ngắn; không chạy lại tác vụ cũ. Timeout/JSON lỗi không được âm thầm nối hoặc thay yêu cầu: giữ trạng thái và hỏi lại.

### 3.3. Song song và chống kết quả cũ

Dùng `asyncio` hiện có; không thêm process hoặc message broker. Microphone tiếp tục hoạt động trong lúc chờ HTTP, ASR và phát tiếng. Chỉ một chủ thể cập nhật trạng thái hội thoại: `Conversation` trên event loop.

Có hai loại khóa phiên bản riêng:

- `(turn_id, revision)` bảo vệ transcript/endpoint của bộ đệm audio.
- `generation` tăng ngay khi xác nhận người dùng bắt đầu nói; bảo vệ resolver, câu trả lời và phát tiếng.

Bất kỳ tác vụ nào trả về cũng phải kiểm tra khóa trước khi sửa trạng thái, lịch sử hoặc phát tiếng. Chỉ `task.cancel()` là chưa đủ: máy chủ hoặc thread có thể vẫn hoàn tất công việc.

Khi ngắt lời: tăng generation → yêu cầu dừng TTS/hủy tác vụ → tiếp tục xử lý frame. Chờ dọn subprocess ở tác vụ nền; giữ tham chiếu, thu exception và chờ dọn trong shutdown. Không tạo vô hạn tác vụ ASR: giữ một công việc native đang chạy; frame mới vẫn được gom, chỉ gửi snapshot mới nhất khi worker rảnh.

Giữ yêu cầu cũ trong lúc nghe câu ngắt lời; chỉ commit thay đổi khi có transcript cuối và quyết định hợp lệ, đúng generation. Nếu bot đã bị ngắt, không tự tiếp tục phát câu cũ.

### 3.4. Nội dung đã phát

Phân biệt `generating`, `speaking`, `completed`, `interrupted`. Với eSpeak hiện tại không biết chính xác từng từ đã nghe. Phản hồi bị ngắt chỉ được đánh dấu đã phát một phần; không đưa toàn bộ nội dung đó vào lịch sử như câu đã nghe trọn vẹn. Câu hỏi nằm trong phần chưa xác nhận đã phát không đủ làm căn cứ tự suy diễn câu trả lời ngắn.

## 4. Cấu hình và ràng buộc

- Python `>=3.12,<3.13`; dùng `asyncio`, `dataclasses`, `unittest`, `httpx` hiện có; không thêm framework agent.
- Giữ `RATE=16000`, `FRAME=512`; `start_speech=0.16`, `pre_roll=0.32` giây.
- Giữ `min_silence=0.8`, `incomplete_silence=2.5`, `max_wait=4.0`, `max_turn=30.0`, `partial_interval=1.0` giây.
- Giữ endpoint `semantic_timeout=1.5`, response `response_timeout=45.0` giây; thêm `resolve_timeout=2.0` giây, độc lập endpoint.
- Giữ Silero VAD CPU trong CLI, Gipformer CUDA. Không đổi cấu hình GPU notebook trong công việc này.
- Dùng chat-completions server hiện có; resolver dùng `turn_model` nếu đã cấu hình, ngược lại dùng `model`.
- Transcript là dữ liệu; không làm theo chỉ dẫn yêu cầu đổi schema hoặc bỏ qua quy tắc bên trong transcript.
- Không tự lưu audio, transcript, API key vào log trên đĩa. Log chẩn đoán chỉ chứa ID, action, trạng thái, độ trễ, loại lỗi.

## 5. Tiêu chí nghiệm thu

| Yêu cầu hiện tại | Câu mới/ngữ cảnh | Kết quả mong đợi |
|---|---|---|
| Chưa có | Tôi muốn đặt vé đi Đà Nẵng | `new`, lưu yêu cầu |
| Đặt vé Đà Nẵng | Vào ngày mai | `append`, thêm ngày đi |
| Đặt vé Đà Nẵng ngày mai | À không, đi Huế | `update`, giữ ngày, đổi nơi đến |
| Đặt vé Đà Nẵng ngày mai | Thôi, tìm khách sạn Hà Nội | `replace`, không giữ ngày mai nếu câu mới không nêu |
| Yêu cầu bất kỳ | Không cần nữa | `cancel`, yêu cầu thành null |
| Đặt vé Đà Nẵng | Hà Nội; chưa có câu hỏi rõ | `clarify`, không tự đổi điểm đi/đến |
| Đặt vé Đà Nẵng | Bot đã hỏi xong “Bạn đi từ đâu?”; Hà Nội | `append`, bổ sung điểm đi |

Test giả lập kiểm chứng cách áp dụng quyết định và race condition; không chứng minh chất lượng phân loại của LLM thật. Cần chạy bộ câu trên với model thực tế, thêm cách nói tương đương và kiểm tra thủ công trước khi kết luận dùng được.

Mục tiêu đo ban đầu, chưa phải kết quả đạt được: p95 từ xác nhận `speech_started` tới dừng âm thanh ≤150 ms trên máy đích, ghi riêng 160 ms xác nhận VAD và độ trễ thiết bị; không tràn hàng đợi; không phát phản hồi sai generation. Đo bằng microphone/loopback thật, không suy ra từ thời điểm gọi cancel.

# Thiết kế Web-ASR chạy Gradio trên Colab

Ngày: 22/10/2026. Trạng thái: bản thiết kế và kế hoạch; chưa triển khai ứng dụng.

## 1. Yêu cầu đã thống nhất

- Tạo thư mục mới `Web-ASR`; viết ứng dụng bằng file Python như bình thường.
- Notebook `Run_Web_ASR_Colab.ipynb` chỉ chuẩn bị môi trường, tải model và khởi chạy ứng dụng.
- Giao diện Gradio đơn giản: một ô nhập text, một ô hiển thị phản hồi LLM, một nút Bật mic/Tắt mic và một nút Bật loa/Tắt loa độc lập; cột bên phải để theo dõi trạng thái, sự kiện và phản hồi phục vụ debug.
- Nhấn Enter trong ô text để gửi, không bắt buộc thêm nút gửi. Vẫn nhập/gửi được khi đang nói; text này bổ sung vào cùng lượt đa phương thức, không hủy audio đã thu.
- Có bot phát tiếng trong browser; khi người dùng nói tiếp thì dừng tiếng bot, tiếp tục nhận dạng và cập nhật ngữ cảnh.
- Giữ luồng new/append/update/replace/cancel/clarify đã thử. Câu thiếu ngày/ga đi vẫn lưu yêu cầu, không mặc định clarify.

## 2. Chỉ cần một notebook

```text
Một notebook Colab
  ├─ tìm/clone repo và đọc cấu hình server
  ├─ uv tạo Python 3.12, cài Gradio + bộ CUDA đã ghim
  ├─ tải model một lần
  └─ chạy Web-ASR/app.py dưới dạng process thường trực
       ├─ Gradio phục vụ giao diện và xử lý sự kiện
       ├─ runtime VAD/ASR/LLM/TTS
       └─ link Gradio → người dùng mở bằng browser
```

Frontend là nội dung web Gradio gửi cho browser, không phải một notebook khác. Process Python 3.12 nằm cùng máy Colab với kernel; process riêng để không phụ thuộc kernel đang dùng Python 3.13. Hai notebook/hai máy không cần thiết cho MVP.

Link share chỉ là đường truy cập tạm thời, không phải một backend thứ hai hoặc nơi chứa model. Phiên Colab phải đang chạy. Không dùng keep-alive, né giới hạn hoặc coi gói trả phí là quyền hosting vô hạn; điều kiện sử dụng và độ ổn định của Colab vẫn áp dụng.

## 3. Giao diện

```text
┌──────────────────────────────────┬──────────────────────────┐
│ WEB-ASR                          │ Debug                    │
│                                  │ [Trạng thái] [Sự kiện]   │
│ [Ô nhập text — Enter để gửi]      │ [Yêu cầu hiện tại]       │
│                                  │                          │
│ [Ô phản hồi LLM, cập nhật dần]    │ đang nghe / ASR / LLM    │
│                                  │ đang phát / bị ngắt      │
│ [Bật mic] [Bật loa / Tắt loa]     │ generation, turn_id     │
│ [Audio bot / thông báo phát tiếng]│ action, lỗi, độ trễ      │
└──────────────────────────────────┴──────────────────────────┘
```

Dùng `gr.Blocks`, Row/Column tỷ lệ khoảng 3:2; cột phải có Tabs hoặc Accordion. Ô nhập text dành cho bản nháp người dùng gõ; không tự ghi transcript mic vào đó vì có thể đè phần đang nhập. Transcript mic tạm/cuối nằm ở debug bên phải. Chỉ Enter mới gửi text, cập nhật transcript không tự gửi lần nữa. Ô output chỉ hiển thị câu của generation hiện tại, không ghép lẫn câu đã bị hủy.

Một nút điều khiển mic bằng thuộc tính `recording` công khai của `gr.Audio`; không bấm giả vào selector nội bộ Gradio. Component thu âm vẫn được render trong DOM; có thể ẩn waveform để giữ UI gọn. Nút có nhãn và trạng thái rõ ràng, xử lý trường hợp người dùng từ chối quyền mic.

## 4. Luồng audio và hội thoại

```text
Mic browser → gr.Audio streaming → resample liên tục mono 16 kHz
→ FIFO frame 512 sample → Silero → chốt sau 1s im lặng
→ Gipformer CUDA → resolver LLM → cập nhật yêu cầu → LLM tạo câu trả lời
→ eSpeak tổng hợp WAV → gửi audio về browser → browser phát
```

Đường text bỏ qua VAD/ASR cho chính nội dung gõ, nhưng dùng chung ngữ cảnh với mic. Gửi text khi bot đang trả lời vẫn ngắt phản hồi cũ.

### 4.1. Gửi text trong lúc nói

- Nếu có audio đang thu hoặc đoạn hiện tại đã chốt nhưng ASR chưa xong, gắn text vào **cùng input_group_id** như phần bổ sung. Không xóa/reset buffer mic, không tắt mic, không tạo phản hồi text độc lập ngay.
- Giữ `spoken_text` (ASR cuối), `typed_additions` (danh sách text theo thứ tự Enter), nguồn và thời điểm riêng. ASR trả muộn điền đúng phần audio của group; không được đè hoặc làm mất phần gõ.
- Chờ ranh giới im lặng 1s hoặc thao tác tắt mic, rồi chờ transcript của đoạn đó. Gửi các phần cùng một lần xử lý resolver cho revision mới nhất; không trả lời sớm khi người dùng vẫn đang nói.
- Enter tăng input_revision/generation của group và vô hiệu hóa tác vụ phản hồi cũ, nhưng không vô hiệu hóa việc ASR hoàn thành phần audio đúng group. Khi submit sau ASR, dùng generation hiện tại của group, không dùng generation cũ lúc bắt đầu nói.
- Nếu Enter và segment_closed/ASR xảy ra đồng thời, loop phiên quyết định thứ tự: text thuộc group hiện tại nếu group đó chưa commit quyết định; group đã commit thì xử lý như text tiếp theo theo ngữ cảnh hiện có. Không gửi cùng một phần hai lần.
- Mic bật nhưng chưa có lời nói/group đang chờ không phải “đang nói”: text xử lý ngay như lượt text bình thường.
- Nếu ASR rỗng/lỗi, giữ text gõ; báo tình trạng audio. Không loại text chỉ vì audio không nhận ra lời.

Ví dụ: nói “Đặt vé tàu đi Đà Nẵng”, đồng thời gõ “ngày 30 tháng 10, 2 vé” → sau khi hết lượt, hiểu cả điểm đến, ngày và số vé; chỉ một phản hồi dựa trên dữ liệu kết hợp.

“Append” ở đây là **bổ sung nguồn text vào đầu vào của cùng lượt**, không phải nối chuỗi mù hoặc ép mọi quyết định LLM thành action append. Nếu chưa có yêu cầu thì action là new; nếu đã có thì có thể append. Chỉ dẫn sửa/thay/hủy rõ ràng trong nội dung vẫn được hiểu đúng theo quy tắc hội thoại đã thống nhất.

### 4.2. Điều phối các lượt

- Mic tiếp tục thu khi bot trả lời/phát tiếng.
- Khi xác nhận speech_started: tăng generation, gửi lệnh dừng tiếng ra browser và hủy tác vụ cũ. Không đợi LLM hiểu câu mới.
- Audio được đóng băng khi đủ im lặng, kể cả ASR còn chạy; giữ FIFO và một native ASR đang chạy.
- Transcript đến muộn giữ làm ngữ cảnh; không bật lại phản hồi cũ.
- Khi tắt mic, dừng browser capture, thu nốt đuôi audio và chốt phần lời đã có; không phải nút hủy toàn bộ hội thoại. Không tự lặp gửi nguyên bản ghi ở stop_recording sau khi đã gửi các chunk.
- Khi mất kết nối, dọn session và hủy phản hồi; mở lại không tự phát audio của phiên trước.

## 5. Phát và ngắt tiếng trong browser

Không gọi `speak_local()` để phát ra loa Colab. Tái sử dụng eSpeak chỉ để sinh WAV bằng subprocess có thể hủy. Bản đầu sinh audio theo cả câu sau khi LLM sinh xong; streaming TTS theo từng mẩu để một đợt riêng.

Gradio điều phối dữ liệu và controls. Dùng một player HTMLAudio nhỏ do ứng dụng sở hữu trong vùng Gradio, điều khiển bởi `browser_audio.js` qua các client event công khai; không phụ thuộc CSS class nội bộ Gradio hoặc tự mở thêm WebSocket server.

Giao thức phía ứng dụng:

- Command: `{command_id, generation, playback_id, audio_epoch, kind: play|stop, audio_url?}`.
- Ack: `{command_id, generation, playback_id, audio_epoch, status: started|ended|stopped|muted|blocked|error}`.
- Browser giữ generation và audio_epoch mới nhất. Bỏ command/ack cũ, không cho file trả muộn phát lại sau khi tắt/bật loa; dừng player cũ trước play mới.
- Stop phải `pause()`, reset thời gian, gỡ src và giải phóng tài nguyên audio đang chờ; vẫn giữ mic.
- `completed` chỉ khi browser ack ended đúng playback_id/generation. Hết thời lượng ước tính hoặc server gửi file xong không phải bằng chứng đã nghe xong.
- Browser bị chặn autoplay: hiện trạng thái blocked và cho thao tác phát thủ công; không giả completed.
- Callback Gradio poll/đồng bộ trạng thái tối đa 200ms/lần, đường riêng khỏi ASR/LLM. Không hứa p95 150ms vì chunk, poll và mạng có độ trễ cần đo.

Python adapter chờ ack ở tác vụ phản hồi, không chặn đường nhận mic. Có timeout ack để không giữ task vô hạn; kết thúc bằng error/interrupted chứ không ghi là đã phát xong. Chỉ file audio đã tạo cho phiên được phục vụ; không public thư mục repo hoặc `.env`.

### 5.1. Nút Bật loa/Tắt loa độc lập với microphone

- Mặc định bật tiếng bot trong phiên; browser có thể yêu cầu một thao tác người dùng để cho autoplay.
- Tắt loa dừng ngay player phía client, xóa audio đang chờ và thông báo server hủy TTS/audio của phản hồi hiện tại. Không hủy LLM đang sinh text, không đổi yêu cầu, không tắt mic hoặc mất history.
- Khi loa tắt, vẫn cập nhật text, ASR và resolver; không tạo/phát TTS cho các phản hồi hoàn tất trong thời gian đó.
- Bật loa áp dụng cho lần phát mới tiếp theo, không tự phát lại phần đã bỏ hoặc những phản hồi cũ khi loa tắt.
- Mỗi lần đổi trạng thái loa tăng audio_epoch riêng để chống command play trả muộn; không tăng generation hội thoại chỉ vì bật/tắt loa. Browser stop ngay ở thao tác click, không đợi poll server.
- Ack muted/stopped khác ended. Nếu đang phát dở rồi tắt loa, không đánh dấu đã nghe trọn. Tác vụ chờ audio phải được giải phóng có trạng thái muted/interrupted, không làm response bị treo.
- Câu trả lời text hoàn chỉnh vẫn được ghi vào lịch sử với trạng thái hiển thị text; metadata phát tiếng tách riêng (muted/interrupted/completed). Không coi hiện text đồng nghĩa người dùng đã nghe hoặc đã đọc. Resolver được biết cả câu đã hiển thị và trạng thái audio khi xét câu trả lời tiếp theo.
- Nút này chỉ điều khiển âm thanh bot của ứng dụng, không thay đổi volume hoặc loa hệ điều hành.

## 6. Tái sử dụng code, môi trường và phân phiên

- Tái sử dụng `Sub-ASR/voice.py`: Session, RequestState, Decision, LLM, AudioRunner và generation guards mới; không lấy bộ điều phối semantic endpoint cũ trong `sub_asr/app.py`.
- Tái sử dụng `Sub-ASR/sub_asr/asr.py`, `turns.py`, `setup.py` và pin model/source hiện có. Không chép lại các thuật toán thành bản fork trong Web-ASR.
- Bổ sung giao diện nhận chunk/nhập text, hook audio sink và delivery=browser. Giữ tương thích notebook Testing cũ bằng giá trị mặc định.
- `Web-ASR` có uv project riêng Python `>=3.12,<3.13`; copy cùng các pin CUDA đã xác minh trong Sub-ASR, ghi `uv.lock` mới.
- Gradio `6.30.0`, soxr `1.1.0`; soxr cung cấp resampler có trạng thái qua các chunk browser 44,1/48 kHz. Không dùng resample từng chunk mà bỏ phase/filter state.
- Model ASR tải một lần mỗi server, một worker native tại một thời điểm; không nạp lại khi bấm mic hoặc gửi text.
- Silero có state streaming: mỗi phiên mic có instance/state riêng; không chia sẻ một module có state mutable giữa người dùng.
- MVP: tối đa 2 phiên UI, tối đa 1 phiên mic đang hoạt động. Phiên còn lại nhận thông báo bận; không lẫn history/request/voice của người khác. Chưa cam kết tải nhiều người dùng.
- Một vòng lặp asyncio nền quản lý các Session; callback Gradio chỉ enqueue và đọc snapshot, không chờ ASR/LLM/TTS.
- Hàng đợi audio tối đa 3 giây; event debug giữ tối đa 200 dòng/phiên; audio đã chốt tối đa 8 đoạn như runtime hiện có. Quá tải báo lỗi/dừng nhận, không bỏ frame âm thầm.
- Session hết hạn sau 120 giây mất heartbeat; đóng chủ động thì dọn ngay. Không lưu audio/transcript dài hạn.

## 7. Cấu hình

Mặc định đọc `.env` gốc repo; `ASR_ENV_PATH` cho đường dẫn khác. `Web-ASR/.env.example` chỉ là mẫu, không tự sao chép secret. Key chỉ ở server, không trả vào component, event hoặc URL browser.

Giữ `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `TURN_MODEL`, `SEGMENT_SILENCE_SECONDS=1.0`, `RESOLVE_TIMEOUT_SECONDS=5.0`. Thêm:

```dotenv
WEB_SHARE=true
WEB_HOST=0.0.0.0
WEB_PORT=7860
WEB_USERNAME=
WEB_PASSWORD=
WEB_MAX_SESSIONS=2
WEB_MAX_ACTIVE_MIC=1
WEB_POLL_SECONDS=0.2
WEB_MIC_CHUNK_SECONDS=0.25
WEB_SPEAKER_ENABLED=true
```

Khi share ra ngoài, cấu hình cặp tài khoản/mật khẩu Gradio; không nhúng API key trong notebook. Ưu tiên tai nghe cho lần kiểm định đầu. AEC/loa ngoài không được mặc định coi là đã hỗ trợ.

## 8. Tiêu chí nghiệm thu

1. Một notebook trên Colab T4 khởi chạy được một app Gradio; không cần notebook frontend riêng.
2. Enter text → output LLM cập nhật. Khi nói + gõ đồng thời, chỉ gọi ASR cho audio, giữ đầy đủ cả hai nguồn và phản hồi dựa trên group kết hợp; không mất bản nháp đang gõ.
3. Một nút bật/tắt được mic của browser; dữ liệu 44,1/48 kHz được đổi thành frame 16 kHz liên tục, không mất đuôi hoặc lặp dữ liệu khi stop.
4. Bot phát tiếng ở máy người dùng; mic vẫn thu. Nói tiếp → browser dừng audio cũ. Nút loa độc lập: tắt dừng ngay tiếng nhưng vẫn có text/mic/LLM; bật lại không phát file/command cũ.
5. Ví dụ đặt vé Đà Nẵng rồi bổ sung ngày 30/10: giữ đúng yêu cầu; update/replace/cancel/clarify cũng qua các test có điều khiển.
6. Debug bên phải thể hiện trạng thái mic, đoạn nói, ASR, action, generation, resolver failure và browser ack, phân biệt lệnh dừng với tiếng thật đã dừng.
7. Hai tab không lẫn ngữ cảnh; không tạo hai native ASR cùng lúc; mất mạng/tắt mic/tab bị đóng không để tác vụ/audio chạy vô hạn.
8. Test offline/browser dùng fake xác minh kỹ thuật, không thay cho kiểm định GPU/LLM/microphone và ngắt tiếng thật trên Colab.

## 9. Tham khảo đã kiểm tra

- [Gradio realtime speech recognition](https://gradio.app/guides/real-time-speech-recognition)
- [Streaming inputs](https://gradio.app/guides/streaming-inputs), [Audio API](https://gradio.app/docs/gradio/audio)
- [Sharing your app](https://gradio.app/guides/sharing-your-app), [share link là proxy về server đang chạy](https://gradio.app/guides/understanding-gradio-share-links)
- [Gradio 6.30.0](https://pypi.org/project/gradio/6.30.0/), [soxr 1.1.0](https://pypi.org/project/soxr/1.1.0/)
- [Colab FAQ: giới hạn và hoạt động bị hạn chế](https://research.google.com/colaboratory/faq.html)

Đã kiểm tra source wheel Gradio 6.30.0: Audio có recording, streaming, visible=hidden và event stream/start_recording/stop_recording/play/pause/stop. Chưa chứng minh tích hợp capture + playback + ack hoạt động trên browser thật; đó là nhiệm vụ đầu tiên của kế hoạch.

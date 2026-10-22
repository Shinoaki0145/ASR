import json
import gradio as gr


def build_ui(controller, settings):
    sessions = {}

    def open_session(request: gr.Request):
        token = controller.dispatch(None, "open")
        sessions[request.session_hash] = token
        return token

    def resolve_sid(sid, request: gr.Request = None):
        if sid:
            return sid
        return sessions.get(getattr(request, "session_hash", None), "")

    def unload(request: gr.Request):
        token = sessions.pop(request.session_hash, None)
        if token:
            try:
                controller.dispatch(token, "close")
            except KeyError:
                pass

    def submit(sid, text, payload, request: gr.Request):
        sid = resolve_sid(sid, request)
        if not sid:
            raise gr.Error("Đang khởi tạo phiên, vui lòng đợi 1–2 giây rồi thử lại.")
        packet = json.loads(payload)
        packet["text"] = text
        ingest(sid, packet)
        return ""

    def mic(sid, payload, request: gr.Request):
        sid = resolve_sid(sid, request)
        if not sid:
            raise gr.Error("Đang khởi tạo phiên, vui lòng đợi 1–2 giây rồi thử lại.")
        packet = json.loads(payload)
        ingest(sid, packet)
        return gr.Button("Tắt mic" if packet["mic"] else "Bật mic")

    def ingest(sid, packet):
        if packet.get("error"):
            gr.Warning(packet["error"])
        controller.dispatch(sid, "packet", packet)

    def speaker(sid, enabled, request: gr.Request):
        sid = resolve_sid(sid, request)
        if not sid:
            return enabled, gr.Button("Tắt loa" if enabled else "Bật loa")
        enabled = not enabled
        controller.dispatch(sid, "speaker", enabled)
        return enabled, gr.Button("Tắt loa" if enabled else "Bật loa")

    def poll(sid, payload, request: gr.Request):
        sid = resolve_sid(sid, request)
        if not sid:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        if payload:
            packet = json.loads(payload)
            ingest(sid, packet)
        data = controller.dispatch(sid, "snapshot")
        events = data.pop("events")
        output = data.pop("output")
        status = {k: v for k, v in data.items() if k != "command"}
        bridge = {
            k: data[k]
            for k in (
                "generation",
                "audio_epoch",
                "speaker",
                "command",
                "capture_error",
            )
        }
        return (
            output,
            status,
            "\n".join(json.dumps(e, ensure_ascii=False) for e in events),
            bridge,
            gr.Button("Tắt mic" if data["mic"] else "Bật mic"),
        )

    with gr.Blocks(title="Web-ASR") as ui:
        sid = gr.State("")
        speaker_on = gr.State(True)
        packet = gr.Textbox(visible=False)
        gr.Markdown(
            "# Web-ASR\nNói hoặc nhập text. Khi đang nói, text gửi thêm được ghép cùng lượt. Dùng tai nghe để hạn chế vọng tiếng bot."
        )
        if settings.fake:
            gr.Markdown(
                "**DEMO: mic nhận diện theo mức âm lượng, transcript/LLM giả lập; chưa dùng GPU.**"
            )
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="Nhập text rồi Enter",
                    placeholder="Có thể gửi cả khi đang nói",
                )
                output = gr.Textbox(
                    label="Phản hồi của bot", interactive=False, lines=10
                )
                with gr.Row():
                    mic_btn = gr.Button("Bật mic")
                    speaker_btn = gr.Button("Tắt loa")
                gr.HTML(
                    '<div id="web-asr-audio-slot"></div><button type="button" id="web-asr-retry" style="display:none">Phát tiếng bị browser chặn</button>'
                )
                gr.Markdown(
                    "Nếu browser chặn tự phát tiếng, bấm “Phát tiếng bị browser chặn”. Text vẫn hoạt động khi tắt loa."
                )
            with gr.Column(scale=2):
                with gr.Tab("Trạng thái"):
                    status = gr.JSON(label="Session / request / nhóm input")
                with gr.Tab("Log"):
                    log = gr.Textbox(
                        label="200 sự kiện gần nhất", interactive=False, lines=23
                    )
        bridge = gr.JSON(visible=False)
        timer = gr.Timer(0.2)
        ui.load(open_session, outputs=sid)
        ui.unload(unload)
        text.submit(
            submit,
            [sid, text, packet],
            text,
            concurrency_limit=None,
            js="(sid,text)=>[sid,text,window.webASR.drain(true)]",
        )
        mic_btn.click(
            mic,
            [sid, packet],
            mic_btn,
            concurrency_limit=1,
            js="async(sid)=>{await window.webASR.toggleMic();return [sid,window.webASR.drain(true)]}",
        )
        speaker_btn.click(
            speaker,
            [sid, speaker_on],
            [speaker_on, speaker_btn],
            concurrency_limit=1,
            js="(sid,on)=>{window.webASR?.mute(on);return [sid,on]}",
        )
        timer.tick(
            poll,
            [sid, packet],
            [output, status, log, bridge, mic_btn],
            concurrency_limit=1,
            trigger_mode="multiple",
            js="(sid)=>[sid,window.webASR?.drain()||'']",
        ).then(None, bridge, None, js="s=>{window.webASR?.apply(s)}")
    return ui

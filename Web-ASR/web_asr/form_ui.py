import json
import gradio as gr
from .form import FIELDS


def build_ui(controller, settings):
    sessions = {}

    def open_session(request: gr.Request):
        sid = controller.dispatch(None, "open")
        sessions[request.session_hash] = sid
        return (sid, gr.Button(interactive=True), gr.Radio(interactive=True),
                gr.Button(interactive=True), gr.Button(interactive=True))

    def resolve(sid, request):
        sid = sid or sessions.get(request.session_hash)
        if not sid:
            raise gr.Error("Đợi khởi tạo phiên rồi thử lại. Tải lại trang nếu phiên đã hết hạn.")
        return sid

    def ingest(sid, payload):
        if payload:
            packet = json.loads(payload)
            if packet.get("error"):
                gr.Warning(packet["error"])
            controller.dispatch(sid, "packet", packet)

    def select(sid, field, payload, request: gr.Request):
        sid = resolve(sid, request)
        control(sid, payload, "select_field", field)
        return controller.dispatch(sid, "snapshot")["form"]["values"][field]

    def edit(sid, field, value, payload, request: gr.Request):
        sid = resolve(sid, request)
        try:
            control(sid, payload, "edit_field", field, value)
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        return "Đã lưu sửa tay."

    def mic(sid, payload, request: gr.Request):
        sid = resolve(sid, request)
        ingest(sid, payload)

    def export(sid, payload, request: gr.Request):
        sid = resolve(sid, request)
        try:
            return control(sid, payload, "export_form")
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc

    def control(sid, payload, action, *args):
        if payload:
            packet = json.loads(payload)
            if packet.get("error"):
                gr.Warning(packet["error"])
            return controller.dispatch(sid, "form_packet", packet, action, *args)
        return controller.dispatch(sid, action, *args)

    def poll(sid, payload, request: gr.Request):
        sid = resolve(sid, request)
        ingest(sid, payload)
        data = controller.dispatch(sid, "snapshot")
        form = data["form"]
        timings = [e["asr_ms"] for e in data["events"] if "asr_ms" in e]
        status = form["error"] or data["capture_error"] or (
            "Đang nhận dạng…" if data["asr_busy"] else
            "Đang nghe…" if data["mic"] else "Sẵn sàng."
        )
        if timings:
            status += f" | ASR gần nhất: {timings[-1]:.0f} ms (chưa gồm chờ khoảng nghỉ và truyền mạng)"
        bridge = {k: data[k] for k in (
            "generation", "audio_epoch", "speaker", "command", "capture_error"
        )}
        return (*form["values"].values(), form["transcript"], status, bridge,
                gr.Button("Tắt mic" if data["mic"] else "Bật mic"))

    def unload(request: gr.Request):
        sid = sessions.pop(request.session_hash, None)
        if sid:
            try:
                controller.dispatch(sid, "close")
            except KeyError:
                pass

    with gr.Blocks(title="ASR điền form") as ui:
        sid = gr.State("")
        packet = gr.Textbox(visible=False)
        bridge = gr.JSON(visible=False)
        export_payload = gr.Textbox(visible=False)
        gr.Markdown("# Điền form bằng giọng nói\nChọn trường → bật mic → nói → nghỉ để điền. Không dùng LLM.")
        if settings.fake:
            gr.Markdown("**DEMO: VAD theo âm lượng, transcript cố định ‘âm thanh demo’; chưa dùng model thật.**")
        with gr.Row():
            with gr.Column():
                field = gr.Radio([(label, key) for key, label in FIELDS.items()],
                                 value="full_name", label="Trường đang điền", interactive=False)
                mic_btn = gr.Button("Bật mic", variant="primary", interactive=False)
                transcript = gr.Textbox(label="Đoạn vừa nhận dạng", interactive=False)
                status = gr.Textbox(label="Trạng thái", interactive=False)
                gr.Markdown("Các đoạn nói nối vào trường đã chọn. Với điện thoại, đọc từng chữ số. Tắt mic và chờ nhận dạng xong trước khi xuất.")
            with gr.Column():
                values = [gr.Textbox(label=label, interactive=False) for label in FIELDS.values()]
                editor = gr.Textbox(label="Sửa giá trị trường đang chọn", placeholder="Nhập toàn bộ giá trị mới; để trống để xóa")
                save = gr.Button("Lưu sửa tay", interactive=False)
                download = gr.Button("Tải JSON", interactive=False)
        gr.Markdown("Sửa tay có hiệu lực khi bấm Lưu sửa tay. Dữ liệu chỉ ở phiên hiện tại; tải JSON trước khi đóng trang. Form mẫu kiểm tra định dạng số điện thoại Việt Nam.")
        timer = gr.Timer(0.2)
        ui.load(open_session, outputs=[sid, mic_btn, field, save, download])
        ui.unload(unload)
        field.change(select, [sid, field, packet], editor,
                     js="(sid,field)=>[sid,field,window.webASR?.drain(true)||'']")
        save.click(edit, [sid, field, editor, packet], status,
                   js="(sid,field,value)=>[sid,field,value,window.webASR?.drain(true)||'']")
        mic_btn.click(mic, [sid, packet], None,
                      js="async(sid)=>{await window.webASR.toggleMic();return [sid,window.webASR.drain(true)]}")
        download.click(export, [sid, packet], export_payload,
                       js="(sid)=>[sid,window.webASR?.drain(true)||'']").success(
            None, export_payload, None,
            js="text=>{if(!text)return;const url=URL.createObjectURL(new Blob([text],{type:'application/json;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='form.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}")
        timer.tick(poll, [sid, packet], [*values, transcript, status, bridge, mic_btn],
                   concurrency_limit=1, trigger_mode="once",
                   js="(sid)=>[sid,window.webASR?.drain()||'']").then(
                       None, bridge, None, js="s=>window.webASR?.apply(s)")
    return ui

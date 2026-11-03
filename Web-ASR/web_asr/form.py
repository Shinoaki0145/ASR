"""Selected-field dictation; no LLM, TTS or server persistence."""

import asyncio
import json
import re
import time
from .backend import Client, Controller, Session


FIELDS = {"full_name": "Họ tên", "phone": "Số điện thoại", "address": "Địa chỉ"}
DIGITS = dict(zip("không một hai ba bốn năm sáu bảy tám chín".split(), "0123456789"))
DIGITS.update({"mốt": "1", "tư": "4", "lăm": "5", "bẩy": "7"})


def normalize_value(field, text):
    if field not in FIELDS or not isinstance(text, str) or len(text) > 2000:
        raise ValueError("Trường hoặc nội dung không hợp lệ (tối đa 2000 ký tự)")
    text = " ".join(text.split())
    if field != "phone" or not text:
        return text
    tokens = re.sub(r"[.,;:()\-]", " ", text.lower()).split()
    result = "".join(DIGITS.get(token, token) for token in tokens)
    if not re.fullmatch(r"\+?[0-9]{1,15}", result):
        raise ValueError("Số điện thoại: đọc từng chữ số hoặc nhập số, tối đa 15 chữ số")
    return result


class FormSession(Session):
    recover_asr_error = True

    def __init__(self, owner):
        super().__init__(None, emit=owner.event)
        self.owner = owner
        self.groups = {}
        self.values = dict.fromkeys(FIELDS, "")
        self.revisions = dict.fromkeys(FIELDS, 0)
        self.selected = "full_name"
        self.targets = {}
        self.transcript = ""
        self.error = ""

    def speech_started(self):
        self.generation += 1
        self.targets[self.owner.runner.turn.turn_id] = (
            self.selected, self.revisions[self.selected]
        )
        self.emit("speech_started")

    def emit(self, event, **data):
        if event == "asr_failed":
            self.error = "ASR lỗi; dữ liệu cũ được giữ. Hãy nói lại hoặc nhập tay."
        super().emit(event, **data)

    def typed(self, text):
        self.submit(text)

    def submit(self, text, utterance_generation=None, turn_id=None):
        target = self.targets.pop(turn_id, None)
        field, revision = target or (self.selected, self.revisions[self.selected])
        self.transcript = text.strip()
        self.emit("transcript_final", text=self.transcript, turn_id=turn_id, field=field)
        if not self.transcript:
            return
        if revision != self.revisions[field]:
            self.error = f"Giữ bản sửa tay của {FIELDS[field]}; bỏ kết quả ASR cũ."
            return
        try:
            value = normalize_value(field, self.transcript)
            separator = "" if field == "phone" else " "
            joined = separator.join(x for x in (self.values[field], value) if x)
            self.values[field] = normalize_value(field, joined)
            self.error = ""
            self.emit("form_updated", field=field)
        except ValueError as exc:
            self.error = str(exc)

    def select(self, field):
        if field not in FIELDS:
            raise ValueError("Trường không hợp lệ")
        self.selected = field

    def edit(self, field, text):
        value = normalize_value(field, text)
        self.values[field] = value
        self.revisions[field] += 1
        self.error = ""

    def export(self):
        if self.owner.mic or self.owner.runner.busy or self.owner.audio_queued:
            raise ValueError("Tắt mic và chờ nhận dạng xong trước khi xuất")
        if not self.values["full_name"] or not self.values["address"]:
            raise ValueError("Điền họ tên và địa chỉ trước khi xuất")
        # shortcut: validate phone format only; add country-specific rules for a real form.
        if not re.fullmatch(r"(?:0[0-9]{9}|\+84[0-9]{9})", self.values["phone"]):
            raise ValueError("Số điện thoại cần 10 chữ số bắt đầu bằng 0 hoặc +84 và 9 chữ số")
        return json.dumps(self.values, ensure_ascii=False, indent=2)


class FormClient(Client):
    def create_session(self):
        self.llm = None
        self.speaker = False
        return FormSession(self)

    def packet_control(self, packet):
        control = packet.get("_form_control")
        if isinstance(control, tuple) and len(control) == 2 and callable(control[0]) and isinstance(control[1], asyncio.Future):
            self.queue.put_nowait(control)

    def snapshot_extra(self):
        session = self.session
        return {"form": {
            "values": dict(session.values),
            "selected": session.selected,
            "transcript": session.transcript,
            "error": session.error,
        }}


class FormController(Controller):
    client_type = FormClient

    def dispatch(self, sid, action, *args):
        if action == "form_packet":
            packet, command, *arguments = args
            async def prepare():
                client = self.clients[sid]
                if client.worker.done() or client.closed:
                    raise ValueError("Phiên đã dừng; tải lại trang")
                methods = {"select_field": client.session.select,
                           "edit_field": client.session.edit,
                           "export_form": client.session.export}
                callback = lambda: methods[command](*arguments)
                return callback, asyncio.get_running_loop().create_future()

            control = self.call(prepare())
            super().dispatch(sid, "packet", {**packet, "_form_control": control})

            async def wait():
                return await asyncio.wait_for(control[1], 5)

            return self.call(wait())
        if action not in {"select_field", "edit_field", "export_form"}:
            return super().dispatch(sid, action, *args)

        async def run():
            client = self.clients[sid]
            client.last_seen = time.monotonic()
            if client.closed or client.worker.done():
                raise ValueError("Phiên đã dừng; tải lại trang")
            methods = {"select_field": client.session.select,
                       "edit_field": client.session.edit,
                       "export_form": client.session.export}
            future = asyncio.get_running_loop().create_future()
            # Queue control changes behind already received audio to keep its original target.
            client.queue.put_nowait((lambda: methods[action](*args), future))
            return await asyncio.wait_for(future, 5)

        return self.call(run())

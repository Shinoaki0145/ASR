# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo>=0.20,<1"]
# ///
"""Upload audio on Molab/Marimo. GPU inference runs in the separate uv project."""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    from pathlib import Path
    import shutil
    import subprocess
    import tempfile

    return Path, json, mo, shutil, subprocess, tempfile


@app.cell
def _(mo):
    mo.md("""
    # Sub-ASR — Upload audio

    Mỗi file là **một lượt nói**, giữ ngữ cảnh qua các khoảng nghỉ.
    Không dùng micro hoặc DF3. VAD kiểm tra có lời nói; Gipformer greedy nhận dạng nguyên file.

    **Molab:** mở notebook mirror từ GitHub, bật GPU rồi bấm cài môi trường.
    Source được lấy tự động; bạn chỉ cần upload audio.
    **Local:** mở notebook từ thư mục `Sub-ASR`.
    File tối đa **25 MB / 60 giây**. Không dùng bản này để đo tốc độ chốt lượt realtime.
    """)
    return


@app.cell
def _(Path, subprocess):
    def resolve_project(here, cwd, update=False):
        # Molab mirrors the notebook; fetch sibling modules if they are not present.
        for candidate in (here, cwd, cwd/'Sub-ASR', here/'Sub-ASR'):
            if (candidate/'sub_asr/upload.py').is_file():
                return candidate
        checkout = cwd/'.sub_asr_source'
        repo = 'https://github.com/Shinoaki0145/ASR.git'
        branch = 'main'
        if not checkout.exists():
            subprocess.run(['git','clone','--depth','1','--branch',branch,repo,str(checkout)],check=True)
        elif update:
            subprocess.run(['git','-C',str(checkout),'fetch','--depth','1','origin',branch],check=True)
            subprocess.run(['git','-C',str(checkout),'checkout','--detach','FETCH_HEAD'],check=True)
        project = checkout/'Sub-ASR'
        if not (project/'sub_asr/upload.py').is_file():
            raise RuntimeError('Sub-ASR source is missing on GitHub. Push the Sub-ASR directory first.')
        return project
    return (resolve_project,)


@app.cell
def _(mo):
    prepare = mo.ui.run_button(label='1. Cài môi trường CUDA và tải model')
    prepare
    return (prepare,)


@app.cell
def _(Path, mo, prepare, resolve_project, shutil, subprocess):
    mo.stop(not prepare.value)
    _project_path = resolve_project(Path(__file__).resolve().parent,Path.cwd(),update=True)
    _uv = shutil.which('uv') or str(__import__('pathlib').Path.home()/'.local/bin/uv')
    with mo.status.spinner(title='Cài Python 3.12, thư viện CUDA và tải Gipformer…'):
        for _command in [
            ['nvidia-smi'],
            [_uv,'sync','--locked','--extra','ui','--extra','cuda'],
            [_uv,'run','--locked','--extra','ui','--extra','cuda','python','-m','sub_asr.setup'],
        ]:
            _process = subprocess.run(_command,cwd=_project_path,capture_output=True,text=True)
            if _process.returncode:
                mo.output.append(mo.plain_text((_process.stdout+'\n'+_process.stderr)[-16000:]))
                mo.stop(True,mo.md('**Chuẩn bị chưa thành công.** Xem log trên; kiểm tra GPU và các thư viện CUDA.'))
    mo.md('Đã chuẩn bị xong. Chọn file bên dưới và bấm **Nhận dạng**.')
    return


@app.cell
def _(mo):
    upload = mo.ui.file(filetypes=['.wav','.flac','.ogg','.mp3','.m4a','.webm'],multiple=False,
                        kind='area',max_size=25_000_000,label='Upload audio')
    ask_assistant = mo.ui.checkbox(value=False,label='Gửi transcript cho trợ lý (cần cấu hình LLM trong .env hoặc biến môi trường)')
    recognize = mo.ui.run_button(label='2. Nhận dạng')
    clear_history = mo.ui.run_button(label='Xóa lịch sử trợ lý')
    mo.vstack([upload,ask_assistant,recognize,clear_history])
    return ask_assistant, clear_history, recognize, upload


@app.cell
def _(mo, upload):
    mo.stop(not upload.value)
    mo.vstack([mo.plain_text(upload.name()),mo.audio(upload.contents())])
    return


@app.cell
def _(mo):
    get_history, set_history = mo.state([])
    return get_history, set_history


@app.cell
def _(clear_history, mo, set_history):
    mo.stop(not clear_history.value)
    set_history([])
    mo.md('Đã xóa lịch sử.')
    return


@app.cell
def _(
    ask_assistant,
    get_history,
    json,
    mo,
    Path,
    resolve_project,
    recognize,
    set_history,
    shutil,
    subprocess,
    tempfile,
    upload,
):
    mo.stop(not recognize.value)
    mo.stop(not upload.value,mo.md('Hãy chọn một file audio trước.'))
    _project_path = resolve_project(Path(__file__).resolve().parent,Path.cwd())
    _uv = shutil.which('uv') or str(__import__('pathlib').Path.home()/'.local/bin/uv')
    with mo.status.spinner(title='Đang chạy VAD và Gipformer…'), tempfile.TemporaryDirectory() as _temp:
        from pathlib import Path as _Path
        _input = _Path(_temp)/'input.audio'
        _output = _Path(_temp)/'result.json'
        _history = _Path(_temp)/'history.json'
        _input.write_bytes(upload.contents())
        _history.write_text(json.dumps(get_history(),ensure_ascii=False))
        _command = [_uv,'run','--locked','--extra','ui','--extra','cuda','python','-m','sub_asr.upload',
                    '--input-file',str(_input),'--output',str(_output),'--history',str(_history)]
        if ask_assistant.value:
            _command.append('--reply')
        _process = subprocess.run(_command,cwd=_project_path,capture_output=True,text=True)
        if _process.returncode:
            mo.output.append(mo.plain_text((_process.stdout+'\n'+_process.stderr)[-16000:]))
            mo.stop(True,mo.md('**Nhận dạng chưa thành công.** Kiểm tra model, GPU và codec. MP3/M4A/WebM có thể cần FFmpeg.'))
        result = json.loads(_output.read_text())
    if ask_assistant.value and not result.get('reply_error'):
        set_history(result['history'])
    mo.vstack([
        mo.md('### Transcript'),mo.plain_text(result['transcript'] or '[VAD không phát hiện lời nói]'),
        mo.md('### Trợ lý'),mo.plain_text(result.get('reply_error') or result['reply'] or '[Chưa yêu cầu phản hồi]'),
        mo.md(f"Audio: {result['audio_seconds']:.2f} giây · VAD: {result['vad_seconds']*1000:.0f} ms · ASR: {result['asr_seconds']*1000:.0f} ms"),
        mo.md('Thời gian ASR không gồm tải model/khởi động worker. Mỗi lần bấm tạo worker mới.'),
        mo.accordion({'Chi tiết VAD / model / lịch sử':mo.json(result)}),
        mo.download(json.dumps(result,ensure_ascii=False,indent=2).encode(),filename='sub_asr_result.json',label='Tải JSON kết quả'),
    ])
    return


if __name__ == "__main__":
    app.run()

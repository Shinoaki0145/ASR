"""Đóng gói source đã kiểm thử và audio mẫu vào notebook chạy độc lập."""
import base64
import io
import json
from pathlib import Path
from textwrap import dedent
import zipfile

ROOT=Path(__file__).resolve().parents[1]
buffer=io.BytesIO()
files={
    'voice.py':ROOT/'notebook_runtime/voice.py',
    'test_voice.py':ROOT/'notebook_runtime/test_voice.py',
    'pyproject.toml':ROOT/'Sub-ASR/pyproject.toml',
    'uv.lock':ROOT/'Sub-ASR/uv.lock',
    '.python-version':ROOT/'Sub-ASR/.python-version',
    'config.toml':ROOT/'Sub-ASR/config.toml',
    'ttshub-ghep-im-lang-3s.mp3':ROOT/'ttshub-ghep-im-lang-3s.mp3',
}
for name in ['__init__.py','turns.py','audio.py','asr.py','setup.py']:
    files['sub_asr/'+name]=ROOT/'Sub-ASR/sub_asr'/name
with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
    for name,path in files.items():
        info=zipfile.ZipInfo(name,date_time=(2026,10,21,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED
        z.writestr(info,path.read_bytes())
blob=base64.b64encode(buffer.getvalue()).decode()
cells=[]
def add(kind,name,source):
    c=dict(cell_type=kind,id=name,metadata={},source=dedent(source).strip().splitlines(keepends=True))
    if kind=='code': c.update(execution_count=None,outputs=[])
    cells.append(c)
add('markdown','intro','''
# Hội thoại tiếng Việt: ngắt lời, bổ sung, sửa, thay thế và hủy

Notebook mới triển khai luồng thử nghiệm theo kế hoạch đã trao đổi, không sửa notebook VAD cũ.

**Đường chạy thật:** MP3 → FFmpeg trong RAM → frame 16 kHz theo thời gian thực → Silero → bộ đệm lượt → chốt đoạn sau 1 giây im lặng → Gipformer CUDA → LLM cập nhật yêu cầu → câu trả lời → phát tiếng hoặc mô phỏng phát tiếng.
Microphone/file tiếp tục được xử lý trong lúc ASR, HTTP và phản hồi chạy nền. Khi có tiếng nói mới được xác nhận, tăng generation và hủy phản hồi cũ; kết quả trả muộn không được commit.

**Có sẵn:** source runtime, test và file `ttshub-ghep-im-lang-3s.mp3` đã được nhúng vào notebook; mở trên Colab không cần clone repo hay upload lại audio mẫu.

**Hai chế độ:**
- `demo`: fixture định sẵn cho đủ 6 action, không cần GPU hoặc LLM. Dùng để kiểm tra orchestration, không chứng minh mô hình hiểu ngữ cảnh.
- `real`: gọi LLM của bạn; với audio, Silero và Gipformer thật cần NVIDIA GPU. Có thể thử LLM bằng text trước mà không cần GPU.

Môi trường model được `uv` quản lý ở Python 3.12 riêng; kernel Colab Python 3.13 vẫn điều khiển được worker. Audio được giải mã trong RAM; chỉ source và file MP3 đầu vào nhúng được khôi phục ra thư mục làm việc.
''')
add('markdown','quickstart','''
## Cách chạy

1. Chạy cài công cụ, bung runtime và cấu hình; giữ `RUN_MODE="demo"` để thử ngay.
2. Chạy kiểm thử offline, rồi cell chạy hội thoại và bảng kết quả.
3. Muốn thử audio thật: bật **T4 GPU** trên Colab, đặt `RUN_MODE=real`, `REAL_INPUT=audio` và điền URL/model LLM trong `.env`, chạy cell chuẩn bị CUDA/model rồi chạy hội thoại.
4. URL LLM cần truy cập được từ máy chạy notebook; `127.0.0.1` trong Colab là máy Colab, không phải máy cá nhân. API theo giao thức chat-completions HTTP/SSE, không bắt buộc nhà cung cấp cụ thể.

**Phát tiếng:** mặc định `PLAYBACK="simulate"`: hiện câu trả lời và giữ tác vụ phát giả lập 4 giây để thử ngắt; không tạo âm thanh bot. `PLAYBACK="local_speaker"` dùng eSpeak và thiết bị âm thanh trên máy worker, chỉ phù hợp notebook chạy local có loa/tai nghe. Nó không phát về browser Colab. Chưa có microphone trực tiếp từ browser hoặc AEC; chế độ micro chỉ dùng thiết bị trên máy worker.

Đây là thử nghiệm theo file đã thu sẵn, không phải đo độ trễ micro thực. Không bảo đảm khoảng nghỉ 3 giây đủ để LLM/ASR phản hồi trước đoạn thứ hai; timeline cho biết ngắt ở giai đoạn resolve, generate hay playback.
''')
add('code','install','''
%pip install uv python-dotenv
''')
add('code','bootstrap',f'''
import asyncio
import base64
import getpass
import html
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import zipfile
from IPython.display import Audio, HTML, display
from dotenv import dotenv_values

BUNDLE_B64 = {blob!r}
WORKSPACE = Path.cwd() / '.turn_taking_notebook'
WORKSPACE.mkdir(parents=True, exist_ok=True)
(WORKSPACE / '.gitignore').write_text('*\\n', encoding='utf-8')
with zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE_B64))) as archive:
    for item in archive.infolist():
        target = (WORKSPACE / item.filename).resolve()
        if not target.is_relative_to(WORKSPACE.resolve()):
            raise ValueError('Đường dẫn bundle không hợp lệ')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(item))
UV = shutil.which('uv')
if not UV:
    raise RuntimeError('Chạy cell cài uv trước; restart kernel nếu cần.')
print('Runtime:', WORKSPACE.resolve())
print('Đã khôi phục source, test và audio mẫu. Không có API key trong bundle.')
''')
add('markdown','configuration-guide','''
## Cấu hình thử nghiệm

Đặt `.env` cạnh notebook (Colab: upload vào `/content/.env`). Điền `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` và đổi `RUN_MODE=real` trong file; chạy lại cell cấu hình và các cell sau. API key có thể để trống nếu server không yêu cầu. File được đọc lại mỗi lần chạy cell, không in API key và không nhúng file bí mật vào notebook. Nếu để file ở nơi khác, sửa `ENV_PATH` bên dưới.

`REAL_INPUT`: `audio`, `text` hoặc `microphone`. Chế độ text gửi từng câu theo thứ tự sau khi phản hồi trước hoàn tất; thử ngắt song song có trong demo và chế độ audio/microphone.

Silero notebook mới mặc định CUDA, có thể đổi `VAD_DEVICE="cpu"` để so sánh. Gipformer vẫn bắt buộc CUDA. Cấu hình CLI gốc trong repo không thay đổi.
Chốt mỗi đoạn audio sau `SEGMENT_SILENCE_SECONDS=1.0` giây im lặng, không gọi LLM kiểm tra kết thúc lượt. Xác nhận bắt đầu nói vẫn 0,16s, pre-roll 0,32s, tối đa 30s/đoạn. LLM chỉ quản lý ý nghĩa yêu cầu sau khi ASR có transcript. Các ngưỡng semantic cũ trong config.toml không điều khiển chốt đoạn ở runtime mới.
`RESOLVE_TIMEOUT_SECONDS=5.0` trong `.env` giới hạn bước cập nhật yêu cầu; JSON lỗi/timeout sẽ hỏi lại và giữ yêu cầu cũ.
''')
add('code','config','''
ENV_PATH = Path.cwd() / '.env'     # Colab thường là /content/.env
settings = dotenv_values(ENV_PATH) if ENV_PATH.is_file() else {}
def setting(name, default=''):
    # Đọc lại file mỗi lần chạy cell; giá trị trong file ưu tiên hơn biến môi trường.
    value = settings.get(name, os.getenv(name, default))
    return value if value is not None else default

RUN_MODE = setting('RUN_MODE', 'demo')
REAL_INPUT = setting('REAL_INPUT', 'audio')
VAD_DEVICE = setting('VAD_DEVICE', 'cuda')
PLAYBACK = setting('PLAYBACK', 'simulate')
SIMULATED_PLAYBACK_SECONDS = 4.0
RESOLVE_TIMEOUT = float(setting('RESOLVE_TIMEOUT_SECONDS', '5.0'))
SEGMENT_SILENCE_SECONDS = float(setting('SEGMENT_SILENCE_SECONDS', '1.0'))

AUDIO_PATH = WORKSPACE / 'ttshub-ghep-im-lang-3s.mp3'
# AUDIO_PATH = Path('/content/my_audio.mp3')
TEXT_INPUTS = ['Tôi muốn đặt vé đi Đà Nẵng', 'Vào ngày mai',
               'À không, đi Huế', 'Thôi, tìm khách sạn Hà Nội', 'Không cần nữa']

LLM_BASE_URL = setting('LLM_BASE_URL')
LLM_MODEL = setting('LLM_MODEL')
TURN_MODEL = setting('TURN_MODEL')          # Bỏ trống để dùng chung model
ASK_API_KEY = False                         # True nếu cần nhập bí mật bằng getpass
LLM_API_KEY = setting('LLM_API_KEY')
if RUN_MODE == 'real' and ASK_API_KEY:
    LLM_API_KEY = getpass.getpass('LLM API key (không hiển thị): ')

assert RUN_MODE in {'demo', 'real'}
assert REAL_INPUT in {'audio', 'text', 'microphone'}
assert PLAYBACK in {'simulate', 'local_speaker'}
assert VAD_DEVICE in {'cpu', 'cuda'}
assert SIMULATED_PLAYBACK_SECONDS >= 0 and RESOLVE_TIMEOUT > 0
assert 0 < SEGMENT_SILENCE_SECONDS < 30
if RUN_MODE == 'real' and (not LLM_BASE_URL or not LLM_MODEL):
    raise ValueError('Điền LLM_BASE_URL và LLM_MODEL để chạy thật.')
print('File cấu hình:', ENV_PATH, '| tồn tại:', ENV_PATH.is_file())
print('Chế độ:', RUN_MODE, '| input:', REAL_INPUT, '| playback:', PLAYBACK)
''')
add('code','listen-original','''
print('Audio đầu vào mẫu có thêm 3 giây im lặng ở giữa:')
if AUDIO_PATH.is_file():
    display(Audio(filename=str(AUDIO_PATH)))
else:
    print('Chưa tìm thấy audio. Sửa AUDIO_PATH nếu chạy real/audio.')
''')
add('markdown','tests-guide','''
## Kiểm thử offline

Không gọi LLM hoặc model thật. Các test kiểm tra schema, trạng thái cập nhật, clarification, cancellation không chặn, kết quả cũ trả muộn, single-flight ASR, hai lượt cách nhau khoảng nghỉ và nhánh kill TTS. Không đo tốc độ GPU hoặc độ chính xác LLM.
''')
add('code','offline-tests','''
subprocess.run([UV, 'run', '--locked', 'python', '-m', 'unittest', 'discover',
                '-s', '.', '-p', 'test_voice.py', '-v'], cwd=WORKSPACE, check=True)
''')
add('markdown','models-guide','''
## Chuẩn bị model thật (chỉ real/audio hoặc real/microphone)

Bước này tải bộ CUDA của dự án, source Gipformer/Icefall và checkpoint đã ghim revision. Có thể mất vài phút và cần dung lượng ổ đĩa. Không cần chạy cho demo hoặc real/text. Chưa cài TTS/PortAudio tự động: nếu dùng thiết bị local, cài `espeak-ng` và `libportaudio2` bằng trình quản lý gói của máy trước.
''')
add('code','prepare-models','''
if RUN_MODE == 'real' and REAL_INPUT in {'audio', 'microphone'}:
    subprocess.run(['nvidia-smi'], check=True)
    subprocess.run([UV, 'sync', '--locked', '--extra', 'cuda'], cwd=WORKSPACE, check=True)
    subprocess.run([UV, 'run', '--locked', '--extra', 'cuda', 'python', '-m', 'sub_asr.setup'],
                   cwd=WORKSPACE, check=True)
else:
    print('Bỏ qua tải CUDA/model trong chế độ hiện tại.')
''')
add('markdown','flow','''
## Chạy hội thoại và quan sát song song

- `speech_started`: người dùng bắt đầu nói; generation tăng, yêu cầu dừng phản hồi cũ.
- `transcript_partial` / `transcript_final`: ASR tạm / cuối.
- `segmentation_config`: chốt theo khoảng im lặng, không gọi LLM endpoint.
- `segment_closed`: đã đóng băng audio; hiện số giây im lặng, thời điểm trong file và trạng thái ASR.
- `response_deferred`: ASR của đoạn trước về muộn khi đang nghe đoạn mới; giữ transcript làm ngữ cảnh, không phát lại câu cũ.
- `request_updated`: `new`, `append`, `update`, `replace`, `cancel`, `clarify`.
- `resolve_failed`: lỗi kỹ thuật (HTTP, timeout, JSON/schema), khác với clarify hợp lệ; hiện reason và HTTP status, không in key.
- `playback_started/stopped`: tác vụ phát thật ở local hoặc phát giả lập, theo cấu hình.
- `stale_result_discarded` / `stale_asr_discarded`: đã loại kết quả cũ.

Một lượt audio mới không đồng nghĩa một yêu cầu mới. Chỉ resolver được cập nhật yêu cầu chuẩn hóa. Transcript gốc vẫn giữ nguyên. Thiếu ngày/giờ/ga đi vẫn phải lưu phần yêu cầu đã hiểu bằng `new` hoặc `append`; bot hỏi dữ liệu thiếu ở bước trả lời. `clarify` chỉ dành cho ý nghĩa hoặc quan hệ còn mơ hồ, không thay cho việc thu thập thông tin nghiệp vụ. Nếu xuất hiện `resolve_failed` trước clarify thì đó là nhánh lỗi dự phòng, không phải LLM hiểu câu nói là mơ hồ.

Mỗi lần chạy cell dưới tạo phiên mới, không giữ state từ lần chạy trước. Dừng cell sẽ gửi SIGINT để worker dọn tài nguyên, rồi kill nếu không dừng sau 5 giây. Event chỉ giữ trong RAM và output notebook; hãy xóa output trước khi chia sẻ nếu có nội dung riêng tư.
''')
add('code','worker','''
async def run_worker(arguments, needs_cuda=False):
    command = [UV, 'run', '--locked']
    if needs_cuda:
        command += ['--extra', 'cuda']
    command += ['--with', 'imageio-ffmpeg', 'python', '-u', 'voice.py', *arguments]
    environment = dict(os.environ)
    environment.update(LLM_BASE_URL=LLM_BASE_URL, LLM_MODEL=LLM_MODEL,
                       TURN_MODEL=TURN_MODEL, LLM_API_KEY=LLM_API_KEY)
    process = await asyncio.create_subprocess_exec(
        *command, cwd=WORKSPACE, env=environment,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    events, diagnostics = [], []
    async def read_diagnostics():
        while line := await process.stderr.readline():
            diagnostics.append(line.decode(errors='replace').rstrip())
            del diagnostics[:-40]
    stderr_task = asyncio.create_task(read_diagnostics())
    try:
        while line := await process.stdout.readline():
            text = line.decode(errors='replace').strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                print(text)
                continue
            events.append(event)
            if event['event'] == 'reply_chunk':
                print(event.get('text', ''), end='', flush=True)
            elif event['event'] == 'summary':
                print('\\nYêu cầu cuối:', event.get('request'))
            else:
                detail = event.get('action') or event.get('text') or event.get('message') or ''
                if event['event'] == 'segment_closed':
                    detail = (f"đoạn {event['turn_id']} | audio={event['audio_time']}s | "
                              f"im lặng={event['silence_seconds']}s | ASR đang chờ={event['asr_pending']}")
                elif event['event'] == 'segmentation_config':
                    detail = f"im lặng {event['silence_seconds']}s, không gọi LLM endpoint"
                elif event['event'] == 'request_updated':
                    detail = f"{event['action']} → {event.get('request') or event.get('question')}"
                elif event['event'] == 'resolve_failed':
                    detail = f"{event['reason']} | HTTP={event.get('http_status')}"
                elif event['event'] == 'response_deferred':
                    detail = f"giữ ngữ cảnh từ g{event['utterance_generation']}, không phát phản hồi cũ"
                print(f"\\n[{event.get('t', 0):6.3f}s | g{event.get('generation', '-')}] {event['event']}: {detail}")
        code = await process.wait()
        await stderr_task
        if code:
            print('\\n'.join(diagnostics))
            raise RuntimeError(f'Worker lỗi (exit {code}); xem sự kiện fatal_error và log trên.')
        return events
    finally:
        if process.returncode is None:
            try:
                process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), 5)
            except TimeoutError:
                process.kill()
                await process.wait()
        await stderr_task
''')
add('code','run','''
if RUN_MODE == 'demo':
    print('DEMO: ASR/LLM và thời gian phát tiếng đều giả lập; fixture có sẵn 6 action.')
    events = await run_worker(['--demo'])
else:
    arguments = ['--segment-silence', str(SEGMENT_SILENCE_SECONDS),
                 '--resolve-timeout', str(RESOLVE_TIMEOUT),
                 '--simulated-seconds', str(SIMULATED_PLAYBACK_SECONDS)]
    if PLAYBACK == 'local_speaker':
        if not shutil.which('espeak-ng'):
            raise RuntimeError('Chưa có espeak-ng trên máy worker. Chọn simulate hoặc cài eSpeak.')
        arguments += ['--speaker']
    if REAL_INPUT == 'audio':
        if not AUDIO_PATH.is_file():
            raise FileNotFoundError(AUDIO_PATH)
        arguments += ['--file', str(AUDIO_PATH.resolve()), '--vad-device', VAD_DEVICE]
    elif REAL_INPUT == 'microphone':
        arguments += ['--mic', '--vad-device', VAD_DEVICE]
    else:
        for text in TEXT_INPUTS:
            arguments += ['--text', text]
    print('REAL:', REAL_INPUT, '| playback:', PLAYBACK)
    events = await run_worker(arguments, needs_cuda=REAL_INPUT in {'audio', 'microphone'})
''')
add('code','results','''
updates = [e for e in events if e['event'] == 'request_updated']
rows = ''.join(
    '<tr>' + ''.join('<td>' + html.escape(str(e.get(k, ''))) + '</td>'
                    for k in ['t', 'generation', 'action', 'request', 'question']) + '</tr>'
    for e in updates)
display(HTML('<table><thead><tr><th>Giây</th><th>Generation</th><th>Action</th>'
             '<th>Yêu cầu sau cập nhật</th><th>Câu hỏi làm rõ</th></tr></thead><tbody>'
             + rows + '</tbody></table>'))
print('Số đoạn audio đã chốt:', sum(e['event'] == 'segment_closed' for e in events))
print('Số lần yêu cầu hủy:', sum(e['event'] == 'cancel_requested' for e in events))
print('Kết quả cũ bị loại:', sum(e['event'] in {'stale_result_discarded', 'stale_asr_discarded'} for e in events))
print('Số transcript cuối:', sum(e['event'] == 'transcript_final' for e in events))
if RUN_MODE == 'demo':
    assert [e['action'] for e in updates] == ['new', 'append', 'update', 'replace', 'cancel', 'clarify']
    assert any(e['event'] == 'cancel_requested' for e in events)
    print('PASS luồng fixture. Chưa phải bằng chứng ASR/LLM hoặc âm thanh thật.')
''')
add('markdown','interpretation','''
## Cách đọc kết quả và bước kiểm định tiếp theo

Với file ghép, xem `transcript_final` để biết Gipformer thực sự nghe được gì; không gán trước transcript hoặc ép kết quả thành append. Sau đó xem action và request mới. Sau khoảng 1 giây im lặng, audio được chốt dù ASR còn chạy. LLM không chắc ý nghĩa thì hỏi lại, không kéo dài ranh giới audio.

Khoảng nghỉ 3 giây sẽ tách hai đoạn nếu VAD xác định ít nhất 1 giây im lặng giữa chúng. Một câu vẫn có thể tách thêm nếu người nói nghỉ dài; resolver dùng ngữ cảnh để hiểu bổ sung/sửa/thay thế, không tự coi mọi đoạn là yêu cầu mới. ASR chậm không được nhập hai đoạn: audio đã đóng băng được xếp hàng theo thứ tự, tối đa 8 đoạn chờ; quá giới hạn sẽ báo lỗi rõ ràng. Nếu response chưa bắt đầu trước câu tiếp theo, hệ thống vẫn hủy tác vụ đang chờ; không được báo đã ngắt âm thanh.

**Bổ sung/sửa/thay thế:** thử real/text với các câu trong `TEXT_INPUTS`. **Ngắt lời:** xem demo hoặc real/audio; để kiểm tra micro/loa và độ trễ vật lý, chạy notebook local với tai nghe, `REAL_INPUT="microphone"`, `PLAYBACK="local_speaker"`. Dừng cell để kết thúc.

Không suy ra p95 ngắt tiếng ≤150ms từ log cancel. Cần đo âm thanh thật/loopback, tách riêng 160ms xác nhận VAD. Chưa hỗ trợ loa ngoài có AEC; không thực hiện đặt vé hoặc tác vụ bên ngoài.

**Phạm vi so với kế hoạch:** đã có cơ chế state, resolver, cancellation, generation, single-flight ASR và test offline trong runtime nhúng; CLI `Sub-ASR` gốc giữ nguyên. Kiểm định GPU/model LLM/microphone/TTS thực tế vẫn cần thực hiện trên máy đích. Notebook không tự đánh dấu nhiệm vụ kiểm định thật là hoàn thành.
''')
nb=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
    language_info=dict(name='python'),colab=dict(provenance=[],gpuType='T4'),accelerator='GPU'),nbformat=4,nbformat_minor=5)
p=ROOT/'Hoi_Thoai_Ngat_Loi_Test.ipynb'
p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n')
print(p.name,len(cells),'cells',p.stat().st_size,'bytes; bundle',len(buffer.getvalue()),'bytes')

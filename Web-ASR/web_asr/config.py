from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    fake: bool = False
    base_url: str = ""
    model: str = ""
    turn_model: str = ""
    api_key: str = field(default="", repr=False)
    share: bool = False
    username: str = ""
    password: str = field(default="", repr=False)
    host: str = "127.0.0.1"
    port: int = 7860
    silence: float = 1.0
    timeout: float = 5.0
    tts_voice: str = "vi-VN-HoaiMyNeural"
    tts_rate: str = "+0%"

    @classmethod
    def load(cls):
        values = {
            **dotenv_values(os.getenv("ASR_ENV_PATH", str(ROOT / ".env"))),
            **os.environ,
        }

        def get(k, d=""):
            return values.get(k, d) or d

        yes = lambda k, d="false": get(k, d).lower() in {"1", "true", "yes"}
        s = cls(
            fake=yes("WEB_FAKE"),
            base_url=get("LLM_BASE_URL"),
            model=get("LLM_MODEL"),
            turn_model=get("TURN_MODEL"),
            api_key=get("LLM_API_KEY"),
            share=yes("WEB_SHARE"),
            username=get("WEB_USERNAME"),
            password=get("WEB_PASSWORD"),
            host=get("WEB_HOST", "127.0.0.1"),
            port=int(get("WEB_PORT", "7860")),
            silence=float(get("SEGMENT_SILENCE_SECONDS", "1")),
            timeout=float(get("RESOLVE_TIMEOUT_SECONDS", "5")),
            tts_voice=get("TTS_VOICE", "vi-VN-HoaiMyNeural"),
            tts_rate=get("TTS_RATE", "+0%"),
        )
        if not s.fake and not (s.base_url and s.model and s.api_key):
            raise ValueError("Thiếu LLM_BASE_URL / LLM_MODEL / LLM_API_KEY")
        if not 0 < s.silence < 30 or not 0 < s.timeout < 120:
            raise ValueError("Silence/timeout không hợp lệ")
        return s

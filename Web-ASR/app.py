import atexit
from pathlib import Path
from web_asr.config import Settings, ROOT
from web_asr.backend import Controller
from web_asr.ui import build_ui


def main():
    settings = Settings.load()
    controller = Controller(settings)
    atexit.register(controller.close)
    ui = build_ui(controller, settings)
    ui.queue(max_size=64).launch(
        server_name=settings.host,
        server_port=settings.port,
        share=settings.share,
        auth=(settings.username, settings.password) if settings.username else None,
        show_error=True,
        allowed_paths=[],
        blocked_paths=[str(ROOT)],
        js=Path(__file__).parent.joinpath("web_asr/browser_audio.js").read_text(),
    )


if __name__ == "__main__":
    main()

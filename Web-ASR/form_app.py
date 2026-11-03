"""Run the form UI without configuring or instantiating an LLM."""
import atexit
from pathlib import Path
from web_asr.config import Settings, ROOT
from web_asr.form import FormController
from web_asr.form_ui import build_ui


def main():
    settings = Settings.load(mode="form")
    controller = FormController(settings)
    atexit.register(controller.close)
    ui = build_ui(controller, settings)
    ui.queue(max_size=64).launch(
        server_name=settings.host, server_port=settings.port, share=settings.share,
        auth=None,
        show_error=False, allowed_paths=[], blocked_paths=[str(ROOT)],
        js=Path(__file__).with_name("web_asr").joinpath("browser_audio.js").read_text(encoding="utf-8"),
    )


if __name__ == "__main__":
    main()

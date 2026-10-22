"""End-to-end Chromium fake microphone/speaker contract, no GPU/API needed."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import httpx
import numpy as np
import soundfile as sf
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
with tempfile.TemporaryDirectory() as tmp:
    wav = Path(tmp) / "mic.wav"
    sf.write(
        wav,
        np.concatenate(
            (
                np.zeros(32000),
                np.sin(np.arange(16000 * 30) * 2 * np.pi * 220 / 16000) * 0.3,
            )
        ),
        16000,
        subtype="PCM_16",
    )
    env = dict(
        os.environ,
        WEB_FAKE="true",
        WEB_PORT="7863",
        WEB_SHARE="false",
        WEB_HOST="127.0.0.1",
    )
    with (Path(tmp) / "server.log").open("w+") as log:
        server = subprocess.Popen(
            [sys.executable, "-u", "app.py"],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            for _ in range(100):
                try:
                    if httpx.get("http://127.0.0.1:7863/config").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            else:
                raise AssertionError("server did not start")
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    args=[
                        "--use-fake-device-for-media-stream",
                        "--use-fake-ui-for-media-stream",
                        f"--use-file-for-fake-audio-capture={wav}",
                    ]
                )
                page = browser.new_page(permissions=["microphone"])
                errors = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto("http://127.0.0.1:7863")
                page.wait_for_timeout(1000)
                text = page.get_by_role("textbox", name="Nhập text rồi Enter")

                def send(value):
                    text.fill(value)
                    text.press("Enter")

                send("Đặt vé Đà Nẵng")
                page.wait_for_function("window.webASR && !window.webASR.player.paused")
                page.get_by_role("button", name="Bật mic", exact=True).click()
                page.wait_for_function("window.webASR.stats().captured>8000")
                assert not page.evaluate("window.webASR.player.paused"), (
                    "capture and playback must overlap"
                )
                page.wait_for_function("window.webASR.player.paused", timeout=7000)
                assert page.evaluate("window.webASR.stats().mic"), (
                    "barge-in must keep recording"
                )
                second = browser.new_page(permissions=["microphone"])
                second.goto("http://127.0.0.1:7863")
                second.wait_for_timeout(800)
                second.get_by_role("button", name="Bật mic", exact=True).click()
                second.wait_for_function(
                    "document.body.innerText.includes('Một phiên khác đang dùng mic')",
                    timeout=5000,
                )
                second.wait_for_function("!window.webASR.stats().mic", timeout=5000)
                send("ngày 30/10, 2 vé")
                page.wait_for_timeout(500)
                page.get_by_role("button", name="Tắt loa", exact=True).click()
                assert page.evaluate("window.webASR.player.paused")
                page.get_by_role("button", name="Tắt mic", exact=True).click()
                page.wait_for_function(
                    "document.querySelector('textarea[disabled]')?.value.includes('2 vé')",
                    timeout=10000,
                )
                assert page.evaluate("window.webASR.player.paused"), (
                    "muted response cannot play"
                )
                page.get_by_role("button", name="Bật loa", exact=True).click()
                page.wait_for_timeout(500)
                assert page.evaluate("window.webASR.player.paused"), (
                    "unmute cannot replay old audio"
                )
                send("thêm hành lý")
                page.wait_for_function("!window.webASR.player.paused")
                page.wait_for_function("window.webASR.player.paused", timeout=7000)
                page.get_by_role("tab", name="Log", exact=True).click()
                page.wait_for_timeout(500)
                events = [
                    json.loads(line)
                    for line in page.get_by_role("textbox", name="200 sự kiện gần nhất")
                    .input_value()
                    .splitlines()
                ]
                assert any(e["event"] == "text_appended" for e in events)
                assert any(
                    e["event"] == "browser_ack" and e["status"] == "stopped"
                    for e in events
                )
                assert any(
                    e["event"] == "browser_ack" and e["status"] == "ended"
                    for e in events
                )
                assert not any(
                    e["event"] in {"audio_error", "audio_overflow", "response_error"}
                    for e in events
                )
                assert not errors, errors
                # Force the browser policy failure path, then retry under a user gesture.
                page.evaluate(
                    '()=>{window.savedPlay=window.webASR.player.play.bind(window.webASR.player);window.webASR.player.play=()=>Promise.reject(new DOMException("blocked","NotAllowedError"));}'
                )
                send("kiểm tra phát thủ công")
                page.get_by_role("button", name="Phát tiếng bị browser chặn").wait_for(
                    state="visible"
                )
                page.evaluate("()=>{window.webASR.player.play=window.savedPlay;}")
                page.get_by_role("button", name="Phát tiếng bị browser chặn").click()
                page.wait_for_function("!window.webASR.player.paused")
                page.wait_for_function("window.webASR.player.paused", timeout=7000)
                # Late play from an earlier audio epoch must be ignored.
                page.evaluate(
                    """() => {window.webASR.mute(true);window.webASR.apply({generation:100,audio_epoch:0,speaker:true,command:{command_id:999,generation:100,audio_epoch:0,kind:'play',audio_url:'bad'}})}"""
                )
                assert page.evaluate("window.webASR.player.paused")
                cache = ROOT / ".cache"
                cache.mkdir(exist_ok=True)
                page.screenshot(path=str(cache / "browser.png"), full_page=True)
                print(
                    "PASS: simultaneous capture/playback, barge-in, audio+text, mute/unmute, ended/stopped ack, stale epoch"
                )
                browser.close()
        finally:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
            log.seek(0)
            content = log.read()
            if "Traceback" in content:
                print(content[-5000:])
                raise AssertionError("server traceback")

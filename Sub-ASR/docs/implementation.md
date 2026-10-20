# Sub-ASR

Approved scope: standalone uv Python application, without DF3. Microphone/file → Silero VAD CPU → buffered Gipformer 1.5 greedy CUDA → semantic endpoint decision → conversational LLM → local interruptible TTS.

- [x] Test turn continuity, silence rejection, semantic waiting, stale results, and timeout.
- [x] Implement sample-clock turn buffering and asynchronous orchestration. Keep audio capture independent of network and ASR calls.
- [x] Reuse the tested Gipformer recipe in a self-contained adapter; pin upstream revisions and provide a setup command.
- [x] Add generic chat-completions HTTP client for semantic JSON decisions and streamed replies; local espeak-ng TTS, with headphones required (no AEC).
- [x] Create uv project, small default environment, optional Linux CUDA dependencies, configuration, and usage documentation.
- [x] Run offline tests, simulated conversation, package/config checks. Clearly distinguish tests from unperformed GPU/microphone checks.

No automatic recording to disk. Semantic requests have timeouts; on failure use a longer silence timeout, never claim semantic detection succeeded. Hard maximum turn duration is a resource bound and may split long speech. File mode is a paced simulation, not a benchmark of actual human turn completion.

Verification: `uv sync` created `.venv`; `uv lock --check` passed; 28 offline tests and the deterministic demo passed; Marimo notebook validation and initial execution passed; all 20 existing parent benchmark tests still passed. No live CUDA, Silero inference, microphone, TTS playback, or real LLM server was exercised on this host. PortAudio is absent. CUDA stack and pinned models are provided via explicit installation/setup commands rather than downloaded into the lightweight environment.

Molab workflow: GitHub mirror of `Sub-ASR/molab_app.py`; bootstrap missing sibling modules from the configured repository. No source archive upload is required. Upload mode treats each audio file as one complete turn and preserves pauses.

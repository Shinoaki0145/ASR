# ASR form MVP

Implement the accepted select-field → speak → fill approach without LLM or TTS.
Reuse the existing browser PCM transport, AudioIngress, Gipformer ModelPool,
Silero VAD and AudioRunner. Keep the conversation entry point for compatibility;
add form_app.py and a dedicated Colab notebook.

Use a sample contact form: full_name, phone, address. The selected field is
captured when speech starts; successive segments append within a field.
Manual edits invalidate older pending recognition for that field. Silence closes
an audio segment, never submits a form. Show transcript, errors and ASR timing.
Allow manual editing and JSON download after validation, with no server persistence.
Names/addresses are plain text; phone accepts digit-by-digit Vietnamese speech
or digits. Do not guess grouped numbers or interpret free-form commands.

Alternatives: free speech with LLM extraction adds inference latency and semantic
errors; deterministic per-field dictation is the smallest useful first version.
User authorized implementation after the per-field recommendation.
No new runtime dependencies. Real GPU/microphone validation is separate from fake tests.

## Deployment update — 2026-11-03
User requested no web login, including public share links. Always launch the form
with auth=None and never prompt for web credentials. The Colab notebook uses
REPO_URL/REPO_BRANCH to clone a GitHub repo with Web-ASR and Sub-ASR at its root,
reuses existing source without pulling or overwriting, and no longer uploads ZIPs.

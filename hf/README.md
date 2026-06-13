# Hugging Face Release Staging

Use this directory for metadata when the DCASE challenge permits public release.

Planned contents for each adapter repository:

- LoRA adapter files from the corresponding local checkpoint.
- `README.md` model card describing the DCASE Task 5 setup.
- Link back to the code repository and official DCASE dataset page.
- No challenge audio files or generated evaluation outputs.

Keep `local_upload_manifest.json` untracked; it may contain machine-local checkpoint paths.

# Lora Integration

Loras remain fully supported in the headless build, but selection is now configuration-driven rather than interactive. This document explains how to organise lora assets and how to enable them for CLI runs.

## Directory Layout
- `loras/` – text-to-video loras (Wan 2.x). Optional subfolders such as `loras/1.3B/` or `loras/14B/` help keep hardware-specific variants separate.  
- `loras_i2v/` – image-to-video loras.  
- `loras_ltxv/`, `loras_flux/`, `loras_qwen/` – model-family specific loras.  
- `.lset` / `.json` files placed alongside the loras act as presets describing multipliers and metadata.

Wan2GP automatically scans these directories on startup. Keep filenames deterministic: avoid spaces and prefer ASCII to ensure downstream tooling resolves paths correctly.

## Enabling Loras in the CLI
Without the legacy UI toggles, you activate loras by embedding their configuration in either:
1. A **finetune definition** (see `docs/FINETUNES.md`) that lists `activated_loras` and `loras_multipliers`.  
2. A **preset** (`*.lset`) referenced by a finetune or default model profile.  

Example snippet inside a finetune JSON:
```json
"settings": {
  "activated_loras": [
    "my_style.safetensors"
  ],
  "loras_multipliers": "1.0"
}
```

With the file saved under `finetunes/`, run:
```bash
python -m cli.generate --prompt "city lights reflected on wet streets" --model-type <finetune-id> --dry-run
```
The dry-run output lists the active loras; confirm before launching a full render.

## Building `.lset` Presets
`.lset` files are simple JSON or INI-style descriptors that capture:
- Lora filenames (relative to the lora directory)
- Default multipliers (including phased values if needed)
- Optional prompt snippets or comments

Copy an existing preset from the repository, edit the filenames, and save it alongside your loras. Reference the preset name from a finetune or default settings file using the `lora_preselected_preset` field.

## Tips
- Keep `loras_url_cache.json` up to date if you rely on auto-downloaded assets; update it manually when adding new sources.  
- Record any lora usage in `PROJECT_PLAN_LIVE.md` (prompt, multiplier, model type) so runs remain reproducible.  
- When experimenting, clone your finetune JSON rather than editing in-place—this makes it easy to revert or compare variants.

The CLI will gain dedicated flags for loras in a future milestone. Until then, treat the configuration files as the source of truth.

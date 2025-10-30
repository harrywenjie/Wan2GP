# Finetune Definitions

Finetunes extend the base Wan2GP model catalogue without mutating the core repository. Each finetune is described by a JSON file that merges additional metadata (download URLs, default prompts, lora lists, etc.) into an existing model profile.

## Directory Layout
- `defaults/` – canonical definitions shipped with Wan2GP. Treat these files as read-only.  
- `finetunes/` – user-supplied overrides and new definitions. Files in this directory take precedence over entries in `defaults/`.  
- `settings/` – preset configuration files referenced by both defaults and finetunes.

## Creating a Finetune
1. Copy the base definition that best matches your target model from `defaults/` to `finetunes/` and rename it (e.g. `t2v_fast.json`).  
2. Edit the new file:
   - Update the `model` block (`name`, `description`, `architecture`, `urls`).  
   - Adjust `settings` (resolution, frames, guidance, etc.) to reflect the desired defaults.  
   - Add or remove loras/modules as needed.  
3. Save the file and keep JSON syntax strict—invalid JSON will break loading across the CLI.

Example skeleton:
```json
{
  "model": {
    "name": "Wan 2.2 Fast",
    "architecture": "t2v",
    "description": "Accelerated variant tuned for quick previews.",
    "urls": [
      "https://example.com/models/wan_fast_fp16.safetensors"
    ],
    "base_model": "t2v"
  },
  "settings": {
    "prompt": "",
    "video_length": 49,
    "num_inference_steps": 18
  }
}
```

## Validating a Definition
- Run `python -m cli.generate --prompt "finetune validation" --model-type <id> --dry-run`.  
- Replace `<id>` with the filename (without `.json`) you created in `finetunes/`.  
- The CLI prints the merged configuration; if errors appear, check for missing URLs, typos in the architecture ID, or malformed JSON.

## Distributing Finetunes
Share only the JSON definition—avoid committing downloaded weights. Document the expected weight locations or provide download scripts separately. Recipients can drop the JSON into their own `finetunes/` directory and reuse the CLI entry point.

## Maintenance Tips
- Keep changelog notes for new or revised finetunes in `PROJECT_PLAN_LIVE.md`.  
- Use versioned filenames (e.g. `t2v_fast_v2.json`) when iterating on definitions to prevent cache confusion.  
- Ensure any additional assets referenced by the finetune (extra modules, loras) are available on disk before kicking off a generation.

# Model Catalogue

Wan2GP exposes multiple pipelines through the `--model-type` flag. This document summarises the key options, their hardware footprints, and typical use cases. Refer to the JSON files in `defaults/` for the authoritative configuration of each identifier.

## Text-to-Video (t2v)
| Model Type | Description | Min VRAM | Notes |
| --- | --- | --- | --- |
| `t2v` | Default Wan 2.2 text-to-video profile | 12 GB | Balanced quality and speed |
| `t2v_1_3b` | Lightweight 1.3B variant | 6 GB | Ideal for prototyping or limited VRAM |
| `t2v_5b` | Mid-sized model | 10 GB | Faster than 14B with good fidelity |
| `t2v_14b` | High-quality 14B pipeline | 16 GB+ | Best quality; expect longer runtimes |

## Image-to-Video (i2v)
| Model Type | Description | Min VRAM | Notes |
| --- | --- | --- | --- |
| `i2v_2_2` | Wan 2.2 image-driven pipeline | 12 GB | Accepts `--image-start` and optional `--image-end` |
| `i2v_1_3b` | Lightweight image animation | 6 GB | Lower quality but fast |
| `i2v_14b` | High fidelity i2v | 16 GB+ | Requires strong VRAM budget |

## Control / Editing Pipelines
| Model Type | Description | Min VRAM | Required Inputs |
| --- | --- | --- | --- |
| `vace_1_3b` | VACE ControlNet (1.3B) | 6 GB | Supply `--video-guide` or `--video-source`, masks optional |
| `vace_14b` | VACE ControlNet (14B) | 16 GB | Higher quality, slower |
| `flux` | Flux editing family | 12 GB | Behaviour depends on presets |
| `ltxv` | LTX long-form video | 12 GB | Tuned for extended durations |

## Audio-Driven / Specialty Models
Check `defaults/` for identifiers such as `chatterbox`, `qwen_edit`, or `mmaudio`. These typically require additional inputs (audio guides, masks). Ensure the needed assets exist before calling the CLI.

## Choosing a Model
- **Limited VRAM (≤ 8 GB):** prefer `t2v_1_3b`, `i2v_1_3b`, or `vace_1_3b`. Shorten `--frames` for longer clips.  
- **Mid-range VRAM (10–12 GB):** `t2v`, `t2v_5b`, `i2v_2_2`, `ltxv`.  
- **High-end VRAM (16 GB+):** `t2v_14b`, `i2v_14b`, `vace_14b`, stacked workflows with heavy loras.  
- **Editing / Motion Transfer:** Use VACE models with `--video-guide`, `--video-mask`, and `--image-ref` assets to steer the result.

## Discovering Available IDs
List the files in `defaults/` or `finetunes/`:
```bash
ls defaults/*.json
ls finetunes/*.json
```
Use the filename stem (without `.json`) as the `--model-type` value.

## Validation
Always dry-run new combinations:
```bash
python -m cli.generate --prompt "model catalogue check" --model-type t2v_1_3b --dry-run
```
Confirm the output lists the expected model filename before committing to a full render. Document each run (model ID, seed, frames, resolution) in `PROJECT_PLAN_LIVE.md` for reproducibility.

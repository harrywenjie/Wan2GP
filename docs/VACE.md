# VACE Workflow (CLI)

VACE provides Wan2GP’s motion-transfer and video-editing capabilities. The CLI exposes the same pipelines through the `--model-type` flag and a series of file-based inputs.

## Prerequisites
- Select a VACE model: `vace_1_3b` (fast, low VRAM) or `vace_14b` (higher quality).  
- Prepare assets on disk:
  - **Control / Guide Video** (`--video-guide`) – supplies motion, depth, or annotation cues.  
  - **Video Source** (`--video-source`) – original footage to continue or restyle.  
  - **Mask** (`--video-mask` or `--image-mask`) – optional area to constrain edits.  
  - **Reference Images** (`--image-ref`) – inject identities or objects. Repeat the flag for multiple references.  
  - **Prompts** – descriptive text via `--prompt` (and optionally `--negative-prompt`).

## Example Command
```bash
python -m cli.generate \
  --prompt "replace the dancer with a chrome android, neon club lighting" \
  --model-type vace_1_3b \
  --video-guide data/guides/dance_pose.mp4 \
  --video-source data/input/club_scene.mp4 \
  --video-mask data/masks/dancer_mask.mp4 \
  --image-ref data/refs/android_pose.png \
  --frames 96 \
  --output-dir outputs/vace_android
```

The CLI validates every file before execution. Use `--dry-run` with the same flags to confirm the configuration without rendering.

## Asset Preparation Tips
- **Masks:** Export alpha masks or binary videos using your favourite editor. Ensure dimensions and frame counts roughly match the guide/source footage; Wan2GP will resample but clean inputs yield better results.  
- **Reference Images:** Remove backgrounds if you want VACE to focus on the subject. Tools like `preprocessing/matanyone` can still generate masks offline.  
- **Prompt Structure:** Describe both the desired subject and the scene. Mention the guide video intent (e.g. “match choreography of the guide video”).

## Sliding Window Considerations
Long videos rely on sliding windows defined in the model defaults. When editing:
- Keep prompts consistent across windows unless you explicitly want changes.  
- Increase overlap (via settings JSON) if transitions show seams.  
- For very long clips, consider generating segments and stitching them in post.

## Logging & Reproducibility
- Record the exact command, guide assets, and masks used in `PROJECT_PLAN_LIVE.md`.  
- Capture VRAM usage and runtime so future runs can budget resources appropriately.  
- If results drift, verify that guide/reference assets have not been modified between runs.

VACE functionality will continue to migrate into dedicated CLI flags as the headless project matures. For complex workflows, maintain your own shell scripts around `cli.generate` to bundle prompts and asset paths.

# CLAUDE.md — Collaboration Guide for Claude & Antigravity

This repository contains the **ESPNER Colab T4 Luxury Commercial Video Studio** (`D:\廣告影片`). Both **Claude** and **Antigravity** collaborate on this project.

## 1. Core Architecture & Source of Truth
- **`build_colab_notebook.py`** is the **single source of truth** that generates the Colab notebooks:
  - `D:\廣告影片\ESPNER_Colab_T4_省記憶體版_v5.1.ipynb`
  - `D:\廣告影片\ESPNER_Colab_T4_廣告生成器.ipynb`
  - And automatically syncs them to Google Drive Desktop (`G:\我的雲端硬碟\Colab Notebooks\`) if mounted.
- Whenever you modify the Colab notebook logic, **edit `build_colab_notebook.py` first**, then run:
  ```bash
  python build_colab_notebook.py
  ```
  so all local and Google Drive `.ipynb` files stay 100% synchronized.
- The three subprocess scripts live as plain Python in module-level constants `STAGE_3A_CODE`, `STAGE_3B_CODE`, `STAGE_3C_CODE` inside `build_colab_notebook.py` (embedded into Step 3 via `embed_stage()`). Edit them there as normal code; never put `"""` inside them.
- `SHOTPLAN_CODE` (→ `/content/temp/shot_plan.py`) is the single source of truth for shots: `resolve_shots(cfg)` returns N shots (portrait prompt, motion prompt, wear, closeup, dur, copy index). `STYLE_PLANS` holds the reference-standard 30s styles (`A_STORY_30S`, `B_COASTAL_30S`, modeled on the two reference ads); legacy styles resolve to 2 x 8s. Stages 3A/3B/3C and Step 4 all iterate over it, so adding a style = adding a plan.
- Step 4 post modules are embedded constants written to `/content/temp` at run time: `TITLES_CODE` (copy/brand-mark layers, split & studio endcards), `GRADE_CODE` (film LUT + grain), `AUDIO_CODE` (procedural royalty-free music bed + loudnorm mux). Output frames are streamed to disk (never hold all 720 frames in RAM).
- Genuine-eyewear lock in motion: 3A saves `glasses_shot{N}.png` + `glasses_meta.json`; after each AI clip, 3C runs `TRACK_CODE` (`espner_track.recomposite_glasses`, LK head tracking + halo cleanup) to paste the real product back on every frame. LTX VAE encode/decode must stay float32 (`ltx_generate`); fp16 VAE on T4 produced negative/garbage frames. `fidelity_check` rejects clips whose frame 0 doesn't match the keyframe.
- Step 4 ends with `QC_CODE` (`espner_qc`): a plain zh-TW quality report (blur/ghosting/static/flicker per shot) saved next to the video on Drive.
- Stage 3C writes `/content/temp/stage3c_status.json` (`ai_ok`, `engine`, per-shot `motion`). Step 3 prints it so a silent fallback is never mistaken for AI motion. `motion < 3.0` = effectively static.

## 2. Critical Hardware Constraints (Standard Google Colab T4)
1. **System RAM Limit (`12.7 GB`)**:
   - Standard Colab only provides `12.7 GB` of CPU System RAM.
   - **Never** load `RealVisXL` and a video diffusion pipeline in the same Python process, and **never** load the 4.7B T5 text encoder (~9.5 GB) simultaneously with the video transformer (~9 GB) in the same process.
   - Keep the 3-stage subprocess architecture (`stage_3a.py` -> `stage_3b.py` -> `stage_3c.py`):
     - `stage_3a.py`: Pure OpenCV background removal (`0 MB` extra RAM; do NOT use `rembg`/`onnxruntime`) + `RealVisXL_V4.0_Lightning` base portrait + 100% genuine eyewear composite.
     - `stage_3b.py`: Standalone text encoder (`transformer=None, vae=None`) -> saves `/content/temp/prompt_embeds.pt` and exits so the OS reclaims 100% of T5 memory.
     - `stage_3c.py`: Standalone video generation (`text_encoder=None, tokenizer=None`) loading `/content/temp/prompt_embeds.pt`.
2. **Tesla T4 GPU Architecture (`sm_75`, Turing, `15 GB VRAM`)**:
   - **CRITICAL**: Tesla T4 is Compute Capability `7.5` (`sm_75`). It does **NOT** support `torch.bfloat16` in PyTorch's CUTLASS / FlashAttention (`scaled_dot_product_attention`) kernels — using `bfloat16` on T4 throws `RuntimeError: cutlassF: no kernel found to launch!`.
   - Always check `cc_major = torch.cuda.get_device_capability(0)[0]` and use **`torch.float16`** when `cc_major < 8` (`infer_dtype = torch.float16 if cc_major < 8 else torch.bfloat16`).
   - Always enable `pipe.enable_model_cpu_offload()` and `pipe.vae.enable_tiling()`.
3. **Subprocess Output Streaming in Colab (`ipykernel`)**:
   - In Colab notebooks, `subprocess.run(..., check=True)` hides child stdout/stderr from the notebook cell UI. Always use `run_stage()` with `subprocess.Popen(..., stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)` to stream logs live to the cell output.

## 3. ESPNER v5 Commercial Visual Standards
- **Resolution & Frame Rate**: `720x1280` (9:16 vertical), `24 fps`, `libx264`, `yuv420p`.
- **Product Identity Lock**:
  - Never let text-to-image hallucinate fake sunglasses. Generate an unobstructed portrait first, then composite the user's actual uploaded eyewear (`product_clean_rgba.png`), and animate from that keyframe.
- **v5 Endcard (`#FFFFFF` Pure White Studio)**:
  - Must use the original front product photo (`product_clean_rgba.png`) on a pure `#FFFFFF` studio background with a soft ground contact shadow, slow zoom (`1.000 -> 1.035`), and luxury typography (`E S P N E R`, model name, `把目光，還給世界。`, `POLARIZED LUXURY EYEWEAR`).
  - Maintain strict WCAG contrast (dark charcoal `#111111` / `#4A4A4A` / `#8C6D1F` text on `#FFFFFF` background; never white text on white background).

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generator script to build the Low-RAM (<= 5.5GB System RAM, <= 12GB T4 VRAM)
Google Colab Notebook (.ipynb) for ESPNER T4 GPU Commercial Video Generation (v5.1).

Key upgrades:
1. ZERO `rembg` / `onnxruntime` dependency -> Uses pure OpenCV FloodFill + Luminance Alpha Matting
   (0 MB extra RAM, 0.01s execution, never fails with `ModuleNotFoundError: No module named 'rembg'`).
2. Subprocess Isolation (`stage_3a.py`, `stage_3b.py`, `stage_3c.py`):
   Linux OS reclaims 100.0% of CPU System RAM & GPU VRAM after each stage exits.
3. Two-Stage Text Encoder Splitting (`text_encoder=None`):
   Computes prompt embeddings in Stage 3B and unloads the 9.5GB T5 encoder before Stage 3C loads the video model.
   Fits comfortably inside Standard Colab's 12.7GB System RAM without needing High-RAM!
"""

import json
import os
import shutil

PATHS_TO_WRITE = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ESPNER_Colab_T4_LowRAM_v5.2.ipynb"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ESPNER_Colab_T4_省記憶體版_v5.1.ipynb"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "ESPNER_Colab_T4_廣告生成器.ipynb"),
    r"D:\廣告影片\ESPNER_Colab_T4_LowRAM_v5.2.ipynb",
    r"D:\廣告影片\ESPNER_Colab_T4_省記憶體版_v5.1.ipynb",
    r"D:\廣告影片\ESPNER_Colab_T4_廣告生成器.ipynb",
    r"G:\我的雲端硬碟\Colab Notebooks\ESPNER_Colab_T4_LowRAM_v5.2.ipynb",
    r"G:\我的雲端硬碟\Colab Notebooks\ESPNER_Colab_T4_省記憶體版_v5.1.ipynb",
    r"G:\我的雲端硬碟\Colab Notebooks\ESPNER_Colab_T4_廣告生成器.ipynb",
    r"G:\我的雲端硬碟\Colab Notebooks\ESPNER_Colab_T4_廣告生成器 (1).ipynb",
]


# ------------------------------------------------------------------------------
# GPU guard: verifies a CUDA GPU is attached and PyTorch is a CUDA build.
# Fixes "AssertionError: Torch not compiled with CUDA enabled" (CPU-only torch).
# Runs the probe in a fresh subprocess so a stale in-kernel torch import never lies.
# ------------------------------------------------------------------------------
GPU_GUARD_CODE = r"""
def _probe_torch_cuda():
    # 在全新子程序中檢查，避免 notebook 核心內舊的 torch 匯入結果造成誤判
    code = "import torch,json;print(json.dumps([torch.__version__, torch.version.cuda, torch.cuda.is_available()]))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        return [None, None, False]

def _pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name).split("+")[0]
    except Exception:
        return None

def ensure_cuda_torch(required=True):
    gpu_ok = shutil.which("nvidia-smi") is not None and subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0
    if not gpu_ok:
        msg = ("❌ 目前的 Colab 執行階段『沒有掛載 GPU』（PyTorch 為 CPU 版，AI 模式會出現 Torch not compiled with CUDA enabled）。\n"
               "👉 請點選上方選單：執行階段 → 變更執行階段類型 → 硬體加速器選「T4 GPU」→ 儲存，然後從『步驟 1』重新執行。\n"
               "   若顯示 GPU 配額已用完，請稍後再試；或在步驟 2 改選「方案 2 (零變形極速)」，此模式不需要 GPU。")
        if required:
            raise RuntimeError(msg)
        print(msg.replace("❌", "⚠️"))
        return False
    ver, cuda_ver, avail = _probe_torch_cuda()
    if cuda_ver and avail:
        print(f"✅ GPU 檢查通過：torch {ver} (CUDA {cuda_ver})")
        return True
    print(f"⚠️ 偵測到 GPU 但 PyTorch 為 CPU 版 (torch {ver}, CUDA={cuda_ver})，正在自動重裝 CUDA 版 PyTorch (約 1~2 分鐘)...")
    idx = "https://download.pytorch.org/whl/cu126"
    pins = []
    for pkg in ["torch", "torchvision", "torchaudio"]:
        v = _pkg_version(pkg)
        if v:
            pins.append(f"{pkg}=={v}")
    if not pins:
        pins = ["torch", "torchvision"]
    base_cmd = [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--no-deps", "--index-url", idx]
    if subprocess.run(base_cmd + pins).returncode != 0:
        print("ℹ️ 原版本號無對應 CUDA wheel，改安裝 cu126 最新相容版本...")
        unpinned = [x.split("==")[0] for x in pins]
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--force-reinstall", "--index-url", idx] + unpinned, check=True)
    ver, cuda_ver, avail = _probe_torch_cuda()
    if not (cuda_ver and avail):
        raise RuntimeError(f"❌ CUDA 版 PyTorch 重裝後仍無法使用 GPU (torch {ver}, CUDA={cuda_ver})。請點『執行階段 → 中斷連線並刪除執行階段』，以 T4 GPU 重新從步驟 1 開始。")
    print(f"✅ 已修復：torch {ver} (CUDA {cuda_ver})")
    return True
"""
GPU_GUARD_SRC = [ln + "\n" for ln in GPU_GUARD_CODE.strip("\n").split("\n")]


def embed_stage(var_name, code):
    """Embed a stage script as `var_name = r'''...'''` in notebook cell source lines."""
    lines = [var_name + " = r'''" + "\n"]
    lines += [ln + "\n" for ln in code.split("\n")[1:-1]]
    lines.append("'''\n")
    return lines

STAGE_3A_CODE = r"""
import os, re, json, gc, urllib.request, urllib.parse, cv2, torch
import numpy as np
from PIL import Image, ImageFilter

with open("/content/temp/job_config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

photo_path = cfg["photo_path"]
style_key = cfg["commercial_style"].split(" ")[0]
is_snow = "SNOW_GOGGLES" in cfg["product_category"] or style_key == "4_ALPINE_EXTREME"
gen_mode = cfg["generation_mode"]
custom_p = cfg.get("custom_scene_prompt", "").strip()

# 若使用者輸入中文場景描述，自動透過免金鑰介面翻譯為英文導演提示詞
custom_p_en = custom_p
if custom_p and any('\u4e00' <= ch <= '\u9fff' for ch in custom_p):
    try:
        url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=zh-TW&tl=en&dt=t&q=" + urllib.parse.quote(custom_p)
        res = json.loads(urllib.request.urlopen(url, timeout=6).read().decode("utf-8"))
        custom_p_en = "".join(x[0] for x in res[0] if x and x[0])
        print(f"[3A] 🌐 已自動將中文場景翻譯為英文導演指令：{custom_p_en}")
    except Exception as e:
        print(f"[3A] ℹ️ 翻譯略過 ({e})，使用預設英文場景補強。")
        custom_p_en = "A charismatic movie star celebrity in a luxury convertible car surrounded by paparazzi camera flashes, stepping out in front of a grand modern building"

cfg["custom_scene_prompt_en"] = custom_p_en
with open("/content/temp/job_config.json", "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

print("[3A] 正在使用 OpenCV 光學演算法精準去背產品正面圖 (0MB 額外記憶體)...")
raw_pil = Image.open(photo_path).convert("RGBA")
raw_np = np.array(raw_pil)
rgb = raw_np[:, :, :3]
h_img, w_img = rgb.shape[:2]

# 檢查是否原本即為透明底 PNG
if raw_np[:, :, 3].min() < 240:
    alpha = raw_np[:, :, 3].astype(np.float32) / 255.0
else:
    # 從四角進行 FloodFill 找出外部純白/淺色背景，並保留鏡片內部反光
    # FIXED_RANGE：與種子色比較 (非相鄰像素)，避免沿漸層滲入鏡片導致整副眼鏡變半透明
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    bg_mask = np.zeros((h_img + 2, w_img + 2), np.uint8)
    work_img = rgb.copy()
    tol = (18, 18, 18)
    for seed_pt in [(0, 0), (w_img - 1, 0), (0, h_img - 1), (w_img - 1, h_img - 1), (w_img // 2, 0), (w_img // 2, h_img - 1)]:
        cv2.floodFill(work_img, bg_mask, seed_pt, (255, 255, 255), tol, tol, flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE)
    ext_bg = bg_mask[1:-1, 1:-1] > 0
    luma_alpha = np.clip((250.0 - gray.astype(np.float32)) / 16.0, 0.0, 1.0)
    alpha = np.where(ext_bg, luma_alpha * 0.15, np.maximum(luma_alpha, 0.88))
    alpha[ext_bg & (gray > 242)] = 0.0
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (3, 3), 0)

rgba_out = raw_np.copy()
rgba_out[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
ys, xs = np.where(rgba_out[:, :, 3] > 25)
if len(ys) > 0:
    pad = 6
    y1, y2 = max(0, ys.min() - pad), min(h_img, ys.max() + pad)
    x1, x2 = max(0, xs.min() - pad), min(w_img, xs.max() + pad)
    product_rgba = Image.fromarray(rgba_out[y1:y2, x1:x2])
else:
    product_rgba = Image.fromarray(rgba_out)

product_rgba.save("/content/temp/product_clean_rgba.png")
print(f"  -> 產品去背完成，尺寸：{product_rgba.size}")

if "方案 2" in gen_mode:
    print("[3A] 方案 2 無須生成真人肖像，跳過 SDXL。")
    raise SystemExit(0)

PORTRAIT_PROMPTS = {
    "0_CLASSIC_MADRID": (
        "Editorial medium portrait of a stylish European fashion model in linen shirt standing in a sunlit Mediterranean coastal stone colonnade, facing camera directly, head level, golden hour sunlight, ocean background, 85mm lens, photorealistic, 8k",
        "Editorial close-up portrait of a stylish European fashion model facing camera directly by the sunlit Mediterranean sea, warm golden hour lighting on face, wind in hair, shallow depth of field, 85mm lens, photorealistic, 8k"
    ),
    "1_CYBERPUNK_NEON": (
        "Editorial medium portrait of a futuristic model in sleek techwear jacket standing in neon-lit Tokyo street at night, facing camera directly, cyan and magenta rim lighting on face, bokeh neon lights, photorealistic, 8k",
        "Editorial close-up portrait of a futuristic model facing camera directly in neon city night, dramatic cyan and pink neon lighting on skin, sharp focus, photorealistic, 8k"
    ),
    "2_VINTAGE_1985": (
        "1985 Kodachrome vintage film medium portrait of a charismatic model in classic leather jacket leaning by a vintage convertible on Pacific Coast Highway at golden sunset, facing camera directly, warm amber glow, photorealistic",
        "1985 Kodachrome close-up portrait of a model facing camera directly on coastal highway at golden sunset, warm golden sunlight on face, nostalgic 35mm film aesthetic, photorealistic"
    ),
    "3_HAUTE_COUTURE_NOIR": (
        "High-fashion Parisian luxury editorial medium portrait of an elegant female model in tailored black coat standing in front of Haussmann architecture in Paris, facing camera directly, dramatic studio lighting, photorealistic, 8k",
        "High-fashion close-up portrait of an elegant model facing camera directly, dramatic chiaroscuro rim lighting, luxury magazine cover aesthetic, sharp facial structure, photorealistic, 8k"
    ),
    "4_ALPINE_EXTREME": (
        "Editorial medium portrait of a professional alpine skier in technical ski jacket standing on snowy Matterhorn glacier ridge in bright sunlight, facing camera directly, crisp snow peaks background, photorealistic, 8k",
        "Editorial close-up portrait of an alpine skier facing camera directly on snowy mountain summit, bright glacier sunlight, snow powder in air, sharp focus, photorealistic, 8k"
    ),
    "5_RIVIERA_SUNSET": (
        "Mediterranean Riviera luxury resort medium portrait of a sophisticated model in white linen shirt on a teak yacht deck at sunset, facing camera directly, sparkling golden sea background, photorealistic, 8k",
        "Mediterranean Riviera close-up portrait of a model facing camera directly at golden sunset, warm sunset glow on skin, luxury coastal terrace background, shallow depth of field, photorealistic, 8k"
    )
}

p_port1, p_port2 = PORTRAIT_PROMPTS.get(style_key, PORTRAIT_PROMPTS["0_CLASSIC_MADRID"])
if custom_p_en:
    # 產生基底人臉時先移除墨鏡字眼，以確保我們貼上的是 100% 真實上傳的產品圖
    clean_scene = re.sub(r"(?i)\b(wearing|with|took off|take off)?\s*(his|her|the)?\s*(aviator|sunglasses|glasses|eyewear|goggles|shades)\b", "", custom_p_en)
    p_port1 = f"Editorial medium portrait of a charismatic celebrity model ({clean_scene}), facing camera directly, head level, clear unobstructed face and eyes, cinematic lighting, 85mm lens, photorealistic, 8k"
    p_port2 = f"Editorial close-up portrait of a charismatic celebrity model ({clean_scene}), facing camera directly, head level, clear unobstructed face and eyes, dramatic flash and golden lighting, 85mm lens, photorealistic, 8k"

print("[3A] 正在於 T4 GPU 載入 RealVisXL V4.0 Lightning 生成真人模特兒...")
from diffusers import AutoPipelineForText2Image, EulerDiscreteScheduler
txt2img = AutoPipelineForText2Image.from_pretrained(
    "SG161222/RealVisXL_V4.0_Lightning",
    torch_dtype=torch.float16,
    variant="fp16",
    low_cpu_mem_usage=True
).to("cuda")
txt2img.scheduler = EulerDiscreteScheduler.from_config(txt2img.scheduler.config, timestep_spacing="trailing")
txt2img.vae.enable_tiling()
txt2img.vae.enable_slicing()

neg_p = "sunglasses, glasses, eyewear, goggles, deformed face, asymmetrical eyes, bad anatomy, blurry, watermark, text, logo"
b1 = txt2img(prompt=p_port1, negative_prompt=neg_p, width=704, height=1280, num_inference_steps=6, guidance_scale=1.8, generator=torch.Generator("cuda").manual_seed(1024)).images[0]
b2 = txt2img(prompt=p_port2, negative_prompt=neg_p, width=704, height=1280, num_inference_steps=6, guidance_scale=1.8, generator=torch.Generator("cuda").manual_seed(2048)).images[0]

# 依場景光源重新打光真實眼鏡（色溫、曝光、鏡面環境反射、邊緣柔化、顆粒），外型 100% 不變
def harmonize_eyewear(portrait_pil, glasses_rgba, paste_x, paste_y):
    # Relight the genuine eyewear cutout to the scene: color temperature, exposure,
    # environment reflection on lenses, edge softening and matching grain.
    # Shape/geometry is untouched (product identity lock).
    W, H = portrait_pil.size
    g = np.array(glasses_rgba).astype(np.float32)
    rgb, a = g[..., :3], g[..., 3:4] / 255.0
    gh, gw = rgb.shape[:2]
    scene = np.array(portrait_pil.convert("RGB")).astype(np.float32)
    # 1) local scene light: a padded window around the glasses
    pad_x, pad_y = int(gw * 0.6), int(gh * 1.6)
    x0, y0 = max(0, paste_x - pad_x), max(0, paste_y - pad_y)
    x1, y1 = min(W, paste_x + gw + pad_x), min(H, paste_y + gh + pad_y)
    local = scene[y0:y1, x0:x1].reshape(-1, 3)
    glob = scene.reshape(-1, 3)
    light = 0.6 * local.mean(0) + 0.4 * glob.mean(0)
    tint = 1.0 + 0.55 * (np.clip(light / max(1e-3, light.mean()), 0.78, 1.25) - 1.0)
    # 2) exposure: product shot is studio-white; pull toward scene luminance
    w = a[..., 0] > 0.5
    prod_l = float(rgb[w].mean()) if w.any() else 128.0
    scene_l = float(local.mean())
    expo = float(np.clip((scene_l / max(prod_l, 1.0)) ** 0.35, 0.68, 1.0))
    out = rgb * expo * tint
    # 3) environment reflection: mirrored, blurred scene in the bright/neutral lens areas
    env = scene[y0:y1, x0:x1][::-1, ::-1]
    env = cv2.resize(env, (gw, gh), interpolation=cv2.INTER_AREA)
    env = cv2.GaussianBlur(env, (0, 0), max(2.0, gw / 60.0))
    hsv = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    mirror = np.clip((hsv[..., 2] - 120.0) / 120.0, 0, 1) * np.clip(1.0 - hsv[..., 1] / 90.0, 0, 1)
    refl = (mirror * 0.5)[..., None]
    screen = 255.0 - (255.0 - out) * (255.0 - env) / 255.0
    out = out * (1 - refl) + screen * refl
    # 4) top-down falloff (light usually comes from above) + specular sheen
    yy = np.linspace(1.06, 0.88, gh, dtype=np.float32)[:, None, None]
    out = out * yy
    # specular sheen: soft diagonal highlight band across mirror/metal areas
    xg, yg = np.meshgrid(np.linspace(0, 1, gw), np.linspace(0, 1, gh))
    band = np.exp(-((xg * 0.8 + yg * 0.6 - 0.45) ** 2) / (2 * 0.09 ** 2)).astype(np.float32)
    out = out + (band * mirror * 70.0)[..., None]
    # 5) grain matched to the scene's noise level
    noise_sigma = float(np.clip(cv2.Laplacian(cv2.cvtColor(scene.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.CV_32F).std() / 18.0, 1.0, 4.0))
    out = out + np.random.RandomState(7).normal(0, noise_sigma, out.shape).astype(np.float32)
    # 6) soften the matte edge by ~1px so it does not look cut out
    a2 = cv2.GaussianBlur(a[..., 0], (3, 3), 0.8)[..., None]
    res = np.concatenate([np.clip(out, 0, 255), a2 * 255.0], axis=2).astype(np.uint8)
    return Image.fromarray(res, "RGBA")

FACE_BOXES = {}

def composite_eyewear_on_model(portrait_pil, prod_rgba, is_closeup=False, shot_key="1"):
    w, h = portrait_pil.size
    img_bgr = cv2.cvtColor(np.array(portrait_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(140, 140))
    angle_deg = 0.0
    if len(faces) > 0:
        fx, fy, fw, fh = sorted(faces, key=lambda b: b[2]*b[3], reverse=True)[0]
        FACE_BOXES[shot_key] = [int(fx) / w, int(fy) / h, int(fw) / w, int(fh) / h]  # 未戴眼鏡前偵測，供 3C 視差分層使用
        face_cx = fx + fw // 2
        eye_y = fy + int(fh * 0.39)
        target_w = int(fw * (1.06 if is_snow else 0.96))
        roi_gray = gray[fy:fy + int(fh * 0.6), fx:fx + fw]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            e1x, e1y = fx + eyes[0][0] + eyes[0][2]//2, fy + eyes[0][1] + eyes[0][3]//2
            e2x, e2y = fx + eyes[1][0] + eyes[1][2]//2, fy + eyes[1][1] + eyes[1][3]//2
            if abs(e2x - e1x) > fw * 0.22:
                face_cx = (e1x + e2x) // 2
                eye_y = (e1y + e2y) // 2
                angle_deg = float(np.clip(np.degrees(np.arctan2(e2y - e1y, e2x - e1x)), -12.0, 12.0))
    else:
        face_cx = w // 2
        eye_y = int(h * (0.42 if is_closeup else 0.36))
        target_w = int(w * (0.56 if is_closeup else 0.42))
    target_h = max(1, int(prod_rgba.height * (target_w / prod_rgba.width)))
    glasses_resized = prod_rgba.resize((target_w, target_h), Image.Resampling.LANCZOS)
    if abs(angle_deg) > 0.5:
        glasses_resized = glasses_resized.rotate(-angle_deg, resample=Image.Resampling.BICUBIC, expand=True)
    gw, gh = glasses_resized.size
    paste_x = face_cx - gw // 2
    paste_y = eye_y - int(gh * 0.48)
    glasses_resized = harmonize_eyewear(portrait_pil, glasses_resized, paste_x, paste_y)
    g_np = np.array(glasses_resized)
    g_alpha = g_np[:, :, 3].astype(np.float32) / 255.0
    shadow_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sh_mask = (g_alpha * 135).astype(np.uint8)
    sh_pil = Image.fromarray(sh_mask, mode="L").filter(ImageFilter.GaussianBlur(radius=8))
    sh_rgba = Image.new("RGBA", (gw, gh), (15, 8, 5, 0))
    sh_rgba.putalpha(sh_pil)
    shadow_layer.paste(sh_rgba, (paste_x, paste_y + int(gh * 0.08)), sh_rgba)
    out_pil = Image.alpha_composite(portrait_pil.convert("RGBA"), shadow_layer)
    out_pil.paste(glasses_resized, (paste_x, paste_y), glasses_resized)
    return out_pil.convert("RGB")

kf1 = composite_eyewear_on_model(b1, product_rgba, is_closeup=False, shot_key="1")
kf2 = composite_eyewear_on_model(b2, product_rgba, is_closeup=True, shot_key="2")
with open("/content/temp/face_boxes.json", "w", encoding="utf-8") as f:
    json.dump(FACE_BOXES, f)
kf1.save("/content/temp/keyframe_shot1.png")
kf2.save("/content/temp/keyframe_shot2.png")
print("[3A] 真人模特兒配戴首幀合成完成！子程序即將退出並釋放 100% 記憶體...")
"""

STAGE_3B_CODE = r"""
import os, json, gc, torch

with open("/content/temp/job_config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

gen_mode = cfg["generation_mode"]
if "方案 2" in gen_mode or "SVD-XT" in gen_mode:
    raise SystemExit(0)

style_key = cfg["commercial_style"].split(" ")[0]
custom_p_en = cfg.get("custom_scene_prompt_en", cfg.get("custom_scene_prompt", "")).strip()

# 偵測 GPU 架構：Tesla T4 (Compute Capability 7.5) 必須使用 float16 以避免 bfloat16 CUTLASS 錯誤！
cc_major = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 7
infer_dtype = torch.float16 if cc_major < 8 else torch.bfloat16
print(f"[3B] 偵測到 GPU Compute Capability {cc_major}.x，自動啟用最佳精度：{infer_dtype}")

# 影片模型需要「具體動作 + 鏡頭語言」才會動；避免 "unchanged / throughout" 這類會讓模型凍結畫面的字眼
ID_LOCK = "The same sunglasses stay on the face, crisp and sharp."
MOTION_PROMPTS = {
    "0_CLASSIC_MADRID": (
        "The model walks forward along a sunlit Mediterranean stone colonnade, turns the head slowly toward the camera and smiles slightly, the linen shirt and hair move in the sea breeze, golden sunlight flickers through the columns and glints across the lenses, the camera tracks backward smoothly in a slow dolly, cinematic luxury eyewear commercial, natural motion.",
        "Close-up: the model lifts the chin and turns the face from left to right toward the camera, hair blowing in the ocean wind, warm golden hour light sweeps across the face and the mirrored lenses reflect the sea, the camera pushes in slowly, shallow depth of field, cinematic luxury commercial."
    ),
    "1_CYBERPUNK_NEON": (
        "The model walks through a rainy neon-lit Tokyo street at night, rain drops falling, neon signs flicker, cyan and magenta reflections slide across the lenses, the model turns the head toward the camera, the camera tracks sideways smoothly, cinematic.",
        "Close-up: the model slowly turns the head and looks into the camera, neon lights pulse and reflect on the lenses, rain falls in the background bokeh, the camera pushes in slowly, cinematic."
    ),
    "2_VINTAGE_1985": (
        "The model leans on a vintage convertible on a coastal highway at sunset, then straightens up and turns toward the camera, hair and jacket move in the warm wind, sun flare drifts across the lenses, the camera pans slowly, 1985 Kodachrome film look.",
        "Close-up: the model turns the head with a confident smile, hair blowing in the sunset breeze, golden light flares across the lenses, the camera pushes in slowly, 35mm film grain."
    ),
    "3_HAUTE_COUTURE_NOIR": (
        "The model walks toward the camera on a Paris street, the long coat flows with each step, the head turns with a poised expression, rim light sweeps across the frame and lenses, the camera dollies backward slowly, high-fashion editorial film.",
        "Close-up: the model turns the face slowly from profile to the camera, dramatic rim light moves across the cheekbones and lenses, hair moves slightly, the camera pushes in, high-fashion editorial."
    ),
    "4_ALPINE_EXTREME": (
        "The skier stands on a snowy glacier ridge, wind blows snow powder past the camera, the skier turns the head toward the camera and adjusts the jacket collar, bright sunlight glints across the goggle lens, the camera orbits slowly, action sports commercial.",
        "Close-up: snow spray flies past the skier's face, the skier looks up into the camera, sharp sunlight sweeps across the goggle lens, the camera pushes in slowly, action sports commercial."
    ),
    "5_RIVIERA_SUNSET": (
        "The model walks across a teak yacht deck at golden sunset, the white linen shirt moves in the sea breeze, waves sparkle behind, the model turns toward the camera and smiles, the camera dollies slowly, luxury resort commercial.",
        "Close-up: the model turns the head toward the camera at golden sunset, hair moves in the sea breeze, warm light and sea reflections glide across the lenses, the camera pushes in slowly, luxury commercial."
    )
}

p_mot1, p_mot2 = MOTION_PROMPTS.get(style_key, MOTION_PROMPTS["0_CLASSIC_MADRID"])
if custom_p_en:
    p_mot1 = custom_p_en + ". The model moves naturally, turns the head toward the camera, hair and clothes move, lights flicker in the background, the camera tracks smoothly, cinematic luxury commercial."
    p_mot2 = "Close-up. " + custom_p_en + ". The model turns the face toward the camera with a confident expression, light sweeps across the lenses, the camera pushes in slowly, cinematic."
p_mot1 = p_mot1 + " " + ID_LOCK
p_mot2 = p_mot2 + " " + ID_LOCK
def check_embeds(*ts):
    for t in ts:
        if not torch.isfinite(t).all():
            raise RuntimeError("[3B] 文字特徵含 NaN/Inf (FP16 溢位)，請改選方案 1C (SVD-XT) 或方案 2。")

neg_video = "static image, frozen frame, no motion, still photo, deformed sunglasses, melting frame, morphing lenses, extra glasses, warped logo, distorted face, on-screen text, subtitles, watermark, blurry, flickering, jerky camera"

if "Wan2.2" in gen_mode:
    print("[3B] 正在單獨載入 Wan2.2 文字編碼器至 T4 GPU (避免與影片網路同時佔用系統 RAM)...")
    from diffusers import WanImageToVideoPipeline
    from transformers import UMT5EncoderModel
    MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    REV_ID = "b8fff7315c768468a5333511427288870b2e9635"
    # 文字編碼器權重直接逐片載入 GPU (device_map=cuda)，避免在 12.7GB 系統 RAM 中先組裝完整模型
    te = UMT5EncoderModel.from_pretrained(MODEL_ID, subfolder="text_encoder", revision=REV_ID, torch_dtype=infer_dtype, device_map="cuda", low_cpu_mem_usage=True)
    pipe_te = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID, revision=REV_ID, text_encoder=te,
        transformer=None, vae=None, image_encoder=None, image_processor=None,
        torch_dtype=infer_dtype
    )
    with torch.no_grad():
        pe1, npe1 = pipe_te.encode_prompt(prompt=p_mot1, negative_prompt=neg_video, do_classifier_free_guidance=True, max_sequence_length=512, device="cuda", dtype=infer_dtype)
        pe2, npe2 = pipe_te.encode_prompt(prompt=p_mot2, negative_prompt=neg_video, do_classifier_free_guidance=True, max_sequence_length=512, device="cuda", dtype=infer_dtype)
    check_embeds(pe1, npe1, pe2, npe2)
    torch.save({"pe1": pe1.cpu(), "npe1": npe1.cpu(), "pe2": pe2.cpu(), "npe2": npe2.cpu()}, "/content/temp/prompt_embeds.pt")
    print("[3B] Wan2.2 文字特徵編碼完成！即將關閉子程序釋放 T5 記憶體...")
elif "LTX-Video" in gen_mode:
    LTX_REPO = "Lightricks/LTX-Video-0.9.5"  # 0.9.5 與 0.9.0 共用同一個 T5-XXL，文字特徵兩版通用
    print("[3B] 正在單獨載入 LTX-Video 文字編碼器至 T4 GPU (避免與影片網路同時佔用系統 RAM)...")
    from diffusers import LTXImageToVideoPipeline
    from transformers import T5EncoderModel
    # LTX 的 T5-XXL 在 HF 上是 19GB fp32；直接逐片載入 GPU 並轉 fp16，系統 RAM 不會爆
    te = T5EncoderModel.from_pretrained(LTX_REPO, subfolder="text_encoder", torch_dtype=infer_dtype, device_map="cuda", low_cpu_mem_usage=True)
    pipe_te = LTXImageToVideoPipeline.from_pretrained(
        LTX_REPO, text_encoder=te,
        transformer=None, vae=None,
        torch_dtype=infer_dtype
    )
    with torch.no_grad():
        pe1, pam1, npe1, npam1 = pipe_te.encode_prompt(prompt=p_mot1, negative_prompt=neg_video, do_classifier_free_guidance=True, max_sequence_length=256, device="cuda", dtype=infer_dtype)
        pe2, pam2, npe2, npam2 = pipe_te.encode_prompt(prompt=p_mot2, negative_prompt=neg_video, do_classifier_free_guidance=True, max_sequence_length=256, device="cuda", dtype=infer_dtype)
    check_embeds(pe1, npe1, pe2, npe2)
    torch.save({
        "pe1": pe1.cpu(), "pam1": pam1.cpu(), "npe1": npe1.cpu(), "npam1": npam1.cpu(),
        "pe2": pe2.cpu(), "pam2": pam2.cpu(), "npe2": npe2.cpu(), "npam2": npam2.cpu()
    }, "/content/temp/prompt_embeds.pt")
    print("[3B] LTX-Video 文字特徵編碼完成！即將關閉子程序釋放 T5 記憶體...")
"""

STAGE_3C_CODE = r"""
import os, json, gc, time, traceback, subprocess, cv2, torch
import numpy as np
from PIL import Image

with open("/content/temp/job_config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

gen_mode = cfg["generation_mode"]
shot1_mp4 = "/content/temp/raw_shot1.mp4"
shot2_mp4 = "/content/temp/raw_shot2.mp4"
vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 15.0
cc_major = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 7
# Tesla T4 (cc 7.5) 原生支援 float16 Tensor Cores；若使用 bfloat16 會觸發 cutlassF: no kernel found to launch!
infer_dtype = torch.float16 if cc_major < 8 else torch.bfloat16

# ---- 2.5D 視差：人物前景與背景以不同速度/縮放運動，產生立體縱深 ----
def build_parallax_layers(kf_bgr, face_box=None):
    # 以 GrabCut 從臉部位置推估人物前景，並修補出背景板，用於 2.5D 視差運鏡
    h, w = kf_bgr.shape[:2]
    s = 0.5
    small = cv2.resize(kf_bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if face_box is not None:
        faces = [tuple(int(v * s) for v in face_box)]
    else:
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
    if len(faces) > 0:
        fx, fy, fw, fh = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
        x0, x1 = max(1, int(fx - fw * 1.5)), min(sw - 2, int(fx + fw * 2.5))
        y0 = max(1, int(fy - fh * 0.7))
        focus = ((fx + fw / 2) / s, (fy + fh * 0.4) / s)
    else:
        x0, x1, y0 = int(sw * 0.18), int(sw * 0.82), int(sh * 0.12)
        focus = (w / 2, h * 0.4)
    rect = (x0, y0, x1 - x0, sh - 2 - y0)
    mask = np.zeros((sh, sw), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    except cv2.error:
        fg = np.zeros((sh, sw), np.float32)
    fpx, fpy = int(focus[0] * s), int(min(sh - 1, focus[1] * s + 0.05 * sh))
    if fg.mean() < 0.06 or fg.mean() > 0.85 or fg[min(sh - 1, fpy), min(sw - 1, fpx)] < 0.5:
        # 找不到可靠的人物遮罩 → 不做分層，整張畫面一起運鏡 (不會產生鬼影)
        ones = np.ones((h, w, 1), np.float32)
        return kf_bgr.astype(np.float32), kf_bgr.astype(np.float32), ones, focus
    # 只保留最大連通區塊 + 平滑邊緣
    n, lab, st, _ = cv2.connectedComponentsWithStats((fg > 0.5).astype(np.uint8))
    if n > 1:
        keep = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        fg = (lab == keep).astype(np.float32)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    fg = cv2.GaussianBlur(fg, (0, 0), 2.0)
    hole = cv2.dilate((fg > 0.05).astype(np.uint8) * 255, np.ones((15, 15), np.uint8))
    bg_small = cv2.inpaint(small, hole, 7, cv2.INPAINT_TELEA)
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)
    bg = cv2.GaussianBlur(bg, (0, 0), 1.2)
    alpha = cv2.resize(fg, (w, h), interpolation=cv2.INTER_LINEAR)[..., None]
    return bg.astype(np.float32), kf_bgr.astype(np.float32), alpha, focus

def parallax_frame(bg, fg, alpha, focus, t, shot_idx):
    h, w = fg.shape[:2]
    e = t * t * (3 - 2 * t)  # ease in-out
    d = 1.0 if shot_idx == 1 else -1.0
    # 背景：慢、反向橫移；前景人物：較大推進 + 同向橫移 + 微幅呼吸 → 立體縱深感
    sb = 1.04 + 0.03 * e
    Mb = cv2.getRotationMatrix2D(focus, 0, sb)
    Mb[0, 2] += -d * 22.0 * (e - 0.5)
    sf = 1.0 + (0.085 * e if shot_idx == 1 else 0.085 * (1 - e)) + 0.006 * np.sin(t * np.pi * 3)
    Mf = cv2.getRotationMatrix2D(focus, d * 0.6 * (e - 0.5), sf)
    Mf[0, 2] += d * 16.0 * (e - 0.5)
    Mf[1, 2] += -6.0 * e
    bgw = cv2.warpAffine(bg, Mb, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fgw = cv2.warpAffine(fg, Mf, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    aw = cv2.warpAffine(alpha, Mf, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)[..., None]
    out = bgw * (1 - aw) + fgw * aw
    # 斜向光斑掃掠
    yg, xg = np.mgrid[0:h, 0:w].astype(np.float32)
    flare = np.exp(-(((xg + yg * 0.35) - (t * 1.4 - 0.2) * w) ** 2) / (2 * 95.0 ** 2))[..., None]
    out = out + flare * np.array([10.0, 18.0, 26.0], np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)

def load_face_box(shot_idx, w, h):
    try:
        with open("/content/temp/face_boxes.json", "r", encoding="utf-8") as f:
            b = json.load(f).get(str(shot_idx))
        return None if not b else (b[0] * w, b[1] * h, b[2] * w, b[3] * h)
    except Exception:
        return None

def render_3d_motion_fallback():
    print("[3C] 正在執行 2.5D 視差運鏡 (人物/背景分層) + 光斑掃掠合成...")
    product_rgba = Image.open("/content/temp/product_clean_rgba.png").convert("RGBA")
    w, h = 720, 1280
    prod_np = np.array(product_rgba)
    for shot_idx, out_mp4 in [(1, shot1_mp4), (2, shot2_mp4)]:
        num_frames = 192
        raw_tmp = out_mp4.replace(".mp4", "_raw.mp4")
        writer = cv2.VideoWriter(raw_tmp, cv2.VideoWriter_fourcc(*'mp4v'), 24.0, (w, h))
        kf_path = f"/content/temp/keyframe_shot{shot_idx}.png"
        has_kf = os.path.exists(kf_path) and ("方案 2" not in gen_mode)
        if has_kf:
            kf_bgr = cv2.cvtColor(np.array(Image.open(kf_path).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)), cv2.COLOR_RGB2BGR)
            px_bg, px_fg, px_alpha, px_focus = build_parallax_layers(kf_bgr, load_face_box(shot_idx, w, h))
            print(f"  -> 第 {shot_idx} 段視差分層完成 (人物前景佔 {float(px_alpha.mean()) * 100:.0f}%)")
        base_w = 500 if shot_idx == 1 else 580
        base_h = int(prod_np.shape[0] * (base_w / prod_np.shape[1]))
        prod_scaled = cv2.resize(prod_np, (base_w, base_h), interpolation=cv2.INTER_LANCZOS4)
        for i in range(num_frames):
            t = i / (num_frames - 1)
            if has_kf:
                # 真人配戴首幀：人物/背景分層視差運鏡 + 光斑掃掠
                writer.write(parallax_frame(px_bg, px_fg, px_alpha, px_focus, t, shot_idx))
            else:
                yg, xg = np.mgrid[0:h, 0:w]
                bg = np.zeros((h, w, 3), dtype=np.float32)
                bg[..., 0] = 145 + 65 * (1.0 - yg / h)
                bg[..., 1] = 160 + 65 * (1.0 - yg / h)
                bg[..., 2] = 175 + 70 * (1.0 - yg / h)
                zoom = (0.94 + 0.12 * t) if shot_idx == 1 else (1.06 - 0.08 * t)
                cur_w, cur_h = int(base_w * zoom), int(base_h * zoom)
                p_cur = cv2.resize(prod_scaled, (cur_w, cur_h), interpolation=cv2.INTER_LANCZOS4)
                alpha = (p_cur[:, :, 3].astype(np.float32) / 255.0)[..., np.newaxis]
                rgb = p_cur[:, :, :3][:, :, ::-1].astype(np.float32)
                pyg, pxg = np.mgrid[0:cur_h, 0:cur_w]
                sweep = np.exp(-(((pxg + pyg * 0.45) - (t * 1.6 - 0.3) * cur_w) ** 2) / (2 * (28.0 ** 2)))[..., np.newaxis]
                rgb = np.clip(rgb + sweep * 65.0 * alpha, 0, 255)
                cy, cx = h // 2, w // 2
                top_y, left_x = cy - cur_h // 2, cx - cur_w // 2
                bg[top_y:top_y+cur_h, left_x:left_x+cur_w] = bg[top_y:top_y+cur_h, left_x:left_x+cur_w] * (1.0 - alpha) + rgb * alpha
                writer.write(np.clip(bg, 0, 255).astype(np.uint8))
        writer.release()
        subprocess.run(["ffmpeg", "-y", "-i", raw_tmp, "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", out_mp4], check=True)
    print("[3C] 兩段 8.0s 動態鏡頭生成完成！")

if "方案 2" in gen_mode:
    render_3d_motion_fallback()
    raise SystemExit(0)

kf1_pil = Image.open("/content/temp/keyframe_shot1.png").convert("RGB")
kf2_pil = Image.open("/content/temp/keyframe_shot2.png").convert("RGB")
from diffusers.utils import export_to_video

def check_frames(frames, tag):
    arr = np.asarray(frames[len(frames) // 2]).astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    if not np.isfinite(arr).all() or arr.std() < 2.0:
        raise RuntimeError(f"{tag} 輸出為全黑/純色畫面 (FP16 數值溢位)")

STATUS = {"mode": gen_mode, "ai_ok": False, "engine": None, "shots": [], "error": None}
MOTION_MIN = 3.0

def _to_u8(fr):
    arr = np.asarray(fr)
    if arr.dtype != np.uint8:
        arr = np.clip(arr.astype(np.float32) * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return arr

def motion_score(frames):
    # 首尾幀與相鄰幀的平均像素差 (0~255)；先前的「靜止」輸出約 1.9
    g = [cv2.resize(cv2.cvtColor(_to_u8(f), cv2.COLOR_RGB2GRAY), (120, 208)).astype(np.float32) for f in frames[::4]]
    if len(g) < 2:
        return 0.0
    end_diff = float(np.abs(g[-1] - g[0]).mean())
    step_diff = float(np.mean([np.abs(g[i + 1] - g[i]).mean() for i in range(len(g) - 1)])) * 4.0
    return max(end_diff, step_diff)

def ltx_preprocess(img, crf):
    # LTX 以壓縮影片訓練；乾淨照片當首幀容易產生「凍結」影片 → 先經過一次 H.264 壓縮
    src, enc = "/content/temp/_ltx_in.png", "/content/temp/_ltx_in.mp4"
    img.save(src)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", enc], check=True)
    cap = cv2.VideoCapture(enc)
    ok, fr = cap.read()
    cap.release()
    return Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)) if ok else img

def finalize_ai_clip(frames, out_p, add_camera_move=False, shot_idx=1, dur=8.0, src_fps=24):
    # AI 片段 → 8 秒 24fps：動態補幀 (minterpolate) 取代單純拉長，慢動作更順
    tmp_raw, tmp_interp = out_p.replace(".mp4", "_ai.mp4"), out_p.replace(".mp4", "_interp.mp4")
    export_to_video([_to_u8(f) for f in frames], tmp_raw, fps=src_fps)
    pts = dur / (len(frames) / float(src_fps))
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", tmp_raw,
        "-vf", f"setpts={pts:.4f}*PTS,minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,scale=720:1280:flags=lanczos,tpad=stop_mode=clone:stop_duration=1",
        "-frames:v", str(int(round(dur * 24))), "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", tmp_interp
    ], check=True)
    if not add_camera_move:
        os.replace(tmp_interp, out_p)
        return
    # 模型動態不足時，再疊加一層緩慢推軌 + 橫移，確保鏡頭一定在動
    print("     ↳ 動態仍不足，疊加電影推軌運鏡")
    cap = cv2.VideoCapture(tmp_interp)
    src = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        src.append(fr)
    cap.release()
    w, h, n = 720, 1280, max(1, len(src))
    tmp_mv = out_p.replace(".mp4", "_mv.mp4")
    wr = cv2.VideoWriter(tmp_mv, cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
    for i, fr in enumerate(src):
        t = i / max(1, n - 1)
        e = t * t * (3 - 2 * t)
        sc = 1.0 + 0.09 * e if shot_idx == 1 else 1.09 - 0.07 * e
        dx = (18.0 if shot_idx == 1 else -18.0) * (e - 0.5)
        M = cv2.getRotationMatrix2D((w / 2, h * 0.42), 0, sc)
        M[0, 2] += dx
        wr.write(cv2.warpAffine(fr, M, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT))
    wr.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp_mv, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", out_p], check=True)

try:
    if "Wan2.2" in gen_mode:
        print(f"[3C] 正在載入無文字編碼器版 Wan2.2-TI2V-5B (text_encoder=None, dtype={infer_dtype}，僅需 ~6GB RAM)...")
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
        REV_ID = "b8fff7315c768468a5333511427288870b2e9635"
        embeds = torch.load("/content/temp/prompt_embeds.pt", map_location="cpu", weights_only=False)
        vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", revision=REV_ID, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        pipe = WanImageToVideoPipeline.from_pretrained(
            MODEL_ID, revision=REV_ID, vae=vae,
            text_encoder=None, tokenizer=None,
            image_encoder=None, image_processor=None,
            torch_dtype=infer_dtype, expand_timesteps=True,
            low_cpu_mem_usage=True
        )
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        torch.backends.cuda.matmul.allow_tf32 = True
        STATUS["engine"] = MODEL_ID

        gen_w, gen_h, gen_f, gen_steps = (704, 1280, 121, 30) if vram_gb >= 20.0 else (480, 832, 81, 22)
        shots = [
            (1, kf1_pil, embeds["pe1"].to("cuda", infer_dtype), embeds["npe1"].to("cuda", infer_dtype), shot1_mp4),
            (2, kf2_pil, embeds["pe2"].to("cuda", infer_dtype), embeds["npe2"].to("cuda", infer_dtype), shot2_mp4)
        ]
        for idx, kf, pe, npe, out_p in shots:
            t0 = time.time()
            print(f"  -> 正在透過 T4 GPU 運算第 {idx}/2 段 Wan2.2 動態鏡頭 ({gen_w}x{gen_h}, {gen_f}f)...")
            kf_in = kf.resize((gen_w, gen_h), Image.Resampling.LANCZOS)
            frames = pipe(
                image=kf_in,
                prompt=None, negative_prompt=None,
                prompt_embeds=pe, negative_prompt_embeds=npe,
                width=gen_w, height=gen_h,
                num_frames=gen_f, num_inference_steps=gen_steps,
                guidance_scale=5.0,
                generator=torch.Generator(device="cuda").manual_seed(20261103 + idx)
            ).frames[0]
            check_frames(frames, "Wan2.2")
            ms = motion_score(frames)
            finalize_ai_clip(frames, out_p, add_camera_move=ms < MOTION_MIN, shot_idx=idx)
            STATUS["shots"].append({"shot": idx, "motion": round(ms, 2)})
            print(f"     ✓ 第 {idx} 段完成 (motion={ms:.2f}，耗時 {time.time() - t0:.1f}s)")
            del frames
            gc.collect()
            torch.cuda.empty_cache()

    elif "LTX-Video" in gen_mode:
        from diffusers import LTXImageToVideoPipeline
        embeds = torch.load("/content/temp/prompt_embeds.pt", map_location="cpu", weights_only=False)
        # 0.9.5 的圖生影片動態遠優於 0.9.0；兩者共用同一個 T5，若 0.9.5 失敗自動退回 0.9.0
        LTX_REPOS = ["Lightricks/LTX-Video-0.9.5", "Lightricks/LTX-Video"]
        LTX_W, LTX_H, LTX_F, LTX_STEPS = 480, 832, 97, 30
        shots = [
            (1, kf1_pil, embeds["pe1"], embeds["pam1"], embeds["npe1"], embeds["npam1"], shot1_mp4),
            (2, kf2_pil, embeds["pe2"], embeds["pam2"], embeds["npe2"], embeds["npam2"], shot2_mp4)
        ]

        def run_ltx(repo):
            print(f"[3C] 正在載入無文字編碼器版 {repo} (text_encoder=None, dtype={infer_dtype}，約 ~5GB RAM)...")
            pipe = LTXImageToVideoPipeline.from_pretrained(
                repo, text_encoder=None, tokenizer=None,
                torch_dtype=infer_dtype, low_cpu_mem_usage=True
            )
            pipe.enable_model_cpu_offload()
            pipe.vae.enable_tiling()
            for idx, kf, pe, pam, npe, npam, out_p in shots:
                best = None
                # 第 1 次 CRF 30；若畫面幾乎靜止，換種子並加重壓縮 (CRF 40) 重試一次
                for attempt, crf in enumerate([30, 40]):
                    t0 = time.time()
                    print(f"  -> 第 {idx}/2 段 LTX 動態鏡頭 ({LTX_W}x{LTX_H}, {LTX_F}f, {LTX_STEPS} steps, 首幀 CRF {crf})...")
                    kf_in = ltx_preprocess(kf.resize((LTX_W, LTX_H), Image.Resampling.LANCZOS), crf)
                    frames = pipe(
                        image=kf_in,
                        prompt=None, negative_prompt=None,
                        prompt_embeds=pe.to("cuda", infer_dtype),
                        prompt_attention_mask=pam.to("cuda"),
                        negative_prompt_embeds=npe.to("cuda", infer_dtype),
                        negative_prompt_attention_mask=npam.to("cuda"),
                        width=LTX_W, height=LTX_H, num_frames=LTX_F, frame_rate=24,
                        num_inference_steps=LTX_STEPS, guidance_scale=3.0,
                        decode_timestep=0.05, decode_noise_scale=0.025,
                        generator=torch.Generator("cuda").manual_seed(42 + idx + 1000 * attempt)
                    ).frames[0]
                    check_frames(frames, "LTX-Video")
                    ms = motion_score(frames)
                    print(f"     動態量 motion={ms:.2f} (門檻 {MOTION_MIN})，耗時 {time.time() - t0:.1f}s")
                    if best is None or ms > best[0]:
                        best = (ms, frames)
                    if ms >= MOTION_MIN:
                        break
                    print("     ⚠️ 畫面幾乎靜止，換種子並加重首幀壓縮重試...")
                finalize_ai_clip(best[1], out_p, add_camera_move=best[0] < MOTION_MIN, shot_idx=idx)
                STATUS["shots"].append({"shot": idx, "motion": round(best[0], 2)})
                print(f"     ✓ 第 {idx} 段完成 (motion={best[0]:.2f})")
                del frames, best
                gc.collect()
                torch.cuda.empty_cache()
            del pipe

        for ri, repo in enumerate(LTX_REPOS):
            try:
                STATUS["shots"] = []
                run_ltx(repo)
                STATUS["engine"] = repo
                break
            except Exception as e:
                if ri == len(LTX_REPOS) - 1:
                    raise
                print(f"[3C] ⚠️ {repo} 失敗 ({e})，自動改用 {LTX_REPOS[ri + 1]} 重試...")
                traceback.print_exc()
                gc.collect()
                torch.cuda.empty_cache()

    else:
        print("[3C] 正在載入 SVD-XT 純圖生影片模型 (無文字編碼器，僅需 ~4.2GB RAM)...")
        from diffusers import StableVideoDiffusionPipeline
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16, variant="fp16", low_cpu_mem_usage=True
        )
        pipe.enable_model_cpu_offload()
        for idx, (kf, out_p) in enumerate([(kf1_pil, shot1_mp4), (kf2_pil, shot2_mp4)], 1):
            print(f"  -> 正在透過 T4 GPU 運算第 {idx}/2 段 SVD-XT 動態鏡頭...")
            kf_in = kf.resize((576, 1024), Image.Resampling.LANCZOS)
            frames = pipe(kf_in, decode_chunk_size=2, num_frames=25, motion_bucket_id=160, noise_aug_strength=0.03,
                          generator=torch.Generator("cuda").manual_seed(7 + idx)).frames[0]
            check_frames(frames, "SVD-XT")
            ms = motion_score(frames)
            finalize_ai_clip(frames, out_p, add_camera_move=ms < MOTION_MIN, shot_idx=idx, src_fps=8)
            STATUS["shots"].append({"shot": idx, "motion": round(ms, 2)})
            print(f"     ✓ 第 {idx} 段完成 (motion={ms:.2f})")
        STATUS["engine"] = "stabilityai/stable-video-diffusion-img2vid-xt"
    STATUS["ai_ok"] = True
except Exception as e:
    STATUS["ai_ok"] = False
    STATUS["error"] = f"{type(e).__name__}: {e}"
    print(f"\n⚠️ [3C] AI 擴散引擎發生例外 ({e})，完整診斷訊息如下：")
    traceback.print_exc()
    print("\n🔄 正在自動啟動真人配戴首幀 2.5D 視差運鏡保底合成，確保廣告順利產出...")
    render_3d_motion_fallback()
finally:
    with open("/content/temp/stage3c_status.json", "w", encoding="utf-8") as f:
        json.dump(STATUS, f, ensure_ascii=False, indent=2)
"""

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🕶️ ESPNER 精品廣告 AI 自動生成系統 (Colab 標準 T4 省記憶體版 v5.1)\n",
            "**✅ 專為「標準 12.7GB 系統 RAM + 15GB T4 GPU」設計 — 免開「大量 RAM (High-RAM)」、不扣額外點數！**\n",
            "\n",
            "---\n",
            "### 💡 v5.1 省記憶體版三大優化（解決 RAM 用盡與 `rembg` 模組問題）：\n",
            "1. **零 `rembg` 依賴（純 OpenCV 光學去背）**：改用與本機 v5 正式版相同的 OpenCV FloodFill + 亮度柔和邊緣去背演算法，**0 MB 額外記憶體、0.01 秒完成**，再也不會出現 `No module named 'rembg'`。\n",
            "2. **文字編碼器預先拆離技術 (`text_encoder=None`)**：先在獨立子程序單獨載入文字編碼器，花 2 秒把提示詞轉成向量 (`prompt_embeds.pt`) 後**立刻關閉釋放 9.5 GB 記憶體**；接著載入影片模型時完全不載入文字編碼器，系統 RAM 用量從 20GB 驟降至 **5.5 GB 以下**。\n",
            "3. **三階段獨立子程序隔離 (`stage_3a` ➔ `stage_3b` ➔ `stage_3c`)**：每個階段結束時由 Linux 作業系統 100% 強制回收所有系統 RAM 與顯存。"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "cellView": "form"
        },
        "outputs": [],
        "source": [
            "#@title 🚀 步驟 1：掛載 Google Drive、檢查 T4 GPU 並安裝環境套件 (初次執行約需 1.5 分鐘)\n",
            "import subprocess, os, sys, json, shutil\n",
            "\n",
            "#@markdown 是否自動掛載 Google 雲端硬碟以自動備份產出的廣告影片？\n",
            "MOUNT_GOOGLE_DRIVE = True #@param {type:\"boolean\"}\n",
            "\n",
            "if MOUNT_GOOGLE_DRIVE:\n",
            "    try:\n",
            "        from google.colab import drive\n",
            "        drive.mount(\"/content/drive\")\n",
            "        os.makedirs(\"/content/drive/MyDrive/ESPNER_Colab_T4_Output\", exist_ok=True)\n",
            "        print(\"✅ Google 雲端硬碟已掛載！成品將自動同步至 /content/drive/MyDrive/ESPNER_Colab_T4_Output\")\n",
            "    except Exception as e:\n",
            "        print(f\"ℹ️ 略過雲端硬碟掛載 ({e})，成品將直接透過瀏覽器下載。\")\n",
            "\n",
            "print(\"\\n🔍 正在檢查 Colab GPU 與系統 RAM 狀態...\")\n",
            "gpu_info = subprocess.getoutput(\"nvidia-smi\")\n",
            "for line in gpu_info.splitlines():\n",
            "    if any(k in line for k in [\"Tesla\", \"T4\", \"L4\", \"A100\", \"MiB\"]):\n",
            "        print(\"  \", line)\n",
            "ram_info = subprocess.getoutput(\"free -h\")\n",
            "print(\"\\n📊 系統記憶體 (System RAM)：\\n\" + ram_info)\n",
            "\n",
            "print(\"\\n📦 正在安裝 Diffusers 0.38 省記憶體引擎與字體套件...\")\n",
            "subprocess.run([\n",
            "    sys.executable, \"-m\", \"pip\", \"install\", \"-q\",\n",
            "    \"diffusers==0.38.0\", \"transformers==4.57.3\", \"huggingface-hub==0.36.2\",\n",
            "    \"accelerate>=1.1\", \"sentencepiece\", \"protobuf\", \"safetensors\", \"ftfy\",\n",
            "    \"opencv-python-headless<5\", \"pillow\", \"imageio-ffmpeg\", \"soundfile\"\n",
            "], check=True)\n",
            "\n",
            "print(\"\\n🧪 正在檢查 PyTorch CUDA (GPU) 是否可用...\")\n",
            "import json, shutil\n",
        ] + GPU_GUARD_SRC + [
            "ensure_cuda_torch(required=False)\n",
            "\n",
            "subprocess.run([\"apt-get\", \"update\", \"-qq\"], check=True)\n",
            "subprocess.run([\"apt-get\", \"install\", \"-y\", \"-qq\", \"ffmpeg\", \"fonts-noto-cjk\", \"fonts-dejavu-core\"], check=True)\n",
            "\n",
            "for d in [\"/content/input\", \"/content/output\", \"/content/temp\"]:\n",
            "    os.makedirs(d, exist_ok=True)\n",
            "print(\"✅ v5.1 省記憶體環境準備完成！（無須開啟大量 RAM）\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "cellView": "form"
        },
        "outputs": [],
        "source": [
            "#@title 📸 步驟 2：設定產品名稱、廣告風格並上傳產品正面圖片\n",
            "import os, json\n",
            "from google.colab import files\n",
            "from IPython.display import display, Image as IPImage\n",
            "\n",
            "#@markdown ### 🏷️ 產品基本設定\n",
            "MODEL_NAME = \"AVIATOR\" #@param {type:\"string\"}\n",
            "PRODUCT_CATEGORY = \"SUNGLASSES (精品太陽眼鏡)\" #@param [\"SUNGLASSES (精品太陽眼鏡)\", \"SNOW_GOGGLES (極限滑雪護目鏡)\"]\n",
            "\n",
            "#@markdown ### 🎨 選擇廣告視覺風格 (與現有 v5 正式版廣告完全相同規格)\n",
            "COMMERCIAL_STYLE = \"0_CLASSIC_MADRID (經典精品：15s雙鏡故事 + 2s金框石牆過場 + 4.5s白底原圖片尾 = 21.5s)\" #@param [\"0_CLASSIC_MADRID (經典精品：15s雙鏡故事 + 2s金框石牆過場 + 4.5s白底原圖片尾 = 21.5s)\", \"1_CYBERPUNK_NEON (賽博龐克：霓虹調色 + 光學HUD + 故障轉場 + 5s白底原圖片尾 = 21.0s)\", \"2_VINTAGE_1985 (復古1985：柯達暖金膠片 + 6s散景金框章節卡 + 5s白底原圖片尾 = 21.0s)\", \"3_HAUTE_COUTURE_NOIR (高訂黑白：巴黎米蘭黑白時尚邊框 + 5s白底原圖片尾 = 21.0s)\", \"4_ALPINE_EXTREME (極限冰川：冰藍橘紅HUD + 上下分割雙視角 + 5s白底原圖片尾 = 21.0s)\", \"5_RIVIERA_SUNSET (里維耶拉夕陽：地中海金色波光 + 5s白底原圖片尾 = 21.0s)\"]\n",
            "\n",
            "#@markdown ### 🤖 選擇標準 12.7GB RAM + T4 GPU 動態運算引擎\n",
            "GENERATION_MODE = \"方案 1A (超省RAM推薦)：LTX-Video-2B 拆離式圖生影片 + 真人模特兒配戴首幀 (僅需 5GB RAM)\" #@param [\"方案 1A (超省RAM推薦)：LTX-Video-2B 拆離式圖生影片 + 真人模特兒配戴首幀 (僅需 5GB RAM)\", \"方案 1B (高畫質省RAM)：Wan2.2-TI2V-5B 兩階段拆離圖生影片 + 真人模特兒配戴首幀 (僅需 6.5GB RAM)\", \"方案 1C (超輕量穩定)：SVD-XT 圖生影片 + 真人模特兒配戴首幀 (無文字編碼器、僅需 4.2GB RAM)\", \"方案 2 (零變形極速)：純產品 3D/2.5D 刀光立體運鏡 + 電影散景 (1分鐘極速完成、僅需 1.5GB RAM)\"]\n",
            "\n",
            "#@markdown ### 📂 圖片來源選擇\n",
            "IMAGE_SOURCE = \"1_點擊下方按鈕上傳圖片\" #@param [\"1_點擊下方按鈕上傳圖片\", \"2_直接使用雲端硬碟內建 AVIATOR 測試圖 (免手動上傳)\", \"3_直接使用雲端硬碟內建 MADRID 測試圖 (免手動上傳)\", \"4_直接使用雲端硬碟內建 ZERMATT 雪鏡測試圖 (免手動上傳)\"]\n",
            "\n",
            "#@markdown ### 🎬 自訂場景描述 (留空則依風格自動配對最佳模特兒與場景)\n",
            "CUSTOM_SCENE_PROMPT = \"\" #@param {type:\"string\"}\n",
            "\n",
            "os.makedirs(\"/content/input\", exist_ok=True)\n",
            "os.makedirs(\"/content/temp\", exist_ok=True)\n",
            "\n",
            "drive_sample_map = {\n",
            "    \"2_\": (\"/content/drive/MyDrive/Colab Notebooks/ESPNER_測試商品圖/AVIATOR_front.jpg\", \"AVIATOR\"),\n",
            "    \"3_\": (\"/content/drive/MyDrive/Colab Notebooks/ESPNER_測試商品圖/MADRID_front.jpg\", \"MADRID\"),\n",
            "    \"4_\": (\"/content/drive/MyDrive/Colab Notebooks/ESPNER_測試商品圖/ZERMATT_front.jpg\", \"ZERMATT\")\n",
            "}\n",
            "\n",
            "PRODUCT_PHOTO_PATH = None\n",
            "for prefix, (sample_p, sample_name) in drive_sample_map.items():\n",
            "    if IMAGE_SOURCE.startswith(prefix) and os.path.exists(sample_p):\n",
            "        PRODUCT_PHOTO_PATH = sample_p\n",
            "        if MODEL_NAME == \"AVIATOR\":\n",
            "            MODEL_NAME = sample_name\n",
            "        break\n",
            "\n",
            "if PRODUCT_PHOTO_PATH is None:\n",
            "    # 若 /content/input 已經有剛剛上傳過的圖片且不想重傳，也可以直接選取\n",
            "    print(\"👇 請點擊下方「選擇檔案 (Choose Files)」上傳一張產品正面照片 (.jpg / .png / .webp)：\")\n",
            "    uploaded = files.upload()\n",
            "    if not uploaded:\n",
            "        existing = [os.path.join(\"/content/input\", x) for x in os.listdir(\"/content/input\") if x.lower().endswith((\".jpg\", \".jpeg\", \".png\", \".webp\"))]\n",
            "        if existing:\n",
            "            PRODUCT_PHOTO_PATH = existing[-1]\n",
            "            print(f\"ℹ️ 使用已存在之上傳圖片：{PRODUCT_PHOTO_PATH}\")\n",
            "        else:\n",
            "            raise ValueError(\"❌ 未上傳任何圖片！請重新執行此儲存格並選擇產品正面圖。\")\n",
            "    else:\n",
            "        fname = list(uploaded.keys())[0]\n",
            "        PRODUCT_PHOTO_PATH = os.path.join(\"/content/input\", fname)\n",
            "        with open(PRODUCT_PHOTO_PATH, \"wb\") as f:\n",
            "            f.write(uploaded[fname])\n",
            "\n",
            "cfg = {\n",
            "    \"model_name\": MODEL_NAME.strip(),\n",
            "    \"product_category\": PRODUCT_CATEGORY,\n",
            "    \"commercial_style\": COMMERCIAL_STYLE,\n",
            "    \"generation_mode\": GENERATION_MODE,\n",
            "    \"custom_scene_prompt\": CUSTOM_SCENE_PROMPT.strip(),\n",
            "    \"photo_path\": PRODUCT_PHOTO_PATH\n",
            "}\n",
            "with open(\"/content/temp/job_config.json\", \"w\", encoding=\"utf-8\") as f:\n",
            "    json.dump(cfg, f, ensure_ascii=False, indent=2)\n",
            "\n",
            "print(f\"\\n✅ 成功載入產品圖片：{PRODUCT_PHOTO_PATH} (款式：{MODEL_NAME})\")\n",
            "display(IPImage(filename=PRODUCT_PHOTO_PATH, width=300))"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "cellView": "form"
        },
        "outputs": [],
        "source": [
            "#@title 🧠 步驟 3：低記憶體分段子程序運算 — (3A 真人配戴首幀 ➔ 3B 拆離式文字編碼 ➔ 3C 圖生影片)\n",
            "import subprocess, sys, os, json\n",
            "from pathlib import Path\n",
            "from IPython.display import display, Image as IPImage\n",
            "\n",
            "# 前置檢查：執行階段重啟後 /content 會被清空，必須依序重新執行步驟 1、2\n",
            "os.makedirs(\"/content/temp\", exist_ok=True)\n",
            "os.makedirs(\"/content/output\", exist_ok=True)\n",
            "_cfg_p = \"/content/temp/job_config.json\"\n",
            "if not os.path.exists(_cfg_p):\n",
            "    raise RuntimeError(\"❌ 找不到步驟 2 的設定檔 (執行階段已重啟或尚未執行步驟 2)。\\n👉 請依序重新執行『步驟 1』→『步驟 2』(上傳產品圖)，再執行步驟 3。\")\n",
            "with open(_cfg_p, \"r\", encoding=\"utf-8\") as _f:\n",
            "    _photo = json.load(_f).get(\"photo_path\")\n",
            "if not _photo or not os.path.exists(_photo):\n",
            "    raise RuntimeError(f\"❌ 找不到產品圖片 {_photo} (執行階段重啟後上傳的檔案會消失)。\\n👉 請重新執行『步驟 2』上傳產品圖，再執行步驟 3。\")\n",
            "\n",
            "# 若執行階段曾重啟，自動確保 diffusers==0.38.0 已安裝\n",
            "try:\n",
            "    import diffusers\n",
            "    assert diffusers.__version__ == \"0.38.0\"\n",
            "except Exception:\n",
            "    print(\"📦 偵測到執行階段剛重啟，正在自動補齊 Diffusers 0.38 套件...\")\n",
            "    subprocess.run([\n",
            "        sys.executable, \"-m\", \"pip\", \"install\", \"-q\",\n",
            "        \"diffusers==0.38.0\", \"transformers==4.57.3\", \"huggingface-hub==0.36.2\",\n",
            "        \"accelerate>=1.1\", \"sentencepiece\", \"protobuf\", \"safetensors\", \"ftfy\",\n",
            "        \"opencv-python-headless<5\", \"pillow\", \"imageio-ffmpeg\", \"soundfile\"\n",
            "    ], check=True)\n",
            "\n",
            "# ==============================================================================\n",
            "# 子程序腳本 3A：純 OpenCV 零記憶體去背 + 中英導演提示詞自動翻譯 + RealVisXL 真人模特兒生成 + 100% 原始眼鏡配戴合成\n",
            "# ==============================================================================\n",
        ] + embed_stage("stage_3a_code", STAGE_3A_CODE) + [
            "\n",
            "# ==============================================================================\n",
            "# 子程序腳本 3B：獨立文字編碼器 (T4 原生 FP16 加速，算完即釋放 9.5GB T5 記憶體！)\n",
            "# ==============================================================================\n",
        ] + embed_stage("stage_3b_code", STAGE_3B_CODE) + [
            "\n",
            "# ==============================================================================\n",
            "# 子程序腳本 3C：無文字編碼器 (`text_encoder=None`) 純圖生影片擴散推論 (T4 原生 FP16)\n",
            "# ==============================================================================\n",
        ] + embed_stage("stage_3c_code", STAGE_3C_CODE) + [
            "\n",
            "Path(\"/content/temp/stage_3a.py\").write_text(stage_3a_code, encoding=\"utf-8\")\n",
            "Path(\"/content/temp/stage_3b.py\").write_text(stage_3b_code, encoding=\"utf-8\")\n",
            "Path(\"/content/temp/stage_3c.py\").write_text(stage_3c_code, encoding=\"utf-8\")\n",
            "\n",
            "import json, shutil\n",
        ] + GPU_GUARD_SRC + [
            "with open(\"/content/temp/job_config.json\", \"r\", encoding=\"utf-8\") as _f:\n",
            "    _gen_mode = json.load(_f)[\"generation_mode\"]\n",
            "ensure_cuda_torch(required=(\"方案 2\" not in _gen_mode))\n",
            "\n",
            "def run_stage(script_path, stage_title):\n",
            "    print(f\"\\n🚀 開始執行【{stage_title}】...\")\n",
            "    proc = subprocess.Popen(\n",
            "        [sys.executable, \"-u\", script_path],\n",
            "        stdout=subprocess.PIPE,\n",
            "        stderr=subprocess.STDOUT,\n",
            "        text=True,\n",
            "        bufsize=1\n",
            "    )\n",
            "    output_lines = []\n",
            "    for line in proc.stdout:\n",
            "        print(line, end=\"\", flush=True)\n",
            "        output_lines.append(line)\n",
            "    proc.wait()\n",
            "    if proc.returncode != 0:\n",
            "        err_tail = \"\".join(output_lines[-30:])\n",
            "        raise RuntimeError(f\"❌ {stage_title} 執行失敗 (exit={proc.returncode}):\\n{err_tail}\")\n",
            "\n",
            "run_stage(\"/content/temp/stage_3a.py\", \"階段 3A：OpenCV 去背與真人模特兒配戴首幀合成 (獨立子程序)\")\n",
            "\n",
            "if os.path.exists(\"/content/temp/keyframe_shot1.png\"):\n",
            "    display(IPImage(filename=\"/content/temp/keyframe_shot1.png\", width=220))\n",
            "    display(IPImage(filename=\"/content/temp/keyframe_shot2.png\", width=220))\n",
            "\n",
            "run_stage(\"/content/temp/stage_3b.py\", \"階段 3B：獨立文字特徵編碼 (T4 FP16 加速・算完即釋放 T5 記憶體)\")\n",
            "run_stage(\"/content/temp/stage_3c.py\", \"階段 3C：T4 GPU 輕量圖生影片擴散推論 (無文字編碼器、低 RAM 模式)\")\n",
            "\n",
            "# ---- 步驟 3 結果摘要：明確告知 AI 是否真的有讓人物動起來 ----\n",
            "_st_p = \"/content/temp/stage3c_status.json\"\n",
            "if os.path.exists(_st_p):\n",
            "    with open(_st_p, \"r\", encoding=\"utf-8\") as _f:\n",
            "        _st = json.load(_f)\n",
            "    if _st.get(\"ai_ok\"):\n",
            "        _m = \", \".join(f\"第{x['shot']}段 motion={x['motion']}\" for x in _st.get(\"shots\", []))\n",
            "        print(f\"\\n✅ AI 動態影片完成｜引擎：{_st.get('engine')}｜{_m}\")\n",
            "        print(\"   (motion ≥ 3 代表人物/鏡頭有明顯動作；低於 3 時已自動重試並疊加推軌運鏡)\")\n",
            "    else:\n",
            "        print(\"\\n⚠️ 注意：AI 影片模型未成功，本次使用『2.5D 視差運鏡』保底畫面 (人物不會有肢體動作)。\")\n",
            "        print(f\"   失敗原因：{_st.get('error')}\")\n",
            "        print(\"   👉 請把上方 [3C] 的錯誤訊息截圖給開發者，或改選其他方案 (1B Wan2.2 / 1C SVD-XT) 再試一次。\")\n",
            "else:\n",
            "    print(\"\\n✅ 步驟 3 動態鏡頭已產出 (方案 2：純產品 3D 運鏡)\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {
            "cellView": "form"
        },
        "outputs": [],
        "source": [
            "#@title 🎞️ 步驟 4：執行 ESPNER v5 精品後製管線 (6大風格 + 金框過場 + 正面原圖片尾合成 + 雲端備份與下載)\n",
            "import os, json, shutil, subprocess, cv2, numpy as np\n",
            "from PIL import Image, ImageDraw, ImageFont\n",
            "from IPython.display import display, HTML, Image as IPImage\n",
            "from base64 import b64encode\n",
            "from google.colab import files\n",
            "\n",
            "for _p, _step in [(\"/content/temp/job_config.json\", \"步驟 2\"), (\"/content/temp/raw_shot1.mp4\", \"步驟 3\"), (\"/content/temp/raw_shot2.mp4\", \"步驟 3\")]:\n",
            "    if not os.path.exists(_p):\n",
            "        raise RuntimeError(f\"❌ 找不到 {_p}，代表{_step}尚未成功完成 (或執行階段已重啟)。\\n👉 請依序執行步驟 1 → 2 → 3，全部成功後再執行步驟 4。\")\n",
            "\n",
            "with open(\"/content/temp/job_config.json\", \"r\", encoding=\"utf-8\") as f:\n",
            "    cfg = json.load(f)\n",
            "\n",
            "PRODUCT_PHOTO_PATH = cfg[\"photo_path\"]\n",
            "MODEL_NAME = cfg[\"model_name\"]\n",
            "style_key = cfg[\"commercial_style\"].split(\" \")[0]\n",
            "is_snow = \"SNOW_GOGGLES\" in cfg[\"product_category\"] or style_key == \"4_ALPINE_EXTREME\"\n",
            "shot1_mp4 = \"/content/temp/raw_shot1.mp4\"\n",
            "shot2_mp4 = \"/content/temp/raw_shot2.mp4\"\n",
            "\n",
            "def get_font_path(candidates):\n",
            "    for p in candidates:\n",
            "        if os.path.exists(p):\n",
            "            return p\n",
            "    return candidates[-1]\n",
            "\n",
            "FONT_SERIF_BOLD = get_font_path([\n",
            "    \"/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf\",\n",
            "    \"/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc\"\n",
            "])\n",
            "FONT_SERIF_REG = get_font_path([\n",
            "    \"/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf\",\n",
            "    \"/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc\"\n",
            "])\n",
            "FONT_MONO = get_font_path([\n",
            "    \"/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf\"\n",
            "])\n",
            "FONT_CHINESE = get_font_path([\n",
            "    \"/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc\",\n",
            "    \"/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc\",\n",
            "    \"/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf\"\n",
            "])\n",
            "\n",
            "def draw_tracked_text(draw, center_y, text, font, fill, width=720, tracking=4):\n",
            "    chars = list(text)\n",
            "    if not chars: return\n",
            "    char_widths = []\n",
            "    for c in chars:\n",
            "        bbox = font.getbbox(c)\n",
            "        char_widths.append(bbox[2] - bbox[0])\n",
            "    total_w = sum(char_widths) + (len(chars) - 1) * tracking\n",
            "    curr_x = (width - total_w) // 2\n",
            "    for i, c in enumerate(chars):\n",
            "        draw.text((curr_x, center_y), c, font=font, fill=fill)\n",
            "        curr_x += char_widths[i] + tracking\n",
            "\n",
            "def load_video_frames(path, max_frames=None, resize=(720, 1280)):\n",
            "    cap = cv2.VideoCapture(path)\n",
            "    frames = []\n",
            "    while True:\n",
            "        ret, f = cap.read()\n",
            "        if not ret: break\n",
            "        if max_frames and len(frames) >= max_frames: break\n",
            "        if resize and (f.shape[1] != resize[0] or f.shape[0] != resize[1]):\n",
            "            f = cv2.resize(f, resize)\n",
            "        frames.append(f)\n",
            "    cap.release()\n",
            "    return frames\n",
            "\n",
            "def build_madrid_wall_transition_frames(display_name, is_snow=False, duration_sec=2.0, fps=24):\n",
            "    num_frames = int(duration_sec * fps)\n",
            "    w, h = 720, 1280\n",
            "    wide_w = int(w * 1.35)\n",
            "    rng = np.random.RandomState(108)\n",
            "    noise = rng.normal(0, 3.5, (h, wide_w, 3)).astype(np.float32)\n",
            "    base_wall = np.full((h, wide_w, 3), (225, 235, 240), dtype=np.float32) + noise\n",
            "    wide_wall = cv2.GaussianBlur(np.clip(base_wall, 0, 255).astype(np.uint8), (9, 9), 0)\n",
            "\n",
            "    font_brand = ImageFont.truetype(FONT_SERIF_BOLD, 50)\n",
            "    font_model = ImageFont.truetype(FONT_SERIF_BOLD, 56 if len(display_name) > 10 else 62)\n",
            "    font_sub = ImageFont.truetype(FONT_SERIF_REG, 21)\n",
            "\n",
            "    INK_PRIMARY = (15, 23, 42)\n",
            "    INK_SECONDARY = (51, 65, 85)\n",
            "    INK_GOLD = (120, 80, 25)\n",
            "\n",
            "    frames = []\n",
            "    for idx in range(num_frames):\n",
            "        t = idx / max(1, num_frames - 1)\n",
            "        x_offset = int((wide_w - w) * (0.1 + 0.8 * t))\n",
            "        frame_bgr = wide_wall[:, x_offset:x_offset + w].copy()\n",
            "\n",
            "        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))\n",
            "        text_layer = Image.new(\"RGBA\", (w, h), (0, 0, 0, 0))\n",
            "        draw = ImageDraw.Draw(text_layer)\n",
            "\n",
            "        box_y1, box_y2 = h // 2 - 160, h // 2 + 150\n",
            "        box_x1, box_x2 = 70, w - 70\n",
            "        draw.rectangle([(box_x1, box_y1), (box_x2, box_y2)], outline=INK_GOLD + (160,), width=2)\n",
            "        draw.rectangle([(box_x1 + 6, box_y1 + 6), (box_x2 - 6, box_y2 - 6)], outline=INK_SECONDARY + (60,), width=1)\n",
            "\n",
            "        if t < 0.52:\n",
            "            alpha = int(255 * min(1.0, t / 0.15)) if t < 0.38 else int(255 * max(0.0, (0.52 - t) / 0.14))\n",
            "            draw_tracked_text(draw, h // 2 - 35, \"E S P N E R\", font_brand, INK_PRIMARY + (alpha,), width=w, tracking=6)\n",
            "        else:\n",
            "            alpha = int(255 * min(1.0, (t - 0.50) / 0.15))\n",
            "            draw_tracked_text(draw, h // 2 - 65, display_name.upper(), font_model, INK_PRIMARY + (alpha,), width=w, tracking=5)\n",
            "            sub1 = \"ESPNER ALPINE SNOW OPTICS\" if is_snow else \"ESPNER EYEWEAR ARCHIVE\"\n",
            "            sub2 = \"DUAL-SPHERICAL POLARIZED SERIES\" if is_snow else \"POLARIZED LUXURY COLLECTION\"\n",
            "            draw_tracked_text(draw, h // 2 + 18, sub1, font_sub, INK_GOLD + (alpha,), width=w, tracking=4)\n",
            "            draw_tracked_text(draw, h // 2 + 52, sub2, font_sub, INK_SECONDARY + (int(alpha * 0.8),), width=w, tracking=2)\n",
            "\n",
            "        pil_img = Image.alpha_composite(pil_img.convert(\"RGBA\"), text_layer)\n",
            "        out_bgr = cv2.cvtColor(np.array(pil_img.convert(\"RGB\")), cv2.COLOR_RGB2BGR)\n",
            "\n",
            "        if idx >= num_frames - 10:\n",
            "            fade_t = (idx - (num_frames - 10)) / 10.0\n",
            "            out_bgr = cv2.addWeighted(out_bgr, 1.0 - fade_t, np.full((h, w, 3), 255, dtype=np.uint8), fade_t, 0)\n",
            "        frames.append(out_bgr)\n",
            "    return frames\n",
            "\n",
            "def build_front_photo_endcard_frames(photo_p, display_name, is_snow=False, duration_sec=4.5, fps=24):\n",
            "    num_frames = int(duration_sec * fps)\n",
            "    w, h = 720, 1280\n",
            "\n",
            "    pil_in = Image.open(photo_p).convert(\"RGBA\")\n",
            "    white_bg = Image.new(\"RGBA\", pil_in.size, (255, 255, 255, 255))\n",
            "    merged = Image.alpha_composite(white_bg, pil_in).convert(\"RGB\")\n",
            "    photo = cv2.cvtColor(np.array(merged), cv2.COLOR_RGB2BGR)\n",
            "\n",
            "    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)\n",
            "    mask = gray < 248\n",
            "    pts = np.argwhere(mask)\n",
            "    if len(pts) > 0:\n",
            "        min_y, min_x = pts.min(axis=0)\n",
            "        max_y, max_x = pts.max(axis=0)\n",
            "        pad = 4\n",
            "        crop = photo[max(0, min_y - pad):min(photo.shape[0], max_y + pad),\n",
            "                     max(0, min_x - pad):min(photo.shape[1], max_x + pad)]\n",
            "    else:\n",
            "        crop = photo\n",
            "\n",
            "    target_base_w = 560\n",
            "    base_h = max(1, int(crop.shape[0] * (target_base_w / crop.shape[1])))\n",
            "    crop_base = cv2.resize(crop, (target_base_w, base_h), interpolation=cv2.INTER_LANCZOS4)\n",
            "\n",
            "    font_brand = ImageFont.truetype(FONT_SERIF_BOLD, 46)\n",
            "    font_model = ImageFont.truetype(FONT_SERIF_BOLD, 28)\n",
            "    font_slogan = ImageFont.truetype(FONT_CHINESE, 34)\n",
            "    font_sub = ImageFont.truetype(FONT_SERIF_REG, 19)\n",
            "\n",
            "    white_canvas = np.full((h, w, 3), 255, dtype=np.uint8)\n",
            "    endcard_frames = []\n",
            "    sub_label = \"DUAL-SPHERICAL POLARIZED SNOW GOGGLE\" if is_snow else \"POLARIZED LUXURY EYEWEAR\"\n",
            "\n",
            "    for i in range(num_frames):\n",
            "        t = i / max(1, num_frames - 1)\n",
            "        scale = 1.0 + 0.035 * t\n",
            "        cur_w = int(target_base_w * scale)\n",
            "        cur_h = int(base_h * scale)\n",
            "        scaled_crop = cv2.resize(crop_base, (cur_w, cur_h), interpolation=cv2.INTER_LANCZOS4)\n",
            "\n",
            "        canvas = white_canvas.copy()\n",
            "\n",
            "        sy = 665\n",
            "        shadow = np.zeros((h, w), dtype=np.uint8)\n",
            "        cv2.ellipse(shadow, (w // 2, sy), (int(cur_w * 0.42), int(22 * scale)), 0, 0, 360, 255, -1)\n",
            "        shadow = cv2.GaussianBlur(shadow, (71, 35), 0)\n",
            "        shadow_val = (shadow.astype(np.float32) / 255.0 * 0.12)[..., np.newaxis]\n",
            "        canvas = np.clip(canvas.astype(np.float32) * (1.0 - shadow_val), 0, 255).astype(np.uint8)\n",
            "\n",
            "        cy = 530\n",
            "        top_y = max(0, min(h - cur_h, cy - cur_h // 2))\n",
            "        left_x = max(0, min(w - cur_w, (w - cur_w) // 2))\n",
            "\n",
            "        crop_gray = cv2.cvtColor(scaled_crop, cv2.COLOR_BGR2GRAY)\n",
            "        crop_mask = np.clip((255.0 - crop_gray.astype(np.float32)) / 15.0, 0.0, 1.0)[..., np.newaxis]\n",
            "\n",
            "        roi = canvas[top_y:top_y+cur_h, left_x:left_x+cur_w].astype(np.float32)\n",
            "        blended = roi * (1.0 - crop_mask) + scaled_crop.astype(np.float32) * crop_mask\n",
            "        canvas[top_y:top_y+cur_h, left_x:left_x+cur_w] = np.clip(blended, 0, 255).astype(np.uint8)\n",
            "\n",
            "        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))\n",
            "        draw = ImageDraw.Draw(pil_img)\n",
            "\n",
            "        draw_tracked_text(draw, 145, \"E S P N E R\", font_brand, (15, 23, 42), width=w, tracking=8)\n",
            "        draw_tracked_text(draw, 210, display_name.upper(), font_model, (71, 85, 105), width=w, tracking=4)\n",
            "        draw_tracked_text(draw, 840, \"把目光，還給世界。\", font_slogan, (15, 23, 42), width=w, tracking=4)\n",
            "        draw_tracked_text(draw, 905, sub_label, font_sub, (100, 116, 139), width=w, tracking=3)\n",
            "\n",
            "        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)\n",
            "        if i < 10:\n",
            "            fade_in = i / 10.0\n",
            "            frame_bgr = cv2.addWeighted(frame_bgr, fade_in, white_canvas, 1.0 - fade_in, 0)\n",
            "        endcard_frames.append(frame_bgr)\n",
            "    return endcard_frames\n",
            "\n",
            "w, h = 720, 1280\n",
            "display_name = MODEL_NAME.strip().upper()\n",
            "\n",
            "if style_key == \"0_CLASSIC_MADRID\":\n",
            "    f1 = load_video_frames(shot1_mp4, max_frames=174)\n",
            "    f2 = load_video_frames(shot2_mp4, max_frames=186)\n",
            "    f_trans = build_madrid_wall_transition_frames(display_name, is_snow=is_snow, duration_sec=2.0, fps=24)\n",
            "    f_end = build_front_photo_endcard_frames(PRODUCT_PHOTO_PATH, display_name, is_snow=is_snow, duration_sec=4.5, fps=24)\n",
            "    all_frames = f1 + f2 + f_trans + f_end\n",
            "elif style_key == \"1_CYBERPUNK_NEON\":\n",
            "    f1 = load_video_frames(shot1_mp4, max_frames=192)\n",
            "    f2 = load_video_frames(shot2_mp4, max_frames=192)\n",
            "    story_raw = f1 + f2\n",
            "    font_hud = ImageFont.truetype(FONT_MONO, 17)\n",
            "    processed = []\n",
            "    for idx, frame in enumerate(story_raw):\n",
            "        img = frame.astype(np.float32) / 255.0\n",
            "        img = 1.0 / (1.0 + np.exp(-9.0 * (img - 0.48)))\n",
            "        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)\n",
            "        if (188 <= idx <= 196) or (idx % 68 in [0, 1]):\n",
            "            b, g, r = cv2.split(img)\n",
            "            img = cv2.merge([np.roll(b, 10, axis=1), g, np.roll(r, -10, axis=1)])\n",
            "        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))\n",
            "        ov = Image.new(\"RGBA\", (w, h), (0, 0, 0, 0))\n",
            "        dr = ImageDraw.Draw(ov)\n",
            "        dr.text((40, 48), \"[SYS: OPTICAL HUD v5.1]\", font=font_hud, fill=(0, 240, 255, 230))\n",
            "        dr.text((40, 72), f\"T+{idx/24.0:05.2f}s // {display_name}\", font=font_hud, fill=(200, 225, 255, 200))\n",
            "        dr.text((40, 96), \"AERODYNAMIC POLARIZED\", font=font_hud, fill=(255, 0, 128, 230))\n",
            "        pil_img = Image.alpha_composite(pil_img.convert(\"RGBA\"), ov)\n",
            "        out_bgr = cv2.cvtColor(np.array(pil_img.convert(\"RGB\")), cv2.COLOR_RGB2BGR)\n",
            "        if idx >= len(story_raw) - 10:\n",
            "            fw = (idx - (len(story_raw) - 10)) / 10.0\n",
            "            out_bgr = cv2.addWeighted(out_bgr, 1.0 - fw, np.full((h, w, 3), 255, dtype=np.uint8), fw, 0)\n",
            "        processed.append(out_bgr)\n",
            "    f_end = build_front_photo_endcard_frames(PRODUCT_PHOTO_PATH, display_name, is_snow=is_snow, duration_sec=5.0, fps=24)\n",
            "    all_frames = processed + f_end\n",
            "elif style_key == \"3_HAUTE_COUTURE_NOIR\":\n",
            "    f1 = load_video_frames(shot1_mp4, max_frames=192)\n",
            "    f2 = load_video_frames(shot2_mp4, max_frames=192)\n",
            "    story_raw = f1 + f2\n",
            "    font_ed = ImageFont.truetype(FONT_SERIF_REG, 17)\n",
            "    processed = []\n",
            "    for idx, frame in enumerate(story_raw):\n",
            "        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)\n",
            "        noir_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)\n",
            "        pil_img = Image.fromarray(cv2.cvtColor(noir_bgr, cv2.COLOR_BGR2RGB))\n",
            "        ov = Image.new(\"RGBA\", (w, h), (0, 0, 0, 0))\n",
            "        dr = ImageDraw.Draw(ov)\n",
            "        m = 32\n",
            "        dr.rectangle([m, m, w - m, h - m], outline=(218, 185, 105, 200), width=1)\n",
            "        dr.text((m + 20, m + 18), \"PARIS • MILANO • ESPNER\", font=font_ed, fill=(218, 185, 105, 240))\n",
            "        dr.text((w - m - 180, m + 18), f\"EDITION {display_name}\", font=font_ed, fill=(218, 185, 105, 240))\n",
            "        pil_img = Image.alpha_composite(pil_img.convert(\"RGBA\"), ov)\n",
            "        out_bgr = cv2.cvtColor(np.array(pil_img.convert(\"RGB\")), cv2.COLOR_RGB2BGR)\n",
            "        if idx >= len(story_raw) - 10:\n",
            "            fw = (idx - (len(story_raw) - 10)) / 10.0\n",
            "            out_bgr = cv2.addWeighted(out_bgr, 1.0 - fw, np.full((h, w, 3), 255, dtype=np.uint8), fw, 0)\n",
            "        processed.append(out_bgr)\n",
            "    f_end = build_front_photo_endcard_frames(PRODUCT_PHOTO_PATH, display_name, is_snow=is_snow, duration_sec=5.0, fps=24)\n",
            "    all_frames = processed + f_end\n",
            "else:\n",
            "    f1 = load_video_frames(shot1_mp4, max_frames=192)\n",
            "    f2 = load_video_frames(shot2_mp4, max_frames=192)\n",
            "    story_raw = f1 + f2\n",
            "    for idx in range(max(0, len(story_raw) - 10), len(story_raw)):\n",
            "        fw = (idx - (len(story_raw) - 10)) / 10.0\n",
            "        story_raw[idx] = cv2.addWeighted(story_raw[idx], 1.0 - fw, np.full((h, w, 3), 255, dtype=np.uint8), fw, 0)\n",
            "    f_end = build_front_photo_endcard_frames(PRODUCT_PHOTO_PATH, display_name, is_snow=is_snow, duration_sec=5.0, fps=24)\n",
            "    all_frames = story_raw + f_end\n",
            "\n",
            "safe_name = display_name.replace(\" \", \"_\")\n",
            "final_mp4 = f\"/content/output/ESPNER_{safe_name}_commercial_v5.mp4\"\n",
            "raw_out = \"/content/temp/final_raw.mp4\"\n",
            "\n",
            "writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*'mp4v'), 24.0, (w, h))\n",
            "for f in all_frames:\n",
            "    writer.write(f)\n",
            "writer.release()\n",
            "\n",
            "subprocess.run([\n",
            "    \"ffmpeg\", \"-y\", \"-i\", raw_out,\n",
            "    \"-c:v\", \"libx264\", \"-crf\", \"16\", \"-preset\", \"medium\", \"-pix_fmt\", \"yuv420p\", \"-movflags\", \"+faststart\",\n",
            "    final_mp4\n",
            "], check=True)\n",
            "\n",
            "sheet_jpg = f\"/content/output/ESPNER_{safe_name}_QA_sheet.jpg\"\n",
            "indices = np.linspace(0, len(all_frames) - 1, 12, dtype=int)\n",
            "tiles = []\n",
            "for idx in indices:\n",
            "    t_img = cv2.resize(all_frames[idx], (240, 426))\n",
            "    cv2.putText(t_img, f\"f:{idx} ({idx/24.0:.1f}s)\", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3)\n",
            "    cv2.putText(t_img, f\"f:{idx} ({idx/24.0:.1f}s)\", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1)\n",
            "    tiles.append(t_img)\n",
            "sheet = np.vstack([np.hstack(tiles[r*4:(r+1)*4]) for r in range(3)])\n",
            "cv2.imwrite(sheet_jpg, sheet)\n",
            "\n",
            "drive_out_dir = \"/content/drive/MyDrive/ESPNER_Colab_T4_Output\"\n",
            "if os.path.isdir(\"/content/drive/MyDrive\"):\n",
            "    os.makedirs(drive_out_dir, exist_ok=True)\n",
            "    shutil.copy2(final_mp4, os.path.join(drive_out_dir, os.path.basename(final_mp4)))\n",
            "    shutil.copy2(sheet_jpg, os.path.join(drive_out_dir, os.path.basename(sheet_jpg)))\n",
            "    print(f\"☁️ 已自動同步備份至 Google 雲端硬碟：{drive_out_dir}\")\n",
            "\n",
            "dur_sec = len(all_frames) / 24.0\n",
            "size_mb = os.path.getsize(final_mp4) / (1024 * 1024)\n",
            "print(f\"\\n🎉 廣告影片運算完成！\")\n",
            "print(f\"📁 檔案路徑：{final_mp4} ({dur_sec:.2f} 秒, {size_mb:.2f} MB)\")\n",
            "display(IPImage(filename=sheet_jpg, width=640))\n",
            "\n",
            "mp4_bytes = open(final_mp4, 'rb').read()\n",
            "data_url = \"data:video/mp4;base64,\" + b64encode(mp4_bytes).decode()\n",
            "display(HTML(f'<video width=360 controls autoplay loop><source src=\"{data_url}\" type=\"video/mp4\"></video>'))\n",
            "files.download(final_mp4)"
        ]
    }
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {
            "name": "ESPNER_Colab_T4_省記憶體版_v5.1.ipynb",
            "provenance": [],
            "gpuType": "T4"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": cells
}

for p in PATHS_TO_WRITE:
    if os.path.isdir(os.path.dirname(p)):
        with open(p, "w", encoding="utf-8") as f:
            json.dump(notebook, f, ensure_ascii=False, indent=2)
        print(f"Wrote: {p}")

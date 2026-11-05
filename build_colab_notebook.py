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

SHOTPLAN_CODE = r"""
# ESPNER shot plan：所有階段 (3A 肖像 / 3B 動態提示詞 / 3C 影片 / 步驟 4 剪輯) 共用的唯一分鏡來源
import re

NEG_PORTRAIT = "sunglasses, glasses, eyewear, goggles, deformed face, asymmetrical eyes, bad anatomy, extra fingers, blurry, watermark, text, logo"

# ---- 參考片標準 30 秒分鏡 (4 鏡 + 片尾) ----
# wear=True 的鏡頭：肖像需正面面向鏡頭，才能精準合成真實眼鏡；動作在影片階段才發生
STYLE_PLANS = {
    "A_STORY_30S": {
        "subject": "a young East Asian woman with a sleek black bob haircut with bangs, natural makeup, wearing a white linen shirt",
        "look": "warm golden hour light, soft film look, natural skin texture, sharp focus, 35mm photo, photorealistic",
        "endcard": "split", "endcard_dur": 6.0, "brand_mark": "top-left", "mood": "warm", "grade": "A_warm_film",
        "copy_default": ["你有多久，/沒好好看看世界？", "今天，/為自己出發。", "把目光，/還給世界。"],
        # 8 鏡 x 3 秒：短鏡頭停在模型品質最好的區段，也符合參考片 A 每 2~5 秒一刀的剪輯節奏
        "shots": [
            {"framing": "Medium shot", "scene": "sitting at an old wooden desk beside a large sunlit window in a cozy creative studio apartment, city rooftops outside the window", "wear": False, "closeup": False, "dur": 3.0, "copy": 0,
             "motion": "She sits calmly at the desk and slowly turns her head toward the window light, warm sunlight flickers softly, the camera slowly dollies in, subtle natural motion, stable framing, cinematic."},
            {"framing": "Close-up portrait", "scene": "beside a sunlit window, warm soft window light on her face, blurred city outside", "wear": False, "closeup": True, "dur": 3.0, "copy": None,
             "motion": "Close-up: she slowly closes and opens her eyes, a gentle breath, warm window light shimmers softly on her face, the camera stays still with a very slow push-in, subtle natural motion, cinematic."},
            {"framing": "Medium shot", "scene": "standing in a quiet old alley beside a weathered white wall, warm afternoon sun, green plants", "wear": True, "closeup": False, "dur": 3.0, "copy": 1,
             "motion": "She stands relaxed by the sunlit wall facing the camera and smiles softly, hair moves gently in the breeze, the camera slowly pushes in, subtle natural motion, stable framing, cinematic."},
            {"framing": "Close-up portrait", "scene": "beside a weathered white wall in warm afternoon sun", "wear": True, "closeup": True, "dur": 3.0, "copy": None,
             "motion": "Close-up: she faces the camera with a calm confident expression and a slight smile, a few strands of hair move in the breeze, the camera pushes in very slowly, subtle natural motion, stable framing, cinematic."},
            {"framing": "Medium shot from the waist up", "scene": "in a quiet sunlit old alley with green plants and a white wall", "wear": True, "closeup": False, "dur": 3.0, "copy": None,
             "motion": "She stands in the sunlit alley facing the camera, her shirt and hair move gently in the breeze, warm sunlight, the camera slowly arcs to the side, subtle natural motion, stable framing, cinematic."},
            {"framing": "Medium shot", "scene": "standing on a grassy seaside path at golden sunset, calm sea and low sun behind her", "wear": True, "closeup": False, "dur": 3.0, "copy": None,
             "motion": "She stands on the seaside path facing the camera, the sea breeze moves her hair and shirt gently, warm sunset backlight shimmers on the water, the camera slowly dollies in, subtle natural motion, stable framing, cinematic."},
            {"framing": "Close-up portrait", "scene": "at the seaside during golden sunset, warm backlight, calm sea behind", "wear": True, "closeup": True, "dur": 3.0, "copy": 2,
             "motion": "Close-up: she looks into the camera with a peaceful expression, warm sunset rim light on her hair, hair moves slightly in the sea breeze, the camera pushes in very slowly, subtle natural motion, stable framing, cinematic."},
            {"framing": "Medium shot", "scene": "on a grassy seaside path at golden sunset, calm sea behind", "wear": True, "closeup": False, "dur": 3.0, "copy": None,
             "motion": "She stands facing the camera at sunset with a soft smile, hair and shirt flutter gently in the sea breeze, the camera slowly pulls back, subtle natural motion, stable framing, cinematic."},
        ],
    },
    "B_COASTAL_30S": {
        "subject": "a handsome Mediterranean man in his thirties with short dark hair and light stubble, wearing a cream white linen shirt",
        "look": "bright natural midday sunlight, crisp clean colors, sharp focus, luxury fashion campaign, photorealistic",
        "endcard": "studio", "endcard_dur": 10.0, "brand_mark": "top-center", "mood": "airy", "grade": "B_bright_natural",
        "shots": [
            {"framing": "Extreme close-up portrait", "scene": "on a sunny Mediterranean coastal cliff, deep blue sea and clear sky behind", "wear": True, "closeup": True, "dur": 3.0, "copy": None,
             "motion": "Extreme close-up: he faces the camera with a calm confident look, the mirrored lenses reflect the sea and sky, a gentle breeze moves his hair slightly, the camera slowly pulls back, subtle natural motion, stable framing, cinematic."},
            {"framing": "Medium shot", "scene": "standing on a sunlit limestone coastal terrace, bright blue sea and horizon behind", "wear": True, "closeup": False, "dur": 4.0, "copy": None,
             "motion": "He stands relaxed facing the camera on the sunlit terrace, the linen shirt ripples gently in the wind, sparkling sea behind him, the camera slowly orbits a little to the side, subtle natural motion, stable framing, cinematic."},
            {"framing": "Medium shot from the waist up", "scene": "on a sunlit limestone coastal path, bright blue sea behind", "wear": True, "closeup": False, "dur": 3.0, "copy": None,
             "motion": "He stands on the coastal path facing the camera, the shirt moves gently in the wind, waves roll softly behind him, the camera slowly dollies in, subtle natural motion, stable framing, cinematic."},
            {"framing": "Close-up portrait", "scene": "on a sunny Mediterranean coast, bright sea behind", "wear": True, "closeup": True, "dur": 3.0, "copy": None,
             "motion": "Close-up: he looks into the camera with a confident look, the mirrored lenses reflect the bright sea, a light breeze moves his hair, the camera pushes in slowly, subtle natural motion, stable framing, cinematic."},
            {"framing": "Medium shot", "scene": "leaning on a white stone railing above the Mediterranean sea in bright sun", "wear": True, "closeup": False, "dur": 4.0, "copy": None,
             "motion": "He leans relaxed on the stone railing facing the camera, the shirt moves in the sea wind, bright sunlight glints on the sea, the camera slowly arcs around him, subtle natural motion, stable framing, cinematic."},
            {"framing": "Close-up portrait", "scene": "on a sunny coastal cliff, deep blue sea and sky behind", "wear": True, "closeup": True, "dur": 3.0, "copy": None,
             "motion": "Hero shot: he gives a subtle confident smile into the camera, the mirrored lenses reflect the bright sea, the camera pushes in slowly, subtle natural motion, stable framing, cinematic."},
        ],
    },}

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


def is_reference_style(style_key):
    return style_key in STYLE_PLANS

def _clean_scene(text):
    # 移除墨鏡字眼：肖像必須不戴眼鏡，之後才合成 100% 真實產品
    return re.sub(r"(?i)\b(wearing|with|took off|take off)?\s*(his|her|the)?\s*(aviator|sunglasses|glasses|eyewear|goggles|shades)\b", "", text).strip(" ,.")

def resolve_shots(cfg, custom_en=""):
    # 回傳 [{id, portrait, motion, wear, closeup, dur, copy}]
    style_key = cfg["commercial_style"].split(" ")[0]
    custom_en = (custom_en or "").strip()
    shots = []
    if style_key in STYLE_PLANS:
        plan = STYLE_PLANS[style_key]
        for i, sh in enumerate(plan["shots"], 1):
            scene = _clean_scene(custom_en) if custom_en else sh["scene"]
            face = "facing the camera directly, head level, clear unobstructed face and eyes" if sh["wear"] else "natural candid pose"
            portrait = f"{sh['framing']} of {plan['subject']}, {scene}, {face}, {plan['look']}"
            motion = sh["motion"] if not custom_en else f"{sh['motion']} Setting: {custom_en}."
            if sh["wear"]:
                motion = motion + " " + ID_LOCK
            shots.append({"id": i, "portrait": portrait, "motion": motion, "wear": sh["wear"],
                          "closeup": sh["closeup"], "dur": sh["dur"], "copy": sh["copy"]})
        return shots
    # 舊版 6 種風格：2 鏡 x 8 秒
    p1, p2 = PORTRAIT_PROMPTS.get(style_key, PORTRAIT_PROMPTS["0_CLASSIC_MADRID"])
    m1, m2 = MOTION_PROMPTS.get(style_key, MOTION_PROMPTS["0_CLASSIC_MADRID"])
    if custom_en:
        cs = _clean_scene(custom_en)
        p1 = f"Editorial medium portrait of a charismatic celebrity model ({cs}), facing camera directly, head level, clear unobstructed face and eyes, cinematic lighting, 85mm lens, photorealistic, 8k"
        p2 = f"Editorial close-up portrait of a charismatic celebrity model ({cs}), facing camera directly, head level, clear unobstructed face and eyes, dramatic flash and golden lighting, 85mm lens, photorealistic, 8k"
        m1 = custom_en + ". The model moves naturally, turns the head toward the camera, hair and clothes move, lights flicker in the background, the camera tracks smoothly, cinematic luxury commercial."
        m2 = "Close-up. " + custom_en + ". The model turns the face toward the camera with a confident expression, light sweeps across the lenses, the camera pushes in slowly, cinematic."
    return [
        {"id": 1, "portrait": p1, "motion": m1 + " " + ID_LOCK, "wear": True, "closeup": False, "dur": 8.0, "copy": None},
        {"id": 2, "portrait": p2, "motion": m2 + " " + ID_LOCK, "wear": True, "closeup": True, "dur": 8.0, "copy": None},
    ]

def output_size(cfg):
    # (寬, 高)：9:16 = 720x1280；16:9 = 1920x1080
    return (1920, 1080) if cfg.get("aspect", "9:16") == "16:9" else (720, 1280)
"""

TRACK_CODE = r"""
# espner_track.py - keep the product's REAL glasses on the face through a generated clip.
# Deps: numpy, opencv-python-headless (<5). No triple double quotes in this file on purpose.
#
# Method: sparse LK tracking (half resolution) of face/hair features AROUND the glasses
# (glasses area masked out), per-frame similarity (frame0 -> frame i) fitted with RANSAC
# from points that carry their frame-0 coordinates (no matrix-product drift), periodic
# direct re-anchoring to frame 0, zero-phase One-Euro smoothing, confidence-driven fade,
# premultiplied-alpha warp + soft contact shadow composite.
import time
import numpy as np
import cv2

cv2.setNumThreads(1)

_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
_LK_WIDE = dict(winSize=(31, 31), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def _apply(M, pts):
    # M: 2x3, pts: Nx2
    return pts @ M[:, :2].T + M[:, 2]


def _inv(M):
    return cv2.invertAffineTransform(M)


def _params_from_M(M, c0):
    A = M[:, :2]
    s = float(np.sqrt(abs(np.linalg.det(A))) + 1e-9)
    ang = float(np.arctan2(A[1, 0], A[0, 0]))
    c = _apply(M, np.asarray(c0, np.float64)[None])[0]
    return np.array([c[0], c[1], np.log(s), ang])


def _M_from_params(p, c0):
    s = np.exp(p[2])
    ca, sa = np.cos(p[3]) * s, np.sin(p[3]) * s
    A = np.array([[ca, -sa], [sa, ca]])
    c0 = np.asarray(c0, np.float64)
    t = np.array([p[0], p[1]]) - A @ c0
    return np.hstack([A, t[:, None]])


class _OneEuro:
    def __init__(self, fps, min_cutoff, beta, dcutoff=1.0):
        self.fps, self.mc, self.beta, self.dc = fps, min_cutoff, beta, dcutoff

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * self.fps)

    def run(self, x):
        # x: N x D (D channels in px-equivalent units)
        out = np.empty_like(x)
        out[0] = x[0]
        dx_prev = np.zeros(x.shape[1])
        for i in range(1, len(x)):
            dx = (x[i] - out[i - 1]) * self.fps
            a_d = self._alpha(self.dc)
            dx_hat = a_d * dx + (1 - a_d) * dx_prev
            dx_prev = dx_hat
            cutoff = self.mc + self.beta * np.abs(dx_hat)
            a = np.array([self._alpha(c) for c in cutoff])
            out[i] = a * x[i] + (1 - a) * out[i - 1]
        return out


def _smooth_params(P, R, fps, min_cutoff=2.0, beta=0.05):
    # zero-phase: filter forward and backward, average. Work in px-equivalent units.
    X = P.copy()
    X[:, 2] *= R
    X[:, 3] *= R
    f = _OneEuro(fps, min_cutoff, beta)
    a = f.run(X)
    b = f.run(X[::-1].copy())[::-1]
    Y = 0.5 * (a + b)
    Y[:, 2] /= R
    Y[:, 3] /= R
    return Y


def _roi_polys(c0, gw, gh, face_box, shape_full):
    # returns (roi_ellipse_params, glasses_exclusion_rect) in frame-0 full-res coords
    H, W = shape_full
    if face_box is not None:
        fx, fy, fw, fh = face_box
        cx, cy = fx + fw / 2.0, fy + fh / 2.0
        rw, rh = fw * 0.75, fh * 0.75
    else:
        cx, cy = c0[0], c0[1] + gh * 0.6
        rw, rh = gw * 1.7, gw * 2.0
    ex = 0.22 * gw
    exclude = (c0[0] - gw / 2 - ex, c0[1] - gh / 2 - ex * 0.8,
               c0[0] + gw / 2 + ex, c0[1] + gh / 2 + ex * 0.8)
    return (cx, cy, rw, rh), exclude


def _make_mask(M_full, roi, excl, shape_half, s):
    # draw ROI ellipse and glasses exclusion at the current pose, in half-res coords
    h, w = shape_half
    cx, cy, rw, rh = roi
    th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    ell = np.stack([cx + rw * np.cos(th), cy + rh * np.sin(th)], 1)
    x0, y0, x1, y1 = excl
    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
    m = np.zeros((h, w), np.uint8)
    e = (_apply(M_full, ell) * s).astype(np.int32)
    cv2.fillConvexPoly(m, e, 255)
    b = (_apply(M_full, box) * s).astype(np.int32)
    cv2.fillConvexPoly(m, b, 0)
    return m


def _sharp(gray, mask):
    lap = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_16S, ksize=3)
    v = np.abs(lap)
    sel = mask > 0
    return float(v[sel].mean()) + 1e-6 if sel.any() else 1e-6


def _fit(p0, p1, thr=2.0):
    if len(p0) < 4:
        return None, None
    M, inl = cv2.estimateAffinePartial2D(p0.reshape(-1, 1, 2), p1.reshape(-1, 1, 2),
                                         method=cv2.RANSAC, ransacReprojThreshold=thr,
                                         maxIters=400, confidence=0.99, refineIters=10)
    if M is None:
        return None, None
    return M.astype(np.float64), inl.ravel().astype(bool)


def _detect(gray, mask, n, existing=None):
    m = mask.copy()
    if existing is not None and len(existing):
        for p in existing:
            cv2.circle(m, (int(p[0]), int(p[1])), 5, 0, -1)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=n, qualityLevel=0.01, minDistance=6,
                                  mask=m, blockSize=5)
    if pts is None:
        return np.zeros((0, 2), np.float32)
    return pts.reshape(-1, 2).astype(np.float32)


def _lk_fb(g_a, g_b, pa, fb_thr, init=None, wide=False):
    kw = _LK_WIDE if wide else _LK
    if init is not None:
        pb, st, _ = cv2.calcOpticalFlowPyrLK(g_a, g_b, pa.reshape(-1, 1, 2), init.reshape(-1, 1, 2),
                                             flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **kw)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(g_b, g_a, pb, pa.reshape(-1, 1, 2).copy(),
                                                flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **kw)
    else:
        pb, st, _ = cv2.calcOpticalFlowPyrLK(g_a, g_b, pa.reshape(-1, 1, 2), None, **kw)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(g_b, g_a, pb, None, **kw)
    pb = pb.reshape(-1, 2)
    back = back.reshape(-1, 2)
    ok = (st.ravel() == 1) & (st2.ravel() == 1)
    ok &= np.linalg.norm(back - pa, axis=1) < fb_thr
    return pb, ok


def _cleanup_patch(bg8, a_full, ring_r):
    # replace bright melted residue in a ring around the real glasses by inpainted skin/hair.
    # all mask work at half res for speed; blend back at full res.
    h, w = bg8.shape[:2]
    hw, hh = max(2, w // 2), max(2, h // 2)
    sm = cv2.resize(bg8, (hw, hh), interpolation=cv2.INTER_AREA)
    a = cv2.resize(a_full, (hw, hh), interpolation=cv2.INTER_AREA)
    k = max(1, int(ring_r * 0.5))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    core = (a > 0.3).astype(np.uint8)
    near = cv2.dilate(core, ker)
    ring = (near > 0) & (a <= 0.5)
    k2 = max(2, int(ring_r * 0.9))
    far = cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k2 + 1, 2 * k2 + 1)))
    band = (far > 0) & (near == 0)
    if band.sum() < 30:
        return bg8
    lum = cv2.GaussianBlur(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    ref = float(np.percentile(lum[band], 60))
    anom = ring & (lum > ref + 22)
    if anom.sum() < 10:
        return bg8
    anom = (cv2.dilate(anom.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ring
    hm = ((anom | (core > 0)).astype(np.uint8)) * 255
    qs = cv2.resize(sm, (max(2, hw // 2), max(2, hh // 2)), interpolation=cv2.INTER_AREA)
    qm = cv2.resize(hm, (qs.shape[1], qs.shape[0]), interpolation=cv2.INTER_NEAREST)
    fill = cv2.inpaint(qs, qm, 3, cv2.INPAINT_TELEA)
    wt = np.minimum(1.0, cv2.GaussianBlur(anom.astype(np.float32), (0, 0), 1.3) * 1.5)
    fill = cv2.GaussianBlur(cv2.resize(fill, (w, h), interpolation=cv2.INTER_LINEAR), (0, 0), 1.5)
    wt = cv2.resize(wt, (w, h), interpolation=cv2.INTER_LINEAR)[:, :, None]
    return np.clip(bg8 * (1 - wt) + fill * wt + 0.5, 0, 255).astype(np.uint8)


def _composite_patch(frame, pm4, M_tot, opacity, shadow_sigma, shadow_dy, shadow_op, corners,
                     ring_r=0):
    H, W = frame.shape[:2]
    mg = int(3 * shadow_sigma + shadow_dy + 3 + ring_r * 1.8 + 3)
    xs, ys = corners[:, 0], corners[:, 1]
    x0 = max(int(np.floor(xs.min())) - mg, 0)
    y0 = max(int(np.floor(ys.min())) - mg, 0)
    x1 = min(int(np.ceil(xs.max())) + mg, W)
    y1 = min(int(np.ceil(ys.max())) + mg, H)
    if x1 <= x0 or y1 <= y0:
        return frame
    Mp = M_tot.copy()
    Mp[0, 2] -= x0
    Mp[1, 2] -= y0
    pw = cv2.warpAffine(pm4, Mp, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = pw[:, :, 3]
    sh = cv2.GaussianBlur(a, (0, 0), shadow_sigma)
    dy = int(round(shadow_dy))
    if dy > 0:
        sh = np.concatenate([np.zeros((dy, sh.shape[1]), np.float32), sh[:-dy]], 0)
    out = frame.copy()
    if ring_r > 0:
        out[y0:y1, x0:x1] = _cleanup_patch(out[y0:y1, x0:x1], a, ring_r)
    bg = out[y0:y1, x0:x1].astype(np.float32)
    bg *= (1.0 - shadow_op * opacity * np.clip(sh, 0, 1))[:, :, None]
    bg = pw[:, :, :3] * opacity + bg * (1.0 - a * opacity)[:, :, None]
    out[y0:y1, x0:x1] = np.clip(bg + 0.5, 0, 255).astype(np.uint8)
    return out


def recomposite_glasses(frames, glasses_rgba, paste_x, paste_y, face_box=None,
                        min_conf=0.35, debug=False, cleanup=True):
    t_start = time.time()
    n = len(frames)
    H, W = frames[0].shape[:2]
    gh, gw = glasses_rgba.shape[:2]
    fps = 24.0
    s = 0.5
    c0 = np.array([paste_x + gw / 2.0, paste_y + gh / 2.0])
    R = gw / 2.0

    # ---------------- tracking (half res) ----------------
    def to_gray(f):
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    roi, excl = _roi_polys(c0, gw, gh, face_box, (H, W))
    g_prev = to_gray(frames[0])
    g0 = g_prev
    hh, ww = g0.shape
    I = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    mask0 = _make_mask(I, roi, excl, (hh, ww), s)
    sharp0 = _sharp(g0, mask0)
    anchor_pts = _detect(g0, mask0, 200)
    p0 = anchor_pts.copy()          # frame-0 coords of live tracks (half res)
    p_prev = anchor_pts.copy()      # their positions in previous frame
    Ms_full = [I.copy()]            # frame0 -> frame i, full-res coords (raw)
    Ms_half = I.copy()
    conf = np.zeros(n)
    conf[0] = 1.0
    ninl = np.zeros(n, int)
    reproj = np.zeros(n)
    ANCHOR_EVERY = 8
    last_good_i = 0
    lost = 0 if len(p0) >= 8 else 1

    for i in range(1, n):
        g = to_gray(frames[i])
        M_h = None
        inl_mask = None
        n_in, ratio, err = 0, 0.0, 9.0
        if len(p_prev) >= 4:
            pb, ok = _lk_fb(g_prev, g, p_prev, 1.0)
            if ok.sum() >= 4:
                M_h, inl = _fit(p0[ok], pb[ok])
                if M_h is not None:
                    idx = np.where(ok)[0]
                    keep = np.zeros(len(p0), bool)
                    keep[idx[inl]] = True
                    n_in = int(inl.sum())
                    ratio = n_in / max(1, int(ok.sum()))
                    res = np.linalg.norm(_apply(M_h, p0[ok][inl]) - pb[ok][inl], axis=1)
                    err = float(np.median(res)) if len(res) else 9.0
                    p0 = p0[keep]
                    p_prev = pb[keep]
        # periodic / recovery direct re-anchor to frame 0
        need_anchor = (i % ANCHOR_EVERY == 0) or M_h is None or n_in < 12
        if need_anchor and len(anchor_pts) >= 8:
            pred = _apply(Ms_half, anchor_pts.astype(np.float64)).astype(np.float32)
            pb2, ok2 = _lk_fb(g0, g, anchor_pts, 1.5, init=pred, wide=True)
            # reject matches far from prediction (wrong lock)
            ok2 &= np.linalg.norm(pb2 - pred, axis=1) < 25 * s * 2
            if ok2.sum() >= 8:
                M2, inl2 = _fit(anchor_pts[ok2], pb2[ok2])
                if M2 is not None and inl2.sum() >= max(10, n_in * 0.8 if M_h is not None else 10):
                    n2 = int(inl2.sum())
                    r2 = n2 / int(ok2.sum())
                    if r2 >= 0.5 or M_h is None:
                        M_h = M2
                        idx = np.where(ok2)[0][inl2]
                        p0 = anchor_pts[idx].copy()
                        p_prev = pb2[idx].copy()
                        n_in, ratio = n2, r2
                        res = np.linalg.norm(_apply(M2, p0) - p_prev, axis=1)
                        err = float(np.median(res))
        # confidence
        if M_h is None:
            c = 0.0
        else:
            A = M_h[:, :2]
            sc = np.sqrt(abs(np.linalg.det(A)))
            sane = 0.6 < sc < 1.8 and abs(np.arctan2(A[1, 0], A[0, 0])) < np.radians(45)
            c_n = np.clip(n_in / 20.0, 0, 1)
            c_r = np.clip(ratio / 0.7, 0, 1)
            c_e = np.clip(1.0 - err / 2.5, 0, 1)
            M_full_try = M_h.copy()
            M_full_try[:, 2] /= s
            m_try = _make_mask(M_full_try, roi, excl, (hh, ww), s)
            sh_r = _sharp(g, m_try) / sharp0
            c_s = np.clip((sh_r - 0.25) / 0.35, 0, 1)
            c = float(c_n * c_r * c_e * c_s) if sane else 0.0
            ninl[i], reproj[i] = n_in, err
        conf[i] = c
        if M_h is not None and c >= 0.15:
            Ms_half = M_h
            last_good_i = i
        # keep track set healthy (only when pose is trustworthy)
        if c >= 0.5 and len(p_prev) < 70:
            M_full_c = Ms_half.copy()
            M_full_c[:, 2] /= s
            mk = _make_mask(M_full_c, roi, excl, (hh, ww), s)
            new = _detect(g, mk, 150 - len(p_prev), p_prev)
            if len(new):
                Mi = _inv(Ms_half)
                p0 = np.vstack([p0, _apply(Mi, new.astype(np.float64)).astype(np.float32)])
                p_prev = np.vstack([p_prev, new])
        Mf = Ms_half.copy()
        Mf[:, 2] /= s
        Ms_full.append(Mf)
        g_prev = g
        if len(p_prev) < 4 and not (len(anchor_pts) >= 8):
            pass

    # ---------------- smoothing (zero-phase One-Euro on px-equivalent params) ----------------
    P = np.array([_params_from_M(M, c0) for M in Ms_full])
    P[:, 3] = np.unwrap(P[:, 3])
    Ps = _smooth_params(P, R, fps)
    Ms = [_M_from_params(p, c0) for p in Ps]
    raw_c = P[:, :2]
    sm_c = Ps[:, :2]
    d2 = np.linalg.norm(sm_c[2:] - 2 * sm_c[1:-1] + sm_c[:-2], axis=1) if n > 2 else np.zeros(1)
    max_jitter = float(d2.max()) if len(d2) else 0.0

    # ---------------- opacity (fade over ~6 frames, hysteresis) ----------------
    opac = np.zeros(n)
    step = 1.0 / 6.0
    on = True
    cur = 1.0
    for i in range(n):
        if on and conf[i] < min_conf:
            on = False
        elif (not on) and conf[i] >= min_conf + 0.1:
            on = True
        tgt = 1.0 if on else 0.0
        cur = min(1.0, cur + step) if tgt >= 0.5 else max(0.0, cur - step)
        opac[i] = cur

    # ---------------- composite ----------------
    g = glasses_rgba
    if g.shape[2] == 3:
        g = np.dstack([g, np.full(g.shape[:2], 255, np.uint8)])
    a = g[:, :, 3:4].astype(np.float32) / 255.0
    pm4 = np.concatenate([g[:, :, :3].astype(np.float32) * a, a], 2)
    corners0 = np.array([[paste_x, paste_y], [paste_x + gw, paste_y],
                         [paste_x + gw, paste_y + gh], [paste_x, paste_y + gh]], np.float64)
    shadow_sigma = max(1.5, 0.025 * gw)
    shadow_dy = max(1.0, 0.012 * gw)
    out = []
    for i in range(n):
        if opac[i] <= 0.004:
            out.append(frames[i])
            continue
        T = np.array(Ms[i])
        M_tot = np.hstack([T[:, :2], (T[:, :2] @ np.array([paste_x, paste_y]) + T[:, 2])[:, None]])
        corners = _apply(T, corners0)
        fr = _composite_patch(frames[i], pm4, M_tot, float(opac[i]), shadow_sigma,
                              shadow_dy, 0.25, corners,
                              ring_r=(0.12 * gw * float(np.sqrt(abs(np.linalg.det(T[:, :2]))))) if cleanup else 0)
        if debug:
            cv2.polylines(fr, [corners.astype(np.int32)], True, (0, 255, 0), 1)
            cv2.putText(fr, "c=%.2f o=%.2f" % (conf[i], opac[i]), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        out.append(fr)

    el = time.time() - t_start
    stats = {
        "mean_conf": float(conf.mean()),
        "frames_overlaid": int((opac > 0.5).sum()),
        "frames_faded": int((opac < 0.999).sum()),
        "max_jitter_px": max_jitter,
        "ms_per_frame": 1000.0 * el / n,
        "conf": conf, "opacity": opac, "inliers": ninl, "reproj_err": reproj,
        "transforms": Ms, "transforms_raw": Ms_full,
    }
    return out, stats
"""

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

import sys
sys.path.insert(0, "/content/temp")
from shot_plan import resolve_shots, output_size, NEG_PORTRAIT
SHOTS = resolve_shots(cfg, custom_p_en)
OUT_W, OUT_H = output_size(cfg)
GEN_W, GEN_H = (1280, 704) if OUT_W > OUT_H else (704, 1280)
print(f"[3A] 分鏡數：{len(SHOTS)}｜輸出比例 {OUT_W}x{OUT_H}｜肖像生成 {GEN_W}x{GEN_H}")

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

neg_p = NEG_PORTRAIT

def gen_portrait(prompt, seed):
    return txt2img(prompt=prompt, negative_prompt=neg_p, width=GEN_W, height=GEN_H, num_inference_steps=6, guidance_scale=1.8, generator=torch.Generator("cuda").manual_seed(seed)).images[0]

def detect_face(portrait_pil):
    # 逐步放寬最小臉部尺寸，讓中景/全身鏡頭的小臉也能偵測到
    gray = cv2.cvtColor(np.array(portrait_pil.convert("RGB")), cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    short = min(gray.shape[:2])
    for frac in (0.20, 0.12, 0.07, 0.045):
        m = max(24, int(short * frac))
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(m, m))
        if len(faces) > 0:
            return sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
    return None

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
GLASSES_META = {}

def composite_eyewear_on_model(portrait_pil, prod_rgba, is_closeup=False, shot_key="1"):
    w, h = portrait_pil.size
    img_bgr = cv2.cvtColor(np.array(portrait_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    face = detect_face(portrait_pil)
    angle_deg = 0.0
    if face is not None:
        fx, fy, fw, fh = face
        FACE_BOXES[shot_key] = [int(fx) / w, int(fy) / h, int(fw) / w, int(fh) / h]  # 未戴眼鏡前偵測，供 3C 視差分層使用
        face_cx = fx + fw // 2
        eye_y = fy + int(fh * 0.39)
        target_w = int(fw * (1.06 if is_snow else 0.96))
        roi_gray = gray[fy:fy + int(fh * 0.6), fx:fx + fw]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(max(8, fw // 9), max(8, fw // 9)))
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
        target_w = int(min(w, h) * (0.56 if is_closeup else 0.42))
    target_h = max(1, int(prod_rgba.height * (target_w / prod_rgba.width)))
    glasses_resized = prod_rgba.resize((target_w, target_h), Image.Resampling.LANCZOS)
    if abs(angle_deg) > 0.5:
        glasses_resized = glasses_resized.rotate(-angle_deg, resample=Image.Resampling.BICUBIC, expand=True)
    gw, gh = glasses_resized.size
    paste_x = face_cx - gw // 2
    paste_y = eye_y - int(gh * 0.48)
    glasses_resized = harmonize_eyewear(portrait_pil, glasses_resized, paste_x, paste_y)
    # 保存真實眼鏡圖與位置：3C 會在 AI 影片每一格依頭部運動重新貼回真品 (避免 AI 把眼鏡畫變形)
    glasses_resized.save(f"/content/temp/glasses_shot{shot_key}.png")
    GLASSES_META[shot_key] = {"x": int(paste_x), "y": int(paste_y), "src_w": int(w), "src_h": int(h)}
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

for sh in SHOTS:
    sid = sh["id"]
    seed = 1024 * sid
    base = gen_portrait(sh["portrait"], seed)
    if sh["wear"]:
        # 偵測不到正面人臉就換種子重生 (最多 3 次)，避免把眼鏡貼在錯誤位置
        tries = 1
        while detect_face(base) is None and tries < 3:
            print(f"  -> 第 {sid} 鏡未偵測到正面人臉，換種子重新生成 ({tries}/2)...")
            base = gen_portrait(sh["portrait"], seed + 7919 * tries)
            tries += 1
        if detect_face(base) is None and not sh["closeup"]:
            print(f"  ⚠️ 第 {sid} 鏡仍偵測不到人臉，此鏡改為不配戴 (避免錯位)")
            kf = base.convert("RGB")
        else:
            kf = composite_eyewear_on_model(base, product_rgba, is_closeup=sh["closeup"], shot_key=str(sid))
    else:
        kf = base.convert("RGB")
        f0 = detect_face(base)
        if f0 is not None:
            FACE_BOXES[str(sid)] = [int(f0[0]) / GEN_W, int(f0[1]) / GEN_H, int(f0[2]) / GEN_W, int(f0[3]) / GEN_H]
    kf.save(f"/content/temp/keyframe_shot{sid}.png")
    print(f"  ✓ 第 {sid}/{len(SHOTS)} 鏡首幀完成 ({'配戴真實眼鏡' if sh['wear'] else '未配戴'})")
with open("/content/temp/face_boxes.json", "w", encoding="utf-8") as f:
    json.dump(FACE_BOXES, f)
with open("/content/temp/glasses_meta.json", "w", encoding="utf-8") as f:
    json.dump(GLASSES_META, f)
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

import sys
sys.path.insert(0, "/content/temp")
from shot_plan import resolve_shots
SHOTS = resolve_shots(cfg, custom_p_en)
print(f"[3B] 共 {len(SHOTS)} 鏡需要編碼動態提示詞")
def check_embeds(*ts):
    for t in ts:
        if not torch.isfinite(t).all():
            raise RuntimeError("[3B] 文字特徵含 NaN/Inf (FP16 溢位)，請改選方案 1C (SVD-XT) 或方案 2。")

neg_video = "frozen frame, still photo, fast motion, sudden movement, walking, running, morphing, melting face, distorted face, deformed sunglasses, melting frame, morphing lenses, extra glasses, warped logo, motion blur, ghosting, double exposure, on-screen text, subtitles, watermark, blurry, flickering, jerky camera"

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
    out = {}
    with torch.no_grad():
        for sh in SHOTS:
            pe, npe = pipe_te.encode_prompt(prompt=sh["motion"], negative_prompt=neg_video, do_classifier_free_guidance=True, max_sequence_length=512, device="cuda", dtype=infer_dtype)
            check_embeds(pe, npe)
            out[sh["id"]] = {"pe": pe.cpu(), "npe": npe.cpu()}
    torch.save(out, "/content/temp/prompt_embeds.pt")
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
    out = {}
    with torch.no_grad():
        for sh in SHOTS:
            pe, pam, npe, npam = pipe_te.encode_prompt(prompt=sh["motion"], negative_prompt=neg_video, do_classifier_free_guidance=True, max_sequence_length=256, device="cuda", dtype=infer_dtype)
            check_embeds(pe, npe)
            out[sh["id"]] = {"pe": pe.cpu(), "pam": pam.cpu(), "npe": npe.cpu(), "npam": npam.cpu()}
    torch.save(out, "/content/temp/prompt_embeds.pt")
    print("[3B] LTX-Video 文字特徵編碼完成！即將關閉子程序釋放 T5 記憶體...")
"""

STAGE_3C_CODE = r"""
import os, json, gc, time, traceback, subprocess, cv2, torch
import numpy as np
from PIL import Image

with open("/content/temp/job_config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

gen_mode = cfg["generation_mode"]
import sys
sys.path.insert(0, "/content/temp")
from shot_plan import resolve_shots, output_size
SHOTS = resolve_shots(cfg, cfg.get("custom_scene_prompt_en", cfg.get("custom_scene_prompt", "")))
OUT_W, OUT_H = output_size(cfg)
LANDSCAPE = OUT_W > OUT_H
def free_ram():
    # 釋放 Python 物件並把 glibc 堆積還給作業系統 (實測可回收 ~2.7GB)
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def peak_ram_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576.0
    except Exception:
        return 0.0

LTX_CHILD = os.environ.get("ESPNER_LTX_REPO", "")
CHILD_STATUS = "/content/temp/stage3c_child_status.json"

def shot_mp4(sid):
    return f"/content/temp/raw_shot{sid}.mp4"
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
    odd = (shot_idx % 2 == 1)
    d = 1.0 if odd else -1.0
    # 背景：慢、反向橫移；前景人物：較大推進 + 同向橫移 + 微幅呼吸 → 立體縱深感
    sb = 1.04 + 0.03 * e
    Mb = cv2.getRotationMatrix2D(focus, 0, sb)
    Mb[0, 2] += -d * 22.0 * (e - 0.5)
    sf = 1.0 + (0.085 * e if odd else 0.085 * (1 - e)) + 0.006 * np.sin(t * np.pi * 3)
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
    w, h = OUT_W, OUT_H
    prod_np = np.array(product_rgba)
    for sh in SHOTS:
        shot_idx, out_mp4 = sh["id"], shot_mp4(sh["id"])
        num_frames = int(round(sh["dur"] * 24))
        raw_tmp = out_mp4.replace(".mp4", "_raw.mp4")
        writer = cv2.VideoWriter(raw_tmp, cv2.VideoWriter_fourcc(*'mp4v'), 24.0, (w, h))
        kf_path = f"/content/temp/keyframe_shot{shot_idx}.png"
        has_kf = os.path.exists(kf_path) and ("方案 2" not in gen_mode)
        if has_kf:
            kf_bgr = cv2.cvtColor(np.array(Image.open(kf_path).convert("RGB").resize((w, h), Image.Resampling.LANCZOS)), cv2.COLOR_RGB2BGR)
            px_bg, px_fg, px_alpha, px_focus = build_parallax_layers(kf_bgr, load_face_box(shot_idx, w, h))
            print(f"  -> 第 {shot_idx} 段視差分層完成 (人物前景佔 {float(px_alpha.mean()) * 100:.0f}%)")
        base_w = int(min(w, h) * (0.69 if shot_idx % 2 == 1 else 0.8))
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
                zoom = (0.94 + 0.12 * t) if shot_idx % 2 == 1 else (1.06 - 0.08 * t)
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
    print(f"[3C] {len(SHOTS)} 段動態鏡頭生成完成！")

if "方案 2" in gen_mode:
    render_3d_motion_fallback()
    raise SystemExit(0)

KFS = {sh["id"]: Image.open(f"/content/temp/keyframe_shot{sh['id']}.png").convert("RGB") for sh in SHOTS}
DURS = {sh["id"]: sh["dur"] for sh in SHOTS}

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

def fidelity_check(frames, kf_img, tag):
    # 圖生影片的第 1 格必須跟輸入首幀幾乎一樣；顏色反轉 (負片) 會得到負相關，亂碼色塊則連續性很低
    def g(x):
        return cv2.resize(cv2.cvtColor(_to_u8(x), cv2.COLOR_RGB2GRAY), (64, 112)).astype(np.float32).ravel()
    k = g(np.asarray(kf_img.convert("RGB")))
    c0 = float(np.corrcoef(k, g(frames[0]))[0, 1])
    step = max(1, len(frames) // 8)
    seq = [g(f) for f in frames[::step]]
    cc = float(np.mean([np.corrcoef(seq[i], seq[i + 1])[0, 1] for i in range(len(seq) - 1)])) if len(seq) > 1 else 1.0
    print(f"     畫面檢查：首幀相似度 {c0:.2f} (需 ≥ 0.5)｜連續性 {cc:.2f} (需 ≥ 0.55)")
    if not (c0 >= 0.5 and cc >= 0.55):
        raise RuntimeError(f"{tag} 輸出畫面異常 (首幀相似度 {c0:.2f}、連續性 {cc:.2f}；可能為顏色反轉或亂碼)")
    return c0, cc

def ltx_generate(pipe, vae32, kf_in, W, H, F, steps, pe, pam, npe, npam, seed, dev, dtype, decode_t=0.05, decode_noise=0.025):
    # T4 修正：VAE 編碼/解碼一律在 float32 執行 (fp16 VAE 會讓畫面變負片/亂碼)，轉換器只負責去噪
    g = torch.Generator(dev).manual_seed(seed)
    sp, st = pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size
    lf = (F - 1) // pipe.vae_temporal_compression_ratio + 1
    lh, lw = H // pipe.vae_spatial_compression_ratio, W // pipe.vae_spatial_compression_ratio
    img = pipe.video_processor.preprocess(kf_in, height=H, width=W).to(dev, torch.float32)
    with torch.no_grad():
        enc = vae32.encode(img.unsqueeze(2)).latent_dist.mode().float()
    lat0 = pipe._normalize_latents(enc, vae32.latents_mean, vae32.latents_std).repeat(1, 1, lf, 1, 1)
    mask = torch.zeros((1, 1, lf, lh, lw), device=dev, dtype=torch.float32)
    mask[:, :, 0] = 1.0
    noise = torch.randn(lat0.shape, generator=g, device=dev, dtype=torch.float32)
    lat = pipe._pack_latents(lat0 * mask + noise * (1 - mask), sp, st)
    out = pipe(image=None, latents=lat, prompt=None, negative_prompt=None,
               prompt_embeds=pe.to(dev, dtype), prompt_attention_mask=pam.to(dev),
               negative_prompt_embeds=npe.to(dev, dtype), negative_prompt_attention_mask=npam.to(dev),
               width=W, height=H, num_frames=F, frame_rate=24, num_inference_steps=steps, guidance_scale=3.0,
               generator=g, output_type="latent").frames
    lat = pipe._unpack_latents(out, lf, lh, lw, sp, st)
    lat = pipe._denormalize_latents(lat, vae32.latents_mean, vae32.latents_std, vae32.config.scaling_factor).float()
    if not torch.isfinite(lat).all():
        raise RuntimeError("LTX 去噪結果含 NaN/Inf (FP16 溢位)")
    ts = None
    if vae32.config.timestep_conditioning:
        n2 = torch.randn(lat.shape, generator=g, device=dev, dtype=torch.float32)
        lat = (1 - decode_noise) * lat + decode_noise * n2
        ts = torch.tensor([decode_t], device=dev, dtype=torch.float32)
    with torch.no_grad():
        video = vae32.decode(lat, ts, return_dict=False)[0]
    return pipe.video_processor.postprocess_video(video, output_type="pil")[0]

WEAR = {sh["id"]: sh["wear"] for sh in SHOTS}
try:
    with open("/content/temp/glasses_meta.json", "r", encoding="utf-8") as _f:
        GLASSES_META = json.load(_f)
except Exception:
    GLASSES_META = {}

def apply_real_glasses(sid, out_p):
    # AI 會把眼鏡畫變形 → 追蹤頭部運動，把 3A 的真實眼鏡逐格貼回 (產品 100% 真品)
    gp = f"/content/temp/glasses_shot{sid}.png"
    meta = GLASSES_META.get(str(sid))
    if not WEAR.get(sid) or meta is None or not os.path.exists(gp):
        return None
    import espner_track as TRK
    sx, sy = OUT_W / float(meta["src_w"]), OUT_H / float(meta["src_h"])
    g = cv2.imread(gp, cv2.IMREAD_UNCHANGED)
    g = cv2.resize(g, (max(1, int(round(g.shape[1] * sx))), max(1, int(round(g.shape[0] * sy)))), interpolation=cv2.INTER_AREA)
    cap = cv2.VideoCapture(out_p)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        return None
    out, st = TRK.recomposite_glasses(frames, g, int(round(meta["x"] * sx)), int(round(meta["y"] * sy)),
                                      face_box=load_face_box(sid, OUT_W, OUT_H), cleanup=True)
    cv2.setNumThreads(0)
    tmp = out_p.replace(".mp4", "_glasses.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (OUT_W, OUT_H))
    for fr in out:
        vw.write(fr)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", out_p], check=True)
    print(f"     👓 真實眼鏡逐格貼回：追蹤信心 {st.get('mean_conf', 0):.2f}｜貼回 {st.get('frames_overlaid', 0)} 格｜淡出 {st.get('frames_faded', 0)} 格")
    return st

def salvage_clip(sid, out_p, glasses_opacity=None):
    # 只保留清晰、沒變形、眼鏡追蹤穩定的連續片段；不夠長就用最後一個好畫面做慢推鏡補足 (不補幀、不拉長)
    import espner_salvage as SV
    cap = cv2.VideoCapture(out_p)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        return None
    n = int(round(DURS[sid] * 24))
    ref = cv2.resize(cv2.imread(f"/content/temp/keyframe_shot{sid}.png"), (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    op = None
    if glasses_opacity is not None and len(glasses_opacity) == len(frames):
        op = np.asarray(glasses_opacity, dtype=np.float32)
    out, info = SV.select_usable(frames, n, ref_bgr=ref, glasses_opacity=op)
    tmp = out_p.replace(".mp4", "_salv.mp4")
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (OUT_W, OUT_H))
    for fr in out:
        vw.write(fr)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp, "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", out_p], check=True)
    print(f"     ✂️ 挑選可用片段：AI 畫面 {info.get('used_ai_frames')} 格 + 補足 {info.get('held_frames')} 格 (共 {n} 格){'｜改用首幀運鏡' if info.get('fallback') else ''}")
    return info

def post_shot(sid, out_p, entry):
    st = None
    try:
        st = apply_real_glasses(sid, out_p)
        if st:
            entry["glasses_conf"] = round(float(st.get("mean_conf", 0)), 2)
    except Exception as e:
        print(f"     ⚠️ 真實眼鏡貼回略過 ({e})")
    try:
        info = salvage_clip(sid, out_p, (st or {}).get("opacity"))
        if info:
            entry["ai_frames_used"] = int(info.get("used_ai_frames", 0))
            entry["held_frames"] = int(info.get("held_frames", 0))
    except Exception as e:
        print(f"     ⚠️ 片段挑選略過 ({e})")

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
    # 以 OpenCV 寫暫存檔 (不依賴 imageio-ffmpeg)
    f0 = _to_u8(frames[0])
    vw = cv2.VideoWriter(tmp_raw, cv2.VideoWriter_fourcc(*"mp4v"), float(src_fps), (f0.shape[1], f0.shape[0]))
    for f in frames:
        vw.write(cv2.cvtColor(_to_u8(f), cv2.COLOR_RGB2BGR))
    vw.release()
    pts = dur / (len(frames) / float(src_fps))
    if src_fps == 24 and pts <= 1.02:
        vf = f"scale={OUT_W}:{OUT_H}:flags=lanczos,tpad=stop_mode=clone:stop_duration=1"   # 原生長度：不補幀
    else:
        vf = f"setpts={pts:.4f}*PTS,minterpolate=fps=24:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,scale={OUT_W}:{OUT_H}:flags=lanczos,tpad=stop_mode=clone:stop_duration=1"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-i", tmp_raw,
        "-vf", vf,
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
    w, h, n = OUT_W, OUT_H, max(1, len(src))
    tmp_mv = out_p.replace(".mp4", "_mv.mp4")
    wr = cv2.VideoWriter(tmp_mv, cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (w, h))
    for i, fr in enumerate(src):
        t = i / max(1, n - 1)
        e = t * t * (3 - 2 * t)
        sc = 1.0 + 0.09 * e if shot_idx % 2 == 1 else 1.09 - 0.07 * e
        dx = (18.0 if shot_idx % 2 == 1 else -18.0) * (e - 0.5)
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
        if LANDSCAPE:
            gen_w, gen_h = gen_h, gen_w
        shots = [(sid, KFS[sid], embeds[sid]["pe"].to("cuda", infer_dtype), embeds[sid]["npe"].to("cuda", infer_dtype), shot_mp4(sid)) for sid in KFS]
        for idx, kf, pe, npe, out_p in shots:
            t0 = time.time()
            print(f"  -> 正在透過 T4 GPU 運算第 {idx}/{len(shots)} 段 Wan2.2 動態鏡頭 ({gen_w}x{gen_h}, {gen_f}f)...")
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
            fidelity_check(frames, kf_in, f"Wan2.2 第 {idx} 鏡")
            ms = motion_score(frames)
            finalize_ai_clip(frames, out_p, add_camera_move=ms < MOTION_MIN, shot_idx=idx, dur=DURS[idx])
            entry = {"shot": idx, "motion": round(ms, 2)}
            post_shot(idx, out_p, entry)
            STATUS["shots"].append(entry)
            print(f"     ✓ 第 {idx} 段完成 (motion={ms:.2f}，耗時 {time.time() - t0:.1f}s)")
            del frames
            gc.collect()
            torch.cuda.empty_cache()

    elif "LTX-Video" in gen_mode:
        from diffusers import LTXImageToVideoPipeline, AutoencoderKLLTXVideo, LTXVideoTransformer3DModel
        embeds = torch.load("/content/temp/prompt_embeds.pt", map_location="cpu", weights_only=False) if LTX_CHILD else {}
        # 0.9.5 的圖生影片動態遠優於 0.9.0；兩者共用同一個 T5，若 0.9.5 失敗自動退回 0.9.0
        LTX_REPOS = ["Lightricks/LTX-Video-0.9.5", "Lightricks/LTX-Video"]
        # 576x1024：比 480x832 清晰 1.4 倍，運算量約 1.5 倍；搭配 3 秒短鏡頭控制 T4 時間
        LTX_W, LTX_H, LTX_STEPS = (1024, 576, 30) if LANDSCAPE else (576, 1024, 30)

        def ltx_frames_for(dur):
            # 直接生成整段長度 (8k+1 格)，不靠補幀拉長 → 避免殘影；上限 145 格 (6 秒) 以控制 T4 時間
            n = int(round(dur * 24))
            return min(145, ((n - 1 + 7) // 8) * 8 + 1)
        shots = [(sid, KFS[sid], embeds[sid]["pe"], embeds[sid]["pam"], embeds[sid]["npe"], embeds[sid]["npam"], shot_mp4(sid)) for sid in KFS] if LTX_CHILD else []

        def run_ltx(repo):
            print(f"[3C] 正在載入無文字編碼器版 {repo} (text_encoder=None, dtype={infer_dtype}，約 ~5GB RAM)...")
            # 系統 RAM 只有 12.7GB：先載 VAE(fp32) 並立刻搬上 GPU，再把轉換器(fp16) 直接逐片載入 GPU (device_map)。
            # 2B 轉換器 3.9GB + VAE 2.5GB ≈ 6.4GB VRAM，T4 15GB 放得下，因此不需 CPU offload，系統 RAM 峰值大幅下降。
            vae32 = AutoencoderKLLTXVideo.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True).to("cuda")
            vae32.enable_tiling()
            free_ram()
            tr = LTXVideoTransformer3DModel.from_pretrained(repo, subfolder="transformer", torch_dtype=infer_dtype, low_cpu_mem_usage=True, device_map="cuda")
            free_ram()
            pipe = LTXImageToVideoPipeline.from_pretrained(
                repo, transformer=tr, text_encoder=None, tokenizer=None, vae=None, torch_dtype=infer_dtype
            )
            free_ram()
            print(f"     模型載入完成｜系統 RAM 峰值 {peak_ram_gb():.1f} GB")
            for idx, kf, pe, pam, npe, npam, out_p in shots:
                best = None
                # 第 1 次 CRF 30；若畫面幾乎靜止，換種子並加重壓縮 (CRF 40) 重試一次
                for attempt, crf in enumerate([30, 40]):
                    t0 = time.time()
                    LTX_F = ltx_frames_for(DURS[idx])
                    print(f"  -> 第 {idx}/{len(shots)} 段 LTX 動態鏡頭 ({LTX_W}x{LTX_H}, {LTX_F}f, {LTX_STEPS} steps, 首幀 CRF {crf})...")
                    kf_in = ltx_preprocess(kf.resize((LTX_W, LTX_H), Image.Resampling.LANCZOS), crf)
                    frames = ltx_generate(pipe, vae32, kf_in, LTX_W, LTX_H, LTX_F, LTX_STEPS,
                                          pe, pam, npe, npam, 42 + idx + 1000 * attempt, "cuda", infer_dtype)
                    check_frames(frames, "LTX-Video")
                    fid = fidelity_check(frames, kf_in, f"{repo} 第 {idx} 鏡")
                    ms = motion_score(frames)
                    print(f"     動態量 motion={ms:.2f} (門檻 {MOTION_MIN})，耗時 {time.time() - t0:.1f}s")
                    if best is None or ms > best[0]:
                        best = (ms, frames)
                    if ms >= MOTION_MIN:
                        break
                    print("     ⚠️ 畫面幾乎靜止，換種子並加重首幀壓縮重試...")
                finalize_ai_clip(best[1], out_p, add_camera_move=best[0] < MOTION_MIN, shot_idx=idx, dur=DURS[idx])
                entry = {"shot": idx, "motion": round(best[0], 2), "first_frame_sim": round(fid[0], 2), "continuity": round(fid[1], 2)}
                post_shot(idx, out_p, entry)
                STATUS["shots"].append(entry)
                print(f"     ✓ 第 {idx} 段完成 (motion={best[0]:.2f})")
                del frames, best
                gc.collect()
                torch.cuda.empty_cache()
            del pipe, vae32

        if LTX_CHILD:
            # 子程序：只跑指定的模型版本，結束時作業系統 100% 回收記憶體
            STATUS["shots"] = []
            run_ltx(LTX_CHILD)
            STATUS["engine"] = LTX_CHILD
        else:
            # 每個模型版本各開一個獨立子程序：失敗的版本不會殘留在 RAM，再載入下一版也不會爆記憶體
            last_err = None
            for ri, repo in enumerate(LTX_REPOS):
                if os.path.exists(CHILD_STATUS):
                    os.remove(CHILD_STATUS)
                rc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__)], env={**os.environ, "ESPNER_LTX_REPO": repo}).returncode
                child = {}
                if os.path.exists(CHILD_STATUS):
                    with open(CHILD_STATUS, "r", encoding="utf-8") as _f:
                        child = json.load(_f)
                if rc == 0 and child.get("ai_ok"):
                    STATUS["shots"] = child.get("shots", [])
                    STATUS["engine"] = child.get("engine")
                    break
                last_err = child.get("error") or f"子程序結束碼 {rc} (可能記憶體不足)"
                if ri < len(LTX_REPOS) - 1:
                    print(f"[3C] ⚠️ {repo} 失敗 ({last_err})，自動改用 {LTX_REPOS[ri + 1]} 重試 (新子程序)...")
            else:
                raise RuntimeError(last_err or "LTX 全部版本失敗")

    else:
        print("[3C] 正在載入 SVD-XT 純圖生影片模型 (無文字編碼器，僅需 ~4.2GB RAM)...")
        from diffusers import StableVideoDiffusionPipeline
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16, variant="fp16", low_cpu_mem_usage=True
        )
        pipe.enable_model_cpu_offload()
        for idx, kf in KFS.items():
            out_p = shot_mp4(idx)
            print(f"  -> 正在透過 T4 GPU 運算第 {idx}/{len(KFS)} 段 SVD-XT 動態鏡頭...")
            kf_in = kf.resize((1024, 576) if LANDSCAPE else (576, 1024), Image.Resampling.LANCZOS)
            frames = pipe(kf_in, decode_chunk_size=2, num_frames=25, motion_bucket_id=160, noise_aug_strength=0.03,
                          generator=torch.Generator("cuda").manual_seed(7 + idx)).frames[0]
            check_frames(frames, "SVD-XT")
            fidelity_check(frames, kf_in, f"SVD-XT 第 {idx} 鏡")
            ms = motion_score(frames)
            finalize_ai_clip(frames, out_p, add_camera_move=ms < MOTION_MIN, shot_idx=idx, dur=DURS[idx], src_fps=8)
            entry = {"shot": idx, "motion": round(ms, 2)}
            post_shot(idx, out_p, entry)
            STATUS["shots"].append(entry)
            print(f"     ✓ 第 {idx} 段完成 (motion={ms:.2f})")
        STATUS["engine"] = "stabilityai/stable-video-diffusion-img2vid-xt"
    STATUS["ai_ok"] = True
except Exception as e:
    STATUS["ai_ok"] = False
    STATUS["error"] = f"{type(e).__name__}: {e}"
    print(f"\n⚠️ [3C] AI 擴散引擎發生例外 ({e})，完整診斷訊息如下：")
    traceback.print_exc()
    if LTX_CHILD:
        raise SystemExit(1)
    print("\n🔄 正在自動啟動真人配戴首幀 2.5D 視差運鏡保底合成，確保廣告順利產出...")
    render_3d_motion_fallback()
finally:
    with open(CHILD_STATUS if LTX_CHILD else "/content/temp/stage3c_status.json", "w", encoding="utf-8") as f:
        json.dump(STATUS, f, ensure_ascii=False, indent=2)
    try:
        import resource
        print(f"[3C] {'子程序' if LTX_CHILD else '主程序'}記憶體峰值：{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.1f} GB")
    except Exception:
        pass
"""

def cell_lines(code):
    # notebook cell source lines (last line without trailing newline, as Jupyter stores it)
    parts = code.split("\n")[1:]
    return [ln + "\n" for ln in parts[:-1]] + [parts[-1]]

# ---- 步驟 4 後製模組 (字卡/片尾、調色、配樂)；於步驟 4 寫入 /content/temp 後 import ----
TITLES_CODE = r"""
# espner_titles.py
# ESPNER luxury eyewear: typography overlays and endcards (numpy + OpenCV + Pillow only).
# Frames are uint8 BGR (OpenCV convention). Text layers are RGBA uint8 (RGB channel order, straight alpha).
# Style: minimal clean sans-serif, charcoal text on light backgrounds, generous whitespace, no boxes.
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHARCOAL = (31, 35, 40)      # #1F2328 primary text
GREY_TEXT = (74, 74, 74)     # #4A4A4A secondary text
MUTED_TEXT = (90, 90, 90)    # #5A5A5A tiny labels (still > 6:1 on white)
WHITE = (255, 255, 255)

_FONT_CANDIDATES = {
    "regular": [
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 3),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
        ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 3),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    ],
    "bold": [
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 3),
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 0),
        ("/usr/share/fonts/truetype/noto/NotoSansCJK-Bold.ttc", 3),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),
    ],
}
_font_cache = {}


def _font(weight, size):
    # weight: "regular" or "bold". Falls back gracefully; last resort is PIL default font.
    size = max(6, int(round(size)))
    key = (weight, size)
    if key in _font_cache:
        return _font_cache[key]
    f = None
    for path, idx in _FONT_CANDIDATES[weight]:
        try:
            f = ImageFont.truetype(path, size, index=idx)
            break
        except Exception:
            continue
    if f is None:
        try:
            f = ImageFont.load_default(size)
        except Exception:
            f = ImageFont.load_default()
    _font_cache[key] = f
    return f


def _unit(W, H):
    # reference pixel unit: designs are specified for 1080p landscape / 720-wide portrait
    return (W / 720.0) if H > W else (H / 1080.0)


def _is_cjk(text):
    return any(ord(c) > 0x2E80 for c in text)


def _measure(text, size, tracking_px, weight="regular"):
    f = _font("regular" if weight == "medium" else weight, size)
    return sum(f.getlength(c) for c in text) + tracking_px * max(0, len(text) - 1)


def _draw_mask(W, H, text, x, cy, size, tracking_px, weight, anchor):
    # returns float32 (H,W) mask 0..1 for tracked text; x is left edge (anchor "l") or center ("c")
    # cy is the visual vertical center of the text line.
    cjk = _is_cjk(text)
    base_off = 0.38 * size if cjk else 0.5 * 0.73 * size
    baseline = cy + base_off
    width = _measure(text, size, tracking_px, weight)
    x0 = x - width / 2.0 if anchor == "c" else x
    wlist = ["regular", "bold"] if weight == "medium" else [weight]
    out = None
    fr = _font("regular" if weight == "medium" else weight, size)
    for w in wlist:
        f = _font(w, size)
        img = Image.new("L", (W, H), 0)
        d = ImageDraw.Draw(img)
        px = x0
        for c in text:
            d.text((px, baseline), c, font=f, fill=255, anchor="ls")
            px += fr.getlength(c) + tracking_px
        m = np.asarray(img, dtype=np.float32) / 255.0
        out = m if out is None else out + m
    return out / len(wlist)


def _compose_items(W, H, items):
    # items: dict(text,x,cy,size,tracking(em),weight,anchor,color,opacity,shadow)
    acc_rgb = np.zeros((H, W, 3), np.float32)
    acc_a = np.zeros((H, W), np.float32)

    def over(m, col):
        nonlocal acc_rgb, acc_a
        acc_rgb = acc_rgb * (1 - m)[..., None] + np.array(col, np.float32) * m[..., None]
        acc_a = acc_a + m - acc_a * m

    for it in items:
        size = it["size"]
        m = _draw_mask(W, H, it["text"], it["x"], it["cy"], size,
                       it.get("tracking", 0.0) * size, it.get("weight", "regular"), it.get("anchor", "l"))
        op = it.get("opacity", 1.0)
        if it.get("shadow"):
            sh = cv2.GaussianBlur(m, (0, 0), max(1.5, size * 0.07)) * 0.45 * op
            over(np.clip(sh, 0, 1), (0, 0, 0))
        over(m * op, it["color"])
    rgb = acc_rgb / np.maximum(acc_a, 1e-4)[..., None]
    rgba = np.dstack([np.clip(rgb, 0, 255), np.clip(acc_a, 0, 1) * 255.0])
    return np.rint(rgba).astype(np.uint8)


def _col(color):
    return WHITE if color == "light" else CHARCOAL


# --------------------------------------------------------------------------- overlays on footage

def render_brand_mark_layer(W, H, position="top-left", color="dark"):
    # tiny tracked ESPNER mark. top-left like ref A, top-center like ref B.
    u = _unit(W, H)
    c = _col(color)
    shadow = color == "light"
    if position == "top-center":
        item = dict(text="ESPNER", x=W / 2.0, cy=0.113 * H, size=34 * u, tracking=0.08,
                    weight="medium", anchor="c", color=c, shadow=shadow)
    else:
        if H > W:
            item = dict(text="E S P N E R", x=0.08 * W, cy=0.055 * H, size=19 * u, tracking=0.22,
                        weight="bold", anchor="l", color=c, shadow=shadow)
        else:
            item = dict(text="E S P N E R", x=0.063 * W, cy=0.081 * H, size=18 * u, tracking=0.22,
                        weight="bold", anchor="l", color=c, shadow=shadow)
    return _compose_items(W, H, [item])


def render_copy_layer(W, H, lines, brand_mark=True, align="left", color="dark"):
    # copy lines (1-3, CJK) left aligned at ~8% margin. Landscape: vertical center ~0.35H (ref A).
    # Portrait: lower-middle third (center ~0.70H) so the face stays free.
    lines = list(lines)[:3]
    u = _unit(W, H)
    size = 42 * u
    lead = 1.32 * size
    portrait = H > W
    block_cy = (0.70 if portrait else 0.355) * H
    x = (0.08 if portrait else 0.0766) * W
    c = _col(color)
    shadow = color == "light"
    items = []
    n = len(lines)
    for i, ln in enumerate(lines):
        cy = block_cy + (i - (n - 1) / 2.0) * lead
        if align == "center":
            items.append(dict(text=ln, x=W / 2.0, cy=cy, size=size, tracking=0.06, weight="medium",
                              anchor="c", color=c, shadow=shadow))
        else:
            items.append(dict(text=ln, x=x, cy=cy, size=size, tracking=0.06, weight="medium",
                              anchor="l", color=c, shadow=shadow))
    layer = _compose_items(W, H, items)
    if brand_mark:
        bm = render_brand_mark_layer(W, H, "top-left", color)
        # combine straight-alpha layers (non overlapping regions in practice)
        a1 = layer[..., 3:4].astype(np.float32) / 255.0
        a2 = bm[..., 3:4].astype(np.float32) / 255.0
        a = a1 + a2 - a1 * a2
        rgb = (layer[..., :3] * a1 * (1 - a2) + bm[..., :3] * a2) / np.maximum(a, 1e-4)
        layer = np.rint(np.dstack([rgb, a * 255.0])).astype(np.uint8)
    return layer


# --------------------------------------------------------------------------- blending / timing

_bbox_cache = {}


def _bbox(layer):
    k = id(layer)
    hit = _bbox_cache.get(k)
    if hit is not None and hit[0] is layer:
        return hit[1]
    a = np.ascontiguousarray(layer[..., 3])
    x, y, w, h = cv2.boundingRect((a > 0).astype(np.uint8))
    bb = (x, y, x + w, y + h)
    if len(_bbox_cache) > 16:
        _bbox_cache.clear()
    _bbox_cache[k] = (layer, bb)
    return bb


def overlay_layer(frame_bgr, layer_rgba, opacity):
    # alpha-blend an RGBA layer onto a BGR frame with global opacity (0..1). Returns a new frame.
    out = frame_bgr.copy()
    opacity = float(min(1.0, max(0.0, opacity)))
    if opacity <= 0.002:
        return out
    x0, y0, x1, y1 = _bbox(layer_rgba)
    if x1 <= x0 or y1 <= y0:
        return out
    roi = layer_rgba[y0:y1, x0:x1]
    a = roi[..., 3:4].astype(np.float32) * (opacity / 255.0)
    src = roi[..., 2::-1].astype(np.float32)  # RGB -> BGR
    dst = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = np.rint(dst + (src - dst) * a).astype(np.uint8)
    return out


def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def fade_opacity(t, start, end, fade=0.5):
    # 0 before start, ramps in over fade seconds, holds, ramps out to 0 at end
    if t <= start or t >= end:
        return 0.0
    f = min(float(fade), (end - start) / 2.0)
    if f <= 1e-6:
        return 1.0
    return _smooth(min((t - start) / f, (end - t) / f, 1.0))


def _lin(c):
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _rel_lum(rgb):
    return float(0.2126 * _lin(np.float32(rgb[0])) + 0.7152 * _lin(np.float32(rgb[1])) + 0.0722 * _lin(np.float32(rgb[2])))


def text_color_for(frame_bgr, layer_rgba):
    # choose "dark" (charcoal) or "light" (white) text by WCAG contrast against the footage under the text
    x0, y0, x1, y1 = _bbox(layer_rgba)
    if x1 <= x0:
        return "dark"
    a = layer_rgba[y0:y1:2, x0:x1:2, 3]
    region = frame_bgr[y0:y1:2, x0:x1:2]
    m = cv2.dilate((a > 20).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    px = region[m] if m.any() else region.reshape(-1, 3)
    rgb = px[:, ::-1].astype(np.float32)
    lum = 0.2126 * _lin(rgb[:, 0]) + 0.7152 * _lin(rgb[:, 1]) + 0.0722 * _lin(rgb[:, 2])
    L = float(lum.mean())
    cr_dark = (L + 0.05) / (_rel_lum(CHARCOAL) + 0.05)
    cr_light = (1.05) / (L + 0.05)
    return "dark" if cr_dark >= cr_light else "light"


# --------------------------------------------------------------------------- product handling

def load_product_rgba(path):
    # returns BGRA uint8 cropped to the product. A 3 channel image (white background) is matted by
    # flood-filling near-white pixels connected to the border (lens interiors survive).
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        near_white = (img.min(axis=2) > 236).astype(np.uint8)
        n, lab = cv2.connectedComponents(near_white, connectivity=4)
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        bg = np.isin(lab, [b for b in border if b != 0]) & (near_white > 0)
        fg = (~bg).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.erode(fg, np.ones((3, 3), np.uint8))
        alpha = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), 1.0)
        img = np.dstack([img, np.rint(alpha * 255).astype(np.uint8)])
    ys, xs = np.where(img[..., 3] > 8)
    if len(xs):
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return img


class _Sprite(object):
    # premultiplied float32 BGRA with an anchor point, pasted with sub-pixel affine placement
    def __init__(self, arr, ax, ay):
        self.arr, self.ax, self.ay = arr, float(ax), float(ay)


def _premult(bgra_u8):
    f = bgra_u8.astype(np.float32)
    a = f[..., 3:4] / 255.0
    return np.concatenate([f[..., :3] * a, a], axis=2)


def _over_premult(dst, src):
    # both premultiplied float32 BGRA, same shape
    return src + dst * (1 - src[..., 3:4])


def _paste(frame, sp, x, y, sx, sy):
    h, w = sp.arr.shape[:2]
    ox, oy = x - sp.ax * sx, y - sp.ay * sy
    bx0, by0 = int(max(0, math.floor(ox))), int(max(0, math.floor(oy)))
    bx1, by1 = int(min(frame.shape[1], math.ceil(ox + w * sx) + 1)), int(min(frame.shape[0], math.ceil(oy + h * sy) + 1))
    if bx1 <= bx0 or by1 <= by0:
        return
    M = np.array([[sx, 0, ox - bx0], [0, sy, oy - by0]], np.float32)
    roi = cv2.warpAffine(sp.arr, M, (bx1 - bx0, by1 - by0), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    f = frame[by0:by1, bx0:bx1].astype(np.float32)
    frame[by0:by1, bx0:bx1] = np.clip(np.rint(roi[..., :3] + f * (1 - roi[..., 3:4])), 0, 255).astype(np.uint8)


def _fit_product(prod_bgra, target_w, max_h):
    h, w = prod_bgra.shape[:2]
    s = min(target_w / w, max_h / h)
    nw, nh = max(2, int(round(w * s))), max(2, int(round(h * s)))
    pm = _premult(prod_bgra)
    pm = cv2.resize(pm, (nw, nh), interpolation=cv2.INTER_AREA)
    return pm


def _ellipse_blob(w, h, cx, cy, rx, ry, alpha, blur):
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (int(round(cx)), int(round(cy))), (max(1, int(rx)), max(1, int(ry))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    if blur > 0:
        m = cv2.GaussianBlur(m, (0, 0), blur)
    return m * alpha


def _grain(H, W, amp=1.2, seed=7):
    rng = np.random.RandomState(seed)
    return (rng.randn(H, W, 1).astype(np.float32) * amp)


# --------------------------------------------------------------------------- Endcard A (split / white)

def build_endcard_split(product_rgba_path, W, H, model_name, slogan_lines=("把目光，", "還給世界。"),
                        category="SUNGLASSES", duration=5.0, fps=24):
    u = _unit(W, H)
    portrait = H > W
    prod = load_product_rgba(product_rgba_path)
    zmax = 1.035
    if portrait:
        pw, cx, cy = 0.86 * W, 0.5 * W, 0.40 * H
        tx = 0.08 * W
        brand_cy, model_cy = 0.105 * H, 0.105 * H + 50 * u
        slog_cy0, slog_lead, slog_size = 0.625 * H, 78 * u, 40 * u
        cat_cy = None
        bsize = 46 * u
    else:
        pw, cx, cy = 0.53 * W, 0.65 * W, 0.48 * H
        tx = 0.085 * W
        brand_cy, model_cy = 0.34 * H, 0.34 * H + 50 * u
        slog_cy0, slog_lead, slog_size = 0.475 * H + 8 * u, 79 * u, 36 * u
        bsize = 42 * u
    pm = _fit_product(prod, pw * zmax, 0.62 * H * zmax)
    ph, pwid = pm.shape[:2]
    # sprite with soft ground contact shadow below the product
    padx, pady = int(0.06 * pwid), int(0.12 * ph)
    SW, SH = pwid + 2 * padx, int(0.10 * ph) + ph + int(0.24 * ph)
    sprite = np.zeros((SH, SW, 4), np.float32)
    sh_cy = int(0.10 * ph) + ph + 0.07 * ph
    # wide faint shadow + tighter darker core
    s1 = _ellipse_blob(SW, SH, SW / 2.0, sh_cy, 0.40 * pwid, 0.030 * ph + 2, 0.16, 0.030 * ph + 3)
    s2 = _ellipse_blob(SW, SH, SW / 2.0, sh_cy - 0.01 * ph, 0.30 * pwid, 0.012 * ph + 1, 0.20, 0.012 * ph + 2)
    sh = np.clip(s1 + s2, 0, 0.30)
    sprite[..., 3] = sh  # black shadow, premultiplied rgb = 0
    top = int(0.10 * ph)
    prem = np.zeros_like(sprite)
    prem[top:top + ph, padx:padx + pwid] = pm
    sprite = _over_premult(sprite, prem)
    sp = _Sprite(sprite, padx + pwid / 2.0, top + ph / 2.0)
    # text layer (rendered once)
    items = [
        dict(text="ESPNER", x=tx, cy=brand_cy, size=bsize, tracking=0.08, weight="bold", color=CHARCOAL),
        dict(text=str(model_name).upper(), x=tx, cy=model_cy, size=19 * u, tracking=0.22, weight="regular", color=GREY_TEXT),
    ]
    for i, ln in enumerate(slogan_lines):
        items.append(dict(text=ln, x=tx, cy=slog_cy0 + i * slog_lead, size=slog_size, tracking=0.08,
                          weight="regular", color=GREY_TEXT))
    last_cy = slog_cy0 + (len(slogan_lines) - 1) * slog_lead
    cat_y = (last_cy + 104 * u) if portrait else 0.688 * H
    items.append(dict(text=category.upper(), x=tx, cy=cat_y, size=11.5 * u, tracking=0.12, weight="bold", color=MUTED_TEXT))
    text_layer = _compose_items(W, H, items)
    bg = np.full((H, W, 3), 255, np.uint8)
    n = int(round(duration * fps))
    frames = []
    for i in range(n):
        t = i / float(fps)
        p = i / float(max(1, n - 1))
        z = 1.0 + (zmax - 1.0) * (p * p * (3 - 2 * p) * 0.35 + p * 0.65)
        f = bg.copy()
        s = z / zmax
        _paste(f, sp, cx, cy, s, s)
        f = overlay_layer(f, text_layer, fade_opacity(t, -1.0, 1e9, 0.0) * _smooth(t / 0.6))
        frames.append(f)
    return frames


# --------------------------------------------------------------------------- Endcard B (studio / podium)

def _studio_background(W, H, podium):
    cx, cy, rx, ry, side = podium
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # radial studio gradient, lighter toward the upper-middle, darker at the edges
    dx = (xx - 0.5 * W) / (0.75 * W)
    dy = (yy - 0.42 * H) / (0.75 * H) if H > W else (yy - 0.42 * H) / (0.95 * H)
    r = np.clip(np.sqrt(dx * dx + dy * dy), 0, 1.25)
    r = r ** 1.5
    c0, c1 = np.array([0xEA, 0xEA, 0xEC], np.float32), np.array([0xCC, 0xCC, 0xD0], np.float32)
    bg = c0[None, None, :] * (1 - r[..., None]) + c1[None, None, :] * r[..., None]
    # floor haze: very slightly lighter band behind the podium
    bg = bg[..., ::-1].copy()  # RGB -> BGR (symmetric palette anyway)

    ss = 3  # supersample masks for clean podium edges
    def ell_mask(ecx, ecy, erx, ery):
        m = np.zeros((H * ss, W * ss), np.uint8)
        cv2.ellipse(m, (int(ecx * ss), int(ecy * ss)), (int(erx * ss), int(ery * ss)), 0, 0, 360, 255, -1, cv2.LINE_8)
        return cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0

    top_m = ell_mask(cx, cy, rx, ry)
    bot_m = ell_mask(cx, cy + side, rx, ry)
    rect = np.zeros((H, W), np.float32)
    x0, x1 = int(round(cx - rx)), int(round(cx + rx))
    rect[int(round(cy)):int(round(cy + side)) + 1, x0:x1] = 1.0
    rect = cv2.GaussianBlur(rect, (0, 0), 0.6)
    body_m = np.clip(np.maximum(rect, bot_m), 0, 1)

    # ground shadow beneath the podium
    gs = _ellipse_blob(W, H, cx, cy + side + ry * 0.55, rx * 1.02, ry * 0.95, 0.38, ry * 0.45)
    bg = bg * (1 - gs[..., None])
    # soft contact darkening hugging the base
    gs2 = _ellipse_blob(W, H, cx, cy + side + ry * 0.2, rx * 0.99, ry * 1.02, 0.22, ry * 0.12)
    bg = bg * (1 - gs2[..., None])

    # cylinder side shading: bright center-left, darker toward the right edge, gentle vertical falloff
    nx = np.clip((xx - cx) / rx, -1, 1)
    shade = 0.935 - 0.20 * np.abs(nx) ** 2.2 - 0.04 * np.clip(nx, 0, 1)
    vy = np.clip((yy - cy) / (side + ry), 0, 1)
    shade = shade - 0.05 * vy
    body_col = (255.0 * shade)[..., None] * np.array([1.0, 0.995, 0.99], np.float32)[None, None, :]  # BGR slight warm
    bg = bg * (1 - body_m[..., None]) + body_col * body_m[..., None]

    # top face: near white, slightly brighter toward the back, soft falloff to the front rim
    tv = np.clip((yy - (cy - ry)) / (2 * ry), 0, 1)
    tnx = np.clip((xx - cx) / rx, -1, 1)
    tshade = 0.985 - 0.045 * tv - 0.03 * np.abs(tnx) ** 2
    top_col = (255.0 * tshade)[..., None] * np.ones((1, 1, 3), np.float32)
    bg = bg * (1 - top_m[..., None]) + top_col * top_m[..., None]

    # rim highlight (front edge of top face) and underside shadow line
    rim = np.zeros((H, W), np.float32)
    ring = np.zeros((H * ss, W * ss), np.uint8)
    cv2.ellipse(ring, (int(cx * ss), int(cy * ss)), (int(rx * ss), int(ry * ss)), 0, 8, 172, 255, max(1, int(1.4 * ss)), cv2.LINE_8)
    rim = cv2.resize(ring, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    bg = bg * (1 - 0.10 * rim[..., None])
    # thin seam line around the body at ~30% of its height (dark thin line + faint light line below)
    seam_y = cy + side * 0.30
    for off, col, alpha, th in ((0.0, (92, 100, 112), 0.70, 1.5), (2.2, (255, 255, 255), 0.55, 1.2)):
        ring = np.zeros((H * ss, W * ss), np.uint8)
        cv2.ellipse(ring, (int(cx * ss), int((seam_y + off) * ss)), (int((rx - 0.5) * ss), int(ry * ss)), 0, 0, 180,
                    255, max(1, int(th * ss)), cv2.LINE_8)
        sm = cv2.resize(ring, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0 * alpha
        sm = sm * body_m
        bg = bg * (1 - sm[..., None]) + np.array(col, np.float32)[None, None, :] * sm[..., None]
    bg = np.clip(bg + _grain(H, W, 1.0), 0, 255).astype(np.uint8)
    return bg, top_m


def build_endcard_studio(product_rgba_path, W, H, model_name, duration=10.0, fps=24):
    u = _unit(W, H)
    portrait = H > W
    rx = min(0.45 * W, 0.36 * H)
    ry = 0.22 * rx
    side = 0.42 * rx
    cx = 0.5 * W
    cy = (0.655 if portrait else 0.66) * H
    bg, top_m = _studio_background(W, H, (cx, cy, rx, ry, side))
    prod = load_product_rgba(product_rgba_path)
    push_max, sx_min = 1.06, 0.94
    pw = 1.28 * rx
    pm = _fit_product(prod, pw * push_max, 0.55 * H * push_max)
    ph, pwid = pm.shape[:2]
    # sprite: reflection (below), contact shadow, product. anchor = contact point at the product bottom center.
    padx = int(0.08 * pwid)
    refl_h = int(0.55 * ph * 0.5)
    top_pad = 4
    SW, SH = pwid + 2 * padx, top_pad + ph + int(0.5 * ry * push_max * 1.2) + refl_h + 8
    spr = np.zeros((SH, SW, 4), np.float32)
    base_y = top_pad + ph  # contact line inside the sprite
    # contact shadow on the top face (soft, elliptical, slightly offset)
    sh1 = _ellipse_blob(SW, SH, SW / 2.0, base_y, 0.43 * pwid, 0.32 * ry * push_max, 0.24, 0.20 * ry * push_max)
    sh2 = _ellipse_blob(SW, SH, SW / 2.0, base_y - 1, 0.33 * pwid, 0.10 * ry * push_max, 0.30, 0.07 * ry * push_max + 1)
    spr[..., 3] = np.clip(sh1 + sh2, 0, 0.42)
    # faint reflection: vertically flipped, squashed, fading out
    flip = pm[::-1].copy()
    flip = cv2.resize(flip, (pwid, refl_h), interpolation=cv2.INTER_AREA)
    fade = np.linspace(0.20, 0.0, refl_h, dtype=np.float32) ** 1.4 / (0.20 ** 0.4)
    flip = flip * fade[:, None, None]
    flip = cv2.GaussianBlur(flip, (0, 0), 1.4)
    refl = np.zeros_like(spr)
    refl[base_y + 2:base_y + 2 + refl_h, padx:padx + pwid] = flip
    spr = _over_premult(spr, refl)
    prem = np.zeros_like(spr)
    prem[top_pad:top_pad + ph, padx:padx + pwid] = pm
    spr = _over_premult(spr, prem)
    sp = _Sprite(spr, padx + pwid / 2.0, base_y)

    # text (rendered once)
    items = [
        dict(text="ESPNER", x=W / 2.0, cy=0.139 * H, size=50 * u, tracking=0.10, weight="bold", anchor="c", color=CHARCOAL),
        dict(text=str(model_name).upper(), x=W / 2.0, cy=0.139 * H + 62 * u, size=24 * u, tracking=0.20,
             weight="regular", anchor="c", color=GREY_TEXT),
    ]
    text_layer = _compose_items(W, H, items)

    n = int(round(duration * fps))
    foot_x, foot_y = cx, cy + 0.42 * ry   # contact point on the podium top
    c0x, c0y = 0.5 * W, 0.52 * H           # camera push center
    frames = []
    for i in range(n):
        t = i / float(fps)
        p = i / float(max(1, n - 1))
        push = 1.0 + (push_max - 1.0) * (0.5 - 0.5 * math.cos(math.pi * p))  # ease in-out push-in
        # gentle yaw simulation: horizontal squash 1 -> 0.94 -> 1 over the clip, tiny sway and bob
        yaw = 0.5 - 0.5 * math.cos(2 * math.pi * p)
        sx = 1.0 - (1.0 - sx_min) * yaw
        sway = 0.010 * pwid * math.sin(2 * math.pi * p) * (1.0)
        bob = 0.8 * u * math.sin(4 * math.pi * p)
        if push > 1.0005:
            M = np.array([[push, 0, c0x * (1 - push)], [0, push, c0y * (1 - push)]], np.float32)
            f = cv2.warpAffine(bg, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            f = bg.copy()
        px = c0x + push * (foot_x + sway - c0x)
        py = c0y + push * (foot_y + bob - c0y)
        k = push / push_max
        _paste(f, sp, px, py, sx * k, k)
        f = overlay_layer(f, text_layer, _smooth(t / 0.6))
        frames.append(f)
    return frames
"""

GRADE_CODE = r"""
# espner_grade.py
#
# Color grading helpers that move ESPNER frames toward two reference looks.
# Dependencies: numpy, opencv-python-headless only.
#
# Measured reference targets (mean over the sampled frames, Y = YCrCb luma):
#   Ref A warm film   : p1 17.5  p5 33.5  p50 156.8  p95 239.5  p99 242.3
#                       HSV S 55.1   Lab a* 2.5  b* 10.1   RMS contrast 0.25
#   Ref B bright natur: p1 28.6  p5 49.2  p50 158.8  p95 220.0  p99 230.2
#                       HSV S 47.4   Lab a* 0.9  b* 2.9    RMS contrast 0.19
#   Ours f_1 (raw)    : p1 4.0   p5 5.0  p50 37.0   p95 146.0  p99 206.0
#                       HSV S 108.6  Lab a* 3.2  b* 9.1    RMS contrast 0.19
#   Ours s2 (raw)     : p1 4.0   p5 8.0  p50 62.0   p95 172.0  p99 215.0
#                       HSV S 149.8  Lab a* 8.2  b* 17.4   RMS contrast 0.21
#
# Derivation summary:
#   lift   : the reference black point (p1) sits near 0.07 (A) and 0.11 (B);
#            ours is crushed at about 0.016, so the lift raises blacks to match.
#   gamma  : mapping our median (about 0.20) to the reference median (about 0.62)
#            needs gamma near 0.36 to 0.40. That is too aggressive for one
#            fixed curve, so the value is damped to 0.50 (A) and 0.48 (B).
#   knee / ceiling : the reference highlight ceiling (p99) is about 0.95 (A)
#            and 0.90 (B); the soft shoulder keeps highlights below clipping.
#   contrast: A has RMS 0.25 and B 0.19, against ours at about 0.20.
#   wb     : B is cooler than ours (b* 2.9 against about 13); A is warm but
#            not as yellow as our s2 frame. BGR gains are small so skin
#            tones stay natural.
#   sat    : HSV S of our raw frames is inflated by the dark exposure.
#            The chroma scale (YCrCb) is set below 1.0 to pull it toward the
#            references, then checked after grading.
#
# Public API:
#   GRADES                      dict of grade parameters
#   build_lut(grade_name)       256x1x3 uint8 BGR tone LUT for cv2.LUT
#   grade_frame(frame_bgr, grade_name, lut=None, strength=1.0)
#   add_grain(frame_bgr, amount=3.0, seed=None)

import math

import cv2
import numpy as np


GRADES = {
    # Warm lifestyle film look: lifted blacks, soft highlights, warm split.
    "A_warm_film": {
        "lift": 0.07,
        "gamma": 0.45,
        "contrast": 0.35,
        "knee": 0.86,
        "ceiling": 0.975,
        "wb": (0.99, 1.00, 1.01),   # (B, G, R) gains
        "sat": 0.75,                # chroma scale, 1.0 = unchanged
        "vignette": 0.12,           # corner darkening, 0 = off
    },
    # Bright natural coastal look: brighter mids, cooler, airy highlights.
    "B_bright_natural": {
        "lift": 0.10,
        "gamma": 0.45,
        "contrast": 0.15,
        "knee": 0.86,
        "ceiling": 0.97,
        "wb": (1.04, 1.00, 0.97),
        "sat": 0.80,
        "vignette": 0.06,
    },
    # Identity grade: pass-through, useful as a control.
    "neutral": {
        "lift": 0.0,
        "gamma": 1.0,
        "contrast": 0.0,
        "knee": 1.0,
        "ceiling": 1.0,
        "wb": (1.0, 1.0, 1.0),
        "sat": 1.0,
        "vignette": 0.0,
    },
}

_LUT_CACHE = {}
_AUX_CACHE = {}
_GRAIN_CACHE = {}


def _params(grade_name):
    if grade_name not in GRADES:
        raise ValueError("unknown grade: %s (have %s)" % (grade_name, sorted(GRADES)))
    return GRADES[grade_name]


def _tone_curve(p, x):
    # Lift, gamma, symmetric S-curve, then soft highlight shoulder.
    lift = float(p["lift"])
    y0 = np.clip(lift + (1.0 - lift) * x, 0.0, 1.0)
    # Gamma acts on mids only: shadows keep the lift-only value so the
    # black point does not float up, and highlights are left to the shoulder.
    lo = float(p.get("mid_from", 0.04))
    hi = float(p.get("mid_to", 0.22))
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    w = t * t * (3.0 - 2.0 * t)
    y1 = y0 ** float(p["gamma"])
    y = y0 + (y1 - y0) * w
    c = float(p["contrast"])
    if c != 0.0:
        y = y + c * y * (1.0 - y) * (2.0 * y - 1.0)
    knee = float(p["knee"])
    ceiling = float(p["ceiling"])
    if knee < 1.0 and ceiling > knee:
        span = ceiling - knee
        over = y > knee
        shoulder = knee + span * (1.0 - np.exp(-(y - knee) / span))
        y = np.where(over, shoulder, y)
    return np.clip(y, 0.0, 1.0)


def build_lut(grade_name):
    # 256x1x3 uint8 BGR tone curves for cv2.LUT. Each channel is the tone
    # curve times its white-balance gain.
    _params(grade_name)
    key = ("lut", grade_name)
    if key not in _LUT_CACHE:
        p = GRADES[grade_name]
        x = np.arange(256, dtype=np.float32) / 255.0
        base = _tone_curve(p, x)
        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        for ch in range(3):
            v = np.clip(base * float(p["wb"][ch]), 0.0, 1.0) * 255.0
            lut[:, 0, ch] = np.clip(np.round(v), 0, 255).astype(np.uint8)
        _LUT_CACHE[key] = lut
    return _LUT_CACHE[key].copy()


def _chroma_lut(sat):
    # YCrCb LUT: luma untouched, Cr and Cb scaled around 128.
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    idx = np.arange(256, dtype=np.float32)
    lut[:, 0, 0] = np.arange(256, dtype=np.uint8)
    chroma = np.clip(128.0 + (idx - 128.0) * float(sat), 0, 255)
    lut[:, 0, 1] = np.round(chroma).astype(np.uint8)
    lut[:, 0, 2] = np.round(chroma).astype(np.uint8)
    return lut


def _vignette_gain(h, w, strength):
    # Fixed-point (x256) gain map: 256 at the centre, darker toward corners.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    d = np.sqrt(((xx - cx) / (w / 2.0)) ** 2 + ((yy - cy) / (h / 2.0)) ** 2)
    t = np.clip((d - 0.55) / (1.41 - 0.55), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    gain = 1.0 - float(strength) * t
    return np.clip(np.round(gain * 256.0), 0, 256).astype(np.uint16)


def _aux(shape, grade_name):
    # Per-(shape, grade) cache: chroma LUT and vignette gain map.
    key = (shape[0], shape[1], grade_name)
    if key not in _AUX_CACHE:
        p = _params(grade_name)
        chroma = _chroma_lut(p["sat"]) if float(p["sat"]) != 1.0 else None
        vign = None
        if float(p["vignette"]) > 0.0:
            vign = _vignette_gain(shape[0], shape[1], p["vignette"])[:, :, None]
        _AUX_CACHE[key] = (chroma, vign)
    return _AUX_CACHE[key]


def grade_frame(frame_bgr, grade_name, lut=None, strength=1.0):
    # Apply tone LUT, chroma (saturation) scale, vignette, then blend.
    if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3:
        raise ValueError("grade_frame expects a uint8 BGR image")
    frame_bgr = np.ascontiguousarray(frame_bgr)
    if lut is None:
        lut = build_lut(grade_name)
    chroma, vign = _aux(frame_bgr.shape, grade_name)

    out = cv2.LUT(frame_bgr, lut)
    if chroma is not None:
        ycc = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
        ycc = cv2.LUT(ycc, chroma)
        out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    if vign is not None:
        out = ((out.astype(np.uint16) * vign) >> 8).astype(np.uint8)

    s = min(max(float(strength), 0.0), 1.0)
    if s < 1.0:
        out = cv2.addWeighted(frame_bgr, 1.0 - s, out, s, 0.0)
    return out


def _grain_tiles(h, w, amount):
    # Two uint8 tiles per call (positive and negative parts of a gaussian
    # noise field), 3 channels identical so the grain is monochrome.
    # Tiles are slightly larger than the frame so each frame can take a
    # random crop, which changes the grain pattern frame to frame.
    key = (h, w, round(float(amount) * 10))
    if key not in _GRAIN_CACHE:
        rng = np.random.default_rng(20240601)
        pad = 32
        tiles = []
        for _ in range(4):
            n = rng.normal(0.0, float(amount), (h + pad, w + pad))
            pos = np.clip(np.round(n), 0, 255).astype(np.uint8)
            neg = np.clip(np.round(-n), 0, 255).astype(np.uint8)
            tiles.append((cv2.merge([pos, pos, pos]), cv2.merge([neg, neg, neg])))
        _GRAIN_CACHE[key] = tiles
    return _GRAIN_CACHE[key]


def add_grain(frame_bgr, amount=3.0, seed=None):
    # Fine monochrome film grain. amount is the noise sigma in 8-bit levels.
    if frame_bgr.dtype != np.uint8 or frame_bgr.ndim != 3:
        raise ValueError("add_grain expects a uint8 BGR image")
    if amount <= 0:
        return frame_bgr.copy()
    h, w = frame_bgr.shape[:2]
    tiles = _grain_tiles(h, w, amount)
    if seed is None:
        ti = int(np.random.randint(0, len(tiles)))
        y0 = int(np.random.randint(0, 33))
        x0 = int(np.random.randint(0, 33))
    else:
        rng = np.random.default_rng(seed)
        ti = int(rng.integers(0, len(tiles)))
        y0 = int(rng.integers(0, 33))
        x0 = int(rng.integers(0, 33))
    pos_t, neg_t = tiles[ti]
    pos = np.ascontiguousarray(pos_t[y0:y0 + h, x0:x0 + w])
    neg = np.ascontiguousarray(neg_t[y0:y0 + h, x0:x0 + w])
    out = cv2.add(frame_bgr, pos)
    out = cv2.subtract(out, neg)
    return out
"""

AUDIO_CODE = r"""
# ESPNER procedural soundtrack: synth_music_bed / mux_audio / pick_mood
# numpy + ffmpeg CLI only. Fully procedural (royalty-free), no downloads.
import json
import os
import re
import subprocess
import wave

import numpy as np

_TWO_PI = 2.0 * np.pi


def _mtof(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=np.float64) - 69.0) / 12.0)


# ---------------------------------------------------------------- mood presets
# chords: list of (voicing midi list, bass midi). last entry of the cycle is the
# "suspended / dominant" chord; the final resolve chord is `final`.
_MOODS = {
    "warm": dict(
        bpm=88, bar_beats=4, key=0,
        chords=[([55, 59, 64, 67], 36), ([55, 60, 64, 67], 33),
                ([53, 57, 60, 64], 41), ([55, 60, 62, 65], 43)],
        final=([52, 55, 59, 64, 67], 36),
        pad_tilt=1.6, pad_amp=0.55, pad_attack=0.55, cutoff=5200, rt60=2.8,
        send=0.42, wet=0.55, arp=True, bass_amp=0.55, shimmer=0.0, noise=0.0,
        target_rms=-20.5),
    "airy": dict(
        bpm=60, bar_beats=8, key=0,
        chords=[([60, 64, 67, 71, 74], 41), ([62, 67, 71, 74, 78], 43),
                ([57, 64, 67, 72, 76], 45), ([55, 60, 62, 67, 69], 43)],
        final=([60, 64, 67, 71, 76], 48),
        pad_tilt=1.0, pad_amp=0.50, pad_attack=1.8, cutoff=9500, rt60=5.0,
        send=0.6, wet=0.75, arp=False, bass_amp=0.18, shimmer=0.5, noise=0.55,
        target_rms=-24.0),
    "noir": dict(
        bpm=62, bar_beats=4, key=0,
        chords=[([52, 57, 60, 64], 33), ([53, 57, 60, 64], 29 + 12),
                ([50, 57, 60, 65], 38), ([52, 56, 59, 62], 40)],
        final=([45, 52, 57, 60, 64], 33),
        pad_tilt=2.1, pad_amp=0.60, pad_attack=1.0, cutoff=2300, rt60=4.2,
        send=0.5, wet=0.6, arp=False, bass_amp=0.75, shimmer=0.0, noise=0.0,
        target_rms=-22.0),
    "neon": dict(
        bpm=108, bar_beats=4, key=0,
        chords=[([57, 60, 64, 67], 33), ([53, 57, 60, 64], 29 + 12),
                ([55, 60, 64, 67], 36), ([55, 59, 62, 67], 31 + 12)],
        final=([57, 60, 64, 69], 33),
        pad_tilt=1.3, pad_amp=0.5, pad_attack=0.25, cutoff=6500, rt60=1.9,
        send=0.35, wet=0.4, arp=False, bass_amp=0.7, shimmer=0.0, noise=0.0,
        target_rms=-19.5),
}


def pick_mood(style_key):
    k = str(style_key or "")
    if k.startswith("A_"):
        return "warm"
    if k.startswith("B_"):
        return "airy"
    if k == "3_HAUTE_COUTURE_NOIR":
        return "noir"
    if k == "1_CYBERPUNK_NEON":
        return "neon"
    return "warm"


# ---------------------------------------------------------------- voices
def _add(buf, start_s, sig_l, sig_r, sr):
    i0 = int(round(start_s * sr))
    if i0 >= buf.shape[1]:
        return
    n = min(sig_l.shape[0], buf.shape[1] - i0)
    if n <= 0:
        return
    buf[0, i0:i0 + n] += sig_l[:n]
    buf[1, i0:i0 + n] += sig_r[:n]


def _adsr_pad(n, sr, a, hold, rel):
    t = np.arange(n, dtype=np.float64) / sr
    env = np.ones(n)
    m = t < a
    env[m] = np.sin(0.5 * np.pi * t[m] / a) ** 2
    m = t > hold
    env[m] = np.cos(0.5 * np.pi * np.clip((t[m] - hold) / rel, 0, 1)) ** 2
    return env, t


def _pad_voice(freq, hold, sr, attack, rel, tilt, amp, detune_c=4.0, maxf=9000.0):
    # additive pad: harmonics with 1/k^tilt, L/R slightly detuned (+/- cents)
    n = int((hold + rel) * sr)
    env, t = _adsr_pad(n, sr, attack, hold, rel)
    out = []
    for sgn in (1.0, -1.0):
        f = freq * 2.0 ** (sgn * detune_c / 1200.0)
        ph = _TWO_PI * f * t
        s1 = np.sin(ph).astype(np.float32)
        c2 = (2.0 * np.cos(ph)).astype(np.float32)
        # Chebyshev recurrence: sin((k+1)x) = 2cos(x) sin(kx) - sin((k-1)x)
        prev = np.zeros(n, np.float32)
        cur = s1
        acc = s1.copy()
        k = 1
        while k < 10 and f * (k + 1) < maxf:
            nxt = c2 * cur - prev
            prev, cur = cur, nxt
            k += 1
            acc += cur * np.float32(1.0 / (k ** tilt))
        out.append(acc * (env * amp).astype(np.float32))
    return out[0], out[1]


def _pluck(freq, dur, sr, amp, bright=1.0, pan=0.0, tau=None):
    # soft piano-like pluck: harmonics decay faster the higher they are
    n = int(dur * sr)
    t = np.arange(n, dtype=np.float64) / sr
    tau = tau or (1.4 / (1.0 + freq / 700.0) + 0.35)
    ph = _TWO_PI * freq * t
    s = np.zeros(n)
    for k in range(1, 8):
        if freq * k > 9000:
            break
        b = 1.0 + 0.00025 * k * k  # slight inharmonicity
        a = (1.0 / k ** (1.5 / bright)) * np.exp(-t * (k - 1) * 1.6 / tau)
        s += a * np.sin(ph * k * b)
    atk = 1.0 - np.exp(-t / 0.004)
    env = atk * np.exp(-t / tau) * np.clip((dur - t) / 0.08, 0, 1)
    s = s * env * amp
    gl = np.sqrt(0.5 * (1 - pan))
    gr = np.sqrt(0.5 * (1 + pan))
    return (s * gl).astype(np.float32), (s * gr).astype(np.float32)


def _bell(freq, dur, sr, amp, pan, ratio=3.5, index=2.2):
    n = int(dur * sr)
    t = np.arange(n, dtype=np.float64) / sr
    tau = dur / 4.5
    mod = index * np.exp(-t / (tau * 0.5)) * np.sin(_TWO_PI * freq * ratio * t)
    s = np.sin(_TWO_PI * freq * t + mod) * np.exp(-t / tau)
    s *= (1.0 - np.exp(-t / 0.006)) * np.clip((dur - t) / 0.2, 0, 1) * amp
    return ((s * np.sqrt(0.5 * (1 - pan))).astype(np.float32),
            (s * np.sqrt(0.5 * (1 + pan))).astype(np.float32))


def _bass(freq, hold, sr, amp, attack=0.04, rel=0.5):
    n = int((hold + rel) * sr)
    env, t = _adsr_pad(n, sr, attack, hold, rel)
    ph = _TWO_PI * freq * t
    s = np.sin(ph) + 0.28 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)
    s = (s * env * amp).astype(np.float32)
    return s, s


def _saw(freq, dur, sr, amp, detune_c=0.0, nh=12, attack=0.005, tau=0.25, maxf=7000.0):
    n = int(dur * sr)
    t = np.arange(n, dtype=np.float64) / sr
    f = freq * 2.0 ** (detune_c / 1200.0)
    ph = _TWO_PI * f * t
    s = np.zeros(n)
    for k in range(1, nh + 1):
        if f * k > maxf:
            break
        s += np.sin(k * ph) / k
    env = (1 - np.exp(-t / attack)) * np.exp(-t / tau) * np.clip((dur - t) / 0.03, 0, 1)
    return (s * env * amp).astype(np.float32)


# ---------------------------------------------------------------- reverb
def _comb_block(x, D, g):
    n = x.shape[-1]
    y = np.zeros_like(x)
    for k in range(0, n, D):
        e = min(k + D, n)
        if k == 0:
            y[..., k:e] = x[..., k:e]
        else:
            y[..., k:e] = x[..., k:e] + g * y[..., k - D:k - D + (e - k)]
    return y


def _allpass_block(x, D, g):
    n = x.shape[-1]
    v = np.zeros_like(x)
    for k in range(0, n, D):
        e = min(k + D, n)
        if k == 0:
            v[..., k:e] = x[..., k:e]
        else:
            v[..., k:e] = x[..., k:e] + g * v[..., k - D:k - D + (e - k)]
    vd = np.zeros_like(v)
    vd[..., D:] = v[..., :-D]
    return -g * v + vd


def _reverb(x, sr, rt60, tail_s=0.0):
    # Schroeder/Freeverb style: 8 parallel combs -> 4 series allpass, per channel
    # (channel R uses slightly different delays => decorrelated stereo tail)
    base = np.array([1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617], dtype=np.float64)
    ap = [556, 441, 341, 225]
    scale = sr / 44100.0
    out = np.zeros_like(x)
    for ch in range(2):
        spread = 23 * ch
        acc = np.zeros(x.shape[1], dtype=np.float32)
        xin = x[ch].astype(np.float32)
        # pre-delay ~ 18 ms
        pd = int(0.018 * sr)
        xin = np.concatenate([np.zeros(pd, np.float32), xin[:-pd]])
        for d in base:
            D = int((d + spread) * scale)
            g = 10.0 ** (-3.0 * D / (sr * rt60))
            acc += _comb_block(xin, D, g)
        acc /= 8.0
        for d in ap:
            acc = _allpass_block(acc, int((d + spread) * scale), 0.5)
        out[ch] = acc
    return out


def _fast_len(n):
    best = 1 << int(np.ceil(np.log2(n)))
    a = 1
    while a < best:
        b = a
        while b < best:
            c = b
            while c < best:
                if c >= n:
                    best = min(best, c)
                c *= 5
            b *= 3
        a *= 2
    return best


def _fft_eq(x, sr, hp=30.0, lp=8000.0):
    # zero-phase magnitude EQ: 2nd-order HP and LP (Butterworth-like)
    n = x.shape[-1]
    nf = _fast_len(n + 4096)
    X = np.fft.rfft(x, nf, axis=-1)
    f = np.fft.rfftfreq(nf, 1.0 / sr)
    H = 1.0 / np.sqrt(1.0 + (f / lp) ** 4)
    H *= 1.0 / np.sqrt(1.0 + (hp / np.maximum(f, 1e-3)) ** 4)
    return np.fft.irfft(X * H, nf, axis=-1)[..., :n]


def _write_wav(path, y, sr):
    pcm = np.clip(y, -1.0, 1.0)
    pcm = (pcm.T * 32767.0).round().astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- composer
def synth_music_bed(duration, out_wav, mood="warm", bpm=None, seed=0, sr=48000):
    mood = mood if mood in _MOODS else "warm"
    P = _MOODS[mood]
    rng = np.random.RandomState(int(seed) & 0x7FFFFFFF)
    T = float(duration)
    N = int(round(T * sr))
    bpm = float(bpm or P["bpm"])
    beat = 60.0 / bpm
    bar = beat * P["bar_beats"]
    resolve_t = max(T - 4.2, T * 0.6)  # final chord start
    n_cyc = len(P["chords"])

    # chord schedule: (start, end, voicing, bass)
    sched = []
    t0, i = 0.0, 0
    while t0 + bar * 0.75 < resolve_t:
        vo, bs = P["chords"][i % n_cyc]
        t1 = t0 + bar
        if t1 + bar * 0.75 >= resolve_t:
            t1 = resolve_t  # stretch last bar into the resolve chord
        sched.append((t0, t1, vo, bs))
        t0, i = t1, i + 1
    fvo, fbs = P["final"]
    sched.append((resolve_t, T + 1.0, fvo, fbs))

    pad = np.zeros((2, N), np.float32)
    pl = np.zeros((2, N), np.float32)
    bs_buf = np.zeros((2, N), np.float32)
    fx = np.zeros((2, N), np.float32)

    for ci, (a, b, vo, bsn) in enumerate(sched):
        final = ci == len(sched) - 1
        hold = (b - a) + (0.35 if not final else 0.0)
        rel = 1.6 if not final else max(T - a, 1.0)
        if final:
            hold = max(T - a - 1.5, 0.5)
        # pad voices (slightly staggered entry => less robotic)
        for vi, m in enumerate(vo):
            st = a + 0.02 * vi + rng.uniform(0, 0.03)
            att = P["pad_attack"] * (1.0 + 0.1 * vi)
            amp = P["pad_amp"] / (len(vo) ** 0.6) * (1.0 if not final else 1.05)
            l, r = _pad_voice(_mtof(m), hold, sr, att, rel, P["pad_tilt"], amp,
                              detune_c=3.0 + 1.5 * vi)
            _add(pad, st, l, r, sr)
        # bass
        l, r = _bass(_mtof(bsn), min(b - a, T - a) * 0.95 if not final else hold,
                     sr, P["bass_amp"] * 0.5, rel=0.7 if not final else rel)
        if mood == "noir":
            l, r = _bass(_mtof(bsn - 12 if bsn > 33 else bsn), (b - a) * 0.97, sr,
                         P["bass_amp"] * 0.55, attack=0.5, rel=1.2)
        _add(bs_buf, a, l, r, sr)

    # ---- warm: soft piano-like arpeggio, gentle pulse
    if P["arp"]:
        patt = [0, 2, 1, 3, 2, 1, 3, 2]
        for ci, (a, b, vo, bsn) in enumerate(sched[:-1]):
            if ci == 0:
                continue  # pad-only intro
            vv = sorted(vo)
            lift = 0.65 + 0.35 * np.sin(np.pi * min(1.0, a / max(resolve_t, 1)))  # swell mid-piece
            nstep = int((b - a) / (beat / 2) + 0.5)
            for s in range(nstep):
                if rng.rand() < 0.14 and s % 4 != 0:
                    continue
                m = vv[patt[s % 8] % len(vv)] + 12
                st = a + s * beat / 2 + rng.uniform(-0.012, 0.012)
                vel = (1.0 if s % 4 == 0 else 0.62) * rng.uniform(0.8, 1.05) * lift
                l, r = _pluck(_mtof(m), 2.6, sr, 0.20 * vel, bright=1.0,
                              pan=rng.uniform(-0.35, 0.35))
                _add(pl, st, l, r, sr)
        # final: rolled chord + one high note, then ring-out
        a = sched[-1][0]
        for j, m in enumerate(sorted(sched[-1][2])):
            l, r = _pluck(_mtof(m + 12), 3.5, sr, 0.17 * (1 - 0.08 * j), pan=-0.3 + 0.15 * j)
            _add(pl, a + 0.09 * j, l, r, sr)
        # soft beat-1/3 bass-pulse accent (the 'gentle pulse')
        for ci, (a, b, vo, bsn) in enumerate(sched[:-1]):
            for k in range(int((b - a) / (2 * beat) + 0.5)):
                l, r = _bass(_mtof(bsn + 12), 0.2, sr, 0.10 if k % 2 else 0.14, attack=0.01, rel=0.45)
                _add(bs_buf, a + k * 2 * beat, l, r, sr)

    # ---- airy: shimmer bells (pentatonic, sparse) + ocean-air noise
    if P["shimmer"] > 0:
        penta = [0, 2, 4, 7, 9]
        tt = 2.0
        while tt < T - 5.0:
            m = 84 + penta[rng.randint(5)] + 12 * (rng.rand() < 0.3) - 12 * (rng.rand() < 0.35)
            l, r = _bell(_mtof(m), 3.2, sr, 0.045 * P["shimmer"] * rng.uniform(0.6, 1.2),
                         pan=rng.uniform(-0.8, 0.8))
            _add(fx, tt, l, r, sr)
            tt += rng.uniform(1.4, 3.6)
        # one final soft bell on resolve
        l, r = _bell(_mtof(88), 3.8, sr, 0.05, pan=0.2)
        _add(fx, resolve_t + 0.4, l, r, sr)
    if P["noise"] > 0:
        nz = np.zeros((2, N), np.float32)
        tg = np.arange(N) / sr
        for ch in range(2):
            w = rng.randn(N)
            W = np.fft.rfft(w)
            f = np.fft.rfftfreq(N, 1.0 / sr)
            H = (1.0 / np.sqrt(1 + (150.0 / np.maximum(f, 1))**2 * 4)) / np.sqrt(1 + (f / 1800.0) ** 4)
            H /= np.sqrt(1 + np.maximum(f, 1) / 300.0)
            x = np.fft.irfft(W * H, N)
            x /= np.sqrt((x ** 2).mean()) + 1e-9
            sw = (0.5 + 0.5 * np.sin(_TWO_PI * tg / (9.0 + 2 * ch) + 1.3 * ch)) ** 2
            sw2 = (0.5 + 0.5 * np.sin(_TWO_PI * tg / 5.3 + 0.7 + ch)) ** 2
            nz[ch] = (x * (0.05 + 0.65 * sw + 0.3 * sw2) * 0.06 * P["noise"]).astype(np.float32)
        fx += nz

    # ---- noir: sparse low piano notes
    if mood == "noir":
        tt = beat * 2
        while tt < resolve_t - 2:
            cur = [s for s in sched if s[0] <= tt < s[1]][0]
            m = sorted(cur[2])[rng.randint(0, 3)] - (12 if rng.rand() < 0.4 else 0)
            l, r = _pluck(_mtof(m), 4.0, sr, 0.17 * rng.uniform(0.6, 1.0), bright=0.7,
                          pan=rng.uniform(-0.3, 0.3), tau=1.8)
            _add(pl, tt, l, r, sr)
            tt += beat * rng.choice([2, 3, 4, 4, 6])

    # ---- neon: saw arp, pumping pad (sidechain), soft kick
    if mood == "neon":
        sc = np.ones(N, np.float32)
        tg = np.arange(N, dtype=np.float64) / sr
        ph_b = np.mod(tg, beat)
        sc = (1.0 - 0.55 * np.exp(-ph_b / 0.11)).astype(np.float32)
        pad *= sc[None, :]
        sixt = beat / 4
        for ci, (a, b, vo, bsn) in enumerate(sched[:-1]):
            vv = sorted(vo)
            arpn = [vv[0] + 12, vv[2] + 12, vv[3] + 12, vv[2] + 12, vv[1] + 12, vv[3] + 12, vv[2] + 12, vv[0] + 24]
            for s in range(int((b - a) / sixt)):
                if ci == 0 and s < 16:
                    continue
                if s % 2 == 1 and rng.rand() < 0.2:
                    continue
                m = arpn[s % 8]
                vel = 1.0 if s % 4 == 0 else 0.55
                x = _saw(_mtof(m), 0.22, sr, 0.055 * vel, detune_c=0, tau=0.11)
                x2 = _saw(_mtof(m), 0.22, sr, 0.055 * vel, detune_c=9, tau=0.11)
                _add(pl, a + s * sixt, x, x2, sr)
            for s in range(int((b - a) / (beat / 2))):
                if s % 2 == 0 and False:
                    continue
                l, r = _bass(_mtof(bsn), beat * 0.4, sr, 0.20 if s % 2 else 0.28, attack=0.01, rel=0.08)
                _add(bs_buf, a + s * beat / 2, l, r, sr)
        k = 0
        while k * beat < resolve_t:
            n = int(0.28 * sr)
            t = np.arange(n) / sr
            fr = 45 + 90 * np.exp(-t / 0.04)
            ph = _TWO_PI * np.cumsum(fr) / sr
            kk = (np.sin(ph) * np.exp(-t / 0.09) * 0.45).astype(np.float32)
            _add(bs_buf, k * beat, kk, kk, sr)
            k += 1
        a = sched[-1][0]
        for j, m in enumerate(sorted(sched[-1][2])):
            x = _saw(_mtof(m + 12), 3.0, sr, 0.05, tau=1.2)
            _add(pl, a + 0.06 * j, x, x, sr)

    # ---- mix: reverb send, EQ, level, ring-out, fade-in, soft limiter
    send = (pad + pl * 0.8 + fx * 0.7) * P["send"]
    wet = _reverb(send, sr, P["rt60"])
    wet = _fft_eq(wet, sr, hp=120.0, lp=4800.0 if mood != "airy" else 8500.0)
    mix = pad + pl + bs_buf + fx + wet * P["wet"] * 2.2
    mix = _fft_eq(mix, sr, hp=32.0, lp=P["cutoff"])

    tg = np.arange(N, dtype=np.float64) / sr
    g = np.ones(N)
    fi = min(1.0, T / 4)
    m = tg < fi
    g[m] = np.sin(0.5 * np.pi * tg[m] / fi) ** 2
    ro = min(3.0, T / 3)
    m = tg > T - ro
    x = (tg[m] - (T - ro)) / ro
    g[m] = (1 - x) ** 2 * (1 + 0.0 * x)  # ring-out: smooth decay to silence
    mix *= g[None, :]
    # tiny edge guard so the file starts/ends at exactly zero
    mix[:, :16] *= np.linspace(0, 1, 16)
    mix[:, -64:] *= np.linspace(1, 0, 64)

    lo, hi = int(sr * min(1.5, T * 0.1)), int(sr * max(T - 4.0, T * 0.5))
    rms = np.sqrt((mix[:, lo:hi] ** 2).mean()) + 1e-9
    mix *= 10.0 ** (P["target_rms"] / 20.0) / rms
    ceil = 0.88  # about -1.1 dBFS
    mix = ceil * np.tanh(mix / ceil)
    mix -= mix.mean(axis=1, keepdims=True) * 0  # DC already removed by HP
    _write_wav(out_wav, mix, sr)
    return out_wav


# ---------------------------------------------------------------- mux
def _run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _probe_duration(path):
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
              "stream=duration", "-of", "default=nw=1:nk=1", path])
    try:
        d = float(r.stdout.strip().split()[0])
        if d > 0:
            return d
    except Exception:
        pass
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
              "-of", "default=nw=1:nk=1", path])
    return float(r.stdout.strip())


def mux_audio(video_in, audio_in, video_out, target_lufs=-16.0, fade_in=0.5, fade_out=1.5):
    D = _probe_duration(video_in)
    fi = max(0.0, min(float(fade_in), D / 3))
    fo = max(0.0, min(float(fade_out), D / 3))
    base = "aresample=48000,apad=whole_dur=%.3f,atrim=0:%.3f,asetpts=N/SR/TB" % (D, D)
    ln = "loudnorm=I=%.1f:TP=-1.5:LRA=11" % target_lufs
    # pass 1: measure
    r = _run(["ffmpeg", "-hide_banner", "-nostdin", "-i", audio_in, "-vn", "-ac", "2", "-af",
              base + "," + ln + ":print_format=json", "-f", "null", "-"])
    meas = None
    mm = re.findall(r"\{[^{}]*\"input_i\"[^{}]*\}", r.stderr, re.S)
    if mm:
        try:
            j = json.loads(mm[-1])
            if np.isfinite(float(j["input_i"])) and float(j["input_i"]) > -70:
                meas = j
        except Exception:
            meas = None
    if meas:
        ln = ("%s:measured_I=%s:measured_TP=%s:measured_LRA=%s:measured_thresh=%s:"
              "offset=%s:linear=true" % (ln, meas["input_i"], meas["input_tp"], meas["input_lra"],
                                         meas["input_thresh"], meas["target_offset"]))
    fades = []
    if fi > 0:
        fades.append("afade=t=in:st=0:d=%.3f" % fi)
    if fo > 0:
        fades.append("afade=t=out:st=%.3f:d=%.3f" % (D - fo, fo))
    chain = ",".join([base, ln, "aresample=48000"] + fades)
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-i", video_in, "-i", audio_in,
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-af", chain,
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
           "-t", "%.3f" % D, "-shortest", "-movflags", "+faststart", video_out]
    r = _run(cmd)
    if r.returncode != 0:
        raise RuntimeError("mux_audio ffmpeg failed: " + r.stderr[-800:])
    return video_out
"""

QC_CODE = r"""
# -*- coding: utf-8 -*-
# ESPNER video quality check (espner_qc.py)
#
# Scores a finished ad video shot by shot and turns the numbers into a plain
# Traditional Chinese report for a non-technical user.
#
# Public API:
#   analyze_shot(frames_bgr)              -> dict of metrics for one shot
#   analyze_video(mp4_path, shot_bounds)  -> dict with per-shot results, grade, issues
#   format_report(result)                 -> zh-TW multi-line report string
#
# Dependencies: numpy, opencv-python-headless, ffprobe CLI (no torch, no scipy).
#
# Metric definitions
#   sharpness           median variance of Laplacian over frames (gray, 360 px wide)
#   sharpness_min_ratio worst contrast-normalised frame / median ("mushy" frames);
#                       frames without contrast (fades to white) are skipped
#   ghosting            share of frames flagged by the motion-compensated ghost test.
#                       Each frame is predicted from its t-1 and t+1 neighbours after DIS
#                       optical flow (180 px wide) has removed real motion. A frame is
#                       flagged when the leftover error is large on average (> GHOST_MEAN)
#                       and a visible share of pixels (> GHOST_PIX_FRAC) is off by more
#                       than GHOST_PIX_DIFF. Double exposure and AI morph flicker survive
#                       the motion model; ordinary camera motion does not.
#   flicker             std of per-frame mean luma after removing a moving average (fades
#                       and slow ramps are not flicker)
#   motion              mean absolute difference between consecutive gray frames (0..255)
#   face_rate           share of every 8th frame where the Haar frontal face cascade fires
#   skin_frac           share of skin-tone pixels; used to tell people from product shots
#
# Calibration (reference ads refA/refB, v.mp4, and a synthetic double-exposure clip):
#   ghosting per shot: refs 0.00-0.06, synthetic 0.56   -> GHOST_SHOT_FRAC 0.15
#   motion per shot:   v.mp4 0.10-0.31, refs >= 0.84    -> STATIC_MOTION 0.575
#   motion, fast refA whip shot 10.4, same at 2x speed 15.9 -> VIOLENT_MOTION 14
#   sharpness: refs 106-613, synthetic 21              -> BLUR_LIMIT 40
#   Haar frontal face misses back and profile shots, so the face check is a minor
#   (penalty 5) hint, gated by skin presence.

import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

ANALYSIS_WIDTH = 360          # sharpness, luma and face use this width
FLOW_WIDTH = 180              # optical flow / ghost residual use this width
FACE_EVERY = 8                # Haar face test on every 8th frame
CUT_PIXDIFF = 25.0            # mean abs diff of 96x54 gray frames: hard cut
CUT_PIXDIFF_SOFT = 12.0       # soft cut: needs a histogram change as well
CUT_HIST = 0.25               # Bhattacharyya distance of HS histograms
MIN_SHOT_FRAMES = 8           # shorter segments are merged into the previous shot

GHOST_MEAN = 2.2              # average residual (0..255) the motion model cannot explain
GHOST_PIX_FRAC = 0.01         # share of pixels with a large residual
GHOST_PIX_DIFF = 20.0         # what counts as a large residual (0..255)
GHOST_SHOT_FRAC = 0.15        # shot is ghosted when this share of its frames is flagged

BLUR_LIMIT = 40.0             # median Laplacian variance at 360 px below this = blurred
MUSHY_RATIO = 0.25            # worst contrast-normalised frame below this x median = mushy
FLICKER_TREND = 15            # moving-average window (frames) removed before the flicker std
FLICKER_LUMA = 4.0            # std of detrended per-frame mean luma (fades are not flicker)
STATIC_MOTION = 0.575         # consecutive-frame mean diff below this = almost no motion
VIOLENT_MOTION = 14.0         # consecutive-frame mean diff above this = too violent
FACE_MIN_RATE = 0.05          # below this share of sampled frames with a frontal face...
SKIN_MIN = 0.08               # ...AND this share of skin-tone pixels = person without a face
ENDCARD_BRIGHT = 0.60         # share of near-white pixels that marks an endcard

PENALTY = {
    "ghost": 40, "blur": 25, "mushy": 15, "flicker": 15,
    "static": 25, "violent": 20, "face": 5,
}

PHRASE = {
    "ghost": "有殘影（畫面疑似重影或變形）",
    "blur": "畫面模糊",
    "mushy": "部分畫面突然變糊",
    "flicker": "畫面亮度閃爍",
    "static": "幾乎沒有動",
    "violent": "動作太劇烈",
    "face": "人物臉部偵測不到（可能變形）",
}

SUGGEST = {
    "ghost": "重新生成有殘影的段落，並降低人物動作幅度或鏡頭移動速度。",
    "blur": "提高輸出解析度，或重新生成模糊的段落。",
    "mushy": "檢查變糊段落的前後銜接，必要時剪掉變糊的幾秒。",
    "flicker": "統一各段亮度與色調，避免忽明忽暗。",
    "static": "加入推近、平移或人物轉頭等動作，讓畫面不要靜止太久。",
    "violent": "放慢動作速度，讓人物與鏡頭的移動更平順。",
    "face": "確認人物臉部清楚可見，必要時重新生成該段。",
}

_FACE_CASCADE = None
_TLS = threading.local()    # one DIS flow object per worker thread


def _face_cascade():
    # Haar frontal face cascade, loaded once (bundled with opencv-python-headless).
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _FACE_CASCADE


def _dis():
    # DIS optical flow (medium preset), one instance per thread.
    if getattr(_TLS, "dis", None) is None:
        _TLS.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _TLS.dis


def _resize_w(img, width=ANALYSIS_WIDTH):
    # Resize keeping aspect ratio to the given width.
    h, w = img.shape[:2]
    if w == width:
        return img
    height = max(1, int(round(h * width / float(w))))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def _ghost_flags(g360):
    # Motion-compensated ghost test. g360: list of uint8 gray frames (360 px wide).
    # Returns (flags bool array, mean residual array, pixel-fraction array).
    n = len(g360)
    flags = np.zeros(n, dtype=bool)
    mean_res = np.zeros(n)
    pix_frac = np.zeros(n)
    if n < 3:
        return flags, mean_res, pix_frac
    small = [cv2.resize(g, (FLOW_WIDTH, max(1, int(round(g.shape[0] / 2.0)))),
                        interpolation=cv2.INTER_AREA) for g in g360]
    H, W = small[0].shape
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    def one(t):
        # Motion-compensated residual for frame t (independent of the others).
        cur = small[t]
        dis = _dis()
        u = dis.calc(cur, small[t - 1], None)   # cur(x) ~ prev(x + u)
        v = dis.calc(cur, small[t + 1], None)   # cur(x) ~ next(x + v)
        pm = cv2.remap(small[t - 1].astype(np.float32), gx + u[..., 0], gy + u[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        pn = cv2.remap(small[t + 1].astype(np.float32), gx + v[..., 0], gy + v[..., 1],
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        d = np.abs(cur.astype(np.float32) - 0.5 * (pm + pn))
        return t, float(d.mean()), float(np.mean(d > GHOST_PIX_DIFF))

    workers = max(1, min(4, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for t, mr, pf in ex.map(one, range(1, n - 1)):
            mean_res[t] = mr
            pix_frac[t] = pf
            flags[t] = (mr > GHOST_MEAN) and (pf > GHOST_PIX_FRAC)
    return flags, mean_res, pix_frac


def _face_rate(g360):
    # Share of every FACE_EVERY-th frame where the Haar frontal cascade finds a face.
    casc = _face_cascade()
    if casc.empty():
        return 0.0
    idx = list(range(0, len(g360), FACE_EVERY))
    if not idx:
        return 0.0
    hits = 0
    for i in idx:
        faces = casc.detectMultiScale(cv2.equalizeHist(g360[i]), scaleFactor=1.1,
                                      minNeighbors=5, minSize=(24, 24))
        if len(faces) > 0:
            hits += 1
    return hits / float(len(idx))


def _skin_frac(bgr):
    # Share of pixels in a skin-tone band (HSV). Separates people from product shots.
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 0] <= 22) & (hsv[..., 1] >= 35) & (hsv[..., 1] <= 170) & (hsv[..., 2] >= 90)
    return float(np.mean(m))


def _metrics(bgr360):
    # All shot metrics from a list of 360 px wide BGR frames.
    n = len(bgr360)
    if n == 0:
        return {"frames": 0}
    g360 = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in bgr360]
    lap = np.array([float(cv2.Laplacian(g, cv2.CV_64F).var()) for g in g360])
    var = np.array([float(g.std()) ** 2 for g in g360])
    med = max(float(np.median(lap)), 1e-6)
    # Mushy: worst frame relative to the median, judged only on frames that have contrast,
    # normalised by contrast so that fades to white do not look like blur.
    content = var >= 0.25 * max(float(np.median(var)), 1e-6)
    if int(content.sum()) >= 3:
        norm = lap[content] / np.maximum(var[content], 1.0)
        min_ratio = float(np.min(norm) / max(float(np.median(norm)), 1e-9))
    else:
        min_ratio = 1.0
    flags, mean_res, _pix = _ghost_flags(g360)
    means = np.array([float(g.mean()) for g in g360])
    if n > FLICKER_TREND:
        k = np.ones(FLICKER_TREND) / FLICKER_TREND
        trend = np.convolve(np.pad(means, FLICKER_TREND // 2, mode="edge"), k, mode="valid")[:n]
        flicker = float(np.std(means - trend))
    else:
        flicker = float(np.std(means))
    if n >= 2:
        motion = float(np.mean([np.mean(np.abs(g360[i].astype(np.int16) - g360[i - 1].astype(np.int16)))
                                for i in range(1, n)]))
    else:
        motion = 0.0
    sample = bgr360[::FACE_EVERY]
    skin = float(np.mean([_skin_frac(f) for f in sample]))
    inner = slice(1, max(1, n - 1))
    return {
        "frames": n,
        "sharpness": med,
        "sharpness_min_ratio": min_ratio,
        "ghosting": float(np.mean(flags)),
        "ghost_mean_res_p50": float(np.median(mean_res[inner])) if n >= 3 else 0.0,
        "flicker": flicker,
        "motion": motion,
        "face_rate": _face_rate(g360),
        "skin_frac": skin,
        "mean_luma": float(np.mean(means)),
        "bright_frac": float(np.mean([np.mean(g > 235) for g in g360])),
    }


def analyze_shot(frames_bgr):
    # Metrics for one shot. frames_bgr: list of HxWx3 uint8 BGR frames in order.
    # Returns the metric dict described in the module header.
    return _metrics([_resize_w(f) for f in frames_bgr])


def _probe_audio(mp4_path):
    # True when ffprobe finds at least one audio stream.
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "json", mp4_path],
            capture_output=True, text=True, timeout=60)
        data = json.loads(out.stdout or "{}")
        return len(data.get("streams", [])) > 0
    except Exception:
        return False


def _hs_hist(small_bgr):
    # Hue/saturation histogram used as a secondary cut signal.
    hsv = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [24, 8], [0, 180, 0, 256])
    cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
    return h


def _detect_shots(pix, hists):
    # A cut is a hard jump in pixels (96x54 gray), or a softer jump that also
    # changes the colour histogram. Short segments are merged into the previous one.
    n = len(pix)
    if n == 0:
        return []
    cuts = [0]
    for i in range(1, n):
        d_pix = float(np.mean(np.abs(pix[i] - pix[i - 1])))
        d_hist = cv2.compareHist(hists[i - 1], hists[i], cv2.HISTCMP_BHATTACHARYYA)
        if d_pix >= CUT_PIXDIFF or (d_pix >= CUT_PIXDIFF_SOFT and d_hist >= CUT_HIST):
            cuts.append(i)
    bounds = []
    for j, s in enumerate(cuts):
        e = cuts[j + 1] if j + 1 < len(cuts) else n
        bounds.append([s, e])
    merged = []
    for s, e in bounds:
        if merged and (e - s) < MIN_SHOT_FRAMES:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    if len(merged) > 1 and (merged[0][1] - merged[0][0]) < MIN_SHOT_FRAMES:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    return [(s, e) for s, e in merged]


def _judge(m):
    # Turn one shot's metrics into a list of problem keys (see PENALTY).
    problems = []
    if m.get("frames", 0) == 0:
        return problems
    endcard = m["bright_frac"] >= ENDCARD_BRIGHT
    if m["ghosting"] >= GHOST_SHOT_FRAC:
        problems.append("ghost")
    if m["sharpness"] < BLUR_LIMIT:
        problems.append("blur")
    if m["sharpness_min_ratio"] < MUSHY_RATIO and "blur" not in problems:
        problems.append("mushy")
    if m["flicker"] > FLICKER_LUMA:
        problems.append("flicker")
    if not endcard and m["motion"] < STATIC_MOTION:
        problems.append("static")
    if m["motion"] > VIOLENT_MOTION:
        problems.append("violent")
        # Very fast motion breaks the flow model and looks like ghosting; attribute it to motion.
        if "ghost" in problems:
            problems.remove("ghost")
    if not endcard and m["skin_frac"] >= SKIN_MIN and m["face_rate"] < FACE_MIN_RATE and m["frames"] >= 8:
        problems.append("face")
    return problems


def _shot_score(problems):
    return max(0, 100 - sum(PENALTY[p] for p in problems))


def _letter(score):
    if score >= 85:
        return "A"
    if score >= 72:
        return "B"
    if score >= 58:
        return "C"
    return "D"


def analyze_video(mp4_path, shot_bounds=None):
    # Analyze an mp4 shot by shot.
    # shot_bounds: list of (start_frame, end_frame) with end exclusive, or None to
    #              detect cuts automatically.
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise IOError("cannot open video: %s" % mp4_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    small_frames = []
    pix = []
    hists = []
    width = height = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if width == 0:
            height, width = frame.shape[:2]
        small = _resize_w(frame)
        small_frames.append(small)
        if shot_bounds is None:
            g96 = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (96, 54),
                             interpolation=cv2.INTER_AREA).astype(np.float32)
            pix.append(g96)
            hists.append(_hs_hist(small))
    cap.release()
    n = len(small_frames)
    duration = n / fps if fps > 0 else 0.0

    if shot_bounds is None:
        bounds = _detect_shots(pix, hists)
    else:
        bounds = []
        for s, e in shot_bounds:
            s, e = max(0, int(s)), min(n, int(e))
            if e > s:
                bounds.append((s, e))

    shots = []
    for idx, (s, e) in enumerate(bounds):
        m = _metrics(small_frames[s:e])
        problems = _judge(m)
        score = _shot_score(problems)
        shot = dict(m)
        shot.update({
            "index": idx + 1,
            "start_frame": s,
            "end_frame": e,
            "start_s": round(s / fps, 2),
            "end_s": round(e / fps, 2),
            "problems": problems,
            "score": score,
            "grade": _letter(score),
        })
        shots.append(shot)
    small_frames = None

    has_audio = _probe_audio(mp4_path)
    issues = []
    for sh in shots:
        if sh["problems"]:
            names = [PHRASE[p] for p in sorted(sh["problems"], key=lambda k: -PENALTY[k])]
            issues.append("第%d段（%.1f–%.1f 秒）：%s。" % (
                sh["index"], sh["start_s"], sh["end_s"], "、".join(names)))
    if not has_audio:
        issues.append("影片沒有聲音軌道，若需要配樂請重新輸出。")

    total = sum(sh["end_frame"] - sh["start_frame"] for sh in shots)
    if total > 0:
        overall_score = sum(sh["score"] * (sh["end_frame"] - sh["start_frame"])
                            for sh in shots) / float(total)
        ghost_share = sum(sh["end_frame"] - sh["start_frame"] for sh in shots
                          if "ghost" in sh["problems"]) / float(total)
    else:
        overall_score, ghost_share = 0.0, 0.0
    overall = _letter(overall_score)
    if ghost_share >= 0.25 and overall in ("A", "B"):
        overall = "C"

    return {
        "path": mp4_path,
        "duration": round(duration, 2),
        "fps": round(fps, 2),
        "size": [width, height],
        "has_audio": has_audio,
        "shots": shots,
        "overall_score": round(overall_score, 1),
        "overall_grade": overall,
        "issues": issues,
    }


_GRADE_TEXT = {
    "A": "優良，可直接使用",
    "B": "良好，有小問題可視情況修改",
    "C": "尚可，建議修改後再發布",
    "D": "不建議使用，請重新生成",
}


def _suggestions(result):
    # Top 2 suggestions, weighted by how much footage each problem touches.
    weight = {}
    for sh in result.get("shots", []):
        for p in sh.get("problems", []):
            weight[p] = weight.get(p, 0) + PENALTY[p] * (sh["end_frame"] - sh["start_frame"])
    ranked = sorted(weight.items(), key=lambda kv: -kv[1])
    out = [SUGGEST[p] for p, _ in ranked[:2]]
    if not out:
        out = ["畫面與動作表現良好，可以直接使用。"]
    return out


def format_report(result):
    # Plain zh-TW report for a non-technical user.
    lines = []
    w, h = result.get("size", [0, 0])
    lines.append("【影片品質檢查報告】")
    lines.append("影片長度：%.1f 秒｜解析度：%dx%d｜聲音：%s" % (
        result.get("duration", 0.0), w, h, "有" if result.get("has_audio") else "無"))
    lines.append("")
    lines.append("逐段檢查：")
    if not result.get("shots"):
        lines.append("（找不到可分析的畫面段落）")
    for sh in result.get("shots", []):
        time_txt = "%.1f–%.1f 秒" % (sh["start_s"], sh["end_s"])
        if not sh["problems"]:
            lines.append("✅ 第%d段（%s）：畫面清晰，動作自然" % (sh["index"], time_txt))
            continue
        probs = set(sh["problems"])
        labels = []
        if "blur" in probs and "ghost" in probs:
            labels.append("畫面模糊／有殘影")
            probs -= {"blur", "ghost"}
        labels += [PHRASE[p] for p in sorted(probs, key=lambda k: -PENALTY[k])]
        lines.append("⚠️ 第%d段（%s）：%s" % (sh["index"], time_txt, "、".join(labels)))
    lines.append("")
    grade = result.get("overall_grade", "D")
    lines.append("整體評等：%s（%s）" % (grade, _GRADE_TEXT.get(grade, "")))
    lines.append("")
    lines.append("建議：")
    for i, s in enumerate(_suggestions(result), 1):
        lines.append("%d. %s" % (i, s))
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    print(format_report(analyze_video(sys.argv[1])))
"""

STEP2_CODE = r"""
#@title 📸 步驟 2：設定產品名稱、廣告風格並上傳產品正面圖片
import os, json
from google.colab import files
from IPython.display import display, Image as IPImage

#@markdown ### 🏷️ 產品基本設定
MODEL_NAME = "AVIATOR" #@param {type:"string"}
PRODUCT_CATEGORY = "SUNGLASSES (精品太陽眼鏡)" #@param ["SUNGLASSES (精品太陽眼鏡)", "SNOW_GOGGLES (極限滑雪護目鏡)"]

#@markdown ### 🎨 選擇廣告視覺風格 (★ = 依參考廣告片標準的 30 秒版本；其餘為舊版 21 秒風格)
COMMERCIAL_STYLE = "A_STORY_30S (★參考片A標準：生活敘事 4 鏡 24s + 文案字幕 + 白底分割片尾 6s = 30s)" #@param ["A_STORY_30S (★參考片A標準：生活敘事 4 鏡 24s + 文案字幕 + 白底分割片尾 6s = 30s)", "B_COASTAL_30S (★參考片B標準：特寫→中景→漫步→英雄推近 4 鏡 20s + 灰階展台片尾 10s = 30s)", "0_CLASSIC_MADRID (經典精品：15s雙鏡故事 + 2s金框石牆過場 + 4.5s白底原圖片尾 = 21.5s)", "1_CYBERPUNK_NEON (賽博龐克：霓虹調色 + 光學HUD + 故障轉場 + 5s白底原圖片尾 = 21.0s)", "2_VINTAGE_1985 (復古1985：柯達暖金膠片 + 6s散景金框章節卡 + 5s白底原圖片尾 = 21.0s)", "3_HAUTE_COUTURE_NOIR (高訂黑白：巴黎米蘭黑白時尚邊框 + 5s白底原圖片尾 = 21.0s)", "4_ALPINE_EXTREME (極限冰川：冰藍橘紅HUD + 上下分割雙視角 + 5s白底原圖片尾 = 21.0s)", "5_RIVIERA_SUNSET (里維耶拉夕陽：地中海金色波光 + 5s白底原圖片尾 = 21.0s)"]

#@markdown ### 📐 輸出比例
ASPECT_RATIO = "9:16 直式 (IG Reels / 抖音 720x1280)" #@param ["9:16 直式 (IG Reels / 抖音 720x1280)", "16:9 橫式 (YouTube / 官網 1920x1080)"]

#@markdown ### ✍️ 文案字幕 (★參考片A 使用；用 / 分行；留空則該鏡不顯示文案)
COPY_LINE_1 = "你有多久，/沒好好看看世界？" #@param {type:"string"}
COPY_LINE_2 = "今天，/為自己出發。" #@param {type:"string"}
COPY_LINE_3 = "把目光，/還給世界。" #@param {type:"string"}

#@markdown ### 🎵 配樂
MUSIC_SOURCE = "1_自動生成配樂 (依風格，免版權)" #@param ["1_自動生成配樂 (依風格，免版權)", "2_上傳自己的音樂檔 (mp3/wav/m4a)", "3_無配樂"]

#@markdown ### 🤖 選擇標準 12.7GB RAM + T4 GPU 動態運算引擎
GENERATION_MODE = "方案 1A (超省RAM推薦)：LTX-Video-2B 拆離式圖生影片 + 真人模特兒配戴首幀 (僅需 5GB RAM)" #@param ["方案 1A (超省RAM推薦)：LTX-Video-2B 拆離式圖生影片 + 真人模特兒配戴首幀 (僅需 5GB RAM)", "方案 1B (高畫質省RAM)：Wan2.2-TI2V-5B 兩階段拆離圖生影片 + 真人模特兒配戴首幀 (僅需 6.5GB RAM)", "方案 1C (超輕量穩定)：SVD-XT 圖生影片 + 真人模特兒配戴首幀 (無文字編碼器、僅需 4.2GB RAM)", "方案 2 (零變形極速)：純產品 3D/2.5D 刀光立體運鏡 + 電影散景 (1分鐘極速完成、僅需 1.5GB RAM)"]

#@markdown ### 📂 圖片來源選擇
IMAGE_SOURCE = "1_點擊下方按鈕上傳圖片" #@param ["1_點擊下方按鈕上傳圖片", "2_直接使用雲端硬碟內建 AVIATOR 測試圖 (免手動上傳)", "3_直接使用雲端硬碟內建 MADRID 測試圖 (免手動上傳)", "4_直接使用雲端硬碟內建 ZERMATT 雪鏡測試圖 (免手動上傳)"]

#@markdown ### 🎬 自訂場景描述 (留空則依風格自動配對最佳模特兒與場景)
CUSTOM_SCENE_PROMPT = "" #@param {type:"string"}

os.makedirs("/content/input", exist_ok=True)
os.makedirs("/content/temp", exist_ok=True)

drive_sample_map = {
    "2_": ("/content/drive/MyDrive/Colab Notebooks/ESPNER_測試商品圖/AVIATOR_front.jpg", "AVIATOR"),
    "3_": ("/content/drive/MyDrive/Colab Notebooks/ESPNER_測試商品圖/MADRID_front.jpg", "MADRID"),
    "4_": ("/content/drive/MyDrive/Colab Notebooks/ESPNER_測試商品圖/ZERMATT_front.jpg", "ZERMATT")
}

PRODUCT_PHOTO_PATH = None
for prefix, (sample_p, sample_name) in drive_sample_map.items():
    if IMAGE_SOURCE.startswith(prefix) and os.path.exists(sample_p):
        PRODUCT_PHOTO_PATH = sample_p
        if MODEL_NAME == "AVIATOR":
            MODEL_NAME = sample_name
        break

if PRODUCT_PHOTO_PATH is None:
    # 若 /content/input 已經有剛剛上傳過的圖片且不想重傳，也可以直接選取
    print("👇 請點擊下方「選擇檔案 (Choose Files)」上傳一張產品正面照片 (.jpg / .png / .webp)：")
    uploaded = files.upload()
    if not uploaded:
        existing = [os.path.join("/content/input", x) for x in os.listdir("/content/input") if x.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        if existing:
            PRODUCT_PHOTO_PATH = existing[-1]
            print(f"ℹ️ 使用已存在之上傳圖片：{PRODUCT_PHOTO_PATH}")
        else:
            raise ValueError("❌ 未上傳任何圖片！請重新執行此儲存格並選擇產品正面圖。")
    else:
        fname = list(uploaded.keys())[0]
        PRODUCT_PHOTO_PATH = os.path.join("/content/input", fname)
        with open(PRODUCT_PHOTO_PATH, "wb") as f:
            f.write(uploaded[fname])

MUSIC_PATH = None
if MUSIC_SOURCE.startswith("2_"):
    print("🎵 請上傳一個音樂檔 (mp3 / wav / m4a)：")
    _up = files.upload()
    if _up:
        _mn = list(_up.keys())[0]
        MUSIC_PATH = os.path.join("/content/input", "music_" + _mn)
        with open(MUSIC_PATH, "wb") as _f:
            _f.write(_up[_mn])
    else:
        print("ℹ️ 未上傳音樂，改用自動生成配樂。")

cfg = {
    "model_name": MODEL_NAME.strip(),
    "product_category": PRODUCT_CATEGORY,
    "commercial_style": COMMERCIAL_STYLE,
    "generation_mode": GENERATION_MODE,
    "custom_scene_prompt": CUSTOM_SCENE_PROMPT.strip(),
    "photo_path": PRODUCT_PHOTO_PATH,
    "aspect": "16:9" if ASPECT_RATIO.startswith("16:9") else "9:16",
    "copy": [x.strip() for x in (COPY_LINE_1, COPY_LINE_2, COPY_LINE_3)],
    "music": "none" if MUSIC_SOURCE.startswith("3_") else ("file" if MUSIC_PATH else "auto"),
    "music_path": MUSIC_PATH
}
with open("/content/temp/job_config.json", "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=2)

print(f"\n✅ 成功載入產品圖片：{PRODUCT_PHOTO_PATH} (款式：{MODEL_NAME})")
display(IPImage(filename=PRODUCT_PHOTO_PATH, width=300))"""

STEP4_CODE = r"""
#@title 🎞️ 步驟 4：ESPNER 精品後製 (★30秒參考片標準：調色 + 文案字卡 + 片尾 + 配樂 ／ 舊版6風格) + 雲端備份與下載
import os, json, shutil, subprocess, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, HTML, Image as IPImage
from base64 import b64encode
from google.colab import files

# __EMBED_MODULES__

for _p, _step in [("/content/temp/job_config.json", "步驟 2"), ("/content/temp/raw_shot1.mp4", "步驟 3"), ("/content/temp/raw_shot2.mp4", "步驟 3")]:
    if not os.path.exists(_p):
        raise RuntimeError(f"❌ 找不到 {_p}，代表{_step}尚未成功完成 (或執行階段已重啟)。\n👉 請依序執行步驟 1 → 2 → 3，全部成功後再執行步驟 4。")

with open("/content/temp/job_config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)

PRODUCT_PHOTO_PATH = cfg["photo_path"]
MODEL_NAME = cfg["model_name"]
style_key = cfg["commercial_style"].split(" ")[0]
is_snow = "SNOW_GOGGLES" in cfg["product_category"] or style_key == "4_ALPINE_EXTREME"
shot1_mp4 = "/content/temp/raw_shot1.mp4"
shot2_mp4 = "/content/temp/raw_shot2.mp4"

def get_font_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[-1]

FONT_SERIF_BOLD = get_font_path([
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
])
FONT_SERIF_REG = get_font_path([
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
])
FONT_MONO = get_font_path([
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
])
FONT_CHINESE = get_font_path([
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
])

def draw_tracked_text(draw, center_y, text, font, fill, width=720, tracking=4):
    chars = list(text)
    if not chars: return
    char_widths = []
    for c in chars:
        bbox = font.getbbox(c)
        char_widths.append(bbox[2] - bbox[0])
    total_w = sum(char_widths) + (len(chars) - 1) * tracking
    curr_x = (width - total_w) // 2
    for i, c in enumerate(chars):
        draw.text((curr_x, center_y), c, font=font, fill=fill)
        curr_x += char_widths[i] + tracking

def load_video_frames(path, max_frames=None, resize=(720, 1280)):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, f = cap.read()
        if not ret: break
        if max_frames and len(frames) >= max_frames: break
        if resize and (f.shape[1] != resize[0] or f.shape[0] != resize[1]):
            f = cv2.resize(f, resize)
        frames.append(f)
    cap.release()
    return frames

def build_madrid_wall_transition_frames(display_name, is_snow=False, duration_sec=2.0, fps=24):
    num_frames = int(duration_sec * fps)
    w, h = 720, 1280
    wide_w = int(w * 1.35)
    rng = np.random.RandomState(108)
    noise = rng.normal(0, 3.5, (h, wide_w, 3)).astype(np.float32)
    base_wall = np.full((h, wide_w, 3), (225, 235, 240), dtype=np.float32) + noise
    wide_wall = cv2.GaussianBlur(np.clip(base_wall, 0, 255).astype(np.uint8), (9, 9), 0)

    font_brand = ImageFont.truetype(FONT_SERIF_BOLD, 50)
    font_model = ImageFont.truetype(FONT_SERIF_BOLD, 56 if len(display_name) > 10 else 62)
    font_sub = ImageFont.truetype(FONT_SERIF_REG, 21)

    INK_PRIMARY = (15, 23, 42)
    INK_SECONDARY = (51, 65, 85)
    INK_GOLD = (120, 80, 25)

    frames = []
    for idx in range(num_frames):
        t = idx / max(1, num_frames - 1)
        x_offset = int((wide_w - w) * (0.1 + 0.8 * t))
        frame_bgr = wide_wall[:, x_offset:x_offset + w].copy()

        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        text_layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(text_layer)

        box_y1, box_y2 = h // 2 - 160, h // 2 + 150
        box_x1, box_x2 = 70, w - 70
        draw.rectangle([(box_x1, box_y1), (box_x2, box_y2)], outline=INK_GOLD + (160,), width=2)
        draw.rectangle([(box_x1 + 6, box_y1 + 6), (box_x2 - 6, box_y2 - 6)], outline=INK_SECONDARY + (60,), width=1)

        if t < 0.52:
            alpha = int(255 * min(1.0, t / 0.15)) if t < 0.38 else int(255 * max(0.0, (0.52 - t) / 0.14))
            draw_tracked_text(draw, h // 2 - 35, "E S P N E R", font_brand, INK_PRIMARY + (alpha,), width=w, tracking=6)
        else:
            alpha = int(255 * min(1.0, (t - 0.50) / 0.15))
            draw_tracked_text(draw, h // 2 - 65, display_name.upper(), font_model, INK_PRIMARY + (alpha,), width=w, tracking=5)
            sub1 = "ESPNER ALPINE SNOW OPTICS" if is_snow else "ESPNER EYEWEAR ARCHIVE"
            sub2 = "DUAL-SPHERICAL POLARIZED SERIES" if is_snow else "POLARIZED LUXURY COLLECTION"
            draw_tracked_text(draw, h // 2 + 18, sub1, font_sub, INK_GOLD + (alpha,), width=w, tracking=4)
            draw_tracked_text(draw, h // 2 + 52, sub2, font_sub, INK_SECONDARY + (int(alpha * 0.8),), width=w, tracking=2)

        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), text_layer)
        out_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

        if idx >= num_frames - 10:
            fade_t = (idx - (num_frames - 10)) / 10.0
            out_bgr = cv2.addWeighted(out_bgr, 1.0 - fade_t, np.full((h, w, 3), 255, dtype=np.uint8), fade_t, 0)
        frames.append(out_bgr)
    return frames

def build_front_photo_endcard_frames(photo_p, display_name, is_snow=False, duration_sec=4.5, fps=24):
    num_frames = int(duration_sec * fps)
    w, h = 720, 1280

    pil_in = Image.open(photo_p).convert("RGBA")
    white_bg = Image.new("RGBA", pil_in.size, (255, 255, 255, 255))
    merged = Image.alpha_composite(white_bg, pil_in).convert("RGB")
    photo = cv2.cvtColor(np.array(merged), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    mask = gray < 248
    pts = np.argwhere(mask)
    if len(pts) > 0:
        min_y, min_x = pts.min(axis=0)
        max_y, max_x = pts.max(axis=0)
        pad = 4
        crop = photo[max(0, min_y - pad):min(photo.shape[0], max_y + pad),
                     max(0, min_x - pad):min(photo.shape[1], max_x + pad)]
    else:
        crop = photo

    target_base_w = 560
    base_h = max(1, int(crop.shape[0] * (target_base_w / crop.shape[1])))
    crop_base = cv2.resize(crop, (target_base_w, base_h), interpolation=cv2.INTER_LANCZOS4)

    font_brand = ImageFont.truetype(FONT_SERIF_BOLD, 46)
    font_model = ImageFont.truetype(FONT_SERIF_BOLD, 28)
    font_slogan = ImageFont.truetype(FONT_CHINESE, 34)
    font_sub = ImageFont.truetype(FONT_SERIF_REG, 19)

    white_canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    endcard_frames = []
    sub_label = "DUAL-SPHERICAL POLARIZED SNOW GOGGLE" if is_snow else "POLARIZED LUXURY EYEWEAR"

    for i in range(num_frames):
        t = i / max(1, num_frames - 1)
        scale = 1.0 + 0.035 * t
        cur_w = int(target_base_w * scale)
        cur_h = int(base_h * scale)
        scaled_crop = cv2.resize(crop_base, (cur_w, cur_h), interpolation=cv2.INTER_LANCZOS4)

        canvas = white_canvas.copy()

        sy = 665
        shadow = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(shadow, (w // 2, sy), (int(cur_w * 0.42), int(22 * scale)), 0, 0, 360, 255, -1)
        shadow = cv2.GaussianBlur(shadow, (71, 35), 0)
        shadow_val = (shadow.astype(np.float32) / 255.0 * 0.12)[..., np.newaxis]
        canvas = np.clip(canvas.astype(np.float32) * (1.0 - shadow_val), 0, 255).astype(np.uint8)

        cy = 530
        top_y = max(0, min(h - cur_h, cy - cur_h // 2))
        left_x = max(0, min(w - cur_w, (w - cur_w) // 2))

        crop_gray = cv2.cvtColor(scaled_crop, cv2.COLOR_BGR2GRAY)
        crop_mask = np.clip((255.0 - crop_gray.astype(np.float32)) / 15.0, 0.0, 1.0)[..., np.newaxis]

        roi = canvas[top_y:top_y+cur_h, left_x:left_x+cur_w].astype(np.float32)
        blended = roi * (1.0 - crop_mask) + scaled_crop.astype(np.float32) * crop_mask
        canvas[top_y:top_y+cur_h, left_x:left_x+cur_w] = np.clip(blended, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        draw_tracked_text(draw, 145, "E S P N E R", font_brand, (15, 23, 42), width=w, tracking=8)
        draw_tracked_text(draw, 210, display_name.upper(), font_model, (71, 85, 105), width=w, tracking=4)
        draw_tracked_text(draw, 840, "把目光，還給世界。", font_slogan, (15, 23, 42), width=w, tracking=4)
        draw_tracked_text(draw, 905, sub_label, font_sub, (100, 116, 139), width=w, tracking=3)

        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        if i < 10:
            fade_in = i / 10.0
            frame_bgr = cv2.addWeighted(frame_bgr, fade_in, white_canvas, 1.0 - fade_in, 0)
        endcard_frames.append(frame_bgr)
    return endcard_frames

w, h = 720, 1280
display_name = MODEL_NAME.strip().upper()

import sys
sys.path.insert(0, "/content/temp")
from shot_plan import resolve_shots, output_size, STYLE_PLANS, is_reference_style
import espner_titles as T
import espner_grade as G
import espner_audio as AU
IS_REF = is_reference_style(style_key)
TOTAL_FRAMES = None

def iter_reference_timeline():
    # ★ 參考片標準 30 秒剪輯：N 鏡 (調色 + 顆粒 + 品牌標 + 文案淡入淡出) → 片尾 (白底分割 / 灰階展台)
    plan = STYLE_PLANS[style_key]
    shots = resolve_shots(cfg, cfg.get("custom_scene_prompt_en", ""))
    W, H = output_size(cfg)
    copies = [c for c in (cfg.get("copy") or [])]
    if not any(copies):
        copies = list(plan.get("copy_default", []))
        if copies:
            print("ℹ️ 步驟 2 的文案是空的，改用預設文案")
    lut = G.build_lut(plan["grade"])
    mark = T.render_brand_mark_layer(W, H, position=plan["brand_mark"])
    prev_last = None
    for sh in shots:
        n = int(round(sh["dur"] * 24))
        frames = load_video_frames(f"/content/temp/raw_shot{sh['id']}.mp4", max_frames=n, resize=(W, H))
        while len(frames) < n:
            frames.append(frames[-1].copy())
        layer = None
        ci = sh.get("copy")
        if ci is not None and ci < len(copies) and copies[ci]:
            lines = [x for x in copies[ci].split("/") if x.strip()]
            print(f"  ✍️ 第 {sh['id']} 鏡文案：{' '.join(lines)}")
            layer = T.render_copy_layer(W, H, lines, brand_mark=False, align="left")
            if T.text_color_for(G.grade_frame(frames[len(frames) // 2], plan["grade"], lut=lut, strength=0.3), layer) == "light":
                layer = T.render_copy_layer(W, H, lines, brand_mark=False, align="left", color="light")
        for i, fr in enumerate(frames):
            t = i / 24.0
            out = G.grade_frame(fr, plan["grade"], lut=lut, strength=0.3)
            out = G.add_grain(out, amount=1.5, seed=i)
            if prev_last is not None and i < 4:
                # 4 格柔和溶接，避免硬切的跳動感
                out = cv2.addWeighted(prev_last, 1.0 - (i + 1) / 5.0, out, (i + 1) / 5.0, 0)
            if plan["brand_mark"] == "top-left" or sh["id"] == 1:
                mark_op = 1.0 if plan["brand_mark"] == "top-left" else T.fade_opacity(t, 0.6, 4.0, fade=0.5)
                if mark_op > 0:
                    out = T.overlay_layer(out, mark, 0.85 * mark_op)
            if layer is not None:
                op = T.fade_opacity(t, 0.5, sh["dur"] - 0.4, fade=0.6)
                if op > 0:
                    out = T.overlay_layer(out, layer, op)
            yield out
        prev_last = out
    slogan = [x for x in (copies[2] if len(copies) > 2 and copies[2] else "把目光，/還給世界。").split("/") if x.strip()]
    if plan["endcard"] == "split":
        end = T.build_endcard_split("/content/temp/product_clean_rgba.png", W, H, display_name, slogan_lines=tuple(slogan), category="SNOW GOGGLES" if is_snow else "SUNGLASSES", duration=plan["endcard_dur"], fps=24)
    else:
        end = T.build_endcard_studio("/content/temp/product_clean_rgba.png", W, H, display_name, duration=plan["endcard_dur"], fps=24)
    for i, fr in enumerate(end):
        if prev_last is not None and i < 12:
            fr = cv2.addWeighted(prev_last, 1.0 - (i + 1) / 13.0, fr, (i + 1) / 13.0, 0)
        yield fr

if IS_REF:
    _plan = STYLE_PLANS[style_key]
    w, h = output_size(cfg)
    TOTAL_FRAMES = sum(int(round(x["dur"] * 24)) for x in resolve_shots(cfg, cfg.get("custom_scene_prompt_en", ""))) + int(_plan["endcard_dur"] * 24)
    all_frames = iter_reference_timeline()
elif style_key == "0_CLASSIC_MADRID":
    f1 = load_video_frames(shot1_mp4, max_frames=174)
    f2 = load_video_frames(shot2_mp4, max_frames=186)
    f_trans = build_madrid_wall_transition_frames(display_name, is_snow=is_snow, duration_sec=2.0, fps=24)
    f_end = build_front_photo_endcard_frames(PRODUCT_PHOTO_PATH, display_name, is_snow=is_snow, duration_sec=4.5, fps=24)
    all_frames = f1 + f2 + f_trans + f_end
elif style_key == "1_CYBERPUNK_NEON":
    f1 = load_video_frames(shot1_mp4, max_frames=192)
    f2 = load_video_frames(shot2_mp4, max_frames=192)
    story_raw = f1 + f2
    font_hud = ImageFont.truetype(FONT_MONO, 17)
    processed = []
    for idx, frame in enumerate(story_raw):
        img = frame.astype(np.float32) / 255.0
        img = 1.0 / (1.0 + np.exp(-9.0 * (img - 0.48)))
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        if (188 <= idx <= 196) or (idx % 68 in [0, 1]):
            b, g, r = cv2.split(img)
            img = cv2.merge([np.roll(b, 10, axis=1), g, np.roll(r, -10, axis=1)])
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        dr.text((40, 48), "[SYS: OPTICAL HUD v5.1]", font=font_hud, fill=(0, 240, 255, 230))
        dr.text((40, 72), f"T+{idx/24.0:05.2f}s // {display_name}", font=font_hud, fill=(200, 225, 255, 200))
        dr.text((40, 96), "AERODYNAMIC POLARIZED", font=font_hud, fill=(255, 0, 128, 230))
        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), ov)
        out_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        if idx >= len(story_raw) - 10:
            fw = (idx - (len(story_raw) - 10)) / 10.0
            out_bgr = cv2.addWeighted(out_bgr, 1.0 - fw, np.full((h, w, 3), 255, dtype=np.uint8), fw, 0)
        processed.append(out_bgr)
    f_end = build_front_photo_endcard_frames(PRODUCT_PHOTO_PATH, display_name, is_snow=is_snow, duration_sec=5.0, fps=24)
    all_frames = processed + f_end
elif style_key == "3_HAUTE_COUTURE_NOIR":
    f1 = load_video_frames(shot1_mp4, max_frames=192)
    f2 = load_video_frames(shot2_mp4, max_frames=192)
    story_raw = f1 + f2
    font_ed = ImageFont.truetype(FONT_SERIF_REG, 17)
    processed = []
    for idx, frame in enumerate(story_raw):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        noir_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        pil_img = Image.fromarray(cv2.cvtColor(noir_bgr, cv2.COLOR_BGR2RGB))
        ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dr = ImageDraw.Draw(ov)
        m = 32
        dr.rectangle([m, m, w - m, h - m], outline=(218, 185, 105, 200), width=1)
        dr.text((m + 20, m + 18), "PARIS • MILANO • ESPNER", font=font_ed, fill=(218, 185, 105, 240))
        dr.text((w - m - 180, m + 18), f"EDITION {display_name}", font=font_ed, fill=(218, 185, 105, 240))
        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), ov)
        out_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        if idx >= len(story_raw) - 10:
            fw = (idx - (len(story_raw) - 10)) / 10.0
            out_bgr = cv2.addWeighted(out_bgr, 1.0 - fw, np.full((h, w, 3), 255, dtype=np.uint8), fw, 0)
        processed.append(out_bgr)
    f_end = build_front_photo_endcard_frames(PRODUCT_PHOTO_PATH, display_name, is_snow=is_snow, duration_sec=5.0, fps=24)
    all_frames = processed + f_end
else:
    f1 = load_video_frames(shot1_mp4, max_frames=192)
    f2 = load_video_frames(shot2_mp4, max_frames=192)
    story_raw = f1 + f2
    for idx in range(max(0, len(story_raw) - 10), len(story_raw)):
        fw = (idx - (len(story_raw) - 10)) / 10.0
        story_raw[idx] = cv2.addWeighted(story_raw[idx], 1.0 - fw, np.full((h, w, 3), 255, dtype=np.uint8), fw, 0)
    f_end = build_front_photo_endcard_frames(PRODUCT_PHOTO_PATH, display_name, is_snow=is_snow, duration_sec=5.0, fps=24)
    all_frames = story_raw + f_end

safe_name = display_name.replace(" ", "_")
final_mp4 = f"/content/output/ESPNER_{safe_name}_commercial_v5.mp4"
raw_out = "/content/temp/final_raw.mp4"

# 逐格串流寫檔 (不把 720 格全部放進記憶體；1080p 30 秒約 4.5GB)，只保留 12 張 QA 縮圖
total = TOTAL_FRAMES if TOTAL_FRAMES else len(all_frames)
indices = [int(x) for x in np.linspace(0, total - 1, 12)]
qa = {}
n_written = 0
writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*'mp4v'), 24.0, (w, h))
for f in all_frames:
    if f.shape[1] != w or f.shape[0] != h:
        f = cv2.resize(f, (w, h))
    writer.write(f)
    if n_written in indices:
        qa[n_written] = f.copy()
    n_written += 1
writer.release()
all_frames = None

video_noaudio = "/content/temp/final_noaudio.mp4"
subprocess.run([
    "ffmpeg", "-y", "-i", raw_out,
    "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    video_noaudio
], check=True)

# 🎵 配樂：使用者上傳的音樂，或依風格自動生成的免版權配樂
music_mode = cfg.get("music", "auto")
if music_mode == "none":
    shutil.copy2(video_noaudio, final_mp4)
    print("🔇 依設定不加配樂")
else:
    try:
        if music_mode == "file" and cfg.get("music_path") and os.path.exists(cfg["music_path"]):
            music_src = cfg["music_path"]
            print(f"🎵 使用上傳的音樂：{os.path.basename(music_src)}")
        else:
            music_src = AU.synth_music_bed(n_written / 24.0, "/content/temp/music_bed.wav", mood=AU.pick_mood(style_key), seed=7)
            print(f"🎵 已自動生成配樂 (風格：{AU.pick_mood(style_key)})")
        AU.mux_audio(video_noaudio, music_src, final_mp4)
    except Exception as e:
        print(f"⚠️ 配樂合成失敗 ({e})，輸出無聲版本")
        shutil.copy2(video_noaudio, final_mp4)

# 🔍 自動品質檢查報告 (只檢查故事鏡頭；片尾沒有人臉不檢查)
qc_txt = f"/content/output/ESPNER_{safe_name}_品質報告.txt"
try:
    import espner_qc as QC
    bounds = None
    if IS_REF:
        bounds, acc = [], 0
        for _sh in resolve_shots(cfg, cfg.get("custom_scene_prompt_en", "")):
            n_sh = int(round(_sh["dur"] * 24))
            bounds.append((acc, acc + n_sh))
            acc += n_sh
    qc_report = QC.format_report(QC.analyze_video(final_mp4, shot_bounds=bounds))
    print("\n" + qc_report)
    with open(qc_txt, "w", encoding="utf-8") as _f:
        _f.write(qc_report)
except Exception as e:
    qc_txt = None
    print(f"ℹ️ 品質檢查略過 ({e})")

sheet_jpg = f"/content/output/ESPNER_{safe_name}_QA_sheet.jpg"
tw, th = (426, 240) if w > h else (240, 426)
tiles = []
for idx in indices:
    t_img = cv2.resize(qa[idx], (tw, th))
    cv2.putText(t_img, f"f:{idx} ({idx/24.0:.1f}s)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 3)
    cv2.putText(t_img, f"f:{idx} ({idx/24.0:.1f}s)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 1)
    tiles.append(t_img)
sheet = np.vstack([np.hstack(tiles[r*4:(r+1)*4]) for r in range(3)])
cv2.imwrite(sheet_jpg, sheet)

drive_out_dir = "/content/drive/MyDrive/ESPNER_Colab_T4_Output"
if os.path.isdir("/content/drive/MyDrive"):
    os.makedirs(drive_out_dir, exist_ok=True)
    shutil.copy2(final_mp4, os.path.join(drive_out_dir, os.path.basename(final_mp4)))
    shutil.copy2(sheet_jpg, os.path.join(drive_out_dir, os.path.basename(sheet_jpg)))
    if qc_txt and os.path.exists(qc_txt):
        shutil.copy2(qc_txt, os.path.join(drive_out_dir, os.path.basename(qc_txt)))
    print(f"☁️ 已自動同步備份至 Google 雲端硬碟：{drive_out_dir}")

dur_sec = n_written / 24.0
size_mb = os.path.getsize(final_mp4) / (1024 * 1024)
print(f"\n🎉 廣告影片運算完成！")
print(f"📁 檔案路徑：{final_mp4} ({dur_sec:.2f} 秒, {size_mb:.2f} MB)")
display(IPImage(filename=sheet_jpg, width=640))

mp4_bytes = open(final_mp4, 'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4_bytes).decode()
display(HTML(f'<video width={640 if w > h else 360} controls autoplay loop><source src="{data_url}" type="video/mp4"></video>'))
files.download(final_mp4)"""

def step4_lines():
    # 把三個後製模組以 r'''...''' 嵌入步驟 4，執行時寫到 /content/temp 供 import
    out = []
    for ln in cell_lines(STEP4_CODE):
        if ln.strip() == "# __EMBED_MODULES__":
            for var, fname, code in [("titles_code", "espner_titles.py", TITLES_CODE), ("grade_code", "espner_grade.py", GRADE_CODE), ("audio_code", "espner_audio.py", AUDIO_CODE), ("qc_code", "espner_qc.py", QC_CODE)]:
                out += embed_stage(var, code)
                out.append(f'open("/content/temp/{fname}", "w", encoding="utf-8").write({var})\n')
            out.append("import importlib, sys as _sys\n")
            out.append("for _m in (\"espner_titles\", \"espner_grade\", \"espner_audio\", \"espner_qc\", \"shot_plan\"):\n")
            out.append("    _sys.modules.pop(_m, None)\n")
        else:
            out.append(ln)
    return out

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
        ] + cell_lines(STEP2_CODE) + [
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
            "    # 只查版本號，不在筆記本主程序 import diffusers/torch (省下 ~1GB 系統 RAM 給子程序)\n",
            "    from importlib.metadata import version as _ver\n",
            "    assert _ver(\"diffusers\") == \"0.38.0\"\n",
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
        ] + embed_stage("shot_plan_code", SHOTPLAN_CODE) + embed_stage("track_code", TRACK_CODE) + embed_stage("stage_3a_code", STAGE_3A_CODE) + [
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
            "Path(\"/content/temp/shot_plan.py\").write_text(shot_plan_code, encoding=\"utf-8\")\n",
            "Path(\"/content/temp/espner_track.py\").write_text(track_code, encoding=\"utf-8\")\n",
            "import glob as _glob\n",
            "for _old in _glob.glob(\"/content/temp/keyframe_shot*.png\") + _glob.glob(\"/content/temp/glasses_shot*.png\") + _glob.glob(\"/content/temp/raw_shot*.mp4\") + [\"/content/temp/glasses_meta.json\"] + [\"/content/temp/stage3c_status.json\", \"/content/temp/prompt_embeds.pt\"]:\n",
            "    if os.path.exists(_old):\n",
            "        os.remove(_old)\n",
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
            "    try:\n",
            "        import resource\n",
            "        print(f\"📊 目前為止子程序記憶體峰值：{resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1048576:.1f} GB (上限 12.7 GB)\")\n",
            "    except Exception:\n",
            "        pass\n",
            "    if proc.returncode in (-9, 137):\n",
            "        raise RuntimeError(f\"❌ {stage_title} 被系統強制終止：系統 RAM 用盡。請截圖此訊息與上方的記憶體峰值給開發者。\")\n",
            "    if proc.returncode != 0:\n",
            "        err_tail = \"\".join(output_lines[-30:])\n",
            "        raise RuntimeError(f\"❌ {stage_title} 執行失敗 (exit={proc.returncode}):\\n{err_tail}\")\n",
            "\n",
            "run_stage(\"/content/temp/stage_3a.py\", \"階段 3A：OpenCV 去背與真人模特兒配戴首幀合成 (獨立子程序)\")\n",
            "\n",
            "import glob as _glob\n",
            "for _kf in sorted(_glob.glob(\"/content/temp/keyframe_shot*.png\")):\n",
            "    print(\"🖼️\", os.path.basename(_kf))\n",
            "    display(IPImage(filename=_kf, width=220))\n",
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
            "        _m = \"\\n   \".join(f\"第{x['shot']}鏡：動態 {x['motion']}｜首幀相似度 {x.get('first_frame_sim', '-')}｜連續性 {x.get('continuity', '-')}｜真實眼鏡追蹤信心 {x.get('glasses_conf', '-')}｜AI 可用 {x.get('ai_frames_used', '-')} 格、補足 {x.get('held_frames', '-')} 格\" for x in _st.get(\"shots\", []))\n",
            "        print(f\"\\n✅ AI 動態影片完成｜引擎：{_st.get('engine')}\\n   {_m}\")\n",
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
        ] + step4_lines() + [
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

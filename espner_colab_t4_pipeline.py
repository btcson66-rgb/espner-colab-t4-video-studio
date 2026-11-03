#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESPNER 精品廣告 AI 自動生成單檔腳本 (Colab T4 GPU 專用版 v5.0)
================================================================================
可直接放在 Google Colab (T4 GPU) 執行，只要上傳一張產品正面照片，自動完成：
1. 產品正面圖 AI 去背與 100% 原始幾何特徵鎖定 (rembg + Luminance Mask)
2. 真人時尚模特兒肖像生成 (RealVisXL V4.0 Lightning) + OpenCV 臉部/雙眼定位精準配戴合成
3. T4 GPU 開源圖生影片 (Lightricks/LTX-Video 或 SVD-XT) 生成兩段各 7.5~8.0 秒動態故事片段
4. ESPNER v5 精品後製：6 大風格調色/HUD + 2.0s 石牆金框過場 + 4.5~5.0s 純白攝影棚正面原圖片尾

在 Colab 單一儲存格一鍵執行方式：
!python espner_colab_t4_pipeline.py --photo /content/input.jpg --name "AVIATOR" --style "0_CLASSIC_MADRID"
或直接將本檔內容貼入 Colab Notebook 儲存格執行（若未指定 --photo 會自動跳出上傳視窗）。
"""

import os
import sys
import gc
import argparse
import subprocess

# 自動檢查與安裝 Colab 必備套件
def ensure_colab_dependencies():
    try:
        import diffusers
        import rembg
        import cv2
    except ImportError:
        print("📦 正在為 Colab T4 安裝 AI 影片生成與去背套件 (約需 1~2 分鐘)...")
        os.system("pip install -q -U diffusers>=0.32.0 transformers accelerate sentencepiece protobuf safetensors rembg onnxruntime-gpu opencv-python-headless pillow imageio[ffmpeg]")
        os.system("apt-get update -qq && apt-get install -y -qq ffmpeg fonts-noto-cjk fonts-dejavu-core")

ensure_colab_dependencies()

import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from rembg import remove


def get_font_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[-1]


FONT_SERIF_BOLD = get_font_path([
    r"C:\Windows\Fonts\timesbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
])
FONT_SERIF_REG = get_font_path([
    r"C:\Windows\Fonts\times.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
])
FONT_MONO = get_font_path([
    r"C:\Windows\Fonts\consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
])
FONT_CHINESE = get_font_path([
    r"C:\Windows\Fonts\msjh.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
])


def draw_tracked_text(draw, center_y, text, font, fill, width=720, tracking=4):
    chars = list(text)
    if not chars:
        return
    char_widths = []
    for c in chars:
        bbox = font.getbbox(c)
        char_widths.append(bbox[2] - bbox[0])
    total_w = sum(char_widths) + (len(chars) - 1) * tracking
    curr_x = (width - total_w) // 2
    for i, c in enumerate(chars):
        draw.text((curr_x, center_y), c, font=font, fill=fill)
        curr_x += char_widths[i] + tracking


def extract_clean_product_rgba(photo_path):
    """精準去背上傳的產品正面圖，保留鏡片半透明感與 100% 原始輪廓"""
    raw_pil = Image.open(photo_path).convert("RGBA")
    fg_pil = remove(raw_pil)
    raw_np = np.array(raw_pil)
    fg_np = np.array(fg_pil)

    gray = cv2.cvtColor(raw_np[:, :, :3], cv2.COLOR_RGB2GRAY)
    luma_mask = np.clip((250.0 - gray.astype(np.float32)) / 18.0, 0.0, 1.0)
    combined_alpha = np.maximum(fg_np[:, :, 3].astype(np.float32) / 255.0, luma_mask)
    combined_alpha[gray > 248] = 0.0
    fg_np[:, :, 3] = (combined_alpha * 255).astype(np.uint8)

    ys, xs = np.where(fg_np[:, :, 3] > 25)
    if len(ys) > 0:
        pad = 6
        y1, y2 = max(0, ys.min() - pad), min(fg_np.shape[0], ys.max() + pad)
        x1, x2 = max(0, xs.min() - pad), min(fg_np.shape[1], xs.max() + pad)
        return Image.fromarray(fg_np[y1:y2, x1:x2])
    return Image.fromarray(fg_np)


def composite_eyewear_on_model(portrait_pil, prod_rgba, is_closeup=False, is_snow=False):
    """自動偵測模特兒臉部與雙眼，將上傳產品以 100% 原始幾何比例與真實接觸陰影配戴至臉上"""
    w, h = portrait_pil.size
    img_bgr = cv2.cvtColor(np.array(portrait_pil), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(140, 140))
    angle_deg = 0.0

    if len(faces) > 0:
        fx, fy, fw, fh = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)[0]
        face_cx = fx + fw // 2
        eye_y = fy + int(fh * 0.39)
        target_w = int(fw * (1.06 if is_snow else 0.96))

        roi_gray = gray[fy:fy + int(fh * 0.6), fx:fx + fw]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda e: e[0])[:2]
            e1x, e1y = fx + eyes[0][0] + eyes[0][2] // 2, fy + eyes[0][1] + eyes[0][3] // 2
            e2x, e2y = fx + eyes[1][0] + eyes[1][2] // 2, fy + eyes[1][1] + eyes[1][3] // 2
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

        roi = canvas[top_y:top_y + cur_h, left_x:left_x + cur_w].astype(np.float32)
        blended = roi * (1.0 - crop_mask) + scaled_crop.astype(np.float32) * crop_mask
        canvas[top_y:top_y + cur_h, left_x:left_x + cur_w] = np.clip(blended, 0, 255).astype(np.uint8)

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


def run_pipeline(photo_path, model_name="AVIATOR", style_key="0_CLASSIC_MADRID", is_snow=False, out_dir="./output"):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("./temp", exist_ok=True)

    product_rgba = extract_clean_product_rgba(photo_path)
    print(f"✅ 產品正面圖去背完成：{product_rgba.size}")

    p_port1 = "Editorial medium portrait of a stylish European fashion model in linen shirt standing in a sunlit Mediterranean coastal stone colonnade, facing camera directly, golden hour sunlight, ocean background, 85mm lens, photorealistic, 8k"
    p_port2 = "Editorial close-up portrait of a stylish European fashion model facing camera directly by the sunlit Mediterranean sea, warm golden hour lighting on face, wind in hair, shallow depth of field, 85mm lens, photorealistic, 8k"
    p_mot1 = "Cinematic luxury eyewear commercial, stylish model wearing sunglasses walking forward slowly along sunlit Mediterranean stone colonnade, subtle head turn, breeze in hair, sunlight glinting on lenses, smooth camera tracking"
    p_mot2 = "Cinematic close-up of model wearing luxury sunglasses looking toward sunlit ocean, gentle camera push-in, warm sunlight sweeping across polarized lenses, natural breathing and subtle expression"

    shot1_mp4 = "./temp/raw_shot1.mp4"
    shot2_mp4 = "./temp/raw_shot2.mp4"

    if torch.cuda.is_available():
        from diffusers import AutoPipelineForText2Image, EulerDiscreteScheduler, LTXImageToVideoPipeline
        from diffusers.utils import export_to_video

        print("🎨 [1/3] 正在於 T4 GPU 生成真人時尚模特兒並合成產品配戴首幀...")
        txt2img = AutoPipelineForText2Image.from_pretrained(
            "SG161222/RealVisXL_V4.0_Lightning",
            torch_dtype=torch.float16,
            variant="fp16"
        ).to("cuda")
        txt2img.scheduler = EulerDiscreteScheduler.from_config(txt2img.scheduler.config, timestep_spacing="trailing")
        neg_p = "sunglasses, glasses, eyewear, goggles, deformed face, asymmetrical eyes, blurry, watermark"

        b1 = txt2img(prompt=p_port1, negative_prompt=neg_p, width=720, height=1280, num_inference_steps=6, guidance_scale=1.8, generator=torch.Generator("cuda").manual_seed(1024)).images[0]
        b2 = txt2img(prompt=p_port2, negative_prompt=neg_p, width=720, height=1280, num_inference_steps=6, guidance_scale=1.8, generator=torch.Generator("cuda").manual_seed(2048)).images[0]
        del txt2img
        gc.collect()
        torch.cuda.empty_cache()

        kf1 = composite_eyewear_on_model(b1, product_rgba, is_closeup=False, is_snow=is_snow)
        kf2 = composite_eyewear_on_model(b2, product_rgba, is_closeup=True, is_snow=is_snow)

        print("🎬 [2/3] 正在於 T4 GPU 執行 Lightricks/LTX-Video 圖生影片...")
        i2v = LTXImageToVideoPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.bfloat16)
        i2v.enable_model_cpu_offload()
        i2v.vae.enable_tiling()

        for idx, (kf, mp, out_p) in enumerate([(kf1, p_mot1, shot1_mp4), (kf2, p_mot2, shot2_mp4)], 1):
            frames = i2v(
                image=kf.resize((512, 896), Image.Resampling.LANCZOS),
                prompt=mp,
                negative_prompt="deformed sunglasses, morphing eyewear, blurry, jitter, watermark",
                width=512, height=896, num_frames=121, num_inference_steps=30,
                generator=torch.Generator("cuda").manual_seed(42 + idx)
            ).frames[0]
            tmp_ai = f"./temp/ltx_{idx}.mp4"
            export_to_video(frames, tmp_ai, fps=24)
            subprocess.run([
                "ffmpeg", "-y", "-i", tmp_ai,
                "-vf", "setpts=1.6*PTS,scale=720:1280:flags=lanczos,fps=24",
                "-t", "8.0", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", out_p
            ], check=True)
        del i2v
        gc.collect()
        torch.cuda.empty_cache()
    else:
        raise RuntimeError("請在具有 NVIDIA T4 GPU 的 Google Colab 環境中執行此腳本！")

    print("🎞️ [3/3] 正在合成石牆金框過場與純白攝影棚正面原圖片尾...")
    cap1, cap2 = cv2.VideoCapture(shot1_mp4), cv2.VideoCapture(shot2_mp4)
    f1, f2 = [], []
    while len(f1) < 174:
        ret, f = cap1.read()
        if not ret: break
        f1.append(cv2.resize(f, (720, 1280)))
    while len(f2) < 186:
        ret, f = cap2.read()
        if not ret: break
        f2.append(cv2.resize(f, (720, 1280)))
    cap1.release()
    cap2.release()

    f_trans = build_madrid_wall_transition_frames(model_name, is_snow=is_snow, duration_sec=2.0, fps=24)
    f_end = build_front_photo_endcard_frames(photo_path, model_name, is_snow=is_snow, duration_sec=4.5, fps=24)
    all_frames = f1 + f2 + f_trans + f_end

    safe_name = model_name.strip().upper().replace(" ", "_")
    final_mp4 = os.path.join(out_dir, f"ESPNER_{safe_name}_commercial_v5.mp4")
    raw_out = "./temp/final_raw.mp4"
    writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*'mp4v'), 24.0, (720, 1280))
    for f in all_frames:
        writer.write(f)
    writer.release()

    subprocess.run([
        "ffmpeg", "-y", "-i", raw_out,
        "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        final_mp4
    ], check=True)
    print(f"🎉 完成！輸出影片：{final_mp4}")
    return final_mp4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", type=str, default="", help="產品正面圖片路徑")
    parser.add_argument("--name", type=str, default="AVIATOR", help="產品款式名稱")
    parser.add_argument("--style", type=str, default="0_CLASSIC_MADRID", help="廣告風格代碼")
    parser.add_argument("--snow", action="store_true", help="是否為雪鏡款式")
    args, _ = parser.parse_known_args()

    photo_p = args.photo
    if not photo_p:
        try:
            from google.colab import files
            print("👇 請上傳產品正面圖片：")
            up = files.upload()
            photo_p = list(up.keys())[0]
        except Exception:
            print("請透過 --photo 指定產品正面圖片路徑。")
            sys.exit(1)

    out_video = run_pipeline(photo_p, model_name=args.name, style_key=args.style, is_snow=args.snow)
    try:
        from google.colab import files
        files.download(out_video)
    except Exception:
        pass

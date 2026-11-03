#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build 5 Creative Styles Master - Front Photo Studio Edition (v5.0)
Implements user instruction:
"參考 [ESPNER_把目光還給世界_品牌廣告_30s_v2.mp4] 把這次所有廣告影片最後的旋轉台都移除 然後用一個正面墨鏡圖片取代就好"

Features:
- Preserves 100% of the restored bespoke creative storytelling, grading & visual FX:
  * Style 1: Cyberpunk Neon & Glitch HUD (RL PRO)
  * Style 2: Vintage 1985 Kodachrome Golden Highway (PILOT)
  * Style 3: Haute Couture Noir & High-Fashion Monochrome (VIVIENNE)
  * Style 4: Alpine Glacier Extreme Sports Rush (ZERMATT)
  * Style 5: Mediterranean Riviera Luxury Sunset (AVIATOR)
- Replaces the turntable segment with a 5.0s (120 frames) Studio Front Photo Endcard:
  * Official front-facing product photograph from catalog_sources.
  * Pure white #FFFFFF studio background with soft grounded drop shadow.
  * Subtle slow push-in (scale 1.000 -> 1.035).
  * Taste-Skill luxury typography: "ESPNER", model name, "把目光，還給世界。", "POLARIZED LUXURY EYEWEAR".
  * Smooth white flash / dip-to-white cross-fade transition from story into studio card.
- Target duration: Exactly 21.00s (504 frames @ 24fps) for each style.
- Zero Flow API credits consumed.
"""

import os, sys, subprocess, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = r"D:\廣告影片\ESPNER_Blender廣告製作_20261031\creative_styles_test"
TEMP_DIR = os.path.join(BASE_DIR, "temp_front_v5")
QA_PREVIEW = os.path.join(BASE_DIR, "qa_preview")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(QA_PREVIEW, exist_ok=True)

CATALOG_DIR = r"D:\廣告影片\catalog_sources"

FONT_SERIF_BOLD = r"C:\Windows\Fonts\timesbd.ttf" if os.path.exists(r"C:\Windows\Fonts\timesbd.ttf") else r"C:\Windows\Fonts\times.ttf"
FONT_SERIF_REG  = r"C:\Windows\Fonts\times.ttf"
FONT_SANS_BOLD   = r"C:\Windows\Fonts\arialbd.ttf" if os.path.exists(r"C:\Windows\Fonts\arialbd.ttf") else r"C:\Windows\Fonts\arial.ttf"
FONT_MONO        = r"C:\Windows\Fonts\consola.ttf" if os.path.exists(r"C:\Windows\Fonts\consola.ttf") else r"C:\Windows\Fonts\arial.ttf"
FONT_IMPACT      = r"C:\Windows\Fonts\impact.ttf" if os.path.exists(r"C:\Windows\Fonts\impact.ttf") else FONT_SANS_BOLD
FONT_CHINESE     = r"C:\Windows\Fonts\msjh.ttc"

def draw_tracked_text_centered(draw, center_y, text, font, fill, width=720, tracking=4):
    chars = list(text)
    if not chars: return
    char_widths = []
    for c in chars:
        bbox = font.getbbox(c)
        char_widths.append(bbox[2] - bbox[0])
    total_w = sum(char_widths) + (len(chars) - 1) * tracking
    start_x = (width - total_w) // 2
    curr_x = start_x
    for i, c in enumerate(chars):
        draw.text((curr_x, center_y), c, font=font, fill=fill)
        curr_x += char_widths[i] + tracking

def load_video_frames(path, max_frames=None, resize=(720, 1280)):
    cap = cv2.VideoCapture(path)
    frames = []
    count = 0
    while True:
        ret, f = cap.read()
        if not ret: break
        if max_frames is not None and count >= max_frames: break
        if resize and (f.shape[1] != resize[0] or f.shape[0] != resize[1]):
            f = cv2.resize(f, resize)
        frames.append(f)
        count += 1
    cap.release()
    return frames

def get_product_bbox(photo):
    h, w = photo.shape[:2]
    gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
    binary = (gray < 240).astype(np.uint8)
    margin = int(min(h, w) * 0.015)
    margin = max(15, margin)
    binary[:margin, :] = 0
    binary[-margin:, :] = 0
    binary[:, :margin] = 0
    binary[:, -margin:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed)
    min_area = min(15000, int(h * w * 0.005))
    valid_boxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            vx = stats[i, cv2.CC_STAT_LEFT]
            vy = stats[i, cv2.CC_STAT_TOP]
            vw = stats[i, cv2.CC_STAT_WIDTH]
            vh = stats[i, cv2.CC_STAT_HEIGHT]
            valid_boxes.append((vx, vy, vx + vw, vy + vh))
    if not valid_boxes:
        return 0, 0, w, h
    min_x = max(0, min(b[0] for b in valid_boxes) - 2)
    min_y = max(0, min(b[1] for b in valid_boxes) - 2)
    max_x = min(w, max(b[2] for b in valid_boxes) + 2)
    max_y = min(h, max(b[3] for b in valid_boxes) + 2)
    return min_x, min_y, max_x, max_y

def build_front_photo_endcard_frames(photo_p, display_name, duration_sec=5.0, fps=24,
                                     slogan="把目光，還給世界。", sub="POLARIZED LUXURY EYEWEAR"):
    num_frames = int(duration_sec * fps)
    w, h = 720, 1280
    
    photo = cv2.imdecode(np.fromfile(photo_p, dtype=np.uint8), cv2.IMREAD_COLOR)
    x1, y1, x2, y2 = get_product_bbox(photo)
    tight_crop = photo[y1:y2, x1:x2]
        
    target_base_w = 540
    base_h = int(tight_crop.shape[0] * (target_base_w / tight_crop.shape[1]))
    crop_base = cv2.resize(tight_crop, (target_base_w, base_h), interpolation=cv2.INTER_LANCZOS4)
    
    font_brand = ImageFont.truetype(FONT_SERIF_BOLD, 46)
    font_model = ImageFont.truetype(FONT_SERIF_BOLD, 28)
    font_slogan = ImageFont.truetype(FONT_CHINESE, 34)
    font_sub = ImageFont.truetype(FONT_SERIF_REG, 20)
    
    white_canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    endcard_frames = []
    
    for i in range(num_frames):
        t = i / (num_frames - 1)
        scale = 1.0 + 0.035 * t
        
        cur_w = int(target_base_w * scale)
        cur_h = int(base_h * scale)
        scaled_crop = cv2.resize(crop_base, (cur_w, cur_h), interpolation=cv2.INTER_LANCZOS4)
        
        canvas = white_canvas.copy()
        
        cy = 530
        top_y = cy - cur_h // 2
        # Mathematically perfect center: left_x guarantees product center is at w // 2 (360)
        left_x = (w - cur_w) // 2
        
        # Soft ground contact shadow
        sy = top_y + cur_h - int(cur_h * 0.05)
        shadow = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(shadow, (w // 2, sy), (int(cur_w * 0.44), int(22 * scale)), 0, 0, 360, 255, -1)
        shadow = cv2.GaussianBlur(shadow, (71, 35), 0)
        shadow_val = (shadow.astype(np.float32) / 255.0 * 0.12)[..., np.newaxis]
        canvas = np.clip(canvas.astype(np.float32) * (1.0 - shadow_val), 0, 255).astype(np.uint8)
        
        crop_gray = cv2.cvtColor(scaled_crop, cv2.COLOR_BGR2GRAY)
        crop_mask = np.clip((255.0 - crop_gray.astype(np.float32)) / 15.0, 0.0, 1.0)[..., np.newaxis]
        
        roi = canvas[top_y:top_y+cur_h, left_x:left_x+cur_w].astype(np.float32)
        blended = roi * (1.0 - crop_mask) + scaled_crop.astype(np.float32) * crop_mask
        canvas[top_y:top_y+cur_h, left_x:left_x+cur_w] = np.clip(blended, 0, 255).astype(np.uint8)
        
        # Typography
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        draw_tracked_text_centered(draw, 145, "E S P N E R", font_brand, (15, 23, 42), width=w, tracking=8)
        draw_tracked_text_centered(draw, 210, display_name.upper(), font_model, (71, 85, 105), width=w, tracking=4)
        draw_tracked_text_centered(draw, 840, slogan, font_slogan, (15, 23, 42), width=w, tracking=4)
        draw_tracked_text_centered(draw, 905, sub, font_sub, (100, 116, 139), width=w, tracking=3)
        
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        # Fade in from white in first 10 frames (0.4s)
        if i < 10:
            fade_in = i / 10.0
            frame_bgr = cv2.addWeighted(frame_bgr, fade_in, white_canvas, 1.0 - fade_in, 0)
            
        endcard_frames.append(frame_bgr)
        
    return endcard_frames

def export_frames_to_mp4(frames, out_mp4, fps=24.0, crf="17"):
    raw_temp = os.path.join(TEMP_DIR, "raw_" + os.path.basename(out_mp4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w, c = frames[0].shape
    writer = cv2.VideoWriter(raw_temp, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    cmd = [
        "ffmpeg", "-y", "-i", raw_temp,
        "-c:v", "libx264", "-crf", crf, "-preset", "medium", "-pix_fmt", "yuv420p", "-an",
        out_mp4
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(raw_temp):
        os.remove(raw_temp)
    size_mb = os.path.getsize(out_mp4) / (1024 * 1024)
    print(f"  -> Built {os.path.basename(out_mp4)}: {len(frames)} frames ({len(frames)/fps:.2f}s, {size_mb:.2f} MB)")

def generate_contact_sheet(mp4_path, out_sheet_jpg, num_cols=4, num_rows=3):
    cap = cv2.VideoCapture(mp4_path)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_tiles = num_cols * num_rows
    indices = np.linspace(0, max(0, total_f - 1), num_tiles, dtype=int)
    tiles = []
    tile_w, tile_h = 240, 426
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((1280, 720, 3), dtype=np.uint8)
        t = cv2.resize(frame, (tile_w, tile_h))
        cv2.putText(t, f"f:{idx} ({idx/24.0:.1f}s)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(t, f"f:{idx} ({idx/24.0:.1f}s)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        tiles.append(t)
    cap.release()
    rows = []
    for r in range(num_rows):
        row_tiles = tiles[r * num_cols : (r + 1) * num_cols]
        rows.append(np.hstack(row_tiles))
    sheet = np.vstack(rows)
    _, buf = cv2.imencode('.jpg', sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    buf.tofile(out_sheet_jpg)


# ==============================================================================
# STYLE 1: CYBERPUNK NEON & GLITCH NOIR (RL PRO)
# ==============================================================================
def build_style1():
    print("\n[1/5] Building Style 1: Cyberpunk Neon Velocity (RL PRO)...")
    out_mp4 = os.path.join(BASE_DIR, "STYLE_1_CYBERPUNK_RL_PRO.mp4")
    
    clip1_p = r"D:\廣告影片\ESPNER_Blender廣告製作_20261031\formal\new_models\RL_PRO\RL_PRO_gemini_vertical_original.mp4"
    clip2_p = r"D:\廣告影片\EP000103_RL_PRO\01_gemini_departure.mp4"
    photo_p = os.path.join(CATALOG_DIR, "EP000103 RL PRO", "RL1.jpg")
    
    f_s1 = load_video_frames(clip1_p, max_frames=192) # 8.0s
    f_s2 = load_video_frames(clip2_p, max_frames=192) # 8.0s
    story_raw = f_s1 + f_s2 # 384 frames (16.0s)
    
    font_hud_sm = ImageFont.truetype(FONT_MONO, 17)
    cut1_f = len(f_s1)
    
    processed_story = []
    w, h = 720, 1280
    
    for idx, frame in enumerate(story_raw):
        # 1. Cyberpunk color grading
        img = frame.astype(np.float32) / 255.0
        img = 1.0 / (1.0 + np.exp(-9.0 * (img - 0.48)))
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) / 255.0
        shadow_mask = np.clip(1.0 - gray * 1.5, 0.0, 1.0)[..., np.newaxis]
        high_mask = np.clip((gray - 0.5) * 2.0, 0.0, 1.0)[..., np.newaxis]
        img[..., 0] += shadow_mask[..., 0] * 0.14
        img[..., 1] += shadow_mask[..., 0] * 0.08
        img[..., 2] += high_mask[..., 0] * 0.12
        img[..., 0] += high_mask[..., 0] * 0.08
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        
        # 2. Glitch on cut
        is_glitch = (cut1_f - 4 <= idx <= cut1_f + 4) or (idx % 68 in [0, 1])
        if is_glitch:
            shift = np.random.randint(6, 16)
            b, g, r = cv2.split(img)
            b = np.roll(b, shift, axis=1)
            r = np.roll(r, -shift, axis=1)
            for _ in range(np.random.randint(2, 5)):
                sy = np.random.randint(60, h - 120)
                sh = np.random.randint(10, 35)
                dx = np.random.randint(-22, 22)
                img[sy:sy+sh] = np.roll(img[sy:sy+sh], dx, axis=1)
            img = cv2.merge([b, g, r])
            
        # 3. Scanlines
        scanline = np.ones((h, 1, 1), dtype=np.float32)
        scanline[::3] = 0.88
        img = np.clip(img * scanline, 0, 255).astype(np.uint8)
        
        # 4. HUD Overlays
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        c_color = (0, 240, 255, 230)
        m_color = (255, 0, 128, 230)
        t_sec = idx / 24.0
        draw.text((40, 48), "[SYS: OPTICAL HUD v4.2]", font=font_hud_sm, fill=c_color)
        draw.text((40, 72), f"T+{t_sec:05.2f}s // 24 FPS", font=font_hud_sm, fill=(200, 225, 255, 200))
        draw.text((40, 96), "AERODYNAMIC POLARIZED", font=font_hud_sm, fill=m_color)
        draw.text((470, 48), "GEO: 35.689°N", font=font_hud_sm, fill=c_color)
        draw.text((470, 72), "ALT: 42M // SPD: 28KM/H", font=font_hud_sm, fill=(200, 225, 255, 200))
        draw.text((470, 96), "BUFFER: [█████████ ] 94%", font=font_hud_sm, fill=c_color)
        
        m = 32
        bl = 28
        draw.line([(m, m), (m + bl, m)], fill=c_color, width=2)
        draw.line([(m, m), (m, m + bl)], fill=c_color, width=2)
        draw.line([(w - m, m), (w - m - bl, m)], fill=c_color, width=2)
        draw.line([(w - m, m), (w - m, m + bl)], fill=c_color, width=2)
        draw.line([(m, h - m), (m + bl, h - m)], fill=c_color, width=2)
        draw.line([(m, h - m), (m, h - m - bl)], fill=c_color, width=2)
        draw.line([(w - m, h - m), (w - m - bl, h - m)], fill=c_color, width=2)
        draw.line([(w - m, h - m), (w - m, h - m - bl)], fill=c_color, width=2)
        
        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay)
        out_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        # Fade to white in last 10 frames of story
        if idx >= len(story_raw) - 10:
            fade_w = (idx - (len(story_raw) - 10)) / 10.0
            out_bgr = cv2.addWeighted(out_bgr, 1.0 - fade_w, np.full((h, w, 3), 255, dtype=np.uint8), fade_w, 0)
            
        processed_story.append(out_bgr)
        
    f_endcard = build_front_photo_endcard_frames(photo_p, "RL PRO", duration_sec=5.0, fps=24,
                                                 slogan="超越感知，直擊霓虹未來。", sub="BEYOND PERCEPTION // CYBER OPTICS") # 120 frames
    all_frames = processed_story + f_endcard # 384 + 120 = 504 frames = 21.00s!
    export_frames_to_mp4(all_frames, out_mp4)
    generate_contact_sheet(out_mp4, os.path.join(QA_PREVIEW, "STYLE_1_CYBERPUNK_RL_PRO_sheet.jpg"))


# ==============================================================================
# STYLE 2: VINTAGE 1985 KODACHROME GOLDEN HIGHWAY (PILOT)
# ==============================================================================
def build_style2():
    print("\n[2/5] Building Style 2: Vintage 1985 Kodachrome Golden Highway (PILOT)...")
    out_mp4 = os.path.join(BASE_DIR, "STYLE_2_VINTAGE_1985_CLASSIC.mp4")
    
    story_p = r"D:\廣告影片\ESPNER_Blender廣告製作_20261031\creative_styles_test\gemini_raw\STYLE_2_PILOT_VINTAGE_1985_NATIVE.mp4"
    photo_p = os.path.join(CATALOG_DIR, "EP00022 PILOT", "EP-000022-BLK_RED-F.jpg")
    
    f_s1 = load_video_frames(story_p, max_frames=240) # 10.0s (240 frames)
    
    w, h = 720, 1280
    f_bridge = []
    font_brand = ImageFont.truetype(FONT_SERIF_BOLD, 48)
    font_model = ImageFont.truetype(FONT_SERIF_BOLD, 54)
    font_sub = ImageFont.truetype(FONT_SERIF_REG, 22)
    font_ed = ImageFont.truetype(FONT_SERIF_REG, 18)
    
    base_last = f_s1[-1].copy()
    base_blurred = cv2.GaussianBlur(base_last, (45, 45), 0)
    
    for i in range(144): # 6.0s bridge (total story + bridge = 384 frames = 16.0s)
        t = i / 143.0
        scale = 1.0 + 0.05 * t
        M = cv2.getRotationMatrix2D((w/2, h/2), 0, scale)
        bg = cv2.warpAffine(base_blurred, M, (w, h))
        
        pil_bg = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        alpha = int(255 * min(1.0, t / 0.20)) if t < 0.80 else int(255 * max(0.0, (1.0 - t) / 0.20))
        gold_c = (235, 195, 110, alpha)
        white_c = (255, 245, 225, alpha)
        
        bx1, by1, bx2, by2 = 60, h//2 - 130, w - 60, h//2 + 130
        draw.rounded_rectangle([(bx1, by1), (bx2, by2)], radius=12,
                               fill=(28, 20, 14, int(alpha * 0.90)), outline=(235, 195, 110, int(alpha * 0.85)), width=2)
        draw.rounded_rectangle([(bx1 + 6, by1 + 6), (bx2 - 6, by2 - 6)], radius=8,
                               outline=(210, 175, 100, int(alpha * 0.40)), width=1)
        
        draw_tracked_text_centered(draw, h//2 - 95, "E S P N E R", font_brand, gold_c, width=w, tracking=6)
        draw_tracked_text_centered(draw, h//2 - 35, "PILOT · 1985", font_model, white_c, width=w, tracking=5)
        draw_tracked_text_centered(draw, h//2 + 35, "THE VINTAGE ARCHIVE // PACIFIC COAST", font_sub, gold_c, width=w, tracking=3)
        draw_tracked_text_centered(draw, h//2 + 75, "HANDCRAFTED POLARIZED AVIATOR", font_ed, (220, 200, 160, int(alpha * 0.75)), width=w, tracking=2)
        
        pil_bg = Image.alpha_composite(pil_bg.convert("RGBA"), overlay)
        out_bgr = cv2.cvtColor(np.array(pil_bg.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        # Fade to white in last 10 frames
        if i >= 134:
            fade_w = (i - 134) / 10.0
            out_bgr = cv2.addWeighted(out_bgr, 1.0 - fade_w, np.full((h, w, 3), 255, dtype=np.uint8), fade_w, 0)
            
        f_bridge.append(out_bgr)
        
    # Kodachrome 64 LUT
    lut_bgr = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        v = i / 255.0
        v_s = 0.06 + 0.94 * (v**0.92)
        r_val = np.clip((v_s * 1.08) ** 0.95 * 255.0, 0, 255)
        g_val = np.clip((v_s * 1.02) * 255.0, 0, 255)
        b_val = np.clip((v_s * 0.84) ** 1.08 * 255.0, 0, 255)
        lut_bgr[i, 0, 0] = int(b_val)
        lut_bgr[i, 0, 1] = int(g_val)
        lut_bgr[i, 0, 2] = int(r_val)
        
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_sun = np.sqrt((x_coords - 680)**2 + (y_coords - 80)**2)
    flare_mask = np.clip(1.0 - dist_sun / 650.0, 0.0, 1.0) ** 2.2
    flare_layer = np.zeros((h, w, 3), dtype=np.float32)
    flare_layer[..., 2] = flare_mask * 55.0
    flare_layer[..., 1] = flare_mask * 35.0
    flare_layer[..., 0] = flare_mask * 10.0
    dist_center = np.sqrt(((x_coords - w/2) / (w/2))**2 + ((y_coords - h/2) / (h/2))**2)
    vignette = np.clip(1.0 - 0.45 * (dist_center ** 2.8), 0.35, 1.0)[..., np.newaxis]
    grain_loop = [np.random.normal(0, 2.0, (h, w, 1)).astype(np.float32) for _ in range(24)]
    font_timestamp = ImageFont.truetype(FONT_MONO, 22)
    
    processed_s1 = []
    for idx, frame in enumerate(f_s1):
        graded = cv2.LUT(frame, lut_bgr)
        img_f = graded.astype(np.float32) * vignette + flare_layer
        img_f = np.clip(img_f + grain_loop[idx % 24], 0, 255).astype(np.uint8)
        
        pil_img = Image.fromarray(cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        amber_glow = (255, 175, 45, 240)
        t_sec = idx / 24.0
        sec_int = int(t_sec) % 60
        draw.text((45, h - 85), "JUL 18 1985", font=font_timestamp, fill=amber_glow)
        draw.text((220, h - 85), f"17:42:{sec_int:02d}", font=font_timestamp, fill=amber_glow)
        draw.text((45, 55), "▶ PLAY  SP", font=font_timestamp, fill=(245, 235, 210, 200))
        draw.text((w - 190, 55), "KODAK 35MM", font=font_timestamp, fill=(245, 210, 120, 180))
        
        processed_s1.append(cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR))
        
    f_endcard = build_front_photo_endcard_frames(photo_p, "PILOT", duration_sec=5.0, fps=24,
                                                 slogan="致敬時光，經典從不退色。", sub="CALIFORNIA 1985 // KODACHROME") # 120 frames
    all_frames = processed_s1 + f_bridge + f_endcard # 240 + 144 + 120 = 504 frames = 21.00s!
    export_frames_to_mp4(all_frames, out_mp4)
    generate_contact_sheet(out_mp4, os.path.join(QA_PREVIEW, "STYLE_2_VINTAGE_1985_CLASSIC_sheet.jpg"))


# ==============================================================================
# STYLE 3: HAUTE COUTURE NOIR & LUXURY EDITORIAL (VIVIENNE)
# ==============================================================================
def build_style3():
    print("\n[3/5] Building Style 3: Haute Couture Noir (VIVIENNE)...")
    out_mp4 = os.path.join(BASE_DIR, "STYLE_3_HAUTE_COUTURE_NOIR_VIVIENNE.mp4")
    
    clip1_p = r"D:\廣告影片\EP00067_VIVIENNE\01_gemini_departure.mp4"
    clip2_p = r"D:\廣告影片\EP00067_VIVIENNE\02_gemini_carving.mp4"
    photo_p = os.path.join(CATALOG_DIR, "EP00067 VIVIENNE", "S__5898267_0.jpg")
    
    f_s1 = load_video_frames(clip1_p, max_frames=192) # 8.0s
    f_s2 = load_video_frames(clip2_p, max_frames=192) # 8.0s
    story_raw = f_s1 + f_s2 # 384 frames (16.0s)
    
    lut_noir = np.zeros((256, 1), dtype=np.uint8)
    for i in range(256):
        v = i / 255.0
        v_s = 1.0 / (1.0 + np.exp(-10.0 * (v - 0.46)))
        lut_noir[i, 0] = int(np.clip(v_s * 255.0, 0, 255))
    lut_noir = np.ascontiguousarray(lut_noir)
    
    w, h = 720, 1280
    font_ed = ImageFont.truetype(FONT_SERIF_REG, 17)
    gold_color = (218, 185, 105, 240)
    
    processed_story = []
    for idx, frame in enumerate(story_raw):
        b, g, r = cv2.split(frame)
        gray = (0.45 * r + 0.45 * g + 0.10 * b).astype(np.uint8)
        noir_gray = cv2.LUT(gray, lut_noir)
        noir_bgr = cv2.cvtColor(noir_gray, cv2.COLOR_GRAY2BGR)
        
        pil_img = Image.fromarray(cv2.cvtColor(noir_bgr, cv2.COLOR_BGR2RGB))
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        margin = 32
        draw.rectangle([margin, margin, w - margin, h - margin], outline=(218, 185, 105, 200), width=1)
        draw.text((margin + 20, margin + 18), "PARIS • MILANO • ESPNER", font=font_ed, fill=gold_color)
        draw.text((w - margin - 150, margin + 18), "EDITION N° 67", font=font_ed, fill=gold_color)
        draw.text((margin + 20, h - margin - 35), "HAUTE COUTURE EYEWEAR", font=font_ed, fill=(200, 200, 200, 200))
        draw.text((w - margin - 145, h - margin - 35), "AUTUMN / WINTER", font=font_ed, fill=gold_color)
        
        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay)
        out_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        # Fade to white in last 10 frames
        if idx >= len(story_raw) - 10:
            fade_w = (idx - (len(story_raw) - 10)) / 10.0
            out_bgr = cv2.addWeighted(out_bgr, 1.0 - fade_w, np.full((h, w, 3), 255, dtype=np.uint8), fade_w, 0)
            
        processed_story.append(out_bgr)
        
    f_endcard = build_front_photo_endcard_frames(photo_p, "VIVIENNE", duration_sec=5.0, fps=24,
                                                 slogan="鋒芒初露，冷冽即是極致。", sub="PARIS HAUTE COUTURE NOIR") # 120 frames
    all_frames = processed_story + f_endcard # 384 + 120 = 504 frames = 21.00s!
    export_frames_to_mp4(all_frames, out_mp4)
    generate_contact_sheet(out_mp4, os.path.join(QA_PREVIEW, "STYLE_3_HAUTE_COUTURE_NOIR_VIVIENNE_sheet.jpg"))


# ==============================================================================
# STYLE 4: ALPINE GLACIER EXTREME SPORTS (ZERMATT)
# ==============================================================================
def build_style4():
    print("\n[4/5] Building Style 4: Alpine Glacier Extreme Sports (ZERMATT)...")
    out_mp4 = os.path.join(BASE_DIR, "STYLE_4_EXTREME_SPORTS_ZERMATT.mp4")
    
    clip1_p = r"D:\廣告影片\EP00042_ZERMATT\01_gemini_departure.mp4"
    clip2_p = r"D:\廣告影片\EP00042_ZERMATT\02_gemini_carving.mp4"
    photo_p = os.path.join(CATALOG_DIR, "EP00042 ZERMATT", "4N3A6295-1.jpg")
    
    f_downhill = load_video_frames(clip1_p, max_frames=180) # 7.5s
    f_closeup = load_video_frames(clip2_p, max_frames=204)  # 8.5s
    
    action_frames = []
    # Part 1: downhill solo (180 frames)
    action_frames.extend(f_downhill[:180])
    # Part 2: split screen (40 frames)
    for i in range(40):
        f_top = f_downhill[140 + i]
        f_bot = f_closeup[i]
        split_f = np.zeros_like(f_top)
        split_f[:640, :] = f_top[:640, :]
        split_f[640:, :] = f_bot[640:, :]
        action_frames.append(split_f)
    # Part 3: closeup carving finish (164 frames) -> total action = 384 frames (16.0s)
    action_frames.extend(f_closeup[40:204])
    
    lut_glacier = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        v = i / 255.0
        v_s = 1.0 / (1.0 + np.exp(-11.0 * (v - 0.47)))
        b_val = np.clip((v_s * 1.15) * 255.0, 0, 255)
        g_val = np.clip((v_s * 1.04) * 255.0, 0, 255)
        r_val = np.clip((v_s * 1.00) * 255.0, 0, 255)
        lut_glacier[i, 0, 0] = int(b_val)
        lut_glacier[i, 0, 1] = int(g_val)
        lut_glacier[i, 0, 2] = int(r_val)
    lut_glacier = np.ascontiguousarray(lut_glacier)
    
    w, h = 720, 1280
    ice_blue = (0, 230, 255, 240)
    vivid_orange = (255, 95, 0, 240)
    pure_white = (255, 255, 255, 255)
    font_hud_sm = ImageFont.truetype(FONT_MONO, 17)
    
    processed_story = []
    for idx, frame in enumerate(action_frames):
        graded = cv2.LUT(frame, lut_glacier)
        pil_img = Image.fromarray(cv2.cvtColor(graded, cv2.COLOR_BGR2RGB))
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        draw.rectangle([35, 45, w - 35, 95], fill=(10, 20, 32, 210), outline=(255, 95, 0, 230), width=1)
        draw.text((48, 52), "▲ MATTERHORN ALPINE SUMMIT", font=font_hud_sm, fill=vivid_orange)
        draw.text((48, 72), "ALT: 3,883 M  |  VLT 12% SNOW OPTICS", font=font_hud_sm, fill=ice_blue)
        spd = 65 + int(15 * np.sin(idx / 12.0))
        draw.text((470, 52), f"SPEED: {spd} KM/H", font=font_hud_sm, fill=pure_white)
        draw.text((470, 72), "SLOPE: 38° | UV400", font=font_hud_sm, fill=pure_white)
        
        is_split = (140 <= idx < 180)
        if is_split:
            mid_y = h // 2
            draw.line([(0, mid_y), (w, mid_y)], fill=(255, 95, 0, 255), width=3)
            draw.rectangle([40, mid_y - 28, 220, mid_y], fill=(255, 95, 0, 240))
            draw.text((48, mid_y - 25), "CAM 1: POV OPTICS", font=font_hud_sm, fill=(0, 0, 0, 255))
            draw.rectangle([40, mid_y + 4, 220, mid_y + 32], fill=(0, 230, 255, 240))
            draw.text((48, mid_y + 7), "CAM 2: CHASE ACTION", font=font_hud_sm, fill=(0, 0, 0, 255))
            
        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay)
        out_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        # Fade to white in last 10 frames
        if idx >= len(action_frames) - 10:
            fade_w = (idx - (len(action_frames) - 10)) / 10.0
            out_bgr = cv2.addWeighted(out_bgr, 1.0 - fade_w, np.full((h, w, 3), 255, dtype=np.uint8), fade_w, 0)
            
        processed_story.append(out_bgr)
        
    f_endcard = build_front_photo_endcard_frames(photo_p, "ZERMATT", duration_sec=5.0, fps=24,
                                                 slogan="攀越群峰，目極冰川巔峰。", sub="MATTERHORN SUMMIT // 4478M") # 120 frames
    all_frames = processed_story + f_endcard # 384 + 120 = 504 frames = 21.00s!
    export_frames_to_mp4(all_frames, out_mp4)
    generate_contact_sheet(out_mp4, os.path.join(QA_PREVIEW, "STYLE_4_EXTREME_SPORTS_ZERMATT_sheet.jpg"))


# ==============================================================================
# STYLE 5: MEDITERRANEAN RIVIERA LUXURY SUNSET (AVIATOR)
# ==============================================================================
def build_style5():
    print("\n[5/5] Building Style 5: Mediterranean Riviera Luxury Sunset (AVIATOR)...")
    out_mp4 = os.path.join(BASE_DIR, "STYLE_5_RIVIERA_SUNSET_AVIATOR.mp4")
    
    clip1_p = r"D:\廣告影片\EP0002_AVIATOR\01_gemini_departure.mp4"
    clip2_p = r"D:\廣告影片\EP0002_AVIATOR\02_gemini_carving.mp4"
    photo_p = os.path.join(CATALOG_DIR, "EP0002 AVIATOR", "EP-000002-SLV-SLV-F#1.jpg")
    
    f_s1 = load_video_frames(clip1_p, max_frames=192) # 8.0s
    f_s2 = load_video_frames(clip2_p, max_frames=192) # 8.0s
    story_raw = f_s1 + f_s2 # 384 frames (16.0s)
    
    lut_riviera = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        v = i / 255.0
        v_s = v ** 0.94
        b_val = np.clip((v_s * 1.08) ** 0.96 * 255.0, 0, 255)
        g_val = np.clip((v_s * 1.04) * 255.0, 0, 255)
        r_val = np.clip((v_s * 1.10) ** 0.94 * 255.0, 0, 255)
        lut_riviera[i, 0, 0] = int(b_val)
        lut_riviera[i, 0, 1] = int(g_val)
        lut_riviera[i, 0, 2] = int(r_val)
    lut_riviera = np.ascontiguousarray(lut_riviera)
    
    w, h = 720, 1280
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_flare = np.sqrt(((x_coords - w/2) / 1.8)**2 + (y_coords - 20)**2)
    flare_mask = np.clip(1.0 - dist_flare / 550.0, 0.0, 1.0) ** 2.4
    flare_layer = np.zeros((h, w, 3), dtype=np.float32)
    flare_layer[..., 2] = flare_mask * 55.0
    flare_layer[..., 1] = flare_mask * 42.0
    flare_layer[..., 0] = flare_mask * 18.0
    
    font_tag = ImageFont.truetype(FONT_SERIF_REG, 17)
    soft_gold = (220, 195, 135, 240)
    
    processed_story = []
    for idx, frame in enumerate(story_raw):
        graded = cv2.LUT(frame, lut_riviera)
        t = idx * 0.12
        caustic_y = np.clip((y_coords - int(h * 0.65)) / (h * 0.35), 0.0, 1.0)
        wave1 = np.sin(x_coords * 0.03 + t) * np.cos(y_coords * 0.02 - t * 0.8)
        wave2 = np.cos(x_coords * 0.02 - t * 0.5) * np.sin(y_coords * 0.03 + t * 0.6)
        caustic_pattern = np.clip((wave1 + wave2 + 1.2) / 2.4, 0.0, 1.0) * caustic_y * 0.18
        
        img_f = graded.astype(np.float32) + flare_layer
        img_f[..., 0] += caustic_pattern * 45.0
        img_f[..., 1] += caustic_pattern * 35.0
        img_f[..., 2] += caustic_pattern * 20.0
        img_f = np.clip(img_f, 0, 255).astype(np.uint8)
        
        pil_img = Image.fromarray(cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB))
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        draw.text((45, 50), "THE RIVIERA ARCHIVE", font=font_tag, fill=soft_gold)
        draw.text((w - 220, 50), "COASTAL LUXURY 2026", font=font_tag, fill=soft_gold)
        
        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay)
        out_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
        
        # Fade to white in last 10 frames
        if idx >= len(story_raw) - 10:
            fade_w = (idx - (len(story_raw) - 10)) / 10.0
            out_bgr = cv2.addWeighted(out_bgr, 1.0 - fade_w, np.full((h, w, 3), 255, dtype=np.uint8), fade_w, 0)
            
        processed_story.append(out_bgr)
        
    f_endcard = build_front_photo_endcard_frames(photo_p, "AVIATOR", duration_sec=5.0, fps=24,
                                                 slogan="馳騁天地，尊享自由航道。", sub="AMALFI COAST // RIVIERA SUNSET") # 120 frames
    all_frames = processed_story + f_endcard # 384 + 120 = 504 frames = 21.00s!
    export_frames_to_mp4(all_frames, out_mp4)
    generate_contact_sheet(out_mp4, os.path.join(QA_PREVIEW, "STYLE_5_RIVIERA_SUNSET_AVIATOR_sheet.jpg"))

def main():
    print("================================================================================")
    print("5 CREATIVE STYLES - FRONT PHOTO STUDIO EDITION (v5.0)")
    print("================================================================================")
    build_style1()
    build_style2()
    build_style3()
    build_style4()
    build_style5()
    print("\nALL 5 CREATIVE STYLES REBUILT WITH FRONT PHOTO ENDCARD SUCCESSFULLY!")

if __name__ == "__main__":
    main()

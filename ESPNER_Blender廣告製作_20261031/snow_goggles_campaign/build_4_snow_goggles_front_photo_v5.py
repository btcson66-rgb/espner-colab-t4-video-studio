#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build 4 Snow Goggles Master - Front Photo Studio Edition (v5.0)
Implements user instruction:
"參考 [ESPNER_把目光還給世界_品牌廣告_30s_v2.mp4] 把這次所有廣告影片最後的旋轉台都移除 然後用一個正面墨鏡圖片取代就好"

Features:
- Completely removes the turntable from all 4 snow goggles videos.
- Replaces with 4.5s (108 frames) Front-Facing Snow Goggle Photo Studio Endcard:
  * Official front-facing packshot from catalog_sources / product archives.
  * Pure white #FFFFFF studio canvas with soft contact shadow.
  * Subtle slow push-in (scale 1.000 -> 1.035).
  * Taste-Skill luxury typography: "ESPNER", model name, "把目光，還給世界。", "DUAL-SPHERICAL POLARIZED SNOW GOGGLE".
  * Smooth white flash transition from snow action into studio card.
- Target duration: 21.0s to 23.5s (comfortably in 20s-40s QA standard).
- Zero Flow API credits consumed.
"""

import os, sys, subprocess, cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = r"D:\廣告影片\ESPNER_Blender廣告製作_20261031\snow_goggles_campaign"
TEMP_DIR = os.path.join(BASE_DIR, "temp_snow_v5")
QA_DIR = os.path.join(BASE_DIR, "qa")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(QA_DIR, exist_ok=True)

CATALOG_DIR = r"D:\廣告影片\catalog_sources"

FONT_SERIF_BOLD = r"C:\Windows\Fonts\timesbd.ttf"
FONT_SERIF      = r"C:\Windows\Fonts\times.ttf"
FONT_SANS_BOLD   = r"C:\Windows\Fonts\arialbd.ttf"
FONT_CHINESE     = r"C:\Windows\Fonts\msjh.ttc"

def draw_tracked_text(draw, center_y, text, font, fill, width=720, tracking=4):
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

def generate_madrid_wall_transition_v5(model_name, out_path, duration_sec=2.0, fps=24):
    num_frames = int(duration_sec * fps)
    w, h = 720, 1280
    wall_ref = r"D:\廣告影片\ESPNER_Blender廣告製作_20261031\formal\qa\madrid_wall_transition_1s8.mp4"
    cap = cv2.VideoCapture(wall_ref)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 20)
    ret, wall_bg = cap.read()
    cap.release()
    
    if not ret or wall_bg is None:
        wall_bg = np.full((h, w, 3), (225, 235, 240), dtype=np.uint8)
    else:
        clean_patch = wall_bg[100:350, 50:670]
        wall_bg[500:750, 50:670] = cv2.resize(clean_patch, (620, 250))
        wall_bg = cv2.GaussianBlur(wall_bg, (5, 5), 0)

    wide_w = int(w * 1.35)
    wide_wall = cv2.resize(wall_bg, (wide_w, h))

    font_brand = ImageFont.truetype(FONT_SERIF_BOLD, 52)
    font_model = ImageFont.truetype(FONT_SERIF_BOLD, 60 if len(model_name) > 10 else 66)
    font_sub = ImageFont.truetype(FONT_SERIF, 22)

    raw_temp = out_path.replace(".mp4", "_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_temp, fourcc, fps, (w, h))

    INK_PRIMARY = (15, 23, 42)
    INK_SECONDARY = (51, 65, 85)
    INK_GOLD = (120, 80, 25)

    for idx in range(num_frames):
        t = idx / (num_frames - 1)
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
            draw_tracked_text(draw, h // 2 - 65, model_name.upper(), font_model, INK_PRIMARY + (alpha,), width=w, tracking=5)
            draw_tracked_text(draw, h // 2 + 18, "ESPNER ALPINE SNOW OPTICS", font_sub, INK_GOLD + (alpha,), width=w, tracking=4)
            draw_tracked_text(draw, h // 2 + 52, "DUAL-SPHERICAL POLARIZED SERIES", font_sub, INK_SECONDARY + (int(alpha * 0.8),), width=w, tracking=2)

        pil_img = Image.alpha_composite(pil_img.convert("RGBA"), text_layer)
        out_bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

        # Fade to white in last 10 frames
        if idx >= num_frames - 10:
            fade_t = (idx - (num_frames - 10)) / 10.0
            out_bgr = cv2.addWeighted(out_bgr, 1.0 - fade_t, np.full((h, w, 3), 255, dtype=np.uint8), fade_t, 0)

        out.write(out_bgr)

    out.release()
    cmd = ["ffmpeg", "-y", "-i", raw_temp, "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-an", out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(raw_temp): os.remove(raw_temp)

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

def generate_front_photo_endcard_v5(photo_p, model_name, display_name, out_path, duration_sec=4.5, fps=24,
                                   slogan="把目光，還給世界。", sub="DUAL-SPHERICAL POLARIZED SNOW GOGGLE"):
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
    font_sub = ImageFont.truetype(FONT_SERIF, 20)
    
    raw_temp = out_path.replace(".mp4", "_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_temp, fourcc, fps, (w, h))
    
    white_canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    
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
        
        # Ground shadow
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
        
        pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        draw_tracked_text(draw, 145, "E S P N E R", font_brand, (15, 23, 42), width=w, tracking=8)
        draw_tracked_text(draw, 210, display_name.upper(), font_model, (71, 85, 105), width=w, tracking=4)
        draw_tracked_text(draw, 840, slogan, font_slogan, (15, 23, 42), width=w, tracking=4)
        draw_tracked_text(draw, 905, sub, font_sub, (100, 116, 139), width=w, tracking=3)
        
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        if i < 10:
            fade_in = i / 10.0
            frame_bgr = cv2.addWeighted(frame_bgr, fade_in, white_canvas, 1.0 - fade_in, 0)
            
        out.write(frame_bgr)
        
    out.release()
    cmd = ["ffmpeg", "-y", "-i", raw_temp, "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-an", out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.exists(raw_temp): os.remove(raw_temp)

def reencode_story_clip(in_mp4, out_mp4, duration=None, start_time=None):
    cmd = ["ffmpeg", "-y"]
    if start_time is not None:
        cmd.extend(["-ss", str(start_time)])
    if duration is not None:
        cmd.extend(["-t", str(duration)])
    cmd.extend([
        "-i", in_mp4,
        "-vf", "scale=720:1280:force_original_aspect_ratio=increase,crop=720:1280,fps=24",
        "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-an",
        out_mp4
    ])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def concat_parts_filter(clip_list, out_mp4):
    cmd = ["ffmpeg", "-y"]
    for c in clip_list:
        cmd.extend(["-i", c])
    
    n = len(clip_list)
    filter_parts = []
    for i in range(n):
        filter_parts.append(f"[{i}:v]fps=24,format=yuv420p[v{i}]")
    concat_part = "".join(f"[v{i}]" for i in range(n)) + f"concat=n={n}:v=1:a=0[outv]"
    filter_str = ";".join(filter_parts) + ";" + concat_part
    
    cmd.extend([
        "-filter_complex", filter_str,
        "-map", "[outv]",
        "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p", "-an",
        out_mp4
    ])
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    size_mb = os.path.getsize(out_mp4) / (1024 * 1024)
    print(f"-> Snow Commercial Built: {out_mp4} ({size_mb:.2f} MB)")

SNOW_CONFIGS = [
    {
        "out_name": "SNOW_1_ZERMATT_GLACIER_FREERIDE.mp4",
        "display": "ZERMATT FREERIDE",
        "story": [
            {"path": r"D:\廣告影片\EP00042_ZERMATT\02_gemini_carving.mp4", "t": 7.5},
            {"path": r"D:\廣告影片\EP00042_ZERMATT\01_gemini_departure.mp4", "t": 7.5}
        ],
        "photo": os.path.join(CATALOG_DIR, "EP00042 ZERMATT", "4N3A6295-1.jpg"),
        "slogan": "極境巡航，縱橫無垠雪野。",
        "sub": "ALPINE FREERIDE DUAL-SPHERICAL"
    },
    {
        "out_name": "SNOW_2_ENGELBERG_TITLIS_BLIZZARD.mp4",
        "display": "ENGELBERG TITLIS",
        "story": [
            {"path": r"D:\廣告影片\EP00041_ENGELBERG\01_gemini_departure.mp4", "t": 7.5},
            {"path": r"D:\廣告影片\EP00041_ENGELBERG\02_flow_carving.mp4", "t": 7.5}
        ],
        "photo": os.path.join(CATALOG_DIR, "EP00041 ENGELBERG", "4N3A6282.jpg"),
        "slogan": "狂風暴雪，無阻清晰前行。",
        "sub": "TITLIS STORM ALL-WEATHER OPTICS"
    },
    {
        "out_name": "SNOW_3_ANDORRA_PYRENEES_SUNSET.mp4",
        "display": "ANDORRA PYRENEES",
        "story": [
            {"path": r"D:\廣告影片\EP00043_ANDORRA\01_gemini_departure.mp4", "t": 7.5},
            {"path": r"D:\廣告影片\EP00043_ANDORRA\02_gemini_carving.mp4", "t": 7.5}
        ],
        "photo": r"D:\廣告影片\EP00043_ANDORRA\4N3A6310-2.jpg",
        "slogan": "暮光粉雪，收盡落日餘暉。",
        "sub": "PYRENEES SUNSET FRAMELESS PANORAMA"
    },
    {
        "out_name": "SNOW_4_ZERMATT_PRO_DOWNHILL_RUSH.mp4",
        "display": "ZERMATT PRO",
        "story": [
            {"path": r"D:\廣告影片\EP00042_ZERMATT\02_gemini_carving.mp4", "t": 7.5},
            {"path": r"D:\廣告影片\EP00042_ZERMATT\01_gemini_departure.mp4", "t": 7.5}
        ],
        "photo": os.path.join(CATALOG_DIR, "EP00042 ZERMATT", "4N3A6295-1.jpg"),
        "slogan": "極速破風，決勝毫釐之間。",
        "sub": "WORLD CUP SPEED RACING GOGGLE"
    }
]

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

def main():
    print("================================================================================")
    print("ESPNER 4 SNOW GOGGLES - FRONT PHOTO STUDIO EDITION (v5.0)")
    print("================================================================================")
    
    for cfg in SNOW_CONFIGS:
        out_name = cfg["out_name"]
        display = cfg["display"]
        print(f"\n---> Building Snow Goggle: {display} ({out_name})...")
        
        story_parts = []
        for i, s in enumerate(cfg["story"]):
            s_out = os.path.join(TEMP_DIR, f"{display}_story_{i}.mp4")
            reencode_story_clip(s["path"], s_out, duration=s.get("t"), start_time=s.get("ss"))
            story_parts.append(s_out)
            
        trans_out = os.path.join(TEMP_DIR, f"{display}_trans.mp4")
        generate_madrid_wall_transition_v5(display, trans_out, duration_sec=2.0)
        
        endcard_out = os.path.join(TEMP_DIR, f"{display}_endcard.mp4")
        generate_front_photo_endcard_v5(cfg["photo"], display, display, endcard_out, duration_sec=4.5,
                                        slogan=cfg.get("slogan", "把目光，還給世界。"),
                                        sub=cfg.get("sub", "DUAL-SPHERICAL POLARIZED SNOW GOGGLE"))
        
        final_mp4 = os.path.join(BASE_DIR, out_name)
        clip_list = story_parts + [trans_out, endcard_out]
        concat_parts_filter(clip_list, final_mp4)
        
        sheet_p = os.path.join(QA_DIR, out_name.replace(".mp4", "_v5_sheet.jpg"))
        generate_contact_sheet(final_mp4, sheet_p)
        
    print("\nALL 4 SNOW GOGGLES REBUILT WITH FRONT PHOTO ENDCARD SUCCESSFULLY!")

if __name__ == "__main__":
    main()

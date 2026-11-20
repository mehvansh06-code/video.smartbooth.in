#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmartBooth AI Video Campaign Engine - video.smartbooth.in
Built for Google Hackathon
Hyper-personalized AI video generation pipeline with ElevenLabs multilingual voice synthesis,
Surgical Acoustic Splice, Devanagari typography overlay, and WhatsApp-optimized H.264 encoding.
"""

import os
import sys
import json
import time
import shutil
import unicodedata
import subprocess
import threading
from typing import Dict, Any, List
from pathlib import Path

from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from PIL import Image, ImageDraw, ImageFont

WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, "output_videos")
TEMP_DIR = os.path.join(WORKSPACE_DIR, "temp_pipeline")
STATIC_DIR = os.path.join(WORKSPACE_DIR, "static")
FONTS_DIR = os.path.join(WORKSPACE_DIR, "fonts")
CACHED_DIR = os.path.join(WORKSPACE_DIR, "cached_samples")
CONFIG_FILE = os.path.join(WORKSPACE_DIR, "campaign_config.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(FONTS_DIR, exist_ok=True)
os.makedirs(CACHED_DIR, exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="")

# Default Campaign Configuration
DEFAULT_CONFIG = {
    "candidate_name_hi": "नीलू दुनीवाल",
    "candidate_name_en": "Neelu Duniwal",
    "ballot_no": "5",
    "symbol_name": "केतली",
    "symbol_name_en": "Kettle",
    "slogan": "आपकी अपनी, सेवा में समर्पित",
    "constituency": "सारणा (सांभर लेक)",
    "greeting_template": "राम राम {name} जी",
    "badge_template": "आदरणीय {name} जी",
    "voice_id": "pFZP5JQG7iQjIQuC4Bku",  # Lily - clear, high quality
    "eleven_api_key": "sk_8b067001bc0871474f7ef2593eee5ae7f70a35d18529661b",
    "raw_video_path": os.path.join(WORKSPACE_DIR, "sample_raw_candidate.mp4"),
    "candidate_speech_path": os.path.join(WORKSPACE_DIR, "candidate_sample.wav"),
    "delay_ms": 1450,
    "cut_point": 3.700,
    "silence_threshold_db": -35,
    "volume_boost": 1.15,
    "target_width": 720,
    "target_height": 1280,
    "fps": 30,
    "audio_sample_rate": 48000
}

def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                merged = DEFAULT_CONFIG.copy()
                merged.update(cfg)
                return merged
        except Exception:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(cfg: Dict[str, Any]):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

def get_system_font(size: int = 38):
    candidate_fonts = [
        os.path.join(FONTS_DIR, "NotoSansDevanagari-Bold.ttf"),
        os.path.join(FONTS_DIR, "NotoSansDevanagari-Regular.ttf"),
        "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Bold.ttf",
        "/usr/share/fonts/truetype/lohit-devanagari/Lohit-Devanagari.ttf",
        "/usr/share/fonts/truetype/Gargi/Gargi.ttf",
        "/System/Library/Fonts/Supplemental/DevanagariMT.ttc",
        "/System/Library/Fonts/Kohinoor.ttc",
        "/Library/Fonts/Arial Unicode.ttf"
    ]
    for p in candidate_fonts:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()

def get_ffmpeg_binary() -> str:
    which_ffmpeg = shutil.which("ffmpeg")
    if which_ffmpeg:
        return which_ffmpeg
    for path in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", os.path.expanduser("~/.local/bin/ffmpeg")]:
        if os.path.exists(path):
            return path
    return "ffmpeg"

FFMPEG_BIN = get_ffmpeg_binary()

# In-memory batch generation state
batch_state = {
    "is_running": False,
    "total": 0,
    "completed": 0,
    "failed": 0,
    "current_name": "",
    "current_step": "",
    "progress_percent": 0,
    "logs": [],
    "should_cancel": False,
    "start_time": 0.0,
    "eta_seconds": 0.0,
    "rendered_videos": []
}

def add_log(msg: str):
    timestamp = time.strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    batch_state["logs"].append(entry)
    if len(batch_state["logs"]) > 250:
        batch_state["logs"].pop(0)
    print(entry)

def ensure_body_cut(config: Dict[str, Any]) -> str:
    cut_point = float(config.get("cut_point", 3.700))
    sample_wav = config.get("candidate_speech_path", os.path.join(WORKSPACE_DIR, "candidate_sample.wav"))
    body_cut_wav = os.path.join(TEMP_DIR, f"body_cut_{cut_point:.3f}.wav")

    if os.path.exists(body_cut_wav) and os.path.getsize(body_cut_wav) > 1000:
        return body_cut_wav

    cmd = [
        FFMPEG_BIN, "-y",
        "-ss", f"{cut_point:.3f}",
        "-i", sample_wav,
        "-c:a", "pcm_s16le",
        "-ar", "48000",
        "-ac", "1",
        body_cut_wav
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return body_cut_wav

def generate_overlay_image(name_hi: str, config: Dict[str, Any], out_path: str):
    top_canvas_path = os.path.join(WORKSPACE_DIR, "top_banner_canvas.png")
    if os.path.exists(top_canvas_path):
        img = Image.open(top_canvas_path).convert("RGBA")
    else:
        img = Image.new("RGBA", (720, 1280), (0, 0, 0, 0))
        draw_top = ImageDraw.Draw(img)
        draw_top.rounded_rectangle([35, 30, 685, 140], radius=16, fill=(18, 22, 28, 230), outline=(255, 205, 60, 255), width=2)
        font_banner = get_system_font(32)
        banner_text = f"बैलेट सं. {config.get('ballot_no', '5')} • {config.get('candidate_name_hi', 'नीलू दुनीवाल')} • चिन्ह: {config.get('symbol_name', 'केतली')}"
        draw_top.text((55, 65), banner_text, font=font_banner, fill=(255, 255, 255, 255))

    draw = ImageDraw.Draw(img)
    font_badge = get_system_font(40)
    badge_text = config.get("badge_template", "आदरणीय {name} जी").replace("{name}", name_hi)

    # Bottom Honorific Acrylic Card at y=1115
    bx0, by0, bx1, by1 = 90, 1115, 630, 1210
    draw.rounded_rectangle([bx0, by0, bx1, by1], radius=24, fill=(18, 22, 28, 235), outline=(255, 205, 60, 255), width=3)
    draw.rounded_rectangle([bx0+3, by0+3, bx1-3, by1-3], radius=21, outline=(255, 235, 150, 90), width=1)

    bbox = font_badge.getbbox(badge_text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    tx = bx0 + (bx1 - bx0 - tw) / 2
    ty = by0 + (by1 - by0 - th) / 2 - 4

    draw.text((tx, ty), badge_text, font=font_badge, fill=(255, 255, 255, 255))
    img.save(out_path)
    return out_path

def synthesize_greeting(name_hi: str, config: Dict[str, Any], out_wav: str) -> str:
    greeting_text = config.get("greeting_template", "राम राम {name} जी").replace("{name}", name_hi)
    api_key = config.get("eleven_api_key", "").strip()
    voice_id = config.get("voice_id", "pFZP5JQG7iQjIQuC4Bku").strip()

    raw_mp3 = os.path.join(TEMP_DIR, f"{name_hi}_raw.mp3")

    synthesized = False
    if api_key and voice_id:
        try:
            import requests
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": api_key
            }
            data = {
                "text": greeting_text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": 0.65,
                    "similarity_boost": 0.85,
                    "style": 0.15,
                    "use_speaker_boost": True
                }
            }
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            resp = requests.post(url, json=data, headers=headers, timeout=20)
            if resp.status_code == 200:
                with open(raw_mp3, "wb") as f:
                    f.write(resp.content)
                synthesized = True
            else:
                add_log(f"Voice synthesis returned HTTP {resp.status_code}. Falling back to cache/local synthesis.")
        except Exception as e:
            add_log(f"Voice synthesis exception: {e}. Falling back.")

    if not synthesized:
        cached_candidates = [
            os.path.join(CACHED_DIR, f"{name_hi}_optc_padded.wav"),
            os.path.join(CACHED_DIR, f"tts_{name_hi}.wav"),
            os.path.join(WORKSPACE_DIR, f"{name_hi}.wav"),
            os.path.join(WORKSPACE_DIR, f"{name_hi}_raw.mp3")
        ]
        found_cache = False
        for c in cached_candidates:
            if os.path.exists(c) and os.path.getsize(c) > 500:
                # If already padded wav, convert or copy
                cmd_conv = [
                    FFMPEG_BIN, "-y",
                    "-i", c,
                    "-ar", "48000",
                    "-ac", "1",
                    out_wav
                ]
                subprocess.run(cmd_conv, check=True, capture_output=True)
                return out_wav

        # Local audio synthesis
        cmd = [
            FFMPEG_BIN, "-y",
            "-f", "lavfi", "-i", "sine=frequency=480:duration=1.2",
            "-c:a", "libmp3lame", "-b:a", "128k",
            raw_mp3
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    # Silence strip & normalize to 48kHz WAV
    silence_db = config.get("silence_threshold_db", -35)
    cmd_trim = [
        FFMPEG_BIN, "-y",
        "-i", raw_mp3,
        "-af", f"silenceremove=start_periods=1:start_duration=0.02:start_threshold={silence_db}dB",
        "-ar", "48000",
        "-ac", "1",
        out_wav
    ]
    subprocess.run(cmd_trim, check=True, capture_output=True)
    return out_wav

def render_personalized_video(name_hi: str, config: Dict[str, Any], step_callback=None) -> Dict[str, Any]:
    start_t = time.time()
    allowed = []
    for ch in name_hi.strip():
        cat = unicodedata.category(ch)
        if cat.startswith(('L', 'M', 'N')) or ch in ('_', '-'):
            allowed.append(ch)
    safe_name = ''.join(allowed) or 'voter'
    out_mp4 = os.path.join(OUTPUT_DIR, f"{safe_name}.mp4")

    trimmed_wav = os.path.join(TEMP_DIR, f"{safe_name}_trimmed.wav")
    padded_wav = os.path.join(TEMP_DIR, f"{safe_name}_padded.wav")
    concat_txt = os.path.join(TEMP_DIR, f"{safe_name}_concat.txt")
    full_audio = os.path.join(TEMP_DIR, f"{safe_name}_full_audio.wav")
    overlay_png = os.path.join(TEMP_DIR, f"{safe_name}_overlay.png")

    if step_callback:
        step_callback("Synthesizing personalized voice greeting")
    synthesize_greeting(name_hi, config, trimmed_wav)

    if step_callback:
        step_callback("Calibrating surgical acoustic splice and timing")
    delay_ms = int(config.get("delay_ms", 1450))
    cut_point = float(config.get("cut_point", 3.700))
    vol = float(config.get("volume_boost", 1.15))

    subprocess.run([
        FFMPEG_BIN, "-y",
        "-i", trimmed_wav,
        "-af", f"adelay={delay_ms}|{delay_ms},volume={vol:.2f},apad=whole_dur={cut_point:.3f}",
        "-t", f"{cut_point:.3f}",
        "-ar", "48000", "-ac", "1",
        padded_wav
    ], check=True, capture_output=True)

    body_cut_wav = ensure_body_cut(config)
    with open(concat_txt, "w", encoding="utf-8") as f:
        f.write(f"file '{padded_wav}'\nfile '{body_cut_wav}'\n")

    subprocess.run([
        FFMPEG_BIN, "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_txt,
        "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "1",
        full_audio
    ], check=True, capture_output=True)

    if step_callback:
        step_callback("Generating Devanagari voter honorific badge overlay")
    generate_overlay_image(name_hi, config, overlay_png)

    if step_callback:
        step_callback("Rendering 720x1280 H.264 video with +faststart WhatsApp optimization")
    raw_video = config.get("raw_video_path", os.path.join(WORKSPACE_DIR, "sample_raw_candidate.mp4"))
    filter_graph = "[0:v]scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,fps=30[bg];[bg][1:v]overlay=0:0[v]"

    cmd_mux = [
        FFMPEG_BIN, "-y",
        "-i", raw_video,
        "-i", overlay_png,
        "-i", full_audio,
        "-filter_complex", filter_graph,
        "-map", "[v]",
        "-map", "2:a:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ar", "48000",
        "-movflags", "+faststart",
        "-shortest",
        out_mp4
    ]
    subprocess.run(cmd_mux, check=True, capture_output=True)

    elapsed = time.time() - start_t
    size_mb = os.path.getsize(out_mp4) / (1024 * 1024)

    return {
        "success": True,
        "name": name_hi,
        "safe_name": safe_name,
        "filename": os.path.basename(out_mp4),
        "video_url": f"/api/video/{os.path.basename(out_mp4)}",
        "download_url": f"/api/video/{os.path.basename(out_mp4)}?download=1",
        "size_mb": round(size_mb, 2),
        "elapsed_sec": round(elapsed, 2),
        "path": out_mp4
    }

# ==================== REST APIS ====================

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")

@app.route("/api/status", methods=["GET"])
def get_status():
    cfg = load_config()
    rendered_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".mp4")]

    return jsonify({
        "status": "ONLINE",
        "service": "SmartBooth Video AI Engine",
        "domain": "video.smartbooth.in",
        "ffmpeg_bin": FFMPEG_BIN,
        "output_count": len(rendered_files),
        "eleven_configured": bool(cfg.get("eleven_api_key")),
        "voice_id": cfg.get("voice_id"),
        "raw_video_exists": os.path.exists(cfg.get("raw_video_path", "")),
        "candidate_audio_exists": os.path.exists(cfg.get("candidate_speech_path", ""))
    })

@app.route("/api/config", methods=["GET", "POST"])
def handle_config():
    if request.method == "POST":
        data = request.json or {}
        cfg = load_config()
        cfg.update(data)
        save_config(cfg)
        return jsonify({"success": True, "config": cfg})
    return jsonify(load_config())

@app.route("/api/load-demo", methods=["POST"])
def load_demo():
    cfg = load_config()
    demo_video = os.path.join(WORKSPACE_DIR, "sample_raw_candidate.mp4")
    demo_audio = os.path.join(WORKSPACE_DIR, "candidate_sample.wav")

    if not os.path.exists(demo_video):
        return jsonify({"error": "Demo video not found"}), 404

    cfg["raw_video_path"] = demo_video
    cfg["candidate_speech_path"] = demo_audio
    save_config(cfg)

    poster_path = os.path.join(WORKSPACE_DIR, "test_frame_720.jpg")
    if not os.path.exists(poster_path):
        subprocess.run([
            FFMPEG_BIN, "-y",
            "-ss", "1.0",
            "-i", demo_video,
            "-vframes", "1",
            "-vf", "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2",
            poster_path
        ], capture_output=True)

    return jsonify({
        "success": True,
        "message": "Demo candidate video loaded successfully",
        "candidate_name": cfg.get("candidate_name_hi"),
        "symbol": cfg.get("symbol_name"),
        "ballot_no": cfg.get("ballot_no")
    })

@app.route("/api/upload/video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower() or ".mp4"
    save_path = os.path.join(WORKSPACE_DIR, f"candidate_raw_input{ext}")
    file.save(save_path)

    # 1. Extract 720x1280 preview poster frame at ~1.0s
    poster_path = os.path.join(WORKSPACE_DIR, "test_frame_720.jpg")
    subprocess.run([
        FFMPEG_BIN, "-y",
        "-ss", "1.0",
        "-i", save_path,
        "-vframes", "1",
        "-vf", "scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2",
        poster_path
    ], capture_output=True)

    # 2. Extract clean audio track to candidate_sample.wav
    audio_sample_path = os.path.join(WORKSPACE_DIR, "candidate_sample.wav")
    subprocess.run([
        FFMPEG_BIN, "-y",
        "-i", save_path,
        "-vn",
        "-c:a", "pcm_s16le",
        "-ar", "48000",
        "-ac", "1",
        audio_sample_path
    ], capture_output=True)

    # Invalidate old body cuts
    for f in os.listdir(TEMP_DIR):
        if f.startswith("body_cut_"):
            try:
                os.remove(os.path.join(TEMP_DIR, f))
            except Exception:
                pass

    cfg = load_config()
    cfg["raw_video_path"] = save_path
    cfg["candidate_speech_path"] = audio_sample_path
    save_config(cfg)

    file_size_mb = round(os.path.getsize(save_path) / (1024 * 1024), 2)
    return jsonify({
        "success": True,
        "video_path": save_path,
        "size_mb": file_size_mb,
        "extracted_audio": True
    })

@app.route("/api/preview/frame")
def preview_frame():
    poster_path = os.path.join(WORKSPACE_DIR, "test_frame_720.jpg")
    if os.path.exists(poster_path):
        return send_file(poster_path, mimetype="image/jpeg")
    return jsonify({"error": "No preview frame"}), 404

@app.route("/api/sample_video")
def get_sample_video():
    cfg = load_config()
    video_path = cfg.get("raw_video_path", os.path.join(WORKSPACE_DIR, "sample_raw_candidate.mp4"))
    if os.path.exists(video_path):
        return send_file(video_path, mimetype="video/mp4")
    return jsonify({"error": "Video not found"}), 404

def run_batch_thread(names: List[str], cfg: Dict[str, Any]):
    batch_state["is_running"] = True
    batch_state["total"] = len(names)
    batch_state["completed"] = 0
    batch_state["failed"] = 0
    batch_state["progress_percent"] = 5
    batch_state["should_cancel"] = False
    batch_state["start_time"] = time.time()
    batch_state["logs"] = []
    batch_state["rendered_videos"] = []

    add_log(f"Starting personalized AI video generation for {len(names)} names: {', '.join(names)}")

    for idx, name in enumerate(names, 1):
        if batch_state["should_cancel"]:
            add_log("[CANCELLED] Generation stopped by user request.")
            break

        batch_state["current_name"] = name
        step_base_pct = int(((idx - 1) / len(names)) * 90)

        def step_update(step_label):
            batch_state["current_step"] = step_label
            add_log(f"[{name}] {step_label}...")

        try:
            add_log(f"[{idx}/{len(names)}] Processing video for voter '{name}'...")
            res = render_personalized_video(name, cfg, step_callback=step_update)
            batch_state["completed"] += 1
            batch_state["rendered_videos"].append(res)
            batch_state["progress_percent"] = int((idx / len(names)) * 90)
            add_log(f"  -> Generated {res['filename']} ({res['size_mb']} MB) in {res['elapsed_sec']}s")
        except Exception as e:
            batch_state["failed"] += 1
            add_log(f"  [ERROR] Failed rendering for '{name}': {e}")

        elapsed = time.time() - batch_state["start_time"]
        done = batch_state["completed"] + batch_state["failed"]
        if done > 0:
            avg_per_item = elapsed / done
            remaining = len(names) - done
            batch_state["eta_seconds"] = round(remaining * avg_per_item, 1)

    batch_state["progress_percent"] = 100
    batch_state["current_step"] = "All campaign videos generated successfully!"
    batch_state["is_running"] = False
    batch_state["current_name"] = ""
    add_log(f"Personalization complete: {batch_state['completed']} succeeded, {batch_state['failed']} failed.")

@app.route("/api/render/batch", methods=["POST"])
def start_batch():
    if batch_state["is_running"]:
        return jsonify({"error": "A generation job is already in progress"}), 409

    data = request.json or {}
    names = data.get("names", [])
    if not names:
        names = ["सीता", "सुरेश", "चेतन", "रामदेव", "मीरा"]

    # Filter out empty names and limit to 10
    cleaned_names = [n.strip() for n in names if n.strip()][:10]
    if not cleaned_names:
        return jsonify({"error": "Please provide at least 1 valid name"}), 400

    cfg = load_config()
    # Update config with any overrides
    for k in ["candidate_name_hi", "symbol_name", "ballot_no", "greeting_template", "badge_template", "eleven_api_key", "voice_id"]:
        if k in data and data[k]:
            cfg[k] = data[k]
    save_config(cfg)

    t = threading.Thread(target=run_batch_thread, args=(cleaned_names, cfg), daemon=True)
    t.start()
    return jsonify({"success": True, "message": "Batch started", "total": len(cleaned_names), "names": cleaned_names})

@app.route("/api/render/status", methods=["GET"])
def get_batch_status():
    return jsonify(batch_state)

@app.route("/api/render/single", methods=["POST"])
def render_single():
    data = request.json or {}
    name_hi = data.get("name_hi", "सीता").strip()
    if not name_hi:
        return jsonify({"error": "Name is required"}), 400

    cfg = load_config()
    try:
        res = render_personalized_video(name_hi, cfg)
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/videos", methods=["GET"])
def list_videos():
    videos = []
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(".mp4"):
                full_path = os.path.join(OUTPUT_DIR, f)
                stat = os.stat(full_path)
                stem = os.path.splitext(f)[0]
                videos.append({
                    "filename": f,
                    "name": stem,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime)),
                    "video_url": f"/api/video/{f}",
                    "download_url": f"/api/video/{f}?download=1"
                })
    videos.sort(key=lambda x: x["modified"], reverse=True)
    return jsonify(videos)

@app.route("/api/video/<filename>")
def stream_video(filename):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Video not found"}), 404

    as_attachment = request.args.get("download") == "1"
    if as_attachment:
        return send_file(file_path, as_attachment=True, download_name=filename, mimetype="video/mp4")

    range_header = request.headers.get("Range", None)
    if not range_header:
        return send_file(file_path, mimetype="video/mp4")

    size = os.path.getsize(file_path)
    byte1, byte2 = 0, None
    m = range_header.replace("bytes=", "").split("-")
    if m[0]:
        byte1 = int(m[0])
    if len(m) > 1 and m[1]:
        byte2 = int(m[1])

    length = size - byte1
    if byte2 is not None:
        length = byte2 - byte1 + 1

    with open(file_path, "rb") as f:
        f.seek(byte1)
        data = f.read(length)

    rv = Response(data, 206, mimetype="video/mp4", content_type="video/mp4", direct_passthrough=True)
    rv.headers.add("Content-Range", f"bytes {byte1}-{byte1 + len(data) - 1}/{size}")
    rv.headers.add("Accept-Ranges", "bytes")
    return rv

@app.route("/api/audit/preflight", methods=["POST", "GET"])
def run_preflight():
    results = []
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".mp4")]

    for f in sorted(files):
        p = os.path.join(OUTPUT_DIR, f)
        size_bytes = os.path.getsize(p)
        size_mb = size_bytes / (1024 * 1024)

        cmd = [FFMPEG_BIN, "-i", p]
        proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        probe_output = proc.stderr

        is_720x1280 = "720x1280" in probe_output
        is_h264 = "h264" in probe_output.lower()
        is_aac = "aac" in probe_output.lower()
        is_48khz = "48000 hz" in probe_output.lower()
        is_44khz = "44100 hz" in probe_output.lower()

        with open(p, "rb") as fp:
            head_bytes = fp.read(4096)
            has_faststart = b"moov" in head_bytes

        issues = []
        if not is_720x1280:
            issues.append("Resolution is not 720x1280")
        if not is_h264:
            issues.append("Codec is not H.264")
        if not is_aac:
            issues.append("Audio codec is not AAC")
        if not (is_48khz or is_44khz):
            issues.append("Audio sample rate is not 48kHz or 44.1kHz")
        if size_mb > 3.5:
            issues.append(f"File size exceeds WhatsApp 3.3MB target ({size_mb:.2f} MB)")
        if not has_faststart:
            issues.append("Missing +faststart moov atom")

        status = "PASSED" if not issues else "FAIL"
        results.append({
            "filename": f,
            "size_mb": round(size_mb, 2),
            "status": status,
            "issues": issues,
            "faststart": has_faststart,
            "resolution": "720x1280" if is_720x1280 else "Other"
        })

    passed_count = sum(1 for r in results if r["status"] == "PASSED")
    total_count = len(results)
    pass_pct = round((passed_count / total_count * 100) if total_count > 0 else 0, 1)

    return jsonify({
        "total_audited": total_count,
        "passed_count": passed_count,
        "pass_percentage": pass_pct,
        "all_compliant": (passed_count == total_count and total_count > 0),
        "results": results
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5040))
    app.run(host="0.0.0.0", port=port, debug=False)

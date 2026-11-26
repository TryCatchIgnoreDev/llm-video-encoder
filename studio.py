"""LLM Video Encoder — unified end-to-end CLI pipeline for lossless LLM arithmetic coding videos.

Works on any video containing speech, narration, or song lyrics (YouTube URL or local MP4).

Subcommands:
  python studio.py init     <url-or-mp4> --slug <name> [--transcript file.txt] [--lang en]
  python studio.py align    --slug <name> [--lang en] [--device auto|cuda|cpu]
  python studio.py compress --slug <name> --model <path.gguf> [--gpu-layers -1]
  python studio.py stills   --slug <name> [--t 15,60,120,200]
  python studio.py render   --slug <name> [--fps 50|60] [--workers 2]
  python studio.py shorts   --slug <name> --start 30 --duration 35 [--headline "..."]
  python studio.py thumb    --slug <name> --t 60 --big "635 BYTES" --zip "1.63 KB ZIP"
  python studio.py all      <url-or-mp4> --slug <name> --model <path.gguf> [--render]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request

STUDIO_ROOT = Path(__file__).resolve().parent
JOBS_ROOT = STUDIO_ROOT / "jobs"
MAIN_PY = Path(sys.executable)


def resolve_whisper_python(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_py = os.environ.get("WHISPER_PYTHON", "").strip()
    if env_py:
        return Path(env_py).expanduser().resolve()
    return MAIN_PY


def run_cmd(cmd: list[str], *, cwd: Path | None = None) -> None:
    print(">>", " ".join(str(x) for x in cmd), flush=True)
    res = subprocess.run([str(x) for x in cmd], cwd=str(cwd) if cwd else None)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(str(x) for x in cmd)}")


def ffprobe_meta(video_path: Path) -> dict:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration:stream=codec_type,width,height,r_frame_rate",
            "-of", "json", str(video_path),
        ],
        text=True,
    )
    info = json.loads(out)
    dur = float(info["format"]["duration"])
    vstream = next((s for s in info["streams"] if s.get("codec_type") == "video"), {})
    num, den = (vstream.get("r_frame_rate") or "30/1").split("/")
    native_fps = float(num) / max(1.0, float(den))
    return {
        "duration": round(dur, 3),
        "width": int(vstream.get("width", 1920)),
        "height": int(vstream.get("height", 1080)),
        "native_fps": round(native_fps, 3),
        "recommended_fps": 50 if abs(native_fps - 25.0) < 0.5 else 60,
    }


def fetch_lrclib(query: str, out_lrc: Path) -> bool:
    try:
        url = f"https://lrclib.net/api/search?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={"User-Agent": "llm-video-encoder/1.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            items = json.loads(resp.read().decode("utf-8"))
        synced_items = [x for x in items if (x.get("syncedLyrics") or "").strip()]
        if not synced_items:
            return False
        best = max(synced_items, key=lambda x: len(x["syncedLyrics"]))
        out_lrc.write_text(best["syncedLyrics"], encoding="utf-8")
        print(f"Saved synced LRC from lrclib.net ({len(best['syncedLyrics'].splitlines())} lines)", flush=True)
        return True
    except Exception as exc:
        print(f"lrclib lookup skipped ({exc})", flush=True)
        return False


def parse_json3_cues(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cues = []
    for ev in data.get("events", []):
        t0 = ev.get("tStartMs")
        dur = ev.get("dDurationMs", 0)
        segs = ev.get("segs")
        if t0 is None or not segs:
            continue
        txt = re.sub(r"\s+", " ", "".join(s.get("utf8", "") for s in segs)).strip()
        if txt:
            cues.append({"start": round(t0 / 1000.0, 3), "end": round((t0 + dur) / 1000.0, 3), "text": txt})
    return cues


def cmd_init(args: argparse.Namespace) -> int:
    job_dir = JOBS_ROOT / args.slug
    work = job_dir / "work"
    data = job_dir / "data"
    out = job_dir / "out"
    for d in (work, data, out):
        d.mkdir(parents=True, exist_ok=True)

    source_arg = args.source
    is_url = source_arg.startswith(("http://", "https://"))
    src_video = work / "source.mp4"
    title = args.slug
    lang = args.lang
    transcript_arg = args.transcript or args.lyrics

    if is_url:
        meta_raw = subprocess.check_output(
            ["yt-dlp", "--no-playlist", "--dump-single-json", "--no-warnings", source_arg],
            text=True,
        )
        yt_meta = json.loads(meta_raw)
        title = yt_meta.get("title") or args.slug
        manual_subs = list((yt_meta.get("subtitles") or {}).keys())
        print(f"YouTube title: {title} | manual subs: {manual_subs}", flush=True)

        # Pick best manual subtitle matching requested lang (exact or prefix like en-nP7-...)
        chosen_sub = None
        candidates = ([lang] if lang else []) + ["en"]
        for cand in candidates:
            if cand in manual_subs:
                chosen_sub = cand
                break
            pref = next((k for k in manual_subs if k.lower().startswith(f"{cand.lower()}-")), None)
            if pref:
                chosen_sub = pref
                break
        if not chosen_sub and manual_subs and not lang:
            chosen_sub = manual_subs[0]
        if not lang and chosen_sub:
            lang = chosen_sub.split("-")[0]

        if chosen_sub and not transcript_arg:
            run_cmd([
                "yt-dlp", "--no-playlist", "--skip-download",
                "--write-subs", "--sub-lang", chosen_sub, "--sub-format", "json3",
                "-o", str(work / "yt_sub"), source_arg,
            ])
            sub_file = work / f"yt_sub.{chosen_sub}.json3"
            if sub_file.is_file():
                cues = parse_json3_cues(sub_file)
                (work / "yt_manual_cues.json").write_text(
                    json.dumps(cues, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"Extracted {len(cues)} official manual '{chosen_sub}' cues!", flush=True)

        if not src_video.is_file():
            run_cmd([
                "yt-dlp", "--no-playlist",
                "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<=1080]+bestaudio/best[height<=1080]",
                "--merge-output-format", "mp4",
                "-o", str(src_video), source_arg,
            ])
    else:
        local_src = Path(source_arg).expanduser().resolve()
        if not local_src.is_file():
            raise FileNotFoundError(f"Source video not found: {local_src}")
        src_video = local_src

    if transcript_arg:
        shutil.copyfile(Path(transcript_arg).expanduser().resolve(), work / "transcript.txt")

    if args.lrclib and not (work / "yt_manual_cues.json").is_file() and not (work / "transcript.txt").is_file():
        q = re.sub(r"\([^)]*\)|\[[^\]]*\]|official|music|video|dance", " ", title, flags=re.I)
        q = re.sub(r"\s+", " ", q).strip()
        fetch_lrclib(q, work / "lrclib.lrc")

    meta = ffprobe_meta(src_video)
    wav_path = work / "audio16k.wav"
    if not wav_path.is_file():
        run_cmd(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_video), "-vn", "-ac", "1", "-ar", "16000", str(wav_path)])

    # Extract 5 reference frames for palette & corner placement inspection
    fdir = work / "preview_frames"
    fdir.mkdir(parents=True, exist_ok=True)
    for frac in (0.12, 0.30, 0.50, 0.72, 0.88):
        ts = round(meta["duration"] * frac, 1)
        run_cmd([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(ts), "-i", str(src_video),
            "-frames:v", "1", "-vf", "scale=960:-1",
            str(fdir / f"frame_{ts:.1f}.jpg"),
        ])

    job_cfg = {
        "slug": args.slug,
        "title": title,
        "source_url": source_arg if is_url else None,
        "source_video": str(src_video),
        "lang": lang,
        "duration": meta["duration"],
        "native_fps": meta["native_fps"],
        "fps": args.fps or meta["recommended_fps"],
        "workers": args.workers,
        "theme": args.theme,
        "position": args.position,
        "scale": args.scale,
        "margin_x": 40,
        "margin_y": 32,
        "palette": {},
    }
    cfg_path = job_dir / "job.json"
    cfg_path.write_text(json.dumps(job_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nInitialized job at {job_dir}")
    print(f"Edit {cfg_path} anytime to tweak theme/palette/position, or inspect {fdir}")
    return 0


def cmd_align(args: argparse.Namespace) -> int:
    job_dir = JOBS_ROOT / args.slug
    whisper_py = resolve_whisper_python(getattr(args, "whisper_python", None))
    cmd = [str(whisper_py), str(STUDIO_ROOT / "core" / "align_words.py"), "--job-dir", str(job_dir)]
    if args.lang:
        cmd.extend(["--lang", args.lang])
    if getattr(args, "chunk_sec", None):
        cmd.extend(["--chunk-sec", str(args.chunk_sec)])
    if getattr(args, "whisper_model", None):
        cmd.extend(["--whisper-model", str(args.whisper_model)])
    if getattr(args, "device", None):
        cmd.extend(["--device", str(args.device)])
    run_cmd(cmd)
    return 0


def cmd_compress(args: argparse.Namespace) -> int:
    job_dir = JOBS_ROOT / args.slug
    cmd = [str(MAIN_PY), str(STUDIO_ROOT / "core" / "compress_text.py"), "--job-dir", str(job_dir)]
    if getattr(args, "model", None):
        cmd.extend(["--model", str(args.model)])
    if getattr(args, "model_label", None):
        cmd.extend(["--model-label", str(args.model_label)])
    if getattr(args, "gpu_layers", None) is not None:
        cmd.extend(["--gpu-layers", str(args.gpu_layers)])
    if getattr(args, "n_ctx", 0):
        cmd.extend(["--n-ctx", str(args.n_ctx)])
    run_cmd(cmd)
    return 0


def cmd_stills(args: argparse.Namespace) -> int:
    job_dir = JOBS_ROOT / args.slug
    cmd = ["node", str(STUDIO_ROOT / "renderer" / "render.mjs"), "stills", "--job-dir", str(job_dir)]
    if getattr(args, "t", None):
        cmd.extend(["--t", args.t])
    run_cmd(cmd)
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    job_dir = JOBS_ROOT / args.slug
    cmd = ["node", str(STUDIO_ROOT / "renderer" / "render.mjs"), "video", "--job-dir", str(job_dir)]
    if getattr(args, "fps", None):
        cmd.extend(["--fps", str(args.fps)])
    if getattr(args, "workers", None):
        cmd.extend(["--workers", str(args.workers)])
    if getattr(args, "out", None):
        cmd.extend(["--out", str(args.out)])
    run_cmd(cmd)
    return 0


def cmd_shorts(args: argparse.Namespace) -> int:
    job_dir = JOBS_ROOT / args.slug
    cmd = [
        "node",
        str(STUDIO_ROOT / "renderer" / "render_shorts.mjs"),
        "--job-dir", str(job_dir),
        "--start", str(args.start),
        "--duration", str(args.duration),
    ]
    if args.headline:
        cmd.extend(["--headline", args.headline])
    if args.subhead:
        cmd.extend(["--subhead", args.subhead])
    if args.footer:
        cmd.extend(["--footer", args.footer])
    if args.crop:
        cmd.extend(["--crop", args.crop])
    if args.fps:
        cmd.extend(["--fps", str(args.fps)])
    if args.out:
        cmd.extend(["--out", str(args.out)])
    if args.still_only:
        cmd.append("--still-only")
    run_cmd(cmd)
    return 0


def cmd_thumb(args: argparse.Namespace) -> int:
    job_dir = JOBS_ROOT / args.slug
    work = job_dir / "work"
    out = job_dir / "out"
    out.mkdir(parents=True, exist_ok=True)
    job_cfg = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    src_video = Path(job_cfg.get("source_video") or (work / "source.mp4"))
    bg_path = Path(args.bg) if args.bg else (work / "thumb_bg.jpg")
    if not args.bg:
        run_cmd([
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(args.t), "-i", str(src_video),
            "-frames:v", "1", "-vf", "scale=1280:720:flags=lanczos",
            str(bg_path),
        ])
    out_png = Path(args.out) if args.out else (out / f"{args.slug}-thumb.png")
    cmd = [
        "node",
        str(STUDIO_ROOT / "renderer" / "render_thumb.mjs"),
        "--bg", str(bg_path),
        "--out", str(out_png),
        "--big", args.big,
        "--zip", args.zip,
        "--badge", args.badge,
        "--tag", args.tag,
        "--accent", args.accent,
        "--signal", args.signal,
        "--bg-shift-x", str(args.bg_shift_x),
    ]
    if args.footer:
        cmd.extend(["--footer", args.footer])
    run_cmd(cmd)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    cmd_init(args)
    cmd_align(args)
    cmd_compress(args)
    cmd_stills(args)
    if args.render:
        cmd_render(args)
    else:
        print(f"\nSemi-auto pause: inspect QA stills in {JOBS_ROOT / args.slug / 'out' / 'stills'}, tweak job.json if needed, then run:")
        print(f"  python studio.py render --slug {args.slug}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM Video Encoder — end-to-end pipeline for live LLM Arithmetic Coding video overlays.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_init_flags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("source", help="YouTube URL or local video path (.mp4, .mkv, .mov, etc.).")
        sp.add_argument("--slug", required=True, help="Short job folder name (e.g. shannon-talk or my-song).")
        sp.add_argument("--transcript", default=None, help="Optional local plain-text transcript file.")
        sp.add_argument("--lyrics", default=None, help="Alias for --transcript.")
        sp.add_argument("--lrclib", action="store_true", help="Optionally query lrclib.net for synced song lyrics when no subtitles/transcript exist.")
        sp.add_argument("--lang", default=None, help="Language code (e.g. en, es, fr; omit for auto-detect).")
        sp.add_argument("--theme", default="dark-slate", help="Theme preset: dark-slate, dark-forest, warm-cinema, neon-pop, light-lab.")
        sp.add_argument("--position", default="bottom-left", help="Overlay position: top-left, top-center, top-right, bottom-left, bottom-center, bottom-right.")
        sp.add_argument("--scale", type=float, default=1.38, help="Overlay scale factor (default 1.38).")
        sp.add_argument("--fps", type=int, default=None, help="Output FPS override (default auto: 50 for 25fps sources, else 60).")
        sp.add_argument("--workers", type=int, default=2, help="Parallel Chromium render workers.")

    p_init = sub.add_parser("init", help="Download/probe video, extract audio & preview frames, create job.json.")
    add_init_flags(p_init)
    p_init.set_defaults(func=cmd_init)

    p_al = sub.add_parser("align", help="Run Whisper word-level transcription/alignment -> data/transcript.json.")
    p_al.add_argument("--slug", required=True)
    p_al.add_argument("--lang", default=None)
    p_al.add_argument("--chunk-sec", type=float, default=None)
    p_al.add_argument("--whisper-model", default="large-v3", help="faster-whisper model name (e.g. large-v3, medium, small).")
    p_al.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p_al.add_argument("--whisper-python", default=None, help="Optional Python executable with faster-whisper installed (or set WHISPER_PYTHON).")
    p_al.set_defaults(func=cmd_align)

    p_co = sub.add_parser("compress", help="Run lossless LLM Arithmetic Coding + zoom trace -> data/compression.json.")
    p_co.add_argument("--slug", required=True)
    p_co.add_argument("--model", default=None, help="Path to local .gguf model (or set LLM_MODEL_PATH).")
    p_co.add_argument("--model-label", default=None, help="Display label shown in the video HUD (defaults to model filename).")
    p_co.add_argument("--gpu-layers", type=int, default=-1, help="GPU offload layers (-1 = all GPU, 0 = CPU only).")
    p_co.add_argument("--n-ctx", type=int, default=0, help="Context window (0 = auto based on text length).")
    p_co.set_defaults(func=cmd_compress)

    p_st = sub.add_parser("stills", help="Render composite QA stills using job.json theme/position.")
    p_st.add_argument("--slug", required=True)
    p_st.add_argument("--t", default=None, help="Comma-separated timestamps in seconds.")
    p_st.set_defaults(func=cmd_stills)

    p_rd = sub.add_parser("render", help="Render full 1080p MP4 with parallel Chromium workers + ffmpeg.")
    p_rd.add_argument("--slug", required=True)
    p_rd.add_argument("--fps", type=int, default=None)
    p_rd.add_argument("--workers", type=int, default=None)
    p_rd.add_argument("--out", default=None)
    p_rd.set_defaults(func=cmd_render)

    p_sh = sub.add_parser("shorts", help="Render a true-vertical 1080x1920 60fps vertical Short (no rotation needed).")
    p_sh.add_argument("--slug", required=True)
    p_sh.add_argument("--start", type=float, default=0.0, help="Clip start time in seconds.")
    p_sh.add_argument("--duration", type=float, default=35.0, help="Clip duration in seconds.")
    p_sh.add_argument("--headline", default=None, help="Top hook headline.")
    p_sh.add_argument("--subhead", default=None, help="Accent metric subtitle.")
    p_sh.add_argument("--footer", default=None, help="Bottom technical footer line.")
    p_sh.add_argument("--crop", default=None, help="Optional ffmpeg crop/scale expression for the 1080x608 video window.")
    p_sh.add_argument("--fps", type=int, default=60)
    p_sh.add_argument("--out", default=None)
    p_sh.add_argument("--still-only", action="store_true", help="Only render preview still for layout QA.")
    p_sh.set_defaults(func=cmd_shorts)

    p_th = sub.add_parser("thumb", help="Generate a high-CTR 1280x720 'AI VS ZIP' thumbnail PNG.")
    p_th.add_argument("--slug", required=True)
    p_th.add_argument("--t", type=float, default=30.0, help="Timestamp (seconds) to extract background frame from source video.")
    p_th.add_argument("--bg", default=None, help="Optional pre-extracted background image path.")
    p_th.add_argument("--out", default=None, help="Output PNG path.")
    p_th.add_argument("--big", required=True, help="Headline size text (e.g. '635 BYTES').")
    p_th.add_argument("--zip", required=True, help="Baseline ZIP text (e.g. '1.63 KB ZIP').")
    p_th.add_argument("--badge", required=True, help="Comparison badge (e.g. '2.6x SMALLER').")
    p_th.add_argument("--tag", default="LLM ARITHMETIC CODING", help="Top subtitle tag.")
    p_th.add_argument("--footer", default=None, help="Bottom footer text.")
    p_th.add_argument("--accent", default="#38D9FF")
    p_th.add_argument("--signal", default="#FF4D2E")
    p_th.add_argument("--bg-shift-x", type=float, default=-160.0)
    p_th.set_defaults(func=cmd_thumb)

    p_all = sub.add_parser("all", help="Run init -> align -> compress -> stills (add --render to also encode final MP4).")
    add_init_flags(p_all)
    p_all.add_argument("--chunk-sec", type=float, default=None)
    p_all.add_argument("--whisper-model", default="large-v3")
    p_all.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    p_all.add_argument("--whisper-python", default=None)
    p_all.add_argument("--model", default=None, help="Path to local .gguf model (or set LLM_MODEL_PATH).")
    p_all.add_argument("--model-label", default=None)
    p_all.add_argument("--gpu-layers", type=int, default=-1)
    p_all.add_argument("--n-ctx", type=int, default=0)
    p_all.add_argument("--t", default=None)
    p_all.add_argument("--out", default=None)
    p_all.add_argument("--render", action="store_true", help="Automatically continue into full MP4 render after stills.")
    p_all.set_defaults(func=cmd_all)

    return p


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

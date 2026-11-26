"""Modular word-level transcript & lyric aligner for llm-video-encoder.

Supports four text sources automatically (in priority order):
  1. Official manual YouTube subtitles (work/yt_manual_cues.json) — already synced to video
  2. Synced LRC (work/lrclib.lrc) — auto-estimates piecewise video offset via pass-1 Whisper
  3. Plain text transcript/lyrics (work/transcript.txt or work/lyrics.txt) — aligned via Whisper + DP
  4. Automatic Whisper transcription — when no subtitle or transcript file is provided,
     transcribes audio16k.wav directly with word-level timestamps!

Refines word-level start/end timestamps using faster-whisper (CUDA with automatic CPU fallback)
and Dynamic Programming alignment, writing data/transcript.json.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import statistics
import time


def norm_word(s: str) -> str:
    return re.sub(r"[^\w]", "", s.lower(), flags=re.UNICODE).replace("_", "")


def sim(a: str, b: str) -> float:
    na, nb = norm_word(a), norm_word(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def parse_lrc(path: Path) -> list[tuple[float, str]]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\[(\d+):(\d+(?:\.\d+)?)\]\s*(.*)", ln)
        if m and m.group(3).strip():
            out.append((int(m.group(1)) * 60.0 + float(m.group(2)), m.group(3).strip()))
    return out


def align_window_dp(gt_words: list[dict], asr_words: list[dict], max_dt: float = 5.0) -> dict[int, int]:
    n, m = len(gt_words), len(asr_words)
    if n == 0 or m == 0:
        return {}
    gap_gt, gap_asr = -0.35, -0.10
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_gt
        bt[i][0] = 2
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_asr
        bt[0][j] = 3

    for i in range(1, n + 1):
        gw = gt_words[i - 1]
        c0 = gw.get("cue_start")
        c1 = gw.get("cue_end")
        pt = gw.get("prior_t")
        for j in range(1, m + 1):
            aw = asr_words[j - 1]
            out_of_win = False
            if c0 is not None and c1 is not None:
                if aw["start"] < c0 - 1.5 or aw["start"] > c1 + 1.2:
                    out_of_win = True
            elif pt is not None and abs(aw["start"] - pt) > max_dt:
                out_of_win = True

            if out_of_win:
                match_score = -4.0
            else:
                sc = sim(gw["w"], aw["w"])
                dt = abs(aw["start"] - pt) if pt is not None else 0.0
                t_pen = max(-1.5, 0.3 * (1.0 - dt / 2.2)) if pt is not None else 0.0
                match_score = (2.3 * sc - 0.55) + t_pen

            best = dp[i - 1][j - 1] + match_score
            mv = 1
            up = dp[i - 1][j] + gap_gt
            if up > best:
                best = up
                mv = 2
            left = dp[i][j - 1] + gap_asr
            if left > best:
                best = left
                mv = 3
            dp[i][j] = best
            bt[i][j] = mv

    mapping: dict[int, int] = {}
    i, j = n, m
    while i > 0 or j > 0:
        mv = bt[i][j]
        if mv == 1:
            gw = gt_words[i - 1]
            aw = asr_words[j - 1]
            sc = sim(gw["w"], aw["w"])
            c0, c1 = gw.get("cue_start"), gw.get("cue_end")
            in_bounds = True
            if c0 is not None and c1 is not None:
                in_bounds = (c0 - 1.2 <= aw["start"] <= c1 + 0.9)
            elif gw.get("prior_t") is not None:
                in_bounds = abs(aw["start"] - gw["prior_t"]) <= max_dt
            if sc >= 0.45 and in_bounds:
                mapping[i - 1] = j - 1
            i -= 1
            j -= 1
        elif mv == 2:
            i -= 1
        else:
            j -= 1
    return mapping


def build_cues_from_lrc(lrc: list[tuple[float, str]], model, wav_path: Path, lang: str, duration: float) -> list[dict]:
    """Run a quick coarse Whisper pass to measure piecewise LRC->video offset and build video-timed cues."""
    print("Estimating piecewise LRC->video offset via coarse Whisper pass...", flush=True)
    segs, _ = model.transcribe(
        str(wav_path),
        language=lang,
        word_timestamps=False,
        vad_filter=True,
        condition_on_previous_text=False,
        beam_size=3,
        temperature=0.0,
    )
    pass1 = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segs]

    anchors: list[tuple[float, float]] = []
    for lt, ltxt in lrc:
        nl = " ".join(norm_word(w) for w in ltxt.split())
        if len(nl) < 8:
            continue
        best_sc, best_dt = 0.0, None
        for s in pass1:
            dt = s["start"] - lt
            if -10.0 <= dt <= 60.0:
                ns = " ".join(norm_word(w) for w in s["text"].split())
                sc = SequenceMatcher(None, nl, ns).ratio()
                if sc > best_sc:
                    best_sc, best_dt = sc, dt
        if best_sc >= 0.68 and best_dt is not None:
            anchors.append((lt, best_dt))

    cues = []
    for idx, (lt, ltxt) in enumerate(lrc):
        near = [dt for alt, dt in anchors if abs(alt - lt) <= 25.0]
        if not near and anchors:
            near = [dt for _, dt in sorted(anchors, key=lambda x: abs(x[0] - lt))[:5]]
        off = statistics.median(near) if near else 0.0
        t0 = max(0.0, lt + off)
        cues.append({"start": round(t0, 3), "end": round(min(duration, t0 + 4.0), 3), "text": ltxt})

    for idx in range(len(cues)):
        if idx > 0 and cues[idx]["start"] < cues[idx - 1]["start"] + 0.15:
            cues[idx]["start"] = round(cues[idx - 1]["start"] + 0.15, 3)
        nxt = cues[idx + 1]["start"] if idx + 1 < len(cues) else min(duration, cues[idx]["start"] + 4.0)
        cues[idx]["end"] = round(max(cues[idx]["start"] + 0.35, min(nxt, cues[idx]["start"] + 6.0)), 3)
    return cues


def _load_whisper_model(model_name: str, device: str) -> tuple[object, str, str]:
    from faster_whisper import WhisperModel

    if device == "cpu":
        return WhisperModel(model_name, device="cpu", compute_type="int8"), "cpu", "int8"
    if device == "cuda":
        return WhisperModel(model_name, device="cuda", compute_type="int8_float16"), "cuda", "int8_float16"

    try:
        return WhisperModel(model_name, device="cuda", compute_type="int8_float16"), "cuda", "int8_float16"
    except Exception as exc:
        print(f"CUDA Whisper unavailable ({exc}); falling back to CPU (int8)...", flush=True)
        return WhisperModel(model_name, device="cpu", compute_type="int8"), "cpu", "int8"


def transcribe_directly(model, wav_path: Path, lang: str | None) -> list[dict]:
    """Transcribe any spoken/sung video directly when no subtitle or transcript file exists."""
    print("No subtitle or transcript file found — running direct Whisper word-level transcription...", flush=True)
    t0 = time.perf_counter()
    segs, info = model.transcribe(
        str(wav_path),
        language=lang,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
        temperature=0.0,
    )
    lines_out: list[dict] = []
    for seg in segs:
        txt = re.sub(r"\s+", " ", (seg.text or "")).strip()
        if not txt:
            continue
        words = []
        for w in (seg.words or []):
            tw = (w.word or "").strip()
            if not tw:
                continue
            ws = float(w.start)
            we = max(ws + 0.04, float(w.end))
            words.append(
                {
                    "w": tw,
                    "start": round(ws, 3),
                    "end": round(we, 3),
                    "conf": round(float(getattr(w, "probability", 0.9)), 3),
                }
            )
        if not words:
            ws = float(seg.start)
            we = max(ws + 0.06, float(seg.end))
            parts = txt.split()
            step = (we - ws) / max(1, len(parts))
            words = [
                {
                    "w": tok,
                    "start": round(ws + idx * step, 3),
                    "end": round(ws + (idx + 1) * step, 3),
                    "conf": 0.7,
                }
                for idx, tok in enumerate(parts)
            ]
        lines_out.append(
            {
                "i": len(lines_out),
                "text": " ".join(w["w"] for w in words),
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "words": words,
            }
        )
    det_lang = getattr(info, "language", lang or "unknown")
    print(f"Direct Whisper transcription finished in {time.perf_counter() - t0:.1f}s (lang={det_lang}, {len(lines_out)} lines)", flush=True)
    return lines_out


def main() -> int:
    parser = argparse.ArgumentParser(description="Word-level forced alignment or transcription for llm-video-encoder.")
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--lang", default=None, help="Language code override (e.g. en, es, fr; omit for auto-detect).")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--chunk-sec", type=float, default=14.0)
    args = parser.parse_args()

    job_dir = args.job_dir.resolve()
    work = job_dir / "work"
    data_dir = job_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    job_cfg = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))

    wav_path = work / "audio16k.wav"
    duration = float(job_cfg.get("duration", 300.0))
    lang = args.lang or job_cfg.get("lang") or None

    t0 = time.perf_counter()
    model, used_dev, used_ct = _load_whisper_model(args.whisper_model, args.device)
    print(f"Loaded faster-whisper '{args.whisper_model}' ({used_dev}/{used_ct}) in {time.perf_counter() - t0:.1f}s", flush=True)

    manual_cues_path = work / "yt_manual_cues.json"
    lrc_path = work / "lrclib.lrc"
    plain_path = work / "transcript.txt"
    if not plain_path.is_file() and (work / "lyrics.txt").is_file():
        plain_path = work / "lyrics.txt"

    cues: list[dict] = []
    source_kind = ""
    if manual_cues_path.is_file():
        raw_cues = json.loads(manual_cues_path.read_text(encoding="utf-8"))
        # Strip music-note wrappers and drop pure non-speech cues like [Music] or (Applause)
        cues = []
        for c in raw_cues:
            txt = re.sub(r"[♪♫]+", "", c.get("text", "")).strip()
            txt = re.sub(r"\s+", " ", txt)
            if not txt:
                continue
            if re.fullmatch(r"[\[\(][^()\[\]]*[\]\)]", txt):
                continue
            cues.append({"start": c["start"], "end": c["end"], "text": txt})
        source_kind = "yt_manual"
    elif lrc_path.is_file():
        lrc = parse_lrc(lrc_path)
        if lrc:
            cues = build_cues_from_lrc(lrc, model, wav_path, lang or "en", duration)
            source_kind = "lrclib"

    if not cues and plain_path.is_file():
        lines_txt = [ln.strip() for ln in plain_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        # Evenly spaced fallback priors if no cue file exists; Whisper pass below will align
        step = max(0.5, (duration - 6.0) / max(1, len(lines_txt)))
        cues = [
            {"start": round(3.0 + i * step, 3), "end": round(3.0 + (i + 1) * step, 3), "text": ln}
            for i, ln in enumerate(lines_txt)
        ]
        source_kind = "plain_transcript"

    if not cues:
        lines_out = transcribe_directly(model, wav_path, lang)
        if not lines_out:
            raise RuntimeError(f"Whisper detected no speech in {wav_path}")
        out_path = data_dir / "transcript.json"
        out_path.write_text(json.dumps({"lines": lines_out}, ensure_ascii=False, indent=1), encoding="utf-8")
        total_w = sum(len(ln["words"]) for ln in lines_out)
        print(f"Wrote {out_path} ({len(lines_out)} lines, {total_w} words, {lines_out[0]['start']}s -> {lines_out[-1]['end']}s)")
        return 0

    print(f"Using cue source '{source_kind}': {len(cues)} lines ({cues[0]['start']}s -> {cues[-1]['end']}s)", flush=True)

    # Chunk cues into singing-safe windows (~14s) with exact ground-truth prompt
    chunks: list[tuple[float, float, str]] = []
    i = 0
    while i < len(cues):
        t_lo = max(0.0, cues[i]["start"] - 0.7)
        j = i
        while j < len(cues) and (cues[j]["end"] - cues[i]["start"]) < args.chunk_sec:
            j += 1
        if j == i:
            j = i + 1
        t_hi = min(duration, cues[j - 1]["end"] + 0.9)
        prompt = " ".join(cues[k]["text"] for k in range(i, j))
        chunks.append((round(t_lo, 2), round(t_hi, 2), prompt))
        i = j

    t1 = time.perf_counter()
    raw_asr = []
    for ci, (t_lo, t_hi, prompt) in enumerate(chunks):
        segs, _ = model.transcribe(
            str(wav_path),
            language=lang,
            word_timestamps=True,
            clip_timestamps=[t_lo, t_hi],
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=prompt,
            beam_size=5,
            temperature=0.0,
            no_speech_threshold=None,
            log_prob_threshold=None,
            compression_ratio_threshold=None,
        )
        cnt = 0
        for s in segs:
            for w in (s.words or []):
                tw = w.word.strip()
                if not tw:
                    continue
                ws, we = float(w.start), float(w.end)
                if ws < t_lo - 0.15 or ws > t_hi + 0.15:
                    continue
                raw_asr.append(
                    {
                        "w": tw,
                        "start": round(ws, 3),
                        "end": round(max(ws + 0.04, we), 3),
                        "prob": round(float(w.probability), 4),
                    }
                )
                cnt += 1
        if (ci + 1) % 5 == 0 or ci + 1 == len(chunks):
            print(f"  chunk {ci + 1:2d}/{len(chunks)} [{t_lo:5.1f}..{t_hi:5.1f}s] -> +{cnt:2d}w (total {len(raw_asr)}w)", flush=True)

    # Deduplicate overlapping chunk boundary words
    raw_asr.sort(key=lambda x: x["start"])
    asr_words = []
    for w in raw_asr:
        if asr_words and abs(w["start"] - asr_words[-1]["start"]) < 0.16 and norm_word(w["w"]) == norm_word(asr_words[-1]["w"]):
            if w["prob"] > asr_words[-1]["prob"]:
                asr_words[-1] = w
            continue
        asr_words.append(w)

    (work / "whisper_words.json").write_text(json.dumps({"words": asr_words}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Whisper chunked pass finished in {time.perf_counter() - t1:.1f}s -> {len(asr_words)} words", flush=True)

    # Build GT word list with syllable/length-weighted cue priors
    gt_words = []
    for li, c in enumerate(cues):
        ws = c["text"].split()
        span = max(0.25, c["end"] - c["start"])
        weights = [max(1.5, float(len(norm_word(w)))) for w in ws]
        tot_w = sum(weights)
        acc = 0.0
        for wi, (w, wt) in enumerate(zip(ws, weights)):
            p0 = c["start"] + span * (acc / tot_w)
            acc += wt
            p1 = c["start"] + span * (acc / tot_w)
            gt_words.append(
                {
                    "li": li,
                    "wi": wi,
                    "w": w,
                    "prior_t": p0,
                    "prior_end": p1,
                    "cue_start": c["start"] if source_kind != "plain_transcript" else None,
                    "cue_end": c["end"] if source_kind != "plain_transcript" else None,
                    "start": None,
                    "end": None,
                    "conf": 0.6,
                }
            )

    # Block DP alignment (24-line blocks so both 50-line and 400-line songs stay O(N) fast!)
    total_matched = 0
    block_lines = 24
    for b0 in range(0, len(cues), block_lines):
        b1 = min(len(cues), b0 + block_lines)
        blk_idx = [k for k, gw in enumerate(gt_words) if b0 <= gw["li"] < b1]
        if not blk_idx:
            continue
        blk_gt = [gt_words[k] for k in blk_idx]
        t_lo = (blk_gt[0]["cue_start"] if blk_gt[0]["cue_start"] is not None else blk_gt[0]["prior_t"]) - 4.0
        t_hi = (blk_gt[-1]["cue_end"] if blk_gt[-1]["cue_end"] is not None else blk_gt[-1]["prior_end"]) + 4.0
        blk_asr = [w for w in asr_words if t_lo <= w["start"] <= t_hi]
        mp = align_window_dp(blk_gt, blk_asr)
        for loc_i, asr_j in mp.items():
            gidx = blk_idx[loc_i]
            aw = blk_asr[asr_j]
            gt_words[gidx]["start"] = float(aw["start"])
            gt_words[gidx]["end"] = max(float(aw["start"]) + 0.05, float(aw["end"]))
            gt_words[gidx]["conf"] = round(float(aw.get("prob", 0.9)), 3)
            total_matched += 1

    print(f"DP matched {total_matched}/{len(gt_words)} words ({total_matched / max(1, len(gt_words)) * 100:.1f}%)")

    # Interpolate unmatched words strictly inside their own cue boundaries
    by_li: dict[int, list[dict]] = {}
    for gw in gt_words:
        by_li.setdefault(gw["li"], []).append(gw)

    for li, wl in by_li.items():
        m = len(wl)
        for wi in range(m):
            if wl[wi]["start"] is not None:
                continue
            l = wi - 1
            while l >= 0 and wl[l]["start"] is None:
                l -= 1
            r = wi + 1
            while r < m and wl[r]["start"] is None:
                r += 1
            c_lo = wl[wi]["cue_start"] if wl[wi]["cue_start"] is not None else wl[wi]["prior_t"]
            c_hi = wl[wi]["cue_end"] if wl[wi]["cue_end"] is not None else wl[wi]["prior_end"]
            t0 = wl[l]["end"] if l >= 0 else c_lo
            t1 = wl[r]["start"] if r < m else c_hi
            if t1 <= t0 + 0.05:
                wl[wi]["start"] = wl[wi]["prior_t"]
                wl[wi]["end"] = max(wl[wi]["prior_t"] + 0.06, wl[wi]["prior_end"])
            else:
                sub = wl[(l + 1) : r]
                wts = [max(1.5, float(len(norm_word(x["w"])))) for x in sub]
                tot = sum(wts)
                pos = wi - (l + 1)
                a0 = sum(wts[:pos]) / tot
                a1 = sum(wts[: pos + 1]) / tot
                ws = t0 + (t1 - t0) * a0
                we = t0 + (t1 - t0) * a1
                wl[wi]["start"] = ws
                wl[wi]["end"] = max(ws + 0.06, we)

    # Global monotonic pass
    for idx in range(len(gt_words)):
        if idx > 0:
            prev, cur = gt_words[idx - 1], gt_words[idx]
            if cur["start"] < prev["start"] + 0.04:
                cur["start"] = prev["start"] + 0.05
            if prev["end"] > cur["start"]:
                prev["end"] = max(prev["start"] + 0.03, cur["start"])
            if cur["end"] < cur["start"] + 0.05:
                cur["end"] = cur["start"] + 0.12
        if gt_words[idx]["end"] - gt_words[idx]["start"] > 1.3:
            gt_words[idx]["end"] = gt_words[idx]["start"] + 1.3

    lines_out = []
    for li, c in enumerate(cues):
        wl = by_li[li]
        lines_out.append(
            {
                "i": li,
                "text": c["text"],
                "start": round(wl[0]["start"], 3),
                "end": round(wl[-1]["end"], 3),
                "words": [
                    {"w": w["w"], "start": round(w["start"], 3), "end": round(w["end"], 3), "conf": w["conf"]}
                    for w in wl
                ],
            }
        )

    out_path = data_dir / "transcript.json"
    out_path.write_text(json.dumps({"lines": lines_out}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wrote {out_path} ({len(lines_out)} lines, {len(gt_words)} words, {lines_out[0]['start']}s -> {lines_out[-1]['end']}s)")
    step = max(1, len(lines_out) // 10)
    for sid in list(range(0, len(lines_out), step)) + [len(lines_out) - 1]:
        ln = lines_out[sid]
        print(f"  [{ln['i']:3d}] {ln['start']:6.2f} -> {ln['end']:6.2f}s ({len(ln['words']):2d}w) | {ln['text'][:68]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

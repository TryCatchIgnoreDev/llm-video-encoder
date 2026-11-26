"""Modular lossless LLM arithmetic coder + visualization trace generator for llm-video-encoder.

Automatically selects optimal n_ctx and Decimal precision for any transcript length (100 bytes to 30+ KB),
performs exact full-vocab Softmax CDF + 62-bit Arithmetic Coding, verifies 100% lossless SHA-256
round-trip decompression, and emits data/encoded.llmc + data/compression.json.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, getcontext
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import zlib

import numpy as np

CORE_DIR = Path(__file__).resolve().parent


def _infer_model_label(model_path: Path) -> str:
    stem = model_path.stem
    # Strip common GGUF quantization suffixes like .Q4_K_S, -Q4_K_M, .Q8_0
    cleaned = re.sub(r"[._-](?:I?Q\d+(?:_[A-Z0-9]+)*|F16|BF16|F32)$", "", stem, flags=re.IGNORECASE)
    return cleaned or stem


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def format_token_disp(piece: str) -> str:
    if piece == "\n":
        return "\\n"
    if piece == "\t":
        return "\\t"
    s = piece.replace("\n", "\\n").replace("\t", "\\t")
    if s.startswith(" "):
        s = "_" + s[1:]
    return s


def build_byte_timeline(lines: list[dict]) -> tuple[str, bytes, list[dict]]:
    parts: list[str] = []
    byte_meta: list[dict] = []

    def push_str(s: str, line_idx: int, word_idx: int, kind: str, t0: float, t1: float) -> None:
        parts.append(s)
        enc = s.encode("utf-8")
        nb = max(1, len(enc))
        for bi in range(len(enc)):
            byte_meta.append(
                {
                    "lineIdx": line_idx,
                    "wordIdx": word_idx,
                    "kind": kind,
                    "t0": lerp(t0, t1, bi / nb),
                    "t1": lerp(t0, t1, (bi + 1) / nb),
                }
            )

    for li, line in enumerate(lines):
        if li > 0:
            prev_end = float(lines[li - 1]["end"])
            next_start = float(line["start"])
            gap = next_start - prev_end
            if gap > 0.05:
                nl_t0 = prev_end + min(0.06, gap * 0.2)
                nl_t1 = prev_end + min(0.24, gap * 0.65)
            else:
                nl_t0 = max(0.0, prev_end - 0.03)
                nl_t1 = prev_end
            push_str("\n", li - 1, -1, "nl", nl_t0, max(nl_t0 + 0.02, nl_t1))

        words = line["words"]
        for wi, w in enumerate(words):
            ws = float(w["start"])
            we = float(w["end"])
            if wi > 0:
                push_str(" ", li, wi, "space", ws, ws)
            push_str(str(w["w"]), li, wi, "char", ws, we)

    full_text = "".join(parts)
    raw_bytes = full_text.encode("utf-8")
    assert len(raw_bytes) == len(byte_meta)
    return full_text, raw_bytes, byte_meta


def assign_token_times(piece_bytes_list: list[bytes], raw_bytes: bytes, byte_meta: list[dict]) -> list[dict]:
    cursor = 0
    raw_spans: list[dict] = []
    for i, pb in enumerate(piece_bytes_list):
        n = len(pb)
        assert raw_bytes[cursor : cursor + n] == pb, f"Token {i} byte mismatch at {cursor}"
        span_meta = byte_meta[cursor : cursor + n]
        non_space = [m for m in span_meta if m["kind"] != "space"]
        primary = non_space if non_space else span_meta
        t0 = float(primary[0]["t0"])
        t1 = float(primary[-1]["t1"])
        raw_spans.append(
            {
                "byteStart": cursor,
                "byteEnd": cursor + n,
                "lineIdx": int(primary[0]["lineIdx"]),
                "wordIdx": int(primary[0]["wordIdx"]),
                "t0": t0,
                "t1": max(t0 + 0.03, t1),
                "isNewline": pb == b"\n",
            }
        )
        cursor += n
    assert cursor == len(raw_bytes)

    for i in range(1, len(raw_spans)):
        if raw_spans[i]["t0"] <= raw_spans[i - 1]["t0"] + 0.02:
            raw_spans[i]["t0"] = raw_spans[i - 1]["t0"] + 0.025
            if raw_spans[i]["t1"] <= raw_spans[i]["t0"] + 0.025:
                raw_spans[i]["t1"] = raw_spans[i]["t0"] + 0.035
    return raw_spans


def _log10_dec(val: Decimal) -> float:
    t = val.as_tuple()
    n_dig = len(t.digits)
    if n_dig == 0:
        return -1e9
    k = min(12, n_dig)
    lead = 0.0
    for d in t.digits[:k]:
        lead = lead * 10.0 + d
    lead /= 10.0 ** (k - 1)
    return math.log10(lead) + (t.exponent + n_dig - 1)


def decimal_prefix_and_ticks(L_prev: Decimal, H_prev: Decimal, L_cur: Decimal, H_cur: Decimal) -> dict:
    W_prev = H_prev - L_prev
    W_cur = H_cur - L_cur
    C_prev = (L_prev + H_prev) / Decimal(2)
    C_cur = (L_cur + H_cur) / Decimal(2)

    log10_wprev = _log10_dec(W_prev)
    log10_wcur = _log10_dec(W_cur)
    d_prev = max(1, math.floor(-log10_wprev))
    d_cur = max(1, math.ceil(-log10_wcur) + 1)

    d_min = max(1, d_prev - 1)
    d_max = max(d_min, d_cur)

    def fmt_tick(m: int, d: int) -> str:
        if m == 0:
            return "0.0"
        if m == 10**d:
            return "1.0"
        if d <= 7:
            s = str(m).rjust(d + 1, "0")
            whole = s[: len(s) - d]
            frac = s[len(s) - d :].rstrip("0") or "0"
            return f"{whole}.{frac}"
        return f"…{m % 10000:04d}"

    ten_pow_dcur = Decimal(1).scaleb(d_cur)
    mid_scaled = int((C_cur * ten_pow_dcur).to_integral_value())
    if d_cur <= 9:
        mid_str = str(max(0, mid_scaled)).rjust(d_cur + 1, "0")
        prefix_str = f"0.{mid_str[len(mid_str) - d_cur :]}"
    else:
        lead4 = int((C_cur * Decimal(10000)).to_integral_value())
        tail5 = max(0, mid_scaled) % 100000
        prefix_str = f"0.{lead4:04d}…{tail5:05d}"

    levels = []
    log_tot = (log10_wprev - log10_wcur) * math.log(10.0)
    for d in range(d_min, d_max + 1):
        step = Decimal(1).scaleb(-d)
        step_rel = float(step / W_cur)
        w_lo = max(W_cur, step * Decimal("1.1"))
        w_hi = min(W_prev * Decimal("1.6"), step * Decimal("22"))
        if w_lo > w_hi:
            if d not in (d_min, d_cur):
                continue
            w_lo = W_cur
            w_hi = W_prev * Decimal("1.6")

        centers = []
        for w_cam in (w_lo, w_hi):
            if log_tot > 1e-9:
                alpha = max(0.0, min(1.0, ((log10_wprev - _log10_dec(w_cam)) * math.log(10.0)) / log_tot))
            else:
                alpha = 1.0
            c_est = C_prev + (C_cur - C_prev) * Decimal(f"{alpha:.6f}")
            centers.append((c_est - w_cam * Decimal("0.9"), c_est + w_cam * Decimal("0.9")))

        v_lo = max(Decimal(0), min(c[0] for c in centers))
        v_hi = min(Decimal(1), max(c[1] for c in centers))
        ten_d = Decimal(1).scaleb(d)
        m0 = max(0, int((v_lo * ten_d).to_integral_value()) - 1)
        m1 = min(10**d, int((v_hi * ten_d).to_integral_value()) + 1)
        if m1 - m0 > 24:
            mc = int((C_cur * ten_d).to_integral_value())
            m0 = max(0, mc - 12)
            m1 = min(10**d, mc + 12)

        origin_rel = float((Decimal(m0) * step - L_cur) / W_cur)
        ticks = [[m - m0, fmt_tick(m, d), m % 5 == 0] for m in range(m0, m1 + 1)]
        levels.append({"d": d, "originRel": origin_rel, "stepRel": step_rel, "ticks": ticks})

    return {"dCur": d_cur, "prefix": prefix_str, "levels": levels}


def select_row_segments(comp, cdf: np.ndarray, chosen_id: int, total_freq: int, label_cache: dict[int, str]) -> tuple[list[list], int]:
    freqs = np.diff(cdf)
    vocab_size = len(freqs)
    top_k = min(38, vocab_size)
    top_ids = np.argpartition(freqs, -top_k)[-top_k:]

    lo_id = max(0, chosen_id - 150)
    hi_id = min(vocab_size, chosen_id + 151)
    local_freqs = freqs[lo_id:hi_id]
    local_top_k = min(16, len(local_freqs))
    local_top = lo_id + np.argpartition(local_freqs, -local_top_k)[-local_top_k:]

    adj = np.arange(max(0, chosen_id - 4), min(vocab_size, chosen_id + 5), dtype=np.int64)
    keep_ids = np.unique(np.concatenate([top_ids, local_top, adj, np.array([chosen_id], dtype=np.int64)]))
    keep_ids.sort()

    def get_label(tid: int) -> str:
        lbl = label_cache.get(tid)
        if lbl is None:
            raw = comp.token_piece_display(tid)
            lbl = format_token_disp(raw)
            if len(label_cache) < 32768:
                label_cache[tid] = lbl
        return lbl

    segs: list[list] = []
    chosen_seg_idx = 0
    for idx, tid_np in enumerate(keep_ids):
        tid = int(tid_np)
        u0 = int(cdf[tid]) / total_freq
        u1 = int(cdf[tid + 1]) / total_freq
        if tid == chosen_id:
            chosen_seg_idx = idx
        is_adj = abs(tid - chosen_id) <= 3
        lbl = get_label(tid) if (tid == chosen_id or is_adj or (u1 - u0) >= 0.0022) else ""
        segs.append([u0, u1, lbl])

    return segs, chosen_seg_idx


def main() -> int:
    parser = argparse.ArgumentParser(description="Lossless LLM arithmetic compression for an llm-video-encoder job.")
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None, help="Path to local .gguf model (or set LLM_MODEL_PATH).")
    parser.add_argument("--model-label", default=None, help="Short model name shown in the HUD (defaults to filename).")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="GPU offload layers (-1 = all GPU, 0 = CPU only).")
    parser.add_argument("--n-ctx", type=int, default=0, help="Context window (0 = auto based on text size).")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        env_model = os.environ.get("LLM_MODEL_PATH", "").strip()
        if env_model:
            model_path = Path(env_model)
    if model_path is None:
        raise SystemExit(
            "Error: Provide a local .gguf model via --model /path/to/model.gguf or set LLM_MODEL_PATH."
        )
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"GGUF model not found: {model_path}")
    model_label = args.model_label or _infer_model_label(model_path)

    sys.path.insert(0, str(CORE_DIR))
    from arithmetic_coder import ArithmeticEncoder, BitOutputStream, DEFAULT_CDF_BITS, FULL_RANGE, logits_to_cdf
    from compressor import FORMAT_VERSION, HEADER_V2_STRUCT, LLMCompressor, MAGIC

    job_dir = args.job_dir.resolve()
    data_dir = job_dir / "data"
    transcript_json = data_dir / "transcript.json"
    if not transcript_json.is_file() and (data_dir / "lyrics.json").is_file():
        transcript_json = data_dir / "lyrics.json"
    out_llmc = data_dir / "encoded.llmc"
    out_json = data_dir / "compression.json"

    transcript_data = json.loads(transcript_json.read_text(encoding="utf-8"))
    lines = transcript_data["lines"]
    full_text, raw_bytes, byte_meta = build_byte_timeline(lines)
    zlib_blob = zlib.compress(raw_bytes, level=9)
    zlib_bytes = len(zlib_blob)
    zlib_bitstring = "".join(f"{b:08b}" for b in zlib_blob)

    # Dynamically size Decimal precision & n_ctx so any transcript length works cleanly with zero underflow
    est_max_bits = max(2000, len(raw_bytes) * 4)
    prec = max(2500, int(est_max_bits * 0.36) + 500)
    getcontext().prec = prec
    n_ctx = args.n_ctx if args.n_ctx > 0 else (2048 if len(raw_bytes) < 3500 else (4096 if len(raw_bytes) < 7500 else 8192))
    print(f"Input: {len(raw_bytes)} B ({len(full_text)} chars) | zlib -9: {zlib_bytes} B | auto n_ctx={n_ctx}, decimal prec={prec}", flush=True)

    t0 = time.perf_counter()
    with LLMCompressor(model_path, n_ctx=n_ctx, n_gpu_layers=args.gpu_layers, fast_cdf=False, streams=1) as comp:
        print(f"Loaded {model_path.name} in {time.perf_counter() - t0:.1f}s", flush=True)
        tokens = comp.tokenize(raw_bytes)
        print(f"Tokenized into {len(tokens)} BPE tokens. Running verified single-pass compress + trace...", flush=True)

        piece_bytes_list = [comp.detokenize_bytes([tid]) for tid in tokens]
        assert b"".join(piece_bytes_list) == raw_bytes
        spans = assign_token_times(piece_bytes_list, raw_bytes, byte_meta)

        total_freq = 1 << DEFAULT_CDF_BITS
        comp._ensure_context(required_streams=1, total_n_ctx=comp.n_ctx)
        comp._prime_bos_streams(1)
        histories: list[list[int]] = [[]]
        stream_pos = 1

        bits_trace: list[int] = []

        class TrackBitOut(BitOutputStream):
            def write_bit(self, bit: int) -> None:
                bits_trace.append(bit & 1)
                super().write_bit(bit)

        bit_out = TrackBitOut()
        enc = ArithmeticEncoder(bit_out)
        label_cache: dict[int, str] = {}

        L_prev, H_prev = Decimal(0), Decimal(1)
        B0_prev, B1_prev = Decimal(0), Decimal(1)
        zlib_prev_bits = 0
        entropy_bits = 0.0
        step_entries: list[dict] = []
        E_int = 0
        t_enc = time.perf_counter()

        for i, tid in enumerate(tokens):
            logits = comp._logits_view(0)
            cdf = logits_to_cdf(logits, total_freq=total_freq)
            u0 = int(cdf[tid]) / total_freq
            u1 = int(cdf[tid + 1]) / total_freq
            prob = (int(cdf[tid + 1]) - int(cdf[tid])) / total_freq
            bits = -math.log2(prob)
            entropy_bits += bits

            top1_id = int(np.argmax(logits))
            top1_prob = (int(cdf[top1_id + 1]) - int(cdf[top1_id])) / total_freq
            model_rank = int(np.sum(logits > logits[tid]))
            top1_piece = comp.token_piece_display(top1_id)

            segs, chosen_seg_idx = select_row_segments(comp, cdf, tid, total_freq, label_cache)

            bit_start = len(bits_trace)
            enc.encode_symbol(tid, cdf)
            bit_end = len(bits_trace)
            for b_val in bits_trace[bit_start:bit_end]:
                E_int = (E_int << 1) | b_val

            zlib_bit_start = zlib_prev_bits
            prefix_bytes = raw_bytes[: spans[i]["byteEnd"]]
            zlib_bit_end = max(zlib_bit_start, len(zlib.compress(prefix_bytes, level=9)) * 8)
            zlib_prev_bits = zlib_bit_end

            m = bit_end
            p = enc.pending_bits
            denom = Decimal(1 << (m + p)) * Decimal(FULL_RANGE)
            e3_off = ((1 << p) - 1) * (FULL_RANGE >> 1) if p > 0 else 0
            base = (E_int << p) * FULL_RANGE + e3_off
            L_cur = Decimal(base + enc.low) / denom
            H_cur = Decimal(base + enc.high + 1) / denom

            if m > 0:
                dy_denom = Decimal(1 << m)
                B0 = Decimal(E_int) / dy_denom
                B1 = Decimal(E_int + 1) / dy_denom
            else:
                B0, B1 = Decimal(0), Decimal(1)

            span_cur = H_cur - L_cur
            bin_rel = [float((B0 - L_cur) / span_cur), float((B1 - L_cur) / span_cur)]
            bin_rel_prev = [float((B0_prev - L_cur) / span_cur), float((B1_prev - L_cur) / span_cur)]
            axis_data = decimal_prefix_and_ticks(L_prev, H_prev, L_cur, H_cur)

            pb = piece_bytes_list[i]
            step_entries.append(
                {
                    "i": i,
                    "id": tid,
                    "piece": pb.decode("utf-8", errors="replace"),
                    "pieceBytes": len(pb),
                    "disp": format_token_disp(comp.token_piece_display(tid)),
                    "lineIdx": spans[i]["lineIdx"],
                    "wordIdx": spans[i]["wordIdx"],
                    "t0": round(spans[i]["t0"], 4),
                    "t1": round(spans[i]["t1"], 4),
                    "prob": round(prob, 8),
                    "bits": round(bits, 4),
                    "rank": model_rank,
                    "top1": format_token_disp(top1_piece),
                    "top1Prob": round(top1_prob, 6),
                    "u0": u0,
                    "u1": u1,
                    "segs": segs,
                    "chosenSeg": chosen_seg_idx,
                    "bitStart": bit_start,
                    "bitEnd": bit_end,
                    "zlibBitStart": zlib_bit_start,
                    "zlibBitEnd": zlib_bit_end,
                    "pending": p,
                    "binRel": bin_rel,
                    "binRelPrev": bin_rel_prev,
                    "axis": axis_data,
                }
            )

            L_prev, H_prev = L_cur, H_cur
            B0_prev, B1_prev = B0, B1
            if i + 1 < len(tokens):
                stream_pos = comp._advance_streams([tid], histories, 1, stream_pos, comp.n_ctx)
            if (i + 1) % 250 == 0 or i + 1 == len(tokens):
                print(f"  encoded {i + 1:4d}/{len(tokens)} tokens | bits={bit_end} | elapsed={time.perf_counter() - t_enc:.1f}s", flush=True)

        payload = enc.finish()
        payload_bits = "".join(f"{b:08b}" for b in payload)
        header = HEADER_V2_STRUCT.pack(MAGIC, FORMAT_VERSION, DEFAULT_CDF_BITS, 0, len(tokens), comp.n_ctx, 1, 160, 512)
        blob = header + payload
        out_llmc.write_bytes(blob)

        print("Verifying 100% lossless round-trip decompression...", flush=True)
        t_dec = time.perf_counter()
        restored, _ = comp.decompress_bytes(blob)
        assert restored == full_text, "Lossless decompression mismatch!"
        sha_orig = hashlib.sha256(raw_bytes).hexdigest()
        assert sha_orig == hashlib.sha256(restored.encode("utf-8")).hexdigest()
        print(f"Verified 100% lossless decode in {time.perf_counter() - t_dec:.1f}s (SHA-256={sha_orig})", flush=True)
        vocab_size = comp.vocab_size

    compressed_bytes = len(blob)
    orig_bytes = len(raw_bytes)
    ratio = orig_bytes / compressed_bytes
    bpb = (compressed_bytes * 8.0) / orig_bytes

    doc = {
        "schemaVersion": 1,
        "summary": {
            "modelName": model_label,
            "modelFile": model_path.name,
            "method": "Exact Softmax CDF + 62-bit Arithmetic Coding",
            "vocabSize": vocab_size,
            "cdfBits": DEFAULT_CDF_BITS,
            "numTokens": len(tokens),
            "originalBytes": orig_bytes,
            "originalChars": len(full_text),
            "compressedBytes": compressed_bytes,
            "headerBytes": len(header),
            "payloadBytes": len(payload),
            "payloadBits": len(payload_bits),
            "zlibBytes": zlib_bytes,
            "ratio": round(ratio, 4),
            "bitsPerByte": round(bpb, 4),
            "bitsPerChar": round((compressed_bytes * 8.0) / len(full_text), 4),
            "entropyBits": round(entropy_bits, 2),
            "exactMatch": True,
            "sha256": sha_orig,
        },
        "payloadBitstring": payload_bits,
        "zlibBitstring": zlib_bitstring,
        "tokens": step_entries,
    }
    out_json.write_text(json.dumps(doc, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    size_mb = out_json.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_llmc} ({compressed_bytes} B) and {out_json} ({size_mb:.2f} MB)")
    print(f"Original: {orig_bytes} B | LLM: {compressed_bytes} B ({ratio:.2f}x, {bpb:.2f} bpb) | zlib -9: {zlib_bytes} B ({orig_bytes / zlib_bytes:.2f}x) | LLM vs zlib: {zlib_bytes / compressed_bytes:.2f}x smaller")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

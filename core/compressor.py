"""Lossless LLM-based text compression using llama-cpp-python (GGUF) and Arithmetic Coding.

Implements the compression paradigm popularized by 3Blue1Brown ("Compression is
Intelligence"), accelerated with:
  1. Direct `llama_decode` + pre-allocated `llama_batch` (eliminates per-token
     Python/KV-cache management overhead).
  2. Sparse Threshold/Top-K integer CDF (`logits_to_sparse_cdf`), computing exact
     float64 softmax only on candidate tokens while assigning frequency 1 to tail
     tokens without materializing >250k-element CDF arrays.
  3. Optional multi-stream parallel batching (`streams = B`, `n_seq_max = B`)
     to evaluate B chunks simultaneously in a single GPU forward pass for both
     encoding and decoding.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field, replace
import gc
import hashlib
import math
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import time
from typing import Any, Callable, Sequence
import zlib

import numpy as np

from arithmetic_coder import (
    ArithmeticDecoder,
    ArithmeticEncoder,
    BitInputStream,
    BitOutputStream,
    DEFAULT_CDF_BITS,
    DEFAULT_SPARSE_DELTA,
    DEFAULT_SPARSE_TOP_K,
    SparseCDF,
    logits_to_cdf,
    logits_to_sparse_cdf,
)


MAGIC: bytes = b"LLMC"
FORMAT_VERSION_V1: int = 1
FORMAT_VERSION: int = 2

# V1 Header layout (16 bytes, little-endian):
#   magic (4s), version (B=1), cdf_bits (B), flags (H), num_tokens (I), ctx_window (I)
HEADER_V1_STRUCT = struct.Struct("<4sBBHII")

# V2 Header layout (22 bytes, little-endian):
#   magic (4s), version (B=2), cdf_bits (B), mode_flags (H), num_tokens (I),
#   ctx_window (I), streams (H), sparse_delta_tenths (H), sparse_top_k (H)
HEADER_V2_STRUCT = struct.Struct("<4sBBHIIHHH")
HEADER_STRUCT = HEADER_V2_STRUCT

MODE_FLAG_SPARSE_CDF: int = 1 << 0
MAX_INLINE_BITS: int = 10_000


def bytes_to_bitstring(data: bytes, *, max_bits: int = MAX_INLINE_BITS) -> str:
    """Convert binary bytes to an MSB-first `'0'/'1'` bitstring up to `max_bits`."""
    total_bits = len(data) * 8
    if total_bits > max_bits:
        raise ValueError(
            f"Binary size ({total_bits} bits) exceeds the {max_bits}-bit inline limit; "
            f"use binary file export instead."
        )
    return "".join(f"{byte:08b}" for byte in data)


def bitstring_to_bytes(bitstring: str, *, max_bits: int = MAX_INLINE_BITS) -> bytes:
    """Parse an MSB-first `'0'/'1'` bitstring (ignoring whitespace) back into bytes."""
    cleaned = re.sub(r"\s+", "", bitstring)
    if not cleaned:
        raise ValueError("Bitstring cannot be empty.")
    if len(cleaned) > max_bits:
        raise ValueError(
            f"Bitstring length ({len(cleaned):,} bits) exceeds the {max_bits:,}-bit limit; "
            f"please upload a .llmc file instead."
        )
    if any(ch not in ("0", "1") for ch in cleaned):
        raise ValueError("Bitstring must contain only '0' and '1' characters (and optional whitespace).")
    if len(cleaned) % 8 != 0:
        raise ValueError(
            f"Bitstring length ({len(cleaned)} bits) must be a multiple of 8 (full .llmc bit output)."
        )
    return bytes(int(cleaned[i : i + 8], 2) for i in range(0, len(cleaned), 8))


def bytes_to_hex(data: bytes, *, max_bits: int = MAX_INLINE_BITS) -> str:
    """Convert binary bytes to a lowercase hex string up to `max_bits`."""
    total_bits = len(data) * 8
    if total_bits > max_bits:
        raise ValueError(
            f"Binary size ({total_bits} bits) exceeds the {max_bits}-bit inline limit; "
            f"use binary file export instead."
        )
    return data.hex()


def hex_to_bytes(hexstring: str, *, max_bits: int = MAX_INLINE_BITS) -> bytes:
    """Parse a hex string (ignoring whitespace and optional 0x prefix) into bytes."""
    cleaned = re.sub(r"\s+", "", hexstring)
    if cleaned.lower().startswith("0x"):
        cleaned = cleaned[2:]
    if not cleaned:
        raise ValueError("Hex string cannot be empty.")
    if len(cleaned) * 4 > max_bits:
        raise ValueError(
            f"Hex payload size ({len(cleaned) * 4:,} bits) exceeds the {max_bits:,}-bit limit; "
            f"please upload a .llmc file instead."
        )
    if len(cleaned) % 2 != 0:
        raise ValueError(
            f"Hex string length ({len(cleaned)} chars) must be even (2 hex characters per byte)."
        )
    try:
        return bytes.fromhex(cleaned)
    except ValueError as exc:
        raise ValueError(
            "Hex string must contain only 0-9 and a-f characters (and optional whitespace)."
        ) from exc


def _ensure_cuda_dlls_on_path() -> None:
    """Ensure CUDA 12 runtime DLLs are discoverable on Windows if installed locally."""
    if sys.platform != "win32":
        return

    current_path = os.environ.get("PATH", "")
    for entry in current_path.split(os.pathsep):
        if entry and (Path(entry) / "cudart64_12.dll").is_file():
            return

    candidate_dirs: list[Path] = []

    for site_dir in sys.path:
        if not site_dir:
            continue
        nvidia_dir = Path(site_dir) / "nvidia"
        if nvidia_dir.is_dir():
            for bin_dir in nvidia_dir.glob("*/bin"):
                if bin_dir.is_dir():
                    candidate_dirs.append(bin_dir)

    cuda_root = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if cuda_root.is_dir():
        for ver_dir in sorted(cuda_root.glob("v12.*"), reverse=True):
            bin_dir = ver_dir / "bin"
            if bin_dir.is_dir():
                candidate_dirs.append(bin_dir)

    lmstudio_vendor = Path.home() / ".lmstudio" / "extensions" / "backends" / "vendor"
    if lmstudio_vendor.is_dir():
        for vendor_dir in sorted(lmstudio_vendor.glob("win-llama-cuda12*"), reverse=True):
            if vendor_dir.is_dir():
                candidate_dirs.append(vendor_dir)

    added: list[str] = []
    for directory in candidate_dirs:
        if (directory / "cudart64_12.dll").is_file() or (directory / "cublas64_12.dll").is_file():
            dir_str = str(directory)
            if dir_str not in current_path:
                added.append(dir_str)
                try:
                    os.add_dll_directory(dir_str)
                except OSError:
                    pass

    if added:
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + current_path


_ensure_cuda_dlls_on_path()
import llama_cpp  # noqa: E402
from llama_cpp import _internals as llama_internals  # noqa: E402
from llama_cpp._utils import suppress_stdout_stderr  # noqa: E402


@dataclass(frozen=True)
class TokenTraceEntry:
    """Detailed per-token probability, rank, and bit-expenditure recorded during encoding."""

    index: int
    token_id: int
    piece: str
    probability: float
    bits: float
    piece_bytes: int = 0
    piece_hex: str = ""
    raw_bits: int = 0
    emitted_bits: int = 0
    pending_bits: int = 0
    cumulative_bits: float = 0.0
    stream_index: int = 0
    stream_step: int = 0
    model_rank: int = 0
    top1_token_id: int = 0
    top1_piece: str = ""
    top1_probability: float = 0.0


@dataclass(frozen=True)
class CompressionStats:
    """Metrics describing a single lossless compression run."""

    model_name: str
    original_bytes: int
    compressed_bytes: int
    header_bytes: int
    payload_bytes: int
    num_tokens: int
    vocab_size: int
    cdf_bits: int
    bos_token_id: int
    theoretical_entropy_bits: float
    elapsed_seconds: float
    streams_used: int = 1
    fast_cdf: bool = True
    token_trace: tuple[TokenTraceEntry, ...] = field(default=(), repr=False)

    @property
    def compression_ratio(self) -> float:
        return (self.original_bytes / self.compressed_bytes) if self.compressed_bytes > 0 else 0.0

    @property
    def bits_per_byte(self) -> float:
        return (self.compressed_bytes * 8.0 / self.original_bytes) if self.original_bytes > 0 else 0.0

    @property
    def bits_per_token(self) -> float:
        return (self.payload_bytes * 8.0 / self.num_tokens) if self.num_tokens > 0 else 0.0

    @property
    def space_savings_percent(self) -> float:
        if self.original_bytes == 0:
            return 0.0
        return (1.0 - self.compressed_bytes / self.original_bytes) * 100.0

    @property
    def coder_efficiency_percent(self) -> float:
        if self.payload_bytes == 0 or self.theoretical_entropy_bits <= 0.0:
            return 100.0
        return min(100.0, (self.theoretical_entropy_bits / (self.payload_bytes * 8.0)) * 100.0)


@dataclass(frozen=True)
class DecompressionStats:
    """Metrics describing a single lossless decompression run."""

    model_name: str
    compressed_bytes: int
    restored_bytes: int
    num_tokens: int
    cdf_bits: int
    elapsed_seconds: float
    streams_used: int = 1
    fast_cdf: bool = True


@dataclass(frozen=True)
class ModelRunResult:
    """Full compress + binary file write/read + decompress result for one model."""

    label: str
    model_path: str
    model_name: str
    model_size_gb: float
    n_ctx: int
    n_gpu_layers: int
    load_seconds: float
    compression: CompressionStats
    decompression: DecompressionStats
    restored_text: str
    exact_match: bool
    original_sha256: str
    restored_sha256: str
    header_hex: str
    binary_path: str | None = None


@dataclass(frozen=True)
class ComparisonResult:
    """Side-by-side comparison between Model A and Model B on the same text."""

    original_text: str
    original_bytes: int
    original_chars: int
    zlib_bytes: int
    model_a: ModelRunResult
    model_b: ModelRunResult

    @property
    def winner_by_size(self) -> str:
        a_size = self.model_a.compression.compressed_bytes
        b_size = self.model_b.compression.compressed_bytes
        if a_size < b_size:
            return "A"
        if b_size < a_size:
            return "B"
        return "TIE"

    @property
    def byte_difference(self) -> int:
        return self.model_b.compression.compressed_bytes - self.model_a.compression.compressed_bytes

    @property
    def ratio_difference(self) -> float:
        return self.model_a.compression.compression_ratio - self.model_b.compression.compression_ratio

    @property
    def entropy_difference_bits(self) -> float:
        return (
            self.model_b.compression.theoretical_entropy_bits
            - self.model_a.compression.theoretical_entropy_bits
        )


_SYSTEM_SCAN_SKIP_DIRS = frozenset(
    {
        "windows",
        "program files",
        "program files (x86)",
        "programdata",
        "$recycle.bin",
        "system volume information",
        "recovery",
        "perflogs",
        "msocache",
        "node_modules",
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".venv",
        "venv",
        ".tox",
        ".pytest_cache",
        ".mypy_cache",
        ".npm",
        ".cargo",
        ".rustup",
        ".gradle",
        ".nuget",
        "temp",
        "tmp",
        "proc",
        "sys",
        "dev",
        "run",
        "boot",
        "sbin",
        "bin",
        "lib",
        "lib64",
        "system",
        "applications",
    }
)


def _env_model_roots() -> list[Path]:
    raw = os.environ.get("LLM_COMPRESSOR_MODELS_DIR", "").strip()
    if not raw:
        return []
    return [Path(part).expanduser() for part in raw.split(os.pathsep) if part.strip()]


def discover_gguf_models(
    search_roots: Sequence[Path] | None = None,
    *,
    skip_system_dirs: bool = False,
    max_depth: int | None = None,
) -> list[dict[str, Any]]:
    """Discover standalone LLM `.gguf` files in the provided roots (excluding mmproj).

    When `search_roots` is None, only directories listed in the optional
    `LLM_COMPRESSOR_MODELS_DIR` environment variable are scanned (no app-specific
    default paths are assumed).
    """
    roots = list(search_roots) if search_roots is not None else _env_model_roots()
    found: list[dict[str, Any]] = []
    seen: set[Path] = set()

    for root in roots:
        try:
            root_resolved = root.expanduser().resolve()
        except OSError:
            continue
        if not root_resolved.is_dir():
            continue
        base_depth = len(root_resolved.parts)
        for dirpath, dirnames, filenames in os.walk(
            root_resolved,
            topdown=True,
            onerror=lambda _err: None,
            followlinks=False,
        ):
            cur_path = Path(dirpath)
            depth = len(cur_path.parts) - base_depth
            if max_depth is not None and depth >= max_depth:
                dirnames[:] = []
            elif skip_system_dirs:
                dirnames[:] = [
                    d for d in dirnames if d.lower() not in _SYSTEM_SCAN_SKIP_DIRS
                ]

            for fname in filenames:
                low = fname.lower()
                if not low.endswith(".gguf") or low.startswith("mmproj"):
                    continue
                candidate = cur_path / fname
                try:
                    resolved = candidate.resolve()
                    if resolved in seen or not resolved.is_file():
                        continue
                    size_bytes = resolved.stat().st_size
                except OSError:
                    continue
                seen.add(resolved)
                found.append(
                    {
                        "name": resolved.name,
                        "path": str(resolved),
                        "size_bytes": size_bytes,
                        "size_gb": round(size_bytes / (1024 ** 3), 2),
                        "parent": resolved.parent.name,
                    }
                )

    found.sort(key=lambda item: (item["size_bytes"], item["name"].lower()))
    return found


def discover_system_gguf_models(*, max_depth: int = 8) -> list[dict[str, Any]]:
    """Scan local user directories and mounted drives for `.gguf` files, skipping OS/system dirs."""
    roots: list[Path] = [Path.home()]
    if sys.platform == "win32":
        for letter in "CDEFGHIJKLMNOPQRSTUVWXYZ":
            drive = Path(f"{letter}:\\")
            if drive.exists():
                roots.append(drive)
    else:
        for extra in ("/mnt", "/media", "/Volumes", "/opt", "/srv", "/var/lib/ollama", "/usr/share/ollama"):
            p = Path(extra)
            if p.is_dir():
                roots.append(p)
    return discover_gguf_models(roots, skip_system_dirs=True, max_depth=max_depth)


def _compute_stream_lengths(num_tokens: int, streams: int) -> list[int]:
    base, rem = divmod(num_tokens, streams)
    return [base + (1 if b < rem else 0) for b in range(streams)]


class LLMCompressor:
    """Lossless text compressor/decompressor backed by any GGUF model via llama-cpp-python."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        cdf_bits: int = DEFAULT_CDF_BITS,
        fast_cdf: bool = True,
        sparse_delta: float = DEFAULT_SPARSE_DELTA,
        sparse_top_k: int = DEFAULT_SPARSE_TOP_K,
        streams: int = 1,
        bos_token_id: int | None = None,
        verbose: bool = False,
    ) -> None:
        self._model: llama_internals.LlamaModel | None = None
        self._ctx: llama_internals.LlamaContext | None = None
        self._batch: llama_cpp.llama_batch | None = None
        self._batch_cap: int = 0
        self._ctx_streams: int = 0
        self._ctx_total_n_ctx: int = 0

        self.model_path: Path = Path(model_path)
        self._requested_n_ctx = n_ctx
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.cdf_bits = cdf_bits
        self.fast_cdf = fast_cdf
        self.sparse_delta = sparse_delta
        self.sparse_top_k = sparse_top_k
        self.streams = max(1, streams)
        self._bos_override = bos_token_id
        self.verbose = verbose

        self.vocab_size: int = 0
        self.bos_token_id: int = 0

        self.load_model(
            model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            bos_token_id=bos_token_id,
            verbose=verbose,
        )

    def __enter__(self) -> LLMCompressor:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _free_context(self) -> None:
        if self._batch is not None:
            llama_cpp.llama_batch_free(self._batch)
            self._batch = None
            self._batch_cap = 0
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None
        self._ctx_streams = 0
        self._ctx_total_n_ctx = 0

    def close(self) -> None:
        """Release the loaded GGUF model and free associated CPU/GPU memory."""
        self._free_context()
        if self._model is not None:
            self._model.close()
            self._model = None
            gc.collect()

    def load_model(
        self,
        model_path: str | Path,
        *,
        n_ctx: int | None = None,
        n_gpu_layers: int | None = None,
        n_threads: int | None = None,
        bos_token_id: int | None = None,
        verbose: bool | None = None,
    ) -> None:
        """Load (or hot-swap) any `.gguf` model from disk."""
        resolved = Path(model_path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"GGUF model file not found: {resolved}")

        if n_ctx is not None:
            self._requested_n_ctx = n_ctx
        if n_gpu_layers is not None:
            self.n_gpu_layers = n_gpu_layers
        if n_threads is not None:
            self.n_threads = n_threads
        if bos_token_id is not None:
            self._bos_override = bos_token_id
        if verbose is not None:
            self.verbose = verbose

        self.close()

        mparams = llama_cpp.llama_model_default_params()
        mparams.n_gpu_layers = 0x7FFFFFFF if self.n_gpu_layers == -1 else self.n_gpu_layers
        mparams.load_mode = llama_cpp.LLAMA_LOAD_MODE_MMAP
        mparams.vocab_only = False

        model = llama_internals.LlamaModel(
            path_model=str(resolved),
            params=mparams,
            verbose=self.verbose,
        )
        self._model = model
        self.model_path = resolved
        train_ctx = model.n_ctx_train()
        self.n_ctx = train_ctx if self._requested_n_ctx == 0 else self._requested_n_ctx
        self.vocab_size = model.n_vocab()
        self.bos_token_id = self._select_bos_token(model, self._bos_override)

        total_freq = 1 << self.cdf_bits
        if total_freq < self.vocab_size:
            self.close()
            raise ValueError(
                f"cdf_bits={self.cdf_bits} (total_freq={total_freq}) is smaller than "
                f"model vocab_size={self.vocab_size}. Increase cdf_bits."
            )

        # Pre-create context for the default single-stream case (upgraded lazily if streams > 1).
        self._ensure_context(required_streams=1, total_n_ctx=self.n_ctx)

    def _ensure_context(self, *, required_streams: int, total_n_ctx: int) -> None:
        if self._model is None:
            raise RuntimeError("No GGUF model is currently loaded.")
        if (
            self._ctx is not None
            and self._ctx_streams == required_streams
            and (
                (required_streams == 1 and self._ctx_total_n_ctx >= total_n_ctx)
                or self._ctx_total_n_ctx == total_n_ctx
            )
        ):
            return

        self._free_context()

        n_batch = min(total_n_ctx, max(64, required_streams))
        cparams = llama_cpp.llama_context_default_params()
        cparams.n_ctx = total_n_ctx
        cparams.n_batch = n_batch
        cparams.n_ubatch = n_batch
        cparams.n_seq_max = required_streams
        cparams.flash_attn_type = llama_cpp.LLAMA_FLASH_ATTN_TYPE_AUTO
        if self.n_threads is not None:
            cparams.n_threads = self.n_threads
            cparams.n_threads_batch = self.n_threads

        with suppress_stdout_stderr(disable=self.verbose):
            self._ctx = llama_internals.LlamaContext(
                model=self._model,
                params=cparams,
                verbose=self.verbose,
            )
        self._batch_cap = n_batch
        self._batch = llama_cpp.llama_batch_init(n_batch, 0, 1)
        self._ctx_streams = required_streams
        self._ctx_total_n_ctx = total_n_ctx

    @staticmethod
    def _select_bos_token(model: llama_internals.LlamaModel, override: int | None) -> int:
        if override is not None:
            return override
        vocab = model.vocab
        bos = model.token_bos()
        if bos >= 0 and (
            model.add_bos_token() or bool(llama_cpp.llama_vocab_is_control(vocab, bos))
        ):
            return bos
        eos = model.token_eos()
        if eos >= 0:
            return eos
        if bos >= 0:
            return bos
        return 0

    @property
    def model(self) -> llama_internals.LlamaModel:
        if self._model is None:
            raise RuntimeError("No GGUF model is currently loaded.")
        return self._model

    @property
    def ctx(self) -> llama_internals.LlamaContext:
        if self._ctx is None:
            raise RuntimeError("Context is not initialized.")
        return self._ctx

    def tokenize(self, text: str | bytes) -> list[int]:
        """Tokenize UTF-8 text into model token IDs without prepending BOS."""
        raw = text.encode("utf-8") if isinstance(text, str) else text
        return self.model.tokenize(raw, add_bos=False, special=False)

    def detokenize_bytes(self, token_ids: Sequence[int]) -> bytes:
        """Convert token IDs back into exact UTF-8 bytes via llama_detokenize."""
        if not token_ids:
            return b""
        n_tokens = len(token_ids)
        arr = (llama_cpp.llama_token * n_tokens)(*token_ids)
        cap = max(64, n_tokens * 16)
        buf = ctypes.create_string_buffer(cap)
        written = llama_cpp.llama_detokenize(
            self.model.vocab,
            arr,
            n_tokens,
            buf,
            cap,
            False,
            False,
        )
        if written < 0:
            cap = -written + 1
            buf = ctypes.create_string_buffer(cap)
            written = llama_cpp.llama_detokenize(
                self.model.vocab,
                arr,
                n_tokens,
                buf,
                cap,
                False,
                False,
            )
        if written < 0:
            raise RuntimeError(f"llama_detokenize failed with code {written}")
        return buf.raw[:written]

    def detokenize(self, token_ids: Sequence[int]) -> str:
        """Convert token IDs back into a UTF-8 string."""
        return self.detokenize_bytes(token_ids).decode("utf-8")

    def token_piece_display(self, token_id: int) -> str:
        """Return a safe display string for a single token piece."""
        return self.detokenize_bytes([token_id]).decode("utf-8", errors="replace")

    def _compute_total_ctx(self, *, num_tokens: int, streams: int, ctx_window: int) -> int:
        if streams == 1:
            return ctx_window
        max_chunk = (num_tokens + streams - 1) // streams
        per_stream = max(64, min(ctx_window, ((max_chunk + 32 + 31) // 32) * 32))
        return streams * per_stream

    def _decode_batch_tokens(
        self,
        tokens: Sequence[int],
        pos: int,
        count: int,
    ) -> None:
        batch = self._batch
        assert batch is not None
        batch.n_tokens = count
        for b in range(count):
            batch.token[b] = tokens[b]
            batch.pos[b] = pos
            batch.n_seq_id[b] = 1
            batch.seq_id[b][0] = b
            batch.logits[b] = True
        rc = llama_cpp.llama_decode(self.ctx.ctx, batch)
        if rc != 0:
            raise RuntimeError(f"llama_decode failed with code {rc}")

    def _prime_bos_streams(self, active_count: int) -> int:
        self.ctx.kv_cache_clear()
        self._decode_batch_tokens([self.bos_token_id] * active_count, pos=0, count=active_count)
        return 1

    def _advance_streams(
        self,
        step_tokens: Sequence[int],
        histories: list[list[int]],
        next_active_count: int,
        stream_pos: int,
        ctx_window: int,
    ) -> int:
        for b in range(next_active_count):
            histories[b].append(step_tokens[b])

        if stream_pos < ctx_window:
            self._decode_batch_tokens(step_tokens, pos=stream_pos, count=next_active_count)
            return stream_pos + 1

        # Deterministic context-window rollover across all active streams.
        keep = max(0, (ctx_window // 2) - 1)
        if keep == 0:
            return self._prime_bos_streams(next_active_count)

        self.ctx.kv_cache_clear()
        batch = self._batch
        assert batch is not None
        cap = self._batch_cap
        filled = 0

        for p in range(keep):
            for b in range(next_active_count):
                tok_p = self.bos_token_id if p == 0 else histories[b][-keep + p - 1]
                batch.token[filled] = tok_p
                batch.pos[filled] = p
                batch.n_seq_id[filled] = 1
                batch.seq_id[filled][0] = b
                batch.logits[filled] = False
                filled += 1
                if filled == cap:
                    batch.n_tokens = filled
                    rc = llama_cpp.llama_decode(self.ctx.ctx, batch)
                    if rc != 0:
                        raise RuntimeError(f"llama_decode rollover prefill failed: {rc}")
                    filled = 0

        if filled > 0:
            batch.n_tokens = filled
            rc = llama_cpp.llama_decode(self.ctx.ctx, batch)
            if rc != 0:
                raise RuntimeError(f"llama_decode rollover prefill tail failed: {rc}")

        last_tokens = [histories[b][-1] for b in range(next_active_count)]
        self._decode_batch_tokens(last_tokens, pos=keep, count=next_active_count)
        return keep + 1

    def _logits_view(self, batch_index: int) -> np.ndarray:
        ptr = llama_cpp.llama_get_logits_ith(self.ctx.ctx, batch_index)
        return np.ctypeslib.as_array(ptr, shape=(self.vocab_size,))

    def _build_cdf(
        self,
        logits: np.ndarray,
        *,
        total_freq: int,
        use_sparse: bool,
        sparse_delta: float,
        sparse_top_k: int,
    ) -> np.ndarray | SparseCDF:
        if use_sparse:
            return logits_to_sparse_cdf(
                logits,
                total_freq=total_freq,
                logit_delta=sparse_delta,
                max_top_k=sparse_top_k,
            )
        return logits_to_cdf(logits, total_freq=total_freq)

    def compress_bytes(
        self,
        text: str,
        *,
        streams: int | None = None,
        fast_cdf: bool | None = None,
        sparse_delta: float | None = None,
        sparse_top_k: int | None = None,
        collect_trace: bool = False,
        max_trace_tokens: int | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> tuple[bytes, CompressionStats]:
        """Losslessly compress `text` and return `(binary_blob, stats)`."""
        t0 = time.perf_counter()
        raw_bytes = text.encode("utf-8")
        token_ids = self.tokenize(raw_bytes)

        if self.detokenize_bytes(token_ids) != raw_bytes:
            raise ValueError(
                f"Tokenizer for '{self.model_path.name}' is not byte-exact reversible on this input."
            )

        num_tokens = len(token_ids)
        req_streams = max(1, self.streams if streams is None else streams)
        use_sparse = self.fast_cdf if fast_cdf is None else fast_cdf
        delta = self.sparse_delta if sparse_delta is None else sparse_delta
        top_k = self.sparse_top_k if sparse_top_k is None else sparse_top_k

        actual_streams = (
            1
            if num_tokens == 0 or req_streams == 1
            else min(req_streams, max(1, num_tokens // 16))
        )

        ctx_window = self.n_ctx
        total_freq = 1 << self.cdf_bits
        mode_flags = MODE_FLAG_SPARSE_CDF if use_sparse else 0
        delta_tenths = max(1, min(65535, int(round(delta * 10.0))))
        clamped_top_k = max(1, min(65535, int(top_k)))

        header = HEADER_V2_STRUCT.pack(
            MAGIC,
            FORMAT_VERSION,
            self.cdf_bits,
            mode_flags,
            num_tokens,
            ctx_window,
            actual_streams,
            delta_tenths,
            clamped_top_k,
        )

        if num_tokens == 0:
            elapsed = time.perf_counter() - t0
            return header, CompressionStats(
                model_name=self.model_path.name,
                original_bytes=0,
                compressed_bytes=len(header),
                header_bytes=len(header),
                payload_bytes=0,
                num_tokens=0,
                vocab_size=self.vocab_size,
                cdf_bits=self.cdf_bits,
                bos_token_id=self.bos_token_id,
                theoretical_entropy_bits=0.0,
                elapsed_seconds=elapsed,
                streams_used=1,
                fast_cdf=use_sparse,
            )

        total_n_ctx = self._compute_stream_lengths_ctx(
            num_tokens=num_tokens,
            streams=actual_streams,
            ctx_window=ctx_window,
        )
        self._ensure_context(required_streams=actual_streams, total_n_ctx=total_n_ctx)

        lengths = _compute_stream_lengths(num_tokens, actual_streams)
        offsets: list[int] = []
        cursor = 0
        stream_slices: list[list[int]] = []
        for length in lengths:
            offsets.append(cursor)
            stream_slices.append(token_ids[cursor : cursor + length])
            cursor += length

        max_steps = lengths[0]
        stream_pos = self._prime_bos_streams(actual_streams)
        histories: list[list[int]] = [[] for _ in range(actual_streams)]

        bit_out = BitOutputStream()
        encoder = ArithmeticEncoder(bit_out)
        entropy_bits = 0.0
        trace_cap = num_tokens if max_trace_tokens is None else min(num_tokens, max(0, max_trace_tokens))
        trace_slots: list[TokenTraceEntry | None] = (
            [None] * trace_cap if collect_trace else []
        )
        piece_cache: dict[int, bytes] = {}
        encoded_count = 0

        def _cached_piece_bytes(tid: int) -> bytes:
            cached = piece_cache.get(tid)
            if cached is None:
                cached = self.detokenize_bytes([tid])
                if len(piece_cache) < 8192:
                    piece_cache[tid] = cached
            return cached

        for step in range(max_steps):
            active_count = sum(1 for length in lengths if step < length)
            step_tokens = [stream_slices[b][step] for b in range(active_count)]

            for b in range(active_count):
                token_id = step_tokens[b]
                logits_view = self._logits_view(b)
                cdf = self._build_cdf(
                    logits_view,
                    total_freq=total_freq,
                    use_sparse=use_sparse,
                    sparse_delta=delta_tenths / 10.0,
                    sparse_top_k=clamped_top_k,
                )
                freq = (
                    cdf.frequency(token_id)
                    if isinstance(cdf, SparseCDF)
                    else (int(cdf[token_id + 1]) - int(cdf[token_id]))
                )
                prob = freq / total_freq
                bits = -math.log2(prob)
                entropy_bits += bits

                global_idx = offsets[b] + step
                record_this = collect_trace and global_idx < trace_cap

                if record_this:
                    top1_id = int(np.argmax(logits_view))
                    top1_freq = (
                        cdf.frequency(top1_id)
                        if isinstance(cdf, SparseCDF)
                        else (int(cdf[top1_id + 1]) - int(cdf[top1_id]))
                    )
                    top1_prob = top1_freq / total_freq
                    model_rank = int(np.sum(logits_view > logits_view[token_id]))
                    piece_raw = _cached_piece_bytes(token_id)
                    top1_raw = _cached_piece_bytes(top1_id)
                    bits_before = bit_out.total_bits_written

                encoder.encode_symbol(token_id, cdf)

                if record_this:
                    emitted_bits = bit_out.total_bits_written - bits_before
                    trace_slots[global_idx] = TokenTraceEntry(
                        index=global_idx,
                        token_id=token_id,
                        piece=piece_raw.decode("utf-8", errors="replace"),
                        probability=prob,
                        bits=bits,
                        piece_bytes=len(piece_raw),
                        piece_hex=piece_raw.hex(),
                        raw_bits=len(piece_raw) * 8,
                        emitted_bits=emitted_bits,
                        pending_bits=encoder.pending_bits,
                        cumulative_bits=0.0,
                        stream_index=b,
                        stream_step=step,
                        model_rank=model_rank,
                        top1_token_id=top1_id,
                        top1_piece=top1_raw.decode("utf-8", errors="replace"),
                        top1_probability=top1_prob,
                    )

                encoded_count += 1
                if progress_callback is not None:
                    progress_callback("encode", encoded_count, num_tokens)

            if step + 1 < max_steps:
                next_active = sum(1 for length in lengths if step + 1 < length)
                stream_pos = self._advance_streams(
                    step_tokens,
                    histories,
                    next_active,
                    stream_pos,
                    ctx_window,
                )

        payload = encoder.finish()
        blob = header + payload
        elapsed = time.perf_counter() - t0
        if collect_trace:
            running = 0.0
            ordered_trace: list[TokenTraceEntry] = []
            for entry in trace_slots:
                if entry is not None:
                    running += entry.bits
                    ordered_trace.append(replace(entry, cumulative_bits=running))
            trace = tuple(ordered_trace)
        else:
            trace = ()

        stats = CompressionStats(
            model_name=self.model_path.name,
            original_bytes=len(raw_bytes),
            compressed_bytes=len(blob),
            header_bytes=len(header),
            payload_bytes=len(payload),
            num_tokens=num_tokens,
            vocab_size=self.vocab_size,
            cdf_bits=self.cdf_bits,
            bos_token_id=self.bos_token_id,
            theoretical_entropy_bits=entropy_bits,
            elapsed_seconds=elapsed,
            streams_used=actual_streams,
            fast_cdf=use_sparse,
            token_trace=trace,
        )
        return blob, stats

    def _compute_stream_lengths_ctx(self, *, num_tokens: int, streams: int, ctx_window: int) -> int:
        return self._compute_total_ctx(
            num_tokens=num_tokens,
            streams=streams,
            ctx_window=ctx_window,
        )

    def compress(self, text: str, **kwargs: Any) -> bytes:
        """Losslessly compress `text` into binary bytes."""
        blob, _ = self.compress_bytes(text, **kwargs)
        return blob

    def decompress_bytes(
        self,
        compressed_data: bytes,
        *,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> tuple[str, DecompressionStats]:
        """Decompress binary bytes (v1 or v2) back into `(text, stats)`."""
        t0 = time.perf_counter()
        if len(compressed_data) < HEADER_V1_STRUCT.size:
            raise ValueError(
                f"Compressed data too short ({len(compressed_data)} bytes); expected >= {HEADER_V1_STRUCT.size}."
            )

        magic, version = compressed_data[:4], compressed_data[4]
        if magic != MAGIC:
            raise ValueError(f"Invalid file magic {magic!r}; expected {MAGIC!r}.")

        if version == FORMAT_VERSION_V1:
            _, _, cdf_bits, flags, num_tokens, ctx_window = HEADER_V1_STRUCT.unpack_from(
                compressed_data, 0
            )
            if flags != 0:
                raise ValueError(f"Unsupported v1 header flags {flags}.")
            use_sparse = False
            streams = 1
            sparse_delta = DEFAULT_SPARSE_DELTA
            sparse_top_k = DEFAULT_SPARSE_TOP_K
            header_size = HEADER_V1_STRUCT.size
        elif version == FORMAT_VERSION:
            if len(compressed_data) < HEADER_V2_STRUCT.size:
                raise ValueError(
                    f"Compressed data too short ({len(compressed_data)} bytes); expected >= {HEADER_V2_STRUCT.size}."
                )
            (
                _,
                _,
                cdf_bits,
                mode_flags,
                num_tokens,
                ctx_window,
                streams,
                delta_tenths,
                sparse_top_k,
            ) = HEADER_V2_STRUCT.unpack_from(compressed_data, 0)
            if mode_flags & ~MODE_FLAG_SPARSE_CDF:
                raise ValueError(f"Unsupported header flags {mode_flags}.")
            use_sparse = bool(mode_flags & MODE_FLAG_SPARSE_CDF)
            sparse_delta = delta_tenths / 10.0
            streams = max(1, streams)
            header_size = HEADER_V2_STRUCT.size
        else:
            raise ValueError(f"Unsupported format version {version}.")

        total_freq = 1 << cdf_bits
        if total_freq < self.vocab_size:
            raise ValueError(
                f"Header cdf_bits={cdf_bits} (total_freq={total_freq}) < vocab_size={self.vocab_size}."
            )
        if num_tokens >= ctx_window and self.n_ctx < ctx_window:
            raise ValueError(
                f"Stream was encoded with ctx_window={ctx_window}, but decoder model was loaded "
                f"with smaller n_ctx={self.n_ctx}. Reload with n_ctx>={ctx_window}."
            )

        if num_tokens == 0:
            elapsed = time.perf_counter() - t0
            return "", DecompressionStats(
                model_name=self.model_path.name,
                compressed_bytes=len(compressed_data),
                restored_bytes=0,
                num_tokens=0,
                cdf_bits=cdf_bits,
                elapsed_seconds=elapsed,
                streams_used=streams,
                fast_cdf=use_sparse,
            )

        total_n_ctx = self._compute_total_ctx(
            num_tokens=num_tokens,
            streams=streams,
            ctx_window=ctx_window,
        )
        self._ensure_context(required_streams=streams, total_n_ctx=total_n_ctx)

        lengths = _compute_stream_lengths(num_tokens, streams)
        max_steps = lengths[0]
        stream_pos = self._prime_bos_streams(streams)
        histories: list[list[int]] = [[] for _ in range(streams)]
        decoded_streams: list[list[int]] = [[] for _ in range(streams)]

        payload = memoryview(compressed_data)[header_size:]
        decoder = ArithmeticDecoder(BitInputStream(payload))
        decoded_count = 0

        for step in range(max_steps):
            active_count = sum(1 for length in lengths if step < length)
            step_tokens: list[int] = []

            for b in range(active_count):
                logits_view = self._logits_view(b)
                cdf = self._build_cdf(
                    logits_view,
                    total_freq=total_freq,
                    use_sparse=use_sparse,
                    sparse_delta=sparse_delta,
                    sparse_top_k=sparse_top_k,
                )
                token_id = decoder.decode_symbol(cdf)
                decoded_streams[b].append(token_id)
                step_tokens.append(token_id)

                decoded_count += 1
                if progress_callback is not None:
                    progress_callback("decode", decoded_count, num_tokens)

            if step + 1 < max_steps:
                next_active = sum(1 for length in lengths if step + 1 < length)
                stream_pos = self._advance_streams(
                    step_tokens,
                    histories,
                    next_active,
                    stream_pos,
                    ctx_window,
                )

        decoded_tokens = [tok for chunk in decoded_streams for tok in chunk]
        restored_bytes = self.detokenize_bytes(decoded_tokens)
        restored_text = restored_bytes.decode("utf-8")
        elapsed = time.perf_counter() - t0

        stats = DecompressionStats(
            model_name=self.model_path.name,
            compressed_bytes=len(compressed_data),
            restored_bytes=len(restored_bytes),
            num_tokens=num_tokens,
            cdf_bits=cdf_bits,
            elapsed_seconds=elapsed,
            streams_used=streams,
            fast_cdf=use_sparse,
        )
        return restored_text, stats

    def decompress(self, compressed_data: bytes) -> str:
        """Decompress binary bytes back into the exact original UTF-8 text."""
        text, _ = self.decompress_bytes(compressed_data)
        return text

    def compress_to_file(
        self,
        text: str,
        output_path: str | Path,
        *,
        streams: int | None = None,
        fast_cdf: bool | None = None,
        sparse_delta: float | None = None,
        sparse_top_k: int | None = None,
        collect_trace: bool = False,
        max_trace_tokens: int | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> CompressionStats:
        """Compress `text` and write the resulting binary stream to `output_path`."""
        blob, stats = self.compress_bytes(
            text,
            streams=streams,
            fast_cdf=fast_cdf,
            sparse_delta=sparse_delta,
            sparse_top_k=sparse_top_k,
            collect_trace=collect_trace,
            max_trace_tokens=max_trace_tokens,
            progress_callback=progress_callback,
        )
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blob)
        return stats

    def decompress_from_file(
        self,
        input_path: str | Path,
        *,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> tuple[str, DecompressionStats]:
        """Read binary data from `input_path` and decompress it back to original text."""
        blob = Path(input_path).read_bytes()
        return self.decompress_bytes(blob, progress_callback=progress_callback)

    def compress_to_bitstring(
        self,
        text: str,
        *,
        max_bits: int = MAX_INLINE_BITS,
        **kwargs: Any,
    ) -> tuple[str, CompressionStats]:
        """Compress `text` into an MSB-first `'0'/'1'` bitstring (enforcing `max_bits`)."""
        blob, stats = self.compress_bytes(text, **kwargs)
        return bytes_to_bitstring(blob, max_bits=max_bits), stats

    def decompress_bitstring(
        self,
        bitstring: str,
        *,
        max_bits: int = MAX_INLINE_BITS,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> tuple[str, DecompressionStats]:
        """Decompress an inline `'0'/'1'` bitstring (enforcing `max_bits`) back to text."""
        blob = bitstring_to_bytes(bitstring, max_bits=max_bits)
        return self.decompress_bytes(blob, progress_callback=progress_callback)

    def compress_to_hex(
        self,
        text: str,
        *,
        max_bits: int = MAX_INLINE_BITS,
        **kwargs: Any,
    ) -> tuple[str, CompressionStats]:
        """Compress `text` into a hex string (enforcing `max_bits`)."""
        blob, stats = self.compress_bytes(text, **kwargs)
        return bytes_to_hex(blob, max_bits=max_bits), stats

    def decompress_hex(
        self,
        hexstring: str,
        *,
        max_bits: int = MAX_INLINE_BITS,
        progress_callback: Callable[[str, int, int], None] | None = None,
    ) -> tuple[str, DecompressionStats]:
        """Decompress an inline hex string (enforcing `max_bits`) back to text."""
        blob = hex_to_bytes(hexstring, max_bits=max_bits)
        return self.decompress_bytes(blob, progress_callback=progress_callback)


def _run_single_model_roundtrip(
    *,
    label: str,
    model_path: str | Path,
    text: str,
    binary_path: Path,
    n_ctx: int,
    n_gpu_layers: int,
    n_threads: int | None,
    cdf_bits: int,
    streams: int,
    fast_cdf: bool,
    collect_trace: bool,
    event_callback: Callable[[dict[str, Any]], None] | None,
) -> ModelRunResult:
    resolved = Path(model_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Model {label} not found: {resolved}")

    original_bytes = text.encode("utf-8")
    original_sha256 = hashlib.sha256(original_bytes).hexdigest()

    if event_callback is not None:
        event_callback(
            {
                "type": "status",
                "model": label,
                "stage": "loading",
                "message": f"Loading Model {label}: {resolved.name}",
            }
        )

    t_load0 = time.perf_counter()
    with LLMCompressor(
        resolved,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        cdf_bits=cdf_bits,
        fast_cdf=fast_cdf,
        streams=streams,
    ) as comp:
        load_seconds = time.perf_counter() - t_load0

        def on_progress(phase: str, current: int, total: int) -> None:
            if event_callback is not None:
                event_callback(
                    {
                        "type": "progress",
                        "model": label,
                        "phase": phase,
                        "current": current,
                        "total": total,
                    }
                )

        if event_callback is not None:
            event_callback(
                {
                    "type": "status",
                    "model": label,
                    "stage": "encode",
                    "message": f"Encoding with Model {label}...",
                }
            )
        comp_stats = comp.compress_to_file(
            text,
            binary_path,
            collect_trace=collect_trace,
            progress_callback=on_progress,
        )

        if event_callback is not None:
            event_callback(
                {
                    "type": "status",
                    "model": label,
                    "stage": "decode",
                    "message": f"Decoding Model {label} from binary file...",
                }
            )
        restored_text, dec_stats = comp.decompress_from_file(
            binary_path,
            progress_callback=on_progress,
        )
        actual_n_ctx = comp.n_ctx

    if event_callback is not None:
        event_callback(
            {
                "type": "status",
                "model": label,
                "stage": "unloaded",
                "message": f"Model {label} finished and unloaded from memory.",
            }
        )

    restored_bytes = restored_text.encode("utf-8")
    restored_sha256 = hashlib.sha256(restored_bytes).hexdigest()
    header_hex = binary_path.read_bytes()[: HEADER_V2_STRUCT.size].hex(" ")

    return ModelRunResult(
        label=label,
        model_path=str(resolved),
        model_name=resolved.name,
        model_size_gb=round(resolved.stat().st_size / (1024 ** 3), 2),
        n_ctx=actual_n_ctx,
        n_gpu_layers=n_gpu_layers,
        load_seconds=load_seconds,
        compression=comp_stats,
        decompression=dec_stats,
        restored_text=restored_text,
        exact_match=(restored_bytes == original_bytes),
        original_sha256=original_sha256,
        restored_sha256=restored_sha256,
        header_hex=header_hex,
        binary_path=str(binary_path),
    )


def compare_models(
    *,
    text: str,
    model_a_path: str | Path,
    model_b_path: str | Path,
    output_dir: str | Path | None = None,
    n_ctx: int = 2048,
    n_gpu_layers_a: int = -1,
    n_gpu_layers_b: int = -1,
    n_threads: int | None = None,
    cdf_bits: int = DEFAULT_CDF_BITS,
    streams: int = 1,
    fast_cdf: bool = True,
    collect_trace: bool = True,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ComparisonResult:
    """Compress & decompress `text` sequentially with Model A then Model B, freeing VRAM between runs."""
    raw_bytes = text.encode("utf-8")
    zlib_bytes = len(zlib.compress(raw_bytes, level=9))

    if output_dir is not None:
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory()
        out_root = Path(temp_ctx.name)

    try:
        bin_a = out_root / "model_a.llmc"
        bin_b = out_root / "model_b.llmc"

        result_a = _run_single_model_roundtrip(
            label="A",
            model_path=model_a_path,
            text=text,
            binary_path=bin_a,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers_a,
            n_threads=n_threads,
            cdf_bits=cdf_bits,
            streams=streams,
            fast_cdf=fast_cdf,
            collect_trace=collect_trace,
            event_callback=event_callback,
        )
        if event_callback is not None:
            event_callback({"type": "model_done", "model": "A", "result": result_a})

        result_b = _run_single_model_roundtrip(
            label="B",
            model_path=model_b_path,
            text=text,
            binary_path=bin_b,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers_b,
            n_threads=n_threads,
            cdf_bits=cdf_bits,
            streams=streams,
            fast_cdf=fast_cdf,
            collect_trace=collect_trace,
            event_callback=event_callback,
        )
        if event_callback is not None:
            event_callback({"type": "model_done", "model": "B", "result": result_b})

        return ComparisonResult(
            original_text=text,
            original_bytes=len(raw_bytes),
            original_chars=len(text),
            zlib_bytes=zlib_bytes,
            model_a=result_a,
            model_b=result_b,
        )
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


def token_trace_to_dicts(trace: Sequence[TokenTraceEntry]) -> list[dict[str, Any]]:
    """Convert a sequence of `TokenTraceEntry` items into detailed JSON-serializable dicts."""
    items: list[dict[str, Any]] = []
    for entry in trace:
        bpb = (entry.bits / entry.piece_bytes) if entry.piece_bytes > 0 else 0.0
        items.append(
            {
                "index": entry.index,
                "stream_index": entry.stream_index,
                "stream_step": entry.stream_step,
                "token_id": entry.token_id,
                "piece": entry.piece,
                "piece_hex": entry.piece_hex,
                "piece_bytes": entry.piece_bytes,
                "raw_bits": entry.raw_bits,
                "probability": round(entry.probability, 8),
                "bits": round(entry.bits, 6),
                "bits_per_byte": round(bpb, 6),
                "emitted_bits": entry.emitted_bits,
                "pending_bits": entry.pending_bits,
                "cumulative_bits": round(entry.cumulative_bits, 6),
                "model_rank": entry.model_rank,
                "is_top1": entry.model_rank == 0,
                "top1_token_id": entry.top1_token_id,
                "top1_piece": entry.top1_piece,
                "top1_probability": round(entry.top1_probability, 8),
            }
        )
    return items


def model_run_to_json_dict(run: ModelRunResult) -> dict[str, Any]:
    """Serialize one model's complete evaluation + token-by-token breakdown into JSON."""
    c = run.compression
    d = run.decompression
    top1_hits = sum(1 for entry in c.token_trace if entry.model_rank == 0)
    trace_count = len(c.token_trace)
    top1_rate = (top1_hits / trace_count * 100.0) if trace_count > 0 else 0.0

    return {
        "label": run.label,
        "model_name": run.model_name,
        "model_path": run.model_path,
        "model_size_gb": run.model_size_gb,
        "n_ctx": run.n_ctx,
        "n_gpu_layers": run.n_gpu_layers,
        "load_seconds": round(run.load_seconds, 3),
        "exact_match": run.exact_match,
        "original_sha256": run.original_sha256,
        "restored_sha256": run.restored_sha256,
        "header_hex": run.header_hex,
        "summary": {
            "original_bytes": c.original_bytes,
            "compressed_bytes": c.compressed_bytes,
            "compressed_bits": c.compressed_bytes * 8,
            "header_bytes": c.header_bytes,
            "payload_bytes": c.payload_bytes,
            "payload_bits": c.payload_bytes * 8,
            "num_tokens": c.num_tokens,
            "vocab_size": c.vocab_size,
            "cdf_bits": c.cdf_bits,
            "bos_token_id": c.bos_token_id,
            "streams_used": c.streams_used,
            "fast_cdf": c.fast_cdf,
            "compression_ratio": round(c.compression_ratio, 4),
            "space_savings_percent": round(c.space_savings_percent, 2),
            "bits_per_byte": round(c.bits_per_byte, 4),
            "bits_per_token": round(c.bits_per_token, 4),
            "theoretical_entropy_bits": round(c.theoretical_entropy_bits, 4),
            "theoretical_entropy_bytes": round(c.theoretical_entropy_bits / 8.0, 2),
            "coder_efficiency_percent": round(c.coder_efficiency_percent, 2),
            "encode_seconds": round(c.elapsed_seconds, 3),
            "decode_seconds": round(d.elapsed_seconds, 3),
            "top1_hit_count": top1_hits,
            "top1_hit_rate_percent": round(top1_rate, 2),
        },
        "tokens": token_trace_to_dicts(c.token_trace),
    }


def comparison_to_json_dict(res: ComparisonResult) -> dict[str, Any]:
    """Serialize a two-model comparison with separate per-model token-by-token details."""
    return {
        "schema_version": 1,
        "original_text": res.original_text,
        "original_bytes": res.original_bytes,
        "original_chars": res.original_chars,
        "zlib_bytes": res.zlib_bytes,
        "zlib_ratio": round(res.original_bytes / res.zlib_bytes, 4) if res.zlib_bytes > 0 else 0.0,
        "comparison": {
            "winner_by_size": res.winner_by_size,
            "byte_difference_b_minus_a": res.byte_difference,
            "ratio_difference_a_minus_b": round(res.ratio_difference, 4),
            "entropy_difference_b_minus_a_bits": round(res.entropy_difference_bits, 4),
        },
        "model_a": model_run_to_json_dict(res.model_a),
        "model_b": model_run_to_json_dict(res.model_b),
    }


def token_trace_to_sparse_dicts(trace: Sequence[TokenTraceEntry]) -> list[dict[str, Any]]:
    """Serialize only the 5 most essential, interpretable per-token fields."""
    return [
        {
            "piece": entry.piece,
            "prob": round(entry.probability, 4),
            "bits": round(entry.bits, 3),
            "emitted_bits": entry.emitted_bits,
            "rank": entry.model_rank,
        }
        for entry in trace
    ]


def _top_surprises_from_trace(trace: Sequence[TokenTraceEntry], limit: int = 10) -> list[dict[str, Any]]:
    """Extract the highest-surprise tokens so large runs can be interpreted at a glance."""
    if not trace or limit <= 0:
        return []
    ranked = sorted(trace, key=lambda e: e.bits, reverse=True)[:limit]
    return [
        {
            "index": e.index,
            "piece": e.piece,
            "bits": round(e.bits, 3),
            "prob": round(e.probability, 4),
            "rank": e.model_rank,
            "top1": e.top1_piece,
        }
        for e in ranked
    ]


def build_sparse_model_payload(
    *,
    comp: CompressionStats,
    dec: DecompressionStats | None = None,
    model_name: str | None = None,
    label: str | None = None,
    exact_match: bool | None = None,
    top_surprises_limit: int = 10,
) -> dict[str, Any]:
    """Build a compact sparse JSON dict retaining headline metrics, top surprises, and 5-key tokens."""
    top1_hits = sum(1 for entry in comp.token_trace if entry.model_rank == 0)
    trace_count = len(comp.token_trace)
    top1_rate = (top1_hits / trace_count * 100.0) if trace_count > 0 else 0.0

    payload: dict[str, Any] = {
        "schema_version": 1,
        "detail": "sparse",
    }
    if label is not None:
        payload["label"] = label
    payload["model"] = model_name or comp.model_name
    if exact_match is not None:
        payload["exact_match"] = exact_match

    summary: dict[str, Any] = {
        "original_bytes": comp.original_bytes,
        "compressed_bytes": comp.compressed_bytes,
        "ratio": round(comp.compression_ratio, 4),
        "bits_per_byte": round(comp.bits_per_byte, 4),
        "num_tokens": comp.num_tokens,
        "entropy_bits": round(comp.theoretical_entropy_bits, 2),
        "efficiency_pct": round(comp.coder_efficiency_percent, 2),
        "top1_hit_rate_pct": round(top1_rate, 2),
        "encode_s": round(comp.elapsed_seconds, 3),
    }
    if dec is not None:
        summary["decode_s"] = round(dec.elapsed_seconds, 3)

    payload["summary"] = summary
    payload["top_surprises"] = _top_surprises_from_trace(comp.token_trace, limit=top_surprises_limit)
    payload["tokens"] = token_trace_to_sparse_dicts(comp.token_trace)
    return payload


def model_run_to_sparse_json_dict(run: ModelRunResult) -> dict[str, Any]:
    """Serialize one model's essential evaluation + 5-field sparse per-token breakdown."""
    return build_sparse_model_payload(
        comp=run.compression,
        dec=run.decompression,
        model_name=run.model_name,
        label=run.label,
        exact_match=run.exact_match,
    )


def comparison_to_sparse_json_dict(res: ComparisonResult) -> dict[str, Any]:
    """Serialize a compact comparison report alongside separate sparse Model A / Model B data."""
    return {
        "schema_version": 1,
        "detail": "sparse",
        "original_bytes": res.original_bytes,
        "original_chars": res.original_chars,
        "zlib_bytes": res.zlib_bytes,
        "zlib_ratio": round(res.original_bytes / res.zlib_bytes, 4) if res.zlib_bytes > 0 else 0.0,
        "comparison": {
            "winner_by_size": res.winner_by_size,
            "byte_difference_b_minus_a": res.byte_difference,
            "ratio_difference_a_minus_b": round(res.ratio_difference, 4),
            "entropy_difference_b_minus_a_bits": round(res.entropy_difference_bits, 4),
        },
        "model_a": model_run_to_sparse_json_dict(res.model_a),
        "model_b": model_run_to_sparse_json_dict(res.model_b),
    }

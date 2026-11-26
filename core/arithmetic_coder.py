"""Pure-Python / NumPy 62-bit Arithmetic Coding and Integer CDF conversion.

Implements Witten-Neal-Cleary integer arithmetic coding with E3 underflow scaling
and deterministic softmax-to-integer-CDF quantization guaranteed to assign a
strictly positive frequency (>= 1) to every token in the vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

import numpy as np


# 62-bit arithmetic coding state:
# - Keeps all intermediate products `(range * freq)` cleanly inside Python ints
#   without bit-width overflow and accommodates large LLM vocabularies (> 250k)
#   with high precision (e.g. total_freq = 1 << 28 << QUARTER = 1 << 60).
STATE_BITS: int = 62
FULL_RANGE: int = 1 << STATE_BITS
HALF: int = FULL_RANGE >> 1
QUARTER: int = FULL_RANGE >> 2
THREE_QUARTERS: int = 3 * QUARTER
STATE_MASK: int = FULL_RANGE - 1

DEFAULT_CDF_BITS: int = 28
DEFAULT_SPARSE_DELTA: float = 18.0
DEFAULT_SPARSE_TOP_K: int = 2048


def _quantize_probs(
    probs: np.ndarray,
    total_mass: int,
    distributable: int,
) -> np.ndarray:
    """Quantize float64 probabilities into >=1 int64 counts summing to `total_mass`."""
    n = int(probs.shape[0])
    scaled = probs * float(distributable)
    floored = np.floor(scaled).astype(np.int64)
    freqs = floored + 1

    remainder = total_mass - int(np.sum(freqs))
    if remainder > 0:
        frac = scaled - floored
        if remainder >= n:
            q, remainder = divmod(remainder, n)
            freqs += q
        if remainder > 0:
            top_indices = np.argpartition(-frac, remainder - 1)[:remainder]
            freqs[top_indices] += 1
    elif remainder < 0:
        deficit = -remainder
        while deficit > 0:
            candidates = np.flatnonzero(freqs > 1)
            if candidates.size == 0:
                raise RuntimeError("Cannot adjust CDF frequencies below 1.")
            take = min(deficit, int(candidates.size))
            if take < candidates.size:
                chosen = candidates[np.argpartition(-freqs[candidates], take - 1)[:take]]
            else:
                chosen = candidates
            freqs[chosen] -= 1
            deficit -= take

    return freqs


@dataclass(frozen=True)
class SparseCDF:
    """O(K) cumulative frequency table keeping exact softmax on Top-K tokens and 1 for tail."""

    vocab_size: int
    total_freq: int
    top_ids: np.ndarray
    top_freqs: np.ndarray
    prefix_top: np.ndarray
    cum_start: np.ndarray
    cum_end: np.ndarray

    @property
    def k(self) -> int:
        return int(self.top_ids.shape[0])

    def interval(self, symbol: int) -> tuple[int, int]:
        """Return `[cum_low, cum_high)` for `symbol` in O(log K) without materializing vocab CDF."""
        if not (0 <= symbol < self.vocab_size):
            raise ValueError(f"Symbol {symbol} out of range [0, {self.vocab_size}).")
        r = int(np.searchsorted(self.top_ids, symbol, side="left"))
        if r < self.k and int(self.top_ids[r]) == symbol:
            return int(self.cum_start[r]), int(self.cum_end[r])
        cum_low = (symbol - r) + int(self.prefix_top[r])
        return cum_low, cum_low + 1

    def frequency(self, symbol: int) -> int:
        lo, hi = self.interval(symbol)
        return hi - lo

    def find_symbol(self, scaled_value: int) -> tuple[int, int, int]:
        """Find `(symbol, cum_low, cum_high)` for `0 <= scaled_value < total_freq` in O(log K)."""
        if not (0 <= scaled_value < self.total_freq):
            raise RuntimeError(
                f"scaled_value={scaled_value} out of range [0, {self.total_freq})."
            )
        r_cand = int(np.searchsorted(self.cum_start, scaled_value, side="right")) - 1
        if r_cand >= 0 and scaled_value < int(self.cum_end[r_cand]):
            return (
                int(self.top_ids[r_cand]),
                int(self.cum_start[r_cand]),
                int(self.cum_end[r_cand]),
            )
        r = r_cand + 1
        tail_idx = scaled_value - int(self.prefix_top[r])
        symbol = tail_idx + r
        return symbol, scaled_value, scaled_value + 1


def logits_to_sparse_cdf(
    logits: np.ndarray,
    total_freq: int = 1 << DEFAULT_CDF_BITS,
    *,
    logit_delta: float = DEFAULT_SPARSE_DELTA,
    max_top_k: int = DEFAULT_SPARSE_TOP_K,
) -> SparseCDF | np.ndarray:
    """Build an O(K) SparseCDF keeping tokens within `logit_delta` of max (capped at `max_top_k`).

    All remaining tail tokens receive exact frequency 1 so every token stays codable
    while avoiding float64 exp/cumsum over >250k vocabulary entries.
    """
    logits_arr = np.asarray(logits)
    if logits_arr.ndim != 1:
        raise ValueError(f"Expected 1D logits array, got shape {logits_arr.shape}")

    vocab_size = int(logits_arr.shape[0])
    if vocab_size <= 0:
        raise ValueError("Vocabulary size must be positive.")
    if total_freq < vocab_size:
        raise ValueError(
            f"total_freq ({total_freq}) must be >= vocab_size ({vocab_size})."
        )
    if total_freq > QUARTER:
        raise ValueError(f"total_freq ({total_freq}) must be <= QUARTER ({QUARTER}).")

    if vocab_size <= max_top_k:
        return logits_to_cdf(logits_arr, total_freq=total_freq)

    max_logit = float(np.max(logits_arr))
    if np.isnan(max_logit):
        finite_mask = np.isfinite(logits_arr)
        if not np.any(finite_mask):
            return logits_to_cdf(logits_arr, total_freq=total_freq)
        max_logit = float(np.max(logits_arr[finite_mask]))
    if not np.isfinite(max_logit):
        return logits_to_cdf(logits_arr, total_freq=total_freq)

    threshold = max_logit - logit_delta
    idx = np.flatnonzero(logits_arr >= threshold).astype(np.int64)
    if idx.size == 0:
        return logits_to_cdf(logits_arr, total_freq=total_freq)

    if idx.size > max_top_k:
        top_local = np.argpartition(logits_arr[idx], -max_top_k)[-max_top_k:]
        idx = idx[top_local]
    idx.sort()

    k = int(idx.size)
    top_logits = logits_arr[idx].astype(np.float64) - max_logit
    exp_vals = np.exp(top_logits)
    exp_sum = float(np.sum(exp_vals))
    if not np.isfinite(exp_sum) or exp_sum <= 0.0:
        return logits_to_cdf(logits_arr, total_freq=total_freq)

    probs = exp_vals / exp_sum
    top_total = total_freq - (vocab_size - k)
    distributable = total_freq - vocab_size
    top_freqs = _quantize_probs(probs, total_mass=top_total, distributable=distributable)

    prefix_top = np.empty(k + 1, dtype=np.int64)
    prefix_top[0] = 0
    np.cumsum(top_freqs, out=prefix_top[1:])
    cum_start = (idx - np.arange(k, dtype=np.int64)) + prefix_top[:-1]
    cum_end = cum_start + top_freqs

    return SparseCDF(
        vocab_size=vocab_size,
        total_freq=total_freq,
        top_ids=idx,
        top_freqs=top_freqs,
        prefix_top=prefix_top,
        cum_start=cum_start,
        cum_end=cum_end,
    )


def logits_to_cdf(logits: np.ndarray, total_freq: int = 1 << DEFAULT_CDF_BITS) -> np.ndarray:
    """Convert raw model logits into a strictly monotonic dense integer CDF array.

    Every token in the vocabulary is guaranteed a frequency >= 1 so that no
    symbol interval ever collapses to zero width during arithmetic coding.

    Args:
        logits: 1D array of raw unnormalized logits with shape (vocab_size,).
        total_freq: Fixed integer sum of all token frequencies. Must be at least
            `vocab_size` and at most `QUARTER` (`1 << 60`).

    Returns:
        1D int64 array `cdf` of length `vocab_size + 1` where `cdf[0] == 0`,
        `cdf[-1] == total_freq`, and `cdf[i + 1] > cdf[i]` for all `i`.
    """
    logits_f64 = np.asarray(logits, dtype=np.float64)
    if logits_f64.ndim != 1:
        raise ValueError(f"Expected 1D logits array, got shape {logits_f64.shape}")

    vocab_size = logits_f64.shape[0]
    if vocab_size <= 0:
        raise ValueError("Vocabulary size must be positive.")
    if total_freq < vocab_size:
        raise ValueError(
            f"total_freq ({total_freq}) must be >= vocab_size ({vocab_size}) "
            f"so every token gets frequency >= 1. Use a larger cdf_bits (e.g. 24 or 28)."
        )
    if total_freq > QUARTER:
        raise ValueError(f"total_freq ({total_freq}) must be <= QUARTER ({QUARTER}).")

    # Numerically stable Softmax in float64.
    finite_mask = np.isfinite(logits_f64)
    if not np.any(finite_mask):
        probs = np.full(vocab_size, 1.0 / vocab_size, dtype=np.float64)
    else:
        max_logit = np.max(logits_f64[finite_mask])
        shifted = np.where(np.isnan(logits_f64), -np.inf, logits_f64 - max_logit)
        exp_vals = np.exp(shifted)
        exp_sum = float(np.sum(exp_vals))
        if not np.isfinite(exp_sum) or exp_sum <= 0.0:
            probs = np.full(vocab_size, 1.0 / vocab_size, dtype=np.float64)
        else:
            probs = exp_vals / exp_sum

    # Reserve frequency 1 for every token; distribute remaining mass via softmax.
    distributable = total_freq - vocab_size
    freqs = _quantize_probs(probs, total_mass=total_freq, distributable=distributable)

    cdf = np.empty(vocab_size + 1, dtype=np.int64)
    cdf[0] = 0
    np.cumsum(freqs, out=cdf[1:])
    return cdf


class BitOutputStream:
    """Writes individual bits MSB-first into a byte buffer or binary stream."""

    def __init__(self, stream: BinaryIO | None = None) -> None:
        self._stream = stream
        self._buffer = bytearray()
        self._current_byte = 0
        self._num_bits_filled = 0
        self.total_bits_written = 0

    def write_bit(self, bit: int) -> None:
        self._current_byte = (self._current_byte << 1) | (bit & 1)
        self._num_bits_filled += 1
        self.total_bits_written += 1
        if self._num_bits_filled == 8:
            self._buffer.append(self._current_byte)
            self._current_byte = 0
            self._num_bits_filled = 0
            if self._stream is not None and len(self._buffer) >= 4096:
                self._stream.write(self._buffer)
                self._buffer.clear()

    def flush(self) -> bytes:
        """Pad the trailing partial byte with zero bits and return buffered bytes."""
        if self._num_bits_filled > 0:
            self._current_byte <<= 8 - self._num_bits_filled
            self._buffer.append(self._current_byte)
            self._current_byte = 0
            self._num_bits_filled = 0
        if self._stream is not None:
            if self._buffer:
                self._stream.write(self._buffer)
                self._buffer.clear()
            self._stream.flush()
            return b""
        return bytes(self._buffer)


class BitInputStream:
    """Reads individual bits MSB-first from bytes or a binary stream."""

    def __init__(self, data: bytes | bytearray | memoryview | BinaryIO) -> None:
        if isinstance(data, (bytes, bytearray, memoryview)):
            self._data = memoryview(data)
            self._stream: BinaryIO | None = None
            self._pos = 0
        else:
            self._data = memoryview(b"")
            self._stream = data
            self._pos = 0
        self._current_byte = 0
        self._bits_remaining = 0

    def read_bit(self) -> int:
        if self._bits_remaining == 0:
            if self._stream is not None:
                chunk = self._stream.read(1)
                if not chunk:
                    return 0
                self._current_byte = chunk[0]
            else:
                if self._pos >= len(self._data):
                    return 0
                self._current_byte = self._data[self._pos]
                self._pos += 1
            self._bits_remaining = 8
        self._bits_remaining -= 1
        return (self._current_byte >> self._bits_remaining) & 1


@dataclass
class ArithmeticEncoder:
    """62-bit integer Arithmetic Encoder over dynamic integer CDFs."""

    output: BitOutputStream
    low: int = 0
    high: int = STATE_MASK
    pending_bits: int = 0

    def encode_symbol(self, symbol: int, cdf: np.ndarray | SparseCDF) -> None:
        """Encode one symbol using its cumulative frequency interval `[low, high)`."""
        if isinstance(cdf, SparseCDF):
            cum_low, cum_high = cdf.interval(symbol)
            total_freq = cdf.total_freq
        else:
            cum_low = int(cdf[symbol])
            cum_high = int(cdf[symbol + 1])
            total_freq = int(cdf[-1])

        if not (0 <= cum_low < cum_high <= total_freq):
            raise ValueError(
                f"Invalid CDF interval for symbol {symbol}: "
                f"[{cum_low}, {cum_high}) out of {total_freq}"
            )

        current_range = self.high - self.low + 1
        new_high = self.low + (current_range * cum_high) // total_freq - 1
        new_low = self.low + (current_range * cum_low) // total_freq
        self.low = new_low
        self.high = new_high

        while True:
            if self.high < HALF:
                self._emit_bit_plus_pending(0)
            elif self.low >= HALF:
                self._emit_bit_plus_pending(1)
            elif self.low >= QUARTER and self.high < THREE_QUARTERS:
                self.pending_bits += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low = (self.low << 1) & STATE_MASK
            self.high = ((self.high << 1) | 1) & STATE_MASK

    def _emit_bit_plus_pending(self, bit: int) -> None:
        self.output.write_bit(bit)
        opposite = 1 - bit
        while self.pending_bits > 0:
            self.output.write_bit(opposite)
            self.pending_bits -= 1

    def finish(self) -> bytes:
        """Emit disambiguating tail bits and flush the underlying bit stream."""
        self.pending_bits += 1
        if self.low < QUARTER:
            self._emit_bit_plus_pending(0)
        else:
            self._emit_bit_plus_pending(1)
        return self.output.flush()


class ArithmeticDecoder:
    """62-bit integer Arithmetic Decoder over dynamic integer CDFs."""

    def __init__(self, input_stream: BitInputStream) -> None:
        self.input = input_stream
        self.low = 0
        self.high = STATE_MASK
        self.code = 0
        for _ in range(STATE_BITS):
            self.code = ((self.code << 1) | self.input.read_bit()) & STATE_MASK

    def decode_symbol(self, cdf: np.ndarray | SparseCDF) -> int:
        """Decode the next symbol given the exact same CDF used during encoding."""
        total_freq = cdf.total_freq if isinstance(cdf, SparseCDF) else int(cdf[-1])
        current_range = self.high - self.low + 1
        offset = self.code - self.low
        scaled_value = ((offset + 1) * total_freq - 1) // current_range

        if isinstance(cdf, SparseCDF):
            symbol, cum_low, cum_high = cdf.find_symbol(scaled_value)
        else:
            symbol = int(np.searchsorted(cdf, scaled_value, side="right")) - 1
            if symbol < 0 or symbol >= len(cdf) - 1:
                raise RuntimeError(
                    f"Decoded out-of-range symbol {symbol} for scaled_value={scaled_value}."
                )
            cum_low = int(cdf[symbol])
            cum_high = int(cdf[symbol + 1])

        self.high = self.low + (current_range * cum_high) // total_freq - 1
        self.low = self.low + (current_range * cum_low) // total_freq

        while True:
            if self.high < HALF:
                pass
            elif self.low >= HALF:
                self.code -= HALF
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high < THREE_QUARTERS:
                self.code -= QUARTER
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low = (self.low << 1) & STATE_MASK
            self.high = ((self.high << 1) | 1) & STATE_MASK
            self.code = ((self.code << 1) | self.input.read_bit()) & STATE_MASK

        return symbol

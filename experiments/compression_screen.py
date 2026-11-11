#!/usr/bin/env python3
"""FHERMA lossless compression screening: CPU proxy only, NOT official CUDA time.
Synthetic random W-bit coefficients in the required little-endian uint32 layout.
"""
import bz2, lzma, random, statistics, time, zlib
SEED = 2048
SAMPLE_COEFFICIENTS = 4096
CASES = ((65536, 1747), (32768, 868))

def sample(n, w):
    rng = random.Random(SEED + w)
    limbs = (w + 31) // 32
    high_mask = (1 << (w - 32 * (limbs - 1))) - 1
    out = bytearray()
    for _ in range(min(n, SAMPLE_COEFFICIENTS)):
        for _ in range(limbs - 1):
            out.extend(rng.getrandbits(32).to_bytes(4, "little"))
        out.extend((rng.getrandbits(32) & high_mask).to_bytes(4, "little"))
    return bytes(out)

def measure(fn, data):
    times = []
    out = None
    for _ in range(3):
        start = time.perf_counter()
        out = fn(data)
        times.append((time.perf_counter() - start) * 1000)
    return len(out), statistics.median(times)

def main():
    for n, w in CASES:
        data = sample(n, w)
        padded = ((w + 31) // 32) * 32
        print(f"N={n} W={w} sample_bytes={len(data)} spare_bits={padded-w}")
        algorithms = (
            ("zlib-1", lambda d: zlib.compress(d, 1)),
            ("zlib-6", lambda d: zlib.compress(d, 6)),
            ("bz2-1", lambda d: bz2.compress(d, compresslevel=1)),
            ("lzma-0", lambda d: lzma.compress(d, preset=0)),
        )
        for label, compressor in algorithms:
            size, millis = measure(compressor, data)
            print(f"{label:8s} bytes={size} ratio={len(data)/size:.4f} CPU_encode_ms={millis:.2f}")
        print(f"Bitpacking theoretical size fraction: {w/padded:.5f}")
if __name__ == "__main__":
    main()

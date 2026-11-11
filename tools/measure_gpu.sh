#!/usr/bin/env bash
# FHERMA benchmark + profiler; requires CUDA GPU and an existing test-case directory.
# No submission and no modifications to the competition entry.
set -euo pipefail
if [ "$#" -ne 1 ]; then echo "Usage: bash tools/measure_gpu.sh CASE_DIRECTORY"; exit 2; fi
CASE_DIR="$(realpath "$1")"
test -d "$CASE_DIR" || { echo "Case directory missing"; exit 2; }
command -v nvidia-smi >/dev/null || { echo "No NVIDIA GPU detected"; exit 3; }
command -v nvcc >/dev/null || { echo "No nvcc detected"; exit 3; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/measure-results"
mkdir -p "$OUT"
date -u -Iseconds | tee "$OUT/utc.txt"
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | tee "$OUT/hardware.txt"
nvcc --version | tee "$OUT/compiler.txt"
ARCH="sm_120"
for PROFILE in OFF ON; do
  if [ "$PROFILE" = OFF ]; then LABEL=normal; else LABEL=diagnostic; fi
  cmake -S "$ROOT" -B "$OUT/build-$LABEL" -DCMAKE_BUILD_TYPE=Release -DFHERMA_CUDA_ARCH="$ARCH" -DFHERMA_PROFILE="$PROFILE" -DFHERMA_FUSED_REINDEX=OFF -DFHERMA_CRT_TILE_X=16
  cmake --build "$OUT/build-$LABEL" -j 2
  "$OUT/build-$LABEL/solution" "$CASE_DIR" > "$OUT/$LABEL.stdout" 2> "$OUT/$LABEL.stderr"
done
if command -v nsys >/dev/null; then
  nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt -o "$OUT/timeline" "$OUT/build-normal/solution" "$CASE_DIR" > "$OUT/nsys.stdout" 2> "$OUT/nsys.stderr"
  nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum "$OUT/timeline.nsys-rep" > "$OUT/nsys-summary.txt" || true
fi
echo "Reports in $OUT"
echo "Diagnostic timings include instrumentation overhead: do not treat them as official wall-clock ranking."

# CUDA GPU Optimization Experiment — November 10, 2026

Status: **Experimental; no CUDA compiler or NVIDIA GPU available on the connected computer. NOT benchmarked or submitted.**

## Current experiment
- `FHERMA_FUSED_REINDEX=ON`: replace the separate global-memory bit reversal of input coefficients with a gather inside the encode kernel. The original behavior remains selectable and default OFF.
- `FHERMA_PROFILE=OFF`: compile out profiling events and the explicit upload synchronization used for developmental profiling; output `cudaMemcpy` still supplies the required completion boundary.
- `FHERMA_NTT_TILED=ON`: preserve the pre-existing shared-memory tiled NTT.

## Why
Avoiding a full input permutation might reduce one kernel launch and global-memory read/write traffic. Conversely, a fused gather may have less-coalesced input accesses, so performance could *worsen*. Neither outcome is established without measurements.

## CPU validation completed
Run `node verify-reindex.js`. It compares the old permutation and fused gather across both published points with *all coefficient limbs* for two test inputs. This checks the permutation only; it does NOT validate the NTT, modular arithmetic, CRT, compilation, or GPU performance.

## Required official GPU validation
1. Build baseline in Release using the identical CUDA version/architecture and the original `main.cpp` benchmark harness.
2. Run official correctness and median wall-clock measurements on both N=65536/W=1747/L=55 and N=32768/W=868/L=28, including randomized test seeds.
3. Build experiment with `-DFHERMA_FUSED_REINDEX=ON -DFHERMA_PROFILE=OFF` and benchmark with the same settings and warmup.
4. Build control with `-DFHERMA_FUSED_REINDEX=OFF -DFHERMA_PROFILE=OFF` so profiler removal and permutation fusion are evaluated separately.
5. If experiment is slower or incorrect, do not submit. Check GPU profiling for transfer times, CRT duration, kernel launches, memory coalescing, occupancy and launch overhead. 
6. Submit only a fully verified variant through the authorized official challenge process.

## Other research targets (not implemented)
- CRT: inspect exact integer accumulation schedule and memory reuse before experimenting with tensor-core decomposition; do not substitute approximate floating point without exactness proof.
- NTT: investigate mixed-radix/four-step transforms and permuted layout, *after* measuring current NTT contribution.
- Transfers: benchmark page-locked memory and asynchronous copies **including pinning overhead** with the official end-to-end contract; don't hide work outside the timer.
- CUDA Graphs: potentially beneficial when CPU launch overhead dominates; requires safe handling of varying host buffers and immutable setup.

References:
- NVIDIA CUDA Best Practices Guide: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- Official challenge rules: https://www.fherma.io/kernels/polynomial-multiplication/challenges/polynomial-multiplication-2025?tab=rules

## Feasibility calculations added
Run `node transfer-feasibility.js` to inspect transfer-volume bounds for both official points. The large point transports two 13.750 MiB input tensors and one 13.750 MiB output tensor, or 41.250 MiB total. At a hypothetical aggregate 64 GB/s, serial transfer of these would consume about 675.8 microseconds before math. Reaching 400 microseconds would need at least 108 GB/s of aggregate I/O absent transfer/computation overlap or other optimizations. This is an **illustrative feasibility estimate, not a tested bound**: GPU transfers may be full-duplex, overlapped, cached, mapped, staged or affected by actual PCIe bottlenecks. Do not infer impossibility from this computation.

### Critical next experiments
- Capture 20 official-like wall-clock cases and Nsight Systems GPU/CPU timeline. Prioritize end-to-end, copy engines, synchronization and CRT percentages.
- Compare baseline, profiling-off baseline and fused-reindex variant independently with full verifier.
- Consider cuPQC NTT at existing 31-bit RNS moduli; NTT library supports custom primes and staged transforms, but porting requires correct Montgomery constants/twiddles and proper negacyclic twists.
- Do **not** submit or label faster unless it passes the exact FHERMA verifier and measured GPU benchmark.

NVIDIA cuPQC API references:
https://docs.nvidia.com/cuda/cupqc/index.html
https://docs.nvidia.com/cuda/cupqc/libraries/cupqc_ntt/cupqc_ntt_usage.html
Official benchmark details:
https://www.fherma.io/docs/submit-a-challenge-solution

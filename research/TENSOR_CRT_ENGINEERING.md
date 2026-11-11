# Experimental exact Tensor-Core CRT (11 November 2026)

**Not yet GPU-correctness verified or performance measured.** This branch starts from the FHERMA-verified original commit `c97ec6621d3e` (40/40 previous oracle cases; 2.69 ms at N=65536,W=1747). The official submitted FHERMA implementation stays pinned to the original.

## New architecture

`solve.cu` includes a switch `FHERMA_TENSOR_CRT`:
* **ON**: new `reconstruct_tensor_kernel`, 128 threads / 4 warps per CTA; each CTA reconstructs 16 coefficients × all limbs.
* **OFF**: the original scalar `reconstruct_kernel` remains unchanged.

The kernel computes canonical CRT terms from RNS prime-major residues, computes the original correction term, then evaluates the `16 coefficients × (K+1) primes` by `(K+1) primes × 16 limbs` matrix product using **unsigned INT8 WMMA** (`mma_sync`). Each input 32-bit word is split into four byte planes, using 16 byte-plane products in seven diagonals, followed by exact base-256 carry propagation. It produces per-column low/high 64-bit accumulators and invokes the original exact `carry_and_reduce` / `reduce_sum`, so the original wide-modulus reduction is retained.

One logical CTA handles the complete set of 55 limbs for 16 coefficients, thereby avoiding a new external intermediate-product buffer. This differs from merely changing CRT tile geometry.

## Proof and verification

`research/exact_tensor_crt_gemm_proof.js` compares scalar u32×u32 products against the 16 INT8 matrix-product decomposition and independently checks the kernel's seven-diagonal base-256 carry reconstruction for adversarial and random test vectors.

GitHub Actions workflows:
* `.github/workflows/exact-tensor-crt-proof.yml` runs the CPU algebraic proof.
* `.github/workflows/tensor-crt-cuda-build.yml` compiles both ON/OFF variants under CUDA 12.8.1, `sm_120` (this runner has **no GPU**). Compilation is not evidence of Blackwell behavior.
* The first full official exactness suite must run in **a separate, unsubmitted FHERMA experiment**.
* Do not submit this version for ranking unless it passes all official sizes and establishes a real whole-call speedup over the original baseline, with repeat measurements.

**Risk**: Matrix tiles generate register pressure and shared-memory usage, and preprocessing each CRT term may outweigh WMMA throughput. The theoretical 428.9M scalar products do not by themselves establish that WMMA will be faster. The current prototype is unoptimized and a correctness-first engineering candidate.

## Future metrics

Once a GPU test runs, compare N=65536 W=1747 *and* N=32768 W=868; require identical bytes, 20/20 or more seeds and time measured by FHERMA on the same Blackwell runner. A legitimate target is first <2.69 ms, then <1.12 ms, then <817 us. Do not report 700 us before it is independently confirmed.

No unrelated changes to user source, pricing or project deployment.

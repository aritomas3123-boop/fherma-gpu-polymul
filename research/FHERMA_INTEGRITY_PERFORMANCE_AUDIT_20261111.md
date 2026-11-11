# FHERMA independent integrity and performance audit — 11 Nov 2026

## Scope and evidence standard

This is a *publicly verifiable preliminary audit*, not a forensic investigation of private competitor code or FHERMA administrator logs. An observed large performance lead is **not evidence of illegal conduct, privileged access or cheating**. No competitor is accused. We have not accessed the leader's private implementation, GPU trace, kernel binaries, or privileged runner logs.

### Primary verifiable sources
- Challenge rules: https://www.fherma.io/kernels/polynomial-multiplication/challenges/polynomial-multiplication-2025?tab=rules
- Challenge standings: https://www.fherma.io/kernels/polynomial-multiplication/challenges/polynomial-multiplication-2025?tab=leaderboard
- One public competitor's 828-us run: https://www.fherma.io/kernels/polynomial-multiplication/specifications/negacyclic/runs/6aeed904ed7d31391ff22869
- Transfer discussion and FHERMA-team clarification: https://www.fherma.io/kernels/polynomial-multiplication/discussions/6ab3d18ca9590da538b9af26
- GPU keep-warm fairness discussion: https://www.fherma.io/kernels/polynomial-multiplication/discussions/6ad2305beb1f4723711fb71d
- Runner measurement pipeline: https://www.fherma.io/docs/measurement
- Our fastest known reference commit: https://github.com/aritomas3123-boop/fherma-gpu-polymul/tree/c97ec6621d3e

## Integrity findings

1. **No public evidence of insider information.** Public standings have multiple competitors between ~0.828 and 0.850 ms. The 828-us shared private-code run shows 40/40 exactness, 20 seeds for each official point, 823–837 us spread on the large point, consistent with a reproducible optimization. The private implementation's source cannot be independently audited here.
2. **Actual fairness issue awaiting clarification**: public entrants discussed significant speed changes when the Blackwell GPU/PCIe link goes idle between cases (~100–300 us reported by entrants). In a response, FHERMA had allowed setup-created GPU keep-warm threads between calls, provided they did not touch per-case inputs; NVIDIA technical representatives subsequently sought specifics. This is a known *benchmark policy/consistency issue*, not proof that the leader used the technique or that it violated the stated rules.
3. **Rules expressly forbid** precomputed outputs, recognizing hidden cases, accessing hidden verifier data, harness alteration, unauthorized external compute, exploiting platform vulnerabilities or falsifying measurements. Source-private entries are allowed and do not automatically indicate misconduct. Parameter-only precomputation in setup is allowed.
4. **Caution about timing changes**: during the challenge, reported leader values have varied (~817 to 828 us across different public/previous views). Per-run noise and background retesting are acknowledged by the organizers; a difference in a displayed median does not, alone, establish a rules issue.
5. FHERMA states pre-award re-runs, >=100 extra seeds, >=1,000-seed finalist checks and committee review. Ask the organizer to document uniform GPU power-state policy and hardware environment across finalists.

## Quantitative architecture audit

Point A: N=65536, W=1747, L=55, each of the two original input tensors is 65536*55*4 = 14,417,920 bytes; result tensor same 14,417,920 bytes. Total mandatory host input/output payload = 43,253,760 bytes. These are new arbitrary operands: they cannot be reconstructed from one trigger bit and near-uniform high-entropy inputs cannot be substantially compressed without loss. Pinning buffers, CUDA Graphs and parameter-only setup do not remove these bytes.

An optimistic raw PCIe 5.0 x16 ceiling of 64 GB/s / direction gives:
- two H2D inputs: 450.56 us for 28,835,840 bytes, ignoring protocol overhead;
- one D2H result: 225.28 us for 14,417,920 bytes;
- **non-overlapped** minimum 675.84 us, leaving 152.16 us to an 828-us solution under this generous theoretical assumption. Real throughput and critical path must be measured; different directional overlap, partial-result streaming and CPU stages mean naive addition of stage medians is invalid.

Our original version `c97ec6621d3e` achieved ~2.69 ms on this large point in a non-challenge FHERMA run (previously observed 40/40). It is not the same measurement as the challenge-specific ~3.08 ms reported separately. In our *verified source code*:
- multi-threaded CPU copy -> persistent pinned buffers -> cudaMemcpyAsync to GPU;
- streams, event synchronization, overlapped A/B operand handling;
- WMMA INT8 Tensor Core **input-to-RNS encoding** (function `ari_wmma_encoder::encode_gemm`);
- tiled shared-memory NTT;
- scalar integer CRT reconstruction in `fherma_crt::reconstruct_kernel` (u32 products and sequential carry/reduction);
- output copied back to CPU.
This invalidates any proposed optimization that simply adds pinned input, stream overlap or Tensor Core *input encoding*: these already exist.

### What is not known

We do not have private top-competitor kernel source or full Nsight profile, target-GPU clocks/link metrics, or reliable *stage-by-stage* 2.69-ms baseline timings. We cannot infer from a single total latency that a competitor is cheating or that CRT is the leading bottleneck. A novel exact Tensor-Core CRT is a falsifiable candidate but not yet compiled/verified on the Blackwell competition runner.

## Highest-value next steps

1. **Integrity/fairness inquiry without accusation**: ask FHERMA for documented policy on between-call GPU keep-warm activity; whether power clocks/link are standardized before each seed; whether all entrants get the same policy; what rerun process handles clock variance. Request transparent per-case context, not other contestants' private code or data.
2. **Collect per-stage evidence** from our instrumented `research-profile-fast-baseline-20261111` branch. Compilation passed in CUDA 12.8 for ON/OFF; one instrumented FHERMA measurement returned 3.24ms / 20 of 20 exact. Its instrumented stderr per-stage logs were not yet retrieved. Until those exist, no calibrated AI autotuner can pick optimal kernel architecture.
3. **Reproduce baseline control** on official runner (same commit, image, exact seeds where permitted), and capture hardware clock state, CPU staging and H2D/D2H bandwidth if runner permits. Do not compare different submission commits as though identical.
4. **Novel algorithm candidates**: exact WMMA Tensor-Core CRT (not merely fused epilogue), different NTT layout with on-chip stage consolidation, or a demonstrably lower-traffic mathematical transform. Compare only with exact full-input/output and both official sizes; discard if correctness fails or end-to-end time worsens.
5. **Do not flood official benchmark** with millions of trial submissions. Use static mathematical constraints + calibrated surrogate + narrow hardware benchmarks. Search space count alone is not evidence of performance.

## Outcome

**No proof of foul play. Legitimate policy/fairness question identified. Structural performance gap identified, exact missing timing evidence documented.** Cannot promise or claim a 700-us / 450-us result from public evidence.

No changes to submitted competition implementation or anyone else's accounts.

# Automated FHERMA design search (research only)

This branch starts from the **actual 2.69-ms tested implementation** (commit c97ec6621d3e), not from the slower 5.7-ms experimental branches. Neither solve.cu nor CMakeLists.txt is changed here.

The script `research/billion_design_screen.js` has a combinatorial search space of `8^10 = 1,073,741,824` architectural parameter tuples. It **samples and evaluates 2,000,000 tuples by default**, applying conservative static resource filters. At its first run on November 11, 2026, 95,736 samples passed those filters. The script can be re-run locally as:

```
node research/billion_design_screen.js 2000000 > shortlist.json
```

**Critical limitation:** This is static search-space screening, not one billion Blackwell GPU simulations, not a kernel implementation, not proof of CUDA compilation, not an exactness test, and **not a speedup in microseconds**. It cannot replace profiling or official FHERMA measurements. Some promising candidate dimensions involve prospective code that is not yet implemented. A low proxy score must NEVER be interpreted as predicted latency.

Recommended engineering sequence:
1. Preserve and retest baseline `c97ec6621d3e` (both exact sizes) on the same runner.
2. Measure CUDA stage timings, especially input H2D, residue encode, forward/inverse NTT, CRT, D2H and all CPU launch/staging.
3. Calibrate a performance model with observed stage timing and hardware counters.
4. Select only physically and mathematically implementable shortlisted architectures, beginning with NTT traffic and exact Tensor-Core CRT.
5. Compile + run independent exactness for both points, benchmark on the official runner; reject any full-call regression.
6. Keep competition submission separate from experiments until better exact results are confirmed.

The platform currently runs a single shared Blackwell machine. Avoid automated request flooding or pretending analytical screening is real GPU benchmarking.

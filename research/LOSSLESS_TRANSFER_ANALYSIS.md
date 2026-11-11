# FHERMA lossless data-volume feasibility

The actual N=65536, W=1747 interface uses **L=55 uint32 limbs per coefficient**, totaling **14,417,920 bytes per operand** and **28,835,840 bytes for two inputs**.

An exactly preserved coefficient needs up to 1747 bits, while the interface occupies 1760 bits: at most 13 zero padding bits per coefficient.

* Byte packing (219 bytes per coefficient, instead of 220): saves 65,536 bytes per operand.
* Ideal bit packing (1747/8=218.375 bytes): saves 106,496 bytes per operand.
* Maximum bit-padding reduction for BOTH input operands is **212,992 bytes** (0.7386%), ignoring CPU packing, GPU unpacking, alignment and control overhead.
* Even at an optimistic 64 GB/s effective PCIe, that saves only **3.328 microseconds** of input transfer in the best case.
* Output packing is similarly constrained because all output bytes must be returned intact.
* With dense approximately uniform modular inputs, conventional lossless entropy compression usually cannot materially reduce data size and may enlarge it.

To reproduce CPU synthetic data experiments: `node research/compression_feasibility.js`.

**Different architecture worth researching:** reduce global GPU memory traffic in per-prime NTT and/or exact CRT, where multiple intermediate arrays (N x K x uint32, K~118) are repeatedly read/written. Original fast version already uses INT8 WMMA for coefficient -> RNS encoding, persistent pinned staging, asynchronous two-stream overlap. CRT still computes ~N*L*K = 425,328,640 integer per-column multiply terms, suggesting exact tensor-core CRT as an algorithmic direction. This is an operation count, **not** a timing estimate or speedup.

Research gate before any new FHERMA submission: (1) measure fast baseline stage costs, (2) derive exact algorithm and overflow proof, (3) build correctness against reference, (4) profile real Blackwell, (5) only submit when whole-operation latency improves. **Do not replace the existing challenge submission with an unverified candidate.**

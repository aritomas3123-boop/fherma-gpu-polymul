// Exact negacyclic polynomial multiplication, prepared for Ari Rabinovich.
// RNS NTT + centered CRT. Initial GPU implementation; validation pending.
// Interface/harness: https://github.com/fherma-ai/polynomial-mult-cupqc-example
#include "fherma.h"

// ---- rns_core.h ----
#include <cstdint>
#if defined(__CUDACC__)
#define FHD __host__ __device__
#else
#define FHD
#endif

namespace ari {
struct Prime {
    uint32_t p, neg_inv, r2, one;
};

// All primes are below 2^31. Consequently t + m*p fits in uint64_t.
FHD inline uint32_t mont(uint32_t a, uint32_t b, const Prime& p) {
    const uint64_t t = uint64_t(a) * b;
    const uint32_t m = uint32_t(t) * p.neg_inv;
    const uint32_t u = uint32_t((t + uint64_t(m) * p.p) >> 32);
    return u >= p.p ? u - p.p : u;
}
FHD inline uint32_t add(uint32_t a, uint32_t b, uint32_t p) {
    const uint32_t s = a + b;
    return s >= p ? s-p : s;
}
FHD inline uint32_t sub(uint32_t a, uint32_t b, uint32_t p) {
    return a >= b ? a-b : a+p-b;
}
FHD inline uint32_t reverse_bits(uint32_t x, unsigned bits) {
    uint32_t r=0;
    for (unsigned i=0;i<bits;++i) { r=(r<<1)|(x&1); x>>=1; }
    return r;
}
FHD inline uint32_t reduce_coefficient(const uint32_t* a, unsigned L,
                                      const Prime& p) {
    // mont(r, R^2) = r*2^32 (mod p), even with r in ordinary form.
    // p>2^30 means reducing one input word takes at most three subtractions.
    uint32_t r=0;
    for (unsigned j=L;j-- >0;) {
        uint32_t v=a[j];
        if(v>=p.p)v-=p.p;
        if(v>=p.p)v-=p.p;
        if(v>=p.p)v-=p.p;
        r=add(mont(r,p.r2,p),v,p.p);
    }
    return r;
}
FHD inline void butterfly(uint32_t& a, uint32_t& b, uint32_t w,
                          const Prime& p) {
    const uint32_t u=a, v=mont(b,w,p);
    a=add(u,v,p.p); b=sub(u,v,p.p);
}
} // namespace ari

// ---- rns_plan.h ----
#include <limits>
#include <stdexcept>
#include <vector>

namespace ari {
inline uint32_t powmod(uint32_t a, uint64_t e, uint32_t p) {
    uint32_t r=1;
    while(e) { if(e&1) r=uint32_t(uint64_t(r)*a%p);
        a=uint32_t(uint64_t(a)*a%p); e>>=1; }
    return r;
}
inline bool is_prime(uint32_t p) {
    if(p<2) return false;
    for(uint32_t a: {2u,3u,5u,7u,11u,13u,17u}) {
        if(p==a) return true;
        if(p%a==0) return false;
    }
    uint32_t d=p-1, s=0; while(!(d&1)) {d>>=1;++s;}
    // These bases suffice for the entire uint32_t range.
    for(uint32_t a: {2u,3u,5u,7u,11u}) {
        uint32_t x=powmod(a,d,p);
        if(x==1||x==p-1) continue;
        bool pass=false;
        for(uint32_t i=1;i<s;++i) {x=uint32_t(uint64_t(x)*x%p);
            if(x==p-1){pass=true;break;}}
        if(!pass) return false;
    }
    return true;
}
struct Plan {
    unsigned N,L,W,logN,K;
    std::vector<Prime> primes;
    std::vector<uint32_t> roots, inverse_roots, twist, untwist;
    explicit Plan(const fherma::Point& point): N(point.N),L(point.L),W(point.W),logN(0),K(0) {
        if(N<2 || (N&(N-1)) || N>65536 || W<2 || W>1747 || L!=(W+31)/32 || point.q.data.size()!=L)
            throw std::runtime_error("supported points: power-of-two 2<=N<=65536, 2<=W<=1747, canonical L");
        while((1u<<logN)<N) ++logN;
        // Every chosen prime >2^30, so P > 2^(30*K) >= 16*N*2^(2W).
        // This margin makes centered approximate CRT reconstruction unambiguous.
        K=(2*W+logN+4+29)/30;
        uint32_t p=((uint32_t(1)<<31)-2)/(2*N)*(2*N)+1;
        for(unsigned k=0;k<K;++k) {
            while(p>(1u<<30) && !is_prime(p)) p-=2*N;
            if(p<=(1u<<30)) throw std::runtime_error("not enough suitable RNS primes");
            uint32_t inv=1; for(unsigned j=0;j<5;++j) inv*=2-p*inv;
            const uint32_t one=uint32_t((uint64_t(1)<<32)%p);
            primes.push_back({p,uint32_t(0-inv),uint32_t(uint64_t(one)*one%p),one});
            p-=2*N;
        }
        roots.resize(size_t(K)*(N/2)); inverse_roots.resize(roots.size());
        twist.resize(size_t(K)*N); untwist.resize(twist.size());
        for(unsigned k=0;k<K;++k) {
            const auto& p=primes[k]; uint32_t psi=0;
            for(uint32_t a=2;!psi;++a) {
                uint32_t candidate=powmod(a,(p.p-1)/(2*N),p.p);
                if(powmod(candidate,N,p.p)==p.p-1) psi=candidate;
            }
            const uint32_t psi_inv=powmod(psi,p.p-2,p.p), omega=uint32_t(uint64_t(psi)*psi%p.p);
            const uint32_t omega_inv=powmod(omega,p.p-2,p.p), n_inv=powmod(N,p.p-2,p.p);
            uint32_t a=1,b=1;
            for(unsigned i=0;i<N/2;++i) {
                roots[size_t(k)*(N/2)+i]=mont(a,p.r2,p);
                inverse_roots[size_t(k)*(N/2)+i]=mont(b,p.r2,p);
                a=uint32_t(uint64_t(a)*omega%p.p); b=uint32_t(uint64_t(b)*omega_inv%p.p);
            }
            a=1; b=n_inv;
            for(unsigned i=0;i<N;++i) {
                twist[size_t(k)*N+i]=mont(a,p.r2,p);
                untwist[size_t(k)*N+i]=mont(b,p.r2,p);
                a=uint32_t(uint64_t(a)*psi%p.p); b=uint32_t(uint64_t(b)*psi_inv%p.p);
            }
        }
    }
};
} // namespace ari

// ---- rns_crt_core.hpp ----

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Exact centered CRT reconstruction. All big integers here are little-endian
// 32-bit limbs. Host planning has no dependencies beyond the C++ standard library.
namespace fherma_crt {
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Words = std::vector<u32>;
struct PrimeInfo {
    u32 p, inv, inv_shoup;
    double reciprocal;
};
struct HostPlan {
    unsigned n, limbs, bits;
    u32 gap; // 2^bits - q if representable by one word, zero otherwise
    Words q;
    std::vector<PrimeInfo> primes;
    Words weights; // [limb][prime, followed by one correction column]
};
inline unsigned word_bits(u32 a) {
    unsigned b = 0;
    while (a) { ++b; a >>= 1; }
    return b;
}
inline unsigned bit_length(const Words &a) {
    std::size_t size = a.size();
    while (size && !a[size - 1]) --size;
    return size ? unsigned(32 * (size - 1) + word_bits(a[size - 1])) : 0;
}
inline void multiply_small(Words &a, u32 b) {
    u64 carry = 0;
    for (auto &v : a) {
        carry += u64(v) * b;
        v = u32(carry);
        carry >>= 32;
    }
    if (carry) a.push_back(u32(carry));
}
inline Words divide_small(const Words &a, u32 b) {
    Words out(a.size());
    u64 rem = 0;
    for (std::size_t j = a.size(); j-- > 0;) {
        u64 x = (rem << 32) | a[j];
        out[j] = u32(x / b);
        rem = x % b;
    }
    if (rem) throw std::logic_error("CRT basis division");
    return out;
}
inline u32 mod_small(const Words &a, u32 p) {
    u64 r = 0;
    for (std::size_t i = a.size(); i-- > 0;) r = ((r << 32) | a[i]) % p;
    return u32(r);
}
inline u32 pow_mod(u32 a, u32 e, u32 p) {
    u64 out = 1;
    while (e) {
        if (e & 1u) out = out * a % p;
        a = u32(u64(a) * a % p);
        e >>= 1;
    }
    return u32(out);
}
#ifdef __CUDACC__
#define FHERMA_CRT_HD __host__ __device__
#else
#define FHERMA_CRT_HD
#endif
template<unsigned Stride>
FHERMA_CRT_HD inline u32 shifted_word(const u32 *q, unsigned limbs, unsigned pos, unsigned shift) {
    unsigned whole = shift / 32, off = shift % 32;
    if (pos < whole) return 0;
    unsigned i = pos - whole;
    u32 out = i < limbs ? q[i] << off : 0;
    if (off && i && i - 1 < limbs) out |= q[i - 1] >> (32 - off);
    return out;
}
template<unsigned Stride>
FHERMA_CRT_HD inline bool subtract_if_ge(u32 *a, unsigned size, const u32 *q, unsigned limbs, unsigned shift) {
    for (unsigned j = size; j-- > 0;) {
        u32 b = shifted_word<Stride>(q, limbs, j, shift);
        if (a[j * Stride] < b) return false;
        if (a[j * Stride] > b) break;
    }
    u64 borrow = 0;
    for (unsigned j = 0; j < size; ++j) {
        u64 b = u64(shifted_word<Stride>(q, limbs, j, shift)) + borrow;
        u32 x = a[j * Stride];
        a[j * Stride] = x - u32(b);
        borrow = u64(x) < b;
    }
    return true;
}
inline Words reduce_wide(const Words &value, const Words &q) {
    Words out = value;
    unsigned bits = bit_length(value), qb = bit_length(q);
    if (bits >= qb) {
        for (unsigned s = bits - qb + 1; s-- > 0;)
            subtract_if_ge<1>(out.data(), unsigned(out.size()), q.data(), unsigned(q.size()), s);
    }
    out.resize(q.size());
    return out;
}
inline Words subtract_words(const Words &a, const Words &b) {
    Words out(a.size());
    u64 borrow = 0;
    for (std::size_t j = 0; j < a.size(); ++j) {
        u64 v = u64(j < b.size() ? b[j] : 0) + borrow;
        out[j] = a[j] - u32(v);
        borrow = u64(a[j]) < v;
    }
    if (borrow) throw std::logic_error("negative unsigned CRT weight");
    return out;
}
inline HostPlan make_plan(unsigned n, const u32 *q, unsigned limbs, const std::vector<u32>& ps) {
    if (!n || (n & (n - 1)) || !limbs || ps.empty() || ps.size() > 1024)
        throw std::invalid_argument("invalid CRT dimensions");
    HostPlan plan{};
    plan.n = n; plan.limbs = limbs; plan.q.assign(q, q + limbs);
    plan.bits = bit_length(plan.q);
    if (plan.bits < 2 || (plan.bits + 31) / 32 != limbs)
        throw std::invalid_argument("invalid CRT q width");
    Words two_w(plan.bits / 32 + 1, 0);
    two_w[plan.bits / 32] = u32(1) << (plan.bits % 32);
    Words delta = subtract_words(two_w, plan.q);
    plan.gap = bit_length(delta) <= 32 ? delta[0] : 0;
    Words product{1};
    for (u32 p : ps) {
        if (p < 3 || p >= 0x80000000u) throw std::invalid_argument("CRT expects primes below 2^31");
        multiply_small(product, p);
    }
    // |true coefficient| <= n(q-1)^2. P >= 2^(2W + log2(n) + 2)
    // gives at least two guard bits: |true coefficient/P| < 1/4.
    if (bit_length(product) < 2 * plan.bits + word_bits(n) + 2)
        throw std::invalid_argument("CRT prime product lacks two guard bits");
    std::size_t count = ps.size(), pitch = count + 1;
    plan.weights.resize(limbs * pitch);
    for (std::size_t j = 0; j <= count; ++j) {
        Words reduced;
        if (j < count) {
            u32 p = ps[j];
            Words basis = divide_small(product, p);
            u32 basis_r = mod_small(basis, p);
            u32 inv = pow_mod(basis_r, p - 2, p);
            if (u64(basis_r) * inv % p != 1)
                throw std::invalid_argument("CRT moduli must be distinct primes");
            plan.primes.push_back({p, inv, u32((u64(inv) << 32) / p), 1.0 / double(p)});
            reduced = reduce_wide(basis, plan.q);
        } else {
            // Add k*(-P mod q) so every term in the fixed-width column sum is nonnegative.
            reduced = subtract_words(plan.q, reduce_wide(product, plan.q));
        }
        for (unsigned i = 0; i < limbs; ++i) plan.weights[i * pitch + j] = reduced[i];
    }
    return plan;
}
FHERMA_CRT_HD inline u32 weight_residue(u32 r, const PrimeInfo& p) {
#ifdef __CUDA_ARCH__
    u32 quot = __umulhi(r, p.inv_shoup);
#else
    u32 quot = u32((u64(r) * p.inv_shoup) >> 32);
#endif
    u32 t = r * p.inv - quot * p.p;
    return t >= p.p ? t - p.p : t;
}

template<unsigned Stride>
FHERMA_CRT_HD inline u32 read_window(const u32 *a, unsigned size, unsigned offset) {
    unsigned pos = offset / 32, off = offset % 32;
    u32 lo = pos < size ? a[pos * Stride] >> off : 0;
    if (off && pos + 1 < size) lo |= a[(pos + 1) * Stride] << (32 - off);
    return lo;
}
template<unsigned Stride>
FHERMA_CRT_HD inline void reduce_sum(u32 *a, unsigned limbs, unsigned bits, u32 gap, const u32 *q) {
    const unsigned size = limbs + 2;
    // The sum is < 1025 * 2^31 * q. At most 42 bits lie above bit W.
    // When q = 2^W - gap with small gap, fold those bits into the low
    // words. The two official parameter points both use this fast path.
    if (gap) {
        for (unsigned iteration = 0; iteration < 5; ++iteration) {
            u32 high0 = read_window<Stride>(a, size, bits);
            u32 high1 = read_window<Stride>(a, size, bits + 32);
            if (!(high0 | high1)) {
                subtract_if_ge<Stride>(a, size, q, limbs, 0);
                return;
            }
            unsigned cut = bits / 32, off = bits % 32;
            if (off) a[cut++ * Stride] &= (u32(1) << off) - 1;
            for (unsigned j = cut; j < size; ++j) a[j * Stride] = 0;
            u64 carry = u64(high0) * gap + a[0];
            a[0] = u32(carry);
            carry >>= 32;
            carry += u64(high1) * gap + a[Stride];
            a[Stride] = u32(carry);
            carry >>= 32;
            for (unsigned j = 2; carry && j < size; ++j) {
                carry += a[j * Stride];
                a[j * Stride] = u32(carry);
                carry >>= 32;
            }
        }
    }
    // q's high bits or the degree do not have to be challenge constants.
    // Skip impossible shifts to avoid comparing a truncated divisor.
    for (unsigned s = 43; s-- > 0;) {
        if (bits + s <= size * 32) subtract_if_ge<Stride>(a, size, q, limbs, s);
    }
}

} // namespace fherma_crt

namespace fherma_crt {
template<unsigned Stride>
FHERMA_CRT_HD inline u32 correction(const u32 *terms, const PrimeInfo *primes, unsigned count) {
    // All terms are nonnegative. With at most 1024 terms the absolute
    // rounding error in binary64 is below 1e-9, far inside the 1/4 margin.
    double sum = 0;
    for (unsigned j = 0; j < count; ++j)
        sum += double(terms[j * Stride]) * primes[j].reciprocal;
    return u32(sum + 0.5);
}
template<unsigned Stride>
FHERMA_CRT_HD inline void sum_column(const u32 *terms, const u32 *basis_column,
    unsigned pitch, u64 &low_sum, u64 &high_sum) {
    // Keep accumulators in registers even when the output references are shared memory.
    u64 lo = 0, hi = 0;
    for (unsigned j = 0; j < pitch; ++j) {
        u64 product = u64(terms[j * Stride]) * basis_column[j];
        lo += u32(product);
        hi += product >> 32;
    }
    low_sum = lo;
    high_sum = hi;
}
template<unsigned Stride>
FHERMA_CRT_HD inline void carry_and_reduce(u32 *digits, const u64 *low, const u64 *high,
    unsigned l, unsigned bits, u32 gap, const u32 *q) {
    u64 carry = 0, prior_high = 0;
    for (unsigned i = 0; i < l; ++i) {
        carry += low[i * Stride] + prior_high;
        digits[i * Stride] = u32(carry);
        carry >>= 32;
        prior_high = high[i * Stride];
    }
    carry += prior_high;
    digits[l * Stride] = u32(carry);
    digits[(l + 1) * Stride] = u32(carry >> 32);
    reduce_sum<Stride>(digits, l, bits, gap, q);
}
// CPU mirror of the GPU's per-coefficient arithmetic. Input may be prime-major
// (set stride=n), or one residue vector (stride=1). Used for independent tests.
inline Words reconstruct_one(const HostPlan &p, const u32 *r, std::size_t stride = 1) {
    unsigned k = unsigned(p.primes.size()), pitch = k + 1;
    Words t(pitch), digits(p.limbs + 2);
    std::vector<u64> low(p.limbs), high(p.limbs);
    for (unsigned j = 0; j < k; ++j) t[j] = weight_residue(r[j * stride], p.primes[j]);
    t[k] = correction<1>(t.data(), p.primes.data(), k);
    for (unsigned i = 0; i < p.limbs; ++i)
        sum_column<1>(t.data(), p.weights.data() + i * pitch, pitch, low[i], high[i]);
    carry_and_reduce<1>(digits.data(), low.data(), high.data(), p.limbs, p.bits, p.gap, p.q.data());
    digits.resize(p.limbs);
    return digits;
}
} // namespace fherma_crt
#undef FHERMA_CRT_HD

// ---- rns_crt.cuh ----
#include <cuda_runtime.h>

#include <array>
#include <limits>
#include <vector>

namespace ari_tensor {
using fherma_crt::u32;
using fherma_crt::u64;
using fherma_crt::HostPlan;
using fherma_crt::Words;
inline unsigned ceil_multiple(unsigned val,unsigned multiple) {
    return ((val+multiple-1)/multiple)*multiple;
}
inline u32 fixed_reciprocal(u32 p) {
    // floor(2^52/p) fits in 32 bits for the NTT's p>2^30.
    if(p<=(1u<<20)) throw std::invalid_argument("RNS prime too small for quantised CRT");
    return u32((u64(1)<<52)/p);
}
#ifdef __CUDACC__
#define ARI_TC_HD __host__ __device__
#else
#define ARI_TC_HD
#endif
ARI_TC_HD inline u32 fixed_fraction(u32 t,u32 reciprocal) {
#ifdef __CUDA_ARCH__
    return __umulhi(t,reciprocal);
#else
    return u32(u64(t)*reciprocal>>32);
#endif
}
ARI_TC_HD inline u32 nearest_crt_integer(u32 sum_of_fractions) {
    // Each fixed_fraction differs from 2^20*t/p by < 1+t/2^32 < 1.5.
    // At most 1024 terms => absolute scaled sum error < 0.0015. The
    // certified true sum is within 1/4 of an integer, so rounding is exact.
    return (sum_of_fractions+(1u<<19))>>20;
}
ARI_TC_HD inline std::size_t packed_index(unsigned d,unsigned inner,unsigned row,unsigned kpad,unsigned rowpad) {
    return ((std::size_t(d)*(kpad/16)+inner/16)*rowpad+row)*16+inner%16;
}
ARI_TC_HD inline u32 byte_at(u32 value,unsigned which) {return (value>>(8*which))&255u;}
inline std::vector<std::uint8_t> pack_weights(const HostPlan& p,unsigned kpad,unsigned lpad) {
    std::vector<std::uint8_t> out(std::size_t(4)*lpad*kpad);
    const unsigned terms=unsigned(p.primes.size())+1;
    for(unsigned d=0;d<4;++d)
        for(unsigned l=0;l<p.limbs;++l)for(unsigned k=0;k<terms;++k)
            out[packed_index(d,k,l,kpad,lpad)]=std::uint8_t(byte_at(p.weights[l*terms+k],d));
    return out;
}
} // namespace ari_tensor
#undef ARI_TC_HD

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ari_wmma_crt {
constexpr unsigned tile = 16;
constexpr unsigned warps_per_block = 4;
#ifdef __CUDACC__
#define ARI_WMMA_HD __host__ __device__
#else
#define ARI_WMMA_HD
#endif

ARI_WMMA_HD inline unsigned padded_rows(unsigned rows) {
    return (rows + tile - 1) / tile * tile;
}
// Byte planes are split again into K=16 panels. Each row within a panel
// contributes exactly 16 contiguous bytes. A and B tile bases always lie on
// 256-byte boundaries, including the odd K panels (WMMA requires 32 bytes).
ARI_WMMA_HD inline std::size_t packed_offset(unsigned digit, unsigned row, unsigned inner,
                                              unsigned rows_padded, unsigned k_padded) {
    return ((std::size_t(digit) * (k_padded / tile) + inner / tile) * rows_padded
            + row) * tile + inner % tile;
}
struct SplitSum { std::uint64_t lo, hi; };
// Each d_i is the sum of all byte products at base-256 place i. Separate
// at 2^32 so the complete (possibly >64-bit) result is never formed.
ARI_WMMA_HD inline SplitSum split_diagonals(std::uint32_t d0, std::uint32_t d1,
    std::uint32_t d2, std::uint32_t d3, std::uint32_t d4,
    std::uint32_t d5, std::uint32_t d6) {
    const std::uint64_t lo = d0 + (std::uint64_t(d1) << 8) + (std::uint64_t(d2) << 16)
                               + (std::uint64_t(d3) << 24);
    const std::uint64_t hi = d4 + (std::uint64_t(d5) << 8) + (std::uint64_t(d6) << 16);
    return {std::uint32_t(lo), hi + (lo >> 32)};
}
// Input is the existing CRT-plan weights table [limb][term]. The same generic
// helper can also pack a host term matrix [coefficient][term] in validation.
inline std::vector<std::uint8_t> pack_host_rows(const std::uint32_t *src,
    unsigned rows, unsigned active_terms, unsigned k_padded) {
    if (!rows || !active_terms || (k_padded % tile) || active_terms > k_padded ||
        k_padded > 4096)
        throw std::invalid_argument("invalid tensor CRT dimensions");
    const unsigned pad = padded_rows(rows);
    std::vector<std::uint8_t> out(std::size_t(4) * pad * k_padded);
    for (unsigned row=0; row<rows; ++row)
        for (unsigned j=0; j<active_terms; ++j) {
            const std::uint32_t v = src[std::size_t(row) * active_terms + j];
            for (unsigned b=0; b<4; ++b)
                out[packed_offset(b,row,j,pad,k_padded)] = std::uint8_t(v >> (8*b));
        }
    return out;
}
#undef ARI_WMMA_HD
} // namespace ari_wmma_crt

#include <cuda_runtime.h>
#include <mma.h>

namespace ari_wmma_crt {
namespace wmma = nvcuda::wmma;
using Acc = wmma::fragment<wmma::accumulator,16,16,16,int>;
using FragA = wmma::fragment<wmma::matrix_a,16,16,16,unsigned char,wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b,16,16,16,unsigned char,wmma::col_major>;

// One warp per 16 coefficient x 16 limb tile. Four neighbouring output limb
// tiles reuse each A panel in cache. Data in any padding must be zero.
__global__ void gemm_kernel(const std::uint8_t *__restrict__ a,
    const std::uint8_t *__restrict__ b, std::uint64_t *__restrict__ lo,
    std::uint64_t *__restrict__ hi, unsigned n, unsigned columns, unsigned k_padded) {
    const unsigned lane = threadIdx.x % 32;
    const unsigned warp = threadIdx.x / 32;
    const unsigned row0 = blockIdx.x * tile;
    const unsigned col0 = (blockIdx.y * warps_per_block + warp) * tile;
    if (col0 >= columns) return;  // warp-uniform; there are no CTA barriers.
    const unsigned np = padded_rows(n), lp = padded_rows(columns);
    Acc diag[7];
    #pragma unroll
    for (int d=0; d<7; ++d) wmma::fill_fragment(diag[d],0);

    for (unsigned j=0; j<k_padded; j+=tile) {
        FragA af[4]; FragB bf[4];
        #pragma unroll
        for (unsigned d=0; d<4; ++d) {
            // Packed K panels each start at an aligned, contiguous 16x16 tile.
            wmma::load_matrix_sync(af[d], a + packed_offset(d,row0,j,np,k_padded), tile);
            wmma::load_matrix_sync(bf[d], b + packed_offset(d,col0,j,lp,k_padded), tile);
        }
        #pragma unroll
        for (unsigned u=0; u<4; ++u) {
            #pragma unroll
            for (unsigned v=0; v<4; ++v)
                wmma::mma_sync(diag[u+v], af[u], bf[v], diag[u+v]);
        }
    }
    // On every allowed K: each diagonal <= 4*K*255^2 <= 1,065,369,600,
    // so accumulating into a signed 32-bit integer is exact, with no FP.
    // Accumulator fragment storage layout is undocumented; these matching
    // fragments share the same type and shape. Operate elementwise, then let
    // WMMA map the three resulting word planes back into matrix coordinates.
    #pragma unroll
    for (int t=0; t<Acc::num_elements; ++t) {
        const auto s=split_diagonals(diag[0].x[t],diag[1].x[t],diag[2].x[t],
            diag[3].x[t],diag[4].x[t],diag[5].x[t],diag[6].x[t]);
        diag[0].x[t]=static_cast<int>(static_cast<std::uint32_t>(s.lo));
        diag[1].x[t]=static_cast<int>(static_cast<std::uint32_t>(s.hi));
        diag[2].x[t]=static_cast<int>(static_cast<std::uint32_t>(s.hi>>32));
    }
    __shared__ __align__(32) int words[warps_per_block][3][tile*tile];
    #pragma unroll
    for (unsigned d=0; d<3; ++d)
        wmma::store_matrix_sync(words[warp][d],diag[d],tile,wmma::mem_row_major);
    __syncwarp();
    #pragma unroll
    for (unsigned t=lane; t<tile*tile; t+=32) {
        const unsigned row=row0+t/tile, col=col0+t%tile;
        if (row<n && col<columns) {
            const auto off=std::size_t(row)*columns+col;
            lo[off]=std::uint32_t(words[warp][0][t]);
            hi[off]=std::uint32_t(words[warp][1][t])
                  | (std::uint64_t(std::uint32_t(words[warp][2][t]))<<32);
        }
    }
}
// K may be at most 4096 for the arithmetic guarantee above; K=64 and K=128
// cover the official points. A and B must come from a 32-byte aligned device
// allocation, normally cudaMalloc. All K and matrix tile padding must be zero.
inline cudaError_t launch(const std::uint8_t* a,const std::uint8_t* b,
    std::uint64_t* lo,std::uint64_t* hi,unsigned n,unsigned columns,
    unsigned k_padded,cudaStream_t stream=nullptr) {
    if (!n||!columns||!k_padded||k_padded>4096||k_padded%tile) return cudaErrorInvalidValue;
    gemm_kernel<<<dim3((n+tile-1)/tile,(columns+tile*warps_per_block-1)/(tile*warps_per_block)),
        32*warps_per_block,0,stream>>>(a,b,lo,hi,n,columns,k_padded);
    return cudaGetLastError();
}
} // namespace ari_wmma_crt


namespace ari_wmma_crt {
constexpr unsigned max_fused_limbs = tile * warps_per_block;
#ifdef __CUDACC__
#define ARI_FUSED_HD __host__ __device__
#else
#define ARI_FUSED_HD
#endif

// All three planes are stored in column-major order [limb][tile row]. Each
// row has a single owner. The two extra limbs on the low plane allow final
// carry propagation and the shared q reduction before any global writes.
ARI_FUSED_HD inline void normalize_fused_row(std::uint32_t *low,
        const std::uint32_t *hi0, const std::uint32_t *hi1,
        unsigned limbs,unsigned bits,std::uint32_t gap,const std::uint32_t *q) {
    std::uint64_t carry=0;
    for(unsigned col=0;col<limbs;++col) {
        const unsigned i=col*tile;
        carry+=low[i];
        low[i]=std::uint32_t(carry);
        const std::uint64_t high=hi0[i]|(std::uint64_t(hi1[i])<<32);
        carry=high+(carry>>32);
    }
    low[limbs*tile]=std::uint32_t(carry);
    low[(limbs+1)*tile]=std::uint32_t(carry>>32);
    fherma_crt::reduce_sum<tile>(low,limbs,bits,gap,q);
}
#undef ARI_FUSED_HD
} // namespace ari_wmma_crt


namespace ari_wmma_crt {
// Four warps cover up to 64 limbs for all 16 coefficients. Compute one
// unnormalised limb dot product per matrix cell, retaining all results in
// this CTA. No intermediate per-cell values are written to global memory.
__global__ void gemm_finish_kernel(const std::uint8_t *__restrict__ a,
    const std::uint8_t *__restrict__ b,std::uint32_t *__restrict__ out,
    unsigned n,unsigned limbs,unsigned k_padded,unsigned bits,std::uint32_t gap,
    const std::uint32_t *__restrict__ q) {
    const unsigned tid=threadIdx.x, warp=tid/32;
    const unsigned row0=blockIdx.x*tile, col0=warp*tile;
    __shared__ __align__(32) int low[(max_fused_limbs+2)*tile];
    __shared__ __align__(32) int hi0[max_fused_limbs*tile];
    __shared__ __align__(32) int hi1[max_fused_limbs*tile];

    // This predicate is uniform within a warp. Inactive warps never access
    // unallocated B columns and still reach the block barriers below.
    if(col0<limbs) {
        Acc diag[7];
        #pragma unroll
        for(unsigned d=0;d<7;++d) wmma::fill_fragment(diag[d],0);
        const unsigned np=padded_rows(n),lp=padded_rows(limbs);
        for(unsigned j=0;j<k_padded;j+=tile) {
            FragA af[4];FragB bf[4];
            #pragma unroll
            for(unsigned d=0;d<4;++d) {
                wmma::load_matrix_sync(af[d],a+packed_offset(d,row0,j,np,k_padded),tile);
                wmma::load_matrix_sync(bf[d],b+packed_offset(d,col0,j,lp,k_padded),tile);
            }
            #pragma unroll
            for(unsigned u=0;u<4;++u) {
                #pragma unroll
                for(unsigned v=0;v<4;++v)
                    wmma::mma_sync(diag[u+v],af[u],bf[v],diag[u+v]);
            }
        }
        #pragma unroll
        for(int e=0;e<Acc::num_elements;++e) {
            const SplitSum s=split_diagonals(diag[0].x[e],diag[1].x[e],diag[2].x[e],
                diag[3].x[e],diag[4].x[e],diag[5].x[e],diag[6].x[e]);
            diag[0].x[e]=static_cast<int>(static_cast<std::uint32_t>(s.lo));
            diag[1].x[e]=static_cast<int>(static_cast<std::uint32_t>(s.hi));
            diag[2].x[e]=static_cast<int>(static_cast<std::uint32_t>(s.hi>>32));
        }
        // Each 16x16 tile is stored by column so lanes in the carry warp
        // read 16 consecutive words when they process any particular limb.
        wmma::store_matrix_sync(low+col0*tile,diag[0],tile,wmma::mem_col_major);
        wmma::store_matrix_sync(hi0+col0*tile,diag[1],tile,wmma::mem_col_major);
        wmma::store_matrix_sync(hi1+col0*tile,diag[2],tile,wmma::mem_col_major);
    }
    __syncthreads();

    auto *digits=reinterpret_cast<std::uint32_t*>(low);
    if(tid<tile && row0+tid<n)
        normalize_fused_row(digits+tid,reinterpret_cast<const std::uint32_t*>(hi0)+tid,
            reinterpret_cast<const std::uint32_t*>(hi1)+tid,limbs,bits,gap,q);
    __syncthreads();

    // Consecutive lanes write consecutive words in FHERMA's row-major
    // coefficient layout. These are the kernel's only global stores.
    for(unsigned i=tid;i<tile*limbs;i+=32*warps_per_block) {
        const unsigned row=i/limbs,col=i-row*limbs;
        if(row0+row<n)
            out[std::size_t(row0)*limbs+i]=digits[col*tile+row];
    }
}

inline cudaError_t launch_fused(const std::uint8_t* a,const std::uint8_t* b,
    std::uint32_t* out,unsigned n,unsigned limbs,unsigned k_padded,unsigned bits,
    std::uint32_t gap,const std::uint32_t* q,cudaStream_t stream=nullptr) {
    if(!n||!limbs||limbs>max_fused_limbs||!k_padded||k_padded>4096||
       k_padded%tile||bits<2||bits>limbs*32||(bits+31)/32!=limbs)
        return cudaErrorInvalidValue;
    gemm_finish_kernel<<<(n+tile-1)/tile,32*warps_per_block,0,stream>>>(
        a,b,out,n,limbs,k_padded,bits,gap,q);
    return cudaGetLastError();
}
} // namespace ari_wmma_crt

#include <cuda_runtime.h>

namespace ari_tensor {
constexpr unsigned pack_x=16,pack_y=8;
struct TensorArgs {
    unsigned n, count, limbs, bits, n_pad, k_pad, l_pad;
    u32 gap;
    fherma_crt::PrimeInfo *ps;
    u32 *q, *reciprocal;
};
// Read one prime plane at a time for 16 neighbouring coefficients, transpose
// via shared memory, then write packed 16-byte fragments used by IMMA.
__global__ void pack_terms(const u32 *residues,std::uint8_t *packed,TensorArgs s) {
    extern __shared__ u32 buffer[];
    const unsigned x=threadIdx.x,y=threadIdx.y,start=blockIdx.x*pack_x,c=start+x;
    constexpr unsigned stride=pack_x+1;
    u32 *terms=buffer, *partials=terms+s.k_pad*stride;
    u32 partial=0;
    for(unsigned j=y;j<s.k_pad;j+=pack_y){
        u32 t=0;
        if(j<s.count && c<s.n){
            t=fherma_crt::weight_residue(residues[std::size_t(j)*s.n+c],s.ps[j]);
            partial+=fixed_fraction(t,s.reciprocal[j]);
        }
        terms[j*stride+x]=t;
    }
    partials[y*stride+x]=partial;
    __syncthreads();
    if(y==0){
        u32 q=0;
        for(unsigned b=0;b<pack_y;++b)q+=partials[b*stride+x];
        terms[s.count*stride+x]=nearest_crt_integer(q);
    }
    __syncthreads();
    for(unsigned index=y*pack_x+x;index<s.k_pad*pack_x;index+=pack_x*pack_y){
        unsigned group=index/(pack_x*16),row=(index/16)%pack_x;
        unsigned j=group*16+index%16, val=terms[j*stride+row];
        for(unsigned d=0;d<4;++d){
            auto off=packed_index(d,j,start+row,s.k_pad,s.n_pad);
            packed[off]=std::uint8_t(byte_at(val,d));
        }
    }
}
class DevicePlan {
    TensorArgs s_{};
    std::uint8_t *a_=nullptr,*b_=nullptr;

    std::size_t pack_shared_=0;
    static void check(cudaError_t st) {
        if(st!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(st));
    }
    template<class T>static void allocate(T*& ptr,std::size_t n){
        check(cudaMalloc(reinterpret_cast<void**>(&ptr),n*sizeof(T)));
    }
    template<class T>static void upload(T*& ptr,const std::vector<T>& values){
        allocate(ptr,values.size());
        check(cudaMemcpy(ptr,values.data(),values.size()*sizeof(T),cudaMemcpyHostToDevice));
    }
    void release() noexcept {
        if(a_)cudaFree(a_); if(b_)cudaFree(b_); 
        if(s_.ps)cudaFree(s_.ps); if(s_.reciprocal)cudaFree(s_.reciprocal); if(s_.q)cudaFree(s_.q);
    }
public:
    DevicePlan(unsigned n,const u32 *q,unsigned l,const std::vector<u32>& primes){
        auto h=fherma_crt::make_plan(n,q,l,primes);
        s_.n=n;s_.limbs=l;s_.bits=h.bits;s_.gap=h.gap;s_.count=unsigned(primes.size());
        s_.k_pad=ceil_multiple(s_.count+1,32);s_.l_pad=ceil_multiple(l,16);s_.n_pad=ceil_multiple(n,16);
        std::vector<u32> reciprocals;
        for(auto p:primes)reciprocals.push_back(fixed_reciprocal(p));
        auto weights=pack_weights(h,s_.k_pad,s_.l_pad);
        pack_shared_=(s_.k_pad+pack_y)*(pack_x+1)*sizeof(u32);
        int dev;cudaDeviceProp prop{};
        check(cudaGetDevice(&dev));check(cudaGetDeviceProperties(&prop,dev));
        if(prop.major<8)throw std::runtime_error("integer WMMA CRT requires SM80 or newer");
        if(pack_shared_>prop.sharedMemPerBlock)
            throw std::runtime_error("integer WMMA CRT tile exceeds device shared memory");
        try {
            upload(s_.q,h.q);upload(s_.ps,h.primes);upload(s_.reciprocal,reciprocals);
            upload(b_,weights);allocate(a_,4*std::size_t(s_.k_pad)*s_.n_pad);
        }catch(...){release();throw;}
    }
    ~DevicePlan(){release();}
    DevicePlan(const DevicePlan&)=delete;
    DevicePlan& operator=(const DevicePlan&)=delete;
    cudaError_t run(const u32* residues,u32 *out,cudaStream_t stream=nullptr) const {
        const dim3 block(pack_x,pack_y); const unsigned grid=(s_.n+pack_x-1)/pack_x;
        pack_terms<<<grid,block,pack_shared_,stream>>>(residues,a_,s_);
        auto st=cudaGetLastError();if(st!=cudaSuccess)return st;
        return ari_wmma_crt::launch_fused(a_,b_,out,s_.n,s_.limbs,s_.k_pad,s_.bits,s_.gap,s_.q,stream);
    }
};
} // namespace ari_tensor

// ---- ntt_tile.cuh ----
#ifndef FHERMA_TILE_COORDS_ONLY
#include <cuda_runtime.h>
#endif

// In-place DIT on bit-reversed data. The same two kernels serve forward (two
// arrays) and inverse (one array). Twiddles are in the existing plan's Montgomery
// representation. Data layout and standard-butterfly order are unchanged.
namespace ari_tile {
constexpr unsigned max_tile = 512;
constexpr unsigned low_columns = 16;
constexpr unsigned high_workers = 16;

struct Pair { unsigned first, second, root; };
FHD inline Pair low_pair(unsigned t, unsigned half, unsigned n) {
    unsigned j = t & (half - 1), first = 2 * (t - j) + j;
    return {first, first + half, j * (n / (2 * half))};
}
FHD inline Pair high_pair(unsigned h,unsigned half_high,unsigned low,unsigned n) {
    unsigned within = h & (half_high - 1);
    unsigned first = 2 * (h - within) + within;
    return {first, first + half_high,
        (within * max_tile + low) * (n / (2 * half_high * max_tile))};
}
#ifndef FHERMA_TILE_COORDS_ONLY
template<bool Two>
__global__ void low_stages(uint32_t *a, uint32_t *b, const ari::Prime *primes,
                           const uint32_t *roots, unsigned n, unsigned tile) {
    extern __shared__ uint32_t data[];
    uint32_t *sa = data, *sb = data + tile;
    const unsigned t = threadIdx.x;
    const std::size_t base = std::size_t(blockIdx.y) * n + blockIdx.x * tile;
    const uint32_t *table = roots + std::size_t(blockIdx.y) * (n / 2);
    const ari::Prime p = primes[blockIdx.y];
    sa[t] = a[base + t]; sa[t + blockDim.x] = a[base + t + blockDim.x];
    if constexpr (Two) { sb[t] = b[base + t]; sb[t + blockDim.x] = b[base + t + blockDim.x]; }
    __syncthreads();
    for (unsigned half = 1; half < tile; half *= 2) {
        Pair pair = low_pair(t, half, n);
        uint32_t w = table[pair.root];
        ari::butterfly(sa[pair.first], sa[pair.second], w, p);
        if constexpr (Two) ari::butterfly(sb[pair.first], sb[pair.second], w, p);
        __syncthreads();
    }
    a[base + t] = sa[t]; a[base + t + blockDim.x] = sa[t + blockDim.x];
    if constexpr (Two) { b[base + t] = sb[t]; b[base + t + blockDim.x] = sb[t + blockDim.x]; }
}
template<bool Two>
__global__ void high_stages(uint32_t *a, uint32_t *b, const ari::Prime *primes,
                            const uint32_t *roots, unsigned n) {
    // If index = h * 512 + low, all stages with half >= 512 keep "low"
    // constant. Compute 16 adjacent low columns together to coalesce loads.
    extern __shared__ uint32_t data[];
    const unsigned high_count = n / max_tile;
    uint32_t *sa = data, *sb = data + high_count * low_columns;
    const unsigned lane = threadIdx.x, row_worker = threadIdx.y;
    const unsigned low = blockIdx.x * low_columns + lane;
    const std::size_t plane = std::size_t(blockIdx.y) * n;
    const uint32_t *table = roots + std::size_t(blockIdx.y) * (n / 2);
    const ari::Prime p = primes[blockIdx.y];
    for (unsigned h = row_worker; h < high_count; h += high_workers) {
        unsigned local = h * low_columns + lane;
        std::size_t global = plane + h * max_tile + low;
        sa[local] = a[global];
        if constexpr (Two) sb[local] = b[global];
    }
    __syncthreads();
    for (unsigned half_high = 1; half_high < high_count; half_high *= 2) {
        for (unsigned h = row_worker; h < high_count / 2; h += high_workers) {
            Pair pair = high_pair(h, half_high, low, n);
            unsigned first = pair.first * low_columns + lane;
            unsigned second = pair.second * low_columns + lane;
            uint32_t w = table[pair.root];
            ari::butterfly(sa[first], sa[second], w, p);
            if constexpr (Two) ari::butterfly(sb[first], sb[second], w, p);
        }
        __syncthreads();
    }
    for (unsigned h = row_worker; h < high_count; h += high_workers) {
        unsigned local = h * low_columns + lane;
        std::size_t global = plane + h * max_tile + low;
        a[global] = sa[local];
        if constexpr (Two) b[global] = sb[local];
    }
}
template<bool Two>
inline cudaError_t transform(uint32_t *a, uint32_t *b, const ari::Prime *primes,
                             const uint32_t *roots, unsigned n, unsigned k,
                             cudaStream_t stream = nullptr) {
    if (n < 2 || n > 65536 || (n & (n - 1))) return cudaErrorInvalidValue;
    unsigned tile = n < max_tile ? n : max_tile;
    constexpr unsigned arrays = Two ? 2 : 1;
    low_stages<Two><<<dim3(n / tile, k), tile / 2, arrays * tile * sizeof(uint32_t), stream>>>
        (a, b, primes, roots, n, tile);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess || n == tile) return status;
    high_stages<Two><<<dim3(max_tile / low_columns, k),
        dim3(low_columns,high_workers), arrays * (n / tile) * low_columns * sizeof(uint32_t),stream>>>
        (a, b, primes, roots, n);
    return cudaGetLastError();
}
#endif // FHERMA_TILE_COORDS_ONLY
} // namespace ari_tile

// ---- solve.cu ----
// Work in progress: exact RNS/NTT GPU implementation, November 2026.
// The generated fherma.h and harness are supplied by FHERMA's example.
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <vector>

namespace ari_shoup {
struct Weight { uint32_t value, quotient; };
inline Weight make_weight(uint32_t value, uint32_t p) {
    return {value, uint32_t((uint64_t(value) << 32) / p)};
}
// W_j = 2^(32*(j+1)) mod p. The extra common factor is Montgomery R.
inline std::vector<Weight> make_weights(const std::vector<ari::Prime> &ps, unsigned limbs) {
    std::vector<Weight> w(std::size_t(ps.size()) * limbs);
    for (std::size_t k = 0; k < ps.size(); ++k) {
        uint32_t factor = ps[k].one;
        for (unsigned j = 0; j < limbs; ++j) {
            w[k * limbs + j] = make_weight(factor, ps[k].p);
            factor = uint32_t(uint64_t(factor) * ps[k].one % ps[k].p);
        }
    }
    return w;
}
FHD inline uint32_t lazy_product(uint32_t limb, Weight w, uint32_t p) {
    // For any limb < 2^32 the approximate quotient underestimates the true
    // quotient by at most 1. The exact residual is < 2p < 2^32.
#ifdef __CUDA_ARCH__
    uint32_t quotient = __umulhi(limb, w.quotient);
#else
    uint32_t quotient = uint32_t((uint64_t(limb) * w.quotient) >> 32);
#endif
    return limb * w.value - quotient * p;
}
FHD inline uint32_t canonical_word(uint32_t x, uint32_t p) {
    if (x >= p) x -= p;
    if (x >= p) x -= p;
    if (x >= p) x -= p;
    return x;
}
FHD inline uint32_t finish(uint64_t sum, Weight radix, uint32_t p) {
    uint32_t lo = canonical_word(uint32_t(sum), p);
    uint32_t hi = lazy_product(uint32_t(sum >> 32), radix, p);
    if (hi >= p) hi -= p;
    return ari::add(lo, hi, p);
}
FHD inline uint32_t convert(const uint32_t *input, const Weight *w, unsigned limbs, uint32_t p) {
    uint64_t sum = 0;
    for (unsigned j = 0; j < limbs; ++j) sum += lazy_product(input[j], w[j], p);
    return finish(sum, w[0], p);
}
// Same arithmetic with two input arrays, to interleave independent GPU instruction chains.
FHD inline void convert_both(const uint32_t *a, const uint32_t *b,
                             const Weight *w, unsigned limbs, uint32_t p,
                             uint32_t &x, uint32_t &y) {
    uint64_t sa=0, sb=0;
    for (unsigned j=0;j<limbs;++j) {
        Weight t=w[j];
        sa+=lazy_product(a[j],t,p);
        sb+=lazy_product(b[j],t,p);
    }
    x=finish(sa,w[0],p); y=finish(sb,w[0],p);
}
} // namespace ari_shoup
#include <cuda_runtime.h>
namespace ari_shoup {
constexpr unsigned warp_size = 32, reindex_rows = 8, group_primes = 8;
// Bit reversal is an involution. A single warp owns each pair (i,r) with
// r>i, and lane j swaps one word of that pair; no block shares any address.
__global__ void reindex(uint32_t *a,uint32_t *b,unsigned n,unsigned l,unsigned bits) {
    const unsigned i=blockIdx.x*reindex_rows+threadIdx.y;
    if(i>=n)return;
    const unsigned r=__brev(i)>>(32-bits);
    if(r<=i)return;
    for(unsigned j=threadIdx.x;j<l;j+=warp_size) {
        auto u=std::size_t(i)*l+j,v=std::size_t(r)*l+j;
        uint32_t ai=a[u],ar=a[v],bi=b[u],br=b[v];
        a[u]=ar;a[v]=ai;b[u]=br;b[v]=bi;
    }
}
// Per block: 32 adjacent bit-reversed positions and 8 primes. Once the input
// tile has been staged, advancing through its words uses a conflict-free odd
// stride. Twists are permuted at setup, solely from point parameters.
__global__ void encode_tiled(const uint32_t *ar,const uint32_t *br,uint32_t *ra,uint32_t *rb,
                             const ari::Prime *primes,const Weight *weights,
                             const uint32_t *reversed_twist,unsigned n,unsigned l,unsigned count) {
    extern __shared__ uint32_t stage[];
    const unsigned stride=l|1u;
    uint32_t *sa=stage, *sb=stage+warp_size*stride;
    const unsigned x=threadIdx.x,y=threadIdx.y;
    const unsigned start=blockIdx.x*warp_size;
    const unsigned valid=(n-start<warp_size)?n-start:warp_size;
    for(unsigned t=y*warp_size+x;t<valid*l;t+=warp_size*group_primes) {
        unsigned row=t/l,col=t-row*l;
        std::size_t off=std::size_t(start)*l+t;
        sa[row*stride+col]=ar[off]; sb[row*stride+col]=br[off];
    }
    __syncthreads();
    const unsigned k=blockIdx.y*group_primes+y;
    if(k>=count||x>=valid)return;
    const auto p=primes[k];
    const std::size_t off=std::size_t(k)*n+start+x;
    uint32_t a,b;
    convert_both(sa+x*stride,sb+x*stride,weights+std::size_t(k)*l,l,p.p,a,b);
    uint32_t twist=reversed_twist[off];
    ra[off]=ari::mont(a,twist,p);
    rb[off]=ari::mont(b,twist,p);
}
inline std::vector<uint32_t> reorder_twists(const std::vector<uint32_t>& tw,
                                             unsigned n,unsigned bits) {
    std::vector<uint32_t> ret(tw.size());
    for(std::size_t base=0;base<tw.size();base+=n)
        for(unsigned i=0;i<n;++i)ret[base+ari::reverse_bits(i,bits)]=tw[base+i];
    return ret;
}
inline cudaError_t launch(uint32_t *a,uint32_t *b,
                          uint32_t *ra,uint32_t *rb,const ari::Prime *ps,const Weight *weights,
                          const uint32_t *twist_reversed,unsigned n,unsigned l,unsigned bits,
                          unsigned count,cudaStream_t stream=nullptr) {
    if(n<2||n>65536||(n&(n-1))||!l||l>55||!count) return cudaErrorInvalidValue;
    reindex<<<(n+reindex_rows-1)/reindex_rows,dim3(warp_size,reindex_rows),0,stream>>>(a,b,n,l,bits);
    cudaError_t st=cudaGetLastError();
    if(st!=cudaSuccess)return st;
    size_t shared_bytes=2*warp_size*(l|1u)*sizeof(uint32_t);
    encode_tiled<<<dim3((n+warp_size-1)/warp_size,(count+group_primes-1)/group_primes),
        dim3(warp_size,group_primes),shared_bytes,stream>>>(a,b,ra,rb,ps,weights,twist_reversed,n,l,count);
    return cudaGetLastError();
}
} // namespace ari_shoup
#include <cstdint>
#include <vector>

// WMMA B is column-major in each 16x16 panel. Columns are RNS primes.
// The host writes only the active values; zero-filled padding contributes
// nothing to the dot products on the final prime and limb tiles.
namespace ari_encoder_setup {
struct Weights {
    std::vector<uint8_t> panels;
    std::vector<ari_shoup::Weight> radix;
};
inline Weights make_weights(const std::vector<ari::Prime>& primes, unsigned limbs) {
    const unsigned k_pad = (primes.size() + 15u) / 16u * 16u;
    const unsigned l_pad = (limbs + 15u) / 16u * 16u;
    Weights ret{std::vector<uint8_t>(size_t(4) * k_pad * l_pad), {}};
    ret.radix.reserve(2 * primes.size());
    for (unsigned k = 0; k < primes.size(); ++k) {
        const auto& prime = primes[k];
        ret.radix.push_back(ari_shoup::make_weight(prime.one, prime.p));
        ret.radix.push_back(ari_shoup::make_weight(prime.r2, prime.p));
        uint32_t weight = prime.one;
        for (unsigned j = 0; j < limbs; ++j) {
            for (unsigned d = 0; d < 4; ++d) {
                const size_t off = ((size_t(d) * (l_pad / 16) + j / 16) * k_pad + k) * 16 + j % 16;
                ret.panels[off] = uint8_t(weight >> (8 * d));
            }
            weight = uint32_t(uint64_t(weight) * prime.one % prime.p);
        }
    }
    return ret;
}
} // namespace ari_encoder_setup

#include <cstddef>
#include <cstdint>

namespace ari_wmma_encoder {
struct WideDot {
    std::uint32_t s0, s1, s2;
};
FHD inline unsigned round16(unsigned n) { return (n + 15u) & ~15u; }
FHD inline std::size_t word_offset(unsigned digit,unsigned row,unsigned limb,
                                     unsigned padded_rows,unsigned padded_limbs) {
    return ((std::size_t(digit)*(padded_limbs/16)+limb/16)*padded_rows+row)*16+limb%16;
}
// Exact dot < 55*(2^32-1)*(2^31-1) < 2^69. Separate at the 32-bit
// boundary instead of assembling all seven base-256 positions in uint64.
FHD inline WideDot split_words(std::uint32_t d0,std::uint32_t d1,std::uint32_t d2,
    std::uint32_t d3,std::uint32_t d4,std::uint32_t d5,std::uint32_t d6) {
    const std::uint64_t low=d0+(std::uint64_t(d1)<<8)+(std::uint64_t(d2)<<16)
                           +(std::uint64_t(d3)<<24);
    const std::uint64_t high=d4+(std::uint64_t(d5)<<8)+(std::uint64_t(d6)<<16)+(low>>32);
    return {std::uint32_t(low),std::uint32_t(high),std::uint32_t(high>>32)};
}
FHD inline std::uint32_t reduce_words(std::uint32_t s0,std::uint32_t s1,
    std::uint32_t s2,const ari::Prime& p,const ari_shoup::Weight* radix_two) {
    // radix_two = Shoup precomputations for 2^32 and 2^64 modulo p.
    // All p are >2^30, so reducing s0 takes at most three subtractions.
    std::uint32_t a=ari_shoup::canonical_word(s0,p.p);
    std::uint32_t b=ari_shoup::lazy_product(s1,radix_two[0],p.p);
    std::uint32_t c=ari_shoup::lazy_product(s2,radix_two[1],p.p);
    if(b>=p.p)b-=p.p;
    if(c>=p.p)c-=p.p;
    return ari::add(ari::add(a,b,p.p),c,p.p);
}
FHD inline std::uint32_t reduce_diagonals(const std::uint32_t d[7],const ari::Prime& p,
                                            const ari_shoup::Weight* radix_two) {
    const WideDot s=split_words(d[0],d[1],d[2],d[3],d[4],d[5],d[6]);
    return reduce_words(s.s0,s.s1,s.s2,p,radix_two);
}
FHD inline std::uint32_t twist_residue(std::uint32_t xR,std::uint32_t twist,
                                        const ari::Prime& p) {
    return ari::mont(xR,twist,p);
}
} // namespace ari_wmma_encoder

#include <cuda_runtime.h>
#include <mma.h>

namespace ari_wmma_encoder {
namespace wmma=nvcuda::wmma;
using EncAcc=wmma::fragment<wmma::accumulator,16,16,16,int>;
using EncA=wmma::fragment<wmma::matrix_a,16,16,16,unsigned char,wmma::row_major>;
using EncB=wmma::fragment<wmma::matrix_b,16,16,16,unsigned char,wmma::col_major>;

// Reorder coefficients by the NTT's initial bit permutation while packing;
// the input remains unmodified. Padding is rewritten each run, not assumed.
__global__ void pack_operand(const std::uint32_t* __restrict__ input,
    std::uint8_t* __restrict__ dst,unsigned n,unsigned l,unsigned bits,unsigned operand) {
    constexpr unsigned rows_per_cta=8;
    const unsigned row=blockIdx.x*rows_per_cta+threadIdx.x/16;
    const unsigned limb=blockIdx.y*16+threadIdx.x%16;
    const unsigned np=round16(n),lp=round16(l);
    if(row>=np)return;
    std::uint32_t v=0;
    if(row<n&&limb<l) {
        const unsigned original=__brev(row)>>(32-bits);
        v=input[std::size_t(original)*l+limb];
    }
    const unsigned out_row=operand*np+row;
    #pragma unroll
    for(unsigned d=0;d<4;++d)
        dst[word_offset(d,out_row,limb,2*np,lp)]=std::uint8_t(v>>(8*d));
}

// Each warp produces 16 (already bit-reversed) coefficients x 16 primes.
// Store accumulator fragments column-major in shared so the modulus/twist
// epilogue writes neighbouring coefficients to the same prime plane.
__global__ void encode_gemm(const std::uint8_t* __restrict__ a,
    const std::uint8_t* __restrict__ weights,std::uint32_t* __restrict__ out,
    const ari::Prime* __restrict__ primes,const ari_shoup::Weight* __restrict__ radix,
    const std::uint32_t* __restrict__ twist,unsigned n,unsigned l,unsigned count,unsigned operand) {
    constexpr unsigned warps=4;
    const unsigned lane=threadIdx.x%32,warp=threadIdx.x/32;
    const unsigned r0=blockIdx.x*16,k0=(blockIdx.y*warps+warp)*16;
    if(k0>=count)return;  // uniform within warp, no CTA-wide barriers
    const unsigned np=round16(n),kp=round16(count),lp=round16(l);
    EncAcc diag[7];
    #pragma unroll
    for(unsigned d=0;d<7;++d)wmma::fill_fragment(diag[d],0);
    for(unsigned j=0;j<lp;j+=16) {
        EncA af[4];EncB bf[4];
        #pragma unroll
        for(unsigned d=0;d<4;++d) {
            wmma::load_matrix_sync(af[d],a+word_offset(d,operand*np+r0,j,2*np,lp),16);
            wmma::load_matrix_sync(bf[d],weights+word_offset(d,k0,j,kp,lp),16);
        }
        #pragma unroll
        for(unsigned u=0;u<4;++u) {
            #pragma unroll
            for(unsigned v=0;v<4;++v)wmma::mma_sync(diag[u+v],af[u],bf[v],diag[u+v]);
        }
    }
    // diag[d] <= 4*55*255^2 = 14,305,500 < INT32_MAX.
    #pragma unroll
    for(int t=0;t<EncAcc::num_elements;++t) {
        const WideDot s=split_words(diag[0].x[t],diag[1].x[t],diag[2].x[t],
            diag[3].x[t],diag[4].x[t],diag[5].x[t],diag[6].x[t]);
        diag[0].x[t]=static_cast<int>(s.s0);
        diag[1].x[t]=static_cast<int>(s.s1);
        diag[2].x[t]=static_cast<int>(s.s2);
    }
    __shared__ __align__(32) int words[warps][3][256];
    #pragma unroll
    for(unsigned d=0;d<3;++d)
        wmma::store_matrix_sync(words[warp][d],diag[d],16,wmma::mem_col_major);
    __syncwarp();
    #pragma unroll
    for(unsigned elem=lane;elem<256;elem+=32) {
        const unsigned k=k0+elem/16,r=r0+elem%16;
        if(k<count&&r<n) {
            const auto p=primes[k];
            const auto xR=reduce_words(std::uint32_t(words[warp][0][elem]),
                std::uint32_t(words[warp][1][elem]),std::uint32_t(words[warp][2][elem]),
                p,radix+2*k);
            const std::size_t at=std::size_t(k)*n+r;
            out[at]=twist_residue(xR,twist[at],p);
        }
    }
}
// packed_inputs needs 4 * (2*round16(n)) * round16(l) bytes. packed_weights
// needs 4 * round16(count) * round16(l) bytes, zeroed outside its true shape.
inline cudaError_t launch_operand(const std::uint32_t *input,std::uint32_t *residues,
    unsigned operand,std::uint8_t *packed_inputs,
    const std::uint8_t *packed_weights,const ari::Prime *primes,
    const ari_shoup::Weight *radix_shoup,const std::uint32_t *twist_reversed,
    unsigned n,unsigned limbs,unsigned log_n,unsigned count,cudaStream_t stream=nullptr) {
    if(n<2||n>65536||(n&(n-1))||!limbs||limbs>55||!count||
       log_n>16||(1u<<log_n)!=n||operand>1)return cudaErrorInvalidValue;
    // Each branch reads/writes only its own N padded rows in every panel.
    pack_operand<<<dim3(round16(n)/8,round16(limbs)/16),128,0,stream>>>
        (input,packed_inputs,n,limbs,log_n,operand);
    auto status=cudaGetLastError();if(status!=cudaSuccess)return status;
    encode_gemm<<<dim3(round16(n)/16,(count+63)/64),128,0,stream>>>
       (packed_inputs,packed_weights,residues,primes,radix_shoup,twist_reversed,n,limbs,count,operand);
    return cudaGetLastError();
}
} // namespace ari_wmma_encoder

#ifndef FHERMA_NTT_TILED
#define FHERMA_NTT_TILED 1
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace ari_staggered {
using Address=std::uintptr_t;
struct Range {Address begin=0,end=0;};
inline Address host_page_size(){
    const long size=::sysconf(_SC_PAGESIZE);
    if(size<=0 || (size&(size-1)))throw std::runtime_error("invalid host page size");
    return static_cast<Address>(size);
}
inline void cuda_check(cudaError_t status,const char* action){
    if(status!=cudaSuccess)throw std::runtime_error(std::string(action)+": "+cudaGetErrorString(status));
}
// At most 1, then 2, then 4 new intervals can be needed when registering
// three contiguous buffers. Entries stay separate until unregister, since the
// runtime requires the exact original base address for each registration.
class Registrations {
    std::array<Range,7> entries{};
    unsigned used=0;
    Address page;
    static Address byte_end(Address address,size_t size){
        if(!address||!size||size>std::numeric_limits<Address>::max()-address)
            throw std::invalid_argument("invalid registered buffer");
        return address+size;
    }
    void acquire_gap(Address begin,Address end){
        if(begin==end)return;
        if(used>=entries.size())throw std::logic_error("too many registration segments");
        cuda_check(cudaHostRegister(reinterpret_cast<void*>(begin),end-begin,cudaHostRegisterDefault),
                   "register buffer segment");
        // No allocating or throwing operation follows acquisition.
        entries[used++]={begin,end};
    }
    bool release_one(Range& entry,cudaError_t& first_error){
        if(!entry.begin)return true;
        const auto status=cudaHostUnregister(reinterpret_cast<void*>(entry.begin));
        if(status==cudaSuccess){entry={};return true;}
        if(first_error==cudaSuccess)first_error=status;
        return false;
    }
public:
    explicit Registrations(Address bytes_per_page):page(bytes_per_page){}
    Registrations(const Registrations&)=delete;
    ~Registrations(){
        cudaError_t ignored=cudaSuccess;
        for(unsigned i=0;i<used;++i)release_one(entries[i],ignored);
    }
    Range acquire(const void* pointer,size_t bytes){
        const auto address=reinterpret_cast<Address>(pointer);
        const auto end=byte_end(address,bytes);
        if(end>std::numeric_limits<Address>::max()-(page-1))
            throw std::invalid_argument("host page bound overflow");
        const Range target{address&~(page-1),(end+page-1)&~(page-1)};
        const unsigned old_used=used;
        auto cursor=target.begin;
        for(unsigned i=0;i<old_used && cursor<target.end;++i){
            const auto held=entries[i];
            if(held.end<=cursor)continue;
            if(held.begin>=target.end)break;
            if(held.begin>cursor)acquire_gap(cursor,held.begin);
            cursor=std::max(cursor,held.end);
        }
        if(cursor<target.end)acquire_gap(cursor,target.end);
        std::sort(entries.begin(),entries.begin()+used,[](Range a,Range b){return a.begin<b.begin;});
        return target;
    }
    // Issue separate DMA operations if this logical buffer crosses two
    // independently registered regions. The normal case is a single DMA.
    template<class Copy>
    void for_chunks(const void* pointer,size_t bytes,Copy&& copy)const{
        const auto start=reinterpret_cast<Address>(pointer);
        const auto end=byte_end(start,bytes);
        auto cursor=start;
        for(unsigned i=0;i<used && cursor<end;++i){
            const auto held=entries[i];
            if(held.end<=cursor)continue;
            if(held.begin>cursor)break;
            const auto stop=std::min(held.end,end);
            copy(cursor-start,stop-cursor);
            cursor=stop;
        }
        if(cursor!=end)throw std::logic_error("DMA range has missing registration");
    }
    void release_inputs(Range output){
        cudaError_t first=cudaSuccess;
        for(unsigned i=0;i<used;++i){
            auto& held=entries[i];
            if(held.end<=output.begin || held.begin>=output.end)release_one(held,first);
        }
        cuda_check(first,"unregister input segments");
    }
    void release_all(){
        cudaError_t first=cudaSuccess;
        for(unsigned i=0;i<used;++i)release_one(entries[i],first);
        cuda_check(first,"unregister remaining segments");
    }
};
} // namespace ari_staggered

namespace {
void check(cudaError_t e,const char* where) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(e));
}
template<class T> struct Buffer {
    T* ptr=nullptr;
    Buffer()=default; Buffer(const Buffer&)=delete;
    ~Buffer(){if(ptr)cudaFree(ptr);}
    void alloc(size_t n){check(cudaMalloc(reinterpret_cast<void**>(&ptr),n*sizeof(T)),"cudaMalloc");}
    void copy(const std::vector<T>& x){alloc(x.size());check(cudaMemcpy(ptr,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice),"setup upload");}
};
struct OperandStream {
    cudaStream_t handle=nullptr;
    OperandStream(){
        check(cudaStreamCreateWithFlags(&handle,cudaStreamNonBlocking),"create A stream");
    }
    OperandStream(const OperandStream&)=delete;
    ~OperandStream(){if(handle)cudaStreamDestroy(handle);}
};
struct ReadyEvent {
    cudaEvent_t handle=nullptr;
    ReadyEvent(){
        check(cudaEventCreateWithFlags(&handle,cudaEventDisableTiming),"create pipeline event");
    }
    ReadyEvent(const ReadyEvent&)=delete;
    ~ReadyEvent(){if(handle)cudaEventDestroy(handle);}
};
// The normal output stream joins A before pointwise multiplication. Before
// that join, a partial failure may have submitted work on either stream.
struct RunFence {
    bool completed=false;
    cudaStream_t operand_a;
    explicit RunFence(cudaStream_t a):operand_a(a){}
    RunFence(const RunFence&)=delete;
    ~RunFence(){
        if(!completed){
            cudaStreamSynchronize(operand_a);
            cudaStreamSynchronize(nullptr);
        }
    }
    void complete(){
        check(cudaStreamSynchronize(nullptr),"complete measured work");
        check(cudaStreamSynchronize(operand_a),"check A stream completion");
        completed=true;
    }
};
__global__ void multiply_reverse(const uint32_t* a,const uint32_t* b,uint32_t* c,
                                 const ari::Prime* primes,unsigned N,unsigned logN) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x,k=blockIdx.y;if(i>=N)return;
    const size_t at=size_t(k)*N+i;
    c[size_t(k)*N+ari::reverse_bits(i,logN)]=ari::mont(a[at],b[at],primes[k]);
}
#if !FHERMA_NTT_TILED
__global__ void single_stage(uint32_t* a,const ari::Prime* primes,const uint32_t* roots,
                              unsigned N,unsigned half) {
    const unsigned t=blockIdx.x*blockDim.x+threadIdx.x,k=blockIdx.y;if(t>=N/2)return;
    const unsigned j=t%half,i=(t/half)*(2*half)+j; const size_t base=size_t(k)*N;
    const auto p=primes[k];const auto w=roots[size_t(k)*(N/2)+j*(N/(2*half))];
    uint32_t x=a[base+i],y=a[base+i+half];ari::butterfly(x,y,w,p);a[base+i]=x;a[base+i+half]=y;
}
#endif
__global__ void decode(uint32_t* a,const ari::Prime* primes,const uint32_t* untwist,unsigned N) {
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x,k=blockIdx.y;if(i>=N)return;
    const size_t at=size_t(k)*N+i;const auto p=primes[k];
    a[at]=ari::mont(ari::mont(a[at],untwist[at],p),1,p);
}
// Capability diagnostics only. A missing optional attribute is logged as -1.
void report_host_access_capabilities(){
    int device=-1;
    if(cudaGetDevice(&device)!=cudaSuccess){
        (void)cudaGetLastError();
        std::fprintf(stderr,"ARI_HOST_ACCESS device=-1\n");
        return;
    }
    auto attribute=[device](cudaDeviceAttr attr){
        int value=-1;
        if(cudaDeviceGetAttribute(&value,attr,device)!=cudaSuccess){
            (void)cudaGetLastError();
            return -1;
        }
        return value;
    };
    const int pageable=attribute(cudaDevAttrPageableMemoryAccess);
    const int host_tables=attribute(cudaDevAttrPageableMemoryAccessUsesHostPageTables);
    const int concurrent=attribute(cudaDevAttrConcurrentManagedAccess);
    const int registered_pointer=attribute(cudaDevAttrCanUseHostPointerForRegisteredMem);
    const int uva=attribute(cudaDevAttrUnifiedAddressing);
    std::fprintf(stderr,"ARI_HOST_ACCESS device=%d pageableMemoryAccess=%d "
        "pageableMemoryAccessUsesHostPageTables=%d concurrentManagedAccess=%d "
        "canUseHostPointerForRegisteredMem=%d unifiedAddressing=%d\n",
        device,pageable,host_tables,concurrent,registered_pointer,uva);
}
struct State {
    ari::Plan plan;
    OperandStream stream_a;
    ReadyEvent forward_a,uploaded[2];
    const ari_staggered::Address page_size=ari_staggered::host_page_size();
    Buffer<uint32_t> a,b,c,ra,rb,rc,roots,inverse_roots,twist,untwist;
    Buffer<ari::Prime> primes;
    Buffer<uint8_t> packed_inputs, packed_weights;
    Buffer<ari_shoup::Weight> radix;
    std::unique_ptr<ari_tensor::DevicePlan> crt;
    explicit State(const fherma::Point& p):plan(p){}
};
} // namespace

void* fherma_init(const fherma::Point& p) {
    auto s=std::make_unique<State>(p);
    const size_t words=size_t(p.N)*p.L, residues=size_t(p.N)*s->plan.K;
    s->a.alloc(words);s->b.alloc(words);s->c.alloc(words);
    s->ra.alloc(residues);s->rb.alloc(residues);s->rc.alloc(residues);
    const unsigned n_pad=(p.N+15u)/16u*16u, l_pad=(p.L+15u)/16u*16u;
    s->packed_inputs.alloc(size_t(8)*n_pad*l_pad);
    const auto weights=ari_encoder_setup::make_weights(s->plan.primes,p.L);
    s->packed_weights.copy(weights.panels);
    s->radix.copy(weights.radix);
    s->primes.copy(s->plan.primes);s->roots.copy(s->plan.roots);
    s->inverse_roots.copy(s->plan.inverse_roots);s->twist.copy(ari_shoup::reorder_twists(s->plan.twist,p.N,s->plan.logN));s->untwist.copy(s->plan.untwist);
    std::vector<uint32_t> primes;for(auto prime:s->plan.primes)primes.push_back(prime.p);
    s->crt=std::make_unique<ari_tensor::DevicePlan>(p.N,p.q.data.data(),p.L,primes);
    check(cudaDeviceSynchronize(),"initialize GPU");
    report_host_access_capabilities();
    return s.release();
}

fherma::Outputs fherma_run(void* state,const fherma::Inputs& in) {
    auto& s=*static_cast<State*>(state);const auto& p=s.plan;
    const size_t words=size_t(p.N)*p.L,bytes=words*sizeof(uint32_t);
    if(in.a.data.size()!=words||in.b.data.size()!=words)throw std::runtime_error("input size does not match point");
    // Reverse lifetime order on exceptions: drain streams, unregister all
    // ranges, then free the caller-owned result's ordinary vector storage.
    fherma::Outputs out;out.c.shape={int64_t(p.N),int64_t(p.L)};
    ari_staggered::Registrations registered(s.page_size);
    RunFence fence(s.stream_a.handle);
    const dim3 grid((p.N+255)/256,p.K);
#if !FHERMA_NTT_TILED
    const dim3 butterflies((p.N/2+255)/256,p.K);
#endif
    auto submit_operand=[&](unsigned operand,const uint32_t* input,cudaStream_t stream){
        uint32_t* device=operand?s.b.ptr:s.a.ptr;
        uint32_t* residues=operand?s.rb.ptr:s.ra.ptr;
        registered.acquire(input,bytes);
        registered.for_chunks(input,bytes,[&](size_t offset,size_t amount){
            auto* dst=reinterpret_cast<unsigned char*>(device)+offset;
            const auto* src=reinterpret_cast<const unsigned char*>(input)+offset;
            check(cudaMemcpyAsync(dst,src,amount,cudaMemcpyHostToDevice,stream),"upload registered input");
        });
        check(cudaEventRecord(s.uploaded[operand].handle,stream),"mark last H2D read");
        check(ari_wmma_encoder::launch_operand(device,residues,operand,s.packed_inputs.ptr,
            s.packed_weights.ptr,s.primes.ptr,s.radix.ptr,s.twist.ptr,
            p.N,p.L,p.logN,p.K,stream),"encode operand");
#if FHERMA_NTT_TILED
        // The template selects the number of arrays. The roots select direction.
        check(ari_tile::transform<false>(residues,nullptr,s.primes.ptr,s.roots.ptr,
                                        p.N,p.K,stream),"single-operand forward NTT");
#else
        for(unsigned half=1;half<p.N;half*=2){
            single_stage<<<butterflies,256,0,stream>>>(residues,s.primes.ptr,s.roots.ptr,p.N,half);
            check(cudaGetLastError(),"single-operand forward stage");
        }
#endif
    };
    submit_operand(0,in.a.data.data(),s.stream_a.handle);
    check(cudaEventRecord(s.forward_a.handle,s.stream_a.handle),"mark A transform complete");
    // Register B only after the entire A branch has been submitted.
    submit_operand(1,in.b.data.data(),nullptr);
    check(cudaStreamWaitEvent(nullptr,s.forward_a.handle,0),"join A and B before product");
    multiply_reverse<<<grid,256>>>(s.ra.ptr,s.rb.ptr,s.rc.ptr,s.primes.ptr,p.N,p.logN);
    check(cudaGetLastError(),"multiply launch");
#if FHERMA_NTT_TILED
    check(ari_tile::transform<false>(s.rc.ptr,nullptr,s.primes.ptr,s.inverse_roots.ptr,p.N,p.K),"tiled inverse NTT");
#else
    for(unsigned half=1;half<p.N;half*=2) {
        single_stage<<<butterflies,256>>>(s.rc.ptr,s.primes.ptr,s.inverse_roots.ptr,p.N,half);
        check(cudaGetLastError(),"inverse NTT launch");
    }
#endif
    decode<<<grid,256>>>(s.rc.ptr,s.primes.ptr,s.untwist.ptr,p.N);
    check(cudaGetLastError(),"decode launch");
    check(s.crt->run(s.rc.ptr,s.c.ptr),"CRT launch");
    // Both inputs, transforms and CRT have been enqueued. Prepare the final
    // vector while GPU work may still be running. Every word is constructed.
    out.c.data.resize(words);
    const auto output_pages=registered.acquire(out.c.data.data(),bytes);
    registered.for_chunks(out.c.data.data(),bytes,[&](size_t offset,size_t amount){
        auto* dst=reinterpret_cast<unsigned char*>(out.c.data.data())+offset;
        const auto* src=reinterpret_cast<const unsigned char*>(s.c.ptr)+offset;
        check(cudaMemcpyAsync(dst,src,amount,cudaMemcpyDeviceToHost,nullptr),"download registered output");
    });
    // Only the completed H2D copies read the borrowed inputs. No later
    // kernel uses those pages. Keep any registration shared with C until
    // the final D2H completes, even if C uses only a fragment of its range.
    check(cudaEventSynchronize(s.uploaded[0].handle),"wait for A host read");
    check(cudaEventSynchronize(s.uploaded[1].handle),"wait for B host read");
    registered.release_inputs(output_pages);
    fence.complete();
    registered.release_all();
    return out;
}
void fherma_free(void* state) {delete static_cast<State*>(state);}

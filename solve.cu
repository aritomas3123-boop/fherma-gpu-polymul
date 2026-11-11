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
#include <mma.h>
#ifndef FHERMA_TENSOR_CRT
#define FHERMA_TENSOR_CRT 0
#endif

namespace fherma_crt {
// Prime-major input [p][coefficient], coefficient-major canonical output
// [coefficient][limb]. Threads in a row read neighbouring coefficients.
constexpr unsigned tile_x = 8;
constexpr unsigned tile_y = 16;

struct DeviceArgs {
    unsigned n, limbs, bits, count;
    u32 gap;
    PrimeInfo *primes;
    u32 *weights, *q;
};
__global__ void reconstruct_kernel(DeviceArgs plan, const u32 *residues, u32 *out) {
    extern __shared__ unsigned long long shared[];
    unsigned x = threadIdx.x, y = threadIdx.y;
    unsigned coef = blockIdx.x * tile_x + x;
    unsigned k = plan.count, l = plan.limbs, pitch = k + 1;
    u64 *low = reinterpret_cast<u64 *>(shared);
    u64 *high = low + l * tile_x;
    u32 *terms = reinterpret_cast<u32 *>(high + l * tile_x);
    u32 *digits = terms + pitch * tile_x;
    bool active = coef < plan.n;
    for (unsigned j = y; j < k; j += tile_y) {
        u32 r = active ? residues[std::size_t(j) * plan.n + coef] : 0;
        terms[j * tile_x + x] = weight_residue(r, plan.primes[j]);
    }
    __syncthreads();
    if (y == 0) {
        terms[k * tile_x + x] = correction<tile_x>(terms + x, plan.primes, k);
    }
    __syncthreads();
    for (unsigned i = y; i < l; i += tile_y) {
        sum_column<tile_x>(terms + x, plan.weights + i * pitch, pitch,
            low[i * tile_x + x], high[i * tile_x + x]);
    }
    __syncthreads();
    if (y == 0 && active) {
        carry_and_reduce<tile_x>(digits + x, low + x, high + x, l,
            plan.bits, plan.gap, plan.q);
        for (unsigned i = 0; i < l; ++i)
            out[std::size_t(coef) * l + i] = digits[i * tile_x + x];
    }
}


#if FHERMA_TENSOR_CRT
// New architectural experiment: one CTA owns 16 adjacent coefficients across
// all L limbs. Four Tensor-Core WMMA warps each process 16 output limbs.
// The original scalar CRT remains available when FHERMA_TENSOR_CRT=0.
// NOT A PROVEN PERFORMANCE IMPROVEMENT; exactness must be checked on FHERMA.
namespace crt_wmma = nvcuda::wmma;
constexpr unsigned tc_rows=16,tc_warps=4,tc_tile=16,tc_diagonals=7;
using TcA=crt_wmma::fragment<crt_wmma::matrix_a,16,16,16,unsigned char,crt_wmma::row_major>;
using TcB=crt_wmma::fragment<crt_wmma::matrix_b,16,16,16,unsigned char,crt_wmma::col_major>;
using TcAcc=crt_wmma::fragment<crt_wmma::accumulator,16,16,16,int>;
inline std::size_t tensor_shared_bytes(unsigned k,unsigned l) {
    const unsigned pitch=k+1;
    return std::size_t(tc_rows)*pitch*sizeof(u32)
          +4*256 /* A planes */
          +tc_warps*4*256 /* B planes */
          +tc_warps*tc_diagonals*256*sizeof(int) /* partials */
          +2*tc_rows*l*sizeof(u64)
          +tc_rows*(l+2)*sizeof(u32);
}
__global__ void reconstruct_tensor_kernel(DeviceArgs plan,const u32 *residues,u32 *out) {
    extern __shared__ unsigned char scratch[];
    const unsigned tid=threadIdx.x, warp=tid/32, lane=tid%32;
    const unsigned n=plan.n,k=plan.count,l=plan.limbs,pitch=k+1;
    const unsigned base=blockIdx.x*tc_rows;
    unsigned char *ptr=scratch;
    u32 *terms=reinterpret_cast<u32*>(ptr);
    ptr+=std::size_t(tc_rows)*pitch*sizeof(u32);
    unsigned char *planeA=ptr; ptr+=4*256;
    unsigned char *planeB=ptr; ptr+=tc_warps*4*256;
    int *partials=reinterpret_cast<int*>(ptr);
    ptr+=tc_warps*tc_diagonals*256*sizeof(int);
    u64 *low=reinterpret_cast<u64*>(ptr);
    ptr+=std::size_t(tc_rows)*l*sizeof(u64);
    u64 *high=reinterpret_cast<u64*>(ptr);
    ptr+=std::size_t(tc_rows)*l*sizeof(u64);
    u32 *digits=reinterpret_cast<u32*>(ptr);
    // Build canonical weighted CRT terms once per 16-coefficient tile.
    for(unsigned idx=tid;idx<tc_rows*k;idx+=blockDim.x) {
        unsigned row=idx/k,j=idx%k,coef=base+row;
        const u32 residue=coef<n?residues[std::size_t(j)*n+coef]:0;
        terms[row*pitch+j]=weight_residue(residue,plan.primes[j]);
    }
    __syncthreads();
    if(tid<tc_rows){
        terms[tid*pitch+k]=correction<1>(terms+tid*pitch,plan.primes,k);
    }
    __syncthreads();
    TcAcc acc[tc_diagonals];
    #pragma unroll
    for(unsigned d=0;d<tc_diagonals;d++)crt_wmma::fill_fragment(acc[d],0);
    const unsigned limbBase=warp*tc_tile;
    for(unsigned start=0;start<pitch;start+=16){
        // Every element of the A-plane is shared by four independent warps.
        for(unsigned index=tid;index<256;index+=blockDim.x){
            unsigned row=index/16,j=index%16;
            u32 val=start+j<pitch?terms[row*pitch+start+j]:0;
            #pragma unroll
            for(unsigned d=0;d<4;d++)
                planeA[d*256+index]=static_cast<unsigned char>(val>>(8*d));
        }
        // B is column-major: offset column*16+row, each warp owns 16 limbs.
        for(unsigned index=lane;index<256;index+=32){
            unsigned col=index/16,j=index%16;
            unsigned limb=limbBase+col;
            u32 val=(limb<l && start+j<pitch)?
                     plan.weights[std::size_t(limb)*pitch+start+j]:0;
            #pragma unroll
            for(unsigned d=0;d<4;d++)
                planeB[(warp*4+d)*256+index]=static_cast<unsigned char>(val>>(8*d));
        }
        __syncthreads();
        TcA af[4];TcB bf[4];
        #pragma unroll
        for(unsigned d=0;d<4;d++){
            crt_wmma::load_matrix_sync(af[d],planeA+d*256,16);
            crt_wmma::load_matrix_sync(bf[d],planeB+(warp*4+d)*256,16);
        }
        #pragma unroll
        for(unsigned u=0;u<4;u++){
            #pragma unroll
            for(unsigned v=0;v<4;v++){
                crt_wmma::mma_sync(acc[u+v],af[u],bf[v],acc[u+v]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(unsigned d=0;d<tc_diagonals;d++){
        crt_wmma::store_matrix_sync(partials+(warp*tc_diagonals+d)*256,
                                    acc[d],16,crt_wmma::mem_row_major);
    }
    __syncthreads();
    for(unsigned e=lane;e<256;e+=32){
        unsigned row=e/16,col=e%16,limb=limbBase+col;
        if(limb>=l)continue;
        // Exactly recombine seven base-256 diagonals without FP rounding.
        u64 carry=0;
        u32 words[3]{};
        for(unsigned d=0;d<tc_diagonals;d++){
            u64 t=carry+u32(partials[(warp*tc_diagonals+d)*256+e]);
            const unsigned shift=(d%4)*8;
            const unsigned wi=d/4;
            // Each diagonal is at most 4*119*255^2 < 2^25;
            // propagate into three complete u32 words.
            if(wi<3){
                u64 combined=(u64(words[wi])+(t<<shift));
                words[wi]=u32(combined);
                carry=combined>>32;
                if(wi+1<3)words[wi+1]+=u32(carry);
            }
        }
        // The above recombination is diagnostic candidate code; use exact
        // 96-bit accumulation below for deterministic carry propagation.
        u64 s0=0,s1=0,s2=0;
        for(unsigned d=0;d<tc_diagonals;d++){
            const u64 v=u32(partials[(warp*tc_diagonals+d)*256+e]);
            const unsigned bits=8*d;
            if(bits<32){
                s0+=v<<bits;
                if(bits && bits>8) s1+=v>>(32-bits);
            }else if(bits<64){
                s1+=v<<(bits-32);
            }else{
                s2+=v<<(bits-64);
            }
        }
        s1+=s0>>32;
        s2+=s1>>32;
        // no more than 119 products of two u32 words: total < 2^71
        low[row*l+limb]=u32(s0);
        high[row*l+limb]=u32(s1)+(s2<<32);
    }
    __syncthreads();
    if(tid<tc_rows && base+tid<n){
        carry_and_reduce<1>(digits+tid*(l+2),low+tid*l,high+tid*l,
                             l,plan.bits,plan.gap,plan.q);
        for(unsigned limb=0;limb<l;limb++)
           out[std::size_t(base+tid)*l+limb]=digits[tid*(l+2)+limb];
    }
}
#endif

class DevicePlan {
    DeviceArgs args_{};
    std::size_t shared_ = 0;
    template<class T> static void upload(T *&dst, const std::vector<T>& src) {
        std::size_t bytes = src.size() * sizeof(T);
        cudaError_t status = cudaMalloc(reinterpret_cast<void **>(&dst), bytes);
        if (status == cudaSuccess)
            status = cudaMemcpy(dst, src.data(), bytes, cudaMemcpyHostToDevice);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    }
    void release() noexcept {
        if (args_.primes) cudaFree(args_.primes);
        if (args_.weights) cudaFree(args_.weights);
        if (args_.q) cudaFree(args_.q);
    }
public:
    DevicePlan(unsigned n, const u32 *q, unsigned limbs, const std::vector<u32>& primes) {
        HostPlan plan = make_plan(n, q, limbs, primes);
        args_.n = n; args_.limbs = limbs; args_.bits = plan.bits;
        args_.count = unsigned(primes.size()); args_.gap = plan.gap;
#if FHERMA_TENSOR_CRT
        shared_ = tensor_shared_bytes(unsigned(primes.size()),limbs);
#else
        shared_ = tile_x * (2 * sizeof(u64) * limbs + sizeof(u32) * (primes.size() + 1 + limbs + 2));
#endif
        int device = 0;
        cudaDeviceProp prop{};
        cudaError_t status = cudaGetDevice(&device);
        if (status == cudaSuccess) status = cudaGetDeviceProperties(&prop, device);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        if (shared_ > prop.sharedMemPerBlock) {
            if (shared_ > prop.sharedMemPerBlockOptin)
                throw std::invalid_argument("CRT tile exceeds opt-in shared memory capacity");
#if FHERMA_TENSOR_CRT
            status=cudaFuncSetAttribute(reconstruct_tensor_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,static_cast<int>(shared_));
#else
            status=cudaFuncSetAttribute(reconstruct_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,static_cast<int>(shared_));
#endif
            if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
        }
        try { upload(args_.primes, plan.primes); upload(args_.weights, plan.weights); upload(args_.q, plan.q); }
        catch (...) { release(); throw; }
    }
    DevicePlan(const DevicePlan&) = delete;
    DevicePlan& operator=(const DevicePlan&) = delete;
    ~DevicePlan() { release(); }
    // Asynchronous, caller checks eventual execution status when consuming out.
    cudaError_t run(const u32 *residues, u32 *out, cudaStream_t stream = nullptr) const {
#if FHERMA_TENSOR_CRT
        reconstruct_tensor_kernel<<<(args_.n + tc_rows-1)/tc_rows,128,shared_,stream>>>(args_,residues,out);
#else
        reconstruct_kernel<<<(args_.n + tile_x - 1) / tile_x, dim3(tile_x, tile_y), shared_, stream>>>(args_, residues, out);
#endif
        return cudaGetLastError();
    }
};
} // namespace fherma_crt

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
#ifndef FHERMA_PINNED_INPUT
#define FHERMA_PINNED_INPUT 1
#endif
#ifndef FHERMA_PINNED_OUTPUT
#define FHERMA_PINNED_OUTPUT 1
#endif
static_assert(FHERMA_PINNED_INPUT && FHERMA_PINNED_OUTPUT,
              "tuned staging requires pinned input and output");

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
#include <atomic>
#if defined(__x86_64__) || defined(__i386__)
#include <emmintrin.h>
#endif
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#if defined(__linux__)
#include <sched.h>
#endif

#ifndef FHERMA_COPY_PARTICIPANTS
#define FHERMA_COPY_PARTICIPANTS 24
#endif
static_assert(FHERMA_COPY_PARTICIPANTS >= 1 && FHERMA_COPY_PARTICIPANTS <= 24,
              "expected 1..24 copy participants");
namespace ari_copy {
inline unsigned concurrency() {
    unsigned available=std::thread::hardware_concurrency();
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if(sched_getaffinity(0,sizeof(set),&set)==0)
        available=unsigned(CPU_COUNT(&set));
#endif
    return std::max(1u,std::min(unsigned(FHERMA_COPY_PARTICIPANTS),available));
}

// All public operations are serialized by fherma_run on this State. Workers
// cannot retain or touch borrowed pointers after copy() releases the caller.
class Team {
    std::mutex mu;
    std::condition_variable wake,done;
    std::vector<std::thread> workers;
    uint64_t epoch=0;
    unsigned pending=0;
    bool stopping=false;
    const unsigned participants;
    const uint32_t* src=nullptr;
    uint32_t* dst=nullptr;
    size_t words=0;
    bool wc_job=false;
    static void publish_wc(bool enabled){
        if(!enabled)return;
#if defined(__x86_64__) || defined(__i386__)
        _mm_sfence();
#else
        std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
    }
    static void piece(unsigned me,unsigned count,size_t n,uint32_t* d,const uint32_t* s){
        // Partition cache lines; only the last worker handles the tail.
        const size_t lines=n/16;
        const size_t lo=16*(lines*me/count);
        const size_t hi=me+1==count?n:16*(lines*(me+1)/count);
        if(hi>lo) std::memcpy(d+lo,s+lo,(hi-lo)*sizeof(uint32_t));
    }
    void loop(unsigned me){
        uint64_t seen=0;
        std::unique_lock<std::mutex> l(mu);
        for(;;){
            wake.wait(l,[&]{return stopping||epoch!=seen;});
            if(stopping)return;
            seen=epoch;
            auto s=src; auto d=dst; auto n=words;auto wc=wc_job;
            l.unlock();
            piece(me,participants,n,d,s);
            publish_wc(wc);
            l.lock();
            if(--pending==0)done.notify_one();
        }
    }
    void shutdown(){
        {std::lock_guard<std::mutex> l(mu);stopping=true;}
        wake.notify_all();
        for(auto& w:workers)if(w.joinable())w.join();
    }
public:
    explicit Team(unsigned count=concurrency()):participants(std::max(1u,count)){
        try{
            workers.reserve(participants-1);
            for(unsigned id=1;id<participants;++id)
                workers.emplace_back([this,id]{loop(id);});
        }catch(...){shutdown();throw;}
    }
    ~Team(){shutdown();}
    Team(const Team&)=delete;
    Team& operator=(const Team&)=delete;
    void copy(uint32_t* d,const uint32_t* s,size_t n,bool wc=false){
        if(!n)return;
        if(participants==1||n<(1u<<16)){
            std::memcpy(d,s,n*sizeof(uint32_t)); publish_wc(wc); return;
        }
        {
            std::lock_guard<std::mutex> l(mu);
            dst=d;src=s;words=n;wc_job=wc;pending=participants-1;++epoch;
        }
        wake.notify_all();
        piece(0,participants,n,d,s);
        publish_wc(wc);
        std::unique_lock<std::mutex> l(mu);
        done.wait(l,[&]{return pending==0;});
        dst=nullptr;src=nullptr;
    }
};
} // namespace ari_copy

#include <array>
#include <chrono>
#include <cstdio>
#include <system_error>

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
// Persistent page-locked staging belongs exclusively to State. Every copy of
// operand/result bytes still occurs inside the measured fherma_run call.
template<class T> struct HostBuffer {
    T* ptr=nullptr;
    HostBuffer()=default;
    HostBuffer(const HostBuffer&)=delete;
    HostBuffer& operator=(const HostBuffer&)=delete;
    ~HostBuffer(){if(ptr)cudaFreeHost(ptr);}
    void alloc(size_t n,unsigned flags=cudaHostAllocDefault){
        check(cudaHostAlloc(reinterpret_cast<void**>(&ptr),n*sizeof(T),flags),"cudaHostAlloc staging");
    }
};
struct OperandStream {
    cudaStream_t handle=nullptr;
    OperandStream(){
        check(cudaStreamCreateWithFlags(&handle,cudaStreamNonBlocking),"create A stream");
    }
    OperandStream(const OperandStream&)=delete;
    ~OperandStream(){if(handle)cudaStreamDestroy(handle);}
};
struct ForwardReady {
    cudaEvent_t handle=nullptr;
    ForwardReady(){
        check(cudaEventCreateWithFlags(&handle,cudaEventDisableTiming),"create A ready event");
    }
    ForwardReady(const ForwardReady&)=delete;
    ~ForwardReady(){if(handle)cudaEventDestroy(handle);}
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
struct State {
    ari::Plan plan;
    OperandStream stream_a;
    ForwardReady forward_a;
    Buffer<uint32_t> a,b,c,ra,rb,rc,roots,inverse_roots,twist,untwist;
    Buffer<ari::Prime> primes;
    Buffer<uint8_t> packed_inputs, packed_weights;
    Buffer<ari_shoup::Weight> radix;
#if FHERMA_PINNED_INPUT
    HostBuffer<uint32_t> host_a,host_b;
    bool input_wc=false;
    std::unique_ptr<ari_copy::Team> copy_team;
#endif
#if FHERMA_PINNED_OUTPUT
    HostBuffer<uint32_t> host_c;
#endif
    std::unique_ptr<fherma_crt::DevicePlan> crt;
    explicit State(const fherma::Point& p):plan(p){}
};
// Calibration touches only owned, synthetic point-sized storage. All runtime
// inputs, outputs, CUDA work and waits remain inside their normal call.
void choose_staging(State& s,size_t words){
    using Clock=std::chrono::steady_clock;
    const size_t bytes=words*sizeof(uint32_t);
    const auto available=ari_copy::concurrency();
    const auto stream=s.stream_a.handle;
    unsigned best_count=1;
    bool best_wc=false;
    double best_ms=std::numeric_limits<double>::infinity();
    {
        std::array<std::vector<uint32_t>,2> synthetic;
        uint32_t value=0x238f4a69u;
        for(auto& source:synthetic){
            source.resize(words);
            for(auto& word:source){value=value*1664525u+1013904223u;word=value;}
        }
        HostBuffer<uint32_t> cached,wc;
        cached.alloc(words,cudaHostAllocDefault);
        const auto wc_status=cudaHostAlloc(reinterpret_cast<void**>(&wc.ptr),bytes,cudaHostAllocWriteCombined);
        if(wc_status!=cudaSuccess){
            wc.ptr=nullptr;
            if(wc_status!=cudaErrorMemoryAllocation && wc_status!=cudaErrorNotSupported)
                check(wc_status,"allocate write-combined calibration buffer");
            (void)cudaGetLastError();
        }
        // Destroy this fence before freeing buffers on a failed setup DMA.
        RunFence fence(stream);
        unsigned previous=0;
        std::fprintf(stderr,"ARI_STAGING_TUNE bytes=%zu available=%u",bytes,available);
        for(unsigned requested:{1u,4u,8u,16u,24u}){
            const unsigned count=std::min(available,requested);
            if(count==previous)continue;
            previous=count;
            // Never keep a candidate's threads alive during another trial.
            std::unique_ptr<ari_copy::Team> team;
            try{team=std::make_unique<ari_copy::Team>(count);}
            catch(const std::system_error&){if(count==1)throw;continue;}
            for(unsigned mode=0;mode<(wc.ptr?2u:1u);++mode){
                uint32_t* stage=mode?wc.ptr:cached.ptr;
                std::array<double,3> measured{};
                for(unsigned trial=0;trial<4;++trial){
                    const auto start=Clock::now();
                    team->copy(stage,synthetic[trial&1].data(),words,mode!=0);
                    check(cudaMemcpyAsync(s.a.ptr,stage,bytes,cudaMemcpyHostToDevice,stream),"calibration upload");
                    check(cudaStreamSynchronize(stream),"complete calibration upload");
                    const auto end=Clock::now();
                    if(trial)measured[trial-1]=std::chrono::duration<double,std::milli>(end-start).count();
                }
                std::sort(measured.begin(),measured.end());
                const auto ms=measured[1];
                std::fprintf(stderr," t%u_%s_ms=%.4f",count,mode?"wc":"cached",ms);
                if(ms<best_ms){best_ms=ms;best_count=count;best_wc=mode!=0;}
            }
        }
        fence.complete();
        // Retain one of the tested pinned input buffers. The other goes out
        // of scope before allocating B, keeping peak pinning to three arrays.
        std::swap(s.host_a.ptr,best_wc?wc.ptr:cached.ptr);
    }
    s.input_wc=best_wc;
    s.host_b.alloc(words,best_wc?cudaHostAllocWriteCombined:cudaHostAllocDefault);
    try{s.copy_team=std::make_unique<ari_copy::Team>(best_count);}
    catch(const std::system_error&){
        if(best_count==1)throw;
        best_count=1;
        s.copy_team=std::make_unique<ari_copy::Team>(best_count);
    }
    std::fprintf(stderr," selected_t=%u selected_input=%s\n",best_count,best_wc?"wc":"cached");
}

} // namespace

void* fherma_init(const fherma::Point& p) {
    auto s=std::make_unique<State>(p);
    const size_t words=size_t(p.N)*p.L, residues=size_t(p.N)*s->plan.K;
    s->a.alloc(words);s->b.alloc(words);s->c.alloc(words);
#if FHERMA_PINNED_OUTPUT
    // Output is cacheable: the CPU must read it when constructing the vector.
    s->host_c.alloc(words);
#endif
    s->ra.alloc(residues);s->rb.alloc(residues);s->rc.alloc(residues);
    const unsigned n_pad=(p.N+15u)/16u*16u, l_pad=(p.L+15u)/16u*16u;
    s->packed_inputs.alloc(size_t(8)*n_pad*l_pad);
    const auto weights=ari_encoder_setup::make_weights(s->plan.primes,p.L);
    s->packed_weights.copy(weights.panels);
    s->radix.copy(weights.radix);
    s->primes.copy(s->plan.primes);s->roots.copy(s->plan.roots);
    s->inverse_roots.copy(s->plan.inverse_roots);s->twist.copy(ari_shoup::reorder_twists(s->plan.twist,p.N,s->plan.logN));s->untwist.copy(s->plan.untwist);
    std::vector<uint32_t> primes;for(auto prime:s->plan.primes)primes.push_back(prime.p);
    s->crt=std::make_unique<fherma_crt::DevicePlan>(p.N,p.q.data.data(),p.L,primes);
    check(cudaDeviceSynchronize(),"initialize GPU");
    choose_staging(*s,words);
    return s.release();
}

fherma::Outputs fherma_run(void* state,const fherma::Inputs& in) {
    auto& s=*static_cast<State*>(state);const auto& p=s.plan;
    const size_t words=size_t(p.N)*p.L,bytes=words*sizeof(uint32_t);
    if(in.a.data.size()!=words||in.b.data.size()!=words)throw std::runtime_error("input size does not match point");
    RunFence fence(s.stream_a.handle);
    const dim3 grid((p.N+255)/256,p.K);
#if !FHERMA_NTT_TILED
    const dim3 butterflies((p.N/2+255)/256,p.K);
#endif
    auto submit_operand=[&](unsigned operand,const uint32_t* input,cudaStream_t stream){
        uint32_t* device=operand?s.b.ptr:s.a.ptr;
        uint32_t* residues=operand?s.rb.ptr:s.ra.ptr;
#if FHERMA_PINNED_INPUT
        uint32_t* stage=operand?s.host_b.ptr:s.host_a.ptr;
        s.copy_team->copy(stage,input,words,s.input_wc);
        input=stage;
#endif
        check(cudaMemcpyAsync(device,input,bytes,cudaMemcpyHostToDevice,stream),"upload operand");
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
    // This CPU staging copy and the B upload may overlap A's encoder/NTT.
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
    fherma::Outputs out;out.c.shape={int64_t(p.N),int64_t(p.L)};
#if FHERMA_PINNED_OUTPUT
    check(cudaMemcpyAsync(s.host_c.ptr,s.c.ptr,bytes,cudaMemcpyDeviceToHost,nullptr),"download staged result");
    // Construct elements while D2H and preceding kernels can still run.
    out.c.data.resize(words);
    fence.complete();
    s.copy_team->copy(out.c.data.data(),s.host_c.ptr,words);
#else
    out.c.data.resize(words);
    check(cudaMemcpy(out.c.data.data(),s.c.ptr,bytes,cudaMemcpyDeviceToHost),"download result");
    fence.complete();
#endif
    return out;
}
void fherma_free(void* state) {delete static_cast<State*>(state);}

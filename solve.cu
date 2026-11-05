// Exact negacyclic polynomial multiplication, prepared for Ari Rabinovich.
// RNS NTT + centered CRT. Initial GPU implementation; validation pending.
// Interface/harness: https://github.com/fherma-ai/polynomial-mult-cupqc-example
#include "fherma.h"
#include <cuda_runtime.h>
// Include at global scope. Host-only, read-only topology; no NVML dependency.
#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>
#if defined(__linux__)
#include <sched.h>
#endif

namespace ari_topology {
inline bool canonical_bdf(const char* input,char (&out)[32]) {
    unsigned domain=0,bus=0,device=0,function=0;char extra=0;
    if(!input || std::sscanf(input,"%x:%x:%x.%x%c",&domain,&bus,&device,&function,&extra)!=4 ||
       domain>0xffff || bus>0xff || device>0x1f || function>7)return false;
    std::snprintf(out,sizeof(out),"%04x:%02x:%02x.%x",domain,bus,device,function);
    return true;
}
inline bool read_line(const std::string& path,std::string& line) {
    std::ifstream input(path);return bool(std::getline(input,line));
}
#if defined(__linux__)
inline bool cpu_list(const std::string& text,cpu_set_t& result) {
    CPU_ZERO(&result);const char* at=text.c_str();bool any=false;
    while(*at){
        while(std::isspace(static_cast<unsigned char>(*at)))++at;
        if(!std::isdigit(static_cast<unsigned char>(*at)))return false;
        char* next=nullptr;errno=0;const auto first=std::strtoul(at,&next,10);
        if(errno || next==at)return false;
        at=next;auto last=first;
        if(*at=='-'){
            ++at;if(!std::isdigit(static_cast<unsigned char>(*at)))return false;
            errno=0;last=std::strtoul(at,&next,10);
            if(errno || next==at || last<first)return false;
            at=next;
        }
        if(first<CPU_SETSIZE)
            for(unsigned long cpu=first;cpu<=std::min(last,static_cast<unsigned long>(CPU_SETSIZE-1));++cpu)
                CPU_SET(cpu,&result);
        any=true;
        while(std::isspace(static_cast<unsigned char>(*at)))++at;
        if(!*at)return any;
        if(*at++!=',' || !*at)return false;
    }
    return any;
}
inline std::string cpu_ranges(const cpu_set_t& mask) {
    std::string out;
    for(unsigned first=0;first<CPU_SETSIZE;++first)if(CPU_ISSET(first,&mask)){
        unsigned last=first;
        while(last+1<CPU_SETSIZE && CPU_ISSET(last+1,&mask))++last;
        if(!out.empty())out+=',';
        out+=std::to_string(first);
        if(last!=first)out+='-'+std::to_string(last);
        first=last;
    }
    return out.empty()?"none":out;
}
#endif
inline void report(int device,const char* returned_bdf,int cuda_status=0) noexcept {
    try {
#if defined(__linux__)
        cpu_set_t allowed,local,intersection;CPU_ZERO(&allowed);CPU_ZERO(&local);CPU_ZERO(&intersection);
        const int affinity_status=::sched_getaffinity(0,sizeof(allowed),&allowed);
        const int affinity_errno=affinity_status?errno:0;
        const int current=::sched_getcpu();char bdf[32]={};
        const bool valid=!cuda_status && canonical_bdf(returned_bdf,bdf);
        std::string numa="unknown",list;
        const std::string root=valid?std::string("/sys/bus/pci/devices/")+bdf:std::string();
        if(valid){std::string value;if(read_line(root+"/numa_node",value))numa=value;}
        const bool have_local=valid && read_line(root+"/local_cpulist",list) && cpu_list(list,local);
        if(have_local && !affinity_status)
            for(unsigned cpu=0;cpu<CPU_SETSIZE;++cpu)
                if(CPU_ISSET(cpu,&local) && CPU_ISSET(cpu,&allowed))CPU_SET(cpu,&intersection);
        std::fprintf(stderr,"ARI_CPU_GPU_TOPO device=%d pci=%s current_cpu=%d allowed_count=%d "
            "allowed=%s gpu_numa=%s gpu_local_count=%d gpu_local_allowed_count=%d "
            "gpu_local_allowed=%s affinity_errno=%d cuda_status=%d\n",device,valid?bdf:"unknown",
            current,affinity_status?-1:CPU_COUNT(&allowed),affinity_status?"unknown":cpu_ranges(allowed).c_str(),
            numa.c_str(),have_local?CPU_COUNT(&local):-1,have_local&&!affinity_status?CPU_COUNT(&intersection):-1,
            have_local&&!affinity_status?cpu_ranges(intersection).c_str():"unknown",affinity_errno,cuda_status);
#else
        (void)returned_bdf;std::fprintf(stderr,"ARI_CPU_GPU_TOPO device=%d unsupported_os cuda_status=%d\n",device,cuda_status);
#endif
    }catch(...){std::fprintf(stderr,"ARI_CPU_GPU_TOPO unavailable\n");}
}
} // namespace ari_topology

// Place at global scope after CUDA and topology.hpp are included globally.
// Call once in fherma_init after the existing successful setup synchronization.
void report_gpu_cpu_topology() noexcept {
    int device=-1;char bdf[32]={};
    auto status=cudaGetDevice(&device);
    if(status==cudaSuccess)
        status=cudaDeviceGetPCIBusId(bdf,int(sizeof(bdf)),device);
    if(status!=cudaSuccess)(void)cudaGetLastError();
    ari_topology::report(device,status==cudaSuccess?bdf:nullptr,int(status));
}


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

namespace fherma_crt {
// Prime-major input [p][coefficient], coefficient-major canonical output
// [coefficient][limb]. Threads in a row read neighbouring coefficients.
constexpr unsigned tile_x = 16;
constexpr unsigned tile_y = 8;

struct DeviceArgs {
    unsigned n, limbs, bits, count;
    u32 gap;
    PrimeInfo *primes;
    u32 *weights, *q, *factor;
    const ari::Prime *mont_primes; // owned by enclosing State
};
__global__ void reconstruct_kernel(DeviceArgs plan, const u32 *residues, u32 *out,
                                   unsigned begin,unsigned end) {
    extern __shared__ unsigned long long shared[];
    unsigned x = threadIdx.x, y = threadIdx.y;
    const unsigned base=begin+blockIdx.x*tile_x;
    const unsigned coef=base+x;
    unsigned k = plan.count, l = plan.limbs, pitch = k + 1;
    u64 *low = reinterpret_cast<u64 *>(shared);
    u64 *high = low + l * tile_x;
    u32 *terms = reinterpret_cast<u32 *>(high + l * tile_x);
    u32 *digits = terms + pitch * tile_x;
    const bool active=coef<end;
    for(unsigned j=y;j<k;j+=tile_y){
        u32 t=0;
        if(active){
            const std::size_t at=std::size_t(j)*plan.n+coef;
            t=ari::mont(residues[at],plan.factor[at],plan.mont_primes[j]);
        }
        terms[j*tile_x+x]=t;
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
    }
    __syncthreads();
    // The carry lanes finished writing digit planes. All lanes cooperatively
    // emit contiguous coefficient-major words, even for a short final tile.
    const unsigned rows=end-base<tile_x?end-base:tile_x;
    for(unsigned at=y*tile_x+x;at<rows*l;at+=tile_x*tile_y){
        const unsigned row=at/l,col=at-row*l;
        out[std::size_t(base)*l+at]=digits[col*tile_x+row];
    }
}

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
        if (args_.factor) cudaFree(args_.factor);
    }
public:
    DevicePlan(unsigned n,const u32* q,unsigned limbs,const std::vector<u32>& primes,
               const std::vector<ari::Prime>& mont,const std::vector<u32>& untwist,
               const ari::Prime* device_primes) {
        HostPlan plan = make_plan(n, q, limbs, primes);
        args_.n = n; args_.limbs = limbs; args_.bits = plan.bits;
        args_.count = unsigned(primes.size()); args_.gap = plan.gap;
        shared_ = tile_x * (2 * sizeof(u64) * limbs + sizeof(u32) * (primes.size() + 1 + limbs + 2));
        int device = 0;
        cudaDeviceProp prop{};
        cudaError_t status = cudaGetDevice(&device);
        if (status == cudaSuccess) status = cudaGetDeviceProperties(&prop, device);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        if (shared_ > prop.sharedMemPerBlock) throw std::invalid_argument("CRT tile exceeds shared memory capacity");
        if(mont.size()!=primes.size() || untwist.size()!=std::size_t(n)*primes.size())
            throw std::invalid_argument("invalid CRT untwist table");
        args_.mont_primes=device_primes;
        std::vector<u32> factor(untwist.size());
        for(unsigned j=0;j<primes.size();++j){
            const auto prime=mont[j];
            const auto inv=plan.primes[j].inv;
            for(unsigned i=0;i<n;++i){
                const std::size_t at=std::size_t(j)*n+i;
                // mont(x,factor) = x*untwist*inv/R^2, exactly the old
                // decode(x) times the ordinary CRT reconstruction weight.
                factor[at]=ari::mont(untwist[at],inv,prime);
            }
        }
        try { upload(args_.primes, plan.primes); upload(args_.weights, plan.weights);
              upload(args_.q, plan.q); upload(args_.factor, factor); }
        catch (...) { release(); throw; }
    }
    DevicePlan(const DevicePlan&) = delete;
    DevicePlan& operator=(const DevicePlan&) = delete;
    ~DevicePlan() { release(); }
    // Asynchronous, caller checks eventual execution status when consuming out.
    cudaError_t run(const u32 *residues,u32 *out,unsigned begin,unsigned end,
                    cudaStream_t stream=nullptr) const {
        if(begin>=end||end>args_.n)return cudaErrorInvalidValue;
        reconstruct_kernel<<<(end-begin+tile_x-1)/tile_x,dim3(tile_x,tile_y),shared_,stream>>>
            (args_,residues,out,begin,end);
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

// Include at GLOBAL scope after cuda_runtime.h. Diagnostics only, no counters.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
#ifndef FHERMA_OVERLAP_DIAGNOSTIC
#define FHERMA_OVERLAP_DIAGNOSTIC 1
#endif
#if FHERMA_OVERLAP_DIAGNOSTIC
namespace ari_overlap {
using Clock=std::chrono::steady_clock;
inline double ms(Clock::duration d){return std::chrono::duration<double,std::milli>(d).count();}
enum Cpu : unsigned {RegisterA,RegisterB,ResizeC,RegisterC,InputEventWait,
    UnregisterInputs,FinalWait,UnregisterC,BeforeCapture,CpuCount};
constexpr unsigned event_count=25,gpu_count=17,capacity=128;
constexpr unsigned input_begin(unsigned operand){return 1+operand*6;}
constexpr unsigned crt_begin(unsigned part){return 15+part*2;}
constexpr unsigned output_begin=23,output_end=24;
struct Sample {
    std::array<double,CpuCount> cpu{};
    std::array<double,gpu_count> gpu{};
    unsigned parts=0;
    int crt_ready_after_register=-1;
};
class Profile {
    std::array<cudaEvent_t,event_count> event_{};
    std::vector<Sample> calls_;
    static void checked(cudaError_t status,const char* where){
        if(status!=cudaSuccess)throw std::runtime_error(std::string(where)+": "+cudaGetErrorString(status));
    }
    void release() noexcept {for(auto e:event_)if(e)cudaEventDestroy(e);}
    double elapsed(unsigned first,unsigned last) const {
        float value=0;checked(cudaEventElapsedTime(&value,event_[first],event_[last]),"read diagnostic GPU interval");
        return value;
    }
public:
    Profile(){
        calls_.reserve(capacity);
        try {for(auto& e:event_)checked(cudaEventCreate(&e),"create diagnostic timing event");}
        catch(...){release();throw;}
    }
    ~Profile(){release();}
    Profile(const Profile&)=delete;
    void mark(unsigned index,cudaStream_t stream){
        checked(cudaEventRecord(event_[index],stream),"record diagnostic GPU event");
    }
    int query(unsigned index){
        const auto status=cudaEventQuery(event_[index]);
        if(status==cudaErrorNotReady)return 0;
        checked(status,"query diagnostic CRT ready");return 1;
    }
    // Caller has already completed ALL streams using the original RunFence.
    // No event synchronize, stream synchronize or device synchronize is added.
    void capture(Sample sample){
        if(calls_.size()==calls_.capacity())return;
        sample.gpu.fill(-1.0);unsigned col=0;
        for(unsigned operand=0;operand<2;++operand){
            const auto e=input_begin(operand);
            for(unsigned phase=0;phase<3;++phase)sample.gpu[col++]=elapsed(e+phase*2,e+phase*2+1);
        }
        sample.gpu[col++]=elapsed(13,14);
        for(unsigned part=0;part<4;++part)
            sample.gpu[col++]=part<sample.parts?elapsed(crt_begin(part),crt_begin(part)+1):-1.0;
        sample.gpu[col++]=elapsed(output_begin,output_end);
        for(unsigned e: {6u,12u,14u,crt_begin(sample.parts-1)+1,output_end})
            sample.gpu[col++]=elapsed(0,e);
        calls_.push_back(sample);
    }
    void report(unsigned n,unsigned w) const {
        if(calls_.empty())return;
        std::array<double,capacity> values{};
        auto print=[&](const char* name,auto get){
            unsigned used=0;
            for(const auto& row:calls_){const double v=get(row);if(v>=0 && std::isfinite(v))values[used++]=v;}
            if(!used){std::fprintf(stderr," %s=NA",name);return;}
            std::sort(values.begin(),values.begin()+used);
            std::fprintf(stderr," %s=%.6f",name,(values[(used-1)/2]+values[used/2])*0.5);
        };
        const char* cpu_names[]={"register_A","register_B","resize_C","register_C",
            "input_events_wait","unregister_inputs","final_wait","unregister_C","before_capture"};
        const char* gpu_names[]={"A_H2D_span","A_encode_span","A_forward_span",
            "B_H2D_span","B_encode_span","B_forward_span","multiply_inverse_span",
            "CRT0_span","CRT1_span","CRT2_span","CRT3_span","D2H_span",
            "A_forward_from_origin","B_forward_from_origin","inverse_from_origin",
            "last_CRT_from_origin","D2H_done_from_origin"};
        std::fprintf(stderr,"ARI_OVERLAP_CPU N=%u W=%u calls=%zu median_ms",n,w,calls_.size());
        for(unsigned col=0;col<CpuCount;++col)print(cpu_names[col],[col](const Sample& row){return row.cpu[col];});
        std::fprintf(stderr,"\nARI_OVERLAP_GPU N=%u W=%u calls=%zu median_ms",n,w,calls_.size());
        for(unsigned col=0;col<gpu_count;++col)print(gpu_names[col],[col](const Sample& row){return row.gpu[col];});
        unsigned ready=0,pending=0;
        for(const auto& row:calls_){ready+=row.crt_ready_after_register==1;pending+=row.crt_ready_after_register==0;}
        std::fprintf(stderr,"\nARI_OVERLAP_QUERY N=%u W=%u last_CRT_ready_after_register_C=%u pending=%u "
            "warmups_included=1 diagnostic_only=1 do_not_sum_stage_medians=1\n",n,w,ready,pending);
        std::fflush(stderr);
    }
};
} // namespace ari_overlap
#endif

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
struct AsyncStream {
    cudaStream_t handle=nullptr;
    AsyncStream(){
        check(cudaStreamCreateWithFlags(&handle,cudaStreamNonBlocking),"create non-blocking stream");
    }
    AsyncStream(const AsyncStream&)=delete;
    ~AsyncStream(){if(handle)cudaStreamDestroy(handle);}
};
struct ReadyEvent {
    cudaEvent_t handle=nullptr;
    ReadyEvent(){
        check(cudaEventCreateWithFlags(&handle,cudaEventDisableTiming),"create pipeline event");
    }
    ReadyEvent(const ReadyEvent&)=delete;
    ~ReadyEvent(){if(handle)cudaEventDestroy(handle);}
};
constexpr unsigned max_parts=4;
inline unsigned part_boundary(unsigned n,unsigned i,unsigned parts){return n/parts*i+(n%parts)*i/parts;}
inline unsigned result_parts(size_t bytes){return bytes>=(8u<<20)?max_parts:1u;}
// Own all asynchronous work until the registered host memory can be released.
struct RunFence {
    bool completed=false;
    cudaStream_t operand_a,download;
    RunFence(cudaStream_t a,cudaStream_t result):operand_a(a),download(result){}
    RunFence(const RunFence&)=delete;
    ~RunFence(){
        if(!completed){
            cudaStreamSynchronize(operand_a);
            cudaStreamSynchronize(nullptr);
            if(download)cudaStreamSynchronize(download);
        }
    }
    void complete(){
        check(cudaStreamSynchronize(nullptr),"complete measured work");
        check(cudaStreamSynchronize(operand_a),"check A stream completion");
        if(download)check(cudaStreamSynchronize(download),"complete registered output");
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
#if FHERMA_OVERLAP_DIAGNOSTIC
    ari_overlap::Profile diag;
#endif
    AsyncStream stream_a,download;
    ReadyEvent forward_a,uploaded[2],produced[max_parts];
    const ari_staggered::Address page_size=ari_staggered::host_page_size();
    Buffer<uint32_t> a,b,c,ra,rb,rc,roots,inverse_roots,twist;
    Buffer<ari::Prime> primes;
    Buffer<uint8_t> packed_inputs, packed_weights;
    Buffer<ari_shoup::Weight> radix;
    std::unique_ptr<fherma_crt::DevicePlan> crt;
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
    s->inverse_roots.copy(s->plan.inverse_roots);s->twist.copy(ari_shoup::reorder_twists(s->plan.twist,p.N,s->plan.logN));
    std::vector<uint32_t> primes;for(auto prime:s->plan.primes)primes.push_back(prime.p);
    s->crt=std::make_unique<fherma_crt::DevicePlan>(p.N,p.q.data.data(),p.L,primes,
        s->plan.primes,s->plan.untwist,s->primes.ptr);
    check(cudaDeviceSynchronize(),"initialize GPU");
    report_host_access_capabilities();
    report_gpu_cpu_topology();
    return s.release();
}

fherma::Outputs fherma_run(void* state,const fherma::Inputs& in) {
    auto& s=*static_cast<State*>(state);const auto& p=s.plan;
    const size_t words=size_t(p.N)*p.L,bytes=words*sizeof(uint32_t);
    if(in.a.data.size()!=words||in.b.data.size()!=words)throw std::runtime_error("input size does not match point");
#if FHERMA_OVERLAP_DIAGNOSTIC
    ari_overlap::Sample sample;
    const auto run_started=ari_overlap::Clock::now();
#endif
    // Reverse lifetime order on exceptions: drain streams, unregister all
    // ranges, then free the caller-owned result's ordinary vector storage.
    fherma::Outputs out;out.c.shape={int64_t(p.N),int64_t(p.L)};
    ari_staggered::Registrations registered(s.page_size);
    const unsigned parts=result_parts(bytes);
    cudaStream_t download=parts>1?s.download.handle:nullptr;
    RunFence fence(s.stream_a.handle,download);
#if FHERMA_OVERLAP_DIAGNOSTIC
    sample.parts=parts;
    s.diag.mark(0,nullptr);
#endif
    const dim3 grid((p.N+255)/256,p.K);
#if !FHERMA_NTT_TILED
    const dim3 butterflies((p.N/2+255)/256,p.K);
#endif
    auto submit_operand=[&](unsigned operand,const uint32_t* input,cudaStream_t stream){
        uint32_t* device=operand?s.b.ptr:s.a.ptr;
        uint32_t* residues=operand?s.rb.ptr:s.ra.ptr;
#if FHERMA_OVERLAP_DIAGNOSTIC
        const auto register_started=ari_overlap::Clock::now();
#endif
        registered.acquire(input,bytes);
#if FHERMA_OVERLAP_DIAGNOSTIC
        sample.cpu[operand?ari_overlap::RegisterB:ari_overlap::RegisterA]=ari_overlap::ms(ari_overlap::Clock::now()-register_started);
        const unsigned operand_mark=ari_overlap::input_begin(operand);
        s.diag.mark(operand_mark,stream);
#endif
        registered.for_chunks(input,bytes,[&](size_t offset,size_t amount){
            auto* dst=reinterpret_cast<unsigned char*>(device)+offset;
            const auto* src=reinterpret_cast<const unsigned char*>(input)+offset;
            check(cudaMemcpyAsync(dst,src,amount,cudaMemcpyHostToDevice,stream),"upload registered input");
        });
#if FHERMA_OVERLAP_DIAGNOSTIC
        s.diag.mark(operand_mark+1,stream);
#endif
        check(cudaEventRecord(s.uploaded[operand].handle,stream),"mark last H2D read");
#if FHERMA_OVERLAP_DIAGNOSTIC
        s.diag.mark(operand_mark+2,stream);
#endif
        check(ari_wmma_encoder::launch_operand(device,residues,operand,s.packed_inputs.ptr,
            s.packed_weights.ptr,s.primes.ptr,s.radix.ptr,s.twist.ptr,
            p.N,p.L,p.logN,p.K,stream),"encode operand");
#if FHERMA_OVERLAP_DIAGNOSTIC
        s.diag.mark(operand_mark+3,stream);
        s.diag.mark(operand_mark+4,stream);
#endif
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
#if FHERMA_OVERLAP_DIAGNOSTIC
        s.diag.mark(operand_mark+5,stream);
#endif
    };
    submit_operand(0,in.a.data.data(),s.stream_a.handle);
    check(cudaEventRecord(s.forward_a.handle,s.stream_a.handle),"mark A transform complete");
    // Register B only after the entire A branch has been submitted.
    submit_operand(1,in.b.data.data(),nullptr);
    check(cudaStreamWaitEvent(nullptr,s.forward_a.handle,0),"join A and B before product");
#if FHERMA_OVERLAP_DIAGNOSTIC
    s.diag.mark(13,nullptr);
#endif
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
#if FHERMA_OVERLAP_DIAGNOSTIC
    s.diag.mark(14,nullptr);
#endif
    // Queue every producer first, so constructing/registering C does not
    // delay any CRT stripe. Each writes its own coefficient span in device C.
    for(unsigned part=0;part<parts;++part){
        const unsigned first=part_boundary(p.N,part,parts),last=part_boundary(p.N,part+1,parts);
#if FHERMA_OVERLAP_DIAGNOSTIC
        s.diag.mark(ari_overlap::crt_begin(part),nullptr);
#endif
        check(s.crt->run(s.rc.ptr,s.c.ptr,first,last),"fused CRT stripe");
#if FHERMA_OVERLAP_DIAGNOSTIC
        s.diag.mark(ari_overlap::crt_begin(part)+1,nullptr);
#endif
        if(download)check(cudaEventRecord(s.produced[part].handle,nullptr),"mark CRT stripe ready");
    }
#if FHERMA_OVERLAP_DIAGNOSTIC
    const auto resize_started=ari_overlap::Clock::now();
#endif
    out.c.data.resize(words);
#if FHERMA_OVERLAP_DIAGNOSTIC
    sample.cpu[ari_overlap::ResizeC]=ari_overlap::ms(ari_overlap::Clock::now()-resize_started);
    const auto register_c_started=ari_overlap::Clock::now();
#endif
    const auto output_pages=registered.acquire(out.c.data.data(),bytes);
#if FHERMA_OVERLAP_DIAGNOSTIC
    sample.cpu[ari_overlap::RegisterC]=ari_overlap::ms(ari_overlap::Clock::now()-register_c_started);
    sample.crt_ready_after_register=s.diag.query(ari_overlap::crt_begin(parts-1)+1);
#endif
    for(unsigned part=0;part<parts;++part){
        const size_t first=size_t(part_boundary(p.N,part,parts))*p.L;
        const size_t last=size_t(part_boundary(p.N,part+1,parts))*p.L;
        auto* dst=reinterpret_cast<unsigned char*>(out.c.data.data()+first);
        const auto* src=reinterpret_cast<const unsigned char*>(s.c.ptr+first);
        if(download)check(cudaStreamWaitEvent(download,s.produced[part].handle,0),"wait for CRT stripe");
#if FHERMA_OVERLAP_DIAGNOSTIC
        if(part==0)s.diag.mark(ari_overlap::output_begin,download);
#endif
        registered.for_chunks(dst,(last-first)*sizeof(uint32_t),[&](size_t offset,size_t amount){
            check(cudaMemcpyAsync(dst+offset,src+offset,amount,cudaMemcpyDeviceToHost,download),
                  "download registered CRT stripe");
        });
    }
#if FHERMA_OVERLAP_DIAGNOSTIC
    s.diag.mark(ari_overlap::output_end,download);
    const auto input_wait_started=ari_overlap::Clock::now();
#endif
    // Only the completed H2D copies read the borrowed inputs. No later
    // kernel uses those pages. Keep any registration shared with C until
    // the final D2H completes, even if C uses only a fragment of its range.
    check(cudaEventSynchronize(s.uploaded[0].handle),"wait for A host read");
    check(cudaEventSynchronize(s.uploaded[1].handle),"wait for B host read");
#if FHERMA_OVERLAP_DIAGNOSTIC
    sample.cpu[ari_overlap::InputEventWait]=ari_overlap::ms(ari_overlap::Clock::now()-input_wait_started);
    const auto input_unreg_started=ari_overlap::Clock::now();
#endif
    registered.release_inputs(output_pages);
#if FHERMA_OVERLAP_DIAGNOSTIC
    sample.cpu[ari_overlap::UnregisterInputs]=ari_overlap::ms(ari_overlap::Clock::now()-input_unreg_started);
    const auto final_wait_started=ari_overlap::Clock::now();
#endif
    fence.complete();
#if FHERMA_OVERLAP_DIAGNOSTIC
    sample.cpu[ari_overlap::FinalWait]=ari_overlap::ms(ari_overlap::Clock::now()-final_wait_started);
    const auto output_unreg_started=ari_overlap::Clock::now();
#endif
    registered.release_all();
#if FHERMA_OVERLAP_DIAGNOSTIC
    sample.cpu[ari_overlap::UnregisterC]=ari_overlap::ms(ari_overlap::Clock::now()-output_unreg_started);
    sample.cpu[ari_overlap::BeforeCapture]=ari_overlap::ms(ari_overlap::Clock::now()-run_started);
    s.diag.capture(sample);
#endif
    return out;
}
void fherma_free(void* state) {
    auto* s=static_cast<State*>(state);
#if FHERMA_OVERLAP_DIAGNOSTIC
    if(s)s->diag.report(s->plan.N,s->plan.W);
#endif
    delete s;
}

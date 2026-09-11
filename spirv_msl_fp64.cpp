/*
 * Copyright 2024 SPIRV-Cross Authors
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#include "spirv_msl_fp64.hpp"
#include <cmath>
#include <cstring>
#include <cstdio>

namespace SPIRV_CROSS_NAMESPACE
{

// ===================================================================
// MSL prelude — double-float mode (~46 mantissa bits)
// ===================================================================
static const char *prelude_df = R"(
// ⚠ MGL fp64 emulation — double-float mode (~46 mantissa bits)
// ⚠ This shader must be compiled with mathMode = MTLMathModeSafe
//    (or fastMathEnabled = NO). Fast math will delete error-recovery
//    expressions that are essential for correct rounding.

struct MGLdf { float hi, lo; };

// ---- EFT primitives ----
static inline MGLdf MGLdf_two_sum(float a, float b) {
    float s = a + b, bb = s - a;
    return MGLdf{s, (a - (s - bb)) + (b - bb)};
}
static inline MGLdf MGLdf_quick_two_sum(float a, float b) {
    float s = a + b; return MGLdf{s, b - (s - a)};
}
static inline MGLdf MGLdf_two_prod(float a, float b) {
    float p = a * b; return MGLdf{p, fma(a, b, -p)};
}

// ---- double-float arithmetic ----
static inline MGLdf MGLdf_add(MGLdf a, MGLdf b) {
    MGLdf s = MGLdf_two_sum(a.hi, b.hi), t = MGLdf_two_sum(a.lo, b.lo);
    s.lo += t.hi; s = MGLdf_quick_two_sum(s.hi, s.lo);
    s.lo += t.lo; return MGLdf_quick_two_sum(s.hi, s.lo);
}
static inline MGLdf MGLdf_sub(MGLdf a, MGLdf b) {
    return MGLdf_add(a, MGLdf{-b.hi, -b.lo});
}
static inline MGLdf MGLdf_mul(MGLdf a, MGLdf b) {
    MGLdf p = MGLdf_two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return MGLdf_quick_two_sum(p.hi, p.lo);
}
// Division: 1 NR iteration. float seed ~24 bits, one df NR -> ~48 bits (>46).
static inline MGLdf MGLdf_div(MGLdf a, MGLdf b) {
    float r = 1.0f / b.hi;
    MGLdf x = MGLdf{r, 0.0f};
    MGLdf two = MGLdf{2.0f, 0.0f};
    x = MGLdf_mul(x, MGLdf_sub(two, MGLdf_mul(b, x)));
    return MGLdf_mul(a, x);
}
// Sqrt: 1 NR iteration on rsqrt.
static inline MGLdf MGLdf_sqrt(MGLdf a) {
    float y = rsqrt(a.hi);
    MGLdf yn = MGLdf{y, 0.0f};
    MGLdf half_ = MGLdf{0.5f, 0.0f};
    MGLdf three = MGLdf{3.0f, 0.0f};
    MGLdf yn2 = MGLdf_mul(yn, yn);
    yn = MGLdf_mul(yn, MGLdf_mul(half_, MGLdf_sub(three, MGLdf_mul(a, yn2))));
    return MGLdf_mul(a, yn);
}
static inline MGLdf MGLdf_neg(MGLdf a) { return MGLdf{-a.hi, -a.lo}; }
static inline MGLdf MGLdf_abs(MGLdf a) {
    return MGLdf{fabs(a.hi), (a.hi < 0.0f ? -a.lo : (a.hi > 0.0f ? a.lo : fabs(a.lo)))};
}
static inline MGLdf MGLdf_min(MGLdf a, MGLdf b) {
    float v = a.hi + a.lo, w = b.hi + b.lo;
    return (v < w) ? a : ((w < v) ? b : ((a.hi < b.hi) ? a : b));
}
static inline MGLdf MGLdf_max(MGLdf a, MGLdf b) {
    float v = a.hi + a.lo, w = b.hi + b.lo;
    return (v > w) ? a : ((w > v) ? b : ((a.hi > b.hi) ? a : b));
}

// ---- double-float comparisons ----
static inline bool MGLdf_eq (MGLdf a, MGLdf b) { return (a.hi == b.hi) && (a.lo == b.lo); }
static inline bool MGLdf_ne (MGLdf a, MGLdf b) { return !MGLdf_eq(a, b); }
static inline bool MGLdf_lt (MGLdf a, MGLdf b) { return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo); }
static inline bool MGLdf_le (MGLdf a, MGLdf b) { return (a.hi < b.hi) || (a.hi == b.hi && a.lo <= b.lo); }
static inline bool MGLdf_gt (MGLdf a, MGLdf b) { return MGLdf_lt(b, a); }
static inline bool MGLdf_ge (MGLdf a, MGLdf b) { return MGLdf_le(b, a); }

// ---- Conversions ----
static inline MGLdf MGLdf_from_float(float a) { return MGLdf{a, 0.0f}; }
static inline float MGLdf_to_float(MGLdf a) { return a.hi + a.lo; }
static inline MGLdf MGLdf_from_int(int a) { float f = float(a); return MGLdf{f, float(a) - f}; }
static inline int    MGLdf_to_int(MGLdf a)   { return int(a.hi + a.lo); }
static inline MGLdf MGLdf_from_uint(uint a) { float f = float(a); return MGLdf{f, float(a) - f}; }
static inline uint   MGLdf_to_uint(MGLdf a)  { return uint(a.hi + a.lo); }
static inline MGLdf MGLdf_sign(MGLdf a) {
    float s = a.hi + a.lo;
    if (s > 0.0f) return MGLdf{1.0f, 0.0f};
    if (s < 0.0f) return MGLdf{-1.0f, 0.0f};
    return MGLdf{0.0f, 0.0f};
}

// ---- Rounding ----
static inline MGLdf MGLdf_floor(MGLdf a)     { float f = floor(a.hi + a.lo); return MGLdf{f, 0.0f}; }
static inline MGLdf MGLdf_ceil(MGLdf a)      { float f = ceil(a.hi + a.lo);  return MGLdf{f, 0.0f}; }
static inline MGLdf MGLdf_trunc(MGLdf a)     { float f = trunc(a.hi + a.lo); return MGLdf{f, 0.0f}; }
static inline MGLdf MGLdf_round(MGLdf a)     { float f = round(a.hi + a.lo); return MGLdf{f, 0.0f}; }
static inline MGLdf MGLdf_roundeven(MGLdf a) { float f = rint(a.hi + a.lo);  return MGLdf{f, 0.0f}; }

// ---- Decode / Encode (IEEE binary64 <-> MGLdf) ----
// low word = bits.x, high word = bits.y
static inline MGLdf MGLdf_decode(uint2 bits) {
    // Two floats reach about 46 bits, so the bottom 8 mantissa bits of a
    // binary64 do not survive. That is the cost of the fast mode.
    uint hi = bits.y, lo = bits.x;
    uint sign_bit = hi >> 31;
    int  ex = int((hi >> 20) & 0x7FFu);
    uint mant_hi = hi & 0xFFFFFu;
    float sgn = (sign_bit != 0u) ? -1.0f : 1.0f;
    if (ex == 0x7FF) {
        if (mant_hi == 0u && lo == 0u) return MGLdf{sgn * INFINITY, 0.0f};
        return MGLdf{NAN, 0.0f};
    }
    bool sub = (ex == 0);
    int e = (sub ? 1 : ex) - 1023;
    float lead = (sub ? 0.0f : 1048576.0f) + float(mant_hi);   // < 2^21
    float rest = float(lo >> 8);                               // < 2^24
    MGLdf r;
    r.hi = sgn * ldexp(lead, e - 20);
    r.lo = sgn * ldexp(rest, e - 44);
    return r;
}
static inline uint2 MGLdf_encode(MGLdf v) {
    float s0 = v.hi + v.lo;
    if (s0 == 0.0f) return uint2(0u, 0u);
    if (isinf(s0))  return uint2(0u, (s0 < 0.0f) ? 0xFFF00000u : 0x7FF00000u);
    if (isnan(s0))  return uint2(0u, 0x7FF80000u);
    uint sign = 0u;
    float x = v.hi, y = v.lo;
    if (x < 0.0f) { sign = 1u; x = -x; y = -y; }
    int e = ilogb(x);
    float mx = ldexp(x, -e), my = ldexp(y, -e);
    if (mx + my < 1.0f) { e -= 1; mx += mx; my += my; }
    else if (mx + my >= 2.0f) { e += 1; mx *= 0.5f; my *= 0.5f; }
    float r0 = mx - 1.0f;
    float a = floor(r0 * 1048576.0f);
    uint mant_hi = uint(a);
    float rA = r0 - ldexp(a, -20);
    MGLdf s = MGLdf_two_sum(rA, my);
    float rh = s.hi, rl = s.lo;
    if (rh < 0.0f) {
        mant_hi -= 1u;
        MGLdf g = MGLdf_two_sum(ldexp(1.0f, -20), rh);
        rh = g.hi; rl += g.lo;
    }
    float bh = ldexp(rh, 36), bl = ldexp(rl, 36);
    float b = floor(bh);
    float rem = (bh - b) + bl;
    float k = floor(rem);
    b += k; rem -= k;
    if (b < 0.0f) { b += 65536.0f; mant_hi -= 1u; }
    else if (b >= 65536.0f) { b -= 65536.0f; mant_hi += 1u; }
    uint mid = uint(b);
    float cf = ldexp(rem, 16);
    float c = floor(cf);
    uint tail = uint(c);
    float frac = cf - c;
    ulong mant = (ulong(mant_hi) << 32) | (ulong(mid) << 16) | ulong(tail);
    if (frac >= 0.5f) {
        mant += 1ul;
        if (mant >= (1ul << 52)) { mant = 0ul; e += 1; }
    }
    int biased = e + 1023;
    if (biased <= 0)    return uint2(0u, sign << 31);
    if (biased >= 2047) return uint2(0u, (sign << 31) | 0x7FF00000u);
    ulong bits = (ulong(sign) << 63) | (ulong(uint(biased)) << 52) | mant;
    return uint2(uint(bits & 0xFFFFFFFFul), uint(bits >> 32));
}

// ---- Vector types ----
struct MGLdf2 { MGLdf x, y; };
struct MGLdf3 { MGLdf x, y, z; };
struct MGLdf4 { MGLdf x, y, z, w; };
)";

// ===================================================================
// MSL prelude — triple-float mode (~72.8 mantissa bits, default)
// ===================================================================
static const char *prelude_tf = R"(
// ⚠ MGL fp64 emulation — triple-float mode (~72.8 mantissa bits)
// ⚠ This shader must be compiled with mathMode = MTLMathModeSafe
//    (or fastMathEnabled = NO). Fast math will delete error-recovery
//    expressions that are essential for correct rounding.

struct MGLdf { float hi, lo; };
struct MGLtf { float x, y, z; };  // value == x + y + z, non-overlapping

// ---- EFT primitives (Dekker / Knuth) ----
static inline MGLdf MGLdf_two_sum(float a, float b) {
    float s = a + b, bb = s - a;
    return MGLdf{s, (a - (s - bb)) + (b - bb)};
}
static inline MGLdf MGLdf_quick_two_sum(float a, float b) {
    float s = a + b; return MGLdf{s, b - (s - a)};
}
static inline MGLdf MGLdf_two_prod(float a, float b) {
    float p = a * b; return MGLdf{p, fma(a, b, -p)};
}

// ---- Priest-style renormalisation ----
// IMPORTANT: input terms MUST be in descending magnitude order.
template <int N> static inline MGLtf MGLtf_renorm(thread float (&e)[N]) {
    for (int i = N - 1; i > 0; i--) { MGLdf s = MGLdf_two_sum(e[i - 1], e[i]); e[i - 1] = s.hi; e[i] = s.lo; }
    for (int i = N - 1; i > 1; i--) { MGLdf s = MGLdf_two_sum(e[i - 1], e[i]); e[i - 1] = s.hi; e[i] = s.lo; }
    for (int i = N - 1; i > 2; i--) { MGLdf s = MGLdf_two_sum(e[i - 1], e[i]); e[i - 1] = s.hi; e[i] = s.lo; }
    float t = 0.0f;
    for (int i = 3; i < N; i++) t += e[i];
    return MGLtf{e[0], e[1], e[2] + t};
}

// ---- triple-float arithmetic ----
static inline MGLtf MGLtf_canon(MGLtf v) {
    float e[3] = {v.x, v.y, v.z};
    return MGLtf_renorm(e);
}
static inline MGLtf MGLtf_add(MGLtf a_in, MGLtf b_in) {
    // Merge two already-descending triples into one descending six. The old
    // interleave {a.x,b.x,a.y,b.y,a.z,b.z} is only in order when a and b are
    // the same size; when they are not, the renormalisation loses the tail.
    // Both must be non-overlapping first; a freshly decoded double is not.
    MGLtf a = MGLtf_canon(a_in), b = MGLtf_canon(b_in);
    float A[3] = {a.x, a.y, a.z};
    float B[3] = {b.x, b.y, b.z};
    float e[6];
    int i = 0, j = 0;
    for (int k = 0; k < 6; k++) {
        bool takeA = (j >= 3) || (i < 3 && fabs(A[i]) >= fabs(B[j]));
        e[k] = takeA ? A[i] : B[j];
        i += takeA ? 1 : 0;
        j += takeA ? 0 : 1;
    }
    return MGLtf_renorm(e);
}
static inline MGLtf MGLtf_sub(MGLtf a, MGLtf b) {
    return MGLtf_add(a, MGLtf{-b.x, -b.y, -b.z});
}
static inline MGLtf MGLtf_mul(MGLtf a_in, MGLtf b_in) {
    // The 13-term ordering below assumes each component is about 2^-24 of the
    // one before it. A decoded double is not in that form until it is renormalised.
    MGLtf a = MGLtf_canon(a_in), b = MGLtf_canon(b_in);
    MGLdf p0 = MGLdf_two_prod(a.x, b.x), p1 = MGLdf_two_prod(a.x, b.y), p2 = MGLdf_two_prod(a.y, b.x);
    MGLdf p3 = MGLdf_two_prod(a.y, b.y), p4 = MGLdf_two_prod(a.x, b.z), p5 = MGLdf_two_prod(a.z, b.x);
    float tail = a.y * b.z + a.z * b.y + a.z * b.z;
    // VecSum needs terms in DESCENDING magnitude.
    float e[13] = { p0.hi,
                     p0.lo, p1.hi, p2.hi,
                     p1.lo, p2.lo, p3.hi, p4.hi, p5.hi,
                     p3.lo, p4.lo, p5.lo, tail };
    return MGLtf_renorm(e);
}
// Division: 2 NR iterations. float seed ~24 bits -> 1st NR ~48 bits -> 2nd NR ~96 bits (>72.8).
static inline MGLtf MGLtf_div(MGLtf a, MGLtf b) {
    float r = 1.0f / b.x;
    MGLtf x = MGLtf{r, 0.0f, 0.0f};
    MGLtf two = MGLtf{2.0f, 0.0f, 0.0f};
    x = MGLtf_mul(x, MGLtf_sub(two, MGLtf_mul(b, x)));
    x = MGLtf_mul(x, MGLtf_sub(two, MGLtf_mul(b, x)));
    return MGLtf_mul(a, x);
}
// Sqrt: 2 NR iterations on rsqrt.
static inline MGLtf MGLtf_sqrt(MGLtf a) {
    float y = rsqrt(a.x);
    MGLtf yn = MGLtf{y, 0.0f, 0.0f};
    MGLtf half_ = MGLtf{0.5f, 0.0f, 0.0f};
    MGLtf three = MGLtf{3.0f, 0.0f, 0.0f};
    MGLtf yn2 = MGLtf_mul(yn, yn);
    yn = MGLtf_mul(yn, MGLtf_mul(half_, MGLtf_sub(three, MGLtf_mul(a, yn2))));
    yn2 = MGLtf_mul(yn, yn);
    yn = MGLtf_mul(yn, MGLtf_mul(half_, MGLtf_sub(three, MGLtf_mul(a, yn2))));
    return MGLtf_mul(a, yn);
}
static inline MGLtf MGLtf_neg(MGLtf a) { return MGLtf{-a.x, -a.y, -a.z}; }
static inline MGLtf MGLtf_abs(MGLtf a) {
    return MGLtf{fabs(a.x),
                 (a.x < 0.0f ? -a.y : (a.x > 0.0f ? a.y : fabs(a.y))),
                 (a.x < 0.0f ? -a.z : (a.x > 0.0f ? a.z : ((a.y < 0.0f) ? -a.z : fabs(a.z))))};
}
static inline MGLtf MGLtf_min(MGLtf a, MGLtf b) {
    float va = a.x + a.y + a.z, vb = b.x + b.y + b.z;
    if (va < vb) return a; if (vb < va) return b;
    if (a.x < b.x) return a; if (b.x < a.x) return b;
    if (a.y < b.y) return a; if (b.y < a.y) return b;
    return (a.z < b.z) ? a : b;
}
static inline MGLtf MGLtf_max(MGLtf a, MGLtf b) {
    float va = a.x + a.y + a.z, vb = b.x + b.y + b.z;
    if (va > vb) return a; if (vb > va) return b;
    if (a.x > b.x) return a; if (b.x > a.x) return b;
    if (a.y > b.y) return a; if (b.y > a.y) return b;
    return (a.z > b.z) ? a : b;
}

// ---- triple-float comparisons ----
static inline bool MGLtf_eq(MGLtf a, MGLtf b) { return (a.x == b.x) && (a.y == b.y) && (a.z == b.z); }
static inline bool MGLtf_ne(MGLtf a, MGLtf b) { return !MGLtf_eq(a, b); }
static inline bool MGLtf_lt(MGLtf a, MGLtf b) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z < b.z;
}
static inline bool MGLtf_le(MGLtf a, MGLtf b) {
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z <= b.z;
}
static inline bool MGLtf_gt(MGLtf a, MGLtf b) { return MGLtf_lt(b, a); }
static inline bool MGLtf_ge(MGLtf a, MGLtf b) { return MGLtf_le(b, a); }

// ---- Conversions ----
static inline MGLtf MGLtf_from_float(float a) { return MGLtf{a, 0.0f, 0.0f}; }
static inline float MGLtf_to_float(MGLtf a) { return a.x + a.y + a.z; }
static inline MGLtf MGLtf_from_int(int a) {
    float f = float(a);
    MGLdf r = MGLdf_two_sum(f, float(a) - f);
    return MGLtf{r.hi, r.lo, 0.0f};
}
static inline int MGLtf_to_int(MGLtf a) { return int(a.x + a.y + a.z); }
static inline MGLtf MGLtf_from_uint(uint a) {
    float f = float(a);
    MGLdf r = MGLdf_two_sum(f, float(a) - f);
    return MGLtf{r.hi, r.lo, 0.0f};
}
static inline uint MGLtf_to_uint(MGLtf a) { return uint(a.x + a.y + a.z); }
static inline MGLtf MGLtf_sign(MGLtf a) {
    float s = a.x + a.y + a.z;
    if (s > 0.0f) return MGLtf{1.0f, 0.0f, 0.0f};
    if (s < 0.0f) return MGLtf{-1.0f, 0.0f, 0.0f};
    return MGLtf{0.0f, 0.0f, 0.0f};
}

// ---- Rounding ----
static inline MGLtf MGLtf_floor(MGLtf a)     { float f = floor(a.x + a.y + a.z); return MGLtf{f, 0.0f, 0.0f}; }
static inline MGLtf MGLtf_ceil(MGLtf a)      { float f = ceil(a.x + a.y + a.z);  return MGLtf{f, 0.0f, 0.0f}; }
static inline MGLtf MGLtf_trunc(MGLtf a)     { float f = trunc(a.x + a.y + a.z); return MGLtf{f, 0.0f, 0.0f}; }
static inline MGLtf MGLtf_round(MGLtf a)     { float f = round(a.x + a.y + a.z); return MGLtf{f, 0.0f, 0.0f}; }
static inline MGLtf MGLtf_roundeven(MGLtf a) { float f = rint(a.x + a.y + a.z);  return MGLtf{f, 0.0f, 0.0f}; }

// ---- Decode / Encode (IEEE binary64 <-> MGLtf) ----
static inline MGLtf MGLtf_decode(uint2 bits) {
    // binary64 has 52 stored mantissa bits; a float holds 24 exactly. Split the
    // mantissa into 21 + 16 + 16 bit pieces so every conversion below is exact
    // and the three floats end up non-overlapping without any renormalisation.
    uint hi = bits.y, lo = bits.x;
    uint sign_bit = hi >> 31;
    int  ex = int((hi >> 20) & 0x7FFu);
    uint mant_hi = hi & 0xFFFFFu;
    float sgn = (sign_bit != 0u) ? -1.0f : 1.0f;
    if (ex == 0x7FF) {
        if (mant_hi == 0u && lo == 0u) return MGLtf{sgn * INFINITY, 0.0f, 0.0f};
        return MGLtf{NAN, 0.0f, 0.0f};
    }
    bool sub = (ex == 0);
    int e = (sub ? 1 : ex) - 1023;
    float lead = (sub ? 0.0f : 1048576.0f) + float(mant_hi);   // < 2^21
    float mid  = float(lo >> 16);                              // < 2^16
    float tail = float(lo & 0xFFFFu);                          // < 2^16
    float e3[3];
    e3[0] = sgn * ldexp(lead, e - 20);
    e3[1] = sgn * ldexp(mid,  e - 36);
    e3[2] = sgn * ldexp(tail, e - 52);
    // Exact, but the pieces overlap. MGLtf_canon puts it in the non-overlapping
    // form the arithmetic needs; encode wants it exactly as split here, so the
    // canonicalisation happens at the operations rather than in decode.
    MGLtf r; r.x = e3[0]; r.y = e3[1]; r.z = e3[2];
    return r;
}
static inline uint2 MGLtf_encode(MGLtf v) {
    // Peel the mantissa back out in 20 + 16 + 16 bit pieces. Every scaling here
    // is by a power of two and every floor lands on a value under 2^20, so the
    // only rounding is the deliberate one at the end.
    float s0 = v.x + v.y + v.z;
    if (s0 == 0.0f) return uint2(0u, 0u);
    if (isinf(s0))  return uint2(0u, (s0 < 0.0f) ? 0xFFF00000u : 0x7FF00000u);
    if (isnan(s0))  return uint2(0u, 0x7FF80000u);
    uint sign = 0u;
    float x = v.x, y = v.y, z = v.z;
    if (x < 0.0f) { sign = 1u; x = -x; y = -y; z = -z; }
    int e = ilogb(x);
    float mx = ldexp(x, -e), my = ldexp(y, -e), mz = ldexp(z, -e);
    // the low terms can drag the value out of [1,2), which would break the peel
    if (mx + my + mz < 1.0f) { e -= 1; mx += mx; my += my; mz += mz; }
    else if (mx + my + mz >= 2.0f) { e += 1; mx *= 0.5f; my *= 0.5f; mz *= 0.5f; }
    float r0 = mx - 1.0f;                       // exact, in [0,1]
    float a  = floor(r0 * 1048576.0f);          // <= 2^20
    uint mant_hi = uint(a);
    float rA = r0 - ldexp(a, -20);              // exact
    MGLdf s = MGLdf_two_sum(rA, my);
    MGLdf t = MGLdf_two_sum(s.lo, mz);
    float rh = s.hi, rl = t.hi + t.lo;
    // A negative correction means the peel above took one unit too many. Borrow
    // it back, exactly. Without this the chunk below goes negative and the cast
    // to uint wraps, which is a whole-value error, not a rounding one.
    if (rh < 0.0f) {
        mant_hi -= 1u;
        MGLdf g = MGLdf_two_sum(ldexp(1.0f, -20), rh);
        rh = g.hi; rl += g.lo;
    }
    float bh = ldexp(rh, 36), bl = ldexp(rl, 36);
    float b = floor(bh);
    float rem = (bh - b) + bl;
    float k = floor(rem);
    b += k; rem -= k;
    if (b < 0.0f) { b += 65536.0f; mant_hi -= 1u; }
    else if (b >= 65536.0f) { b -= 65536.0f; mant_hi += 1u; }
    uint mid = uint(b);
    float cf = ldexp(rem, 16);
    float c = floor(cf);
    uint tail = uint(c);
    float frac = cf - c;                        // what is left below 2^-52
    ulong mant = (ulong(mant_hi) << 32) | (ulong(mid) << 16) | ulong(tail);
    if (frac >= 0.5f) {                         // round to nearest
        mant += 1ul;
        if (mant >= (1ul << 52)) { mant = 0ul; e += 1; }
    }
    int biased = e + 1023;
    if (biased <= 0)    return uint2(0u, sign << 31);
    if (biased >= 2047) return uint2(0u, (sign << 31) | 0x7FF00000u);
    ulong bits = (ulong(sign) << 63) | (ulong(uint(biased)) << 52) | mant;
    return uint2(uint(bits & 0xFFFFFFFFul), uint(bits >> 32));
}

// ---- Vector types ----
struct MGLtf2 { MGLtf x, y; };
struct MGLtf3 { MGLtf x, y, z; };
struct MGLtf4 { MGLtf x, y, z, w; };
)";

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

const char *get_fp64_msl_prelude(FP64Mode mode)
{
	switch (mode)
	{
	case FP64Mode::DoubleFloat:
		return prelude_df;
	case FP64Mode::TripleFloat:
		return prelude_tf;
	default:
		return nullptr;
	}
}

const char *get_fp64_type_name(FP64Mode mode)
{
	switch (mode)
	{
	case FP64Mode::DoubleFloat:
		return "MGLdf";
	case FP64Mode::TripleFloat:
		return "MGLtf";
	default:
		return nullptr;
	}
}

void split_double_df(double value, float &hi, float &lo)
{
	hi = static_cast<float>(value);
	lo = static_cast<float>(value - static_cast<double>(hi));
}

void split_double_tf(double value, float &x, float &y, float &z)
{
	x = static_cast<float>(value);
	double remainder = value - static_cast<double>(x);
	y = static_cast<float>(remainder);
	remainder -= static_cast<double>(y);
	z = static_cast<float>(remainder);
}

const char *get_fp64_func_name(const char *op, FP64Mode mode)
{
	static char buf[64];
	const char *prefix = (mode == FP64Mode::DoubleFloat) ? "MGLdf_" : "MGLtf_";
	snprintf(buf, sizeof(buf), "%s%s", prefix, op);
	return buf;
}

const char *get_fp64_dot_name(int vecsize, FP64Mode mode)
{
	static char buf[64];
	const char *prefix = (mode == FP64Mode::DoubleFloat) ? "MGLdf_dot" : "MGLtf_dot";
	snprintf(buf, sizeof(buf), "%s%d", prefix, vecsize);
	return buf;
}

const char *get_fp64_length_name(int vecsize, FP64Mode mode)
{
	static char buf[64];
	const char *prefix = (mode == FP64Mode::DoubleFloat) ? "MGLdf_length" : "MGLtf_length";
	snprintf(buf, sizeof(buf), "%s%d", prefix, vecsize);
	return buf;
}

const char *get_fp64_distance_name(int vecsize, FP64Mode mode)
{
	static char buf[64];
	const char *prefix = (mode == FP64Mode::DoubleFloat) ? "MGLdf_distance" : "MGLtf_distance";
	snprintf(buf, sizeof(buf), "%s%d", prefix, vecsize);
	return buf;
}

const char *get_fp64_normalize_name(int vecsize, FP64Mode mode)
{
	static char buf[64];
	const char *prefix = (mode == FP64Mode::DoubleFloat) ? "MGLdf_normalize" : "MGLtf_normalize";
	snprintf(buf, sizeof(buf), "%s%d", prefix, vecsize);
	return buf;
}

const char *get_fp64_decode_name(FP64Mode mode)
{
	return (mode == FP64Mode::DoubleFloat) ? "MGLdf_decode" : "MGLtf_decode";
}

const char *get_fp64_encode_name(FP64Mode mode)
{
	return (mode == FP64Mode::DoubleFloat) ? "MGLdf_encode" : "MGLtf_encode";
}

const char *get_fp64_convert_name(const char *from, const char *to, FP64Mode mode)
{
	static char buf[64];
	const char *prefix = (mode == FP64Mode::DoubleFloat) ? "MGLdf" : "MGLtf";
	if (strcmp(from, "double") == 0)
		snprintf(buf, sizeof(buf), "%s_to_%s", prefix, to);
	else
		snprintf(buf, sizeof(buf), "%s_from_%s", prefix, from);
	return buf;
}

const char *get_fp64_sign_name(FP64Mode mode)
{
	return (mode == FP64Mode::DoubleFloat) ? "MGLdf_sign" : "MGLtf_sign";
}

} // namespace SPIRV_CROSS_NAMESPACE

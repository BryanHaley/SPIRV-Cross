/*
 * Copyright 2024 SPIRV-Cross Authors
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 */

#include "spirv_msl_fp64.hpp"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <functional>
#include <vector>

namespace SPIRV_CROSS_NAMESPACE
{

// ===================================================================
// MSL prelude — double-float mode (~46 mantissa bits)
// ===================================================================
static const char *prelude_df = R"(
// Every helper is inlined. Called as a real function, with a thread array
// passed by reference, the Apple GPU compiler gave wrong low-order parts or
// dropped the result on some dispatches -- different ones each process.
#define MGL_INL __attribute__((always_inline)) static inline
// ⚠ MGL fp64 emulation — double-float mode (~46 mantissa bits)
// ⚠ This shader must be compiled with mathMode = MTLMathModeSafe
//    (or fastMathEnabled = NO). Fast math will delete error-recovery
//    expressions that are essential for correct rounding.

struct MGLdf { float hi, lo; };

// ---- EFT primitives ----
MGL_INL MGLdf MGLdf_two_sum(float a, float b) {
    float s = a + b, bb = s - a;
    return MGLdf{s, (a - (s - bb)) + (b - bb)};
}
MGL_INL MGLdf MGLdf_quick_two_sum(float a, float b) {
    float s = a + b; return MGLdf{s, b - (s - a)};
}
MGL_INL MGLdf MGLdf_two_prod(float a, float b) {
    float p = a * b; return MGLdf{p, fma(a, b, -p)};
}

// ---- double-float arithmetic ----
MGL_INL MGLdf MGLdf_add(MGLdf a, MGLdf b) {
    MGLdf s = MGLdf_two_sum(a.hi, b.hi), t = MGLdf_two_sum(a.lo, b.lo);
    s.lo += t.hi; s = MGLdf_quick_two_sum(s.hi, s.lo);
    s.lo += t.lo; return MGLdf_quick_two_sum(s.hi, s.lo);
}
MGL_INL MGLdf MGLdf_sub(MGLdf a, MGLdf b) {
    return MGLdf_add(a, MGLdf{-b.hi, -b.lo});
}
MGL_INL MGLdf MGLdf_mul(MGLdf a, MGLdf b) {
    MGLdf p = MGLdf_two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return MGLdf_quick_two_sum(p.hi, p.lo);
}
// Division: 1 NR iteration. float seed ~24 bits, one df NR -> ~48 bits (>46).
MGL_INL MGLdf MGLdf_div(MGLdf a, MGLdf b) {
    float r = 1.0f / b.hi;
    MGLdf x = MGLdf{r, 0.0f};
    MGLdf two = MGLdf{2.0f, 0.0f};
    x = MGLdf_mul(x, MGLdf_sub(two, MGLdf_mul(b, x)));
    return MGLdf_mul(a, x);
}
// Sqrt: 1 NR iteration on rsqrt.
MGL_INL MGLdf MGLdf_sqrt(MGLdf a) {
    float y = rsqrt(a.hi);
    MGLdf yn = MGLdf{y, 0.0f};
    MGLdf half_ = MGLdf{0.5f, 0.0f};
    MGLdf three = MGLdf{3.0f, 0.0f};
    MGLdf yn2 = MGLdf_mul(yn, yn);
    yn = MGLdf_mul(yn, MGLdf_mul(half_, MGLdf_sub(three, MGLdf_mul(a, yn2))));
    return MGLdf_mul(a, yn);
}
MGL_INL MGLdf MGLdf_neg(MGLdf a) { return MGLdf{-a.hi, -a.lo}; }
MGL_INL MGLdf MGLdf_abs(MGLdf a) {
    return MGLdf{fabs(a.hi), (a.hi < 0.0f ? -a.lo : (a.hi > 0.0f ? a.lo : fabs(a.lo)))};
}
MGL_INL MGLdf MGLdf_min(MGLdf a, MGLdf b) {
    float v = a.hi + a.lo, w = b.hi + b.lo;
    return (v < w) ? a : ((w < v) ? b : ((a.hi < b.hi) ? a : b));
}
MGL_INL MGLdf MGLdf_max(MGLdf a, MGLdf b) {
    float v = a.hi + a.lo, w = b.hi + b.lo;
    return (v > w) ? a : ((w > v) ? b : ((a.hi > b.hi) ? a : b));
}

// ---- double-float comparisons ----
MGL_INL bool MGLdf_eq (MGLdf a, MGLdf b) { return (a.hi == b.hi) && (a.lo == b.lo); }
MGL_INL bool MGLdf_ne (MGLdf a, MGLdf b) { return !MGLdf_eq(a, b); }
MGL_INL bool MGLdf_lt (MGLdf a, MGLdf b) { return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo); }
MGL_INL bool MGLdf_le (MGLdf a, MGLdf b) { return (a.hi < b.hi) || (a.hi == b.hi && a.lo <= b.lo); }
MGL_INL bool MGLdf_gt (MGLdf a, MGLdf b) { return MGLdf_lt(b, a); }
MGL_INL bool MGLdf_ge (MGLdf a, MGLdf b) { return MGLdf_le(b, a); }

// ---- Conversions ----
MGL_INL MGLdf MGLdf_from_float(float a) { return MGLdf{a, 0.0f}; }
MGL_INL float MGLdf_to_float(MGLdf a) { return a.hi + a.lo; }
MGL_INL MGLdf MGLdf_from_int(int a) { float f = float(a); return MGLdf{f, float(a) - f}; }
MGL_INL int    MGLdf_to_int(MGLdf a)   { return int(a.hi + a.lo); }
MGL_INL MGLdf MGLdf_from_uint(uint a) { float f = float(a); return MGLdf{f, float(a) - f}; }
MGL_INL uint   MGLdf_to_uint(MGLdf a)  { return uint(a.hi + a.lo); }
MGL_INL MGLdf MGLdf_sign(MGLdf a) {
    float s = a.hi + a.lo;
    if (s > 0.0f) return MGLdf{1.0f, 0.0f};
    if (s < 0.0f) return MGLdf{-1.0f, 0.0f};
    return MGLdf{0.0f, 0.0f};
}

// ---- Rounding ----
MGL_INL MGLdf MGLdf_floor(MGLdf a)     { float f = floor(a.hi + a.lo); return MGLdf{f, 0.0f}; }
MGL_INL MGLdf MGLdf_ceil(MGLdf a)      { float f = ceil(a.hi + a.lo);  return MGLdf{f, 0.0f}; }
MGL_INL MGLdf MGLdf_trunc(MGLdf a)     { float f = trunc(a.hi + a.lo); return MGLdf{f, 0.0f}; }
MGL_INL MGLdf MGLdf_round(MGLdf a)     { float f = round(a.hi + a.lo); return MGLdf{f, 0.0f}; }
MGL_INL MGLdf MGLdf_roundeven(MGLdf a) { float f = rint(a.hi + a.lo);  return MGLdf{f, 0.0f}; }

// ---- Decode / Encode (IEEE binary64 <-> MGLdf) ----
// low word = bits.x, high word = bits.y
MGL_INL MGLdf MGLdf_decode(uint2 bits) {
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
MGL_INL uint2 MGLdf_encode(MGLdf v) {
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

)";

// ===================================================================
// MSL prelude — triple-float mode (~72.8 mantissa bits, default)
// ===================================================================
static const char *prelude_tf = R"(
// Every helper is inlined. Called as a real function, with a thread array
// passed by reference, the Apple GPU compiler gave wrong low-order parts or
// dropped the result on some dispatches -- different ones each process.
#define MGL_INL __attribute__((always_inline)) static inline
// ⚠ MGL fp64 emulation — triple-float mode (~72.8 mantissa bits)
// ⚠ This shader must be compiled with mathMode = MTLMathModeSafe
//    (or fastMathEnabled = NO). Fast math will delete error-recovery
//    expressions that are essential for correct rounding.

struct MGLdf { float hi, lo; };
struct MGLtf { float x, y, z; };  // value == x + y + z, non-overlapping

// ---- EFT primitives (Dekker / Knuth) ----
MGL_INL MGLdf MGLdf_two_sum(float a, float b) {
    float s = a + b, bb = s - a;
    return MGLdf{s, (a - (s - bb)) + (b - bb)};
}
MGL_INL MGLdf MGLdf_quick_two_sum(float a, float b) {
    float s = a + b; return MGLdf{s, b - (s - a)};
}
MGL_INL MGLdf MGLdf_two_prod(float a, float b) {
    float p = a * b; return MGLdf{p, fma(a, b, -p)};
}

// ---- Priest-style renormalisation ----
// IMPORTANT: input terms MUST be in descending magnitude order.
template <int N> MGL_INL MGLtf MGLtf_renorm(thread float (&e)[N]) {
    // unrolled, so every index is a constant and e stays in registers;
    // a counted index puts the array in per-thread memory
#pragma unroll
    for (int i = N - 1; i > 0; i--) { MGLdf s = MGLdf_two_sum(e[i - 1], e[i]); e[i - 1] = s.hi; e[i] = s.lo; }
#pragma unroll
    for (int i = N - 1; i > 1; i--) { MGLdf s = MGLdf_two_sum(e[i - 1], e[i]); e[i - 1] = s.hi; e[i] = s.lo; }
#pragma unroll
    for (int i = N - 1; i > 2; i--) { MGLdf s = MGLdf_two_sum(e[i - 1], e[i]); e[i - 1] = s.hi; e[i] = s.lo; }
    float t = 0.0f;
#pragma unroll
    for (int i = 3; i < N; i++) t += e[i];
    return MGLtf{e[0], e[1], e[2] + t};
}

// ---- triple-float arithmetic ----
MGL_INL MGLtf MGLtf_canon(MGLtf v) {
    float e[3] = {v.x, v.y, v.z};
    return MGLtf_renorm(e);
}
MGL_INL MGLtf MGLtf_add(MGLtf a_in, MGLtf b_in) {
    // Merge two already-descending triples into one descending six. The old
    // interleave {a.x,b.x,a.y,b.y,a.z,b.z} is only in order when a and b are
    // the same size; when they are not, the renormalisation loses the tail.
    // Both must be non-overlapping first; a freshly decoded double is not.
    MGLtf a = MGLtf_canon(a_in), b = MGLtf_canon(b_in);
    // Plain variables and selects only. An array indexed by a counter lives in
    // per-thread memory, and the merge read from it went wrong now and then.
    float ah = a.x, am = a.y, al = a.z, bh = b.x, bm = b.y, bl = b.z;
    int na = 3, nb = 3;
    float e[6];
#pragma unroll
    for (int k = 0; k < 6; k++) {
        bool takeA = (nb == 0) || (na > 0 && fabs(ah) >= fabs(bh));
        e[k] = takeA ? ah : bh;
        // shift the taken triple up by one; an empty slot reads zero
        ah = takeA ? am : ah; am = takeA ? al : am; al = takeA ? 0.0f : al;
        bh = takeA ? bh : bm; bm = takeA ? bm : bl; bl = takeA ? bl : 0.0f;
        na -= takeA ? 1 : 0;
        nb -= takeA ? 0 : 1;
    }
    return MGLtf_renorm(e);
}
MGL_INL MGLtf MGLtf_sub(MGLtf a, MGLtf b) {
    return MGLtf_add(a, MGLtf{-b.x, -b.y, -b.z});
}
MGL_INL MGLtf MGLtf_mul(MGLtf a_in, MGLtf b_in) {
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
MGL_INL MGLtf MGLtf_div(MGLtf a, MGLtf b) {
    float r = 1.0f / b.x;
    MGLtf x = MGLtf{r, 0.0f, 0.0f};
    MGLtf two = MGLtf{2.0f, 0.0f, 0.0f};
    x = MGLtf_mul(x, MGLtf_sub(two, MGLtf_mul(b, x)));
    x = MGLtf_mul(x, MGLtf_sub(two, MGLtf_mul(b, x)));
    return MGLtf_mul(a, x);
}
// Sqrt: 2 NR iterations on rsqrt.
MGL_INL MGLtf MGLtf_sqrt(MGLtf a) {
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
MGL_INL MGLtf MGLtf_neg(MGLtf a) { return MGLtf{-a.x, -a.y, -a.z}; }
MGL_INL MGLtf MGLtf_abs(MGLtf a) {
    return MGLtf{fabs(a.x),
                 (a.x < 0.0f ? -a.y : (a.x > 0.0f ? a.y : fabs(a.y))),
                 (a.x < 0.0f ? -a.z : (a.x > 0.0f ? a.z : ((a.y < 0.0f) ? -a.z : fabs(a.z))))};
}
MGL_INL MGLtf MGLtf_min(MGLtf a, MGLtf b) {
    float va = a.x + a.y + a.z, vb = b.x + b.y + b.z;
    if (va < vb) return a; if (vb < va) return b;
    if (a.x < b.x) return a; if (b.x < a.x) return b;
    if (a.y < b.y) return a; if (b.y < a.y) return b;
    return (a.z < b.z) ? a : b;
}
MGL_INL MGLtf MGLtf_max(MGLtf a, MGLtf b) {
    float va = a.x + a.y + a.z, vb = b.x + b.y + b.z;
    if (va > vb) return a; if (vb > va) return b;
    if (a.x > b.x) return a; if (b.x > a.x) return b;
    if (a.y > b.y) return a; if (b.y > a.y) return b;
    return (a.z > b.z) ? a : b;
}

// ---- triple-float comparisons ----
// One value has many splits into three floats -- a double freshly decoded
// from its bits is split differently from the same double written as a
// literal -- so both sides are put in the one non-overlapping form before
// their parts are compared.
MGL_INL bool MGLtf_eq(MGLtf a_in, MGLtf b_in) {
    // an infinity or NaN is all in the top part, and canon would turn the
    // rest into NaN
    if (!isfinite(a_in.x) || !isfinite(b_in.x)) return a_in.x == b_in.x;
    MGLtf a = MGLtf_canon(a_in), b = MGLtf_canon(b_in);
    return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}
MGL_INL bool MGLtf_ne(MGLtf a, MGLtf b) { return !MGLtf_eq(a, b); }
MGL_INL bool MGLtf_lt(MGLtf a_in, MGLtf b_in) {
    if (!isfinite(a_in.x) || !isfinite(b_in.x)) return a_in.x < b_in.x;
    MGLtf a = MGLtf_canon(a_in), b = MGLtf_canon(b_in);
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z < b.z;
}
MGL_INL bool MGLtf_le(MGLtf a_in, MGLtf b_in) {
    if (!isfinite(a_in.x) || !isfinite(b_in.x)) return a_in.x <= b_in.x;
    MGLtf a = MGLtf_canon(a_in), b = MGLtf_canon(b_in);
    if (a.x != b.x) return a.x < b.x;
    if (a.y != b.y) return a.y < b.y;
    return a.z <= b.z;
}
MGL_INL bool MGLtf_gt(MGLtf a, MGLtf b) { return MGLtf_lt(b, a); }
MGL_INL bool MGLtf_ge(MGLtf a, MGLtf b) { return MGLtf_le(b, a); }

// ---- Conversions ----
MGL_INL MGLtf MGLtf_from_float(float a) { return MGLtf{a, 0.0f, 0.0f}; }
MGL_INL float MGLtf_to_float(MGLtf a) { return a.x + a.y + a.z; }
MGL_INL MGLtf MGLtf_from_int(int a) {
    float f = float(a);
    MGLdf r = MGLdf_two_sum(f, float(a) - f);
    return MGLtf{r.hi, r.lo, 0.0f};
}
MGL_INL int MGLtf_to_int(MGLtf a) { return int(a.x + a.y + a.z); }
MGL_INL MGLtf MGLtf_from_uint(uint a) {
    float f = float(a);
    MGLdf r = MGLdf_two_sum(f, float(a) - f);
    return MGLtf{r.hi, r.lo, 0.0f};
}
MGL_INL uint MGLtf_to_uint(MGLtf a) { return uint(a.x + a.y + a.z); }
MGL_INL MGLtf MGLtf_sign(MGLtf a) {
    float s = a.x + a.y + a.z;
    if (s > 0.0f) return MGLtf{1.0f, 0.0f, 0.0f};
    if (s < 0.0f) return MGLtf{-1.0f, 0.0f, 0.0f};
    return MGLtf{0.0f, 0.0f, 0.0f};
}

// ---- Rounding ----
MGL_INL MGLtf MGLtf_floor(MGLtf a)     { float f = floor(a.x + a.y + a.z); return MGLtf{f, 0.0f, 0.0f}; }
MGL_INL MGLtf MGLtf_ceil(MGLtf a)      { float f = ceil(a.x + a.y + a.z);  return MGLtf{f, 0.0f, 0.0f}; }
MGL_INL MGLtf MGLtf_trunc(MGLtf a)     { float f = trunc(a.x + a.y + a.z); return MGLtf{f, 0.0f, 0.0f}; }
MGL_INL MGLtf MGLtf_round(MGLtf a)     { float f = round(a.x + a.y + a.z); return MGLtf{f, 0.0f, 0.0f}; }
MGL_INL MGLtf MGLtf_roundeven(MGLtf a) { float f = rint(a.x + a.y + a.z);  return MGLtf{f, 0.0f, 0.0f}; }

// ---- Decode / Encode (IEEE binary64 <-> MGLtf) ----
MGL_INL MGLtf MGLtf_decode(uint2 bits) {
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
MGL_INL uint2 MGLtf_encode(MGLtf v) {
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

)";

// ---------------------------------------------------------------------------
// Vectors and matrices
//
// An emulated double is a struct, so a dvec is a struct of those and a dmat a
// struct of columns. Metal has no operators for them, so every GLSL operation
// becomes a call, one component at a time. All of it is written out here
// rather than kept as a fixed string, because the same twelve lines would
// otherwise be repeated for every width.
// ---------------------------------------------------------------------------

static const char *comp_name(uint32_t i)
{
	static const char *names[] = { "x", "y", "z", "w" };
	return names[i];
}

std::string get_fp64_composite_type_name(FP64Mode mode, uint32_t vecsize, uint32_t columns)
{
	const char *base = get_fp64_type_name(mode);
	if (!base)
		return {};

	std::string name = base;
	if (columns > 1)
		name += std::to_string(columns) + "x";
	if (vecsize > 1)
		name += std::to_string(vecsize);
	return name;
}

namespace
{
// Builds MSL call expressions so the generators below read like the maths.
struct Emitter
{
	std::string P;    // the scalar type, MGLtf or MGLdf
	std::string out;

	void line(const std::string &s) { out += s; out += "\n"; }

	std::string call(const std::string &fn, const std::string &a) const { return P + "_" + fn + "(" + a + ")"; }
	std::string call(const std::string &fn, const std::string &a, const std::string &b) const
	{
		return P + "_" + fn + "(" + a + ", " + b + ")";
	}
	std::string call(const std::string &fn, const std::string &a, const std::string &b, const std::string &c) const
	{
		return P + "_" + fn + "(" + a + ", " + b + ", " + c + ")";
	}
	std::string add(const std::string &a, const std::string &b) const { return call("add", a, b); }
	std::string sub(const std::string &a, const std::string &b) const { return call("sub", a, b); }
	std::string mul(const std::string &a, const std::string &b) const { return call("mul", a, b); }
	std::string div(const std::string &a, const std::string &b) const { return call("div", a, b); }
	std::string neg(const std::string &a) const { return call("neg", a); }
	std::string lit(float v) const
	{
		char buf[64];
		snprintf(buf, sizeof(buf), "%s_from_float(%.1ff)", P.c_str(), v);
		return buf;
	}

	std::string vec(uint32_t n) const { return n > 1 ? P + std::to_string(n) : P; }
	std::string mat(uint32_t cols, uint32_t rows) const
	{
		return P + std::to_string(cols) + "x" + std::to_string(rows);
	}
	// component i of a value that may be a scalar (then there is one component)
	std::string at(const std::string &v, uint32_t i, uint32_t n) const
	{
		return n > 1 ? v + "." + comp_name(i) : v;
	}
	std::string col(const std::string &m, uint32_t i) const { return m + ".c" + std::to_string(i); }

	// Wraps per-component expressions in a constructor for the result width.
	std::string build(uint32_t n, const std::vector<std::string> &parts) const
	{
		if (n == 1)
			return parts[0];
		std::string s = vec(n) + "{";
		for (uint32_t i = 0; i < n; i++)
			s += (i ? ", " : "") + parts[i];
		return s + "}";
	}
	std::string buildMat(uint32_t cols, uint32_t rows, const std::vector<std::string> &parts) const
	{
		std::string s = mat(cols, rows) + "{";
		for (uint32_t i = 0; i < cols; i++)
			s += (i ? ", " : "") + parts[i];
		return s + "}";
	}
};
} // namespace

std::string get_fp64_composite_prelude(FP64Mode mode, bool want_vectors, bool want_matrices)
{
	if (mode == FP64Mode::Disabled)
		return {};

	Emitter e;
	e.P = get_fp64_type_name(mode);
	const std::string &P = e.P;

	// ---- scalar operations the fixed prelude does not carry ----
	e.line("// ---- Scalar operations built on the primitives above ----");
	e.line("MGL_INL " + P + " " + P + "_fract(" + P + " a) { return " + e.sub("a", e.call("floor", "a")) +
	       "; }");
	e.line("MGL_INL " + P + " " + P + "_mod(" + P + " a, " + P + " b) { return " +
	       e.sub("a", e.mul("b", e.call("floor", e.div("a", "b")))) + "; }");
	// GLSL's mod floors; C's fmod truncates. SPIR-V has an opcode for each.
	e.line("MGL_INL " + P + " " + P + "_rem(" + P + " a, " + P + " b) { return " +
	       e.sub("a", e.mul("b", e.call("trunc", e.div("a", "b")))) + "; }");
	e.line("MGL_INL " + P + " " + P + "_inversesqrt(" + P + " a) { return " +
	       e.div(e.lit(1.0f), e.call("sqrt", "a")) + "; }");
	e.line("MGL_INL " + P + " " + P + "_mix(" + P + " a, " + P + " b, " + P + " t) { return " +
	       e.add("a", e.mul(e.sub("b", "a"), "t")) + "; }");
	e.line("MGL_INL " + P + " " + P + "_clamp(" + P + " a, " + P + " lo, " + P + " hi) { return " +
	       e.call("min", e.call("max", "a", "lo"), "hi") + "; }");
	e.line("MGL_INL " + P + " " + P + "_step(" + P + " edge, " + P + " a) { return " + P +
	       "_lt(a, edge) ? " + e.lit(0.0f) + " : " + e.lit(1.0f) + "; }");
	e.line("MGL_INL " + P + " " + P + "_smoothstep(" + P + " e0, " + P + " e1, " + P + " a) {");
	e.line("    " + P + " t = " + e.call("clamp", e.div(e.sub("a", "e0"), e.sub("e1", "e0")), e.lit(0.0f),
	                                     e.lit(1.0f)) + ";");
	e.line("    return " + e.mul(e.mul("t", "t"), e.sub(e.lit(3.0f), e.mul(e.lit(2.0f), "t"))) + "; }");
	e.line("MGL_INL " + P + " " + P + "_fma(" + P + " a, " + P + " b, " + P + " c) { return " +
	       e.add(e.mul("a", "b"), "c") + "; }");
	// A value is the sum of its terms, so the leading term decides these.
	e.line("MGL_INL bool " + P + "_isnan(" + P + " a) { return isnan(" + (mode == FP64Mode::TripleFloat ? "a.x" : "a.hi") + "); }");
	e.line("MGL_INL bool " + P + "_isinf(" + P + " a) { return isinf(" + (mode == FP64Mode::TripleFloat ? "a.x" : "a.hi") + "); }");
	if (mode == FP64Mode::TripleFloat)
	{
		e.line("MGL_INL " + P + " " + P + "_ldexp(" + P + " a, int n) { return " + P +
		       "_canon(" + P + "{ldexp(a.x, n), ldexp(a.y, n), ldexp(a.z, n)}); }");
		e.line("MGL_INL " + P + " " + P + "_frexp(" + P + " a, thread int &n) {");
		e.line("    if (a.x == 0.0f) { n = 0; return a; }");
		e.line("    n = ilogb(a.x) + 1;");
		e.line("    return " + P + "_ldexp(a, -n); }");
	}
	else
	{
		e.line("MGL_INL " + P + " " + P + "_ldexp(" + P + " a, int n) { return " + P +
		       "{ldexp(a.hi, n), ldexp(a.lo, n)}; }");
		e.line("MGL_INL " + P + " " + P + "_frexp(" + P + " a, thread int &n) {");
		e.line("    if (a.hi == 0.0f) { n = 0; return a; }");
		e.line("    n = ilogb(a.hi) + 1;");
		e.line("    return " + P + "_ldexp(a, -n); }");
	}
	e.line("MGL_INL " + P + " " + P + "_modf(" + P + " a, thread " + P + " &ip) {");
	e.line("    ip = " + e.call("trunc", "a") + "; return " + e.sub("a", "ip") + "; }");
	// The unordered forms are true whenever either side is a NaN. Unordered
	// not-equal needs no wrapper: not-equal already answers true there.
	for (const char *c : { "eq", "lt", "le", "gt", "ge" })
		e.line("MGL_INL bool " + P + "_un_" + c + "(" + P + " a, " + P + " b) { return " + P +
		       "_isnan(a) || " + P + "_isnan(b) || " + P + "_" + c + "(a, b); }");
	e.line("MGL_INL bool " + P + "_un_ne(" + P + " a, " + P + " b) { return " + P + "_ne(a, b); }");
	e.line("");

	// Everything above is scalar and a plain double needs it: glslang turns
	// "!=" on floats into an unordered compare, so a shader with no dvec at all
	// still calls _un_ne.
	if (!want_vectors)
		return e.out;

	// ---- vector types ----
	e.line("// ---- Vector types ----");
	for (uint32_t n = 2; n <= 4; n++)
	{
		std::string V = e.vec(n);
		std::string members;
		for (uint32_t i = 0; i < n; i++)
			members += (i ? ", " : "") + std::string(comp_name(i));

		e.line("struct " + V + " {");
		e.line("    " + P + " " + members + ";");
		// A dynamic index into a vector has to resolve to a member.
		std::string chain = "z";
		if (n == 2)
			chain = "y";
		else if (n == 4)
			chain = "(i == 2 ? z : w)";
		std::string body = n == 2 ? "return i == 0 ? x : y;" :
		                            "return i == 0 ? x : (i == 1 ? y : " + chain + ");";
		// newer Metal compilers want the address space of 'this' spelled out
		e.line("    thread " + P + " &operator[](int i) thread { " + body + " }");
		e.line("    const thread " + P + " &operator[](int i) const thread { " + body + " }");
		e.line("    device " + P + " &operator[](int i) device { " + body + " }");
		e.line("    const device " + P + " &operator[](int i) const device { " + body + " }");
		e.line("    const constant " + P + " &operator[](int i) const constant { " + body + " }");
		e.line("};");
	}
	e.line("");

	// ---- component-wise vector operations ----
	struct Unary { const char *name; };
	static const Unary unaries[] = { { "neg" },   { "abs" },       { "sqrt" },  { "floor" }, { "ceil" },
		                             { "trunc" }, { "round" },     { "roundeven" }, { "sign" },
		                             { "fract" }, { "inversesqrt" } };
	static const char *binaries[] = { "add", "sub", "mul", "div", "min", "max", "mod", "rem" };
	static const char *compares[] = { "eq",    "ne",    "lt",    "le",    "gt",    "ge",
		                              "un_eq", "un_ne", "un_lt", "un_le", "un_gt", "un_ge" };
	static const char *converts[] = { "float", "int", "uint" };

	e.line("// ---- Vector operations ----");
	for (uint32_t n = 2; n <= 4; n++)
	{
		const std::string V = e.vec(n);
		const std::string B = "bool" + std::to_string(n);

		for (auto &u : unaries)
		{
			std::vector<std::string> parts;
			for (uint32_t i = 0; i < n; i++)
				parts.push_back(e.call(u.name, e.at("a", i, n)));
			e.line("MGL_INL " + V + " " + P + "_" + u.name + "(" + V + " a) { return " + e.build(n, parts) +
			       "; }");
		}

		for (auto *b : binaries)
		{
			std::vector<std::string> vv, vs, sv;
			for (uint32_t i = 0; i < n; i++)
			{
				vv.push_back(e.call(b, e.at("a", i, n), e.at("b", i, n)));
				vs.push_back(e.call(b, e.at("a", i, n), "b"));
				sv.push_back(e.call(b, "a", e.at("b", i, n)));
			}
			e.line("MGL_INL " + V + " " + P + "_" + b + "(" + V + " a, " + V + " b) { return " +
			       e.build(n, vv) + "; }");
			e.line("MGL_INL " + V + " " + P + "_" + b + "(" + V + " a, " + P + " b) { return " +
			       e.build(n, vs) + "; }");
			e.line("MGL_INL " + V + " " + P + "_" + b + "(" + P + " a, " + V + " b) { return " +
			       e.build(n, sv) + "; }");
		}

		for (auto *c : compares)
		{
			std::string s = "MGL_INL " + B + " " + P + "_" + c + "(" + V + " a, " + V + " b) { return " + B +
			                "(";
			for (uint32_t i = 0; i < n; i++)
				s += (i ? ", " : "") + e.call(c, e.at("a", i, n), e.at("b", i, n));
			e.line(s + "); }");
		}

		for (auto *c : converts)
		{
			std::string native = std::string(c) + std::to_string(n);
			std::vector<std::string> from;
			std::string to = native + "(";
			for (uint32_t i = 0; i < n; i++)
			{
				from.push_back(e.call(std::string("from_") + c, e.at("a", i, n)));
				to += (i ? ", " : "") + e.call(std::string("to_") + c, e.at("a", i, n));
			}
			e.line("MGL_INL " + V + " " + P + "_from_" + c + "(" + native + " a) { return " +
			       e.build(n, from) + "; }");
			e.line("MGL_INL " + native + " " + P + "_to_" + c + "(" + V + " a) { return " + to + "); }");
		}

		// three-operand builtins, with the scalar-tail forms GLSL also allows
		static const char *ternaries[] = { "clamp", "mix", "smoothstep", "fma" };
		for (auto *t : ternaries)
		{
			std::vector<std::string> vvv, vvs;
			for (uint32_t i = 0; i < n; i++)
			{
				vvv.push_back(e.call(t, e.at("a", i, n), e.at("b", i, n), e.at("c", i, n)));
				vvs.push_back(e.call(t, e.at("a", i, n), "b", "c"));
			}
			e.line("MGL_INL " + V + " " + P + "_" + t + "(" + V + " a, " + V + " b, " + V + " c) { return " +
			       e.build(n, vvv) + "; }");
			e.line("MGL_INL " + V + " " + P + "_" + t + "(" + V + " a, " + P + " b, " + P + " c) { return " +
			       e.build(n, vvs) + "; }");
			// mix and smoothstep also take a scalar only in the last operand
			std::vector<std::string> vvt;
			for (uint32_t i = 0; i < n; i++)
				vvt.push_back(e.call(t, e.at("a", i, n), e.at("b", i, n), "c"));
			e.line("MGL_INL " + V + " " + P + "_" + t + "(" + V + " a, " + V + " b, " + P + " c) { return " +
			       e.build(n, vvt) + "; }");
		}

		// step takes its edge first, so the scalar form is the other way round
		{
			std::vector<std::string> vv, sv;
			for (uint32_t i = 0; i < n; i++)
			{
				vv.push_back(e.call("step", e.at("edge", i, n), e.at("a", i, n)));
				sv.push_back(e.call("step", "edge", e.at("a", i, n)));
			}
			e.line("MGL_INL " + V + " " + P + "_step(" + V + " edge, " + V + " a) { return " +
			       e.build(n, vv) + "; }");
			e.line("MGL_INL " + V + " " + P + "_step(" + P + " edge, " + V + " a) { return " +
			       e.build(n, sv) + "; }");
		}

		// select on a boolean vector, which is how GLSL's mix(a, b, bvec) arrives
		{
			std::string s = "MGL_INL " + V + " " + P + "_select(" + V + " a, " + V + " b, " + B +
			                " t) { return " + V + "{";
			for (uint32_t i = 0; i < n; i++)
				s += std::string(i ? ", " : "") + "(t." + comp_name(i) + " ? b." + comp_name(i) + " : a." +
				     comp_name(i) + ")";
			e.line(s + "}; }");
		}

		// isnan / isinf answer per component
		for (const char *q : { "isnan", "isinf" })
		{
			std::string s = "MGL_INL " + B + " " + P + "_" + q + "(" + V + " a) { return " + B + "(";
			for (uint32_t i = 0; i < n; i++)
				s += (i ? ", " : "") + e.call(q, e.at("a", i, n));
			e.line(s + "); }");
		}

		// ldexp, frexp and modf carry a second, non-double operand
		{
			std::vector<std::string> parts;
			for (uint32_t i = 0; i < n; i++)
				parts.push_back(e.call("ldexp", e.at("a", i, n), "n." + std::string(comp_name(i))));
			e.line("MGL_INL " + V + " " + P + "_ldexp(" + V + " a, int" + std::to_string(n) +
			       " n) { return " + e.build(n, parts) + "; }");

			std::string s = "MGL_INL " + V + " " + P + "_frexp(" + V + " a, thread int" +
			                std::to_string(n) + " &n) { " + V + " r;";
			for (uint32_t i = 0; i < n; i++)
			{
				std::string c = comp_name(i);
				s += " int e" + std::to_string(i) + "; r." + c + " = " + P + "_frexp(a." + c + ", e" +
				     std::to_string(i) + "); n." + c + " = e" + std::to_string(i) + ";";
			}
			e.line(s + " return r; }");

			s = "MGL_INL " + V + " " + P + "_modf(" + V + " a, thread " + V + " &ip) { return " + V + "{";
			for (uint32_t i = 0; i < n; i++)
			{
				std::string c = comp_name(i);
				s += std::string(i ? ", " : "") + P + "_modf(a." + c + ", ip." + c + ")";
			}
			e.line(s + "}; }");
		}

		// geometric functions
		{
			std::string dot;
			for (uint32_t i = 0; i < n; i++)
			{
				std::string term = e.mul(e.at("a", i, n), e.at("b", i, n));
				dot = i ? e.add(dot, term) : term;
			}
			e.line("MGL_INL " + P + " " + P + "_dot" + std::to_string(n) + "(" + V + " a, " + V +
			       " b) { return " + dot + "; }");
			e.line("MGL_INL " + P + " " + P + "_length" + std::to_string(n) + "(" + V + " a) { return " +
			       e.call("sqrt", P + "_dot" + std::to_string(n) + "(a, a)") + "; }");
			e.line("MGL_INL " + P + " " + P + "_distance" + std::to_string(n) + "(" + V + " a, " + V +
			       " b) { return " + P + "_length" + std::to_string(n) + "(" + P + "_sub(a, b)); }");
			e.line("MGL_INL " + V + " " + P + "_normalize" + std::to_string(n) + "(" + V +
			       " a) { return " + P + "_mul(a, " + e.div(e.lit(1.0f), P + "_length" + std::to_string(n) + "(a)") +
			       "); }");
			// reflect: a - 2 * dot(nrm, a) * nrm
			e.line("MGL_INL " + V + " " + P + "_reflect" + std::to_string(n) + "(" + V + " a, " + V +
			       " nrm) { return " + P + "_sub(a, " + P + "_mul(nrm, " +
			       e.mul(e.lit(2.0f), P + "_dot" + std::to_string(n) + "(nrm, a)") + ")); }");
			// faceforward: dot(ref, i) < 0 ? n : -n
			e.line("MGL_INL " + V + " " + P + "_faceforward" + std::to_string(n) + "(" + V + " nrm, " + V +
			       " i, " + V + " ref) { return " + P + "_lt(" + P + "_dot" + std::to_string(n) +
			       "(ref, i), " + e.lit(0.0f) + ") ? nrm : " + P + "_neg(nrm); }");
			e.line("MGL_INL " + V + " " + P + "_refract" + std::to_string(n) + "(" + V + " i, " + V +
			       " nrm, " + P + " eta) {");
			e.line("    " + P + " d = " + P + "_dot" + std::to_string(n) + "(nrm, i);");
			e.line("    " + P + " k = " + e.sub(e.lit(1.0f), e.mul(e.mul("eta", "eta"),
			                                                       e.sub(e.lit(1.0f), e.mul("d", "d")))) + ";");
			e.line("    if (" + P + "_lt(k, " + e.lit(0.0f) + ")) return " + V + "{" +
			       [&] { std::string s; for (uint32_t i = 0; i < n; i++) s += (i ? ", " : "") + e.lit(0.0f); return s; }() +
			       "};");
			e.line("    return " + P + "_sub(" + P + "_mul(i, eta), " + P + "_mul(nrm, " +
			       e.add(e.mul("eta", "d"), e.call("sqrt", "k")) + ")); }");
		}
		if (n == 3)
		{
			std::vector<std::string> parts;
			static const uint32_t a1[3] = { 1, 2, 0 }, a2[3] = { 2, 0, 1 };
			for (uint32_t i = 0; i < 3; i++)
				parts.push_back(e.sub(e.mul("a." + std::string(comp_name(a1[i])), "b." + std::string(comp_name(a2[i]))),
				                      e.mul("a." + std::string(comp_name(a2[i])), "b." + std::string(comp_name(a1[i])))));
			e.line("MGL_INL " + V + " " + P + "_cross(" + V + " a, " + V + " b) { return " + e.build(3, parts) +
			       "; }");
		}
		e.line("");
	}

	if (!want_matrices)
		return e.out;

	// ---- matrix types ----
	e.line("// ---- Matrix types: c0..cN are columns, as in GLSL ----");
	for (uint32_t cols = 2; cols <= 4; cols++)
	{
		for (uint32_t rows = 2; rows <= 4; rows++)
		{
			std::string M = e.mat(cols, rows), V = e.vec(rows);
			e.line("struct " + M + " {");
			std::string members;
			for (uint32_t i = 0; i < cols; i++)
				members += (i ? ", " : "") + ("c" + std::to_string(i));
			e.line("    " + V + " " + members + ";");
			std::string body;
			if (cols == 2)
				body = "return i == 0 ? c0 : c1;";
			else if (cols == 3)
				body = "return i == 0 ? c0 : (i == 1 ? c1 : c2);";
			else
				body = "return i == 0 ? c0 : (i == 1 ? c1 : (i == 2 ? c2 : c3));";
			e.line("    thread " + V + " &operator[](int i) thread { " + body + " }");
			e.line("    const thread " + V + " &operator[](int i) const thread { " + body + " }");
			e.line("    device " + V + " &operator[](int i) device { " + body + " }");
			e.line("    const device " + V + " &operator[](int i) const device { " + body + " }");
			e.line("    const constant " + V + " &operator[](int i) const constant { " + body + " }");
			e.line("};");
		}
	}
	e.line("");

	e.line("// ---- Matrix operations ----");
	for (uint32_t cols = 2; cols <= 4; cols++)
	{
		for (uint32_t rows = 2; rows <= 4; rows++)
		{
			const std::string M = e.mat(cols, rows), VR = e.vec(rows), VC = e.vec(cols);

			// Component-wise. The linear-algebra product is _mul, below, so the
			// component-wise multiply has to answer to a different name.
			for (const char *b : { "add", "sub", "div", "compmult" })
			{
				const char *op = strcmp(b, "compmult") == 0 ? "mul" : b;
				std::vector<std::string> parts;
				for (uint32_t i = 0; i < cols; i++)
					parts.push_back(e.call(op, e.col("a", i), e.col("b", i)));
				e.line("MGL_INL " + M + " " + P + "_" + b + "(" + M + " a, " + M + " b) { return " +
				       e.buildMat(cols, rows, parts) + "; }");
			}
			{
				std::vector<std::string> parts;
				for (uint32_t i = 0; i < cols; i++)
					parts.push_back(e.call("neg", e.col("a", i)));
				e.line("MGL_INL " + M + " " + P + "_neg(" + M + " a) { return " +
				       e.buildMat(cols, rows, parts) + "; }");
			}
			{
				std::vector<std::string> ms, sm;
				for (uint32_t i = 0; i < cols; i++)
				{
					ms.push_back(e.call("mul", e.col("a", i), "b"));
					sm.push_back(e.call("mul", "a", e.col("b", i)));
				}
				e.line("MGL_INL " + M + " " + P + "_mul(" + M + " a, " + P + " b) { return " +
				       e.buildMat(cols, rows, ms) + "; }");
				e.line("MGL_INL " + M + " " + P + "_mul(" + P + " a, " + M + " b) { return " +
				       e.buildMat(cols, rows, sm) + "; }");
			}
			// matrix * vector: sum of columns scaled by the vector's components
			{
				std::vector<std::string> parts;
				for (uint32_t r = 0; r < rows; r++)
				{
					std::string sum;
					for (uint32_t c = 0; c < cols; c++)
					{
						std::string term = e.mul(e.col("a", c) + "." + comp_name(r), "b." + std::string(comp_name(c)));
						sum = c ? e.add(sum, term) : term;
					}
					parts.push_back(sum);
				}
				e.line("MGL_INL " + VR + " " + P + "_mul(" + M + " a, " + VC + " b) { return " +
				       e.build(rows, parts) + "; }");
			}
			// vector * matrix: each result component is a dot with one column
			{
				std::vector<std::string> parts;
				for (uint32_t c = 0; c < cols; c++)
				{
					std::string sum;
					for (uint32_t r = 0; r < rows; r++)
					{
						std::string term = e.mul("a." + std::string(comp_name(r)), e.col("b", c) + "." + comp_name(r));
						sum = r ? e.add(sum, term) : term;
					}
					parts.push_back(sum);
				}
				e.line("MGL_INL " + VC + " " + P + "_mul(" + VR + " a, " + M + " b) { return " +
				       e.build(cols, parts) + "; }");
			}
			// transpose
			{
				std::vector<std::string> parts;
				for (uint32_t r = 0; r < rows; r++)
				{
					std::vector<std::string> comps;
					for (uint32_t c = 0; c < cols; c++)
						comps.push_back(e.col("a", c) + "." + comp_name(r));
					parts.push_back(e.build(cols, comps));
				}
				e.line("MGL_INL " + e.mat(rows, cols) + " " + P + "_transpose(" + M + " a) { return " +
				       e.buildMat(rows, cols, parts) + "; }");
			}
			// outer product of a rows-vector and a cols-vector makes this shape
			{
				std::vector<std::string> parts;
				for (uint32_t c = 0; c < cols; c++)
				{
					std::vector<std::string> comps;
					for (uint32_t r = 0; r < rows; r++)
						comps.push_back(e.mul("a." + std::string(comp_name(r)), "b." + std::string(comp_name(c))));
					parts.push_back(e.build(rows, comps));
				}
				e.line("MGL_INL " + M + " " + P + "_outer(" + VR + " a, " + VC + " b) { return " +
				       e.buildMat(cols, rows, parts) + "; }");
			}
		}
	}

	// matrix * matrix, for every pair whose inner sizes agree
	for (uint32_t cols = 2; cols <= 4; cols++)
		for (uint32_t rows = 2; rows <= 4; rows++)
			for (uint32_t bcols = 2; bcols <= 4; bcols++)
			{
				// a is cols x rows, b is bcols x cols, result is bcols x rows
				std::vector<std::string> parts;
				for (uint32_t c = 0; c < bcols; c++)
				{
					std::vector<std::string> comps;
					for (uint32_t r = 0; r < rows; r++)
					{
						std::string sum;
						for (uint32_t k = 0; k < cols; k++)
						{
							std::string term = e.mul(e.col("a", k) + "." + comp_name(r),
							                         e.col("b", c) + "." + comp_name(k));
							sum = k ? e.add(sum, term) : term;
						}
						comps.push_back(sum);
					}
					parts.push_back(e.build(rows, comps));
				}
				e.line("MGL_INL " + e.mat(bcols, rows) + " " + P + "_mul(" + e.mat(cols, rows) + " a, " +
				       e.mat(bcols, cols) + " b) { return " + e.buildMat(bcols, rows, parts) + "; }");
			}

	// determinant and inverse, square only
	for (uint32_t n = 2; n <= 4; n++)
	{
		const std::string M = e.mat(n, n);
		auto el = [&](const std::string &m, uint32_t c, uint32_t r) {
			return e.col(m, c) + "." + comp_name(r);
		};
		// minor determinant of the submatrix that drops the listed rows/columns
		std::function<std::string(const std::vector<uint32_t> &, const std::vector<uint32_t> &)> det =
		    [&](const std::vector<uint32_t> &cs, const std::vector<uint32_t> &rs) -> std::string {
			if (cs.size() == 1)
				return el("a", cs[0], rs[0]);
			std::string sum;
			for (size_t i = 0; i < cs.size(); i++)
			{
				std::vector<uint32_t> sub_c, sub_r(rs.begin() + 1, rs.end());
				for (size_t j = 0; j < cs.size(); j++)
					if (j != i)
						sub_c.push_back(cs[j]);
				std::string term = e.mul(el("a", cs[i], rs[0]), det(sub_c, sub_r));
				if (i == 0)
					sum = term;
				else
					sum = (i % 2) ? e.sub(sum, term) : e.add(sum, term);
			}
			return sum;
		};
		std::vector<uint32_t> all;
		for (uint32_t i = 0; i < n; i++)
			all.push_back(i);
		e.line("MGL_INL " + P + " " + P + "_determinant(" + M + " a) { return " + det(all, all) + "; }");

		// inverse from the adjugate; the cofactor of (c,r) lands at (r,c)
		std::vector<std::string> parts;
		for (uint32_t c = 0; c < n; c++)
		{
			std::vector<std::string> comps;
			for (uint32_t r = 0; r < n; r++)
			{
				// transposed: entry (c,r) of the inverse is cofactor (r,c)
				std::vector<uint32_t> sub_c, sub_r;
				for (uint32_t i = 0; i < n; i++)
				{
					if (i != r)
						sub_c.push_back(i);
					if (i != c)
						sub_r.push_back(i);
				}
				std::string cof = det(sub_c, sub_r);
				if ((r + c) % 2)
					cof = e.neg(cof);
				comps.push_back(e.mul(cof, "inv_det"));
			}
			parts.push_back(e.build(n, comps));
		}
		e.line("MGL_INL " + M + " " + P + "_inverse(" + M + " a) {");
		e.line("    " + P + " inv_det = " + e.div(e.lit(1.0f), P + "_determinant(a)") + ";");
		e.line("    return " + e.buildMat(n, n, parts) + "; }");
	}
	e.line("");

	return e.out;
}

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

/**
 * Zero-Knowledge Proof Elliptic Curve Operations Implementation
 * 
 * Implements elliptic curve operations for ZK proof systems.
 * 
 * Requirements: 19, 20.2
 */

#include "zk_elliptic_curve.h"
#include <algorithm>
#include <cstring>
#include <random>

namespace fhe_accelerate {
namespace zk {

// ============================================================================
// Point Comparison
// ============================================================================

bool AffinePoint256::operator==(const AffinePoint256& other) const {
    if (is_infinity && other.is_infinity) return true;
    if (is_infinity != other.is_infinity) return false;
    return x == other.x && y == other.y;
}

bool AffinePoint384::operator==(const AffinePoint384& other) const {
    if (is_infinity && other.is_infinity) return true;
    if (is_infinity != other.is_infinity) return false;
    return x == other.x && y == other.y;
}

// ============================================================================
// Coordinate Conversions
// ============================================================================

ProjectivePoint256 ProjectivePoint256::from_affine(const AffinePoint256& p, 
                                                    const Field256& field) {
    if (p.is_infinity) {
        return ProjectivePoint256();
    }
    return ProjectivePoint256(p.x, p.y, field.one());
}

JacobianPoint256 JacobianPoint256::from_affine(const AffinePoint256& p,
                                                const Field256& field) {
    if (p.is_infinity) {
        return JacobianPoint256();
    }
    return JacobianPoint256(p.x, p.y, field.one());
}

JacobianPoint384 JacobianPoint384::from_affine(const AffinePoint384& p,
                                                const Field384& field) {
    if (p.is_infinity) {
        return JacobianPoint384();
    }
    return JacobianPoint384(p.x, p.y, field.one());
}


// ============================================================================
// EllipticCurve256 Implementation
// ============================================================================

EllipticCurve256::EllipticCurve256(const FieldElement256& a, const FieldElement256& b,
                                   const Field256& field)
    : a_(a), b_(b), field_(field) {
    a_is_zero_ = a_.is_zero();
}

JacobianPoint256 EllipticCurve256::add(const JacobianPoint256& p, 
                                        const JacobianPoint256& q) const {
    // Handle infinity cases
    if (p.is_infinity()) return q;
    if (q.is_infinity()) return p;
    
    // Jacobian addition formula
    // U1 = X1*Z2^2, U2 = X2*Z1^2
    // S1 = Y1*Z2^3, S2 = Y2*Z1^3
    // H = U2 - U1, R = S2 - S1
    // X3 = R^2 - H^3 - 2*U1*H^2
    // Y3 = R*(U1*H^2 - X3) - S1*H^3
    // Z3 = H*Z1*Z2
    
    FieldElement256 z1_sq = field_.square(p.z);
    FieldElement256 z2_sq = field_.square(q.z);
    
    FieldElement256 u1 = field_.mul(p.x, z2_sq);
    FieldElement256 u2 = field_.mul(q.x, z1_sq);
    
    FieldElement256 z1_cu = field_.mul(z1_sq, p.z);
    FieldElement256 z2_cu = field_.mul(z2_sq, q.z);
    
    FieldElement256 s1 = field_.mul(p.y, z2_cu);
    FieldElement256 s2 = field_.mul(q.y, z1_cu);
    
    // Check if points are equal or negatives
    if (u1 == u2) {
        if (s1 == s2) {
            return double_point(p);
        } else {
            return JacobianPoint256();  // Point at infinity
        }
    }
    
    FieldElement256 h = field_.sub(u2, u1);
    FieldElement256 r = field_.sub(s2, s1);
    
    FieldElement256 h_sq = field_.square(h);
    FieldElement256 h_cu = field_.mul(h_sq, h);
    
    FieldElement256 u1_h_sq = field_.mul(u1, h_sq);
    
    // X3 = R^2 - H^3 - 2*U1*H^2
    FieldElement256 r_sq = field_.square(r);
    FieldElement256 x3 = field_.sub(r_sq, h_cu);
    x3 = field_.sub(x3, u1_h_sq);
    x3 = field_.sub(x3, u1_h_sq);
    
    // Y3 = R*(U1*H^2 - X3) - S1*H^3
    FieldElement256 y3 = field_.sub(u1_h_sq, x3);
    y3 = field_.mul(r, y3);
    FieldElement256 s1_h_cu = field_.mul(s1, h_cu);
    y3 = field_.sub(y3, s1_h_cu);
    
    // Z3 = H*Z1*Z2
    FieldElement256 z3 = field_.mul(h, p.z);
    z3 = field_.mul(z3, q.z);
    
    return JacobianPoint256(x3, y3, z3);
}

JacobianPoint256 EllipticCurve256::add_mixed(const JacobianPoint256& p,
                                              const AffinePoint256& q) const {
    if (p.is_infinity()) {
        return JacobianPoint256::from_affine(q, field_);
    }
    if (q.is_infinity) {
        return p;
    }
    
    // Mixed addition (Z2 = 1)
    FieldElement256 z1_sq = field_.square(p.z);
    FieldElement256 u1 = p.x;
    FieldElement256 u2 = field_.mul(q.x, z1_sq);
    
    FieldElement256 z1_cu = field_.mul(z1_sq, p.z);
    FieldElement256 s1 = p.y;
    FieldElement256 s2 = field_.mul(q.y, z1_cu);
    
    if (u1 == u2) {
        if (s1 == s2) {
            return double_point(p);
        } else {
            return JacobianPoint256();
        }
    }
    
    FieldElement256 h = field_.sub(u2, u1);
    FieldElement256 r = field_.sub(s2, s1);
    
    FieldElement256 h_sq = field_.square(h);
    FieldElement256 h_cu = field_.mul(h_sq, h);
    FieldElement256 u1_h_sq = field_.mul(u1, h_sq);
    
    FieldElement256 r_sq = field_.square(r);
    FieldElement256 x3 = field_.sub(r_sq, h_cu);
    x3 = field_.sub(x3, u1_h_sq);
    x3 = field_.sub(x3, u1_h_sq);
    
    FieldElement256 y3 = field_.sub(u1_h_sq, x3);
    y3 = field_.mul(r, y3);
    FieldElement256 s1_h_cu = field_.mul(s1, h_cu);
    y3 = field_.sub(y3, s1_h_cu);
    
    FieldElement256 z3 = field_.mul(h, p.z);
    
    return JacobianPoint256(x3, y3, z3);
}


JacobianPoint256 EllipticCurve256::double_point(const JacobianPoint256& p) const {
    if (p.is_infinity()) return p;
    
    // Check if Y = 0 (point of order 2)
    if (p.y.is_zero()) {
        return JacobianPoint256();
    }
    
    // Jacobian doubling formula (optimized for a=0)
    // For a=0: 
    //   XX = X^2, YY = Y^2, YYYY = YY^2
    //   S = 2*((X+YY)^2 - XX - YYYY)
    //   M = 3*XX
    //   X3 = M^2 - 2*S
    //   Y3 = M*(S - X3) - 8*YYYY
    //   Z3 = 2*Y*Z
    
    FieldElement256 xx = field_.square(p.x);
    FieldElement256 yy = field_.square(p.y);
    FieldElement256 yyyy = field_.square(yy);
    
    // S = 2*((X+YY)^2 - XX - YYYY)
    FieldElement256 x_plus_yy = field_.add(p.x, yy);
    FieldElement256 s = field_.square(x_plus_yy);
    s = field_.sub(s, xx);
    s = field_.sub(s, yyyy);
    s = field_.add(s, s);  // 2*S
    
    // M = 3*XX (for a=0) or 3*XX + a*Z^4 (general)
    FieldElement256 m = field_.add(xx, xx);
    m = field_.add(m, xx);  // 3*XX
    
    if (!a_is_zero_) {
        FieldElement256 z_sq = field_.square(p.z);
        FieldElement256 z_4 = field_.square(z_sq);
        FieldElement256 a_z4 = field_.mul(a_, z_4);
        m = field_.add(m, a_z4);
    }
    
    // X3 = M^2 - 2*S
    FieldElement256 x3 = field_.square(m);
    x3 = field_.sub(x3, s);
    x3 = field_.sub(x3, s);
    
    // Y3 = M*(S - X3) - 8*YYYY
    FieldElement256 y3 = field_.sub(s, x3);
    y3 = field_.mul(m, y3);
    FieldElement256 yyyy_8 = field_.add(yyyy, yyyy);  // 2
    yyyy_8 = field_.add(yyyy_8, yyyy_8);              // 4
    yyyy_8 = field_.add(yyyy_8, yyyy_8);              // 8
    y3 = field_.sub(y3, yyyy_8);
    
    // Z3 = 2*Y*Z
    FieldElement256 z3 = field_.mul(p.y, p.z);
    z3 = field_.add(z3, z3);
    
    return JacobianPoint256(x3, y3, z3);
}

JacobianPoint256 EllipticCurve256::negate(const JacobianPoint256& p) const {
    if (p.is_infinity()) return p;
    return JacobianPoint256(p.x, field_.neg(p.y), p.z);
}

JacobianPoint256 EllipticCurve256::scalar_mul(const JacobianPoint256& p,
                                               const FieldElement256& k) const {
    if (p.is_infinity() || k.is_zero()) {
        return JacobianPoint256();
    }
    
    // Double-and-add algorithm (constant time would use Montgomery ladder)
    JacobianPoint256 result;
    JacobianPoint256 temp = p;
    
    for (int i = 0; i < 4; ++i) {
        uint64_t ki = k.limbs[i];
        for (int j = 0; j < 64; ++j) {
            if (ki & 1) {
                result = add(result, temp);
            }
            temp = double_point(temp);
            ki >>= 1;
        }
    }
    
    return result;
}

JacobianPoint256 EllipticCurve256::scalar_mul(const AffinePoint256& p,
                                               const FieldElement256& k) const {
    return scalar_mul(JacobianPoint256::from_affine(p, field_), k);
}

void EllipticCurve256::bucket_add(std::vector<JacobianPoint256>& buckets,
                                   const AffinePoint256& point, size_t bucket_idx) const {
    if (bucket_idx >= buckets.size()) return;
    buckets[bucket_idx] = add_mixed(buckets[bucket_idx], point);
}

JacobianPoint256 EllipticCurve256::msm(const AffinePoint256* points,
                                        const FieldElement256* scalars,
                                        size_t count) const {
    if (count == 0) return JacobianPoint256();
    if (count == 1) return scalar_mul(points[0], scalars[0]);
    
    // Pippenger's algorithm
    // Choose window size based on count
    size_t c = 1;
    while ((1ULL << (c + 1)) <= count) c++;
    c = std::min(c, size_t(16));  // Cap at 16 bits
    
    size_t num_buckets = (1ULL << c) - 1;
    size_t num_windows = (256 + c - 1) / c;
    
    JacobianPoint256 result;
    
    // Process each window
    for (size_t w = 0; w < num_windows; ++w) {
        // Initialize buckets
        std::vector<JacobianPoint256> buckets(num_buckets);
        
        // Accumulate points into buckets
        for (size_t i = 0; i < count; ++i) {
            // Extract c-bit window from scalar
            size_t bit_offset = w * c;
            size_t limb_idx = bit_offset / 64;
            size_t bit_idx = bit_offset % 64;
            
            uint64_t window_val = 0;
            if (limb_idx < 4) {
                window_val = scalars[i].limbs[limb_idx] >> bit_idx;
                if (bit_idx + c > 64 && limb_idx + 1 < 4) {
                    window_val |= scalars[i].limbs[limb_idx + 1] << (64 - bit_idx);
                }
            }
            window_val &= ((1ULL << c) - 1);
            
            if (window_val > 0) {
                bucket_add(buckets, points[i], window_val - 1);
            }
        }
        
        // Aggregate buckets: sum = sum(i * bucket[i])
        JacobianPoint256 window_sum;
        JacobianPoint256 running_sum;
        
        for (size_t i = num_buckets; i > 0; --i) {
            running_sum = add(running_sum, buckets[i - 1]);
            window_sum = add(window_sum, running_sum);
        }
        
        // Shift result by c bits and add window sum
        for (size_t i = 0; i < c; ++i) {
            result = double_point(result);
        }
        result = add(result, window_sum);
    }
    
    return result;
}

JacobianPoint256 EllipticCurve256::msm_gpu(const AffinePoint256* points,
                                            const FieldElement256* scalars,
                                            size_t count) const {
    // TODO: Implement Metal GPU acceleration
    // For now, fall back to CPU implementation
    return msm(points, scalars, count);
}


AffinePoint256 EllipticCurve256::to_affine(const JacobianPoint256& p) const {
    if (p.is_infinity()) {
        return AffinePoint256();
    }
    
    // (X/Z^2, Y/Z^3)
    FieldElement256 z_inv = field_.inv(p.z);
    FieldElement256 z_inv_sq = field_.square(z_inv);
    FieldElement256 z_inv_cu = field_.mul(z_inv_sq, z_inv);
    
    FieldElement256 x = field_.mul(p.x, z_inv_sq);
    FieldElement256 y = field_.mul(p.y, z_inv_cu);
    
    return AffinePoint256(x, y);
}

JacobianPoint256 EllipticCurve256::to_jacobian(const AffinePoint256& p) const {
    return JacobianPoint256::from_affine(p, field_);
}

bool EllipticCurve256::is_on_curve(const AffinePoint256& p) const {
    if (p.is_infinity) return true;
    
    // Check y^2 = x^3 + ax + b
    FieldElement256 y_sq = field_.square(p.y);
    FieldElement256 x_sq = field_.square(p.x);
    FieldElement256 x_cu = field_.mul(x_sq, p.x);
    
    FieldElement256 rhs = x_cu;
    if (!a_is_zero_) {
        FieldElement256 ax = field_.mul(a_, p.x);
        rhs = field_.add(rhs, ax);
    }
    rhs = field_.add(rhs, b_);
    
    return y_sq == rhs;
}

bool EllipticCurve256::is_on_curve(const JacobianPoint256& p) const {
    return is_on_curve(to_affine(p));
}

// ============================================================================
// EllipticCurve384 Implementation
// ============================================================================

EllipticCurve384::EllipticCurve384(const FieldElement384& a, const FieldElement384& b,
                                   const Field384& field)
    : a_(a), b_(b), field_(field) {
    a_is_zero_ = a_.is_zero();
}

JacobianPoint384 EllipticCurve384::add(const JacobianPoint384& p,
                                        const JacobianPoint384& q) const {
    if (p.is_infinity()) return q;
    if (q.is_infinity()) return p;
    
    FieldElement384 z1_sq = field_.square(p.z);
    FieldElement384 z2_sq = field_.square(q.z);
    
    FieldElement384 u1 = field_.mul(p.x, z2_sq);
    FieldElement384 u2 = field_.mul(q.x, z1_sq);
    
    FieldElement384 z1_cu = field_.mul(z1_sq, p.z);
    FieldElement384 z2_cu = field_.mul(z2_sq, q.z);
    
    FieldElement384 s1 = field_.mul(p.y, z2_cu);
    FieldElement384 s2 = field_.mul(q.y, z1_cu);
    
    if (u1 == u2) {
        if (s1 == s2) {
            return double_point(p);
        } else {
            return JacobianPoint384();
        }
    }
    
    FieldElement384 h = field_.sub(u2, u1);
    FieldElement384 r = field_.sub(s2, s1);
    
    FieldElement384 h_sq = field_.square(h);
    FieldElement384 h_cu = field_.mul(h_sq, h);
    FieldElement384 u1_h_sq = field_.mul(u1, h_sq);
    
    FieldElement384 r_sq = field_.square(r);
    FieldElement384 x3 = field_.sub(r_sq, h_cu);
    x3 = field_.sub(x3, u1_h_sq);
    x3 = field_.sub(x3, u1_h_sq);
    
    FieldElement384 y3 = field_.sub(u1_h_sq, x3);
    y3 = field_.mul(r, y3);
    FieldElement384 s1_h_cu = field_.mul(s1, h_cu);
    y3 = field_.sub(y3, s1_h_cu);
    
    FieldElement384 z3 = field_.mul(h, p.z);
    z3 = field_.mul(z3, q.z);
    
    return JacobianPoint384(x3, y3, z3);
}

JacobianPoint384 EllipticCurve384::add_mixed(const JacobianPoint384& p,
                                              const AffinePoint384& q) const {
    if (p.is_infinity()) {
        return JacobianPoint384::from_affine(q, field_);
    }
    if (q.is_infinity) {
        return p;
    }
    
    FieldElement384 z1_sq = field_.square(p.z);
    FieldElement384 u1 = p.x;
    FieldElement384 u2 = field_.mul(q.x, z1_sq);
    
    FieldElement384 z1_cu = field_.mul(z1_sq, p.z);
    FieldElement384 s1 = p.y;
    FieldElement384 s2 = field_.mul(q.y, z1_cu);
    
    if (u1 == u2) {
        if (s1 == s2) {
            return double_point(p);
        } else {
            return JacobianPoint384();
        }
    }
    
    FieldElement384 h = field_.sub(u2, u1);
    FieldElement384 r = field_.sub(s2, s1);
    
    FieldElement384 h_sq = field_.square(h);
    FieldElement384 h_cu = field_.mul(h_sq, h);
    FieldElement384 u1_h_sq = field_.mul(u1, h_sq);
    
    FieldElement384 r_sq = field_.square(r);
    FieldElement384 x3 = field_.sub(r_sq, h_cu);
    x3 = field_.sub(x3, u1_h_sq);
    x3 = field_.sub(x3, u1_h_sq);
    
    FieldElement384 y3 = field_.sub(u1_h_sq, x3);
    y3 = field_.mul(r, y3);
    FieldElement384 s1_h_cu = field_.mul(s1, h_cu);
    y3 = field_.sub(y3, s1_h_cu);
    
    FieldElement384 z3 = field_.mul(h, p.z);
    
    return JacobianPoint384(x3, y3, z3);
}


JacobianPoint384 EllipticCurve384::double_point(const JacobianPoint384& p) const {
    if (p.is_infinity()) return p;
    if (p.y.is_zero()) return JacobianPoint384();
    
    FieldElement384 xx = field_.square(p.x);
    FieldElement384 yy = field_.square(p.y);
    FieldElement384 yyyy = field_.square(yy);
    
    FieldElement384 x_plus_yy = field_.add(p.x, yy);
    FieldElement384 s = field_.square(x_plus_yy);
    s = field_.sub(s, xx);
    s = field_.sub(s, yyyy);
    s = field_.add(s, s);
    
    FieldElement384 m = field_.add(xx, xx);
    m = field_.add(m, xx);
    
    if (!a_is_zero_) {
        FieldElement384 z_sq = field_.square(p.z);
        FieldElement384 z_4 = field_.square(z_sq);
        FieldElement384 a_z4 = field_.mul(a_, z_4);
        m = field_.add(m, a_z4);
    }
    
    FieldElement384 x3 = field_.square(m);
    x3 = field_.sub(x3, s);
    x3 = field_.sub(x3, s);
    
    FieldElement384 y3 = field_.sub(s, x3);
    y3 = field_.mul(m, y3);
    FieldElement384 yyyy_8 = field_.add(yyyy, yyyy);
    yyyy_8 = field_.add(yyyy_8, yyyy_8);
    yyyy_8 = field_.add(yyyy_8, yyyy_8);
    y3 = field_.sub(y3, yyyy_8);
    
    FieldElement384 z3 = field_.mul(p.y, p.z);
    z3 = field_.add(z3, z3);
    
    return JacobianPoint384(x3, y3, z3);
}

JacobianPoint384 EllipticCurve384::negate(const JacobianPoint384& p) const {
    if (p.is_infinity()) return p;
    return JacobianPoint384(p.x, field_.neg(p.y), p.z);
}

JacobianPoint384 EllipticCurve384::scalar_mul(const JacobianPoint384& p,
                                               const FieldElement256& k) const {
    if (p.is_infinity() || k.is_zero()) {
        return JacobianPoint384();
    }
    
    JacobianPoint384 result;
    JacobianPoint384 temp = p;
    
    for (int i = 0; i < 4; ++i) {
        uint64_t ki = k.limbs[i];
        for (int j = 0; j < 64; ++j) {
            if (ki & 1) {
                result = add(result, temp);
            }
            temp = double_point(temp);
            ki >>= 1;
        }
    }
    
    return result;
}

JacobianPoint384 EllipticCurve384::scalar_mul(const AffinePoint384& p,
                                               const FieldElement256& k) const {
    return scalar_mul(JacobianPoint384::from_affine(p, field_), k);
}

JacobianPoint384 EllipticCurve384::msm(const AffinePoint384* points,
                                        const FieldElement256* scalars,
                                        size_t count) const {
    if (count == 0) return JacobianPoint384();
    if (count == 1) return scalar_mul(points[0], scalars[0]);
    
    // Pippenger's algorithm
    size_t c = 1;
    while ((1ULL << (c + 1)) <= count) c++;
    c = std::min(c, size_t(16));
    
    size_t num_buckets = (1ULL << c) - 1;
    size_t num_windows = (256 + c - 1) / c;
    
    JacobianPoint384 result;
    
    for (size_t w = 0; w < num_windows; ++w) {
        std::vector<JacobianPoint384> buckets(num_buckets);
        
        for (size_t i = 0; i < count; ++i) {
            size_t bit_offset = w * c;
            size_t limb_idx = bit_offset / 64;
            size_t bit_idx = bit_offset % 64;
            
            uint64_t window_val = 0;
            if (limb_idx < 4) {
                window_val = scalars[i].limbs[limb_idx] >> bit_idx;
                if (bit_idx + c > 64 && limb_idx + 1 < 4) {
                    window_val |= scalars[i].limbs[limb_idx + 1] << (64 - bit_idx);
                }
            }
            window_val &= ((1ULL << c) - 1);
            
            if (window_val > 0) {
                size_t bucket_idx = window_val - 1;
                if (bucket_idx < buckets.size()) {
                    buckets[bucket_idx] = add_mixed(buckets[bucket_idx], points[i]);
                }
            }
        }
        
        JacobianPoint384 window_sum;
        JacobianPoint384 running_sum;
        
        for (size_t i = num_buckets; i > 0; --i) {
            running_sum = add(running_sum, buckets[i - 1]);
            window_sum = add(window_sum, running_sum);
        }
        
        for (size_t i = 0; i < c; ++i) {
            result = double_point(result);
        }
        result = add(result, window_sum);
    }
    
    return result;
}

JacobianPoint384 EllipticCurve384::msm_gpu(const AffinePoint384* points,
                                            const FieldElement256* scalars,
                                            size_t count) const {
    // TODO: Implement Metal GPU acceleration
    return msm(points, scalars, count);
}

AffinePoint384 EllipticCurve384::to_affine(const JacobianPoint384& p) const {
    if (p.is_infinity()) {
        return AffinePoint384();
    }
    
    FieldElement384 z_inv = field_.inv(p.z);
    FieldElement384 z_inv_sq = field_.square(z_inv);
    FieldElement384 z_inv_cu = field_.mul(z_inv_sq, z_inv);
    
    FieldElement384 x = field_.mul(p.x, z_inv_sq);
    FieldElement384 y = field_.mul(p.y, z_inv_cu);
    
    return AffinePoint384(x, y);
}

JacobianPoint384 EllipticCurve384::to_jacobian(const AffinePoint384& p) const {
    return JacobianPoint384::from_affine(p, field_);
}

bool EllipticCurve384::is_on_curve(const AffinePoint384& p) const {
    if (p.is_infinity) return true;
    
    FieldElement384 y_sq = field_.square(p.y);
    FieldElement384 x_sq = field_.square(p.x);
    FieldElement384 x_cu = field_.mul(x_sq, p.x);
    
    FieldElement384 rhs = x_cu;
    if (!a_is_zero_) {
        FieldElement384 ax = field_.mul(a_, p.x);
        rhs = field_.add(rhs, ax);
    }
    rhs = field_.add(rhs, b_);
    
    return y_sq == rhs;
}

bool EllipticCurve384::is_on_curve(const JacobianPoint384& p) const {
    return is_on_curve(to_affine(p));
}


// ============================================================================
// Pre-configured Curve Instances
// ============================================================================

const EllipticCurve256& bn254_g1() {
    static EllipticCurve256 curve(
        FieldElement256(0),  // a = 0
        bn254_fq().to_montgomery(FieldElement256(3)),  // b = 3
        bn254_fq()
    );
    return curve;
}

const EllipticCurve384& bls12_381_g1() {
    static EllipticCurve384 curve(
        FieldElement384(),  // a = 0
        bls12_381_fq().to_montgomery(FieldElement384(4)),  // b = 4
        bls12_381_fq()
    );
    return curve;
}

AffinePoint256 bn254_g1_generator() {
    // BN254 G1 generator: (1, 2)
    const Field256& fq = bn254_fq();
    return AffinePoint256(
        fq.to_montgomery(FieldElement256(1)),
        fq.to_montgomery(FieldElement256(2))
    );
}

AffinePoint384 bls12_381_g1_generator() {
    // BLS12-381 G1 generator
    // x = 0x17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb
    // y = 0x08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1
    
    const Field384& fq = bls12_381_fq();
    
    FieldElement384 gx = FieldElement384::from_hex(
        "17f1d3a73197d7942695638c4fa9ac0fc3688c4f9774b905a14e3a3f171bac586c55e83ff97a1aeffb3af00adb22c6bb"
    );
    FieldElement384 gy = FieldElement384::from_hex(
        "08b3f481e3aaa0f1a09e30ed741d8ae4fcf5e095d5d00af600db18cb2c04b3edd03cc744a2888ae40caa232946c5e7e1"
    );
    
    return AffinePoint384(
        fq.to_montgomery(gx),
        fq.to_montgomery(gy)
    );
}

// ============================================================================
// Utility Functions
// ============================================================================

AffinePoint256 random_point_256(const EllipticCurve256& curve) {
    // Generate random scalar and multiply generator
    FieldElement256 k = random_field_element_256(bn254_fr());
    JacobianPoint256 p = curve.scalar_mul(bn254_g1_generator(), k);
    return curve.to_affine(p);
}

AffinePoint384 random_point_384(const EllipticCurve384& curve) {
    FieldElement256 k = random_field_element_256(bls12_381_fr());
    JacobianPoint384 p = curve.scalar_mul(bls12_381_g1_generator(), k);
    return curve.to_affine(p);
}

AffinePoint256 hash_to_curve_256(const uint8_t* data, size_t len, 
                                  const EllipticCurve256& curve) {
    // Simplified hash-to-curve (not constant-time, for testing only)
    // Real implementation should use RFC 9380
    
    std::array<uint8_t, 32> hash;
    // Simple hash: XOR with position
    for (size_t i = 0; i < 32; ++i) {
        hash[i] = 0;
        for (size_t j = 0; j < len; ++j) {
            hash[i] ^= data[j] ^ static_cast<uint8_t>((i + j) & 0xFF);
        }
    }
    
    FieldElement256 x = FieldElement256::from_bytes(hash);
    const Field256& fq = curve.field();
    
    // Try to find y such that y^2 = x^3 + ax + b
    for (int attempt = 0; attempt < 256; ++attempt) {
        FieldElement256 x_mont = fq.to_montgomery(x);
        FieldElement256 x_sq = fq.square(x_mont);
        FieldElement256 x_cu = fq.mul(x_sq, x_mont);
        
        FieldElement256 rhs = x_cu;
        if (!curve.a().is_zero()) {
            FieldElement256 ax = fq.mul(curve.a(), x_mont);
            rhs = fq.add(rhs, ax);
        }
        rhs = fq.add(rhs, curve.b());
        
        // Try to compute square root (simplified)
        // For p ≡ 3 (mod 4): sqrt(a) = a^((p+1)/4)
        FieldElement256 exp = fq.modulus();
        exp.limbs[0] += 1;  // p + 1
        // Divide by 4 (shift right by 2)
        uint64_t carry = 0;
        for (int i = 3; i >= 0; --i) {
            uint64_t new_carry = exp.limbs[i] & 3;
            exp.limbs[i] = (exp.limbs[i] >> 2) | (carry << 62);
            carry = new_carry;
        }
        
        FieldElement256 y = fq.pow(rhs, exp);
        FieldElement256 y_sq = fq.square(y);
        
        if (y_sq == rhs) {
            return AffinePoint256(x_mont, y);
        }
        
        // Increment x and try again
        x.limbs[0]++;
        if (x >= fq.modulus()) {
            x.limbs[0] = 0;
        }
    }
    
    // Fallback to generator
    return bn254_g1_generator();
}

AffinePoint384 hash_to_curve_384(const uint8_t* data, size_t len,
                                  const EllipticCurve384& curve) {
    // Simplified hash-to-curve for testing
    std::array<uint8_t, 48> hash;
    for (size_t i = 0; i < 48; ++i) {
        hash[i] = 0;
        for (size_t j = 0; j < len; ++j) {
            hash[i] ^= data[j] ^ static_cast<uint8_t>((i + j) & 0xFF);
        }
    }
    
    FieldElement384 x = FieldElement384::from_bytes(hash);
    const Field384& fq = curve.field();
    
    for (int attempt = 0; attempt < 256; ++attempt) {
        FieldElement384 x_mont = fq.to_montgomery(x);
        FieldElement384 x_sq = fq.square(x_mont);
        FieldElement384 x_cu = fq.mul(x_sq, x_mont);
        
        FieldElement384 rhs = x_cu;
        rhs = fq.add(rhs, curve.b());
        
        // Simplified sqrt attempt
        FieldElement384 exp = fq.modulus();
        exp.limbs[0] += 1;
        uint64_t carry = 0;
        for (int i = 5; i >= 0; --i) {
            uint64_t new_carry = exp.limbs[i] & 3;
            exp.limbs[i] = (exp.limbs[i] >> 2) | (carry << 62);
            carry = new_carry;
        }
        
        FieldElement384 y = fq.pow(rhs, exp);
        FieldElement384 y_sq = fq.square(y);
        
        if (y_sq == rhs) {
            return AffinePoint384(x_mont, y);
        }
        
        x.limbs[0]++;
    }
    
    return bls12_381_g1_generator();
}

} // namespace zk
} // namespace fhe_accelerate

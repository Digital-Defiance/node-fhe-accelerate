/**
 * Zero-Knowledge Proof Elliptic Curve Operations
 * 
 * Implements elliptic curve operations for ZK proof systems:
 * - Point addition and doubling
 * - Scalar multiplication
 * - Multi-scalar multiplication (MSM)
 * - Metal GPU acceleration for parallel MSM
 * 
 * Supports BLS12-381 and BN254 curves.
 * 
 * Requirements: 19, 20.2
 */

#pragma once

#include "zk_field_arithmetic.h"
#include <vector>
#include <memory>

namespace fhe_accelerate {
namespace zk {

// ============================================================================
// Point Representations
// ============================================================================

/**
 * Affine point on curve over 256-bit field
 * Used for BN254 G1 and input/output
 */
struct AffinePoint256 {
    FieldElement256 x;
    FieldElement256 y;
    bool is_infinity;
    
    AffinePoint256() : is_infinity(true) {}
    AffinePoint256(const FieldElement256& x_, const FieldElement256& y_)
        : x(x_), y(y_), is_infinity(false) {}
    
    bool operator==(const AffinePoint256& other) const;
    bool operator!=(const AffinePoint256& other) const { return !(*this == other); }
};

/**
 * Projective point on curve over 256-bit field (X, Y, Z)
 * Represents affine point (X/Z, Y/Z)
 * Used for efficient point operations
 */
struct ProjectivePoint256 {
    FieldElement256 x;
    FieldElement256 y;
    FieldElement256 z;
    
    ProjectivePoint256() : z(0) {}  // Point at infinity
    ProjectivePoint256(const FieldElement256& x_, const FieldElement256& y_, 
                       const FieldElement256& z_)
        : x(x_), y(y_), z(z_) {}
    
    bool is_infinity() const { return z.is_zero(); }
    
    // Convert from affine
    static ProjectivePoint256 from_affine(const AffinePoint256& p, const Field256& field);
};

/**
 * Jacobian point on curve over 256-bit field (X, Y, Z)
 * Represents affine point (X/Z^2, Y/Z^3)
 * Most efficient for repeated additions
 */
struct JacobianPoint256 {
    FieldElement256 x;
    FieldElement256 y;
    FieldElement256 z;
    
    JacobianPoint256() : z(0) {}
    JacobianPoint256(const FieldElement256& x_, const FieldElement256& y_,
                     const FieldElement256& z_)
        : x(x_), y(y_), z(z_) {}
    
    bool is_infinity() const { return z.is_zero(); }
    
    static JacobianPoint256 from_affine(const AffinePoint256& p, const Field256& field);
};

/**
 * Affine point on curve over 384-bit field
 * Used for BLS12-381 G1
 */
struct AffinePoint384 {
    FieldElement384 x;
    FieldElement384 y;
    bool is_infinity;
    
    AffinePoint384() : is_infinity(true) {}
    AffinePoint384(const FieldElement384& x_, const FieldElement384& y_)
        : x(x_), y(y_), is_infinity(false) {}
    
    bool operator==(const AffinePoint384& other) const;
};

/**
 * Jacobian point on curve over 384-bit field
 */
struct JacobianPoint384 {
    FieldElement384 x;
    FieldElement384 y;
    FieldElement384 z;
    
    JacobianPoint384() : z(0) {}
    JacobianPoint384(const FieldElement384& x_, const FieldElement384& y_,
                     const FieldElement384& z_)
        : x(x_), y(y_), z(z_) {}
    
    bool is_infinity() const { return z.is_zero(); }
    
    static JacobianPoint384 from_affine(const AffinePoint384& p, const Field384& field);
};

// ============================================================================
// Curve Parameters
// ============================================================================

/**
 * Elliptic curve parameters for short Weierstrass form: y^2 = x^3 + ax + b
 */
template<typename FieldElement, typename Field>
struct CurveParams {
    FieldElement a;           // Curve coefficient a
    FieldElement b;           // Curve coefficient b
    FieldElement generator_x; // Generator point x coordinate
    FieldElement generator_y; // Generator point y coordinate
    const Field* field;       // Base field
    
    // Cofactor and order info
    size_t cofactor;
};

// ============================================================================
// Elliptic Curve Operations (256-bit field)
// ============================================================================

/**
 * Elliptic curve arithmetic over 256-bit field
 * 
 * Implements point operations for BN254 G1 curve.
 * Uses Jacobian coordinates for efficient computation.
 */
class EllipticCurve256 {
public:
    /**
     * Construct curve with given parameters
     */
    EllipticCurve256(const FieldElement256& a, const FieldElement256& b,
                     const Field256& field);
    
    // ========================================================================
    // Point Operations
    // ========================================================================
    
    /**
     * Point addition: P + Q
     */
    JacobianPoint256 add(const JacobianPoint256& p, const JacobianPoint256& q) const;
    
    /**
     * Mixed addition: Jacobian + Affine (more efficient)
     */
    JacobianPoint256 add_mixed(const JacobianPoint256& p, const AffinePoint256& q) const;
    
    /**
     * Point doubling: 2P
     */
    JacobianPoint256 double_point(const JacobianPoint256& p) const;
    
    /**
     * Point negation: -P
     */
    JacobianPoint256 negate(const JacobianPoint256& p) const;
    
    /**
     * Scalar multiplication: k * P
     * Uses double-and-add algorithm
     */
    JacobianPoint256 scalar_mul(const JacobianPoint256& p, const FieldElement256& k) const;
    
    /**
     * Scalar multiplication with affine input
     */
    JacobianPoint256 scalar_mul(const AffinePoint256& p, const FieldElement256& k) const;
    
    /**
     * Multi-scalar multiplication (MSM): sum(k_i * P_i)
     * Uses Pippenger's algorithm for efficiency
     * 
     * @param points Array of points
     * @param scalars Array of scalars
     * @param count Number of point-scalar pairs
     * @return Sum of scalar multiplications
     */
    JacobianPoint256 msm(const AffinePoint256* points, const FieldElement256* scalars,
                         size_t count) const;
    
    /**
     * GPU-accelerated MSM using Metal
     * Falls back to CPU if Metal unavailable
     */
    JacobianPoint256 msm_gpu(const AffinePoint256* points, const FieldElement256* scalars,
                             size_t count) const;
    
    // ========================================================================
    // Coordinate Conversion
    // ========================================================================
    
    /**
     * Convert Jacobian to affine coordinates
     */
    AffinePoint256 to_affine(const JacobianPoint256& p) const;
    
    /**
     * Convert affine to Jacobian coordinates
     */
    JacobianPoint256 to_jacobian(const AffinePoint256& p) const;
    
    // ========================================================================
    // Validation
    // ========================================================================
    
    /**
     * Check if point is on the curve
     */
    bool is_on_curve(const AffinePoint256& p) const;
    bool is_on_curve(const JacobianPoint256& p) const;
    
    // ========================================================================
    // Getters
    // ========================================================================
    
    const Field256& field() const { return field_; }
    const FieldElement256& a() const { return a_; }
    const FieldElement256& b() const { return b_; }
    
private:
    FieldElement256 a_;
    FieldElement256 b_;
    const Field256& field_;
    
    // Precomputed constants for a=0 optimization
    bool a_is_zero_;
    
    // MSM bucket accumulation
    void bucket_add(std::vector<JacobianPoint256>& buckets,
                    const AffinePoint256& point, size_t bucket_idx) const;
};

// ============================================================================
// Elliptic Curve Operations (384-bit field)
// ============================================================================

/**
 * Elliptic curve arithmetic over 384-bit field
 * 
 * Implements point operations for BLS12-381 G1 curve.
 */
class EllipticCurve384 {
public:
    EllipticCurve384(const FieldElement384& a, const FieldElement384& b,
                     const Field384& field);
    
    JacobianPoint384 add(const JacobianPoint384& p, const JacobianPoint384& q) const;
    JacobianPoint384 add_mixed(const JacobianPoint384& p, const AffinePoint384& q) const;
    JacobianPoint384 double_point(const JacobianPoint384& p) const;
    JacobianPoint384 negate(const JacobianPoint384& p) const;
    
    JacobianPoint384 scalar_mul(const JacobianPoint384& p, const FieldElement256& k) const;
    JacobianPoint384 scalar_mul(const AffinePoint384& p, const FieldElement256& k) const;
    
    JacobianPoint384 msm(const AffinePoint384* points, const FieldElement256* scalars,
                         size_t count) const;
    JacobianPoint384 msm_gpu(const AffinePoint384* points, const FieldElement256* scalars,
                             size_t count) const;
    
    AffinePoint384 to_affine(const JacobianPoint384& p) const;
    JacobianPoint384 to_jacobian(const AffinePoint384& p) const;
    
    bool is_on_curve(const AffinePoint384& p) const;
    bool is_on_curve(const JacobianPoint384& p) const;
    
    const Field384& field() const { return field_; }
    
private:
    FieldElement384 a_;
    FieldElement384 b_;
    const Field384& field_;
    bool a_is_zero_;
};

// ============================================================================
// Pre-configured Curve Instances
// ============================================================================

/**
 * Get BN254 G1 curve
 * y^2 = x^3 + 3
 */
const EllipticCurve256& bn254_g1();

/**
 * Get BLS12-381 G1 curve
 * y^2 = x^3 + 4
 */
const EllipticCurve384& bls12_381_g1();

/**
 * Get BN254 G1 generator point
 */
AffinePoint256 bn254_g1_generator();

/**
 * Get BLS12-381 G1 generator point
 */
AffinePoint384 bls12_381_g1_generator();

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Generate random point on curve
 */
AffinePoint256 random_point_256(const EllipticCurve256& curve);
AffinePoint384 random_point_384(const EllipticCurve384& curve);

/**
 * Hash to curve (simplified - for testing)
 */
AffinePoint256 hash_to_curve_256(const uint8_t* data, size_t len, const EllipticCurve256& curve);
AffinePoint384 hash_to_curve_384(const uint8_t* data, size_t len, const EllipticCurve384& curve);

} // namespace zk
} // namespace fhe_accelerate

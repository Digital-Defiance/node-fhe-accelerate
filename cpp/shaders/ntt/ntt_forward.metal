//
// ntt_forward.metal
// Forward Number Theoretic Transform (NTT) compute shader
//
// Implements the Cooley-Tukey NTT algorithm using GPU parallelism.
// Each thread processes butterfly operations for a specific stage.
//
// Design Reference: Section 4 - NTT Processor Implementation
// Requirements: 1.1, 1.4, 14.5
//

#include "../common/fhe_common.metal"

// ============================================================================
// Kernel: Forward NTT Butterfly Stage
// ============================================================================

/// Perform one stage of the forward NTT butterfly operations
///
/// The NTT is computed in log2(N) stages, where each stage performs N/2 butterfly
/// operations. This kernel processes one stage for a batch of polynomials.
///
/// Thread organization:
/// - Each thread processes one butterfly operation
/// - Threadgroups process chunks of the polynomial
/// - Multiple polynomials can be processed in parallel (batch dimension)
///
/// @param coeffs Input/output coefficient buffer [batch_size][degree]
/// @param twiddles Precomputed twiddle factors [degree]
/// @param params NTT parameters (degree, modulus, etc.)
/// @param stage Current NTT stage (0 to log2(degree)-1)
/// @param batch_size Number of polynomials to process
kernel void ntt_forward_stage(
    device coeff_t* coeffs [[buffer(0)]],
    device const coeff_t* twiddles [[buffer(1)]],
    constant NTTParams& params [[buffer(2)]],
    constant uint32_t& stage [[buffer(3)]],
    constant uint32_t& batch_size [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 grid_size [[threads_per_grid]]
) {
    // Extract thread indices
    uint32_t batch_idx = gid.z;  // Which polynomial in the batch
    uint32_t butterfly_idx = gid.x;  // Which butterfly in this stage
    
    // Bounds check
    if (batch_idx >= batch_size) return;
    
    // Compute butterfly parameters for this stage
    uint32_t m = 1 << stage;  // Distance between butterfly pairs
    uint32_t k = params.degree >> (stage + 1);  // Number of butterfly groups
    
    if (butterfly_idx >= params.degree / 2) return;
    
    // Compute indices for this butterfly
    uint32_t group = butterfly_idx / m;
    uint32_t offset = butterfly_idx % m;
    uint32_t j = 2 * m * group + offset;
    uint32_t k_idx = j + m;
    
    // Compute twiddle factor index
    uint32_t twiddle_idx = (butterfly_idx / m) * (params.degree / (2 * m));
    
    // Load twiddle factor
    coeff_t omega = twiddles[twiddle_idx];
    
    // Compute base offset for this polynomial in the batch
    uint32_t poly_offset = batch_idx * params.degree;
    
    // Load butterfly inputs
    coeff_t a = coeffs[poly_offset + j];
    coeff_t b = coeffs[poly_offset + k_idx];
    
    // Compute butterfly outputs
    // Standard Cooley-Tukey butterfly:
    // out[j] = a + omega * b
    // out[k] = a - omega * b
    
    coeff_t omega_b = montgomery_mul(omega, b, params.modulus, params.inv_n);
    coeff_t out_j = mod_add(a, omega_b, params.modulus);
    coeff_t out_k = mod_sub(a, omega_b, params.modulus);
    
    // Store butterfly outputs
    coeffs[poly_offset + j] = out_j;
    coeffs[poly_offset + k_idx] = out_k;
}

// ============================================================================
// Kernel: Bit-Reversal Permutation
// ============================================================================

/// Perform bit-reversal permutation on polynomial coefficients
///
/// The NTT algorithm requires input coefficients to be in bit-reversed order.
/// This kernel performs the permutation in parallel.
///
/// @param coeffs Input/output coefficient buffer [batch_size][degree]
/// @param params NTT parameters
/// @param batch_size Number of polynomials to process
kernel void ntt_bit_reverse(
    device coeff_t* coeffs [[buffer(0)]],
    constant NTTParams& params [[buffer(1)]],
    constant uint32_t& batch_size [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint32_t batch_idx = gid.z;
    uint32_t i = gid.x;
    
    if (batch_idx >= batch_size || i >= params.degree) return;
    
    // Compute bit-reversed index
    uint32_t j = bit_reverse(i, params.log_degree);
    
    // Only swap if i < j to avoid double-swapping
    if (i < j) {
        uint32_t poly_offset = batch_idx * params.degree;
        
        coeff_t temp = coeffs[poly_offset + i];
        coeffs[poly_offset + i] = coeffs[poly_offset + j];
        coeffs[poly_offset + j] = temp;
    }
}

// ============================================================================
// Kernel: Batch Forward NTT (Complete)
// ============================================================================

/// Perform complete forward NTT on a batch of polynomials
///
/// This is an optimized kernel that performs all NTT stages in a single dispatch
/// using threadgroup memory for intermediate results. Best for smaller polynomial
/// degrees (up to 4096) that fit in threadgroup memory.
///
/// @param input Input coefficient buffer [batch_size][degree]
/// @param output Output NTT coefficient buffer [batch_size][degree]
/// @param twiddles Precomputed twiddle factors [degree]
/// @param params NTT parameters
/// @param batch_size Number of polynomials to process
kernel void ntt_forward_batch(
    device const coeff_t* input [[buffer(0)]],
    device coeff_t* output [[buffer(1)]],
    device const coeff_t* twiddles [[buffer(2)]],
    constant NTTParams& params [[buffer(3)]],
    constant uint32_t& batch_size [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tpg [[threads_per_threadgroup]]
) {
    uint32_t batch_idx = gid.z;
    
    if (batch_idx >= batch_size) return;
    
    // Allocate threadgroup memory for intermediate results
    // Note: Size must be known at compile time or passed as template parameter
    threadgroup coeff_t shared_coeffs[1024];  // Adjust size as needed
    
    uint32_t poly_offset = batch_idx * params.degree;
    uint32_t local_idx = tid.x;
    
    // Load coefficients into threadgroup memory with bit-reversal
    if (local_idx < params.degree) {
        uint32_t reversed_idx = bit_reverse(local_idx, params.log_degree);
        shared_coeffs[local_idx] = input[poly_offset + reversed_idx];
    }
    
    threadgroup_barrier();
    
    // Perform NTT stages
    for (uint32_t stage = 0; stage < params.log_degree; stage++) {
        uint32_t m = 1 << stage;
        uint32_t butterfly_idx = local_idx;
        
        if (butterfly_idx < params.degree / 2) {
            // Compute butterfly indices
            uint32_t group = butterfly_idx / m;
            uint32_t offset = butterfly_idx % m;
            uint32_t j = 2 * m * group + offset;
            uint32_t k = j + m;
            
            // Compute twiddle factor index
            uint32_t twiddle_idx = (butterfly_idx / m) * (params.degree / (2 * m));
            coeff_t omega = twiddles[twiddle_idx];
            
            // Load butterfly inputs
            coeff_t a = shared_coeffs[j];
            coeff_t b = shared_coeffs[k];
            
            // Compute butterfly outputs
            coeff_t omega_b = montgomery_mul(omega, b, params.modulus, params.inv_n);
            coeff_t out_j = mod_add(a, omega_b, params.modulus);
            coeff_t out_k = mod_sub(a, omega_b, params.modulus);
            
            // Store results back to shared memory
            shared_coeffs[j] = out_j;
            shared_coeffs[k] = out_k;
        }
        
        threadgroup_barrier();
    }
    
    // Write results to output
    if (local_idx < params.degree) {
        output[poly_offset + local_idx] = shared_coeffs[local_idx];
    }
}

# Implementation Plan: @digitaldefiance/node-fhe-accelerate

## Overview

This implementation plan breaks down the FHE acceleration library into incremental coding tasks. Each task builds on previous work, with property tests placed close to implementations to catch errors early. The plan prioritizes core arithmetic primitives first, then builds up to full FHE operations.

## Tasks

- [ ] 1. Project scaffolding and build infrastructure
  - [x] 1.1 Initialize npm package with TypeScript configuration (yarn)
    - Create package.json with name `@digitaldefiance/node-fhe-accelerate`
    - Configure TypeScript with strict mode and ES2022 target
    - Set up dual CJS/ESM exports
    - _Requirements: 12.5_
  
  - [x] 1.2 Set up napi-rs native addon structure
    - Initialize napi-rs with C++ support
    - Configure Cargo.toml for native dependencies
    - Set up build scripts for arm64 macOS
    - _Requirements: 13.1, 13.4_
  
  - [x] 1.3 Configure Metal shader compilation pipeline
    - Set up .metal shader directory structure
    - Create build script to compile Metal shaders to metallib
    - Configure shader hot-reloading for development
    - _Requirements: 13.2_
  
  - [x] 1.4 Set up testing infrastructure
    - Configure Vitest for TypeScript tests
    - Set up fast-check for property-based testing
    - Configure C++ test harness with random generators
    - Create test utilities for ciphertext comparison
    - Add voting simulation test harness
    - Create ballot generation utilities
    - Implement fraud scenario generators
    - **Set up 7-layer testing infrastructure (unit, property, integration, adversarial, formal, cross-validation, stress)**
    - **Configure 1000+ iteration property tests**
    - **Set up cross-validation with SEAL, HElib, libsnark**
    - **Configure continuous testing pipeline**
    - **Set up formal verification tools (Coq/Cryptol)**
    - **Implement test coverage reporting (100% required)**
    - **Set up security testing (timing attacks, side-channels)**
    - _Requirements: Testing Strategy, 15, 17, 19, 20_

- [x] 2. Core modular arithmetic implementation
  - [x] 2.1 Implement Montgomery multiplication for 64-bit coefficients
    - Write Montgomery reduction in C++ with precomputed constants
    - Implement modular addition and subtraction
    - Create NEON SIMD vectorized variants
    - **OPTIMIZED**: Barrett Unrolled (4x) is 2x faster than Montgomery for batch ops
    - _Requirements: 2.1, 2.4_
  
  - [x] 2.2 Implement multi-limb arithmetic for coefficients > 64 bits
    - Create multi-limb integer representation
    - Implement multi-limb Montgomery multiplication
    - Add NEON vectorization for limb operations
    - _Requirements: 2.2_
  
  - [x] 2.3 Write property test for modular multiplication correctness
    - **Property 2: Modular Multiplication Correctness**
    - Generate random coefficient pairs and moduli
    - Verify against reference big-integer implementation
    - **Validates: Requirements 2.5**

- [x] 3. Checkpoint - Core arithmetic validation
  - Ensure all modular arithmetic tests pass
  - Verify NEON optimizations produce correct results
  - Ask the user if questions arise

- [x] 4. NTT processor implementation
  - [x] 4.1 Implement twiddle factor precomputation
    - Compute primitive roots for NTT-friendly primes
    - Generate forward and inverse twiddle factor tables
    - Store in cache-aligned memory layout
    - _Requirements: 1.1_
  
  - [x] 4.2 Implement Cooley-Tukey NTT algorithm
    - Write iterative radix-2 NTT in C++
    - Implement bit-reversal permutation
    - Add in-place computation to minimize memory
    - _Requirements: 1.1, 1.2_
  
  - [x] 4.3 Implement inverse NTT
    - Write inverse NTT with inverse twiddle factors
    - Apply scaling by N^(-1) mod q
    - Verify round-trip correctness
    - _Requirements: 1.2_
  
  - [x] 4.4 Write property test for NTT round-trip
    - **Property 1: NTT Round-Trip Consistency**
    - Generate random polynomials of various degrees
    - Apply forward then inverse NTT
    - Verify coefficient-wise equality
    - **Validates: Requirements 1.6, 1.2**
  
  - [x] 4.5 Implement NEON-optimized NTT butterflies
    - Vectorize butterfly operations using NEON intrinsics
    - Process 2 butterflies per SIMD operation
    - Optimize memory access patterns for cache
    - _Requirements: 1.1, 9.3_
  
  - [x] 4.6 Implement SME-accelerated NTT
    - Use SME matrix registers for butterfly stages
    - Implement streaming mode for large polynomials
    - Add fallback to NEON when SME unavailable
    - _Requirements: 1.3, 9.1, 14.4_

- [x] 5. Checkpoint - NTT validation
  - Ensure NTT round-trip property passes for all degrees
  - Benchmark NEON vs SME performance
  - **COMPLETED**: All NTT tests pass, Montgomery NTT provides 2x speedup
  - Ask the user if questions arise

- [x] 6. Polynomial ring operations
  - [x] 6.1 Implement polynomial data structure
    - Create Polynomial class with coefficient storage
    - Support NTT and coefficient representations
    - Implement cache-aligned memory layout
    - _Requirements: 3.4, 14.2_
  
  - [x] 6.2 Implement polynomial addition and subtraction
    - Write coefficient-wise modular addition
    - Implement NEON vectorized variant
    - Handle different moduli in RNS representation
    - _Requirements: 3.2_
  
  - [x] 6.3 Implement NTT-based polynomial multiplication
    - Convert to NTT domain, pointwise multiply, convert back
    - Handle reduction modulo X^N + 1 via negacyclic NTT
    - Optimize for in-place computation
    - _Requirements: 3.1, 3.3_
  
  - [x] 6.4 Write property tests for polynomial operations
    - **Property 3: Polynomial Multiplication Commutativity**
    - **Property 4: Polynomial Multiplicative Identity**
    - Generate random polynomial pairs
    - Verify commutativity and identity properties
    - **Validates: Requirements 3.5, 3.6**

- [x] 7. Metal GPU backend implementation
  - [x] 7.1 Implement Metal compute pipeline setup
    - Create MTLDevice and command queue management
    - Implement buffer allocation with unified memory
    - Set up compute pipeline state compilation
    - **COMPLETED**: `cpp/include/metal_compute.h`, `cpp/src/metal_compute.mm`
    - _Requirements: 13.2, 14.6_
  
  - [x] 7.2 Implement batch NTT Metal shader
    - Write NTT butterfly kernel in Metal Shading Language
    - Support processing multiple polynomials in parallel
    - Optimize threadgroup size for M4 Max GPU
    - **COMPLETED**: `cpp/shaders/ntt/ntt_forward.metal`
    - _Requirements: 1.4, 14.5_
  
  - [x] 7.3 Implement batch modular multiplication shader
    - Write parallel modular multiplication kernel
    - Support Montgomery and Barrett reduction
    - Optimize for GPU memory bandwidth
    - **COMPLETED**: `cpp/shaders/modular/modmul_direct.metal` (Barrett), `cpp/shaders/modular/modmul_batch.metal` (Montgomery)
    - **GPU provides 1.55x speedup for batches >262K elements**
    - _Requirements: 2.6_

- [x] 8. Hardware dispatcher implementation
  - [x] 8.1 Implement hardware capability detection
    - Detect SME availability via CPU feature flags
    - Query Metal device capabilities
    - Check for Neural Engine availability
    - **COMPLETED**: `cpp/include/hardware_benchmark.h`, `cpp/src/hardware_benchmark.cpp`
    - **Detects: SME, SME2, AMX, Metal (40 cores), Neural Engine (38 TOPS)**
    - _Requirements: 9.1, 9.2, 9.3, 9.4_
  
  - [x] 8.2 Implement backend selection logic
    - Create heuristics for operation routing
    - Support manual override for benchmarking
    - Implement graceful fallback chain
    - **COMPLETED**: `cpp/include/adaptive_dispatcher.h`, `cpp/src/adaptive_dispatcher.cpp`
    - **Winners: Barrett Unrolled (4x) for modmul, Montgomery NTT for transforms**
    - _Requirements: 9.5, 9.6_

- [x] 9. Checkpoint - Hardware acceleration validation
  - Verify all backends produce identical results
  - Benchmark each backend for various operation sizes
  - **COMPLETED**: Full benchmark suite in `cpp/tests/run_hardware_benchmark.cpp`
  - **Results documented in `cpp/HARDWARE_ACCELERATION_REPORT.md`**
  - Ask the user if questions arise

- [x] 10. Parameter sets and security validation
  - [x] 10.1 Implement parameter set data structures
    - Create ParameterSet struct with all FHE parameters
    - Implement preset configurations (TFHE-128-fast, etc.)
    - Calculate derived parameters (noise budget, depth)
    - _Requirements: 10.1, 10.2, 10.3_
  
  - [x] 10.2 Implement security constraint validation
    - Implement lattice security estimator checks
    - Validate polynomial degree vs modulus relationships
    - Return detailed violation messages
    - _Requirements: 10.4, 10.5_
  
  - [x] 10.3 Write property test for parameter validation
    - **Property 11: Parameter Validation Rejects Insecure Configurations**
    - Generate parameter sets violating security bounds
    - Verify rejection with appropriate errors
    - **Validates: Requirements 10.4**

- [x] 11. Key generation and management
  - [x] 11.1 Implement secret key generation
    - Sample coefficients from ternary/Gaussian distribution
    - Use secure random number generator
    - Store in protected memory region
    - _Requirements: 4.1_
  
  - [x] 11.2 Implement public key generation
    - Compute RLWE encryption of zero
    - Parallelize across available cores
    - Target < 100ms for 128-bit security
    - _Requirements: 4.2_
  
  - [x] 11.3 Implement evaluation key generation
    - Generate key switching keys with configurable base
    - Support multiple decomposition levels
    - Parallelize key generation
    - _Requirements: 4.3_
  
  - [x] 11.4 Implement key serialization
    - Create binary serialization format
    - Add integrity checksums
    - Support streaming for large keys
    - Implement compressed ballot format (<10KB per vote)
    - Add versioning for protocol upgrades
    - Support partial deserialization for verification
    - _Requirements: 4.5, 4.6, 16_
  
  - [x] 11.5 Write property test for key serialization round-trip
    - **Property 5: Key Serialization Round-Trip**
    - Generate keys, serialize, deserialize
    - Verify functional equivalence via encrypt/decrypt
    - **Validates: Requirements 4.5**
  
  - [x] 11.6 Implement threshold decryption
    - Support N-of-M threshold key sharing
    - Implement distributed key generation
    - Support partial decryption and combination
    - Verify threshold security properties
    - _Requirements: 15.4_
  
  - [x] 11.7 Implement verifiable encryption
    - Generate zero-knowledge proofs of correct encryption
    - Support public verifiability of ballots
    - Implement receipt generation for voters
    - Verify proof correctness
    - _Requirements: 15.5, 17_

- [x] 12. Encryption and decryption
  - [x] 12.1 Implement RLWE encryption
    - Sample random polynomial and error
    - Compute ciphertext components
    - Support SIMD packing for multiple values
    - _Requirements: 5.1, 5.3_
  
  - [x] 12.2 Implement decryption
    - Compute inner product with secret key
    - Round to nearest plaintext value
    - Check and report noise budget
    - _Requirements: 5.2, 5.4_
  
  - [x] 12.3 Implement batch encryption with Metal GPU
    - Parallelize encryption across GPU threads
    - Use unified memory for zero-copy
    - Target: 10,000+ ballots/second encryption
    - Support streaming ballot ingestion
    - _Requirements: 5.6, 15.2_
  
  - [x] 12.4 Write property test for encryption round-trip
    - **Property 6: Encryption/Decryption Round-Trip**
    - Generate random plaintexts
    - Encrypt then decrypt
    - Verify equality
    - **Validates: Requirements 5.5, 5.2**
  
  - [x] 12.5 Implement batch homomorphic operations
    - Batch addition of thousands of ciphertexts
    - Parallel processing across GPU cores
    - Memory-efficient accumulation patterns
    - Target: Process 10,000+ ballots in under 5 seconds
    - _Requirements: 15.1, 15.2_

- [x] 13. Checkpoint - Basic FHE operations
  - Verify encryption/decryption round-trip passes
  - Benchmark key generation and encryption times
  - Ask the user if questions arise

- [x] 14. Homomorphic addition
  - [x] 14.1 Implement ciphertext-ciphertext addition
    - Add corresponding polynomial components
    - Track noise growth
    - _Requirements: 6.1_
  
  - [x] 14.2 Implement ciphertext-plaintext addition
    - Encode plaintext and add to ciphertext body
    - Optimize for common plaintext values
    - _Requirements: 6.2_
  
  - [x] 14.3 Write property test for homomorphic addition
    - **Property 7: Homomorphic Addition Correctness**
    - Generate random plaintext pairs
    - Encrypt, add ciphertexts, decrypt
    - Verify sum equals plaintext sum
    - Test commutativity and identity
    - **Validates: Requirements 6.1, 6.4, 6.5**

- [x] 15. Homomorphic multiplication
  - [x] 15.1 Implement ciphertext-ciphertext multiplication
    - Compute tensor product of ciphertext polynomials
    - Use NTT for efficient polynomial multiplication
    - _Requirements: 7.1_
  
  - [x] 15.2 Implement relinearization
    - Apply evaluation key to reduce ciphertext size
    - Use decomposition for noise control
    - _Requirements: 7.4_
  
  - [x] 15.3 Implement ciphertext-plaintext multiplication
    - Multiply ciphertext components by plaintext polynomial
    - Optimize for scalar plaintexts
    - _Requirements: 7.2_
  
  - [x] 15.4 Write property test for homomorphic multiplication
    - **Property 8: Homomorphic Multiplication Correctness**
    - Generate random plaintext pairs (bounded to avoid overflow)
    - Encrypt, multiply ciphertexts, decrypt
    - Verify product equals plaintext product
    - Test commutativity and scalar multiplication
    - **Validates: Requirements 7.1, 7.2, 7.5**
  
  - [x] 15.5 Implement ballot aggregation primitives
    - Implement homomorphic vote tallying (encrypted sum)
    - Support weighted voting with encrypted weights
    - Implement threshold detection without decryption
    - Optimize for large-scale ballot aggregation
    - _Requirements: 15.1, 15.8_
  
  - [x] 15.6 Implement comparison operations for fraud detection
    - Implement encrypted greater-than/less-than via PBS
    - Support range checks on encrypted values
    - Implement equality testing for duplicate detection
    - Optimize comparison circuits for voting patterns
    - _Requirements: 15.3, 15.7_

- [x] 16. Checkpoint - Homomorphic operations
  - Verify addition and multiplication properties pass
  - Measure noise growth per operation
  - Ask the user if questions arise

- [x] 17. TFHE bootstrapping implementation
  - [x] 17.1 Implement bootstrapping key generation
    - Generate GGSW encryptions of secret key bits
    - Parallelize across cores
    - Store in optimized memory layout
    - _Requirements: 4.4_
  
  - [x] 17.2 Implement blind rotate operation
    - Accumulator initialization with test polynomial
    - External product with bootstrapping key
    - SME-accelerated polynomial multiplication
    - _Requirements: 8.1, 8.3_
  
  - [x] 17.3 Implement sample extract and key switching
    - Extract LWE sample from GLWE ciphertext
    - Apply key switching to return to original key
    - _Requirements: 8.1_
  
  - [x] 17.4 Implement programmable bootstrapping
    - Support arbitrary lookup tables
    - Encode function in test polynomial
    - _Requirements: 8.6_
  
  - [x] 17.5 Write property tests for bootstrapping
    - **Property 9: Bootstrapping Value Preservation**
    - **Property 10: Programmable Bootstrapping LUT Application**
    - Bootstrap ciphertexts and verify value preservation
    - Test PBS with known functions (negation, threshold)
    - **Validates: Requirements 8.5, 8.6**
  
  - [x] 17.6 Implement fraud detection circuits
    - Detect duplicate voting patterns (encrypted comparison)
    - Implement statistical anomaly detection on encrypted data
    - Support threshold alerts without revealing individual votes
    - Implement time-series analysis for voting patterns
    - Optimize for real-time fraud detection
    - _Requirements: 15.3, 15.7_
  
  - [x] 17.7 Write property tests for fraud detection
    - **Property 13: Fraud Detection Correctness**
    - Verify duplicate detection without false positives
    - Test anomaly detection sensitivity
    - Validate privacy preservation during detection
    - Test with realistic voting patterns
    - **Validates: Requirements 15.3**

- [x] 18. Unconventional hardware optimizations
  - [x] 18.1 Implement Neural Engine modular reduction
    - Create CoreML model for modular reduction
    - Train on specific modulus values
    - Integrate with batch operations
    - Benchmark against CPU/GPU implementations
    - _Requirements: 14.7, 14.8, 14.9, 14.10, 14.11_
  
  - [x] 18.2 Implement AMX-accelerated operations via Accelerate
    - Use BLAS for matrix-form NTT
    - Implement Toeplitz polynomial multiplication
    - Benchmark matrix vs scalar implementations
    - _Requirements: 9.4, 22.2, 22.3_
  
  - [x] 18.3 Implement memory optimizations
    - Configure NTT-aware prefetch patterns
    - Enable hardware memory compression
    - Set up zero-copy IOSurface sharing
    - Benchmark memory bandwidth for each accelerator pair
    - _Requirements: 14.18, 14.19, 14.20, 14.21, 14.22_
  
  - [x] 18.4 Implement speculative execution for PBS
    - Pre-compute results for all possible inputs
    - Obliviously select correct result
    - Implement branch-free selection
    - _Requirements: 14.23, 14.24, 14.25_
  
  - [x] 18.5 Implement matrix-centric NTT
    - Express NTT as sparse butterfly matrix multiplications
    - Implement batched NTT as dense matrix multiply
    - Benchmark SME tile sizes for different polynomial degrees
    - _Requirements: 22.1, 22.6, 22.7_
  
  - [x] 18.6 Implement matrix-centric polynomial multiplication
    - Express polynomial multiplication as Toeplitz matrix-vector product
    - Implement circulant matrix formulation for cyclic convolution
    - Batch polynomial multiplication as matrix-matrix multiply
    - _Requirements: 22.2, 22.8_
  
  - [x] 18.7 Implement ray tracing hardware exploitation
    - Build BVH for key switching decomposition trees
    - Implement decomposition digit extraction via ray tracing
    - Benchmark against CPU tree traversal
    - _Requirements: 14.12, 14.13, 14.14_
  
  - [x] 18.8 Implement texture sampling for polynomial evaluation
    - Encode polynomials as textures
    - Implement twiddle factor texture sampling
    - Benchmark against direct computation
    - _Requirements: 14.15, 14.16, 14.17_
  
  - [x] 18.9 Implement advanced SIMD optimizations
    - Implement gather/scatter loads for NTT butterflies
    - Implement predicated modular reduction
    - Implement horizontal reductions for inner products
    - Benchmark SVE2 predication vs branching
    - _Requirements: 14.26, 14.27, 14.28, 14.29_
  
  - [x] 18.10 Implement SME streaming mode operations
    - Implement continuous polynomial processing pipelines
    - Implement SME2 predicated coefficient processing
    - Benchmark SME tile configurations
    - _Requirements: 14.30, 14.31, 14.32_
  
  - [x] 18.11 Implement Secure Enclave integration
    - Support secret key generation in Secure Enclave
    - Support final decryption in Secure Enclave
    - Support ciphertext signing
    - _Requirements: 14.33, 14.34, 14.35_
  
  - [x] 18.12 Write property tests for hardware backend equivalence
    - **Property 14: Hardware Backend Equivalence**
    - Verify all backends produce identical results
    - Test every operation on every available backend
    - **Validates: Requirements 14.36, 21.1**
  
  - [x] 18.13 Write property tests for matrix formulation equivalence
    - **Property 15: Matrix Formulation Equivalence**
    - Verify matrix and scalar implementations produce identical results
    - Test NTT, polynomial multiplication, key switching
    - **Validates: Requirements 22.8, 22.9**
  
  - [x] 18.14 Write property tests for speculative execution
    - **Property 16: Speculative Execution Correctness**
    - Verify speculative selection matches non-speculative result
    - Test PBS speculation, branch speculation
    - **Validates: Requirements 14.23, 14.24, 14.25**
  
  - [x] 18.15 Write property tests for unified memory consistency
    - **Property 17: Unified Memory Consistency**
    - Verify data consistency across CPU/GPU/Neural Engine
    - Test all accelerator pairs
    - **Validates: Requirements 14.21, 14.22**

- [x] 18-bench. Comprehensive hardware benchmarking suite
  - [x] 18-bench.1 Implement benchmark infrastructure
    - Create BenchmarkResult struct with latency, throughput, bandwidth, power
    - Implement statistical analysis (mean, stddev, significance testing)
    - Create benchmark report generator
    - _Requirements: 21.7, 21.8_
  
  - [x] 18-bench.2 Benchmark NTT on all hardware paths
    - Test SME, AMX, Metal GPU, NEON, CPU fallback
    - Test matrix-form vs scalar-form NTT
    - Test all polynomial degrees (1024 to 32768)
    - _Requirements: 21.1, 21.3_
  
  - [x] 18-bench.3 Benchmark modular multiplication on all hardware paths
    - Test Montgomery, Barrett, Neural Engine approximation
    - Test single-limb and multi-limb variants
    - Test batch sizes from 1 to 1M elements
    - _Requirements: 21.1, 21.4_
  
  - [x] 18-bench.4 Benchmark polynomial multiplication on all hardware paths
    - Test NTT-based, Toeplitz matrix, circulant matrix
    - Test all polynomial degrees
    - _Requirements: 21.1, 21.3_
  
  - [x] 18-bench.5 Benchmark Neural Engine operations
    - Test modular reduction network accuracy and speed
    - Test Poseidon hash network
    - Test LUT evaluation
    - Compare against CPU/GPU implementations
    - _Requirements: 21.4, 21.5_
  
  - [x] 18-bench.6 Benchmark ray tracing operations
    - Test decomposition tree traversal
    - Test Merkle tree traversal
    - Compare against CPU implementations
    - _Requirements: 21.6_
  
  - [x] 18-bench.7 Benchmark texture sampling operations
    - Test polynomial evaluation via texture sampling
    - Test twiddle factor texture lookup
    - Compare against direct computation
    - _Requirements: 21.6_
  
  - [x] 18-bench.8 Benchmark memory system
    - Test bandwidth between all accelerator pairs
    - Test cache effects for different data sizes
    - Test unified memory vs explicit copy
    - _Requirements: 21.9, 21.2_
  
  - [x] 18-bench.9 Benchmark pipelined operations
    - Test combinations of hardware units
    - Test speculative execution overhead
    - Identify optimal pipeline configurations
    - _Requirements: 21.10, 21.11_
  
  - [x] 18-bench.10 Generate comprehensive benchmark report
    - Produce detailed comparison tables
    - Identify optimal hardware path for each operation/size
    - Document statistical significance of results
    - Create visualization of results
    - _Requirements: 21.7, 21.8, 21.12_

- [x] 18a. Zero-Knowledge proof system infrastructure
  - [x] 18a.1 Implement finite field arithmetic for ZK
    - Implement BLS12-381 and BN254 curve operations
    - Implement field addition, multiplication, inversion
    - Use Montgomery form for efficient modular arithmetic
    - Reuse FHE modular arithmetic infrastructure
    - _Requirements: 19, 20_
  
  - [x] 18a.2 Implement elliptic curve operations
    - Implement point addition and doubling
    - Implement scalar multiplication
    - Implement multi-scalar multiplication (MSM)
    - Use Metal GPU for parallel MSM computation
    - _Requirements: 19, 20.2_
  
  - [x] 18a.3 Implement polynomial commitment schemes
    - Implement KZG commitments for PLONK
    - Implement FRI commitments for STARKs
    - Reuse NTT infrastructure for FFT operations
    - Use Metal GPU for parallel commitment computation
    - _Requirements: 19, 20.3_
  
  - [x] 18a.4 Implement cryptographic hash functions
    - Implement Poseidon hash (ZK-friendly)
    - Implement Blake2s for Bulletproofs
    - Use Metal GPU for parallel hash tree construction
    - Optimize for M4 Max cache hierarchy
    - _Requirements: 19, 20.4_

- [x] 18b. Bulletproofs implementation (ballot validity)
  - [x] 18b.1 Implement range proof generation
    - Implement inner product argument
    - Support range proofs for vote validity (choice ∈ {0,1,2,...})
    - Optimize for small ranges (2-10 candidates)
    - Target <50ms proof generation on M4 Max
    - _Requirements: 19.1, 20_
  
  - [x] 18b.2 Implement range proof verification
    - Implement efficient verification algorithm
    - Target <5ms verification time
    - Support batch verification for multiple proofs
    - _Requirements: 19.6, 19.8_
  
  - [x] 18b.3 Write property tests for Bulletproofs
    - **Property 14: Bulletproof Soundness**
    - Verify invalid ranges are rejected
    - Verify valid ranges are accepted
    - Test proof size is under 1KB
    - **Validates: Requirements 19.1, 19.9**

- [x] 18c. Groth16 implementation (eligibility proofs)
  - [x] 18c.1 Implement R1CS constraint system
    - Create constraint builder for eligibility circuits
    - Implement witness generation
    - Support Merkle tree membership proofs
    - _Requirements: 19.2, 20.6_
  
  - [x] 18c.2 Implement Groth16 proving
    - Implement trusted setup ceremony
    - Implement proof generation with Metal GPU acceleration
    - Target <100ms proof generation
    - Optimize MSM using Metal parallel point operations
    - _Requirements: 19.2, 20.2_
  
  - [x] 18c.3 Implement Groth16 verification
    - Implement pairing-based verification
    - Target <1ms verification time
    - Support batch verification
    - _Requirements: 19.6, 19.8_
  
  - [x] 18c.4 Write property tests for Groth16
    - **Property 15: Groth16 Soundness**
    - Verify invalid witnesses are rejected
    - Verify valid witnesses are accepted
    - Test proof size is ~200 bytes
    - **Validates: Requirements 19.2, 19.9**

- [x] 18d. PLONK implementation (tally correctness)
  - [x] 18d.1 Implement PLONK constraint system
    - Create constraint builder for tally circuits
    - Implement custom gates for FHE operations
    - Support variable-size circuits
    - _Requirements: 19.4, 20_
  
  - [x] 18d.2 Implement PLONK proving
    - Implement universal trusted setup
    - Implement proof generation with KZG commitments
    - Use Metal GPU for FFT operations (reuse NTT)
    - Target <150ms proof generation
    - _Requirements: 19.4, 20.3_
  
  - [x] 18d.3 Implement PLONK verification
    - Implement KZG opening verification
    - Target <5ms verification time
    - _Requirements: 19.6, 19.8_
  
  - [x] 18d.4 Write property tests for PLONK
    - **Property 16: PLONK Soundness**
    - Verify incorrect computations are rejected
    - Verify correct computations are accepted
    - Test proof size is ~400 bytes
    - **Validates: Requirements 19.4, 19.9**

- [x] 18e. Hardware-accelerated ZK optimizations
  - [x] 18e.1 Implement Metal GPU MSM
    - Pippenger's algorithm for multi-scalar multiplication
    - Parallel bucket accumulation on GPU
    - Optimize for M4 Max 40-core GPU
    - Target 10x speedup over CPU
    - _Requirements: 20.2, 20.7_
  
  - [x] 18e.2 Implement Metal GPU FFT for ZK
    - Reuse NTT infrastructure for FFT over ZK fields
    - Support BLS12-381 and BN254 scalar fields
    - Batch FFT operations for multiple polynomials
    - _Requirements: 20.3_
  
  - [x] 18e.3 Implement Neural Engine hash acceleration
    - Use Neural Engine for parallel Poseidon hashing
    - Implement as quantized neural network
    - Target 100x speedup for hash tree construction
    - _Requirements: 20.4, 20.5_
  
  - [x] 18e.4 Implement AMX-accelerated constraint evaluation
    - Use AMX for R1CS matrix-vector products
    - Optimize witness generation with BLAS
    - _Requirements: 20.6_
  
  - [x] 18e.5 Implement batch proof generation
    - Parallelize proof generation across GPU cores
    - Support generating 100+ proofs simultaneously
    - Share setup data across proofs
    - _Requirements: 20.8_

- [x] 19. Streaming operations
  - [x] 19.1 Implement chunked ciphertext processing
    - Split large ciphertexts into processable chunks
    - Maintain correctness across chunk boundaries
    - _Requirements: 11.1, 11.2_
  
  - [x] 19.2 Implement async streaming interface
    - Create AsyncIterable for ciphertext streams
    - Support progress callbacks
    - _Requirements: 11.3_
  
  - [x] 19.3 Write property test for streaming equivalence
    - **Property 12: Streaming Equivalence**
    - Process data via streaming and non-streaming
    - Verify bit-identical results
    - **Validates: Requirements 11.6**

- [x] 20. TypeScript API implementation
  - [x] 20.1 Implement napi-rs bindings for all operations
    - Create TypeScript type definitions
    - Implement async wrappers with Promises
    - Handle errors with typed exceptions
    - _Requirements: 12.1, 12.2, 12.3, 12.6_
  
  - [x] 20.2 Implement Node.js Stream interfaces
    - Create Readable/Writable streams for ciphertexts
    - Support pipe() for operation chaining
    - _Requirements: 12.4_
  
  - [x] 20.3 Create high-level convenience API
    - Implement FHEContext for simplified usage
    - Add parameter preset helpers
    - Create example usage documentation
    - _Requirements: 12.1_
  
  - [x] 20.4 Implement real-time tally streaming API
    - WebSocket support for live encrypted tallies
    - Progressive result disclosure (encrypted running totals)
    - Support for multiple concurrent elections
    - Handle 1,000+ concurrent subscribers
    - _Requirements: 18_
  
  - [x] 20.5 Implement audit trail API
    - Cryptographic audit log of all operations
    - Support for post-election verification
    - Export formats for election officials (JSON, CSV)
    - Immutable operation history with timestamps
    - _Requirements: 17_
  
  - [x] 20.6 Implement ZK proof API
    - TypeScript bindings for all proof systems
    - Async proof generation and verification
    - Batch proof operations
    - Proof serialization and deserialization
    - _Requirements: 19_
  
  - [x] 20.7 Create voting system example with ZK
    - Complete end-to-end voting example
    - Demonstrate ballot validity proofs
    - Show eligibility and uniqueness proofs
    - Include tally correctness proofs
    - Demonstrate threshold decryption with proofs
    - Show fraud detection integration
    - Include audit trail generation
    - _Requirements: 15, 17, 18, 19_

- [x] 21. Final checkpoint - Full integration
  - Run all property tests with 100+ iterations
  - Verify sub-100ms latency for typical operations
  - Verify sub-20ms bootstrapping on M4 Max
  - **Verify 10,000+ ballots/second ingestion rate**
  - **Verify <5 second tally time for 100,000 ballots**
  - **Verify <1MB memory per encrypted ballot**
  - **Verify <10KB serialized ballot size**
  - **Test 24-hour continuous operation stability**
  - **Verify threshold decryption with 3-of-5 officials**
  - **Test fraud detection on realistic voting patterns**
  - **Verify audit trail integrity and completeness**
  - **Verify ZK proof generation <200ms per ballot**
  - **Verify ZK proof verification <20ms per ballot**
  - **Verify ZK proof size <2KB per ballot**
  - **Test batch ZK proof generation (100+ proofs)**
  - **Verify 5x ZK speedup on M4 Max vs CPU**
  - Ask the user if questions arise

- [x] 22. Voting system integration and deployment
  - Create deployment guide for Mac Studio/M4 Max
  - Document election official setup procedures
  - Create voter client examples (web, mobile)
  - Implement monitoring and alerting
  - Create disaster recovery procedures
  - Document security best practices
  - **Document ZK proof verification for observers**
  - **Create public verification tools**
  - _Requirements: 15, 16, 17, 18, 19_

## Notes

- All property tests are required for comprehensive correctness validation
- Each property test references its design document property number
- Checkpoints ensure incremental validation before proceeding
- Hardware-specific optimizations (SME, Metal, Neural Engine) have fallbacks to ensure correctness on all platforms
- Performance targets: encryption < 10ms, addition < 1ms, multiplication < 50ms, bootstrapping < 20ms
- **Voting-specific targets: 10,000+ ballots/sec ingestion, <5s tally for 100K ballots, <10KB per ballot**
- **Security targets: M-of-N threshold decryption, zero-knowledge proofs, cryptographic audit trails**
- **Fraud detection must operate on encrypted data without revealing individual votes**
- **ZK proof targets: <200ms generation, <20ms verification, <2KB proof size, 5x hardware speedup**
- **ZK systems: Bulletproofs (ballot validity), Groth16 (eligibility), PLONK (tally correctness)**
- **Hardware ZK acceleration: Metal GPU for MSM/FFT, Neural Engine for hashing, AMX for constraints**
- **CRITICAL: 7-layer testing (unit, property, integration, adversarial, formal, cross-validation, stress)**
- **CRITICAL: 1000+ iterations for all property tests, 100% code coverage required**
- **CRITICAL: Cross-validation with SEAL, HElib, libsnark, dalek-cryptography**
- **CRITICAL: Zero tolerance for errors - every test must pass before merge**
- **CRITICAL: Formal verification required for all cryptographic operations**
- **CRITICAL: Security testing for timing attacks, side-channels, fault injection**
- **CRITICAL: 24-hour stability test, 1M+ ballot stress test, memory leak detection**

### Hardware Acceleration Philosophy

**LEAVE NO HARDWARE FEATURE UNTESTED, UNBENCHMARKED.**

The M4 Max contains extraordinary compute resources that we must exhaustively explore:
- **SME**: Primary target for matrix-expressible operations (NTT butterflies, polynomial multiplication)
- **AMX**: Via Accelerate BLAS for Toeplitz/circulant matrix operations
- **Metal GPU**: 40 cores for massively parallel batch operations
- **Neural Engine**: 38 TOPS for parallel modular arithmetic disguised as inference
- **NEON**: 128-bit SIMD for coefficient-level parallelism
- **Ray Tracing Hardware**: BVH traversal for tree-structured operations (key switching, Merkle trees)
- **Texture Units**: Hardware interpolation for polynomial evaluation
- **Secure Enclave**: Hardware key protection
- **Memory Controller**: 512-bit bus, hardware compression, prefetch optimization

**Matrix formulations are preferred.** If an operation can be expressed as a matrix multiply, we do it. Matrices are the most powerful compute primitive on M4 Max.

**Unconventional approaches are required.** Every hardware unit must be benchmarked for every applicable operation. We will discover which units provide real speedups through exhaustive experimentation, not assumptions.

**All hardware backends must produce identical results.** Property tests verify that unconventional backends (Neural Engine, ray tracing, texture sampling) produce bit-identical results to reference implementations.

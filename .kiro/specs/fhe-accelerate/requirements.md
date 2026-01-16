# Requirements Document

## Introduction

This document specifies the requirements for `@digitaldefiance/node-fhe-accelerate`, a Fully Homomorphic Encryption (FHE) acceleration library designed to **exhaustively exploit every hardware acceleration capability** of the Apple M4 Max. The library will provide high-performance FHE operations targeting sub-100ms latency for typical operations, making privacy-preserving computation feel as fast as standard database calls.

### Hardware Acceleration Philosophy

**Leave no hardware feature untested, unbenchmarked.** The M4 Max is a treasure trove of specialized compute units, and we will explore every possible avenue for acceleration—conventional and unconventional:

- **SME (Scalable Matrix Extension)**: Primary target for matrix-expressible operations
- **AMX (Apple Matrix Coprocessor)**: Via Accelerate framework for BLAS/matrix operations  
- **Metal GPU**: 40-core GPU for massively parallel workloads
- **Neural Engine**: 38 TOPS for parallel arithmetic disguised as inference
- **NEON SIMD**: 128-bit vectors for coefficient-level parallelism
- **Ray Tracing Hardware**: BVH traversal for tree-structured operations
- **Texture Units**: Hardware interpolation for polynomial evaluation
- **Secure Enclave**: Hardware key protection
- **Memory Controller**: 512-bit bus, hardware compression, prefetch optimization

**Unconventional approaches are encouraged.** If we can reformulate an FHE operation as a matrix multiply, we do it. If we can encode modular reduction as a neural network, we try it. If ray tracing hardware can accelerate tree traversal in key switching, we benchmark it. The goal is to discover which hardware units provide real speedups for which operations through exhaustive experimentation.

The library will be implemented as a standalone native addon using C++ with napi-rs bindings, optimized specifically for FHE workloads. While Reed-Solomon operations from node-rs-accelerate share some mathematical foundations (Galois Field arithmetic), FHE requires fundamentally different primitives (NTT over prime fields, large modular arithmetic) that warrant a fresh implementation optimized for FHE-specific access patterns.

## Glossary

- **FHE_Engine**: The core computation engine that orchestrates FHE operations across hardware accelerators
- **NTT_Processor**: Component responsible for Number Theoretic Transform operations
- **Polynomial_Ring**: Data structure representing elements in Z_q[X]/(X^N+1)
- **Ciphertext**: Encrypted data structure containing one or more polynomials
- **Plaintext**: Unencrypted data ready for encryption or result of decryption
- **Key_Manager**: Component handling key generation, storage, and key switching operations
- **Bootstrap_Engine**: Component performing ciphertext refresh operations to reduce noise
- **Hardware_Dispatcher**: Component that routes operations to optimal hardware (SME, Metal, NEON, AMX)
- **Parameter_Set**: Configuration defining security level, polynomial degree, and modulus chain
- **Montgomery_Reducer**: Component performing modular reduction using Montgomery representation
- **Streaming_Processor**: Component handling large ciphertext operations in chunks
- **Ballot_Aggregator**: Component for homomorphic tallying of encrypted votes
- **Fraud_Detector**: Component for detecting voting anomalies on encrypted data
- **Threshold_Decryptor**: Component for M-of-N threshold decryption of election results
- **Audit_Logger**: Component for maintaining cryptographic audit trails
- **ZK_Prover**: Component for generating zero-knowledge proofs of ballot validity and computation correctness
- **ZK_Verifier**: Component for verifying zero-knowledge proofs
- **Proof_Aggregator**: Component for batching and aggregating multiple ZK proofs

## Requirements

### Requirement 1: Number Theoretic Transform (NTT) Operations

**User Story:** As a developer, I want hardware-accelerated NTT operations, so that polynomial multiplication achieves near-theoretical throughput on M4 Max.

#### Acceptance Criteria

1. WHEN performing forward NTT on a polynomial of degree N, THE NTT_Processor SHALL compute the transform in O(N log N) operations using precomputed twiddle factors
2. WHEN performing inverse NTT, THE NTT_Processor SHALL restore the original polynomial representation with exact precision
3. WHEN the SME unit is available, THE Hardware_Dispatcher SHALL route NTT butterfly operations to SME matrix registers
4. WHEN processing multiple polynomials, THE NTT_Processor SHALL batch operations for Metal GPU parallel execution
5. WHEN polynomial degree is 2^14 or larger, THE NTT_Processor SHALL achieve at least 1 million NTT operations per second
6. FOR ALL valid polynomials, applying forward NTT then inverse NTT SHALL produce the original polynomial (round-trip property)

### Requirement 2: Modular Arithmetic Engine

**User Story:** As a developer, I want optimized large-integer modular arithmetic, so that coefficient operations don't bottleneck FHE computations.

#### Acceptance Criteria

1. THE Montgomery_Reducer SHALL perform modular multiplication without division operations
2. WHEN coefficients exceed 64 bits, THE Montgomery_Reducer SHALL use multi-limb representation with NEON SIMD vectorization
3. WHEN performing Barrett reduction, THE FHE_Engine SHALL precompute reduction constants for each modulus in the chain
4. WHEN modulus is a prime of form q = 1 (mod 2N), THE Montgomery_Reducer SHALL exploit the special structure for faster reduction
5. FOR ALL coefficient pairs (a, b) and modulus q, computing (a * b) mod q SHALL produce mathematically correct results
6. WHEN processing coefficient vectors, THE FHE_Engine SHALL achieve at least 10 billion modular multiplications per second using NEON

### Requirement 3: Polynomial Ring Operations

**User Story:** As a developer, I want efficient polynomial arithmetic in cyclotomic rings, so that I can perform homomorphic operations on ciphertexts.

#### Acceptance Criteria

1. WHEN multiplying two polynomials, THE Polynomial_Ring SHALL use NTT-based multiplication for O(N log N) complexity
2. WHEN adding polynomials, THE Polynomial_Ring SHALL perform coefficient-wise addition with modular reduction
3. WHEN the result exceeds the ring modulus, THE Polynomial_Ring SHALL automatically reduce modulo X^N + 1
4. THE Polynomial_Ring SHALL support polynomial degrees N in {1024, 2048, 4096, 8192, 16384, 32768}
5. FOR ALL polynomial pairs (p1, p2), multiplication SHALL be commutative: p1 * p2 = p2 * p1
6. FOR ALL polynomials p, multiplying by the multiplicative identity SHALL return p unchanged

### Requirement 4: Key Generation and Management

**User Story:** As a developer, I want secure and efficient key generation, so that I can set up FHE contexts quickly.

#### Acceptance Criteria

1. WHEN generating a secret key, THE Key_Manager SHALL sample coefficients from a secure random distribution
2. WHEN generating a public key, THE Key_Manager SHALL compute the key pair in under 100ms for 128-bit security
3. WHEN generating evaluation keys for key switching, THE Key_Manager SHALL support configurable decomposition bases
4. WHEN generating bootstrapping keys, THE Key_Manager SHALL parallelize computation across available cores
5. THE Key_Manager SHALL support serialization and deserialization of all key types
6. WHEN loading serialized keys, THE Key_Manager SHALL validate key integrity before use

### Requirement 5: Encryption and Decryption

**User Story:** As a developer, I want fast encryption and decryption, so that data can move in and out of the encrypted domain efficiently.

#### Acceptance Criteria

1. WHEN encrypting a plaintext, THE FHE_Engine SHALL produce a valid ciphertext in under 10ms for typical parameters
2. WHEN decrypting a ciphertext, THE FHE_Engine SHALL recover the original plaintext if noise budget permits
3. WHEN encrypting with SIMD packing, THE FHE_Engine SHALL encode multiple values into a single ciphertext
4. IF the noise budget is exhausted during decryption, THEN THE FHE_Engine SHALL return a descriptive error
5. FOR ALL plaintexts p, encrypting then decrypting SHALL recover the original value (round-trip property)
6. WHEN batch encrypting multiple plaintexts, THE FHE_Engine SHALL use Metal GPU for parallel encryption

### Requirement 6: Homomorphic Addition

**User Story:** As a developer, I want to add encrypted values, so that I can perform additive computations on private data.

#### Acceptance Criteria

1. WHEN adding two ciphertexts, THE FHE_Engine SHALL produce a ciphertext encrypting the sum of plaintexts
2. WHEN adding a ciphertext and plaintext, THE FHE_Engine SHALL perform the operation without decryption
3. THE FHE_Engine SHALL complete homomorphic addition in under 1ms for typical parameters
4. FOR ALL ciphertext pairs (c1, c2), addition SHALL be commutative: c1 + c2 = c2 + c1
5. FOR ALL ciphertexts c, adding the encryption of zero SHALL return an equivalent ciphertext
6. WHEN the noise budget is insufficient, THE FHE_Engine SHALL report the remaining budget before failure

### Requirement 7: Homomorphic Multiplication

**User Story:** As a developer, I want to multiply encrypted values, so that I can perform multiplicative computations on private data.

#### Acceptance Criteria

1. WHEN multiplying two ciphertexts, THE FHE_Engine SHALL produce a ciphertext encrypting the product of plaintexts
2. WHEN multiplying a ciphertext by a plaintext, THE FHE_Engine SHALL perform scalar multiplication efficiently
3. THE FHE_Engine SHALL complete homomorphic multiplication in under 50ms for typical parameters
4. WHEN multiplication increases ciphertext size, THE FHE_Engine SHALL perform relinearization automatically
5. FOR ALL ciphertext pairs (c1, c2), multiplication SHALL be commutative: c1 * c2 = c2 * c1
6. WHEN the noise growth exceeds threshold, THE FHE_Engine SHALL trigger bootstrapping if enabled

### Requirement 8: TFHE Bootstrapping

**User Story:** As a developer, I want fast bootstrapping, so that I can perform unlimited depth computations on encrypted data.

#### Acceptance Criteria

1. WHEN bootstrapping a TFHE ciphertext, THE Bootstrap_Engine SHALL refresh the noise to initial levels
2. THE Bootstrap_Engine SHALL complete bootstrapping in under 20ms on M4 Max hardware
3. WHEN bootstrapping, THE Bootstrap_Engine SHALL use SME for accumulator polynomial multiplication
4. WHEN bootstrapping keys are preloaded, THE Bootstrap_Engine SHALL achieve consistent latency
5. FOR ALL ciphertexts c, bootstrapping SHALL preserve the encrypted plaintext value
6. WHEN performing programmable bootstrapping, THE Bootstrap_Engine SHALL apply the lookup table function

### Requirement 9: Hardware Acceleration Dispatch

**User Story:** As a developer, I want automatic hardware selection, so that operations use the optimal accelerator without manual configuration.

#### Acceptance Criteria

1. WHEN SME is available and operation is matrix-compatible, THE Hardware_Dispatcher SHALL route to SME
2. WHEN processing batches of independent operations, THE Hardware_Dispatcher SHALL route to Metal GPU
3. WHEN operation requires low latency single execution, THE Hardware_Dispatcher SHALL use NEON SIMD
4. WHEN AMX provides benefit via Accelerate framework, THE Hardware_Dispatcher SHALL utilize it
5. THE Hardware_Dispatcher SHALL support manual override for benchmarking purposes
6. WHEN hardware is unavailable, THE Hardware_Dispatcher SHALL fall back gracefully to the next best option

### Requirement 10: Parameter Presets and Security Levels

**User Story:** As a developer, I want predefined parameter sets, so that I can easily configure appropriate security levels.

#### Acceptance Criteria

1. THE Parameter_Set SHALL provide presets for 128-bit, 192-bit, and 256-bit security levels
2. WHEN selecting a preset, THE FHE_Engine SHALL configure all dependent parameters automatically
3. THE Parameter_Set SHALL include recommended parameters for TFHE, BFV, and CKKS schemes
4. WHEN custom parameters are provided, THE Parameter_Set SHALL validate security constraints
5. IF parameters fail security validation, THEN THE Parameter_Set SHALL return specific violation details
6. THE Parameter_Set SHALL expose estimated noise budget and operation depth for each configuration

### Requirement 11: Streaming Operations for Large Ciphertexts

**User Story:** As a developer, I want streaming support for large operations, so that memory constraints don't limit computation size.

#### Acceptance Criteria

1. WHEN ciphertext size exceeds available memory, THE Streaming_Processor SHALL process in chunks
2. WHEN streaming, THE Streaming_Processor SHALL maintain operation correctness across chunk boundaries
3. THE Streaming_Processor SHALL support async iteration with progress callbacks
4. WHEN streaming to disk, THE Streaming_Processor SHALL use memory-mapped I/O for efficiency
5. THE Streaming_Processor SHALL leverage unified memory for zero-copy CPU/GPU data sharing
6. WHEN streaming completes, THE Streaming_Processor SHALL produce results identical to non-streaming execution

### Requirement 12: TypeScript/JavaScript API

**User Story:** As a JavaScript developer, I want a familiar API, so that I can integrate FHE into Node.js applications easily.

#### Acceptance Criteria

1. THE FHE_Engine SHALL expose a TypeScript API following patterns from node-rs-accelerate
2. WHEN performing async operations, THE FHE_Engine SHALL return Promises with proper error handling
3. THE FHE_Engine SHALL provide TypeScript type definitions for all public interfaces
4. WHEN operations support streaming, THE FHE_Engine SHALL expose Node.js Stream interfaces
5. THE FHE_Engine SHALL support both CommonJS and ES Module imports
6. WHEN errors occur, THE FHE_Engine SHALL throw typed exceptions with actionable messages

### Requirement 13: Native C++ Implementation with Node.js Bindings

**User Story:** As a developer, I want a high-performance native implementation, so that FHE operations achieve maximum hardware utilization.

#### Acceptance Criteria

1. THE FHE_Engine SHALL be implemented in C++ with napi-rs bindings for Node.js
2. THE FHE_Engine SHALL use Metal framework directly for GPU compute shaders
3. THE FHE_Engine SHALL use ARM intrinsics for NEON and SME operations
4. WHEN compiling, THE FHE_Engine SHALL produce universal binaries for arm64 macOS
5. THE FHE_Engine SHALL minimize JavaScript-to-native boundary crossings for performance
6. THE FHE_Engine SHALL support prebuilt binaries for common platforms

### Requirement 14: Exhaustive Unconventional Hardware Exploitation

**User Story:** As a performance engineer, I want to exploit every possible hardware acceleration path on M4 Max, including unconventional approaches, so that I can achieve breakthrough performance through exhaustive experimentation.

#### Acceptance Criteria

##### Matrix-Based Acceleration (SME/AMX)
1. THE FHE_Engine SHALL reformulate NTT butterfly operations as matrix multiplications for SME acceleration
2. THE FHE_Engine SHALL express polynomial multiplication as Toeplitz matrix-vector products for AMX
3. THE FHE_Engine SHALL use outer product accumulation on SME for coefficient-wise operations
4. THE FHE_Engine SHALL implement key switching as batched matrix operations
5. WHEN operations can be expressed as matrices, THE Hardware_Dispatcher SHALL prefer matrix form over scalar form
6. THE FHE_Engine SHALL benchmark matrix vs scalar implementations for every applicable operation

##### Neural Engine Exploitation
7. WHEN Neural Engine is available, THE Hardware_Dispatcher SHALL explore using it for parallel modular reductions encoded as quantized neural networks
8. THE FHE_Engine SHALL implement Poseidon hash computation on Neural Engine for ZK proof generation
9. THE FHE_Engine SHALL explore Neural Engine for parallel LUT evaluation in TFHE bootstrapping
10. THE FHE_Engine SHALL benchmark Neural Engine vs GPU for batch element-wise operations
11. THE FHE_Engine SHALL compile modular reduction for specific moduli as optimized ANE models

##### GPU Ray Tracing Hardware
12. THE FHE_Engine SHALL explore ray tracing BVH traversal hardware for key switching decomposition trees
13. THE FHE_Engine SHALL benchmark ray tracing hardware for blind rotate index computation in bootstrapping
14. THE FHE_Engine SHALL encode decomposition digit extraction as ray-scene intersection queries

##### GPU Texture Sampling
15. THE FHE_Engine SHALL explore texture sampling hardware for polynomial evaluation at multiple points
16. THE FHE_Engine SHALL encode twiddle factor tables as textures for on-the-fly NTT computation
17. THE FHE_Engine SHALL benchmark texture interpolation for lookup table evaluation

##### Memory System Optimization
18. THE FHE_Engine SHALL implement custom memory layouts optimized for M4 Max 192KB L1 / 32MB L2 cache hierarchy
19. THE FHE_Engine SHALL exploit hardware memory compression for ciphertext storage
20. THE FHE_Engine SHALL configure NTT-specific prefetch patterns based on butterfly access patterns
21. THE FHE_Engine SHALL use unified memory for zero-copy sharing between CPU/GPU/Neural Engine
22. THE FHE_Engine SHALL benchmark IOSurface shared buffers vs MTLBuffer for accelerator data transfer

##### Speculative and Predictive Execution
23. WHEN processing predictable ciphertext patterns, THE FHE_Engine SHALL use speculative execution
24. THE FHE_Engine SHALL speculatively compute all possible PBS outputs and select based on decrypted value
25. THE FHE_Engine SHALL implement branch-free oblivious selection for speculative results

##### Advanced SIMD (NEON/SVE2)
26. THE FHE_Engine SHALL use gather/scatter loads for NTT butterfly patterns
27. THE FHE_Engine SHALL use predicated operations for conditional modular reduction
28. THE FHE_Engine SHALL implement horizontal reductions for inner products in key switching
29. THE FHE_Engine SHALL benchmark SVE2 predication vs branching for coefficient processing

##### SME Streaming Mode
30. THE FHE_Engine SHALL use SME streaming mode for continuous polynomial processing pipelines
31. THE FHE_Engine SHALL implement SME2 predicated operations for conditional coefficient processing
32. THE FHE_Engine SHALL benchmark SME tile sizes for different polynomial degrees

##### Secure Enclave Integration
33. THE FHE_Engine SHALL support secret key generation inside Secure Enclave
34. THE FHE_Engine SHALL support final decryption inside Secure Enclave (key never exposed)
35. THE FHE_Engine SHALL support ciphertext signing via Secure Enclave

##### Comprehensive Benchmarking
36. THE FHE_Engine SHALL benchmark EVERY hardware path for EVERY applicable operation
37. THE FHE_Engine SHALL produce detailed performance reports comparing all acceleration strategies
38. THE FHE_Engine SHALL support runtime hardware path selection based on benchmark results
39. THE FHE_Engine SHALL log which hardware unit was selected for each operation type

### Requirement 15: Voting-Specific Operations

**User Story:** As an election official, I want to tally encrypted votes in real-time, so that results can be announced quickly without compromising voter privacy.

#### Acceptance Criteria

1. WHEN aggregating encrypted ballots, THE FHE_Engine SHALL compute running totals without decryption
2. THE FHE_Engine SHALL process at least 10,000 ballots per second on M4 Max hardware
3. WHEN detecting fraud patterns, THE FHE_Engine SHALL identify anomalies without revealing individual votes
4. THE FHE_Engine SHALL support threshold decryption requiring M-of-N election officials
5. WHEN generating receipts, THE FHE_Engine SHALL provide verifiable proof without revealing vote content
6. THE FHE_Engine SHALL maintain less than 1MB memory footprint per encrypted ballot
7. WHEN performing fraud detection, THE FHE_Engine SHALL support encrypted comparison operations
8. THE FHE_Engine SHALL implement homomorphic vote tallying with encrypted sum operations

### Requirement 16: Network Efficiency for Ballot Transmission

**User Story:** As a voter, I want to submit my ballot quickly over any network, so that voting is accessible even with limited bandwidth.

#### Acceptance Criteria

1. THE FHE_Engine SHALL compress encrypted ballots to less than 10KB per ballot
2. WHEN transmitting ballots, THE FHE_Engine SHALL support resumable uploads for unreliable networks
3. THE FHE_Engine SHALL validate ballot integrity before processing
4. THE FHE_Engine SHALL support batch submission of multiple ballots
5. THE FHE_Engine SHALL provide progress feedback during ballot submission
6. WHEN serializing ballots, THE FHE_Engine SHALL support versioning for protocol upgrades
7. THE FHE_Engine SHALL support partial deserialization for quick verification

### Requirement 17: Auditability and Compliance

**User Story:** As an election auditor, I want cryptographic proof of correct tallying, so that results can be independently verified.

#### Acceptance Criteria

1. THE FHE_Engine SHALL generate cryptographic audit logs for all ballot operations
2. WHEN tallying completes, THE FHE_Engine SHALL provide zero-knowledge proofs of correctness
3. THE FHE_Engine SHALL support post-election verification without re-decryption
4. THE FHE_Engine SHALL maintain immutable operation history with timestamps
5. THE FHE_Engine SHALL export audit data in standard formats (JSON, CSV)
6. WHEN generating audit trails, THE FHE_Engine SHALL include operation metadata
7. THE FHE_Engine SHALL support cryptographic verification of audit log integrity

### Requirement 18: Real-Time Results and Live Tallying

**User Story:** As an election observer, I want to see encrypted running totals in real-time, so that I can monitor election progress without compromising privacy.

#### Acceptance Criteria

1. THE FHE_Engine SHALL provide WebSocket API for live encrypted tally streaming
2. WHEN processing ballots, THE FHE_Engine SHALL update running totals in under 100ms
3. THE FHE_Engine SHALL support multiple concurrent elections without interference
4. THE FHE_Engine SHALL provide progressive result disclosure with encrypted intermediate totals
5. WHEN streaming results, THE FHE_Engine SHALL maintain consistent state across clients
6. THE FHE_Engine SHALL support real-time fraud detection alerts
7. THE FHE_Engine SHALL handle at least 1,000 concurrent result stream subscribers

### Requirement 19: Zero-Knowledge Verifiability

**User Story:** As an election observer, I want cryptographic proof that all ballots are valid and the tally is correct, without revealing any votes or voter identities.

#### Acceptance Criteria

1. WHEN a voter submits a ballot, THE ZK_Prover SHALL generate a proof that the ballot contains a valid choice without revealing the choice
2. WHEN a voter submits a ballot, THE ZK_Prover SHALL generate a proof of voter eligibility without revealing voter identity
3. WHEN a voter submits a ballot, THE ZK_Prover SHALL generate a proof of uniqueness (no double voting) without revealing identity
4. WHEN computing the tally, THE ZK_Prover SHALL generate a proof of correct homomorphic computation
5. WHEN officials decrypt results, THE ZK_Prover SHALL generate proofs of correct partial decryption
6. THE ZK_Prover SHALL support public verification of all proofs by any party
7. ZK proof generation SHALL complete in under 200ms per ballot on M4 Max
8. ZK proof verification SHALL complete in under 20ms per ballot
9. ZK proofs SHALL serialize to under 2KB per ballot
10. THE ZK_Prover SHALL support multiple proof systems (Bulletproofs, Groth16, PLONK, STARKs)

### Requirement 20: Hardware-Accelerated Zero-Knowledge Proving

**User Story:** As a performance engineer, I want ZK proof generation to leverage M4 Max hardware, so that proving is fast enough for real-time voting.

#### Acceptance Criteria

1. WHEN generating ZK proofs, THE ZK_Prover SHALL use Metal GPU for parallel field operations
2. WHEN computing MSMs (Multi-Scalar Multiplications), THE ZK_Prover SHALL use Metal GPU for parallel point operations
3. WHEN computing FFTs for polynomial commitments, THE ZK_Prover SHALL reuse NTT infrastructure
4. WHEN computing Poseidon hashes, THE ZK_Prover SHALL use Metal GPU for parallel hash tree construction
5. THE ZK_Prover SHALL use Neural Engine for parallel modular arithmetic in proof generation
6. THE ZK_Prover SHALL use AMX for matrix operations in R1CS constraint evaluation
7. WHEN proving on M4 Max, THE ZK_Prover SHALL achieve at least 5x speedup over CPU-only implementation
8. THE ZK_Prover SHALL support batch proof generation for multiple ballots in parallel

### Requirement 21: Comprehensive Hardware Benchmarking Suite

**User Story:** As a developer, I want exhaustive benchmarks of every hardware acceleration path, so that I can understand which hardware units provide real speedups for which operations.

#### Acceptance Criteria

1. THE Benchmark_Suite SHALL test EVERY hardware unit (SME, AMX, Metal GPU, Neural Engine, NEON, Ray Tracing, Texture Units) for EVERY applicable FHE operation
2. THE Benchmark_Suite SHALL measure latency, throughput, and power efficiency for each hardware path
3. THE Benchmark_Suite SHALL test matrix-reformulated operations vs scalar implementations
4. THE Benchmark_Suite SHALL test Neural Engine modular reduction vs CPU/GPU implementations
5. THE Benchmark_Suite SHALL test ray tracing hardware for tree traversal vs CPU implementations
6. THE Benchmark_Suite SHALL test texture sampling for polynomial evaluation vs direct computation
7. THE Benchmark_Suite SHALL produce detailed comparison reports with statistical significance
8. THE Benchmark_Suite SHALL identify the optimal hardware path for each operation at each data size
9. THE Benchmark_Suite SHALL measure memory bandwidth utilization for each hardware path
10. THE Benchmark_Suite SHALL test combinations of hardware units for pipelined operations
11. THE Benchmark_Suite SHALL benchmark speculative execution strategies
12. THE Benchmark_Suite SHALL be runnable as a standalone tool for hardware characterization

### Requirement 22: Matrix-Centric Algorithm Design

**User Story:** As a performance engineer, I want FHE algorithms reformulated as matrix operations wherever possible, so that I can maximize utilization of M4 Max's powerful matrix units (SME, AMX).

#### Acceptance Criteria

1. THE FHE_Engine SHALL express NTT as a sequence of sparse matrix multiplications (butterfly matrices)
2. THE FHE_Engine SHALL express polynomial multiplication as Toeplitz/circulant matrix-vector products
3. THE FHE_Engine SHALL express key switching as batched matrix-matrix multiplications
4. THE FHE_Engine SHALL express gadget decomposition as matrix operations
5. THE FHE_Engine SHALL express TFHE blind rotate as matrix operations where beneficial
6. THE FHE_Engine SHALL support configurable matrix tile sizes optimized for SME register file
7. THE FHE_Engine SHALL implement matrix-based NTT that processes multiple polynomials simultaneously
8. THE FHE_Engine SHALL benchmark matrix formulations against traditional scalar implementations
9. WHEN matrix formulation provides >10% speedup, THE FHE_Engine SHALL prefer matrix implementation
10. THE FHE_Engine SHALL document the mathematical reformulations used for each operation

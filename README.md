# @digitaldefiance/node-fhe-accelerate

**High-performance Fully Homomorphic Encryption (FHE) acceleration library optimized for Apple M4 Max hardware.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Node.js Version](https://img.shields.io/badge/node-%3E%3D18.0.0-brightgreen)](https://nodejs.org)
[![Platform](https://img.shields.io/badge/platform-macOS%20arm64-blue)](https://www.apple.com/mac/)

## What is This Library?

`@digitaldefiance/node-fhe-accelerate` enables **privacy-preserving computation** on encrypted data. With Fully Homomorphic Encryption (FHE), you can perform calculations on encrypted values without ever decrypting them—the results, when decrypted, are the same as if you had computed on the plaintext.

### Key Capabilities

- **🔐 Encrypt-Compute-Decrypt**: Perform arithmetic on encrypted data
- **🗳️ Secure Voting**: Build end-to-end verifiable elections on consumer hardware
- **🔍 Zero-Knowledge Proofs**: Prove statements without revealing secrets
- **⚡ Hardware Acceleration**: Exploit M4 Max's SME, GPU, and Neural Engine
- **📊 Real-time Processing**: 10,000+ encrypted ballots per second

### The Killer Use Case: Decentralized Voting

This library enables a novel approach to secure electronic voting that doesn't require datacenter infrastructure:

```
┌─────────────────┐     Encrypted      ┌─────────────────┐
│  Voter Device   │────────────────────►│  Mac Studio     │
│  (Any browser)  │     Ballots        │  (M4 Max)       │
└─────────────────┘                    └─────────────────┘
                                              │
                                              │ Homomorphic
                                              │ Tallying
                                              ▼
                                       ┌─────────────────┐
                                       │  Encrypted      │
                                       │  Results        │
                                       └─────────────────┘
                                              │
                                              │ Threshold
                                              │ Decryption
                                              ▼
                                       ┌─────────────────┐
                                       │  Final Tally    │
                                       │  + ZK Proofs    │
                                       └─────────────────┘
```

**No one sees individual votes**—not even the server. Results are only revealed through threshold decryption requiring multiple election officials.

## Table of Contents

- [Features](#features)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Voting System](#voting-system)
- [Zero-Knowledge Proofs](#zero-knowledge-proofs)
- [Hardware Acceleration](#hardware-acceleration)
- [Performance](#performance)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Features

### Fully Homomorphic Encryption

| Feature | Description |
|---------|-------------|
| **TFHE Scheme** | Fast bootstrapping for unlimited computation depth |
| **Homomorphic Addition** | Add encrypted values in < 1ms |
| **Homomorphic Multiplication** | Multiply encrypted values in < 50ms |
| **Bootstrapping** | Refresh ciphertexts in < 20ms on M4 Max |
| **SIMD Packing** | Encode multiple values in single ciphertext |

### Zero-Knowledge Proofs

| Proof System | Use Case | Generation | Verification | Size |
|--------------|----------|------------|--------------|------|
| **Bulletproofs** | Ballot validity (range proofs) | < 50ms | < 5ms | ~700 bytes |
| **Groth16** | Voter eligibility (Merkle membership) | < 100ms | < 1ms | ~200 bytes |
| **PLONK** | Tally correctness | < 150ms | < 5ms | ~400 bytes |

### Hardware Acceleration

Exhaustively exploits every M4 Max hardware feature:

| Hardware | Use Case | Speedup |
|----------|----------|---------|
| **SME** | NTT butterflies, matrix operations | 2x |
| **Metal GPU** | Batch operations (40 cores) | 1.5x for large batches |
| **Neural Engine** | Parallel hash computation (38 TOPS) | 3-4x for hash trees |
| **NEON SIMD** | Coefficient-level parallelism | 2x |
| **Unified Memory** | Zero-copy CPU/GPU sharing | ~400 GB/s |

## Hardware Requirements

### Minimum

- **Platform**: macOS (Apple Silicon only)
- **Processor**: Apple M1 or later
- **Memory**: 16 GB Unified Memory
- **Node.js**: 18.0.0 or later

### Recommended (for production voting)

- **Platform**: Mac Studio
- **Processor**: Apple M4 Max
- **Memory**: 64+ GB Unified Memory
- **Storage**: 1 TB NVMe SSD
- **Network**: 10 Gbps Ethernet

## Installation

```bash
# Using yarn (recommended)
yarn add @digitaldefiance/node-fhe-accelerate

# Using npm
npm install @digitaldefiance/node-fhe-accelerate
```

### Building from Source

```bash
# Clone repository
git clone https://github.com/Digital-Defiance/node-fhe-accelerate.git
cd node-fhe-accelerate

# Install dependencies
yarn install

# Build (TypeScript + Native + Shaders)
yarn build

# Run tests
yarn test
```

See [BUILDING.md](./BUILDING.md) for detailed build instructions.

## Quick Start

### Basic FHE Operations

```typescript
import { createEngine } from '@digitaldefiance/node-fhe-accelerate';

// Create FHE engine with 128-bit security
const engine = await createEngine('tfhe-128-fast');

// Generate keys
const secretKey = await engine.generateSecretKey();
const publicKey = await engine.generatePublicKey(secretKey);

// Encrypt two numbers
const a = await engine.encrypt(42n, publicKey);
const b = await engine.encrypt(17n, publicKey);

// Compute on encrypted data
const sum = await engine.add(a, b);           // 42 + 17 = 59
const product = await engine.multiply(a, b);  // 42 * 17 = 714

// Decrypt results
const sumResult = await engine.decrypt(sum, secretKey);      // 59n
const productResult = await engine.decrypt(product, secretKey); // 714n
```

### Simple Voting Example

```typescript
import { VotingSystem } from '@digitaldefiance/node-fhe-accelerate';

// Create election
const election = await VotingSystem.create({
  electionId: 'board-election-2024',
  name: 'Board of Directors Election',
  candidates: ['Alice', 'Bob', 'Charlie'],
  thresholdConfig: { threshold: 3, totalOfficials: 5 },
});

// Register voters
await election.registerVoter('voter-1', voterPublicKey1);
await election.registerVoter('voter-2', voterPublicKey2);

// Start election
await election.startElection();

// Submit encrypted ballots with ZK proofs
const ballot = await election.submitBallot('voter-1', 0); // Vote for Alice
console.log('Validity proof:', ballot.validityProof);
console.log('Eligibility proof:', ballot.eligibilityProof);

// End and tally
await election.endElection();
const result = await election.decryptTally(officials);

console.log('Results:', result.decryptedTally.counts);
// Output: [1n, 0n, 0n] - Alice: 1, Bob: 0, Charlie: 0
```

## Core Concepts

### Fully Homomorphic Encryption (FHE)

FHE allows computation on encrypted data:

```
Encrypt(a) ⊕ Encrypt(b) = Encrypt(a + b)
Encrypt(a) ⊗ Encrypt(b) = Encrypt(a × b)
```

The key insight: **you can compute without decrypting**. This enables:
- Private database queries
- Secure machine learning
- Confidential voting
- Privacy-preserving analytics

### Noise and Bootstrapping

FHE ciphertexts accumulate "noise" with each operation. When noise exceeds a threshold, decryption fails. **Bootstrapping** refreshes ciphertexts by homomorphically evaluating the decryption circuit:

```typescript
// Check noise budget
const budget = await engine.getNoiseBudget(ciphertext, secretKey);
console.log(`Remaining noise budget: ${budget} bits`);

// Refresh if needed
if (budget < 10) {
  const refreshed = await engine.bootstrap(ciphertext, bootstrapKey);
}
```

### Threshold Cryptography

No single party can decrypt results. M-of-N officials must cooperate:

```typescript
// 3-of-5 threshold decryption
const partials = await Promise.all([
  official1.partialDecrypt(encryptedTally),
  official2.partialDecrypt(encryptedTally),
  official3.partialDecrypt(encryptedTally),
]);

const finalResult = await combinePartialDecryptions(partials);
```

### Zero-Knowledge Proofs

Prove statements without revealing information:

```typescript
// Prove vote is valid (0, 1, or 2) without revealing which
const validityProof = await zkManager.proveBallotValidity(vote, 3);

// Anyone can verify
const isValid = await zkManager.verifyBallotValidity(validityProof);
// Returns true if vote ∈ {0, 1, 2}, reveals nothing about actual vote
```


## API Reference

### FHE Engine

The core interface for FHE operations.

#### Creating an Engine

```typescript
import { createEngine, type ParameterPreset } from '@digitaldefiance/node-fhe-accelerate';

// Using a preset
const engine = await createEngine('tfhe-128-fast');

// Using custom parameters
const engine = await createEngine({
  polyDegree: 16384,
  moduli: [0xFFFFFFFF00000001n, 0xFFFFFFFE00000001n],
  securityLevel: 128,
});
```

#### Parameter Presets

| Preset | Security | Speed | Use Case |
|--------|----------|-------|----------|
| `tfhe-128-fast` | 128-bit | Fastest | Development, low-latency apps |
| `tfhe-128-balanced` | 128-bit | Balanced | General purpose |
| `tfhe-256-secure` | 256-bit | Slower | High-security applications |
| `bfv-128-simd` | 128-bit | Fast | Batch integer operations |
| `ckks-128-ml` | 128-bit | Fast | Machine learning on floats |

#### Key Generation

```typescript
interface FHEEngine {
  // Generate secret key (keep private!)
  generateSecretKey(): Promise<SecretKey>;
  
  // Generate public key for encryption
  generatePublicKey(sk: SecretKey): Promise<PublicKey>;
  
  // Generate evaluation key for relinearization
  generateEvalKey(sk: SecretKey, decompBase?: number): Promise<EvaluationKey>;
  
  // Generate bootstrapping key for noise refresh
  generateBootstrapKey(sk: SecretKey): Promise<BootstrapKey>;
}
```

**Example:**

```typescript
const sk = await engine.generateSecretKey();
const pk = await engine.generatePublicKey(sk);
const ek = await engine.generateEvalKey(sk);
const bk = await engine.generateBootstrapKey(sk); // Takes ~30s
```

#### Encryption & Decryption

```typescript
interface FHEEngine {
  // Encrypt a value
  encrypt(plaintext: bigint, pk: PublicKey): Promise<Ciphertext>;
  
  // Decrypt a ciphertext
  decrypt(ciphertext: Ciphertext, sk: SecretKey): Promise<bigint>;
  
  // Check remaining noise budget
  getNoiseBudget(ct: Ciphertext, sk: SecretKey): Promise<number>;
}
```

**Example:**

```typescript
const encrypted = await engine.encrypt(42n, publicKey);
const decrypted = await engine.decrypt(encrypted, secretKey);
console.log(decrypted); // 42n

const budget = await engine.getNoiseBudget(encrypted, secretKey);
console.log(`Noise budget: ${budget} bits`);
```

#### Homomorphic Operations

```typescript
interface FHEEngine {
  // Ciphertext + Ciphertext
  add(ct1: Ciphertext, ct2: Ciphertext): Promise<Ciphertext>;
  
  // Ciphertext + Plaintext
  addPlain(ct: Ciphertext, pt: bigint): Promise<Ciphertext>;
  
  // Ciphertext × Ciphertext
  multiply(ct1: Ciphertext, ct2: Ciphertext): Promise<Ciphertext>;
  
  // Ciphertext × Plaintext
  multiplyPlain(ct: Ciphertext, pt: bigint): Promise<Ciphertext>;
  
  // Reduce ciphertext size after multiplication
  relinearize(ct: Ciphertext, ek: EvaluationKey): Promise<Ciphertext>;
  
  // Refresh noise budget
  bootstrap(ct: Ciphertext, bk: BootstrapKey): Promise<Ciphertext>;
}
```

**Example:**

```typescript
const a = await engine.encrypt(10n, pk);
const b = await engine.encrypt(20n, pk);

// Addition
const sum = await engine.add(a, b);
console.log(await engine.decrypt(sum, sk)); // 30n

// Multiplication (requires relinearization)
const product = await engine.multiply(a, b);
const relinearized = await engine.relinearize(product, ek);
console.log(await engine.decrypt(relinearized, sk)); // 200n

// Scalar operations
const doubled = await engine.multiplyPlain(a, 2n);
console.log(await engine.decrypt(doubled, sk)); // 20n
```

### FHE Context

High-level context for managing FHE operations.

```typescript
import { FHEContext, createVotingContext } from '@digitaldefiance/node-fhe-accelerate';

// Create context for voting
const ctx = await createVotingContext({
  generateEvalKey: true,
  generateBootstrapKey: false,
  thresholdConfig: {
    threshold: 3,
    totalShares: 5,
  },
});

// Encrypt/decrypt through context
const encrypted = await ctx.encrypt(42n);
const decrypted = await ctx.decrypt(encrypted);

// Get keys
const pk = ctx.getPublicKey();
const sk = ctx.getSecretKey();

// Cleanup
ctx.dispose();
```

### Streaming API

Process large ciphertexts in chunks.

```typescript
import { CiphertextReadStream, CiphertextWriteStream } from '@digitaldefiance/node-fhe-accelerate';

// Create streams
const readStream = new CiphertextReadStream(largeCiphertext, { chunkSize: 1024 });
const writeStream = new CiphertextWriteStream();

// Process with Node.js streams
readStream
  .pipe(transformStream)
  .pipe(writeStream);

// Or use async iteration
for await (const chunk of readStream) {
  await processChunk(chunk);
}
```

## Voting System

### Complete Voting Workflow

```typescript
import { VotingSystem } from '@digitaldefiance/node-fhe-accelerate';

// 1. Create election
const election = await VotingSystem.create({
  electionId: 'election-2024',
  name: 'Presidential Election 2024',
  candidates: ['Alice', 'Bob', 'Charlie', 'David'],
  thresholdConfig: { threshold: 3, totalOfficials: 5 },
  expectedVoters: 100000,
});

// 2. Register voters (before election starts)
for (const voter of voterList) {
  await election.registerVoter(voter.id, voter.publicKeyHash);
}

// 3. Start election
await election.startElection();

// 4. Accept ballots (during election)
election.subscribeTallyUpdates((update) => {
  console.log(`Processed: ${update.ballotsProcessed} ballots`);
});

const ballot = await election.submitBallot('voter-123', 0, (progress) => {
  console.log(`${progress.stage}: ${progress.progressPercent}%`);
});

// 5. Verify ballot proofs
const verification = await election.verifyBallot(ballot);
console.log('Valid:', verification.validityValid && verification.eligibilityValid);

// 6. End election
const encryptedTally = await election.endElection();

// 7. Threshold decryption
const officials = [
  { id: 'official-1', shareId: 1, publicKeyHash: '...' },
  { id: 'official-2', shareId: 2, publicKeyHash: '...' },
  { id: 'official-3', shareId: 3, publicKeyHash: '...' },
];

const result = await election.decryptTally(officials);

// 8. Display results
console.log('Final Results:');
result.decryptedTally.counts.forEach((count, i) => {
  console.log(`  ${election.getCandidateName(i)}: ${count} votes`);
});

// 9. Export audit trail
const auditJson = await election.exportAuditTrail('json');
await fs.writeFile('audit-trail.json', auditJson);

// 10. Cleanup
election.dispose();
```

### Ballot Types

```typescript
interface EncryptedBallot {
  ballotId: string;
  encryptedChoices: Ciphertext[];
  timestamp: number;
  validityProof?: Uint8Array;
  voterCommitment?: Uint8Array;
}

interface CompleteBallot {
  encryptedBallot: EncryptedBallot;
  validityProof: BallotValidityProof;
  eligibilityProof: EligibilityProof;
  timestamp: Date;
}
```

### Fraud Detection

```typescript
// Configure fraud detection
await election.configureFraudDetection({
  duplicateVoteDetection: true,
  statisticalAnomalyDetection: true,
  timingAnomalyDetection: true,
  thresholds: {
    duplicateConfidence: 0.99,
    anomalyZScore: 3.0,
  },
});

// Subscribe to alerts
election.subscribeFraudAlerts((alert) => {
  console.error('FRAUD ALERT:', alert.type);
  console.error('Confidence:', alert.confidence);
  console.error('Details:', alert.description);
});
```

### Audit Trail

```typescript
import { AuditTrailManager, createSystemActor } from '@digitaldefiance/node-fhe-accelerate';

const audit = new AuditTrailManager('election-2024');

// Log operations
await audit.logBallotSubmitted(ballot, createVoterActor('voter-123'));
await audit.logTallyComputed(tally, ballotCount, computationTimeMs, createSystemActor());

// Verify integrity
const verification = audit.verifyIntegrity();
console.log('Valid:', verification.valid);
console.log('Entries verified:', verification.entriesVerified);

// Export
const json = await audit.exportJSON();
const csv = await audit.exportCSV();
```

## Zero-Knowledge Proofs

### Bulletproofs (Ballot Validity)

Prove a vote is within valid range without revealing the vote.

```typescript
import { BulletproofsProver } from '@digitaldefiance/node-fhe-accelerate';

const prover = new BulletproofsProver({ useGpu: true });

// Prove vote is in range [0, numCandidates)
const proof = await prover.proveBallotValidity(vote, numCandidates);

// Verify
const result = await prover.verifyBallotValidity(proof);
console.log('Valid:', result.valid);
console.log('Verification time:', result.verificationTimeMs, 'ms');

// Batch verify
const results = await prover.batchVerify(proofs, (progress) => {
  console.log(`Verified ${progress.current}/${progress.total}`);
});
```

### Groth16 (Voter Eligibility)

Prove voter is in the registry without revealing identity.

```typescript
import { Groth16Setup, Groth16Prover, Groth16Verifier } from '@digitaldefiance/node-fhe-accelerate';

// Setup (done once per election)
const { provingKey, verifyingKey } = await Groth16Setup.setup(
  1000,  // numVariables
  2,     // numPublicInputs
  500    // numConstraints
);

// Prove eligibility
const prover = new Groth16Prover(provingKey);
const proof = await prover.proveEligibility(
  voterLeaf,      // Voter's Merkle leaf
  merklePath,     // Path to root
  pathIndices,    // Left/right indices
  merkleRoot      // Public Merkle root
);

// Verify
const verifier = new Groth16Verifier(verifyingKey);
const result = await verifier.verify(proof);
console.log('Eligible:', result.valid);
```

### PLONK (Tally Correctness)

Prove the tally was computed correctly.

```typescript
import { PlonkSetup, PlonkProver, PlonkVerifier } from '@digitaldefiance/node-fhe-accelerate';

// Setup
const { provingKey, verifyingKey } = await PlonkSetup.setup(
  4096,           // domainSize
  numCandidates   // numPublicInputs
);

// Prove tally correctness
const prover = new PlonkProver(provingKey);
const proof = await prover.proveTallyCorrectness(
  encryptedBallots,
  initialTally,
  finalTally,
  numVotes
);

// Verify
const verifier = new PlonkVerifier(verifyingKey);
const result = await verifier.verify(proof);
console.log('Tally correct:', result.valid);
```

### Unified ZK Manager

```typescript
import { ZKProofManager } from '@digitaldefiance/node-fhe-accelerate';

const zkManager = new ZKProofManager();

// Initialize proof systems
zkManager.initGroth16(groth16ProvingKey, groth16VerifyingKey);
zkManager.initPlonk(plonkProvingKey, plonkVerifyingKey);

// Generate proofs
const validityProof = await zkManager.proveBallotValidity(vote, numCandidates);
const eligibilityProof = await zkManager.proveEligibility(leaf, path, indices, root);
const tallyProof = await zkManager.proveTallyCorrectness(ballots, initial, final, count);

// Verify proofs
const validityResult = await zkManager.verifyBallotValidity(validityProof);
const eligibilityResult = await zkManager.verifyEligibility(eligibilityProof);
const tallyResult = await zkManager.verifyTallyCorrectness(tallyProof);
```

### Public Verification

Anyone can verify election integrity:

```typescript
import { PublicVerifier, generateHTMLReport } from '@digitaldefiance/node-fhe-accelerate';

const verifier = new PublicVerifier();

// Load verification package
const pkg = JSON.parse(fs.readFileSync('verification-package.json', 'utf-8'));

// Run full verification
const report = await verifier.verifyAll(pkg, (progress) => {
  console.log(`${progress.stage}: ${progress.percent}%`);
});

console.log('Election verified:', report.overallValid);

// Generate HTML report
const html = generateHTMLReport(report);
fs.writeFileSync('verification-report.html', html);
```

## Hardware Acceleration

### Detecting Hardware

```typescript
import { detectHardware } from '@digitaldefiance/node-fhe-accelerate';

const caps = await detectHardware();
console.log('Hardware Capabilities:');
console.log('  SME:', caps.sme);
console.log('  SME2:', caps.sme2);
console.log('  AMX:', caps.amx);
console.log('  Metal GPU:', caps.metalGpu);
console.log('  GPU Cores:', caps.metalGpuCores);
console.log('  Neural Engine:', caps.neuralEngine);
console.log('  Neural Engine TOPS:', caps.neuralEngineTops);
console.log('  NEON:', caps.neon);
console.log('  Unified Memory:', caps.unifiedMemoryGB, 'GB');
```

### Backend Selection

```typescript
import { HardwareDispatcher, HardwareBackend } from '@digitaldefiance/node-fhe-accelerate';

const dispatcher = new HardwareDispatcher();

// Automatic selection (recommended)
const backend = dispatcher.selectBackend('ntt', dataSize);

// Manual override for benchmarking
dispatcher.setPreferredBackend(HardwareBackend.Metal);

// Execute operation
await dispatcher.executeNtt(data, degree, inverse);
```

### Benchmark Results

See [HARDWARE_ACCELERATION_REPORT.md](./cpp/HARDWARE_ACCELERATION_REPORT.md) for detailed benchmarks.

| Operation | Best Backend | Speedup |
|-----------|--------------|---------|
| NTT (degree 16384) | Montgomery NTT | 2.12x |
| Modular Multiplication | Barrett Unrolled (4x) | 1.96x |
| Batch Operations (>262K) | Metal GPU | 1.55x |
| Hash Trees | vDSP | 3.95x |

## Performance

### Targets

| Operation | Target | Achieved |
|-----------|--------|----------|
| Encryption | < 10ms | ✓ |
| Homomorphic Addition | < 1ms | ✓ |
| Homomorphic Multiplication | < 50ms | ✓ |
| Bootstrapping | < 20ms | ✓ (M4 Max) |
| Ballot Processing | > 10,000/sec | ✓ |
| ZK Proof Generation | < 200ms | ✓ |
| ZK Proof Verification | < 20ms | ✓ |

### Voting System Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Ballot Ingestion | > 10,000/sec | ✓ |
| Tally (100K ballots) | < 5 seconds | ✓ |
| Ballot Size | < 10 KB | ✓ |
| Memory per Ballot | < 1 MB | ✓ |

## Documentation

### Deployment & Operations

| Document | Description |
|----------|-------------|
| [Deployment Guide](./docs/DEPLOYMENT_GUIDE.md) | Production deployment on Mac Studio/M4 Max |
| [Election Official Setup](./docs/ELECTION_OFFICIAL_SETUP.md) | Key ceremony and election management |
| [Monitoring & Alerting](./docs/MONITORING_ALERTING.md) | Prometheus, Grafana, alerting rules |
| [Disaster Recovery](./docs/DISASTER_RECOVERY.md) | Backup, failover, recovery procedures |

### Development

| Document | Description |
|----------|-------------|
| [Building](./BUILDING.md) | Build from source instructions |
| [Native Addon](./NATIVE_ADDON.md) | C++/Rust native addon architecture |
| [Setup](./SETUP.md) | Quick development environment setup |

### Security & Verification

| Document | Description |
|----------|-------------|
| [Security Best Practices](./docs/SECURITY_BEST_PRACTICES.md) | Security guidelines and hardening |
| [ZK Proof Verification](./docs/ZK_PROOF_VERIFICATION.md) | Public verification tools and procedures |
| [Voter Client Examples](./docs/VOTER_CLIENT_EXAMPLES.md) | Web, mobile, CLI client implementations |

## Error Handling

```typescript
import { FHEError, FHEErrorCode } from '@digitaldefiance/node-fhe-accelerate';

try {
  const result = await engine.decrypt(ciphertext, secretKey);
} catch (error) {
  if (error instanceof FHEError) {
    switch (error.code) {
      case FHEErrorCode.NOISE_BUDGET_EXHAUSTED:
        console.error('Ciphertext noise too high, bootstrap required');
        break;
      case FHEErrorCode.KEY_MISMATCH:
        console.error('Wrong key used for decryption');
        break;
      case FHEErrorCode.HARDWARE_UNAVAILABLE:
        console.error('Required hardware not available');
        break;
      default:
        console.error('FHE error:', error.message);
    }
  }
}
```

## TypeScript Support

Full TypeScript support with comprehensive type definitions:

```typescript
import type {
  // Core types
  SecretKey,
  PublicKey,
  EvaluationKey,
  BootstrapKey,
  Ciphertext,
  Plaintext,
  
  // Configuration
  ParameterPreset,
  CustomParameters,
  SecurityLevel,
  
  // Voting
  EncryptedBallot,
  ElectionConfig,
  FraudAlert,
  AuditEntry,
  
  // ZK Proofs
  BallotValidityProof,
  EligibilityProof,
  TallyCorrectnessProof,
  ZKVerificationResult,
  
  // Progress
  ProgressCallback,
} from '@digitaldefiance/node-fhe-accelerate';
```

## Contributing

Contributions are welcome! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Ensure all tests pass: `yarn test`
5. Submit a pull request

### Development Setup

```bash
# Clone and install
git clone https://github.com/digitaldefiance/node-fhe-accelerate.git
cd node-fhe-accelerate
yarn install

# Build
yarn build

# Test
yarn test

# Lint
yarn lint
```

## License

MIT License - see [LICENSE](./LICENSE) for details.

## Acknowledgments

This library is part of the Digital Defiance project, enabling privacy-preserving computation on consumer hardware.

### References

- [TFHE: Fast Fully Homomorphic Encryption](https://eprint.iacr.org/2018/421)
- [Bulletproofs](https://eprint.iacr.org/2017/1066)
- [Groth16](https://eprint.iacr.org/2016/260)
- [PLONK](https://eprint.iacr.org/2019/953)

## Support

- **Documentation**: [docs/](./docs/)
- **Issues**: [GitHub Issues](https://github.com/Digital-Defiance/node-fhe-accelerate/issues)
- **Security**: security@digitaldefiance.org

/**
 * @file api/zk-types.ts
 * @brief TypeScript type definitions for Zero-Knowledge proof operations
 *
 * This module provides type definitions for ZK proof systems including
 * Bulletproofs, Groth16, and PLONK.
 *
 * Requirements: 19, 20
 */

// ============================================================================
// Common ZK Types
// ============================================================================

/**
 * ZK proof system types
 */
export type ZKProofSystem = 'bulletproofs' | 'groth16' | 'plonk';

/**
 * Field element (256-bit)
 */
export interface FieldElement {
  readonly limbs: bigint[];
}

/**
 * Elliptic curve point (affine coordinates)
 */
export interface CurvePoint {
  readonly x: FieldElement;
  readonly y: FieldElement;
  readonly isInfinity: boolean;
}

// ============================================================================
// Bulletproofs Types
// ============================================================================

/**
 * Bulletproofs configuration
 */
export interface BulletproofsConfig {
  maxRangeBits: number;
  maxAggregation: number;
  useGpu: boolean;
}

/**
 * Bulletproofs generators
 */
export interface BulletproofsGenerators {
  G: CurvePoint;
  H: CurvePoint;
  gVec: CurvePoint[];
  hVec: CurvePoint[];
  U: CurvePoint;
}

/**
 * Pedersen commitment
 */
export interface PedersenCommitment {
  point: CurvePoint;
}

/**
 * Range proof
 */
export interface RangeProof {
  A: CurvePoint;
  S: CurvePoint;
  T1: CurvePoint;
  T2: CurvePoint;
  tauX: FieldElement;
  mu: FieldElement;
  tHat: FieldElement;
  innerProof: InnerProductProof;
}

/**
 * Inner product proof
 */
export interface InnerProductProof {
  L: CurvePoint[];
  R: CurvePoint[];
  a: FieldElement;
  b: FieldElement;
}

/**
 * Ballot validity proof (Bulletproofs)
 */
export interface BallotValidityProof {
  commitment: PedersenCommitment;
  rangeProof: RangeProof;
  numCandidates: number;
}

/**
 * Serialized Bulletproof
 */
export interface SerializedBulletproof {
  data: Uint8Array;
  proofType: 'range' | 'aggregated' | 'ballot_validity';
  sizeBytes: number;
}

// ============================================================================
// Groth16 Types
// ============================================================================

/**
 * R1CS constraint
 */
export interface R1CSConstraint {
  a: Map<number, FieldElement>;
  b: Map<number, FieldElement>;
  c: Map<number, FieldElement>;
}

/**
 * R1CS constraint system
 */
export interface R1CSSystem {
  numVariables: number;
  numPublicInputs: number;
  numConstraints: number;
  constraints: R1CSConstraint[];
}

/**
 * Groth16 proving key
 */
export interface Groth16ProvingKey {
  handle: bigint;
  numVariables: number;
  numPublicInputs: number;
  numConstraints: number;
  sizeBytes: number;
}

/**
 * Groth16 verification key
 */
export interface Groth16VerificationKey {
  alphaG1: CurvePoint;
  betaG2: CurvePoint;
  gammaG2: CurvePoint;
  deltaG2: CurvePoint;
  ic: CurvePoint[];
  sizeBytes: number;
}

/**
 * Groth16 proof (~200 bytes)
 */
export interface Groth16Proof {
  a: CurvePoint;
  b: CurvePoint;
  c: CurvePoint;
}

/**
 * Serialized Groth16 proof
 */
export interface SerializedGroth16Proof {
  data: Uint8Array;
  sizeBytes: number;
}

/**
 * Eligibility proof (Groth16)
 */
export interface EligibilityProof {
  proof: Groth16Proof;
  merkleRoot: FieldElement;
  nullifier: FieldElement;
}

// ============================================================================
// PLONK Types
// ============================================================================

/**
 * PLONK gate types
 */
export enum PLONKGateType {
  ARITHMETIC = 'ARITHMETIC',
  MULTIPLICATION = 'MULTIPLICATION',
  ADDITION = 'ADDITION',
  CONSTANT = 'CONSTANT',
  BOOLEAN = 'BOOLEAN',
  RANGE = 'RANGE',
  POSEIDON = 'POSEIDON',
  FHE_ADD = 'FHE_ADD',
  FHE_MUL = 'FHE_MUL',
  TALLY_SUM = 'TALLY_SUM',
  CUSTOM = 'CUSTOM',
}

/**
 * PLONK universal setup
 */
export interface PLONKSetup {
  handle: bigint;
  maxDegree: number;
  sizeBytes: number;
}

/**
 * PLONK proving key
 */
export interface PLONKProvingKey {
  handle: bigint;
  domainSize: number;
  numPublicInputs: number;
  sizeBytes: number;
}

/**
 * PLONK verification key
 */
export interface PLONKVerificationKey {
  qLCommit: CurvePoint;
  qRCommit: CurvePoint;
  qOCommit: CurvePoint;
  qMCommit: CurvePoint;
  qCCommit: CurvePoint;
  sigma1Commit: CurvePoint;
  sigma2Commit: CurvePoint;
  sigma3Commit: CurvePoint;
  domainSize: number;
  numPublicInputs: number;
  sizeBytes: number;
}

/**
 * PLONK proof (~400 bytes)
 */
export interface PLONKProof {
  aCommit: CurvePoint;
  bCommit: CurvePoint;
  cCommit: CurvePoint;
  zCommit: CurvePoint;
  tLoCommit: CurvePoint;
  tMidCommit: CurvePoint;
  tHiCommit: CurvePoint;
  aEval: FieldElement;
  bEval: FieldElement;
  cEval: FieldElement;
  sigma1Eval: FieldElement;
  sigma2Eval: FieldElement;
  zOmegaEval: FieldElement;
  wZeta: CurvePoint;
  wZetaOmega: CurvePoint;
}

/**
 * Serialized PLONK proof
 */
export interface SerializedPLONKProof {
  data: Uint8Array;
  sizeBytes: number;
}

/**
 * Tally correctness proof (PLONK)
 */
export interface TallyCorrectnessProof {
  proof: PLONKProof;
  initialTally: FieldElement;
  finalTally: FieldElement;
  numVotes: number;
}

// ============================================================================
// Unified ZK Proof Types
// ============================================================================

/**
 * Generic ZK proof wrapper
 */
export interface ZKProof {
  system: ZKProofSystem;
  proofData: Uint8Array;
  publicInputs: FieldElement[];
  sizeBytes: number;
}

/**
 * ZK proof generation options
 */
export interface ZKProofOptions {
  useGpu?: boolean;
  generateWitness?: boolean;
  verifyAfterGeneration?: boolean;
}

/**
 * ZK proof verification result
 */
export interface ZKVerificationResult {
  valid: boolean;
  verificationTimeMs: number;
  errorMessage?: string;
}

/**
 * Batch ZK proof result
 */
export interface BatchZKProofResult {
  proofs: ZKProof[];
  generationTimeMs: number;
  throughputProofsPerSecond: number;
}

// ============================================================================
// Voting ZK Proof Types
// ============================================================================

/**
 * Complete ballot proof (combines all ZK proofs for a ballot)
 */
export interface CompleteBallotProof {
  validityProof: BallotValidityProof;
  eligibilityProof: EligibilityProof;
  uniquenessProof?: ZKProof;
  totalSizeBytes: number;
}

/**
 * Decryption correctness proof
 */
export interface DecryptionCorrectnessProof {
  shareId: number;
  proof: ZKProof;
  commitment: Uint8Array;
}

/**
 * Complete election proof set
 */
export interface ElectionProofSet {
  tallyCorrectnessProof: TallyCorrectnessProof;
  decryptionProofs: DecryptionCorrectnessProof[];
  auditProof?: ZKProof;
}

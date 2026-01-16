/**
 * @file api/voting-types.ts
 * @brief TypeScript type definitions for voting-specific operations
 *
 * This module provides type definitions for voting system operations,
 * including ballot management, fraud detection, and audit trails.
 *
 * Requirements: 15, 17, 18
 */

import type { Ciphertext, PublicKey, ProgressCallback } from './types';

// ============================================================================
// Ballot Types
// ============================================================================

/**
 * Encrypted ballot structure
 */
export interface EncryptedBallot {
  readonly ballotId: string;
  readonly encryptedChoices: Ciphertext[];
  readonly timestamp: number;
  readonly validityProof?: Uint8Array;
  readonly voterCommitment?: Uint8Array;
}

/**
 * Ballot submission options
 */
export interface BallotSubmissionOptions {
  generateValidityProof?: boolean;
  generateReceipt?: boolean;
  electionId?: string;
}

/**
 * Ballot submission result
 */
export interface BallotSubmissionResult {
  success: boolean;
  ballotId: string;
  receipt?: VoterReceipt;
  errorMessage?: string;
}

/**
 * Voter receipt for verification
 */
export interface VoterReceipt {
  receiptId: Uint8Array;
  voteCommitment: Uint8Array;
  timestamp: number;
  authoritySignature: Uint8Array;
  verificationData: Uint8Array;
}

// ============================================================================
// Tally Types
// ============================================================================

/**
 * Encrypted tally result
 */
export interface EncryptedTally {
  readonly electionId: string;
  readonly encryptedCounts: Ciphertext[];
  readonly totalBallots: number;
  readonly timestamp: number;
}

/**
 * Tally update event
 */
export interface TallyUpdate {
  electionId: string;
  encryptedTally: EncryptedTally;
  ballotsProcessed: number;
  timestamp: number;
}

/**
 * Decrypted tally result
 */
export interface DecryptedTally {
  electionId: string;
  counts: bigint[];
  totalBallots: number;
  decryptionProof?: Uint8Array;
  timestamp: number;
}

/**
 * Tally options
 */
export interface TallyOptions {
  useGpu?: boolean;
  generateProof?: boolean;
  progressCallback?: ProgressCallback;
}

// ============================================================================
// Fraud Detection Types
// ============================================================================

/**
 * Fraud alert types
 */
export enum FraudAlertType {
  DUPLICATE_VOTE = 'DUPLICATE_VOTE',
  STATISTICAL_ANOMALY = 'STATISTICAL_ANOMALY',
  TIMING_ANOMALY = 'TIMING_ANOMALY',
  THRESHOLD_EXCEEDED = 'THRESHOLD_EXCEEDED',
  PATTERN_ANOMALY = 'PATTERN_ANOMALY',
}

/**
 * Fraud alert structure
 */
export interface FraudAlert {
  type: FraudAlertType;
  description: string;
  timestamp: number;
  confidence: number;
  encryptedEvidence?: Ciphertext;
  ballotIndex?: number;
}

/**
 * Statistical model for expected voting patterns
 */
export interface StatisticalModel {
  expectedDistribution: number[];
  varianceThreshold: number;
  minSampleSize: number;
}

/**
 * Threshold configuration for alerts
 */
export interface FraudThreshold {
  name: string;
  value: number;
  alertOnExceed: boolean;
  alertOnBelow: boolean;
}

/**
 * Fraud detection result
 */
export interface FraudDetectionResult {
  alerts: FraudAlert[];
  ballotsAnalyzed: number;
  analysisTimeMs: number;
  privacyPreserved: boolean;
}

/**
 * Fraud detection options
 */
export interface FraudDetectionOptions {
  checkDuplicates?: boolean;
  checkAnomalies?: boolean;
  checkThresholds?: boolean;
  statisticalModel?: StatisticalModel;
  thresholds?: FraudThreshold[];
  progressCallback?: ProgressCallback;
}

// ============================================================================
// Audit Trail Types
// ============================================================================

/**
 * Audit log entry types
 */
export enum AuditEntryType {
  BALLOT_SUBMITTED = 'BALLOT_SUBMITTED',
  BALLOT_VERIFIED = 'BALLOT_VERIFIED',
  TALLY_UPDATED = 'TALLY_UPDATED',
  TALLY_DECRYPTED = 'TALLY_DECRYPTED',
  FRAUD_ALERT = 'FRAUD_ALERT',
  KEY_GENERATED = 'KEY_GENERATED',
  THRESHOLD_DECRYPTION = 'THRESHOLD_DECRYPTION',
}

/**
 * Audit log entry (simplified for voting types)
 * Note: For full audit trail functionality, use AuditEntry from audit-trail.ts
 */
export interface VotingAuditEntry {
  entryId: string;
  type: AuditEntryType;
  timestamp: number;
  electionId: string;
  data: Record<string, unknown>;
  hash: Uint8Array;
  previousHash: Uint8Array;
  signature?: Uint8Array;
}

/**
 * Audit trail
 */
export interface VotingAuditTrail {
  electionId: string;
  entries: VotingAuditEntry[];
  rootHash: Uint8Array;
  createdAt: number;
  lastUpdated: number;
}

/**
 * Audit export format
 */
export type AuditExportFormat = 'json' | 'csv' | 'binary';

/**
 * Audit export options
 */
export interface AuditExportOptions {
  format: AuditExportFormat;
  includeSignatures?: boolean;
  startTime?: number;
  endTime?: number;
  entryTypes?: AuditEntryType[];
}

// ============================================================================
// Election Types
// ============================================================================

/**
 * Election configuration
 */
export interface ElectionConfig {
  electionId: string;
  name: string;
  candidates: string[];
  startTime: number;
  endTime: number;
  thresholdConfig: {
    threshold: number;
    totalOfficials: number;
  };
  publicKey: PublicKey;
}

/**
 * Election status
 */
export enum ElectionStatus {
  NOT_STARTED = 'NOT_STARTED',
  OPEN = 'OPEN',
  CLOSED = 'CLOSED',
  TALLYING = 'TALLYING',
  DECRYPTING = 'DECRYPTING',
  COMPLETED = 'COMPLETED',
}

/**
 * Election state
 */
export interface ElectionState {
  config: ElectionConfig;
  status: ElectionStatus;
  currentTally: EncryptedTally;
  ballotsReceived: number;
  fraudAlerts: FraudAlert[];
  auditTrail: VotingAuditTrail;
}

// ============================================================================
// Real-time Streaming Types
// ============================================================================

/**
 * Tally stream subscriber
 */
export interface TallyStreamSubscriber {
  subscriberId: string;
  electionId: string;
  onUpdate: (update: TallyUpdate) => void;
  onError: (error: Error) => void;
  onComplete: () => void;
}

/**
 * Tally stream options
 */
export interface TallyStreamOptions {
  updateIntervalMs?: number;
  includeIntermediateTallies?: boolean;
  maxSubscribers?: number;
}

/**
 * Stream subscription handle
 */
export interface StreamSubscription {
  unsubscribe: () => void;
  isActive: () => boolean;
}

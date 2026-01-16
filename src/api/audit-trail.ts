/**
 * @file api/audit-trail.ts
 * @brief Cryptographic audit trail API for FHE voting operations
 *
 * Provides:
 * - Cryptographic audit log of all operations
 * - Support for post-election verification
 * - Export formats for election officials (JSON, CSV)
 * - Immutable operation history with timestamps
 *
 * Requirements: 17
 */

import type { PublicKey, SecretKey } from './types';
import type { EncryptedBallot, EncryptedTally } from './voting-types';
import { createHash } from 'crypto';

// Type aliases for clarity
export type ElectionId = string;
export type VoterId = string;
export type BallotId = string;

// ============================================================================
// Audit Entry Types
// ============================================================================

/**
 * Types of auditable operations
 */
export type AuditOperationType =
  | 'election_created'
  | 'election_started'
  | 'election_ended'
  | 'ballot_submitted'
  | 'ballot_verified'
  | 'ballot_rejected'
  | 'tally_computed'
  | 'tally_verified'
  | 'key_generated'
  | 'key_shared'
  | 'decryption_started'
  | 'partial_decryption'
  | 'decryption_completed'
  | 'proof_generated'
  | 'proof_verified'
  | 'fraud_detected'
  | 'audit_exported';

/**
 * Base audit entry structure
 */
export interface AuditEntry {
  /** Unique entry ID */
  id: string;
  /** Entry sequence number (monotonically increasing) */
  sequenceNumber: number;
  /** Operation type */
  operation: AuditOperationType;
  /** Election ID */
  electionId: ElectionId;
  /** Timestamp of operation */
  timestamp: Date;
  /** Actor who performed the operation */
  actor: AuditActor;
  /** Operation-specific data */
  data: AuditData;
  /** Hash of previous entry (chain integrity) */
  previousHash: string;
  /** Hash of this entry */
  hash: string;
  /** Digital signature (optional) */
  signature?: string | undefined;
}

/**
 * Actor who performed an auditable operation
 */
export interface AuditActor {
  type: 'voter' | 'official' | 'system' | 'verifier';
  id: string;
  publicKeyHash?: string | undefined;
}

/**
 * Union type for operation-specific audit data
 */
export type AuditData =
  | ElectionCreatedData
  | ElectionStartedData
  | ElectionEndedData
  | BallotSubmittedData
  | BallotVerifiedData
  | BallotRejectedData
  | TallyComputedData
  | TallyVerifiedData
  | KeyGeneratedData
  | KeySharedData
  | DecryptionStartedData
  | PartialDecryptionData
  | DecryptionCompletedData
  | ProofGeneratedData
  | ProofVerifiedData
  | FraudDetectedData
  | AuditExportedData;

export interface ElectionCreatedData {
  electionName: string;
  candidateCount: number;
  expectedVoters: number;
  securityLevel: number;
}

export interface ElectionStartedData {
  publicKeyHash: string;
  startTime: Date;
}

export interface ElectionEndedData {
  endTime: Date;
  totalBallots: number;
  tallyHash: string;
}

export interface BallotSubmittedData {
  ballotId: BallotId;
  voterId?: VoterId;
  ballotHash: string;
  proofHash?: string;
}

export interface BallotVerifiedData {
  ballotId: BallotId;
  verificationResult: boolean;
  proofType: string;
}

export interface BallotRejectedData {
  ballotId: BallotId;
  reason: string;
  details?: string;
}

export interface TallyComputedData {
  tallyHash: string;
  ballotCount: number;
  computationTimeMs: number;
}

export interface TallyVerifiedData {
  tallyHash: string;
  verificationResult: boolean;
  proofHash: string;
}

export interface KeyGeneratedData {
  keyType: 'public' | 'secret' | 'evaluation' | 'bootstrap' | 'threshold';
  keyHash: string;
  parameters: string;
}

export interface KeySharedData {
  shareId: number;
  totalShares: number;
  threshold: number;
  commitmentHash: string;
}

export interface DecryptionStartedData {
  tallyHash: string;
  requiredShares: number;
}

export interface PartialDecryptionData {
  shareId: number;
  partialHash: string;
  proofHash?: string;
}

export interface DecryptionCompletedData {
  resultHash: string;
  sharesUsed: number[];
  verificationProofHash: string;
}

export interface ProofGeneratedData {
  proofType: 'bulletproof' | 'groth16' | 'plonk';
  proofHash: string;
  statementHash: string;
  generationTimeMs: number;
}

export interface ProofVerifiedData {
  proofType: 'bulletproof' | 'groth16' | 'plonk';
  proofHash: string;
  verificationResult: boolean;
  verificationTimeMs: number;
}

export interface FraudDetectedData {
  fraudType: 'duplicate_vote' | 'invalid_proof' | 'anomaly' | 'tampering';
  severity: 'low' | 'medium' | 'high' | 'critical';
  details: string;
  evidenceHash: string;
}

export interface AuditExportedData {
  format: 'json' | 'csv';
  entryCount: number;
  exportHash: string;
}

// ============================================================================
// Audit Trail Manager
// ============================================================================

/**
 * Configuration for audit trail
 */
export interface AuditTrailConfig {
  /** Enable digital signatures on entries */
  enableSignatures?: boolean;
  /** Signing key for signatures */
  signingKey?: SecretKey;
  /** Maximum entries to keep in memory */
  maxInMemoryEntries?: number;
  /** Auto-persist to storage */
  autoPersist?: boolean;
  /** Persistence callback */
  onPersist?: (entries: AuditEntry[]) => Promise<void>;
}

/**
 * Query options for audit entries
 */
export interface AuditQueryOptions {
  /** Filter by operation types */
  operations?: AuditOperationType[];
  /** Filter by actor type */
  actorType?: AuditActor['type'];
  /** Filter by actor ID */
  actorId?: string;
  /** Start time (inclusive) */
  startTime?: Date;
  /** End time (inclusive) */
  endTime?: Date;
  /** Maximum entries to return */
  limit?: number;
  /** Offset for pagination */
  offset?: number;
}

/**
 * Audit trail verification result
 */
export interface AuditVerificationResult {
  valid: boolean;
  entriesVerified: number;
  firstInvalidEntry?: number;
  errorMessage?: string;
}

/**
 * Cryptographic audit trail manager
 *
 * Maintains an immutable, hash-chained log of all election operations.
 * Supports verification, export, and querying of audit entries.
 *
 * @example
 * ```typescript
 * const audit = new AuditTrailManager(electionId);
 *
 * // Log operations
 * await audit.logBallotSubmitted(ballot, voter);
 * await audit.logTallyComputed(tally, ballotCount);
 *
 * // Verify integrity
 * const result = audit.verifyIntegrity();
 *
 * // Export for officials
 * const json = audit.exportJSON();
 * const csv = audit.exportCSV();
 * ```
 */
export class AuditTrailManager {
  private electionId: ElectionId;
  private entries: AuditEntry[] = [];
  private sequenceCounter = 0;
  private config: AuditTrailConfig;
  private genesisHash: string;

  constructor(electionId: ElectionId, config: AuditTrailConfig = {}) {
    this.electionId = electionId;
    this.config = {
      maxInMemoryEntries: 100000,
      autoPersist: false,
      ...config,
    };
    this.genesisHash = this.computeGenesisHash();
  }

  /**
   * Get the election ID
   */
  getElectionId(): ElectionId {
    return this.electionId;
  }

  /**
   * Get total entry count
   */
  getEntryCount(): number {
    return this.entries.length;
  }

  /**
   * Get the latest entry
   */
  getLatestEntry(): AuditEntry | undefined {
    return this.entries[this.entries.length - 1];
  }

  // ========================================================================
  // Logging Methods
  // ========================================================================

  /**
   * Log election creation
   */
  async logElectionCreated(
    electionName: string,
    candidateCount: number,
    expectedVoters: number,
    securityLevel: number,
    actor: AuditActor
  ): Promise<AuditEntry> {
    return this.addEntry('election_created', actor, {
      electionName,
      candidateCount,
      expectedVoters,
      securityLevel,
    });
  }

  /**
   * Log election start
   */
  async logElectionStarted(publicKey: PublicKey, actor: AuditActor): Promise<AuditEntry> {
    return this.addEntry('election_started', actor, {
      publicKeyHash: this.hashPublicKey(publicKey),
      startTime: new Date(),
    });
  }

  /**
   * Log election end
   */
  async logElectionEnded(
    totalBallots: number,
    tally: EncryptedTally,
    actor: AuditActor
  ): Promise<AuditEntry> {
    return this.addEntry('election_ended', actor, {
      endTime: new Date(),
      totalBallots,
      tallyHash: this.hashTally(tally),
    });
  }

  /**
   * Log ballot submission
   */
  async logBallotSubmitted(
    ballot: EncryptedBallot,
    actor: AuditActor,
    proofHash?: string
  ): Promise<AuditEntry> {
    const data: BallotSubmittedData = {
      ballotId: ballot.ballotId,
      ballotHash: this.hashBallot(ballot),
    };
    if (proofHash !== undefined) {
      data.proofHash = proofHash;
    }
    return this.addEntry('ballot_submitted', actor, data);
  }

  /**
   * Log ballot verification
   */
  async logBallotVerified(
    ballotId: BallotId,
    verificationResult: boolean,
    proofType: string,
    actor: AuditActor
  ): Promise<AuditEntry> {
    return this.addEntry('ballot_verified', actor, {
      ballotId,
      verificationResult,
      proofType,
    });
  }

  /**
   * Log ballot rejection
   */
  async logBallotRejected(
    ballotId: BallotId,
    reason: string,
    actor: AuditActor,
    details?: string
  ): Promise<AuditEntry> {
    const data: BallotRejectedData = {
      ballotId,
      reason,
    };
    if (details !== undefined) {
      data.details = details;
    }
    return this.addEntry('ballot_rejected', actor, data);
  }

  /**
   * Log tally computation
   */
  async logTallyComputed(
    tally: EncryptedTally,
    ballotCount: number,
    computationTimeMs: number,
    actor: AuditActor
  ): Promise<AuditEntry> {
    return this.addEntry('tally_computed', actor, {
      tallyHash: this.hashTally(tally),
      ballotCount,
      computationTimeMs,
    });
  }

  /**
   * Log tally verification
   */
  async logTallyVerified(
    tally: EncryptedTally,
    verificationResult: boolean,
    proofHash: string,
    actor: AuditActor
  ): Promise<AuditEntry> {
    return this.addEntry('tally_verified', actor, {
      tallyHash: this.hashTally(tally),
      verificationResult,
      proofHash,
    });
  }

  /**
   * Log key generation
   */
  async logKeyGenerated(
    keyType: KeyGeneratedData['keyType'],
    keyHash: string,
    parameters: string,
    actor: AuditActor
  ): Promise<AuditEntry> {
    return this.addEntry('key_generated', actor, {
      keyType,
      keyHash,
      parameters,
    });
  }

  /**
   * Log key sharing
   */
  async logKeyShared(
    shareId: number,
    totalShares: number,
    threshold: number,
    commitmentHash: string,
    actor: AuditActor
  ): Promise<AuditEntry> {
    return this.addEntry('key_shared', actor, {
      shareId,
      totalShares,
      threshold,
      commitmentHash,
    });
  }

  /**
   * Log decryption start
   */
  async logDecryptionStarted(
    tally: EncryptedTally,
    requiredShares: number,
    actor: AuditActor
  ): Promise<AuditEntry> {
    return this.addEntry('decryption_started', actor, {
      tallyHash: this.hashTally(tally),
      requiredShares,
    });
  }

  /**
   * Log partial decryption
   */
  async logPartialDecryption(
    shareId: number,
    partialHash: string,
    actor: AuditActor,
    proofHash?: string
  ): Promise<AuditEntry> {
    const data: PartialDecryptionData = {
      shareId,
      partialHash,
    };
    if (proofHash !== undefined) {
      data.proofHash = proofHash;
    }
    return this.addEntry('partial_decryption', actor, data);
  }

  /**
   * Log decryption completion
   */
  async logDecryptionCompleted(
    resultHash: string,
    sharesUsed: number[],
    verificationProofHash: string,
    actor: AuditActor
  ): Promise<AuditEntry> {
    return this.addEntry('decryption_completed', actor, {
      resultHash,
      sharesUsed,
      verificationProofHash,
    });
  }

  /**
   * Log proof generation
   */
  async logProofGenerated(
    proofType: ProofGeneratedData['proofType'],
    proofHash: string,
    statementHash: string,
    generationTimeMs: number,
    actor: AuditActor
  ): Promise<AuditEntry> {
    return this.addEntry('proof_generated', actor, {
      proofType,
      proofHash,
      statementHash,
      generationTimeMs,
    });
  }

  /**
   * Log proof verification
   */
  async logProofVerified(
    proofType: ProofVerifiedData['proofType'],
    proofHash: string,
    verificationResult: boolean,
    verificationTimeMs: number,
    actor: AuditActor
  ): Promise<AuditEntry> {
    return this.addEntry('proof_verified', actor, {
      proofType,
      proofHash,
      verificationResult,
      verificationTimeMs,
    });
  }

  /**
   * Log fraud detection
   */
  async logFraudDetected(
    fraudType: FraudDetectedData['fraudType'],
    severity: FraudDetectedData['severity'],
    details: string,
    evidenceHash: string,
    actor: AuditActor
  ): Promise<AuditEntry> {
    return this.addEntry('fraud_detected', actor, {
      fraudType,
      severity,
      details,
      evidenceHash,
    });
  }

  // ========================================================================
  // Query Methods
  // ========================================================================

  /**
   * Query audit entries
   */
  query(options: AuditQueryOptions = {}): AuditEntry[] {
    let results = [...this.entries];

    if (options.operations !== undefined && options.operations.length > 0) {
      results = results.filter((e) => options.operations!.includes(e.operation));
    }

    if (options.actorType !== undefined) {
      results = results.filter((e) => e.actor.type === options.actorType);
    }

    if (options.actorId !== undefined && options.actorId !== '') {
      results = results.filter((e) => e.actor.id === options.actorId);
    }

    if (options.startTime !== undefined) {
      results = results.filter((e) => e.timestamp >= options.startTime!);
    }

    if (options.endTime !== undefined) {
      results = results.filter((e) => e.timestamp <= options.endTime!);
    }

    if (options.offset !== undefined) {
      results = results.slice(options.offset);
    }

    if (options.limit !== undefined) {
      results = results.slice(0, options.limit);
    }

    return results;
  }

  /**
   * Get entry by ID
   */
  getEntry(id: string): AuditEntry | undefined {
    return this.entries.find((e) => e.id === id);
  }

  /**
   * Get entry by sequence number
   */
  getEntryBySequence(sequenceNumber: number): AuditEntry | undefined {
    return this.entries.find((e) => e.sequenceNumber === sequenceNumber);
  }

  /**
   * Get entries for a specific ballot
   */
  getBallotHistory(ballotId: BallotId): AuditEntry[] {
    return this.entries.filter((e) => {
      if ('ballotId' in e.data) {
        return (e.data as { ballotId: string }).ballotId === ballotId;
      }
      return false;
    });
  }

  /**
   * Get entries for a specific voter
   */
  getVoterHistory(voterId: VoterId): AuditEntry[] {
    return this.entries.filter((e) => {
      if ('voterId' in e.data) {
        return (e.data as { voterId: string }).voterId === voterId;
      }
      return e.actor.id === voterId;
    });
  }

  // ========================================================================
  // Verification Methods
  // ========================================================================

  /**
   * Verify the integrity of the audit trail
   */
  verifyIntegrity(): AuditVerificationResult {
    if (this.entries.length === 0) {
      return { valid: true, entriesVerified: 0 };
    }

    // Verify first entry links to genesis
    const firstEntry = this.entries[0]!;
    if (firstEntry.previousHash !== this.genesisHash) {
      return {
        valid: false,
        entriesVerified: 0,
        firstInvalidEntry: 0,
        errorMessage: 'First entry does not link to genesis hash',
      };
    }

    // Verify hash chain
    for (let i = 0; i < this.entries.length; i++) {
      const entry = this.entries[i]!;

      // Verify entry hash
      const computedHash = this.computeEntryHash(entry);
      if (computedHash !== entry.hash) {
        return {
          valid: false,
          entriesVerified: i,
          firstInvalidEntry: i,
          errorMessage: `Entry ${i} hash mismatch`,
        };
      }

      // Verify chain link (except first entry)
      if (i > 0) {
        const previousEntry = this.entries[i - 1]!;
        if (entry.previousHash !== previousEntry.hash) {
          return {
            valid: false,
            entriesVerified: i,
            firstInvalidEntry: i,
            errorMessage: `Entry ${i} does not link to previous entry`,
          };
        }
      }
    }

    return { valid: true, entriesVerified: this.entries.length };
  }

  /**
   * Verify a specific entry
   */
  verifyEntry(entry: AuditEntry): boolean {
    const computedHash = this.computeEntryHash(entry);
    return computedHash === entry.hash;
  }

  // ========================================================================
  // Export Methods
  // ========================================================================

  /**
   * Export audit trail as JSON
   */
  async exportJSON(options: AuditQueryOptions = {}): Promise<string> {
    const entries = this.query(options);
    const exportData = {
      electionId: this.electionId,
      exportedAt: new Date().toISOString(),
      entryCount: entries.length,
      genesisHash: this.genesisHash,
      entries: entries.map((e) => ({
        ...e,
        timestamp: e.timestamp.toISOString(),
      })),
    };

    const json = JSON.stringify(exportData, null, 2);

    // Log the export
    await this.addEntry(
      'audit_exported',
      { type: 'system', id: 'audit_manager' },
      {
        format: 'json',
        entryCount: entries.length,
        exportHash: this.hashString(json),
      }
    );

    return json;
  }

  /**
   * Export audit trail as CSV
   */
  async exportCSV(options: AuditQueryOptions = {}): Promise<string> {
    const entries = this.query(options);

    const headers = [
      'id',
      'sequenceNumber',
      'operation',
      'timestamp',
      'actorType',
      'actorId',
      'previousHash',
      'hash',
      'data',
    ];

    const rows = entries.map((e) => [
      e.id,
      e.sequenceNumber.toString(),
      e.operation,
      e.timestamp.toISOString(),
      e.actor.type,
      e.actor.id,
      e.previousHash,
      e.hash,
      JSON.stringify(e.data),
    ]);

    const csv = [headers.join(','), ...rows.map((r) => r.map(escapeCSV).join(','))].join('\n');

    // Log the export
    await this.addEntry(
      'audit_exported',
      { type: 'system', id: 'audit_manager' },
      {
        format: 'csv',
        entryCount: entries.length,
        exportHash: this.hashString(csv),
      }
    );

    return csv;
  }

  /**
   * Import audit entries from JSON
   */
  importJSON(json: string): { imported: number; errors: string[] } {
    const errors: string[] = [];
    let imported = 0;

    try {
      const data = JSON.parse(json) as {
        electionId: string;
        entries: Array<{
          id: string;
          sequenceNumber: number;
          operation: AuditOperationType;
          electionId: ElectionId;
          timestamp: string;
          actor: AuditActor;
          data: AuditData;
          previousHash: string;
          hash: string;
          signature?: string;
        }>;
      };

      if (data.electionId !== this.electionId) {
        errors.push(`Election ID mismatch: expected ${this.electionId}, got ${data.electionId}`);
        return { imported: 0, errors };
      }

      for (const entryData of data.entries) {
        const entry: AuditEntry = {
          id: entryData.id,
          sequenceNumber: entryData.sequenceNumber,
          operation: entryData.operation,
          electionId: entryData.electionId,
          timestamp: new Date(entryData.timestamp),
          actor: entryData.actor,
          data: entryData.data,
          previousHash: entryData.previousHash,
          hash: entryData.hash,
        };
        if (entryData.signature !== undefined) {
          entry.signature = entryData.signature;
        }

        if (!this.verifyEntry(entry)) {
          errors.push(`Entry ${entry.id} failed verification`);
          continue;
        }

        this.entries.push(entry);
        this.sequenceCounter = Math.max(this.sequenceCounter, entry.sequenceNumber + 1);
        imported++;
      }
    } catch (error) {
      errors.push(`Parse error: ${(error as Error).message}`);
    }

    return { imported, errors };
  }

  // ========================================================================
  // Private Methods
  // ========================================================================

  private async addEntry(
    operation: AuditOperationType,
    actor: AuditActor,
    data: AuditData
  ): Promise<AuditEntry> {
    const sequenceNumber = this.sequenceCounter++;
    const previousHash =
      this.entries.length > 0 ? this.entries[this.entries.length - 1]!.hash : this.genesisHash;

    const entry: AuditEntry = {
      id: this.generateEntryId(),
      sequenceNumber,
      operation,
      electionId: this.electionId,
      timestamp: new Date(),
      actor,
      data,
      previousHash,
      hash: '', // Will be computed
    };

    entry.hash = this.computeEntryHash(entry);

    // Add signature if enabled
    if (this.config.enableSignatures === true && this.config.signingKey !== undefined) {
      entry.signature = await this.signEntry(entry);
    }

    this.entries.push(entry);

    // Auto-persist if enabled
    if (this.config.autoPersist === true && this.config.onPersist !== undefined) {
      await this.config.onPersist([entry]);
    }

    // Trim old entries if needed
    if (
      this.config.maxInMemoryEntries !== undefined &&
      this.entries.length > this.config.maxInMemoryEntries
    ) {
      const toRemove = this.entries.length - this.config.maxInMemoryEntries;
      this.entries.splice(0, toRemove);
    }

    return entry;
  }

  private generateEntryId(): string {
    return `audit_${this.electionId}_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  }

  private computeGenesisHash(): string {
    return this.hashString(`genesis:${this.electionId}:${Date.now()}`);
  }

  private computeEntryHash(entry: AuditEntry): string {
    const content = [
      entry.id,
      entry.sequenceNumber.toString(),
      entry.operation,
      entry.electionId,
      entry.timestamp.toISOString(),
      JSON.stringify(entry.actor),
      JSON.stringify(entry.data),
      entry.previousHash,
    ].join('|');

    return this.hashString(content);
  }

  private hashString(content: string): string {
    return createHash('sha256').update(content).digest('hex');
  }

  private hashPublicKey(pk: PublicKey): string {
    return this.hashString(`pk:${pk.keyId.toString()}:${pk.handle.toString()}`);
  }

  private hashBallot(ballot: EncryptedBallot): string {
    const choicesHash = ballot.encryptedChoices.map((ct) => ct.handle.toString()).join(',');
    return this.hashString(`ballot:${ballot.ballotId}:${choicesHash}`);
  }

  private hashTally(tally: EncryptedTally): string {
    const countsHash = tally.encryptedCounts.map((ct) => ct.handle.toString()).join(',');
    return this.hashString(`tally:${tally.electionId}:${countsHash}:${tally.totalBallots}`);
  }

  private async signEntry(_entry: AuditEntry): Promise<string> {
    // Placeholder for actual signing implementation
    // In production, this would use the signing key to create a digital signature
    return 'signature_placeholder';
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Escape a value for CSV
 */
function escapeCSV(value: string): string {
  if (value.includes(',') || value.includes('"') || value.includes('\n')) {
    return `"${value.replace(/"/g, '""')}"`;
  }
  return value;
}

/**
 * Create a system actor
 */
export function createSystemActor(id: string = 'system'): AuditActor {
  return { type: 'system', id };
}

/**
 * Create a voter actor
 */
export function createVoterActor(voterId: VoterId, publicKeyHash?: string): AuditActor {
  const actor: AuditActor = { type: 'voter', id: voterId };
  if (publicKeyHash !== undefined) {
    actor.publicKeyHash = publicKeyHash;
  }
  return actor;
}

/**
 * Create an official actor
 */
export function createOfficialActor(officialId: string, publicKeyHash?: string): AuditActor {
  const actor: AuditActor = { type: 'official', id: officialId };
  if (publicKeyHash !== undefined) {
    actor.publicKeyHash = publicKeyHash;
  }
  return actor;
}

/**
 * Create a verifier actor
 */
export function createVerifierActor(verifierId: string): AuditActor {
  return { type: 'verifier', id: verifierId };
}

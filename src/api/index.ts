/**
 * @file api/index.ts
 * @brief Main API exports for FHE operations
 *
 * This module re-exports all API types and functions for convenient access.
 *
 * Requirements: 12.1, 12.2, 12.3, 12.6
 */

// Core types
export * from './types';

// Voting types
export * from './voting-types';

// ZK proof types
export * from './zk-types';

// FHE Engine
export { createFHEEngine } from './fhe-engine';
export type { FHEEngine } from './fhe-engine';

// High-level convenience API
export {
  FHEContext,
  createFastContext,
  createBalancedContext,
  createSecureContext,
  createSIMDContext,
  createMLContext,
  createVotingContext,
} from './fhe-context';
export type { FHEContextOptions } from './fhe-context';

// Tally Streaming API
export {
  TallyStreamManager,
  TallyWebSocketAdapter,
  createZeroCiphertext,
  homomorphicAdd,
  serializeTallyEvent,
  serializeStreamingEncryptedTally,
} from './tally-streaming';
export type {
  TallyEventType,
  TallyEvent,
  TallyEventData,
  BallotReceivedData,
  TallyUpdatedData,
  StreamingElectionStartedData,
  StreamingElectionEndedData,
  StreamingErrorData,
  StreamingSubscriberData,
  TallySubscriber,
  SubscriptionOptions,
  WebSocketMessageType,
  WebSocketMessage,
  WebSocketConnection,
} from './tally-streaming';

// Audit Trail API
export {
  AuditTrailManager,
  createSystemActor,
  createVoterActor,
  createOfficialActor,
  createVerifierActor,
} from './audit-trail';
export type {
  ElectionId,
  VoterId,
  BallotId,
  AuditOperationType,
  AuditEntry,
  AuditActor,
  AuditData,
  AuditTrailConfig,
  AuditQueryOptions,
  AuditVerificationResult,
} from './audit-trail';

// ZK Proofs API
export {
  BulletproofsProver,
  Groth16Prover,
  Groth16Verifier,
  Groth16Setup,
  PlonkProver,
  PlonkVerifier,
  PlonkSetup,
  ZKProofManager,
} from './zk-proofs';

// Voting System Example
export { VotingSystem, runVotingExample } from './voting-example';
export type {
  VoterRegistration,
  CompleteBallot,
  ElectionOfficial,
  ElectionResult,
} from './voting-example';

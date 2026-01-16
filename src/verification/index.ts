/**
 * @file verification/index.ts
 * @brief Public verification tools exports
 *
 * Requirements: 17, 19
 */

export {
  PublicVerifier,
  generateHTMLReport,
  exportReportJSON,
  type VerificationElectionConfig,
  type VerificationPackage,
  type BallotProofEntry,
  type EligibilityProofEntry,
  type FinalTally,
  type VerificationResult,
  type VerificationReport,
  type VerificationProgressCallback,
} from './public-verifier';

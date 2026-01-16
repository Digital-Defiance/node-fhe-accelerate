# Election Official Setup Procedures

This document provides step-by-step procedures for election officials to set up and manage elections using the FHE voting system.

## Table of Contents

1. [Overview](#overview)
2. [Role Definitions](#role-definitions)
3. [Pre-Election Setup](#pre-election-setup)
4. [Key Ceremony](#key-ceremony)
5. [Election Configuration](#election-configuration)
6. [Voter Registration](#voter-registration)
7. [Election Day Operations](#election-day-operations)
8. [Post-Election Procedures](#post-election-procedures)
9. [Audit and Verification](#audit-and-verification)

## Overview

The FHE voting system uses threshold cryptography to ensure that no single official can decrypt votes. A minimum of M officials (out of N total) must cooperate to decrypt the final tally.

### Security Model

- **Threshold Decryption**: M-of-N officials required
- **Zero-Knowledge Proofs**: All operations are verifiable
- **Audit Trail**: Cryptographic log of all operations
- **Hardware Security**: Keys protected by Secure Enclave

## Role Definitions

### Chief Election Officer (CEO)

- Overall responsibility for election integrity
- Authorizes key ceremony participants
- Signs final election results
- Access Level: Full administrative access

### Key Custodians (M-of-N)

- Hold threshold key shares
- Participate in key ceremony
- Provide partial decryptions
- Access Level: Key management only

### Election Administrators

- Configure election parameters
- Manage voter registration
- Monitor election progress
- Access Level: Administrative (no key access)

### Auditors

- Verify ZK proofs
- Review audit trails
- Validate election integrity
- Access Level: Read-only verification

## Pre-Election Setup

### 1. System Verification

Before any election, verify the system is properly configured:

```typescript
import { VotingSystem, SystemVerifier } from '@digitaldefiance/node-fhe-accelerate';

// Verify hardware capabilities
const verifier = new SystemVerifier();
const hardwareCheck = await verifier.checkHardware();

console.log('Hardware Verification:');
console.log(`  SME Available: ${hardwareCheck.sme}`);
console.log(`  GPU Cores: ${hardwareCheck.gpuCores}`);
console.log(`  Neural Engine: ${hardwareCheck.neuralEngine}`);
console.log(`  Secure Enclave: ${hardwareCheck.secureEnclave}`);

if (!hardwareCheck.meetsRequirements) {
  throw new Error('Hardware does not meet minimum requirements');
}

// Verify software integrity
const softwareCheck = await verifier.checkSoftware();
console.log(`\nSoftware Verification:`);
console.log(`  Version: ${softwareCheck.version}`);
console.log(`  Checksum Valid: ${softwareCheck.checksumValid}`);
console.log(`  Dependencies OK: ${softwareCheck.dependenciesOk}`);
```

### 2. Security Audit

Run the security audit before each election:

```bash
# Run security audit
yarn audit:security

# Check for known vulnerabilities
yarn audit

# Verify cryptographic implementations
yarn test:crypto
```

### 3. Backup Verification

Ensure backup systems are operational:

```bash
# Test backup to external storage
./scripts/backup-test.sh

# Verify backup integrity
./scripts/verify-backup.sh
```

## Key Ceremony

The key ceremony generates the election's cryptographic keys with threshold sharing.

### Ceremony Requirements

- **Location**: Secure, access-controlled room
- **Witnesses**: At least 2 independent observers
- **Recording**: Video recording of entire ceremony
- **Air-gapped**: No network connectivity during key generation

### Ceremony Procedure

#### Step 1: Gather Participants

All key custodians must be physically present with valid identification.

```typescript
import { KeyCeremony, KeyCustodian } from '@digitaldefiance/node-fhe-accelerate';

// Initialize ceremony
const ceremony = new KeyCeremony({
  electionId: 'election-2024-general',
  threshold: 3,        // M: minimum custodians needed
  totalCustodians: 5,  // N: total custodians
  securityLevel: 128,
});

// Register custodians
const custodians: KeyCustodian[] = [
  { id: 'custodian-1', name: 'Alice Johnson', role: 'CEO' },
  { id: 'custodian-2', name: 'Bob Smith', role: 'Deputy CEO' },
  { id: 'custodian-3', name: 'Carol Williams', role: 'IT Director' },
  { id: 'custodian-4', name: 'David Brown', role: 'Legal Counsel' },
  { id: 'custodian-5', name: 'Eve Davis', role: 'External Auditor' },
];

for (const custodian of custodians) {
  await ceremony.registerCustodian(custodian);
}
```

#### Step 2: Generate Master Keys

```typescript
// Generate election keys (this takes ~30 seconds)
console.log('Generating election keys...');
const keyGenResult = await ceremony.generateKeys({
  useSecureEnclave: true,
  generateBootstrapKey: true,
  generateEvalKey: true,
});

console.log(`Public Key Hash: ${keyGenResult.publicKeyHash}`);
console.log(`Key Generation Time: ${keyGenResult.generationTimeMs}ms`);
```

#### Step 3: Distribute Key Shares

Each custodian receives their encrypted key share:

```typescript
// Generate and distribute shares
const shares = await ceremony.distributeShares();

for (const share of shares) {
  console.log(`\nShare for ${share.custodianId}:`);
  console.log(`  Share ID: ${share.shareId}`);
  console.log(`  Commitment: ${share.commitment}`);
  
  // Each custodian stores their share securely
  // In production, this would be on a hardware security module
  await share.exportToSecureStorage(share.custodianId);
}
```

#### Step 4: Verify Shares

Each custodian verifies their share:

```typescript
// Verification by each custodian
for (const custodian of custodians) {
  const verification = await ceremony.verifyCustodianShare(custodian.id);
  
  if (!verification.valid) {
    throw new Error(`Share verification failed for ${custodian.name}`);
  }
  
  console.log(`${custodian.name}: Share verified ✓`);
}
```

#### Step 5: Test Threshold Decryption

Perform a test decryption to verify the setup:

```typescript
// Test with exactly threshold custodians
const testCustodians = custodians.slice(0, 3); // First 3 custodians

const testResult = await ceremony.testThresholdDecryption(testCustodians);

if (!testResult.success) {
  throw new Error('Threshold decryption test failed');
}

console.log('\nThreshold decryption test: PASSED ✓');
```

#### Step 6: Finalize Ceremony

```typescript
// Generate ceremony report
const report = await ceremony.finalize();

console.log('\n=== Key Ceremony Report ===');
console.log(`Election ID: ${report.electionId}`);
console.log(`Public Key Hash: ${report.publicKeyHash}`);
console.log(`Threshold: ${report.threshold} of ${report.totalCustodians}`);
console.log(`Ceremony Time: ${report.ceremonyDuration}ms`);
console.log(`Witnesses: ${report.witnesses.join(', ')}`);

// Export ceremony report for audit
await report.exportToFile('key-ceremony-report.json');
```

## Election Configuration

### Create Election

```typescript
import { VotingSystem } from '@digitaldefiance/node-fhe-accelerate';

const election = await VotingSystem.create({
  electionId: 'election-2024-general',
  name: 'General Election 2024',
  candidates: [
    'Alice Anderson',
    'Bob Baker',
    'Carol Chen',
    'David Davis',
  ],
  thresholdConfig: {
    threshold: 3,
    totalOfficials: 5,
  },
  expectedVoters: 100000,
  startTime: new Date('2024-11-05T06:00:00Z'),
  endTime: new Date('2024-11-05T20:00:00Z'),
  options: {
    allowWriteIn: false,
    maxSelectionsPerBallot: 1,
    requireEligibilityProof: true,
    requireValidityProof: true,
    enableFraudDetection: true,
    enableRealTimeStreaming: true,
  },
});

console.log('Election created successfully');
console.log(`Election ID: ${election.getElectionId()}`);
```

### Configure Fraud Detection

```typescript
// Configure fraud detection thresholds
await election.configureFraudDetection({
  duplicateVoteDetection: true,
  statisticalAnomalyDetection: true,
  timingAnomalyDetection: true,
  thresholds: {
    duplicateConfidence: 0.99,
    anomalyZScore: 3.0,
    timingDeviationMs: 100,
  },
  alertRecipients: [
    'ceo@election.gov',
    'security@election.gov',
  ],
});
```

## Voter Registration

### Bulk Import

```typescript
import { VoterRegistry } from '@digitaldefiance/node-fhe-accelerate';

const registry = new VoterRegistry(election);

// Import from CSV
await registry.importFromCSV('voters.csv', {
  columns: {
    voterId: 'voter_id',
    publicKeyHash: 'public_key_hash',
    district: 'district',
  },
  validateFormat: true,
  deduplicateOnImport: true,
});

console.log(`Imported ${registry.getVoterCount()} voters`);
```

### Individual Registration

```typescript
// Register individual voter
await registry.registerVoter({
  voterId: 'voter-12345',
  publicKeyHash: voterPublicKeyHash,
  metadata: {
    district: 'District 1',
    registrationDate: new Date(),
  },
});
```

### Build Merkle Tree

After all voters are registered, build the eligibility Merkle tree:

```typescript
// Build Merkle tree for eligibility proofs
const merkleRoot = await registry.buildMerkleTree();

console.log(`Merkle Root: ${merkleRoot.toString('hex')}`);
console.log(`Tree Depth: ${registry.getMerkleTreeDepth()}`);

// Export Merkle tree for public verification
await registry.exportMerkleTree('merkle-tree.json');
```

## Election Day Operations

### Start Election

```typescript
// Start the election (only after all setup is complete)
await election.startElection();

console.log('Election started at:', new Date().toISOString());
```

### Monitor Progress

```typescript
// Subscribe to real-time updates
election.subscribeTallyUpdates((update) => {
  console.log(`Ballots processed: ${update.ballotsProcessed}`);
  console.log(`Processing rate: ${update.ballotsPerSecond}/sec`);
});

// Subscribe to fraud alerts
election.subscribeFraudAlerts((alert) => {
  console.error('FRAUD ALERT:', alert.type);
  console.error('Details:', alert.description);
  console.error('Confidence:', alert.confidence);
});

// Get current statistics
const stats = election.getStats();
console.log('Current Statistics:');
console.log(`  Total Voters: ${stats.totalVoters}`);
console.log(`  Ballots Cast: ${stats.totalBallots}`);
console.log(`  Turnout: ${(stats.totalBallots / stats.totalVoters * 100).toFixed(2)}%`);
```

### Handle Issues

```typescript
// Pause election (emergency only)
await election.pause('Security incident investigation');

// Resume election
await election.resume();

// Extend voting hours (requires authorization)
await election.extendEndTime(
  new Date('2024-11-05T22:00:00Z'),
  'Court order #12345'
);
```

## Post-Election Procedures

### End Election

```typescript
// End the election
const finalTally = await election.endElection();

console.log('Election ended at:', new Date().toISOString());
console.log('Total ballots:', finalTally.totalBallots);
```

### Threshold Decryption

```typescript
// Gather key custodians for decryption
const decryptionCustodians = [
  { id: 'custodian-1', shareId: 1 },
  { id: 'custodian-2', shareId: 2 },
  { id: 'custodian-3', shareId: 3 },
];

// Each custodian provides their partial decryption
const partialDecryptions = [];

for (const custodian of decryptionCustodians) {
  // Load custodian's key share (from secure storage)
  const keyShare = await loadKeyShare(custodian.id);
  
  // Generate partial decryption with proof
  const partial = await election.generatePartialDecryption(
    keyShare,
    custodian.shareId
  );
  
  // Verify partial decryption proof
  const verified = await election.verifyPartialDecryption(partial);
  if (!verified) {
    throw new Error(`Partial decryption verification failed for ${custodian.id}`);
  }
  
  partialDecryptions.push(partial);
  console.log(`${custodian.id}: Partial decryption verified ✓`);
}

// Combine partial decryptions
const result = await election.combinePartialDecryptions(partialDecryptions);

console.log('\n=== Election Results ===');
for (let i = 0; i < result.decryptedTally.counts.length; i++) {
  console.log(`${election.getCandidateName(i)}: ${result.decryptedTally.counts[i]} votes`);
}
console.log(`Total: ${result.decryptedTally.totalBallots} ballots`);
```

### Generate Proofs

```typescript
// Generate tally correctness proof
const tallyProof = await election.generateTallyCorrectnessProof();

console.log('Tally Correctness Proof:');
console.log(`  Proof Size: ${tallyProof.sizeBytes} bytes`);
console.log(`  Generation Time: ${tallyProof.generationTimeMs}ms`);

// Verify the proof
const proofValid = await election.verifyTallyCorrectnessProof(tallyProof);
console.log(`  Verification: ${proofValid ? 'VALID ✓' : 'INVALID ✗'}`);
```

## Audit and Verification

### Export Audit Trail

```typescript
// Export complete audit trail
const auditJson = await election.exportAuditTrail('json');
await fs.writeFile('audit-trail.json', auditJson);

const auditCsv = await election.exportAuditTrail('csv');
await fs.writeFile('audit-trail.csv', auditCsv);

console.log('Audit trail exported');
```

### Verify Audit Trail Integrity

```typescript
// Verify audit trail integrity
const auditVerification = election.verifyAuditTrail();

console.log('Audit Trail Verification:');
console.log(`  Valid: ${auditVerification.valid}`);
console.log(`  Entries Verified: ${auditVerification.entriesVerified}`);

if (!auditVerification.valid) {
  console.error(`  First Invalid Entry: ${auditVerification.firstInvalidEntry}`);
  console.error(`  Error: ${auditVerification.errorMessage}`);
}
```

### Public Verification

Generate verification package for public auditors:

```typescript
// Generate public verification package
const verificationPackage = await election.generateVerificationPackage();

// Package includes:
// - Election configuration
// - Public key
// - Merkle root
// - All ZK proofs
// - Audit trail hash
// - Final results

await verificationPackage.exportToFile('verification-package.zip');

console.log('Verification package generated');
console.log(`Package Hash: ${verificationPackage.hash}`);
```

## Checklist

### Pre-Election

- [ ] Hardware verification complete
- [ ] Software audit passed
- [ ] Backup systems tested
- [ ] Key ceremony completed
- [ ] All custodians verified
- [ ] Election configured
- [ ] Voters registered
- [ ] Merkle tree built
- [ ] Fraud detection configured
- [ ] Monitoring enabled

### Election Day

- [ ] System health check
- [ ] Election started on time
- [ ] Real-time monitoring active
- [ ] Support staff available
- [ ] Incident response ready

### Post-Election

- [ ] Election ended
- [ ] Threshold decryption completed
- [ ] Results verified
- [ ] Proofs generated
- [ ] Audit trail exported
- [ ] Verification package created
- [ ] Results published
- [ ] System secured

## Emergency Procedures

### System Failure

1. Activate backup system
2. Notify all officials
3. Document incident
4. Resume operations when safe

### Security Breach

1. Pause election immediately
2. Isolate affected systems
3. Notify security team
4. Assess impact
5. Document everything
6. Resume only after clearance

### Key Compromise

1. Revoke compromised share
2. Notify all custodians
3. Assess if threshold still met
4. Consider election restart if necessary

## Contact Information

- **Technical Support**: support@digitaldefiance.org
- **Security Hotline**: security@digitaldefiance.org

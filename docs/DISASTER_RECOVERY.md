# Disaster Recovery Procedures

This document outlines disaster recovery procedures for the FHE voting system to ensure election continuity and data integrity.

## Table of Contents

1. [Overview](#overview)
2. [Backup Strategy](#backup-strategy)
3. [Recovery Procedures](#recovery-procedures)
4. [Failover Configuration](#failover-configuration)
5. [Data Integrity Verification](#data-integrity-verification)
6. [Testing and Drills](#testing-and-drills)

## Overview

### Recovery Objectives

| Metric | Target | Description |
|--------|--------|-------------|
| **RTO** | 15 minutes | Recovery Time Objective |
| **RPO** | 0 ballots | Recovery Point Objective (no ballot loss) |
| **MTTR** | 30 minutes | Mean Time To Recovery |

### Critical Components

1. **Election State**: Active elections, configurations
2. **Encrypted Ballots**: All submitted ballots
3. **Cryptographic Keys**: Public keys, key shares (NOT secret keys)
4. **Audit Trail**: Complete operation history
5. **Voter Registry**: Merkle tree, voter records

## Backup Strategy

### Continuous Backup

```typescript
import { BackupManager } from '@digitaldefiance/node-fhe-accelerate';

const backup = new BackupManager({
  // Primary backup location
  primary: {
    type: 'local',
    path: '/backup/fhe-voting',
  },
  // Secondary backup (offsite)
  secondary: {
    type: 's3',
    bucket: 'election-backup',
    region: 'us-west-2',
    encryption: 'AES-256',
  },
  // Backup schedule
  schedule: {
    full: '0 0 * * *',      // Daily full backup
    incremental: '*/5 * * * *', // Every 5 minutes
    auditTrail: '* * * * *',    // Every minute
  },
  // Retention
  retention: {
    full: 90,        // 90 days
    incremental: 7,  // 7 days
    auditTrail: 365, // 1 year
  },
});

// Start backup service
await backup.start();

// Manual backup trigger
await backup.createFullBackup('pre-election-backup');
```

### Backup Components

#### 1. Election State Backup

```typescript
// Backup election state
async function backupElectionState(election: Election): Promise<BackupResult> {
  const state = {
    electionId: election.id,
    config: election.config,
    status: election.status,
    statistics: election.getStats(),
    timestamp: new Date().toISOString(),
  };
  
  const encrypted = await encryptBackup(JSON.stringify(state));
  await writeBackup(`elections/${election.id}/state.enc`, encrypted);
  
  return {
    component: 'election_state',
    size: encrypted.length,
    checksum: computeChecksum(encrypted),
  };
}
```

#### 2. Ballot Backup

```typescript
// Continuous ballot backup with write-ahead log
class BallotBackup {
  private wal: WriteAheadLog;
  
  constructor(backupPath: string) {
    this.wal = new WriteAheadLog(backupPath);
  }
  
  async backupBallot(ballot: EncryptedBallot): Promise<void> {
    // Write to WAL first (synchronous, durable)
    await this.wal.append({
      type: 'ballot',
      data: ballot,
      timestamp: Date.now(),
    });
    
    // Then write to backup storage (async)
    await this.writeToBackupStorage(ballot);
  }
  
  async recover(): Promise<EncryptedBallot[]> {
    // Recover from WAL
    const entries = await this.wal.readAll();
    return entries
      .filter(e => e.type === 'ballot')
      .map(e => e.data as EncryptedBallot);
  }
}
```

#### 3. Audit Trail Backup

```typescript
// Real-time audit trail replication
class AuditTrailReplicator {
  private primary: AuditTrailManager;
  private replicas: AuditTrailManager[];
  
  async replicate(entry: AuditEntry): Promise<void> {
    // Write to all replicas in parallel
    await Promise.all(
      this.replicas.map(replica => replica.append(entry))
    );
  }
  
  async verifyConsistency(): Promise<boolean> {
    const primaryHash = await this.primary.getRootHash();
    
    for (const replica of this.replicas) {
      const replicaHash = await replica.getRootHash();
      if (primaryHash !== replicaHash) {
        return false;
      }
    }
    
    return true;
  }
}
```

### Backup Verification

```bash
#!/bin/bash
# verify-backup.sh - Verify backup integrity

BACKUP_PATH="/backup/fhe-voting"
DATE=$(date +%Y%m%d)

echo "Verifying backup for $DATE..."

# Verify checksums
for file in $BACKUP_PATH/$DATE/*.enc; do
  expected=$(cat "$file.sha256")
  actual=$(shasum -a 256 "$file" | cut -d' ' -f1)
  
  if [ "$expected" != "$actual" ]; then
    echo "CHECKSUM MISMATCH: $file"
    exit 1
  fi
done

# Verify decryption
for file in $BACKUP_PATH/$DATE/*.enc; do
  if ! decrypt-backup "$file" > /dev/null 2>&1; then
    echo "DECRYPTION FAILED: $file"
    exit 1
  fi
done

# Verify data integrity
node verify-backup-data.js "$BACKUP_PATH/$DATE"

echo "Backup verification: PASSED"
```

## Recovery Procedures

### Scenario 1: Server Failure

**Symptoms**: Server unresponsive, hardware failure

**Recovery Steps**:

```bash
#!/bin/bash
# recover-server-failure.sh

echo "=== Server Failure Recovery ==="

# 1. Activate standby server
echo "Step 1: Activating standby server..."
ssh standby-server "sudo systemctl start fhe-voting"

# 2. Update DNS/load balancer
echo "Step 2: Updating traffic routing..."
./update-dns.sh standby-server.election.gov

# 3. Restore latest backup
echo "Step 3: Restoring from backup..."
ssh standby-server "restore-backup.sh /backup/fhe-voting/latest"

# 4. Verify data integrity
echo "Step 4: Verifying data integrity..."
ssh standby-server "verify-election-state.sh"

# 5. Resume operations
echo "Step 5: Resuming election operations..."
ssh standby-server "resume-election.sh"

echo "=== Recovery Complete ==="
```

### Scenario 2: Data Corruption

**Symptoms**: Invalid checksums, decryption failures, audit trail inconsistency

**Recovery Steps**:

```typescript
async function recoverFromCorruption(electionId: string): Promise<void> {
  console.log('Starting corruption recovery...');
  
  // 1. Stop accepting new ballots
  await election.pause('Data corruption recovery');
  
  // 2. Identify corruption extent
  const corruptionReport = await analyzeCorruption(electionId);
  console.log('Corruption report:', corruptionReport);
  
  // 3. Find last known good state
  const lastGoodBackup = await findLastGoodBackup(electionId);
  console.log('Last good backup:', lastGoodBackup.timestamp);
  
  // 4. Restore from backup
  await restoreFromBackup(lastGoodBackup);
  
  // 5. Replay WAL entries after backup
  const walEntries = await getWALEntriesAfter(lastGoodBackup.timestamp);
  for (const entry of walEntries) {
    if (await verifyEntry(entry)) {
      await replayEntry(entry);
    } else {
      console.warn('Skipping corrupted entry:', entry.id);
    }
  }
  
  // 6. Verify recovery
  const integrityCheck = await verifyElectionIntegrity(electionId);
  if (!integrityCheck.valid) {
    throw new Error('Recovery verification failed');
  }
  
  // 7. Resume operations
  await election.resume();
  console.log('Recovery complete');
}
```

### Scenario 3: Security Breach

**Symptoms**: Unauthorized access detected, anomalous activity

**Recovery Steps**:

1. **Immediate Actions** (0-5 minutes)
   ```bash
   # Isolate affected systems
   ./isolate-system.sh primary-server
   
   # Preserve evidence
   ./capture-forensics.sh
   
   # Notify security team
   ./alert-security-team.sh "BREACH DETECTED"
   ```

2. **Assessment** (5-30 minutes)
   ```typescript
   // Analyze breach scope
   const breachAnalysis = await analyzeSecurityBreach({
     startTime: suspectedBreachTime,
     endTime: new Date(),
     systems: ['primary', 'backup', 'network'],
   });
   
   // Determine if election integrity is compromised
   const integrityStatus = await assessElectionIntegrity(breachAnalysis);
   ```

3. **Recovery Decision**
   - If integrity maintained: Resume with enhanced monitoring
   - If integrity compromised: Consult legal/election authorities

### Scenario 4: Key Share Loss

**Symptoms**: Key custodian unavailable, key share corrupted

**Recovery Steps**:

```typescript
async function handleKeyShareLoss(
  lostShareId: number,
  remainingCustodians: KeyCustodian[]
): Promise<void> {
  // Check if threshold can still be met
  const threshold = election.config.thresholdConfig.threshold;
  const totalShares = election.config.thresholdConfig.totalOfficials;
  const availableShares = remainingCustodians.length;
  
  if (availableShares >= threshold) {
    console.log(`Threshold can be met with ${availableShares} shares`);
    // Proceed with available shares
    return;
  }
  
  // Threshold cannot be met - this is a critical situation
  console.error('CRITICAL: Threshold cannot be met');
  
  // Options:
  // 1. Recover share from backup (if available)
  const backupShare = await recoverShareFromBackup(lostShareId);
  if (backupShare) {
    console.log('Share recovered from backup');
    return;
  }
  
  // 2. Contact custodian for recovery
  // 3. Consult legal authorities
  // 4. Consider election restart (last resort)
  
  throw new Error('Key share recovery required - manual intervention needed');
}
```

## Failover Configuration

### Active-Passive Setup

```typescript
// Primary server configuration
const primaryConfig = {
  role: 'primary',
  heartbeatInterval: 5000,
  failoverThreshold: 3, // missed heartbeats
  
  onFailover: async () => {
    // Notify standby to take over
    await notifyStandby('FAILOVER');
  },
};

// Standby server configuration
const standbyConfig = {
  role: 'standby',
  primaryUrl: 'https://primary.election.gov',
  heartbeatInterval: 5000,
  
  onPrimaryFailure: async () => {
    console.log('Primary failure detected, initiating failover...');
    
    // 1. Verify primary is truly down
    const primaryDown = await verifyPrimaryDown();
    if (!primaryDown) {
      console.log('False alarm, primary is responsive');
      return;
    }
    
    // 2. Acquire leadership
    await acquireLeadership();
    
    // 3. Sync latest state
    await syncFromBackup();
    
    // 4. Start accepting traffic
    await startAcceptingTraffic();
    
    // 5. Notify operators
    await notifyOperators('FAILOVER_COMPLETE');
  },
};
```

### Health Check for Failover

```typescript
class FailoverMonitor {
  private missedHeartbeats = 0;
  private readonly threshold = 3;
  
  async checkPrimary(): Promise<boolean> {
    try {
      const response = await fetch(`${primaryUrl}/health`, {
        timeout: 5000,
      });
      
      if (response.ok) {
        this.missedHeartbeats = 0;
        return true;
      }
    } catch (error) {
      this.missedHeartbeats++;
    }
    
    if (this.missedHeartbeats >= this.threshold) {
      await this.initiateFailover();
      return false;
    }
    
    return true;
  }
}
```

## Data Integrity Verification

### Continuous Verification

```typescript
class IntegrityVerifier {
  async verifyElection(electionId: string): Promise<IntegrityReport> {
    const report: IntegrityReport = {
      electionId,
      timestamp: new Date(),
      checks: [],
    };
    
    // 1. Verify ballot count
    report.checks.push(await this.verifyBallotCount(electionId));
    
    // 2. Verify audit trail chain
    report.checks.push(await this.verifyAuditChain(electionId));
    
    // 3. Verify Merkle tree
    report.checks.push(await this.verifyMerkleTree(electionId));
    
    // 4. Verify ZK proofs
    report.checks.push(await this.verifyProofs(electionId));
    
    // 5. Verify tally consistency
    report.checks.push(await this.verifyTallyConsistency(electionId));
    
    report.valid = report.checks.every(c => c.valid);
    return report;
  }
  
  private async verifyAuditChain(electionId: string): Promise<Check> {
    const auditTrail = await getAuditTrail(electionId);
    const verification = auditTrail.verifyIntegrity();
    
    return {
      name: 'audit_chain',
      valid: verification.valid,
      details: verification,
    };
  }
}
```

## Testing and Drills

### Monthly DR Drill

```bash
#!/bin/bash
# dr-drill.sh - Monthly disaster recovery drill

echo "=== Disaster Recovery Drill ==="
echo "Date: $(date)"

# 1. Simulate primary failure
echo "Step 1: Simulating primary failure..."
./simulate-failure.sh primary

# 2. Verify failover
echo "Step 2: Verifying automatic failover..."
sleep 30
./verify-failover.sh

# 3. Test backup restoration
echo "Step 3: Testing backup restoration..."
./test-restore.sh

# 4. Verify data integrity
echo "Step 4: Verifying data integrity..."
./verify-integrity.sh

# 5. Test failback
echo "Step 5: Testing failback to primary..."
./test-failback.sh

# 6. Generate report
echo "Step 6: Generating drill report..."
./generate-drill-report.sh

echo "=== Drill Complete ==="
```

### Checklist

- [ ] Backup verification (daily)
- [ ] Failover test (weekly)
- [ ] Full DR drill (monthly)
- [ ] Key share recovery test (quarterly)
- [ ] Security breach simulation (quarterly)
- [ ] Documentation review (quarterly)

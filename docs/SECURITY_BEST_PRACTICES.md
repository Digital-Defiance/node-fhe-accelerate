# Security Best Practices

This document outlines security best practices for deploying and operating the FHE voting system.

## Table of Contents

1. [Overview](#overview)
2. [Infrastructure Security](#infrastructure-security)
3. [Cryptographic Security](#cryptographic-security)
4. [Access Control](#access-control)
5. [Network Security](#network-security)
6. [Operational Security](#operational-security)
7. [Incident Response](#incident-response)
8. [Compliance](#compliance)

## Overview

### Security Principles

1. **Defense in Depth**: Multiple layers of security controls
2. **Least Privilege**: Minimal access rights for all users and processes
3. **Zero Trust**: Verify everything, trust nothing
4. **Transparency**: All operations are auditable and verifiable
5. **Privacy by Design**: Voter privacy is protected at every layer

### Threat Model

| Threat | Mitigation |
|--------|------------|
| Vote manipulation | FHE encryption, ZK proofs |
| Voter coercion | Receipt-free voting |
| Server compromise | Threshold decryption |
| Network attacks | TLS 1.3, certificate pinning |
| Insider threats | Multi-party computation, audit trails |
| Side-channel attacks | Constant-time implementations |

## Infrastructure Security

### Hardware Security

```bash
# Enable FileVault encryption
sudo fdesetup enable

# Enable Secure Boot
# (Configured in Recovery Mode)

# Verify System Integrity Protection
csrutil status
# Expected: System Integrity Protection status: enabled.

# Enable firewall
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on
```

### Secure Enclave Usage

```typescript
import { SecureEnclaveManager } from '@digitaldefiance/node-fhe-accelerate';

// Generate keys in Secure Enclave
const secureEnclave = new SecureEnclaveManager();

// Secret key never leaves Secure Enclave
const keyHandle = await secureEnclave.generateSecretKey({
  keySize: 256,
  algorithm: 'ECDSA',
  accessControl: {
    biometricRequired: true,
    devicePasscodeRequired: true,
  },
});

// Sign operations happen inside Secure Enclave
const signature = await secureEnclave.sign(keyHandle, data);
```

### System Hardening

```bash
#!/bin/bash
# harden-system.sh

# Disable unnecessary services
sudo launchctl disable system/com.apple.screensharing
sudo launchctl disable system/com.apple.RemoteDesktop

# Configure secure SSH
cat >> /etc/ssh/sshd_config << EOF
PermitRootLogin no
PasswordAuthentication no
PubkeyAuthentication yes
AllowUsers fhevoting
MaxAuthTries 3
LoginGraceTime 30
EOF

# Set secure file permissions
chmod 700 /var/fhevoting
chmod 600 /var/fhevoting/config/*
chmod 400 /var/fhevoting/keys/*

# Enable audit logging
sudo launchctl load -w /System/Library/LaunchDaemons/com.apple.auditd.plist
```

## Cryptographic Security

### Key Management

```typescript
// Key generation with proper entropy
import { KeyManager, EntropySource } from '@digitaldefiance/node-fhe-accelerate';

const keyManager = new KeyManager({
  entropySource: EntropySource.HARDWARE, // Use hardware RNG
  keyDerivation: 'HKDF-SHA256',
  minimumEntropy: 256,
});

// Generate election keys
const electionKeys = await keyManager.generateElectionKeys({
  securityLevel: 128,
  threshold: 3,
  totalShares: 5,
});

// Secure key storage
await keyManager.storeKey(electionKeys.publicKey, {
  storage: 'keychain',
  accessControl: 'biometric',
});
```

### Constant-Time Operations

```cpp
// constant_time.h - Constant-time comparison to prevent timing attacks

// Constant-time comparison
inline bool constant_time_compare(
    const uint8_t* a,
    const uint8_t* b,
    size_t len
) {
    uint8_t result = 0;
    for (size_t i = 0; i < len; i++) {
        result |= a[i] ^ b[i];
    }
    return result == 0;
}

// Constant-time conditional select
inline uint64_t constant_time_select(
    uint64_t condition,  // 0 or 1
    uint64_t a,
    uint64_t b
) {
    uint64_t mask = ~(condition - 1);
    return (a & mask) | (b & ~mask);
}
```

### Random Number Generation

```typescript
// Secure random number generation
import { SecureRandom } from '@digitaldefiance/node-fhe-accelerate';

const rng = new SecureRandom({
  source: 'hardware',  // Use hardware RNG
  reseedInterval: 1000, // Reseed every 1000 calls
});

// Generate random bytes
const randomBytes = await rng.getBytes(32);

// Generate random field element
const randomFieldElement = await rng.getFieldElement(fieldModulus);
```

## Access Control

### Role-Based Access Control

```typescript
// Define roles and permissions
const roles = {
  CEO: {
    permissions: ['*'],
    mfa: 'required',
    sessionTimeout: 3600,
  },
  KEY_CUSTODIAN: {
    permissions: [
      'key:generate_share',
      'key:provide_partial_decryption',
      'election:view',
    ],
    mfa: 'required',
    sessionTimeout: 1800,
  },
  ADMINISTRATOR: {
    permissions: [
      'election:create',
      'election:configure',
      'voter:register',
      'audit:view',
    ],
    mfa: 'required',
    sessionTimeout: 3600,
  },
  AUDITOR: {
    permissions: [
      'election:view',
      'audit:view',
      'audit:export',
      'proof:verify',
    ],
    mfa: 'optional',
    sessionTimeout: 7200,
  },
};

// Enforce access control
function checkPermission(user: User, permission: string): boolean {
  const role = roles[user.role];
  if (!role) return false;
  
  if (role.permissions.includes('*')) return true;
  return role.permissions.includes(permission);
}
```

### Multi-Factor Authentication

```typescript
// MFA configuration
const mfaConfig = {
  methods: ['totp', 'hardware_key', 'biometric'],
  requiredFactors: 2,
  
  // Hardware key (YubiKey) configuration
  hardwareKey: {
    algorithms: ['ES256', 'RS256'],
    attestation: 'direct',
    userVerification: 'required',
  },
  
  // TOTP configuration
  totp: {
    algorithm: 'SHA256',
    digits: 6,
    period: 30,
  },
};

// Verify MFA
async function verifyMFA(user: User, factors: MFAFactor[]): Promise<boolean> {
  if (factors.length < mfaConfig.requiredFactors) {
    return false;
  }
  
  for (const factor of factors) {
    const verified = await verifyFactor(user, factor);
    if (!verified) return false;
  }
  
  return true;
}
```

## Network Security

### TLS Configuration

```typescript
// TLS 1.3 only configuration
const tlsConfig = {
  minVersion: 'TLSv1.3',
  cipherSuites: [
    'TLS_AES_256_GCM_SHA384',
    'TLS_CHACHA20_POLY1305_SHA256',
  ],
  
  // Certificate configuration
  cert: fs.readFileSync('/etc/ssl/certs/voting.crt'),
  key: fs.readFileSync('/etc/ssl/private/voting.key'),
  
  // Client certificate verification (for officials)
  requestCert: true,
  rejectUnauthorized: true,
  ca: fs.readFileSync('/etc/ssl/certs/ca.crt'),
};

// Create HTTPS server
const server = https.createServer(tlsConfig, app);
```

### Certificate Pinning

```typescript
// Client-side certificate pinning
const pinnedCertificates = [
  'sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=', // Primary
  'sha256/BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=', // Backup
];

async function verifyServerCertificate(cert: X509Certificate): Promise<boolean> {
  const certHash = crypto
    .createHash('sha256')
    .update(cert.raw)
    .digest('base64');
  
  const pinned = `sha256/${certHash}`;
  return pinnedCertificates.includes(pinned);
}
```

### Rate Limiting

```typescript
// Rate limiting configuration
const rateLimits = {
  // Ballot submission
  ballotSubmission: {
    windowMs: 60000,      // 1 minute
    maxRequests: 1,       // 1 ballot per minute per voter
    keyGenerator: (req) => req.user.voterId,
  },
  
  // API requests
  api: {
    windowMs: 60000,
    maxRequests: 100,
    keyGenerator: (req) => req.ip,
  },
  
  // Authentication attempts
  auth: {
    windowMs: 900000,     // 15 minutes
    maxRequests: 5,       // 5 attempts
    keyGenerator: (req) => req.ip,
    onLimitReached: async (req) => {
      await logSecurityEvent('rate_limit_exceeded', {
        ip: req.ip,
        endpoint: req.path,
      });
    },
  },
};
```

## Operational Security

### Secure Deployment

```bash
#!/bin/bash
# secure-deploy.sh

# Verify code signature
codesign --verify --deep --strict /opt/fhevoting/bin/*

# Verify checksums
sha256sum -c /opt/fhevoting/checksums.sha256

# Set immutable attributes on critical files
chflags schg /opt/fhevoting/bin/*
chflags schg /opt/fhevoting/config/security.json

# Verify file permissions
find /opt/fhevoting -type f -perm /o+w -exec echo "WARNING: World-writable file: {}" \;

# Start with minimal privileges
sudo -u fhevoting /opt/fhevoting/bin/server
```

### Audit Logging

```typescript
// Comprehensive audit logging
const auditLogger = new AuditLogger({
  // Log all security-relevant events
  events: [
    'authentication',
    'authorization',
    'key_operation',
    'ballot_submission',
    'proof_verification',
    'configuration_change',
    'admin_action',
  ],
  
  // Tamper-evident logging
  integrity: {
    enabled: true,
    algorithm: 'SHA256',
    chainPreviousHash: true,
  },
  
  // Secure storage
  storage: {
    type: 'append_only',
    encryption: 'AES-256-GCM',
    replication: 3,
  },
});

// Log security event
await auditLogger.log({
  event: 'authentication',
  actor: user.id,
  action: 'login',
  result: 'success',
  metadata: {
    ip: req.ip,
    userAgent: req.headers['user-agent'],
    mfaMethod: 'hardware_key',
  },
});
```

### Secret Management

```typescript
// Never hardcode secrets
// Use environment variables or secret management service

import { SecretManager } from '@digitaldefiance/node-fhe-accelerate';

const secrets = new SecretManager({
  provider: 'keychain', // macOS Keychain
  namespace: 'fhe-voting',
});

// Retrieve secrets at runtime
const dbPassword = await secrets.get('database_password');
const apiKey = await secrets.get('monitoring_api_key');

// Rotate secrets periodically
await secrets.rotate('api_key', {
  algorithm: 'random',
  length: 32,
  notifyServices: ['monitoring', 'backup'],
});
```

## Incident Response

### Security Incident Procedure

```typescript
// Incident response workflow
class IncidentResponse {
  async handleIncident(incident: SecurityIncident): Promise<void> {
    // 1. Detect and classify
    const classification = await this.classify(incident);
    
    // 2. Contain
    if (classification.severity >= Severity.HIGH) {
      await this.contain(incident);
    }
    
    // 3. Notify
    await this.notify(incident, classification);
    
    // 4. Investigate
    const investigation = await this.investigate(incident);
    
    // 5. Remediate
    await this.remediate(incident, investigation);
    
    // 6. Document
    await this.document(incident, investigation);
    
    // 7. Review and improve
    await this.postIncidentReview(incident);
  }
  
  private async contain(incident: SecurityIncident): Promise<void> {
    switch (incident.type) {
      case 'unauthorized_access':
        await this.revokeAccess(incident.actor);
        await this.isolateSystem(incident.affectedSystems);
        break;
      case 'data_breach':
        await this.isolateSystem(incident.affectedSystems);
        await this.preserveEvidence(incident);
        break;
      case 'dos_attack':
        await this.enableDDoSProtection();
        await this.blockAttacker(incident.sourceIPs);
        break;
    }
  }
}
```

### Evidence Preservation

```bash
#!/bin/bash
# preserve-evidence.sh

INCIDENT_ID=$1
EVIDENCE_DIR="/evidence/$INCIDENT_ID"

mkdir -p "$EVIDENCE_DIR"

# Capture system state
system_profiler SPSoftwareDataType > "$EVIDENCE_DIR/system_info.txt"
ps aux > "$EVIDENCE_DIR/processes.txt"
netstat -an > "$EVIDENCE_DIR/network.txt"
last > "$EVIDENCE_DIR/logins.txt"

# Capture logs
cp -r /var/log/fhe-voting "$EVIDENCE_DIR/app_logs"
cp /var/log/system.log "$EVIDENCE_DIR/system.log"

# Capture memory dump (if needed)
# sudo vmmap -w $(pgrep node) > "$EVIDENCE_DIR/memory_map.txt"

# Calculate checksums
find "$EVIDENCE_DIR" -type f -exec sha256sum {} \; > "$EVIDENCE_DIR/checksums.sha256"

# Seal evidence
tar -czf "$EVIDENCE_DIR.tar.gz" "$EVIDENCE_DIR"
sha256sum "$EVIDENCE_DIR.tar.gz" > "$EVIDENCE_DIR.tar.gz.sha256"

echo "Evidence preserved: $EVIDENCE_DIR.tar.gz"
```

## Compliance

### Security Checklist

- [ ] All communications use TLS 1.3
- [ ] Multi-factor authentication enabled for all officials
- [ ] Audit logging enabled and verified
- [ ] Backup encryption verified
- [ ] Access controls reviewed
- [ ] Security patches applied
- [ ] Penetration testing completed
- [ ] Incident response plan tested
- [ ] Key ceremony procedures documented
- [ ] Disaster recovery tested

### Regular Security Tasks

| Task | Frequency | Owner |
|------|-----------|-------|
| Security patch review | Weekly | IT Security |
| Access control audit | Monthly | Security Officer |
| Penetration testing | Quarterly | External Auditor |
| Key rotation | Annually | Key Custodians |
| DR drill | Monthly | Operations |
| Security training | Quarterly | All Staff |

# Voter Client Examples

This document provides examples for implementing voter clients across different platforms: web browsers, mobile applications, and command-line interfaces.

> **Note**: These examples show how to build voter clients using the `@digitaldefiance/node-fhe-accelerate` library. The library provides the core FHE and ZK proof functionality; you'll build your client application on top of it.

## Table of Contents

1. [Overview](#overview)
2. [Web Client (Browser)](#web-client-browser)
3. [Mobile Client (React Native)](#mobile-client-react-native)
4. [Command-Line Client](#command-line-client)
5. [API Reference](#api-reference)
6. [Security Considerations](#security-considerations)

## Overview

Voter clients interact with the FHE voting server to:
1. Authenticate the voter
2. Retrieve the election public key
3. Encrypt the ballot locally
4. Generate ZK proofs of ballot validity
5. Submit the encrypted ballot with proofs
6. Receive and verify the voting receipt

### Architecture

```
┌─────────────────┐     HTTPS/WSS      ┌─────────────────┐
│  Voter Client   │◄──────────────────►│  Voting Server  │
│  (Browser/App)  │                    │  (Mac Studio)   │
└─────────────────┘                    └─────────────────┘
        │                                      │
        │ Local Encryption                     │ FHE Operations
        │ ZK Proof Generation                  │ Tally Aggregation
        ▼                                      ▼
┌─────────────────┐                    ┌─────────────────┐
│  WebAssembly    │                    │  Native Addon   │
│  Crypto Module  │                    │  (M4 Max)       │
└─────────────────┘                    └─────────────────┘
```

## Web Client (Browser)

### Installation

The server-side uses the main library:

```bash
# Server-side (Node.js)
npm install @digitaldefiance/node-fhe-accelerate
```

For browser clients, you'll need to build a WebAssembly module from the library's crypto primitives, or use a thin client that delegates encryption to the server.

### Basic Usage (Server-Side API)

The following shows how to use the library's voting APIs on the server side:

```typescript
import { 
  VotingSystem,
  BulletproofsProver,
  type ElectionConfig 
} from '@digitaldefiance/node-fhe-accelerate';

// Create election on server
const election = await VotingSystem.create({
  electionId: 'election-2024',
  name: 'Board Election 2024',
  candidates: ['Alice', 'Bob', 'Charlie'],
  thresholdConfig: { threshold: 3, totalOfficials: 5 },
});

// Register voter
await election.registerVoter('voter-123', voterPublicKeyHash);

// Start election
await election.startElection();

// Submit ballot (called from client via API)
const ballot = await election.submitBallot('voter-123', 0); // Vote for Alice
console.log('Validity proof:', ballot.validityProof);
console.log('Eligibility proof:', ballot.eligibilityProof);

// Verify ballot proofs
const verification = await election.verifyBallot(ballot);
console.log('Valid:', verification.validityValid && verification.eligibilityValid);
```

### Complete Web Application Example

The following shows a complete web voting interface. Note that in a production system, the encryption and proof generation would happen either:
1. Via WebAssembly compiled from the library's Rust code
2. On the server side with the client sending plaintext choices over a secure channel

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FHE Voting</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px; }
    .candidate { padding: 15px; margin: 10px 0; border: 2px solid #ddd; border-radius: 8px; cursor: pointer; }
    .candidate:hover { border-color: #007bff; }
    .candidate.selected { border-color: #28a745; background: #e8f5e9; }
    button { padding: 15px 30px; font-size: 18px; cursor: pointer; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
    .status.success { background: #d4edda; color: #155724; }
    .status.error { background: #f8d7da; color: #721c24; }
    .status.info { background: #cce5ff; color: #004085; }
  </style>
</head>
<body>
  <h1>🗳️ Secure Voting</h1>
  
  <div id="election-info"></div>
  <div id="candidates"></div>
  <div id="status"></div>
  
  <button id="submit-btn" disabled>Submit Vote</button>
  <button id="verify-btn" style="display:none">Verify Receipt</button>
  
  <div id="receipt" style="display:none">
    <h3>Your Receipt</h3>
    <pre id="receipt-data"></pre>
  </div>

  <script type="module">
    // In production, this would use a WebAssembly build of the crypto primitives
    // or communicate with a server API that uses @digitaldefiance/node-fhe-accelerate
    
    // Example API client (you would implement this)
    class VotingApiClient {
      constructor(serverUrl, electionId) {
        this.serverUrl = serverUrl;
        this.electionId = electionId;
      }
      
      async connect() {
        const response = await fetch(`${this.serverUrl}/api/elections/${this.electionId}`);
        return response.json();
      }
      
      async hasVoted() {
        const response = await fetch(`${this.serverUrl}/api/elections/${this.electionId}/has-voted`, {
          credentials: 'include'
        });
        const data = await response.json();
        return data.hasVoted;
      }
      
      async getElectionInfo() {
        const response = await fetch(`${this.serverUrl}/api/elections/${this.electionId}`);
        return response.json();
      }
      
      async submitBallot(choice) {
        const response = await fetch(`${this.serverUrl}/api/elections/${this.electionId}/vote`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ choice })
        });
        return response.json();
      }
      
      async verifyReceipt(receipt) {
        const response = await fetch(`${this.serverUrl}/api/verify-receipt`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(receipt)
        });
        const data = await response.json();
        return data.valid;
      }
    }
    
    let client;
    let election;
    let selectedCandidate = null;
    let receipt = null;
    
    async function init() {
      showStatus('Connecting to voting server...', 'info');
      
      client = new VotingApiClient(
        'https://voting.example.com',
        'election-2024'
      );
      
      await client.connect();
      
      // Check if already voted
      const hasVoted = await client.hasVoted();
      if (hasVoted) {
        showStatus('You have already voted in this election.', 'info');
        return;
      }
      
      election = await client.getElectionInfo();
      renderElection();
      showStatus('Ready to vote', 'success');
    }
    
    function renderElection() {
      document.getElementById('election-info').innerHTML = `
        <h2>${election.name}</h2>
        <p>Select your candidate:</p>
      `;
      
      const candidatesDiv = document.getElementById('candidates');
      candidatesDiv.innerHTML = election.candidates.map((name, i) => `
        <div class="candidate" data-index="${i}">
          <strong>${name}</strong>
        </div>
      `).join('');
      
      candidatesDiv.querySelectorAll('.candidate').forEach(el => {
        el.addEventListener('click', () => selectCandidate(parseInt(el.dataset.index)));
      });
    }
    
    function selectCandidate(index) {
      selectedCandidate = index;
      document.querySelectorAll('.candidate').forEach((el, i) => {
        el.classList.toggle('selected', i === index);
      });
      document.getElementById('submit-btn').disabled = false;
    }
    
    async function submitVote() {
      if (selectedCandidate === null) return;
      
      const submitBtn = document.getElementById('submit-btn');
      submitBtn.disabled = true;
      
      try {
        showStatus('Submitting your vote (encryption happens server-side)...', 'info');
        
        // In this example, the server handles encryption and proof generation
        // using @digitaldefiance/node-fhe-accelerate
        receipt = await client.submitBallot(selectedCandidate);
        
        showStatus('Vote submitted successfully!', 'success');
        
        document.getElementById('receipt').style.display = 'block';
        document.getElementById('receipt-data').textContent = JSON.stringify({
          receiptId: receipt.receiptId,
          timestamp: receipt.timestamp,
          commitment: receipt.voteCommitment,
        }, null, 2);
        
        document.getElementById('verify-btn').style.display = 'inline-block';
        document.getElementById('candidates').style.display = 'none';
        submitBtn.style.display = 'none';
        
      } catch (error) {
        showStatus(`Error: ${error.message}`, 'error');
        submitBtn.disabled = false;
      }
    }
    
    async function verifyReceipt() {
      if (!receipt) return;
      
      showStatus('Verifying receipt...', 'info');
      
      try {
        const valid = await client.verifyReceipt(receipt);
        showStatus(valid ? 'Receipt verified ✓' : 'Receipt verification failed', valid ? 'success' : 'error');
      } catch (error) {
        showStatus(`Verification error: ${error.message}`, 'error');
      }
    }
    
    function showStatus(message, type) {
      const statusDiv = document.getElementById('status');
      statusDiv.className = `status ${type}`;
      statusDiv.textContent = message;
    }
    
    document.getElementById('submit-btn').addEventListener('click', submitVote);
    document.getElementById('verify-btn').addEventListener('click', verifyReceipt);
    
    init().catch(err => showStatus(`Initialization error: ${err.message}`, 'error'));
  </script>
</body>
</html>
```

### WebAssembly Crypto Module

The client uses WebAssembly for local encryption and proof generation:

```typescript
// wasm-crypto.ts - WebAssembly bindings for client-side crypto

import init, { 
  encrypt_ballot, 
  generate_validity_proof,
  verify_receipt 
} from '@digitaldefiance/fhe-wasm';

export class WasmCrypto {
  private initialized = false;
  
  async initialize(): Promise<void> {
    if (this.initialized) return;
    await init();
    this.initialized = true;
  }
  
  async encryptBallot(
    choice: number,
    publicKey: Uint8Array,
    randomness: Uint8Array
  ): Promise<Uint8Array> {
    await this.initialize();
    return encrypt_ballot(choice, publicKey, randomness);
  }
  
  async generateValidityProof(
    choice: number,
    numCandidates: number,
    randomness: Uint8Array
  ): Promise<Uint8Array> {
    await this.initialize();
    return generate_validity_proof(choice, numCandidates, randomness);
  }
  
  async verifyReceipt(
    receipt: Uint8Array,
    serverPublicKey: Uint8Array
  ): Promise<boolean> {
    await this.initialize();
    return verify_receipt(receipt, serverPublicKey);
  }
}
```

## Mobile Client (React Native)

### Installation

For React Native, you'll need to create API bindings to your voting server which uses `@digitaldefiance/node-fhe-accelerate`:

```bash
# Your React Native project
npm install axios  # or your preferred HTTP client
```

### React Native Component

This example shows a mobile voting interface that communicates with a server running the FHE library:

```tsx
// VotingScreen.tsx
import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Alert,
  ActivityIndicator,
} from 'react-native';

// API client for communicating with the voting server
class VotingApiClient {
  constructor(private serverUrl: string, private electionId: string) {}
  
  async connect(): Promise<void> {
    await fetch(`${this.serverUrl}/api/elections/${this.electionId}`);
  }
  
  async hasVoted(): Promise<boolean> {
    const response = await fetch(
      `${this.serverUrl}/api/elections/${this.electionId}/has-voted`,
      { credentials: 'include' }
    );
    const data = await response.json();
    return data.hasVoted;
  }
  
  async getElectionInfo(): Promise<any> {
    const response = await fetch(`${this.serverUrl}/api/elections/${this.electionId}`);
    return response.json();
  }
  
  async submitBallot(choice: number): Promise<any> {
    const response = await fetch(
      `${this.serverUrl}/api/elections/${this.electionId}/vote`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ choice })
      }
    );
    return response.json();
  }
  
  async verifyReceipt(receipt: any): Promise<boolean> {
    const response = await fetch(`${this.serverUrl}/api/verify-receipt`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(receipt)
    });
    const data = await response.json();
    return data.valid;
  }
}

interface Candidate {
  index: number;
  name: string;
}

export const VotingScreen: React.FC = () => {
  const [client, setClient] = useState<VotingApiClient | null>(null);
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [selectedCandidate, setSelectedCandidate] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [receipt, setReceipt] = useState<any>(null);
  const [status, setStatus] = useState('Connecting...');

  useEffect(() => {
    initializeClient();
  }, []);

  const initializeClient = async () => {
    try {
      const votingClient = new VotingApiClient(
        'https://voting.example.com',
        'election-2024'
      );

      await votingClient.connect();
      
      const hasVoted = await votingClient.hasVoted();
      if (hasVoted) {
        setStatus('You have already voted');
        setLoading(false);
        return;
      }

      const election = await votingClient.getElectionInfo();
      setCandidates(election.candidates.map((name: string, index: number) => ({ index, name })));
      setClient(votingClient);
      setStatus('Ready to vote');
      setLoading(false);
    } catch (error: any) {
      setStatus(`Error: ${error.message}`);
      setLoading(false);
    }
  };

  const submitVote = async () => {
    if (!client || selectedCandidate === null) return;

    setSubmitting(true);
    setStatus('Submitting vote...');

    try {
      // Server handles encryption and proof generation using
      // @digitaldefiance/node-fhe-accelerate
      const voteReceipt = await client.submitBallot(selectedCandidate);

      setReceipt(voteReceipt);
      setStatus('Vote submitted successfully!');
      
      Alert.alert(
        'Success',
        'Your vote has been submitted and encrypted.',
        [{ text: 'OK' }]
      );
    } catch (error: any) {
      setStatus(`Error: ${error.message}`);
      Alert.alert('Error', error.message);
    } finally {
      setSubmitting(false);
    }
  };

  const verifyReceipt = async () => {
    if (!client || !receipt) return;

    setStatus('Verifying receipt...');
    try {
      const valid = await client.verifyReceipt(receipt);
      Alert.alert(
        valid ? 'Verified' : 'Invalid',
        valid ? 'Your receipt is valid.' : 'Receipt verification failed.'
      );
      setStatus(valid ? 'Receipt verified ✓' : 'Verification failed');
    } catch (error: any) {
      Alert.alert('Error', error.message);
    }
  };

  if (loading) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#007bff" />
        <Text style={styles.status}>{status}</Text>
      </View>
    );
  }

  if (receipt) {
    return (
      <View style={styles.container}>
        <Text style={styles.title}>🗳️ Vote Submitted</Text>
        <Text style={styles.status}>{status}</Text>
        <View style={styles.receiptBox}>
          <Text style={styles.receiptTitle}>Your Receipt</Text>
          <Text style={styles.receiptText}>
            ID: {receipt.receiptId.substring(0, 16)}...
          </Text>
          <Text style={styles.receiptText}>
            Time: {new Date(receipt.timestamp).toLocaleString()}
          </Text>
        </View>
        <TouchableOpacity style={styles.button} onPress={verifyReceipt}>
          <Text style={styles.buttonText}>Verify Receipt</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Text style={styles.title}>🗳️ Cast Your Vote</Text>
      <Text style={styles.status}>{status}</Text>
      
      {candidates.map((candidate) => (
        <TouchableOpacity
          key={candidate.index}
          style={[
            styles.candidateCard,
            selectedCandidate === candidate.index && styles.selectedCard,
          ]}
          onPress={() => setSelectedCandidate(candidate.index)}
        >
          <Text style={styles.candidateName}>{candidate.name}</Text>
          {selectedCandidate === candidate.index && (
            <Text style={styles.checkmark}>✓</Text>
          )}
        </TouchableOpacity>
      ))}

      <TouchableOpacity
        style={[styles.button, (!selectedCandidate && selectedCandidate !== 0) && styles.buttonDisabled]}
        onPress={submitVote}
        disabled={selectedCandidate === null || submitting}
      >
        {submitting ? (
          <ActivityIndicator color="#fff" />
        ) : (
          <Text style={styles.buttonText}>Submit Vote</Text>
        )}
      </TouchableOpacity>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    backgroundColor: '#f5f5f5',
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 10,
  },
  status: {
    fontSize: 16,
    textAlign: 'center',
    color: '#666',
    marginBottom: 20,
  },
  candidateCard: {
    backgroundColor: '#fff',
    padding: 20,
    borderRadius: 10,
    marginBottom: 10,
    borderWidth: 2,
    borderColor: '#ddd',
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  selectedCard: {
    borderColor: '#28a745',
    backgroundColor: '#e8f5e9',
  },
  candidateName: {
    fontSize: 18,
    fontWeight: '500',
  },
  checkmark: {
    fontSize: 24,
    color: '#28a745',
  },
  button: {
    backgroundColor: '#007bff',
    padding: 18,
    borderRadius: 10,
    marginTop: 20,
    alignItems: 'center',
  },
  buttonDisabled: {
    backgroundColor: '#ccc',
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  receiptBox: {
    backgroundColor: '#fff',
    padding: 20,
    borderRadius: 10,
    marginVertical: 20,
  },
  receiptTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  receiptText: {
    fontSize: 14,
    color: '#666',
    marginBottom: 5,
  },
});
```

## Command-Line Client

### Installation

The library includes a verification CLI. For a full voting CLI, you can build one using the library:

```bash
# Install the main library (includes verification CLI)
npm install -g @digitaldefiance/node-fhe-accelerate

# The verification CLI is available as:
npx fhe-verify --help
```

### Usage (Verification CLI)

```bash
# Verify a complete verification package
fhe-verify --package verification-package.json

# Verify and generate HTML report
fhe-verify --package verification-package.json --report report.html --format html

# Verify just the audit trail
fhe-verify --audit audit_trail.json

# Verbose output
fhe-verify --package verification-package.json --verbose
```

### Building a Custom Voting CLI

Here's an example of building a voting CLI using the library:

```typescript
// voting-cli.ts
#!/usr/bin/env node

import { Command } from 'commander';
import { 
  PublicVerifier,
  generateHTMLReport,
  exportReportJSON 
} from '@digitaldefiance/node-fhe-accelerate';
import * as fs from 'fs';
import * as path from 'path';

const program = new Command();
const configPath = path.join(process.env.HOME || '', '.fhe-vote-config.json');

function loadConfig(): Record<string, string> {
  try {
    return JSON.parse(fs.readFileSync(configPath, 'utf-8'));
  } catch {
    return {};
  }
}

function saveConfig(config: Record<string, string>): void {
  fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
}

program
  .name('fhe-vote')
  .description('FHE Voting CLI Client')
  .version('1.0.0');

// Config commands
const configCmd = program.command('config').description('Manage configuration');

configCmd
  .command('set <key> <value>')
  .description('Set a configuration value')
  .action((key, value) => {
    const config = loadConfig();
    config[key] = value;
    saveConfig(config);
    console.log(`Set ${key} = ${value}`);
  });

configCmd
  .command('get <key>')
  .description('Get a configuration value')
  .action((key) => {
    const config = loadConfig();
    console.log(config[key] || '(not set)');
  });

// Verify command using the library's PublicVerifier
program
  .command('verify')
  .description('Verify election proofs')
  .requiredOption('-p, --package <file>', 'Verification package file')
  .option('-r, --report <file>', 'Output report file')
  .option('-f, --format <format>', 'Output format (text, json, html)', 'text')
  .action(async (options) => {
    const verifier = new PublicVerifier();
    
    console.log(`Loading verification package: ${options.package}`);
    const pkg = JSON.parse(fs.readFileSync(options.package, 'utf-8'));
    
    console.log('Running verification...\n');
    const report = await verifier.verifyAll(pkg, (progress) => {
      console.log(`${progress.stage}: ${progress.percent}%`);
    });
    
    // Output report
    if (options.report) {
      let output: string;
      switch (options.format) {
        case 'html':
          output = generateHTMLReport(report);
          break;
        case 'json':
          output = exportReportJSON(report);
          break;
        default:
          output = report.summary;
      }
      fs.writeFileSync(options.report, output);
      console.log(`Report saved to: ${options.report}`);
    }
    
    console.log('\n' + '='.repeat(60));
    console.log(`OVERALL: ${report.overallValid ? '✓ VERIFIED' : '✗ VERIFICATION FAILED'}`);
    console.log('='.repeat(60));
    
    process.exit(report.overallValid ? 0 : 1);
  });

program.parse();
```

## API Reference

The `@digitaldefiance/node-fhe-accelerate` library provides the following APIs for building voting applications:

### VotingSystem (Server-Side)

```typescript
import { VotingSystem } from '@digitaldefiance/node-fhe-accelerate';

// Create an election
const election = await VotingSystem.create({
  electionId: string;
  name: string;
  candidates: string[];
  thresholdConfig: { threshold: number; totalOfficials: number };
  expectedVoters?: number;
});

// Election lifecycle
await election.registerVoter(voterId: string, publicKeyHash: string);
await election.startElection();
const ballot = await election.submitBallot(voterId: string, choice: number);
const encryptedTally = await election.endElection();
const result = await election.decryptTally(officials: Official[]);

// Verification
const verification = await election.verifyBallot(ballot);
```

### PublicVerifier (Verification)

```typescript
import { PublicVerifier, generateHTMLReport } from '@digitaldefiance/node-fhe-accelerate';

const verifier = new PublicVerifier();

// Verify individual components
const configResult = verifier.verifyConfig(config);
const ballotResult = await verifier.verifyBallotProofs(proofs, numCandidates);
const eligibilityResult = await verifier.verifyEligibilityProofs(proofs, merkleRoot);
const tallyResult = verifier.verifyTallyProof(proof, finalTally);
const auditResult = verifier.verifyAuditTrail(entries);

// Full verification
const report = await verifier.verifyAll(verificationPackage, progressCallback);

// Generate reports
const html = generateHTMLReport(report);
```

### ZK Proof APIs

```typescript
import { 
  BulletproofsProver,
  Groth16Prover,
  PlonkProver 
} from '@digitaldefiance/node-fhe-accelerate';

// Bulletproofs for ballot validity
const bulletproofs = new BulletproofsProver({ useGpu: true });
const validityProof = await bulletproofs.proveBallotValidity(vote, numCandidates);
const result = await bulletproofs.verifyBallotValidity(proof);

// Groth16 for eligibility
const groth16 = new Groth16Prover(provingKey);
const eligibilityProof = await groth16.proveEligibility(leaf, path, indices, root);

// PLONK for tally correctness
const plonk = new PlonkProver(provingKey);
const tallyProof = await plonk.proveTallyCorrectness(ballots, initial, final, count);
```

### Audit Trail

```typescript
import { AuditTrailManager, createSystemActor } from '@digitaldefiance/node-fhe-accelerate';

const audit = new AuditTrailManager('election-2024');

// Log operations
await audit.logBallotSubmitted(ballot, actor);
await audit.logTallyComputed(tally, ballotCount, timeMs, createSystemActor());

// Verify and export
const verification = audit.verifyIntegrity();
const json = await audit.exportJSON();
const csv = await audit.exportCSV();
```

## Security Considerations

### Client-Side Security

1. **Randomness**: Use cryptographically secure random number generation
2. **Memory**: Clear sensitive data from memory after use
3. **Storage**: Never store unencrypted votes or private keys
4. **Transport**: Always use TLS 1.3 for server communication

### Proof Generation

1. **Local Generation**: All proofs are generated locally on the client
2. **No Vote Leakage**: Proofs reveal nothing about the vote content
3. **Verification**: Receipts can be verified without server trust

### Receipt Security

1. **Non-Transferable**: Receipts cannot prove how someone voted
2. **Verifiable**: Anyone can verify a receipt is valid
3. **Timestamped**: Receipts include cryptographic timestamps

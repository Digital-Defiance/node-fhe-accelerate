# Monitoring and Alerting

This document describes the monitoring and alerting infrastructure for the FHE voting system.

## Table of Contents

1. [Overview](#overview)
2. [Metrics Collection](#metrics-collection)
3. [Health Checks](#health-checks)
4. [Alerting Rules](#alerting-rules)
5. [Dashboard Setup](#dashboard-setup)
6. [Log Management](#log-management)
7. [Incident Response](#incident-response)

## Overview

The monitoring system provides real-time visibility into:
- System health and performance
- Election progress and statistics
- Security events and anomalies
- Hardware utilization (GPU, Neural Engine, etc.)

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Voting Server  │────►│   Prometheus    │────►│    Grafana      │
│  (Metrics)      │     │   (Storage)     │     │   (Dashboard)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │
        │                       ▼
        │               ┌─────────────────┐
        │               │  AlertManager   │
        │               │   (Alerts)      │
        │               └─────────────────┘
        │                       │
        ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│   Log Files     │     │  PagerDuty/     │
│   (JSON)        │     │  Slack/Email    │
└─────────────────┘     └─────────────────┘
```

## Metrics Collection

### Enable Metrics Endpoint

```typescript
import { VotingServer, MetricsCollector } from '@digitaldefiance/node-fhe-accelerate';

const server = new VotingServer({
  port: 3000,
  metrics: {
    enabled: true,
    port: 9090,
    path: '/metrics',
  },
});

// Custom metrics
const metrics = new MetricsCollector();

// Register custom metrics
metrics.registerCounter('ballots_submitted_total', 'Total ballots submitted');
metrics.registerHistogram('ballot_processing_seconds', 'Ballot processing time');
metrics.registerGauge('active_elections', 'Number of active elections');
```

### Available Metrics

#### System Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `fhe_cpu_usage_percent` | Gauge | CPU utilization |
| `fhe_memory_usage_bytes` | Gauge | Memory usage |
| `fhe_gpu_usage_percent` | Gauge | GPU utilization |
| `fhe_neural_engine_usage_percent` | Gauge | Neural Engine utilization |
| `fhe_unified_memory_bandwidth_gbps` | Gauge | Memory bandwidth |

#### Election Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `fhe_ballots_submitted_total` | Counter | Total ballots submitted |
| `fhe_ballots_verified_total` | Counter | Total ballots verified |
| `fhe_ballots_rejected_total` | Counter | Total ballots rejected |
| `fhe_ballot_processing_seconds` | Histogram | Ballot processing latency |
| `fhe_tally_computation_seconds` | Histogram | Tally computation time |
| `fhe_active_elections` | Gauge | Active elections count |
| `fhe_registered_voters` | Gauge | Registered voters per election |

#### Cryptographic Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `fhe_encryption_seconds` | Histogram | Encryption latency |
| `fhe_decryption_seconds` | Histogram | Decryption latency |
| `fhe_proof_generation_seconds` | Histogram | ZK proof generation time |
| `fhe_proof_verification_seconds` | Histogram | ZK proof verification time |
| `fhe_ntt_operations_total` | Counter | NTT operations performed |
| `fhe_bootstrap_operations_total` | Counter | Bootstrap operations |

#### Security Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `fhe_fraud_alerts_total` | Counter | Fraud alerts triggered |
| `fhe_duplicate_vote_attempts` | Counter | Duplicate vote attempts |
| `fhe_invalid_proof_submissions` | Counter | Invalid proof submissions |
| `fhe_authentication_failures` | Counter | Auth failures |


### Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

rule_files:
  - "alerts/*.yml"

scrape_configs:
  - job_name: 'fhe-voting'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: /metrics
    scheme: http

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
```

## Health Checks

### Health Endpoint

```typescript
// Health check endpoint implementation
server.get('/health', async (req, res) => {
  const health = await server.getHealthStatus();
  
  res.status(health.healthy ? 200 : 503).json({
    status: health.healthy ? 'healthy' : 'unhealthy',
    timestamp: new Date().toISOString(),
    version: process.env.npm_package_version,
    uptime: process.uptime(),
    checks: {
      database: health.database,
      hardware: health.hardware,
      crypto: health.crypto,
      network: health.network,
    },
  });
});
```

### Health Check Components

```typescript
interface HealthCheck {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  latencyMs: number;
  message?: string;
}

async function performHealthChecks(): Promise<HealthCheck[]> {
  return [
    await checkHardware(),
    await checkCrypto(),
    await checkNetwork(),
    await checkStorage(),
  ];
}

async function checkHardware(): Promise<HealthCheck> {
  const start = Date.now();
  const caps = await detectHardware();
  
  return {
    name: 'hardware',
    status: caps.sme && caps.metalGpu ? 'healthy' : 'degraded',
    latencyMs: Date.now() - start,
    message: `SME: ${caps.sme}, GPU: ${caps.metalGpu}`,
  };
}

async function checkCrypto(): Promise<HealthCheck> {
  const start = Date.now();
  
  try {
    // Perform test encryption/decryption
    const testResult = await performCryptoTest();
    return {
      name: 'crypto',
      status: testResult.success ? 'healthy' : 'unhealthy',
      latencyMs: Date.now() - start,
    };
  } catch (error) {
    return {
      name: 'crypto',
      status: 'unhealthy',
      latencyMs: Date.now() - start,
      message: error.message,
    };
  }
}
```

## Alerting Rules

### Critical Alerts

Create `alerts/critical.yml`:

```yaml
groups:
  - name: critical
    rules:
      - alert: VotingServerDown
        expr: up{job="fhe-voting"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Voting server is down"
          description: "The FHE voting server has been unreachable for more than 1 minute."

      - alert: HighErrorRate
        expr: rate(fhe_ballots_rejected_total[5m]) / rate(fhe_ballots_submitted_total[5m]) > 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High ballot rejection rate"
          description: "More than 10% of ballots are being rejected."

      - alert: FraudDetected
        expr: increase(fhe_fraud_alerts_total[5m]) > 0
        labels:
          severity: critical
        annotations:
          summary: "Fraud alert triggered"
          description: "A fraud detection alert has been triggered."

      - alert: CryptoFailure
        expr: fhe_health_crypto_status == 0
        for: 30s
        labels:
          severity: critical
        annotations:
          summary: "Cryptographic subsystem failure"
          description: "The cryptographic subsystem is not functioning correctly."
```

### Warning Alerts

Create `alerts/warning.yml`:

```yaml
groups:
  - name: warning
    rules:
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(fhe_ballot_processing_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High ballot processing latency"
          description: "95th percentile ballot processing time exceeds 1 second."

      - alert: HighMemoryUsage
        expr: fhe_memory_usage_bytes / fhe_memory_total_bytes > 0.85
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is above 85%."

      - alert: GPUNotAvailable
        expr: fhe_gpu_available == 0
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "GPU not available"
          description: "Metal GPU is not available for acceleration."

      - alert: LowThroughput
        expr: rate(fhe_ballots_submitted_total[5m]) < 100
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Low ballot throughput"
          description: "Ballot submission rate is below 100/minute."
```

### AlertManager Configuration

Create `alertmanager.yml`:

```yaml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical'
      continue: true
    - match:
        severity: warning
      receiver: 'warning'

receivers:
  - name: 'default'
    email_configs:
      - to: 'ops@election.gov'
        from: 'alerts@election.gov'
        smarthost: 'smtp.election.gov:587'

  - name: 'critical'
    pagerduty_configs:
      - service_key: '<pagerduty-service-key>'
    slack_configs:
      - api_url: '<slack-webhook-url>'
        channel: '#election-critical'
        title: '🚨 CRITICAL: {{ .GroupLabels.alertname }}'
        text: '{{ .CommonAnnotations.description }}'

  - name: 'warning'
    slack_configs:
      - api_url: '<slack-webhook-url>'
        channel: '#election-alerts'
        title: '⚠️ WARNING: {{ .GroupLabels.alertname }}'
        text: '{{ .CommonAnnotations.description }}'
```

## Dashboard Setup

### Grafana Dashboard

Import the following dashboard JSON:

```json
{
  "dashboard": {
    "title": "FHE Voting System",
    "panels": [
      {
        "title": "Ballots Submitted",
        "type": "stat",
        "targets": [
          {
            "expr": "fhe_ballots_submitted_total",
            "legendFormat": "Total Ballots"
          }
        ]
      },
      {
        "title": "Ballot Processing Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(fhe_ballots_submitted_total[1m])",
            "legendFormat": "Ballots/sec"
          }
        ]
      },
      {
        "title": "Processing Latency (p95)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(fhe_ballot_processing_seconds_bucket[5m]))",
            "legendFormat": "p95 Latency"
          }
        ]
      },
      {
        "title": "Hardware Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "fhe_cpu_usage_percent",
            "legendFormat": "CPU"
          },
          {
            "expr": "fhe_gpu_usage_percent",
            "legendFormat": "GPU"
          },
          {
            "expr": "fhe_neural_engine_usage_percent",
            "legendFormat": "Neural Engine"
          }
        ]
      },
      {
        "title": "Fraud Alerts",
        "type": "stat",
        "targets": [
          {
            "expr": "fhe_fraud_alerts_total",
            "legendFormat": "Total Alerts"
          }
        ]
      },
      {
        "title": "ZK Proof Generation Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(fhe_proof_generation_seconds_bucket[5m]))",
            "legendFormat": "p95 Proof Gen"
          }
        ]
      }
    ]
  }
}
```

## Log Management

### Structured Logging

```typescript
import { Logger } from '@digitaldefiance/node-fhe-accelerate';

const logger = new Logger({
  level: process.env.LOG_LEVEL || 'info',
  format: 'json',
  outputs: [
    { type: 'console' },
    { type: 'file', path: '/var/log/fhe-voting/app.log' },
  ],
});

// Log ballot submission
logger.info('ballot_submitted', {
  electionId: election.id,
  ballotId: ballot.id,
  processingTimeMs: processingTime,
  proofValid: proofResult.valid,
});

// Log security event
logger.warn('security_event', {
  type: 'duplicate_vote_attempt',
  voterId: voter.id,
  timestamp: new Date().toISOString(),
});

// Log error
logger.error('processing_error', {
  error: error.message,
  stack: error.stack,
  context: { ballotId, electionId },
});
```

### Log Rotation

Configure logrotate (`/etc/logrotate.d/fhe-voting`):

```
/var/log/fhe-voting/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 fhevoting staff
    postrotate
        kill -USR1 $(cat /var/run/fhe-voting.pid 2>/dev/null) 2>/dev/null || true
    endscript
}
```

## Incident Response

### Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| P1 | Critical | 5 minutes | Server down, data breach |
| P2 | High | 15 minutes | High error rate, fraud alert |
| P3 | Medium | 1 hour | Performance degradation |
| P4 | Low | 4 hours | Minor issues |

### Runbooks

#### Server Down (P1)

1. Check server status: `sudo launchctl list | grep fhevoting`
2. Check logs: `tail -100 /var/log/fhe-voting/stderr.log`
3. Restart service: `sudo launchctl kickstart -k system/com.digitaldefiance.fhevoting`
4. Verify health: `curl -k https://localhost/health`
5. If still down, failover to backup server

#### High Error Rate (P2)

1. Check error logs for patterns
2. Identify affected ballots
3. Check hardware status
4. Scale resources if needed
5. Notify election officials

#### Fraud Alert (P1)

1. Do NOT dismiss alert
2. Preserve all logs and evidence
3. Notify security team immediately
4. Document timeline
5. Follow fraud investigation procedure

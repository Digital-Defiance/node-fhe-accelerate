# FHE Voting System Documentation

This directory contains comprehensive documentation for deploying and operating the FHE voting system.

## Documentation Index

### Deployment & Setup

| Document | Description |
|----------|-------------|
| [Deployment Guide](./DEPLOYMENT_GUIDE.md) | Complete deployment instructions for Mac Studio/M4 Max |
| [Election Official Setup](./ELECTION_OFFICIAL_SETUP.md) | Step-by-step procedures for election officials |
| [Voter Client Examples](./VOTER_CLIENT_EXAMPLES.md) | Web, mobile, and CLI client implementations |

### Operations

| Document | Description |
|----------|-------------|
| [Monitoring & Alerting](./MONITORING_ALERTING.md) | Metrics, health checks, and alerting configuration |
| [Disaster Recovery](./DISASTER_RECOVERY.md) | Backup, recovery, and failover procedures |

### Security

| Document | Description |
|----------|-------------|
| [Security Best Practices](./SECURITY_BEST_PRACTICES.md) | Security guidelines and hardening procedures |
| [ZK Proof Verification](./ZK_PROOF_VERIFICATION.md) | Public verification tools and procedures |

## Quick Links

### For Election Officials

1. Start with [Deployment Guide](./DEPLOYMENT_GUIDE.md) to set up the server
2. Follow [Election Official Setup](./ELECTION_OFFICIAL_SETUP.md) for key ceremony and election configuration
3. Review [Security Best Practices](./SECURITY_BEST_PRACTICES.md) before going live
4. Set up [Monitoring & Alerting](./MONITORING_ALERTING.md) for election day

### For Developers

1. See [Voter Client Examples](./VOTER_CLIENT_EXAMPLES.md) for client implementation
2. Review the [API documentation](../src/api/) for TypeScript interfaces
3. Check [ZK Proof Verification](./ZK_PROOF_VERIFICATION.md) for verification tool development

### For Auditors

1. Start with [ZK Proof Verification](./ZK_PROOF_VERIFICATION.md) for verification procedures
2. Review [Security Best Practices](./SECURITY_BEST_PRACTICES.md) for security audit
3. Check [Monitoring & Alerting](./MONITORING_ALERTING.md) for audit trail access

## System Requirements

### Server (Mac Studio/M4 Max)

- macOS Sequoia 15.2+
- Apple M4 Max processor
- 64 GB+ Unified Memory
- 1 TB+ SSD storage
- 10 Gbps network

### Client (Voter)

- Any modern web browser (Chrome, Safari, Firefox)
- Or: iOS 15+ / Android 10+ for mobile
- Or: Node.js 18+ for CLI

## Support

- **Technical Support**: support@digitaldefiance.org
- **Security Issues**: security@digitaldefiance.org
- **Documentation Issues**: Open a GitHub issue

## Version

This documentation is for `@digitaldefiance/node-fhe-accelerate` version 1.0.0.

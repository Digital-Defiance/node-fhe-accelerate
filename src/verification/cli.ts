#!/usr/bin/env node
/**
 * @file verification/cli.ts
 * @brief Command-line interface for public election verification
 *
 * Usage:
 *   fhe-verify --package verification-package.zip
 *   fhe-verify --ballots proofs/ballots/
 *   fhe-verify --tally proofs/tally.bin
 *   fhe-verify --audit audit_trail.json
 *
 * Requirements: 17, 19
 */

import * as fs from 'fs';
import {
  PublicVerifier,
  generateHTMLReport,
  exportReportJSON,
  type VerificationPackage,
  type VerificationReport,
} from './public-verifier';
import type { AuditEntry } from '../api/audit-trail';

// ============================================================================
// CLI Implementation
// ============================================================================

interface CLIOptions {
  package: string | undefined;
  ballots: string | undefined;
  tally: string | undefined;
  audit: string | undefined;
  report: string | undefined;
  format: 'text' | 'json' | 'html' | undefined;
  verbose: boolean | undefined;
}

function parseArgs(args: string[]): CLIOptions {
  const options: CLIOptions = {
    package: undefined,
    ballots: undefined,
    tally: undefined,
    audit: undefined,
    report: undefined,
    format: undefined,
    verbose: undefined,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    const nextArg = args[i + 1];

    switch (arg) {
      case '--package':
      case '-p':
        options.package = nextArg;
        i++;
        break;
      case '--ballots':
      case '-b':
        options.ballots = nextArg;
        i++;
        break;
      case '--tally':
      case '-t':
        options.tally = nextArg;
        i++;
        break;
      case '--audit':
      case '-a':
        options.audit = nextArg;
        i++;
        break;
      case '--report':
      case '-r':
        options.report = nextArg;
        i++;
        break;
      case '--format':
      case '-f':
        options.format = nextArg as 'text' | 'json' | 'html';
        i++;
        break;
      case '--verbose':
      case '-v':
        options.verbose = true;
        break;
      case '--help':
      case '-h':
        printHelp();
        process.exit(0);
    }
  }

  return options;
}

function printHelp(): void {
  console.log(`
FHE Voting Verification Tool

Usage:
  fhe-verify [options]

Options:
  -p, --package <file>    Verify complete verification package (ZIP)
  -b, --ballots <dir>     Verify ballot proofs in directory
  -t, --tally <file>      Verify tally correctness proof
  -a, --audit <file>      Verify audit trail
  -r, --report <file>     Output report to file
  -f, --format <format>   Output format: text, json, html (default: text)
  -v, --verbose           Verbose output
  -h, --help              Show this help message

Examples:
  fhe-verify --package verification-package.zip
  fhe-verify --package verification-package.zip --report report.html --format html
  fhe-verify --ballots proofs/ballots/ --verbose
  fhe-verify --audit audit_trail.json
`);
}

function printProgress(stage: string, current: number, total: number, percent: number): void {
  const bar = '█'.repeat(Math.floor(percent / 5)) + '░'.repeat(20 - Math.floor(percent / 5));
  process.stdout.write(`\r${stage}: [${bar}] ${percent}% (${current}/${total})`);
  if (percent === 100) {
    console.log('');
  }
}

function loadVerificationPackage(packagePath: string): VerificationPackage {
  // In a real implementation, this would unzip and parse the package
  // For now, we assume it's a JSON file
  const content = fs.readFileSync(packagePath, 'utf-8');
  return JSON.parse(content) as VerificationPackage;
}

function printResult(name: string, result: { valid: boolean; errors: string[] }): void {
  const icon = result.valid ? '✓' : '✗';
  const status = result.valid ? 'PASSED' : 'FAILED';
  console.log(`\n${icon} ${name}: ${status}`);
  if (result.errors.length > 0) {
    for (const error of result.errors) {
      console.log(`    ERROR: ${error}`);
    }
  }
}

function printTextReport(report: VerificationReport): void {
  console.log('\n' + '='.repeat(60));
  console.log('ELECTION VERIFICATION REPORT');
  console.log('='.repeat(60));

  console.log(`\nElection: ${report.electionName}`);
  console.log(`Election ID: ${report.electionId}`);
  console.log(`Verification Date: ${report.verificationDate.toISOString()}`);
  console.log(`Total Time: ${report.totalVerificationTimeMs}ms`);

  console.log('\n' + '-'.repeat(60));
  console.log('RESULTS');
  console.log('-'.repeat(60));

  printResult('Configuration', report.configVerification);
  printResult('Ballot Proofs', report.ballotVerification);
  printResult('Eligibility Proofs', report.eligibilityVerification);
  printResult('Tally Correctness', report.tallyVerification);
  printResult('Audit Trail', report.auditTrailVerification);

  console.log('\n' + '='.repeat(60));
  console.log(`OVERALL: ${report.overallValid ? '✓ VERIFIED' : '✗ VERIFICATION FAILED'}`);
  console.log('='.repeat(60) + '\n');
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    printHelp();
    process.exit(1);
  }

  const options = parseArgs(args);
  const verifier = new PublicVerifier();

  try {
    let report: VerificationReport | null = null;

    if (options.package !== undefined && options.package !== '') {
      console.log(`Loading verification package: ${options.package}`);
      const pkg = loadVerificationPackage(options.package);

      console.log('Running verification...\n');
      report = await verifier.verifyAll(pkg, (progress) => {
        if (options.verbose === true) {
          printProgress(progress.stage, progress.current, progress.total, progress.percent);
        }
      });
    } else if (options.audit !== undefined && options.audit !== '') {
      console.log(`Verifying audit trail: ${options.audit}`);
      const content = fs.readFileSync(options.audit, 'utf-8');
      const auditTrail = JSON.parse(content) as AuditEntry[];
      const result = verifier.verifyAuditTrail(auditTrail);

      console.log(`\nAudit Trail Verification: ${result.valid ? '✓ PASSED' : '✗ FAILED'}`);
      console.log(`Entries verified: ${String(result.details['entriesVerified'])}`);
      if (result.errors.length > 0) {
        console.log('Errors:');
        for (const error of result.errors) {
          console.log(`  - ${error}`);
        }
      }
      process.exit(result.valid ? 0 : 1);
    } else {
      console.error('No verification target specified. Use --help for usage.');
      process.exit(1);
    }

    if (report !== null) {
      // Output report
      if (options.report !== undefined && options.report !== '') {
        let output: string;
        const format = options.format ?? 'text';

        switch (format) {
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

      // Print to console
      switch (options.format) {
        case 'json':
          console.log(exportReportJSON(report));
          break;
        case 'html':
          console.log(generateHTMLReport(report));
          break;
        default:
          printTextReport(report);
      }

      process.exit(report.overallValid ? 0 : 1);
    }
  } catch (error) {
    console.error(`Error: ${(error as Error).message}`);
    if (options.verbose === true) {
      console.error((error as Error).stack);
    }
    process.exit(1);
  }
}

// Run if executed directly
if (require.main === module) {
  main().catch((error) => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

export { main, parseArgs, printHelp };

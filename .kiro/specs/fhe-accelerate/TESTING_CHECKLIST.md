# Testing Checklist - Quick Reference

## ⚠️ CRITICAL: Zero Tolerance for Errors

**For a voting system, every bug is a threat to democracy.**

## Pre-Merge Requirements (ALL must pass)

### ✅ Correctness
- [ ] All 17 correctness properties pass (1000+ iterations each)
- [ ] 100% code coverage (lines, branches, functions)
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All adversarial tests pass
- [ ] Cross-validation with SEAL matches
- [ ] Cross-validation with HElib matches
- [ ] Cross-validation with libsnark matches
- [ ] Cross-validation with dalek-cryptography matches
- [ ] NIST test vectors pass
- [ ] Academic paper test vectors pass

### ✅ Security
- [ ] No timing side-channels detected
- [ ] No memory side-channels detected
- [ ] All adversarial tests pass
- [ ] Fuzz testing complete (1 hour minimum)
- [ ] Malicious input rejection verified
- [ ] Chosen ciphertext attack resistance verified
- [ ] Double voting prevention verified
- [ ] Tampered tally detection verified

### ✅ Performance
- [ ] Encryption < 10ms
- [ ] Verification < 20ms
- [ ] Tally < 5s for 100K ballots
- [ ] ZK proof generation < 200ms
- [ ] ZK proof verification < 20ms
- [ ] 10,000+ ballots/second throughput
- [ ] 5x hardware speedup on M4 Max

### ✅ Reliability
- [ ] 24-hour stability test passes
- [ ] 1M+ ballot stress test passes
- [ ] 1000+ concurrent operations test passes
- [ ] No memory leaks detected
- [ ] No resource leaks detected
- [ ] Graceful error handling verified
- [ ] Recovery from failures verified

### ✅ Formal Verification
- [ ] Coq proofs for critical properties
- [ ] Cryptol specifications verified
- [ ] SMT solver verification complete
- [ ] Model checking complete

### ✅ Documentation
- [ ] Every test has comprehensive documentation
- [ ] Every property references requirements
- [ ] Every edge case is documented
- [ ] Every known bug has regression test
- [ ] Test report generated and reviewed

## 7-Layer Testing Pyramid

```
        ┌─────────────────────┐
        │  Formal Verification │  ← Mathematical proofs
        ├─────────────────────┤
        │  Cross-Validation    │  ← Compare with references
        ├─────────────────────┤
        │   Stress Tests       │  ← Scale and performance
        ├─────────────────────┤
        │ Adversarial Tests    │  ← Security validation
        ├─────────────────────┤
        │ Integration Tests    │  ← Component interactions
        ├─────────────────────┤
        │  Property Tests      │  ← Mathematical invariants
        ├─────────────────────┤
        │    Unit Tests        │  ← Individual functions
        └─────────────────────┘
```

## Critical Test Commands

```bash
# Run all tests (MUST pass before merge)
yarn test:all

# Run with coverage (MUST be 100%)
yarn test:coverage

# Run property tests (1000+ iterations)
yarn test:property --iterations=1000

# Run security tests
yarn test:security

# Run cross-validation
yarn test:cross-validate

# Run stress tests
yarn test:stress

# Run 24-hour stability
yarn test:stability

# Run formal verification
make verify-proofs

# Generate test report
yarn test:report
```

## Test Failure Protocol

### If ANY test fails:

1. **STOP** - Do not proceed
2. **INVESTIGATE** - Understand root cause
3. **FIX** - Correct the implementation
4. **VERIFY** - Run all tests again
5. **DOCUMENT** - Add regression test
6. **REVIEW** - Get code review
7. **MERGE** - Only after ALL tests pass

### If property test fails:

1. **EXAMINE** shrunk counterexample
2. **REPRODUCE** failure locally
3. **UNDERSTAND** why property violated
4. **FIX** implementation or property
5. **ADD** regression test for counterexample
6. **VERIFY** with 10,000 iterations

### If cross-validation fails:

1. **COMPARE** outputs in detail
2. **CHECK** if reference is correct
3. **VERIFY** test setup is correct
4. **FIX** implementation to match
5. **DOCUMENT** any intentional differences

## Performance Regression Detection

```typescript
// Track performance over time
const PERFORMANCE_BASELINES = {
    encryption: 10,        // ms
    verification: 20,      // ms
    tally_100k: 5000,     // ms
    zk_proof_gen: 200,    // ms
    zk_proof_verify: 20,  // ms
};

// Fail if performance regresses by >10%
test('No performance regression', () => {
    for (const [operation, baseline] of Object.entries(PERFORMANCE_BASELINES)) {
        const actual = measurePerformance(operation);
        expect(actual).toBeLessThan(baseline * 1.1);
    }
});
```

## Security Test Checklist

- [ ] Timing attack resistance verified
- [ ] Side-channel resistance verified
- [ ] Fault injection resistance verified
- [ ] Malicious voter attacks blocked
- [ ] Malicious server attacks detected
- [ ] Chosen ciphertext attacks blocked
- [ ] Double voting prevented
- [ ] Tampered tally detected
- [ ] Invalid proofs rejected
- [ ] ZK proofs don't leak witnesses

## Remember

**"In cryptography, almost correct is completely wrong."**

- One failing test = DO NOT MERGE
- <100% coverage = DO NOT MERGE
- Performance regression = DO NOT MERGE
- Security vulnerability = DO NOT MERGE
- Missing documentation = DO NOT MERGE

**Zero tolerance. Bulletproof testing. Unbreakable trust.**

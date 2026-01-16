/**
 * Property-Based Tests for Homomorphic Addition
 * 
 * Feature: fhe-accelerate, Property 7: Homomorphic Addition Correctness
 * Validates: Requirements 6.1, 6.4, 6.5
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as fc from 'fast-check';
import { PROPERTY_TEST_CONFIG } from './property-test-config';

// NTT-friendly primes for different polynomial degrees
const NTT_PRIMES: Record<number, bigint> = {
  8: 17n,
  256: 65537n,
  512: 12289n,
  1024: 132120577n,
};

// ============================================================================
// Mathematical Utilities
// ============================================================================

function modPow(base: bigint, exp: bigint, mod: bigint): bigint {
  let result = 1n;
  base = ((base % mod) + mod) % mod;
  while (exp > 0n) {
    if (exp % 2n === 1n) result = (result * base) % mod;
    exp = exp / 2n;
    base = (base * base) % mod;
  }
  return result;
}

function modInverse(a: bigint, m: bigint): bigint {
  let [old_r, r] = [a, m];
  let [old_s, s] = [1n, 0n];
  while (r !== 0n) {
    const quotient = old_r / r;
    [old_r, r] = [r, old_r - quotient * r];
    [old_s, s] = [s, old_s - quotient * s];
  }
  return ((old_s % m) + m) % m;
}

function findPsi(degree: number, modulus: bigint): bigint {
  const twoN = BigInt(degree) * 2n;
  if ((modulus - 1n) % twoN !== 0n) {
    throw new Error(`Modulus ${modulus} is not NTT-friendly for degree ${degree}`);
  }
  const exponent = (modulus - 1n) / twoN;
  for (let g = 2n; g < modulus; g++) {
    const psi = modPow(g, exponent, modulus);
    const psiN = modPow(psi, BigInt(degree), modulus);
    const psi2N = modPow(psi, twoN, modulus);
    if (psi2N === 1n && psiN === modulus - 1n) return psi;
  }
  throw new Error('Could not find primitive 2N-th root of unity');
}

function bitReverse(index: number, bits: number): number {
  let result = 0;
  for (let i = 0; i < bits; i++) {
    result = (result << 1) | (index & 1);
    index >>= 1;
  }
  return result;
}

function bitReversePermutation(coeffs: bigint[], n: number): bigint[] {
  const result = [...coeffs];
  const bits = Math.log2(n);
  for (let i = 0; i < n; i++) {
    const j = bitReverse(i, bits);
    if (i < j) [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

// Caches for twiddle factors
const psiPowersCache = new Map<string, bigint[]>();
const psiInvPowersCache = new Map<string, bigint[]>();

function getPsiPowers(degree: number, modulus: bigint): bigint[] {
  const key = `${degree}-${modulus}`;
  if (!psiPowersCache.has(key)) {
    const psi = findPsi(degree, modulus);
    const powers: bigint[] = [1n];
    for (let i = 1; i < degree; i++) powers.push((powers[i - 1] * psi) % modulus);
    psiPowersCache.set(key, powers);
  }
  return psiPowersCache.get(key)!;
}

function getPsiInvPowers(degree: number, modulus: bigint): bigint[] {
  const key = `${degree}-${modulus}`;
  if (!psiInvPowersCache.has(key)) {
    const psi = findPsi(degree, modulus);
    const psiInv = modInverse(psi, modulus);
    const powers: bigint[] = [1n];
    for (let i = 1; i < degree; i++) powers.push((powers[i - 1] * psiInv) % modulus);
    psiInvPowersCache.set(key, powers);
  }
  return psiInvPowersCache.get(key)!;
}

// ============================================================================
// NTT Operations
// ============================================================================

function forwardNTT(coeffs: bigint[], modulus: bigint): bigint[] {
  const n = coeffs.length;
  const logN = Math.log2(n);
  const psiPowers = getPsiPowers(n, modulus);
  const psi = findPsi(n, modulus);
  const omega = (psi * psi) % modulus;
  let result = coeffs.map((c, i) => (c * psiPowers[i]) % modulus);
  const twiddles: bigint[] = [1n];
  for (let i = 1; i < n; i++) twiddles.push((twiddles[i - 1] * omega) % modulus);
  result = bitReversePermutation(result, n);
  for (let stage = 0; stage < logN; stage++) {
    const m = 1 << stage;
    const groupSize = 2 * m;
    for (let k = 0; k < n; k += groupSize) {
      for (let j = 0; j < m; j++) {
        const twiddleIdx = j * (n / groupSize);
        const w = twiddles[twiddleIdx];
        const idxA = k + j, idxB = k + j + m;
        const a = result[idxA], b = result[idxB];
        const wb = (w * b) % modulus;
        result[idxA] = (a + wb) % modulus;
        result[idxB] = ((a - wb) % modulus + modulus) % modulus;
      }
    }
  }
  return result;
}

function inverseNTT(coeffs: bigint[], modulus: bigint): bigint[] {
  const n = coeffs.length;
  const logN = Math.log2(n);
  const psiInvPowers = getPsiInvPowers(n, modulus);
  const psi = findPsi(n, modulus);
  const omega = (psi * psi) % modulus;
  const omegaInv = modInverse(omega, modulus);
  const invTwiddles: bigint[] = [1n];
  for (let i = 1; i < n; i++) invTwiddles.push((invTwiddles[i - 1] * omegaInv) % modulus);
  let result = [...coeffs];
  for (let stage = logN - 1; stage >= 0; stage--) {
    const m = 1 << stage;
    const groupSize = 2 * m;
    for (let k = 0; k < n; k += groupSize) {
      for (let j = 0; j < m; j++) {
        const twiddleIdx = j * (n / groupSize);
        const wInv = invTwiddles[twiddleIdx];
        const idxA = k + j, idxB = k + j + m;
        const a = result[idxA], b = result[idxB];
        result[idxA] = (a + b) % modulus;
        const diff = ((a - b) % modulus + modulus) % modulus;
        result[idxB] = (diff * wInv) % modulus;
      }
    }
  }
  result = bitReversePermutation(result, n);
  const invN = modInverse(BigInt(n), modulus);
  result = result.map(c => (c * invN) % modulus);
  result = result.map((c, i) => (c * psiInvPowers[i]) % modulus);
  return result;
}

// ============================================================================
// Polynomial Operations
// ============================================================================

function pointwiseMultiply(a: bigint[], b: bigint[], modulus: bigint): bigint[] {
  return a.map((ai, i) => (ai * b[i]) % modulus);
}

function polyAdd(a: bigint[], b: bigint[], modulus: bigint): bigint[] {
  return a.map((ai, i) => (ai + b[i]) % modulus);
}

function polySub(a: bigint[], b: bigint[], modulus: bigint): bigint[] {
  return a.map((ai, i) => ((ai - b[i]) % modulus + modulus) % modulus);
}

function polyNegate(a: bigint[], modulus: bigint): bigint[] {
  return a.map(ai => (modulus - ai) % modulus);
}

// ============================================================================
// Sampling Functions
// ============================================================================

function sampleRandomPoly(degree: number, modulus: bigint): bigint[] {
  const maxVal = Number(modulus % BigInt(Number.MAX_SAFE_INTEGER));
  return Array.from({ length: degree }, () => BigInt(Math.floor(Math.random() * maxVal)));
}

function sampleTernaryPoly(degree: number, modulus: bigint): bigint[] {
  return Array.from({ length: degree }, () => {
    const r = Math.floor(Math.random() * 3);
    if (r === 0) return modulus - 1n;
    if (r === 1) return 0n;
    return 1n;
  });
}

function sampleErrorPoly(degree: number, modulus: bigint): bigint[] {
  return Array.from({ length: degree }, () => {
    const r = Math.floor(Math.random() * 3) - 1;
    return r < 0 ? (modulus + BigInt(r)) % modulus : BigInt(r);
  });
}

// ============================================================================
// Key Generation
// ============================================================================

function generateSecretKey(degree: number, modulus: bigint): bigint[] {
  return sampleTernaryPoly(degree, modulus);
}

function generatePublicKey(sk: bigint[], degree: number, modulus: bigint): { a: bigint[]; b: bigint[] } {
  const a = sampleRandomPoly(degree, modulus);
  const e = sampleErrorPoly(degree, modulus);
  const a_ntt = forwardNTT(a, modulus);
  const s_ntt = forwardNTT(sk, modulus);
  const as_ntt = pointwiseMultiply(a_ntt, s_ntt, modulus);
  const as = inverseNTT(as_ntt, modulus);
  const neg_as = as.map(c => (modulus - c) % modulus);
  const b = polyAdd(neg_as, e, modulus);
  return { a, b };
}

// ============================================================================
// Encoding/Decoding
// ============================================================================

function encodePlaintext(value: number, degree: number, modulus: bigint, plaintextModulus: number): bigint[] {
  const delta = modulus / BigInt(plaintextModulus);
  const coeffs: bigint[] = new Array(degree).fill(0n);
  coeffs[0] = (BigInt(value) * delta) % modulus;
  return coeffs;
}

function decodePlaintext(poly: bigint[], modulus: bigint, plaintextModulus: number): number {
  const coeff = poly[0];
  const t = BigInt(plaintextModulus);
  const numerator = coeff * t + modulus / 2n;
  return Number((numerator / modulus) % t);
}

// ============================================================================
// Encryption/Decryption
// ============================================================================

interface Ciphertext {
  c0: bigint[];
  c1: bigint[];
}

function encrypt(plaintext: bigint[], pk: { a: bigint[]; b: bigint[] }, degree: number, modulus: bigint): Ciphertext {
  const u = sampleTernaryPoly(degree, modulus);
  const e1 = sampleErrorPoly(degree, modulus);
  const e2 = sampleErrorPoly(degree, modulus);
  const u_ntt = forwardNTT(u, modulus);
  const pk_a_ntt = forwardNTT(pk.a, modulus);
  const pk_b_ntt = forwardNTT(pk.b, modulus);
  const bu_ntt = pointwiseMultiply(pk_b_ntt, u_ntt, modulus);
  const bu = inverseNTT(bu_ntt, modulus);
  let c0 = polyAdd(bu, e1, modulus);
  c0 = polyAdd(c0, plaintext, modulus);
  const au_ntt = pointwiseMultiply(pk_a_ntt, u_ntt, modulus);
  const au = inverseNTT(au_ntt, modulus);
  const c1 = polyAdd(au, e2, modulus);
  return { c0, c1 };
}

function decrypt(ct: Ciphertext, sk: bigint[], modulus: bigint): bigint[] {
  const c1_ntt = forwardNTT(ct.c1, modulus);
  const sk_ntt = forwardNTT(sk, modulus);
  const c1sk_ntt = pointwiseMultiply(c1_ntt, sk_ntt, modulus);
  const c1sk = inverseNTT(c1sk_ntt, modulus);
  return polyAdd(ct.c0, c1sk, modulus);
}

// ============================================================================
// Homomorphic Operations
// ============================================================================

/**
 * Homomorphic addition of two ciphertexts
 * 
 * Computes: ct_result = ct1 + ct2 where decrypt(ct_result) = decrypt(ct1) + decrypt(ct2)
 * 
 * Requirements: 6.1
 */
function addCiphertexts(ct1: Ciphertext, ct2: Ciphertext, modulus: bigint): Ciphertext {
  return {
    c0: polyAdd(ct1.c0, ct2.c0, modulus),
    c1: polyAdd(ct1.c1, ct2.c1, modulus),
  };
}

/**
 * Homomorphic subtraction of two ciphertexts
 * 
 * Computes: ct_result = ct1 - ct2 where decrypt(ct_result) = decrypt(ct1) - decrypt(ct2)
 */
function subtractCiphertexts(ct1: Ciphertext, ct2: Ciphertext, modulus: bigint): Ciphertext {
  return {
    c0: polySub(ct1.c0, ct2.c0, modulus),
    c1: polySub(ct1.c1, ct2.c1, modulus),
  };
}

/**
 * Homomorphic negation of a ciphertext
 * 
 * Computes: ct_result = -ct where decrypt(ct_result) = -decrypt(ct)
 */
function negateCiphertext(ct: Ciphertext, modulus: bigint): Ciphertext {
  return {
    c0: polyNegate(ct.c0, modulus),
    c1: polyNegate(ct.c1, modulus),
  };
}

/**
 * Add a plaintext to a ciphertext
 * 
 * Computes: ct_result = ct + pt where decrypt(ct_result) = decrypt(ct) + pt
 * 
 * Requirements: 6.2
 */
function addPlaintext(ct: Ciphertext, plaintext: bigint[], modulus: bigint): Ciphertext {
  return {
    c0: polyAdd(ct.c0, plaintext, modulus),
    c1: [...ct.c1], // c1 unchanged
  };
}

// ============================================================================
// Property Tests
// ============================================================================

describe('Property 7: Homomorphic Addition Correctness', () => {
  /**
   * **Validates: Requirements 6.1, 6.4, 6.5**
   */
  const degree = 256;
  const modulus = NTT_PRIMES[degree];
  const plaintextModulus = 4;
  
  let sk: bigint[];
  let pk: { a: bigint[]; b: bigint[] };
  
  beforeAll(() => {
    sk = generateSecretKey(degree, modulus);
    pk = generatePublicKey(sk, degree, modulus);
  });
  
  it('should satisfy decrypt(ct1 + ct2) = decrypt(ct1) + decrypt(ct2) for all plaintext pairs', () => {
    /**
     * **Validates: Requirements 6.1**
     * 
     * Property: Homomorphic addition preserves the sum of plaintexts
     * For all plaintexts p1, p2: decrypt(encrypt(p1) + encrypt(p2)) = p1 + p2 (mod t)
     */
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: plaintextModulus - 1 }),
        fc.integer({ min: 0, max: plaintextModulus - 1 }),
        (p1, p2) => {
          // Encrypt both plaintexts
          const encoded1 = encodePlaintext(p1, degree, modulus, plaintextModulus);
          const encoded2 = encodePlaintext(p2, degree, modulus, plaintextModulus);
          const ct1 = encrypt(encoded1, pk, degree, modulus);
          const ct2 = encrypt(encoded2, pk, degree, modulus);
          
          // Add ciphertexts homomorphically
          const ctSum = addCiphertexts(ct1, ct2, modulus);
          
          // Decrypt and verify
          const decrypted = decrypt(ctSum, sk, modulus);
          const recovered = decodePlaintext(decrypted, modulus, plaintextModulus);
          
          // Expected sum (mod plaintext modulus)
          const expectedSum = (p1 + p2) % plaintextModulus;
          
          return recovered === expectedSum;
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
    );
  });

  it('should satisfy commutativity: ct1 + ct2 = ct2 + ct1', () => {
    /**
     * **Validates: Requirements 6.4**
     * 
     * Property: Homomorphic addition is commutative
     * For all ciphertexts ct1, ct2: ct1 + ct2 = ct2 + ct1
     */
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: plaintextModulus - 1 }),
        fc.integer({ min: 0, max: plaintextModulus - 1 }),
        (p1, p2) => {
          const encoded1 = encodePlaintext(p1, degree, modulus, plaintextModulus);
          const encoded2 = encodePlaintext(p2, degree, modulus, plaintextModulus);
          const ct1 = encrypt(encoded1, pk, degree, modulus);
          const ct2 = encrypt(encoded2, pk, degree, modulus);
          
          // ct1 + ct2
          const sum1 = addCiphertexts(ct1, ct2, modulus);
          const dec1 = decodePlaintext(decrypt(sum1, sk, modulus), modulus, plaintextModulus);
          
          // ct2 + ct1
          const sum2 = addCiphertexts(ct2, ct1, modulus);
          const dec2 = decodePlaintext(decrypt(sum2, sk, modulus), modulus, plaintextModulus);
          
          return dec1 === dec2;
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
    );
  });

  it('should satisfy identity: ct + encrypt(0) = ct', () => {
    /**
     * **Validates: Requirements 6.5**
     * 
     * Property: Encryption of zero is the additive identity
     * For all ciphertexts ct: ct + encrypt(0) decrypts to the same value as ct
     */
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: plaintextModulus - 1 }),
        (p) => {
          const encoded = encodePlaintext(p, degree, modulus, plaintextModulus);
          const ct = encrypt(encoded, pk, degree, modulus);
          
          // Encrypt zero
          const encodedZero = encodePlaintext(0, degree, modulus, plaintextModulus);
          const ctZero = encrypt(encodedZero, pk, degree, modulus);
          
          // ct + encrypt(0)
          const ctSum = addCiphertexts(ct, ctZero, modulus);
          
          // Decrypt both
          const decOriginal = decodePlaintext(decrypt(ct, sk, modulus), modulus, plaintextModulus);
          const decSum = decodePlaintext(decrypt(ctSum, sk, modulus), modulus, plaintextModulus);
          
          return decOriginal === decSum && decSum === p;
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
    );
  });

  it('should satisfy associativity: (ct1 + ct2) + ct3 = ct1 + (ct2 + ct3)', () => {
    /**
     * Property: Homomorphic addition is associative
     * For all ciphertexts ct1, ct2, ct3: (ct1 + ct2) + ct3 = ct1 + (ct2 + ct3)
     */
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: plaintextModulus - 1 }),
        fc.integer({ min: 0, max: plaintextModulus - 1 }),
        fc.integer({ min: 0, max: plaintextModulus - 1 }),
        (p1, p2, p3) => {
          const ct1 = encrypt(encodePlaintext(p1, degree, modulus, plaintextModulus), pk, degree, modulus);
          const ct2 = encrypt(encodePlaintext(p2, degree, modulus, plaintextModulus), pk, degree, modulus);
          const ct3 = encrypt(encodePlaintext(p3, degree, modulus, plaintextModulus), pk, degree, modulus);
          
          // (ct1 + ct2) + ct3
          const sum12 = addCiphertexts(ct1, ct2, modulus);
          const sum123_left = addCiphertexts(sum12, ct3, modulus);
          const dec_left = decodePlaintext(decrypt(sum123_left, sk, modulus), modulus, plaintextModulus);
          
          // ct1 + (ct2 + ct3)
          const sum23 = addCiphertexts(ct2, ct3, modulus);
          const sum123_right = addCiphertexts(ct1, sum23, modulus);
          const dec_right = decodePlaintext(decrypt(sum123_right, sk, modulus), modulus, plaintextModulus);
          
          // Both should equal (p1 + p2 + p3) mod t
          const expected = (p1 + p2 + p3) % plaintextModulus;
          
          return dec_left === dec_right && dec_left === expected;
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
    );
  });

  it('should correctly add plaintext to ciphertext', () => {
    /**
     * **Validates: Requirements 6.2**
     * 
     * Property: Ciphertext-plaintext addition is correct
     * For all plaintexts p1, p2: decrypt(encrypt(p1) + p2) = p1 + p2 (mod t)
     */
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: plaintextModulus - 1 }),
        fc.integer({ min: 0, max: plaintextModulus - 1 }),
        (p1, p2) => {
          const encoded1 = encodePlaintext(p1, degree, modulus, plaintextModulus);
          const ct = encrypt(encoded1, pk, degree, modulus);
          
          // Add plaintext p2 to ciphertext
          const encoded2 = encodePlaintext(p2, degree, modulus, plaintextModulus);
          const ctSum = addPlaintext(ct, encoded2, modulus);
          
          // Decrypt and verify
          const decrypted = decrypt(ctSum, sk, modulus);
          const recovered = decodePlaintext(decrypted, modulus, plaintextModulus);
          
          const expectedSum = (p1 + p2) % plaintextModulus;
          
          return recovered === expectedSum;
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
    );
  });

  it('should satisfy subtraction: decrypt(ct1 - ct2) = decrypt(ct1) - decrypt(ct2)', () => {
    /**
     * Property: Homomorphic subtraction is correct
     * For all plaintexts p1, p2: decrypt(encrypt(p1) - encrypt(p2)) = p1 - p2 (mod t)
     */
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: plaintextModulus - 1 }),
        fc.integer({ min: 0, max: plaintextModulus - 1 }),
        (p1, p2) => {
          const ct1 = encrypt(encodePlaintext(p1, degree, modulus, plaintextModulus), pk, degree, modulus);
          const ct2 = encrypt(encodePlaintext(p2, degree, modulus, plaintextModulus), pk, degree, modulus);
          
          const ctDiff = subtractCiphertexts(ct1, ct2, modulus);
          const decrypted = decrypt(ctDiff, sk, modulus);
          const recovered = decodePlaintext(decrypted, modulus, plaintextModulus);
          
          // Expected difference (mod plaintext modulus, handling negative)
          const expectedDiff = ((p1 - p2) % plaintextModulus + plaintextModulus) % plaintextModulus;
          
          return recovered === expectedDiff;
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
    );
  });

  it('should satisfy negation: decrypt(-ct) = -decrypt(ct)', () => {
    /**
     * Property: Homomorphic negation is correct
     * For all plaintexts p: decrypt(-encrypt(p)) = -p (mod t)
     */
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: plaintextModulus - 1 }),
        (p) => {
          const ct = encrypt(encodePlaintext(p, degree, modulus, plaintextModulus), pk, degree, modulus);
          
          const ctNeg = negateCiphertext(ct, modulus);
          const decrypted = decrypt(ctNeg, sk, modulus);
          const recovered = decodePlaintext(decrypted, modulus, plaintextModulus);
          
          // Expected negation (mod plaintext modulus)
          const expectedNeg = (plaintextModulus - p) % plaintextModulus;
          
          return recovered === expectedNeg;
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
    );
  });

  it('should handle edge cases: 0 + 0 = 0', () => {
    const encodedZero = encodePlaintext(0, degree, modulus, plaintextModulus);
    const ct1 = encrypt(encodedZero, pk, degree, modulus);
    const ct2 = encrypt(encodedZero, pk, degree, modulus);
    
    const ctSum = addCiphertexts(ct1, ct2, modulus);
    const decrypted = decrypt(ctSum, sk, modulus);
    const recovered = decodePlaintext(decrypted, modulus, plaintextModulus);
    
    expect(recovered).toBe(0);
  });

  it('should handle edge cases: max + max with overflow', () => {
    const maxVal = plaintextModulus - 1;
    const encodedMax = encodePlaintext(maxVal, degree, modulus, plaintextModulus);
    const ct1 = encrypt(encodedMax, pk, degree, modulus);
    const ct2 = encrypt(encodedMax, pk, degree, modulus);
    
    const ctSum = addCiphertexts(ct1, ct2, modulus);
    const decrypted = decrypt(ctSum, sk, modulus);
    const recovered = decodePlaintext(decrypted, modulus, plaintextModulus);
    
    // (max + max) mod t = (2 * (t-1)) mod t = (2t - 2) mod t = t - 2
    const expected = (2 * maxVal) % plaintextModulus;
    expect(recovered).toBe(expected);
  });

  it('should support multiple consecutive additions', () => {
    /**
     * Property: Multiple additions accumulate correctly
     */
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 0, max: plaintextModulus - 1 }), { minLength: 2, maxLength: 5 }),
        (plaintexts) => {
          // Encrypt all plaintexts
          const ciphertexts = plaintexts.map(p => 
            encrypt(encodePlaintext(p, degree, modulus, plaintextModulus), pk, degree, modulus)
          );
          
          // Sum all ciphertexts
          let ctSum = ciphertexts[0];
          for (let i = 1; i < ciphertexts.length; i++) {
            ctSum = addCiphertexts(ctSum, ciphertexts[i], modulus);
          }
          
          // Decrypt and verify
          const decrypted = decrypt(ctSum, sk, modulus);
          const recovered = decodePlaintext(decrypted, modulus, plaintextModulus);
          
          // Expected sum
          const expectedSum = plaintexts.reduce((a, b) => a + b, 0) % plaintextModulus;
          
          return recovered === expectedSum;
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
    );
  });
});

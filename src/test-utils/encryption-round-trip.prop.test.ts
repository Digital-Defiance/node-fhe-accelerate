/**
 * Property-Based Tests for Encryption/Decryption Round-Trip
 * Validates: Requirements 5.5, 5.2
 */

import { describe, it, expect, beforeAll } from 'vitest';
import * as fc from 'fast-check';
import { PROPERTY_TEST_CONFIG } from './property-test-config';

const NTT_PRIMES: Record<number, bigint> = {
  8: 17n,
  256: 65537n,
  512: 12289n,
  1024: 132120577n,
};

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
    throw new Error('Modulus is not NTT-friendly');
  }
  const exponent = (modulus - 1n) / twoN;
  for (let g = 2n; g < modulus; g++) {
    const psi = modPow(g, exponent, modulus);
    const psiN = modPow(psi, BigInt(degree), modulus);
    const psi2N = modPow(psi, twoN, modulus);
    if (psi2N === 1n && psiN === modulus - 1n) return psi;
  }
  throw new Error('Could not find primitive root');
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

const psiPowersCache = new Map<string, bigint[]>();
const psiInvPowersCache = new Map<string, bigint[]>();

function getPsiPowers(degree: number, modulus: bigint): bigint[] {
  const key = degree + '-' + modulus;
  if (!psiPowersCache.has(key)) {
    const psi = findPsi(degree, modulus);
    const powers: bigint[] = [1n];
    for (let i = 1; i < degree; i++) powers.push((powers[i - 1] * psi) % modulus);
    psiPowersCache.set(key, powers);
  }
  return psiPowersCache.get(key)!;
}

function getPsiInvPowers(degree: number, modulus: bigint): bigint[] {
  const key = degree + '-' + modulus;
  if (!psiInvPowersCache.has(key)) {
    const psi = findPsi(degree, modulus);
    const psiInv = modInverse(psi, modulus);
    const powers: bigint[] = [1n];
    for (let i = 1; i < degree; i++) powers.push((powers[i - 1] * psiInv) % modulus);
    psiInvPowersCache.set(key, powers);
  }
  return psiInvPowersCache.get(key)!;
}

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

function pointwiseMultiply(a: bigint[], b: bigint[], modulus: bigint): bigint[] {
  return a.map((ai, i) => (ai * b[i]) % modulus);
}

function polyAdd(a: bigint[], b: bigint[], modulus: bigint): bigint[] {
  return a.map((ai, i) => (ai + b[i]) % modulus);
}

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

function encrypt(plaintext: bigint[], pk: { a: bigint[]; b: bigint[] }, degree: number, modulus: bigint): { c0: bigint[]; c1: bigint[] } {
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

function decrypt(ct: { c0: bigint[]; c1: bigint[] }, sk: bigint[], modulus: bigint): bigint[] {
  const c1_ntt = forwardNTT(ct.c1, modulus);
  const sk_ntt = forwardNTT(sk, modulus);
  const c1sk_ntt = pointwiseMultiply(c1_ntt, sk_ntt, modulus);
  const c1sk = inverseNTT(c1sk_ntt, modulus);
  return polyAdd(ct.c0, c1sk, modulus);
}

describe('Encryption Utility Functions', () => {
  const degree = 256;
  const modulus = NTT_PRIMES[degree];
  const plaintextModulus = 4;

  it('should verify NTT round-trip works', () => {
    const smallDegree = 8;
    const smallModulus = 17n;
    const a: bigint[] = [1n, 2n, 3n, 4n, 5n, 6n, 7n, 8n];
    const a_ntt = forwardNTT(a, smallModulus);
    const a_recovered = inverseNTT(a_ntt, smallModulus);
    for (let i = 0; i < smallDegree; i++) {
      expect(a_recovered[i]).toBe(a[i]);
    }
  });

  it('should correctly encode and decode plaintexts', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: plaintextModulus - 1 }),
        (value) => {
          const encoded = encodePlaintext(value, degree, modulus, plaintextModulus);
          const decoded = decodePlaintext(encoded, modulus, plaintextModulus);
          return decoded === value;
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
    );
  });

  it('should generate valid key pairs', () => {
    fc.assert(
      fc.property(
        fc.constant(null),
        () => {
          const sk = generateSecretKey(degree, modulus);
          const pk = generatePublicKey(sk, degree, modulus);
          if (sk.length !== degree) return false;
          if (pk.a.length !== degree) return false;
          if (pk.b.length !== degree) return false;
          for (const coeff of sk) {
            if (coeff < 0n || coeff >= modulus) return false;
          }
          return true;
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 10 }
    );
  });
});

describe('Property 6: Encryption/Decryption Round-Trip', () => {
  const degree = 256;
  const modulus = NTT_PRIMES[degree];
  const plaintextModulus = 4;
  
  let sk: bigint[];
  let pk: { a: bigint[]; b: bigint[] };
  
  beforeAll(() => {
    sk = generateSecretKey(degree, modulus);
    pk = generatePublicKey(sk, degree, modulus);
  });
  
  it('should satisfy encrypt(decrypt(m)) = m for all plaintexts', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: plaintextModulus - 1 }),
        (plaintext) => {
          const encoded = encodePlaintext(plaintext, degree, modulus, plaintextModulus);
          const ct = encrypt(encoded, pk, degree, modulus);
          const decrypted = decrypt(ct, sk, modulus);
          const recovered = decodePlaintext(decrypted, modulus, plaintextModulus);
          return recovered === plaintext;
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 100 }
    );
  });

  it('should preserve plaintext for SIMD-packed values', () => {
    fc.assert(
      fc.property(
        fc.array(fc.integer({ min: 0, max: plaintextModulus - 1 }), { minLength: 1, maxLength: 8 }),
        (plaintexts) => {
          const delta = modulus / BigInt(plaintextModulus);
          const encoded: bigint[] = new Array(degree).fill(0n);
          for (let i = 0; i < plaintexts.length && i < degree; i++) {
            encoded[i] = (BigInt(plaintexts[i]) * delta) % modulus;
          }
          const ct = encrypt(encoded, pk, degree, modulus);
          const decrypted = decrypt(ct, sk, modulus);
          for (let i = 0; i < plaintexts.length; i++) {
            const coeff = decrypted[i];
            const t = BigInt(plaintextModulus);
            const numerator = coeff * t + modulus / 2n;
            const recovered = Number((numerator / modulus) % t);
            if (recovered !== plaintexts[i]) return false;
          }
          return true;
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
    );
  });

  it('should produce different ciphertexts for same plaintext (semantic security)', () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: plaintextModulus - 1 }),
        (plaintext) => {
          const encoded = encodePlaintext(plaintext, degree, modulus, plaintextModulus);
          const ct1 = encrypt(encoded, pk, degree, modulus);
          const ct2 = encrypt(encoded, pk, degree, modulus);
          let different = false;
          for (let i = 0; i < degree; i++) {
            if (ct1.c0[i] !== ct2.c0[i] || ct1.c1[i] !== ct2.c1[i]) {
              different = true;
              break;
            }
          }
          const dec1 = decodePlaintext(decrypt(ct1, sk, modulus), modulus, plaintextModulus);
          const dec2 = decodePlaintext(decrypt(ct2, sk, modulus), modulus, plaintextModulus);
          return different && dec1 === plaintext && dec2 === plaintext;
        }
      ),
      { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
    );
  });

  it('should handle edge case plaintexts (0 and max)', () => {
    const encoded0 = encodePlaintext(0, degree, modulus, plaintextModulus);
    const ct0 = encrypt(encoded0, pk, degree, modulus);
    const dec0 = decodePlaintext(decrypt(ct0, sk, modulus), modulus, plaintextModulus);
    expect(dec0).toBe(0);
    
    const maxPt = plaintextModulus - 1;
    const encodedMax = encodePlaintext(maxPt, degree, modulus, plaintextModulus);
    const ctMax = encrypt(encodedMax, pk, degree, modulus);
    const decMax = decodePlaintext(decrypt(ctMax, sk, modulus), modulus, plaintextModulus);
    expect(decMax).toBe(maxPt);
  });
});

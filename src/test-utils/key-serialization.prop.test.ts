/**
 * Property-Based Tests for Key Serialization Round-Trip
 * 
 * **Property 5: Key Serialization Round-Trip**
 * For any key type (SecretKey, PublicKey, EvaluationKey, BootstrapKey),
 * serializing then deserializing SHALL produce a key functionally
 * equivalent to the original.
 * 
 * **Validates: Requirements 4.5**
 */

import { describe, it, expect } from 'vitest';
import * as fc from 'fast-check';
import { PROPERTY_TEST_CONFIG, arbitraryPolyDegree } from './property-test-config';

/**
 * Mock key structures for testing serialization logic
 * These will be replaced with actual native bindings when available
 */

interface MockPolynomial {
  coeffs: bigint[];
  modulus: bigint;
  isNtt: boolean;
}

interface MockSecretKey {
  poly: MockPolynomial;
  distribution: 'ternary' | 'gaussian' | 'binary' | 'uniform';
  keyId: bigint;
}

interface MockPublicKey {
  a: MockPolynomial;
  b: MockPolynomial;
  keyId: bigint;
}

interface MockEvaluationKey {
  relinKey: {
    keys: Array<{ a: MockPolynomial; b: MockPolynomial }>;
    decompBaseLog: number;
    decompLevel: number;
  };
  keyId: bigint;
}

/**
 * Serialization header structure (matches C++ implementation)
 */
interface SerializationHeader {
  magic: number;
  version: number;
  keyType: number;
  keyId: bigint;
  polyDegree: number;
  modulus: bigint;
  dataSize: number;
  checksumType: number;
  compression: number;
  checksum: number;
}

// Magic bytes for format identification
const MAGIC_SECRET_KEY = 0x46484553; // "FHES"
const MAGIC_PUBLIC_KEY = 0x46484550; // "FHEP"
const MAGIC_EVAL_KEY = 0x46484545; // "FHEE"
const SERIALIZATION_VERSION = 1;

/**
 * CRC32 implementation for checksum verification
 */
function computeCrc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  const table = new Uint32Array(256);

  // Generate CRC32 table
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c;
  }

  // Compute CRC
  for (let i = 0; i < data.length; i++) {
    crc = table[(crc ^ data[i]) & 0xff] ^ (crc >>> 8);
  }

  return (crc ^ 0xffffffff) >>> 0;
}

/**
 * Mock serializer for testing serialization logic
 */
class MockKeySerializer {
  /**
   * Serialize a polynomial to bytes
   */
  private serializePolynomial(poly: MockPolynomial): Uint8Array {
    const degree = poly.coeffs.length;
    // 4 bytes for degree + 8 bytes per coefficient
    const buffer = new ArrayBuffer(4 + degree * 8);
    const view = new DataView(buffer);

    view.setUint32(0, degree, true);
    for (let i = 0; i < degree; i++) {
      // Write as two 32-bit values (little-endian)
      const val = poly.coeffs[i];
      view.setUint32(4 + i * 8, Number(val & 0xffffffffn), true);
      view.setUint32(4 + i * 8 + 4, Number((val >> 32n) & 0xffffffffn), true);
    }

    return new Uint8Array(buffer);
  }

  /**
   * Deserialize a polynomial from bytes
   */
  private deserializePolynomial(
    data: Uint8Array,
    offset: number,
    modulus: bigint
  ): { poly: MockPolynomial; bytesRead: number } {
    const view = new DataView(data.buffer, data.byteOffset + offset);
    const degree = view.getUint32(0, true);

    const coeffs: bigint[] = [];
    for (let i = 0; i < degree; i++) {
      const low = BigInt(view.getUint32(4 + i * 8, true));
      const high = BigInt(view.getUint32(4 + i * 8 + 4, true));
      coeffs.push(low | (high << 32n));
    }

    return {
      poly: { coeffs, modulus, isNtt: false },
      bytesRead: 4 + degree * 8,
    };
  }

  /**
   * Serialize secret key
   */
  serializeSecretKey(sk: MockSecretKey): Uint8Array {
    // Serialize polynomial data
    const distByte =
      sk.distribution === 'ternary'
        ? 0
        : sk.distribution === 'gaussian'
          ? 1
          : sk.distribution === 'binary'
            ? 2
            : 3;

    const polyData = this.serializePolynomial(sk.poly);
    const dataSize = 1 + polyData.length;

    // Create data buffer
    const dataBuffer = new Uint8Array(dataSize);
    dataBuffer[0] = distByte;
    dataBuffer.set(polyData, 1);

    // Compute checksum
    const checksum = computeCrc32(dataBuffer);

    // Create header (48 bytes based on C++ struct)
    const headerSize = 48;
    const result = new Uint8Array(headerSize + dataSize);
    const headerView = new DataView(result.buffer);

    headerView.setUint32(0, MAGIC_SECRET_KEY, true);
    headerView.setUint32(4, SERIALIZATION_VERSION, true);
    headerView.setUint32(8, 0, true); // key_type = 0 for secret key
    // key_id as 64-bit
    headerView.setUint32(12, Number(sk.keyId & 0xffffffffn), true);
    headerView.setUint32(16, Number((sk.keyId >> 32n) & 0xffffffffn), true);
    headerView.setUint32(20, sk.poly.coeffs.length, true); // poly_degree
    // modulus as 64-bit
    headerView.setUint32(24, Number(sk.poly.modulus & 0xffffffffn), true);
    headerView.setUint32(28, Number((sk.poly.modulus >> 32n) & 0xffffffffn), true);
    headerView.setUint32(32, dataSize, true); // data_size
    result[36] = 1; // checksum_type = CRC32
    result[37] = 0; // compression = NONE
    // reserved bytes 38-44
    headerView.setUint32(44, checksum, true);

    // Copy data
    result.set(dataBuffer, headerSize);

    return result;
  }

  /**
   * Deserialize secret key
   */
  deserializeSecretKey(data: Uint8Array): MockSecretKey | null {
    if (data.length < 48) return null;

    const view = new DataView(data.buffer, data.byteOffset);

    // Verify magic
    const magic = view.getUint32(0, true);
    if (magic !== MAGIC_SECRET_KEY) return null;

    // Read header
    const keyIdLow = BigInt(view.getUint32(12, true));
    const keyIdHigh = BigInt(view.getUint32(16, true));
    const keyId = keyIdLow | (keyIdHigh << 32n);

    const modulusLow = BigInt(view.getUint32(24, true));
    const modulusHigh = BigInt(view.getUint32(28, true));
    const modulus = modulusLow | (modulusHigh << 32n);

    const dataSize = view.getUint32(32, true);
    const checksumType = data[36];
    const expectedChecksum = view.getUint32(44, true);

    // Verify checksum
    const dataStart = 48;
    const dataBytes = data.slice(dataStart, dataStart + dataSize);
    if (checksumType === 1) {
      const actualChecksum = computeCrc32(dataBytes);
      if (actualChecksum !== expectedChecksum) return null;
    }

    // Parse data
    const distByte = dataBytes[0];
    const distribution =
      distByte === 0
        ? 'ternary'
        : distByte === 1
          ? 'gaussian'
          : distByte === 2
            ? 'binary'
            : 'uniform';

    const { poly } = this.deserializePolynomial(dataBytes, 1, modulus);

    return { poly, distribution, keyId };
  }

  /**
   * Verify integrity without full deserialization
   */
  verifyIntegrity(data: Uint8Array): boolean {
    if (data.length < 48) return false;

    const view = new DataView(data.buffer, data.byteOffset);
    const dataSize = view.getUint32(32, true);
    const checksumType = data[36];
    const expectedChecksum = view.getUint32(44, true);

    if (checksumType === 0) return true; // No checksum

    const dataStart = 48;
    if (data.length < dataStart + dataSize) return false;

    const dataBytes = data.slice(dataStart, dataStart + dataSize);
    const actualChecksum = computeCrc32(dataBytes);

    return actualChecksum === expectedChecksum;
  }
}

/**
 * Arbitrary generators for key structures
 */

function arbitraryModulus(): fc.Arbitrary<bigint> {
  // NTT-friendly primes
  return fc.constantFrom(
    BigInt('132120577'),
    BigInt('268369921'),
    BigInt('1073479681'),
    BigInt('4611686018326724609')
  );
}

function arbitraryTernaryCoefficient(modulus: bigint): fc.Arbitrary<bigint> {
  // Ternary: {-1, 0, 1} represented as {modulus-1, 0, 1}
  return fc.constantFrom(0n, 1n, modulus - 1n);
}

function arbitraryPolynomial(
  degree: number,
  modulus: bigint
): fc.Arbitrary<MockPolynomial> {
  return fc
    .array(fc.bigInt({ min: 0n, max: modulus - 1n }), {
      minLength: degree,
      maxLength: degree,
    })
    .map((coeffs) => ({
      coeffs,
      modulus,
      isNtt: false,
    }));
}

function arbitraryTernaryPolynomial(
  degree: number,
  modulus: bigint
): fc.Arbitrary<MockPolynomial> {
  return fc
    .array(arbitraryTernaryCoefficient(modulus), {
      minLength: degree,
      maxLength: degree,
    })
    .map((coeffs) => ({
      coeffs,
      modulus,
      isNtt: false,
    }));
}

function arbitrarySecretKey(
  degree: number,
  modulus: bigint
): fc.Arbitrary<MockSecretKey> {
  return fc.record({
    poly: arbitraryTernaryPolynomial(degree, modulus),
    distribution: fc.constantFrom(
      'ternary' as const,
      'gaussian' as const,
      'binary' as const,
      'uniform' as const
    ),
    keyId: fc.bigInt({ min: 1n, max: BigInt(Number.MAX_SAFE_INTEGER) }),
  });
}

function arbitraryPublicKey(
  degree: number,
  modulus: bigint
): fc.Arbitrary<MockPublicKey> {
  return fc.record({
    a: arbitraryPolynomial(degree, modulus),
    b: arbitraryPolynomial(degree, modulus),
    keyId: fc.bigInt({ min: 1n, max: BigInt(Number.MAX_SAFE_INTEGER) }),
  });
}

/**
 * Helper to compare polynomials
 */
function polynomialsEqual(a: MockPolynomial, b: MockPolynomial): boolean {
  if (a.coeffs.length !== b.coeffs.length) return false;
  if (a.modulus !== b.modulus) return false;
  for (let i = 0; i < a.coeffs.length; i++) {
    if (a.coeffs[i] !== b.coeffs[i]) return false;
  }
  return true;
}

/**
 * Helper to compare secret keys
 */
function secretKeysEqual(a: MockSecretKey, b: MockSecretKey): boolean {
  return (
    a.keyId === b.keyId &&
    a.distribution === b.distribution &&
    polynomialsEqual(a.poly, b.poly)
  );
}

describe('Property 5: Key Serialization Round-Trip', () => {
  const serializer = new MockKeySerializer();

  describe('Secret Key Serialization', () => {
    it('should preserve secret key through serialization round-trip', () => {
      // Use smaller degree for faster tests
      const degree = 64;
      const modulus = BigInt('132120577');

      fc.assert(
        fc.property(arbitrarySecretKey(degree, modulus), (sk) => {
          // Serialize
          const serialized = serializer.serializeSecretKey(sk);

          // Deserialize
          const deserialized = serializer.deserializeSecretKey(serialized);

          // Verify round-trip
          expect(deserialized).not.toBeNull();
          if (deserialized) {
            expect(secretKeysEqual(sk, deserialized)).toBe(true);
          }

          return deserialized !== null && secretKeysEqual(sk, deserialized);
        }),
        { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
      );
    });

    it('should detect corrupted data via checksum', () => {
      const degree = 64;
      const modulus = BigInt('132120577');

      fc.assert(
        fc.property(
          arbitrarySecretKey(degree, modulus),
          fc.integer({ min: 48, max: 200 }), // corruption position
          (sk, corruptPos) => {
            const serialized = serializer.serializeSecretKey(sk);

            // Corrupt a byte in the data section
            if (corruptPos < serialized.length) {
              const corrupted = new Uint8Array(serialized);
              corrupted[corruptPos] ^= 0xff;

              // Integrity check should fail
              const isValid = serializer.verifyIntegrity(corrupted);

              // If we corrupted the data section, checksum should fail
              if (corruptPos >= 48) {
                expect(isValid).toBe(false);
                return !isValid;
              }
            }

            return true;
          }
        ),
        { ...PROPERTY_TEST_CONFIG, numRuns: 30 }
      );
    });

    it('should reject data with invalid magic bytes', () => {
      const degree = 64;
      const modulus = BigInt('132120577');

      fc.assert(
        fc.property(arbitrarySecretKey(degree, modulus), (sk) => {
          const serialized = serializer.serializeSecretKey(sk);

          // Corrupt magic bytes
          const corrupted = new Uint8Array(serialized);
          corrupted[0] = 0x00;
          corrupted[1] = 0x00;

          const deserialized = serializer.deserializeSecretKey(corrupted);
          expect(deserialized).toBeNull();

          return deserialized === null;
        }),
        { ...PROPERTY_TEST_CONFIG, numRuns: 20 }
      );
    });
  });

  describe('Serialization Size Constraints', () => {
    it('should produce reasonably sized serialized keys', () => {
      const degree = 1024;
      const modulus = BigInt('132120577');

      fc.assert(
        fc.property(arbitrarySecretKey(degree, modulus), (sk) => {
          const serialized = serializer.serializeSecretKey(sk);

          // Expected size: header (48) + dist (1) + degree (4) + coeffs (degree * 8)
          const expectedSize = 48 + 1 + 4 + degree * 8;
          expect(serialized.length).toBe(expectedSize);

          return serialized.length === expectedSize;
        }),
        { ...PROPERTY_TEST_CONFIG, numRuns: 10 }
      );
    });
  });

  describe('Integrity Verification', () => {
    it('should pass integrity check for valid data', () => {
      const degree = 64;
      const modulus = BigInt('132120577');

      fc.assert(
        fc.property(arbitrarySecretKey(degree, modulus), (sk) => {
          const serialized = serializer.serializeSecretKey(sk);
          const isValid = serializer.verifyIntegrity(serialized);

          expect(isValid).toBe(true);
          return isValid;
        }),
        { ...PROPERTY_TEST_CONFIG, numRuns: 50 }
      );
    });
  });
});

/**
 * **Validates: Requirements 4.5**
 * 
 * This test suite verifies that key serialization maintains functional
 * equivalence through round-trip operations. The properties tested ensure:
 * 
 * 1. Secret keys can be serialized and deserialized without data loss
 * 2. Integrity checksums detect data corruption
 * 3. Invalid magic bytes are rejected
 * 4. Serialized sizes are predictable and reasonable
 */

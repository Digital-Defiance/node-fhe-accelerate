import { describe, it, expect } from 'vitest';
import { version, FHEErrorCode, FHEError, createEngine } from './index';

describe('@digitaldefiance/node-fhe-accelerate', () => {
  describe('Package metadata', () => {
    it('should export version', () => {
      expect(version).toBe('0.1.0');
    });
  });

  describe('FHEErrorCode enum', () => {
    it('should have all error codes', () => {
      expect(FHEErrorCode.NOISE_BUDGET_EXHAUSTED).toBe('NOISE_BUDGET_EXHAUSTED');
      expect(FHEErrorCode.INVALID_PARAMETERS).toBe('INVALID_PARAMETERS');
      expect(FHEErrorCode.KEY_MISMATCH).toBe('KEY_MISMATCH');
      expect(FHEErrorCode.HARDWARE_UNAVAILABLE).toBe('HARDWARE_UNAVAILABLE');
      expect(FHEErrorCode.SERIALIZATION_ERROR).toBe('SERIALIZATION_ERROR');
    });
  });

  describe('FHEError class', () => {
    it('should create error with code and message', () => {
      const error = new FHEError(
        'Test error',
        FHEErrorCode.INVALID_PARAMETERS,
        { param: 'test' }
      );

      expect(error).toBeInstanceOf(Error);
      expect(error).toBeInstanceOf(FHEError);
      expect(error.message).toBe('Test error');
      expect(error.code).toBe(FHEErrorCode.INVALID_PARAMETERS);
      expect(error.details).toEqual({ param: 'test' });
      expect(error.name).toBe('FHEError');
    });

    it('should work without details', () => {
      const error = new FHEError('Test error', FHEErrorCode.KEY_MISMATCH);

      expect(error.message).toBe('Test error');
      expect(error.code).toBe(FHEErrorCode.KEY_MISMATCH);
      expect(error.details).toBeUndefined();
    });
  });

  describe('createEngine', () => {
    it('should throw not implemented error', async () => {
      await expect(createEngine('tfhe-128-fast')).rejects.toThrow(
        'Not yet implemented - native addon required'
      );
    });

    it('should accept custom parameters', async () => {
      const customParams = {
        polyDegree: 2048,
        moduli: [BigInt('1099511627777')],
        securityLevel: 128 as const,
      };

      await expect(createEngine(customParams)).rejects.toThrow(
        'Not yet implemented - native addon required'
      );
    });
  });
});

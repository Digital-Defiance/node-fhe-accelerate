import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: ['src/**/*.{test,spec,prop.test}.ts'],
    exclude: ['node_modules/', 'dist/', 'target/', 'cpp/'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'dist/',
        'target/',
        'cpp/',
        '**/*.test.ts',
        '**/*.spec.ts',
        '**/*.prop.test.ts',
        'vitest.config.ts',
      ],
    },
    testTimeout: 60000, // 60 seconds for property-based tests with 100+ iterations
    // Property-based testing configuration
    // Tests should use fast-check with minimum 100 iterations
    // Seed logging enabled for reproducibility
    // Shrinking enabled for minimal failing examples
  },
});

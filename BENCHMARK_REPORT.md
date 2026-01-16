# Comprehensive FHE Hardware Benchmark Report

Testing ALL hardware features on Apple Silicon for FHE operations.

## Summary of Winners

| Operation | Winner | Latency | Speedup |
|-----------|--------|---------|--------|
| Modular Multiplication (n=1024) | Barrett Unrolled (4x) | 1.77 µs | - |
| Modular Multiplication (n=4096) | Barrett Unrolled (4x) | 6.93 µs | - |
| Modular Multiplication (n=16384) | Barrett Unrolled (4x) | 26.87 µs | - |
| Modular Multiplication (n=65536) | Barrett Unrolled (4x) | 114.28 µs | - |
| NTT (degree=1024) | Matrix-form NTT (AMX) | 8.58 µs | - |
| NTT (degree=4096) | SME Tile NTT | 39.74 µs | - |
| NTT (degree=16384) | SME Tile NTT | 186.13 µs | - |
| Polynomial Multiplication (degree=1024) | NTT-based | 21.24 µs | - |
| Polynomial Multiplication (degree=4096) | NTT-based | 97.04 µs | - |
| Neural Engine Operations (batch=1024) | CPU Direct | 515.02 ns | - |
| Neural Engine Operations (batch=16384) | CPU Direct | 8.39 µs | - |
| Neural Engine Operations (batch=65536) | CPU Direct | 34.14 µs | - |
| Ray Tracing Operations (base=256, levels=4) | CPU Direct | 36.39 µs | - |
| Texture Sampling (degree=1024, points=10000) | Texture Sampling | 5.08 µs | - |
| Memory System (size=64KB) | Cache-Aligned | 570.85 ns | - |
| Memory System (size=1024KB) | memcpy | 12.35 µs | - |
| Memory System (size=16384KB) | Cache-Aligned | 212.55 µs | - |
| Pipelined Operations (degree=4096) | Parallel NTTs | 85.63 µs | - |
| Pipelined Operations (degree=16384) | Parallel NTTs | 275.40 µs | - |

## Detailed Results

### Modular Multiplication (n=1024)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| CPU Scalar | CPU Scalar | 3.88 µs | 46.61 ns | 257.67 Kops/s | 1.00x | ✓ |
| NEON Basic | NEON | 3.72 µs | 41.23 ns | 269.12 Kops/s | 1.04x | ✓ |
| NEON Unrolled (4x) | NEON Unrolled | 3.70 µs | 30.41 ns | 270.12 Kops/s | 1.05x | ✓ |
| Montgomery | CPU Scalar | 2.44 µs | 26.03 ns | 409.91 Kops/s | 1.59x | ✓ |
| Barrett Unrolled (4x) | CPU Scalar | 1.77 µs | 27.49 ns | 564.44 Kops/s | 2.19x | ✓ |
| Neural Engine | Neural Engine | 2.01 µs | 50.86 ns | 498.03 Kops/s | 1.93x | ✓ |

**Winner:** Barrett Unrolled (4x)

### Modular Multiplication (n=4096)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| CPU Scalar | CPU Scalar | 13.52 µs | 55.04 ns | 73.94 Kops/s | 1.00x | ✓ |
| NEON Basic | NEON | 13.07 µs | 456.80 ns | 76.54 Kops/s | 1.04x | ✓ |
| NEON Unrolled (4x) | NEON Unrolled | 13.73 µs | 1.80 µs | 72.86 Kops/s | 0.99x | ✓ |
| Montgomery | CPU Scalar | 9.23 µs | 542.07 ns | 108.31 Kops/s | 1.46x | ✓ |
| Barrett Unrolled (4x) | CPU Scalar | 6.93 µs | 1.31 µs | 144.40 Kops/s | 1.95x | ✓ |
| Barrett Parallel | Hybrid | 163.55 µs | 13.21 µs | 6.11 Kops/s | 0.08x | ✓ |
| Neural Engine | Neural Engine | 7.80 µs | 986.39 ns | 128.27 Kops/s | 1.73x | ✓ |

**Winner:** Barrett Unrolled (4x)

### Modular Multiplication (n=16384)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| CPU Scalar | CPU Scalar | 46.30 µs | 2.00 µs | 21.60 Kops/s | 1.00x | ✓ |
| NEON Basic | NEON | 44.57 µs | 2.88 µs | 22.44 Kops/s | 1.04x | ✓ |
| NEON Unrolled (4x) | NEON Unrolled | 53.17 µs | 3.36 µs | 18.81 Kops/s | 0.87x | ✓ |
| Montgomery | CPU Scalar | 35.75 µs | 3.54 µs | 27.97 Kops/s | 1.30x | ✓ |
| Barrett Unrolled (4x) | CPU Scalar | 26.87 µs | 1.61 µs | 37.22 Kops/s | 1.72x | ✓ |
| Barrett Parallel | Hybrid | 163.64 µs | 31.98 µs | 6.11 Kops/s | 0.28x | ✓ |
| Neural Engine | Neural Engine | 28.58 µs | 1.83 µs | 34.98 Kops/s | 1.62x | ✓ |

**Winner:** Barrett Unrolled (4x)

### Modular Multiplication (n=65536)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| CPU Scalar | CPU Scalar | 202.46 µs | 20.15 µs | 4.94 Kops/s | 1.00x | ✓ |
| NEON Basic | NEON | 181.04 µs | 14.90 µs | 5.52 Kops/s | 1.12x | ✓ |
| NEON Unrolled (4x) | NEON Unrolled | 194.20 µs | 17.66 µs | 5.15 Kops/s | 1.04x | ✓ |
| Montgomery | CPU Scalar | 138.15 µs | 10.17 µs | 7.24 Kops/s | 1.47x | ✓ |
| Barrett Unrolled (4x) | CPU Scalar | 114.28 µs | 22.36 µs | 8.75 Kops/s | 1.77x | ✓ |
| Barrett Parallel | Hybrid | 174.62 µs | 51.39 µs | 5.73 Kops/s | 1.16x | ✓ |
| Neural Engine | Neural Engine | 127.33 µs | 23.84 µs | 7.85 Kops/s | 1.59x | ✓ |

**Winner:** Barrett Unrolled (4x)

### NTT (degree=1024)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| Scalar NTT | CPU Scalar | 17.71 µs | 5.69 µs | 56.47 Kops/s | 1.00x | ✓ |
| NEON NTT | NEON | 17.35 µs | 1.82 µs | 57.65 Kops/s | 1.02x | ✓ |
| Barrett NTT | CPU Scalar | 13.24 µs | 497.62 ns | 75.55 Kops/s | 1.34x | ✓ |
| Montgomery NTT | CPU Scalar | 8.86 µs | 659.01 ns | 112.89 Kops/s | 2.00x | ✓ |
| Matrix-form NTT (AMX) | AMX | 8.58 µs | 212.56 ns | 116.51 Kops/s | 2.06x | ✓ |
| SME Tile NTT | SME | 8.67 µs | 537.26 ns | 115.35 Kops/s | 2.04x | ✓ |

**Winner:** Matrix-form NTT (AMX)

### NTT (degree=4096)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| Scalar NTT | CPU Scalar | 86.97 µs | 15.18 µs | 11.50 Kops/s | 1.00x | ✓ |
| NEON NTT | NEON | 88.60 µs | 14.64 µs | 11.29 Kops/s | 0.98x | ✓ |
| Barrett NTT | CPU Scalar | 64.38 µs | 7.12 µs | 15.53 Kops/s | 1.35x | ✓ |
| Montgomery NTT | CPU Scalar | 42.04 µs | 4.91 µs | 23.79 Kops/s | 2.07x | ✓ |
| Matrix-form NTT (AMX) | AMX | 41.52 µs | 5.73 µs | 24.09 Kops/s | 2.09x | ✓ |
| SME Tile NTT | SME | 39.74 µs | 867.33 ns | 25.16 Kops/s | 2.19x | ✓ |

**Winner:** SME Tile NTT

### NTT (degree=16384)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| Scalar NTT | CPU Scalar | 404.18 µs | 35.17 µs | 2.47 Kops/s | 1.00x | ✓ |
| NEON NTT | NEON | 398.36 µs | 28.41 µs | 2.51 Kops/s | 1.01x | ✓ |
| Barrett NTT | CPU Scalar | 390.04 µs | 61.04 µs | 2.56 Kops/s | 1.04x | ✓ |
| Montgomery NTT | CPU Scalar | 188.28 µs | 9.84 µs | 5.31 Kops/s | 2.15x | ✓ |
| Matrix-form NTT (AMX) | AMX | 187.03 µs | 7.24 µs | 5.35 Kops/s | 2.16x | ✓ |
| SME Tile NTT | SME | 186.13 µs | 6.85 µs | 5.37 Kops/s | 2.17x | ✓ |

**Winner:** SME Tile NTT

### Polynomial Multiplication (degree=1024)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| NTT-based | CPU Scalar | 21.24 µs | 4.78 µs | 47.07 Kops/s | 1.00x | ✓ |

**Winner:** NTT-based

### Polynomial Multiplication (degree=4096)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| NTT-based | CPU Scalar | 97.04 µs | 5.09 µs | 10.31 Kops/s | 1.00x | ✓ |

**Winner:** NTT-based

### Neural Engine Operations (batch=1024)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| CPU Direct | CPU Scalar | 515.02 ns | 20.13 ns | 1.94 Mops/s | 1.00x | ✓ |
| Neural Engine | Neural Engine | 778.31 ns | 22.07 ns | 1.28 Mops/s | 0.66x | ✓ |
| Neural Engine Poseidon | Neural Engine | 1.99 ms | 63.78 µs | 514.55 Kops/s | 1.00x | ✓ |

**Winner:** CPU Direct

### Neural Engine Operations (batch=16384)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| CPU Direct | CPU Scalar | 8.39 µs | 1.14 µs | 119.12 Kops/s | 1.00x | ✓ |
| Neural Engine | Neural Engine | 15.00 µs | 1.49 µs | 66.69 Kops/s | 0.56x | ✓ |
| Neural Engine Poseidon | Neural Engine | 31.63 ms | 279.02 µs | 518.07 Kops/s | 1.00x | ✓ |

**Winner:** CPU Direct

### Neural Engine Operations (batch=65536)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| CPU Direct | CPU Scalar | 34.14 µs | 3.85 µs | 29.29 Kops/s | 1.00x | ✓ |
| Neural Engine | Neural Engine | 55.11 µs | 2.48 µs | 18.15 Kops/s | 0.62x | ✓ |
| Neural Engine Poseidon | Neural Engine | 125.87 ms | 643.11 µs | 520.66 Kops/s | 1.00x | ✓ |

**Winner:** CPU Direct

### Ray Tracing Operations (base=256, levels=4)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| CPU Direct | CPU Scalar | 36.39 µs | 2.27 µs | 27.48 Kops/s | 1.00x | ✓ |
| Ray Tracing BVH | Ray Tracing | 42.75 µs | 2.93 µs | 23.39 Kops/s | 0.85x | ✓ |

**Winner:** CPU Direct

### Texture Sampling (degree=1024, points=10000)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| Horner's Method | CPU Scalar | 8.97 ms | 56.35 µs | 111.51 ops/s | 1.00x | ✓ |
| Texture Sampling | Texture Sampling | 5.08 µs | 41.09 ns | 196.66 Kops/s | 1763.64x | ✓ |

**Winner:** Texture Sampling

### Memory System (size=64KB)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| memcpy | CPU Scalar | 630.85 ns | 32.97 ns | 0.00 ops/s | 1.00x | ✓ |
| Unified Memory | Hybrid | 936.63 ns | 30.44 ns | 0.00 ops/s | 0.67x | ✓ |
| Cache-Aligned | CPU Scalar | 570.85 ns | 23.32 ns | 0.00 ops/s | 1.11x | ✓ |

**Winner:** Cache-Aligned

### Memory System (size=1024KB)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| memcpy | CPU Scalar | 12.35 µs | 50.03 ns | 0.00 ops/s | 1.00x | ✓ |
| Unified Memory | Hybrid | 14.06 µs | 6.23 µs | 0.00 ops/s | 0.88x | ✓ |
| Cache-Aligned | CPU Scalar | 12.90 µs | 893.27 ns | 0.00 ops/s | 0.96x | ✓ |

**Winner:** memcpy

### Memory System (size=16384KB)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| memcpy | CPU Scalar | 218.01 µs | 25.87 µs | 0.00 ops/s | 1.00x | ✓ |
| Unified Memory | Hybrid | 214.91 µs | 13.38 µs | 0.00 ops/s | 1.01x | ✓ |
| Cache-Aligned | CPU Scalar | 212.55 µs | 11.34 µs | 0.00 ops/s | 1.03x | ✓ |

**Winner:** Cache-Aligned

### Pipelined Operations (degree=4096)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| Sequential | CPU Scalar | 96.65 µs | 7.06 µs | 10.35 Kops/s | 1.00x | ✓ |
| Parallel NTTs | Hybrid | 85.63 µs | 23.02 µs | 11.68 Kops/s | 1.13x | ✓ |

**Winner:** Parallel NTTs

### Pipelined Operations (degree=16384)

| Method | Backend | Latency | Stddev | Throughput | Speedup | Correct |
|--------|---------|---------|--------|------------|---------|--------|
| Sequential | CPU Scalar | 426.50 µs | 17.77 µs | 2.34 Kops/s | 1.00x | ✓ |
| Parallel NTTs | Hybrid | 275.40 µs | 30.73 µs | 3.63 Kops/s | 1.55x | ✓ |

**Winner:** Parallel NTTs


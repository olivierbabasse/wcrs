Starting idea: https://codingchallenges.fyi/challenges/challenge-wc/

Hyperfine benchmarks on i7-9700k, Linux kernel 6.8.0, 1 GB file:

| #  | Step                                                       | Time    | Speedup vs `wc` |
|---:|------------------------------------------------------------|--------:|----------------:|
|  0 | Reference implementation `/usr/bin/wc`                     | 3.830 s |          1.00× |
|  1 | Naive implementation, default release options              | 2.372 s |          1.61× |
|  2 | Naive implementation, native instruction set               | 2.055 s |          1.86× |
|  3 | Buffered read and single pass                              | 1.822 s |          2.10× |
|  4 | Line counting with `memchr`                                | 1.627 s |          2.35× |
|  5 | SIMD word counting                                         |  157 ms |         24.4× |
|  6 | SIMD combined word & newline counting                      |  154 ms |         24.9× |
|  7 | `mmap` instead of buffered reading                         |  137 ms |         27.9× |
|  8 | Shuffle-based whitespace classifier                        |  135 ms |         28.4× |
|  9 | Single shuffle classifier                                  |  124 ms |         30.9× |
| 10 | "Sum of absolute differences" instead of popcount trick    |  120 ms |         31.9× |
| 11 | SAD trick also on word path                                |  116 ms |         33.0× |
| 12 | Software prefetching                                       |  104 ms |         36.8× |
| 13 | Parallel with rayon                                        |   57 ms |         67.2× |
| 14 | Enabling THP                                               |   37 ms |        103.5× |
| 15 | Using `pread` instead of `mmap`                            |   40 ms |         95.8× |

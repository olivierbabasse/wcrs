starting idea : https://codingchallenges.fyi/challenges/challenge-wc/

hyperfine tests on i7-9700, linux kernel 6.8.0, 1GB file :
reference implementation /usr/bin/wc : 3.830s
naive implementation, default release options : 2.372s
naive implementation, native instruction set : 2.055s
buffered read and single pass : 1.822s
line counting with memchr : 1.627s
SIMD word counting : 157ms
SIMD combined word & newline counting : 154ms
mmap instead of buffered reading : 137ms
shuffle-based whitespace classifier : 135ms
single shuffle classifier : 124ms
"sum of absolute differences" instead of popcount trick : 120ms
starting idea : https://codingchallenges.fyi/challenges/challenge-wc/

hyperfine tests on MacBookPro i5 early 2015, 1GB file :
reference implementation /usr/bin/wc : 3.645s
naive implementation, default release options : 3.233s
naive implementation, codegen-units, lto, panic, target native, debug info : no significant change
naive implementation, native instruction set : 3.002s
buffered read and single pass : 2.347s
line counting with memchr : 2.015s
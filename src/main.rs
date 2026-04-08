use std::env;
use std::fs::File;
use std::io::{self, Read};
use memchr::memchr_iter;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn count_words_avx2(chunk: &[u8], in_word: &mut bool) -> u64 { unsafe {
    let mut words: u64 = 0;
    let mut prev_ws_bit: u32 = if *in_word { 0 } else { 1 };

    let space = _mm256_set1_epi8(b' ' as i8);
    let tab = _mm256_set1_epi8(b'\t' as i8);
    let newline = _mm256_set1_epi8(b'\n' as i8);
    let cr = _mm256_set1_epi8(b'\r' as i8);
    let vt = _mm256_set1_epi8(0x0B as i8);
    let ff = _mm256_set1_epi8(0x0C as i8);

    let mut i = 0;
    let len = chunk.len();

    while i + 32 <= len {
        let data = _mm256_loadu_si256(chunk.as_ptr().add(i) as *const __m256i);

        let ws = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_or_si256(
                    _mm256_cmpeq_epi8(data, space),
                    _mm256_cmpeq_epi8(data, tab),
                ),
                _mm256_or_si256(
                    _mm256_cmpeq_epi8(data, newline),
                    _mm256_cmpeq_epi8(data, cr),
                ),
            ),
            _mm256_or_si256(
                _mm256_cmpeq_epi8(data, vt),
                _mm256_cmpeq_epi8(data, ff),
            ),
        );

        let ws_mask = _mm256_movemask_epi8(ws) as u32;
        let not_ws_mask = !ws_mask;
        let prev_was_ws = (ws_mask << 1) | prev_ws_bit;
        let word_starts = not_ws_mask & prev_was_ws;
        words += word_starts.count_ones() as u64;
        prev_ws_bit = (ws_mask >> 31) & 1;

        i += 32;
    }

    *in_word = prev_ws_bit == 0;
    words + count_words_scalar(&chunk[i..], in_word)
}}

fn count_words_scalar(chunk: &[u8], in_word: &mut bool) -> u64 {
    let mut words: u64 = 0;
    let mut iw = *in_word;
    for &b in chunk {
        if b.is_ascii_whitespace() {
            iw = false;
        } else if !iw {
            iw = true;
            words += 1;
        }
    }
    *in_word = iw;
    words
}

fn count_words(chunk: &[u8], in_word: &mut bool) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { count_words_avx2(chunk, in_word) };
        }
    }
    count_words_scalar(chunk, in_word)
}

fn main() -> io::Result<()> {
    let filename = env::args().nth(1).expect("Usage: wcrs <filename>");
    let mut file = File::open(&filename)?;

    let mut buf = [0u8; 64 * 1024];
    let mut bytes: u64 = 0;
    let mut lines: u64 = 0;
    let mut words: u64 = 0;
    let mut in_word = false;

    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        let chunk = &buf[..n];
        bytes += n as u64;
        lines += memchr_iter(b'\n', chunk).count() as u64;
        words += count_words(chunk, &mut in_word);
    }

    println!("  {lines}  {words} {bytes} {filename}");
    Ok(())
}

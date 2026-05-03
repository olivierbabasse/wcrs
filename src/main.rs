use memmap2::Mmap;
use std::env;
use std::fs::File;
use std::io::{self};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn count_avx2(chunk: &[u8], in_word: &mut bool) -> (u64, u64) {
    unsafe {
        let mut words: u64 = 0;
        let mut lines: u64 = 0;
        let mut prev_ws_bit: u32 = if *in_word { 0 } else { 1 };

        let newline = _mm256_set1_epi8(b'\n' as i8);

        // Each whitespace byte sits at the table position equal to its own value :
        // table[0]=0x20, table[9..=0xD]=0x09..0x0D, rest = 0.
        // _mm256_shuffle_epi8 indexes by the low nibble of each input byte, so the
        // lookup returns the byte's own value iff the byte is whitespace. Comparing
        // the lookup result to the original byte then directly produces the WS mask.
        // (High-bit bytes get zeroed by _mm256_shuffle_epi8, so they correctly compare unequal.)
        // Table duplicated across both 128-bit lanes for _mm256_shuffle_epi8.
        let ws_table = _mm256_setr_epi8(
            0x20, 0, 0, 0, 0, 0, 0, 0, 0, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0, 0, 0x20, 0, 0, 0, 0, 0,
            0, 0, 0, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0, 0,
        );

        let mut i = 0;
        let len = chunk.len();

        while i + 32 <= len {
            let data = _mm256_loadu_si256(chunk.as_ptr().add(i) as *const __m256i);

            // Lines : direct cmpeq with '\n'.
            let nl_cmp = _mm256_cmpeq_epi8(data, newline);
            let nl_mask = _mm256_movemask_epi8(nl_cmp) as u32;
            lines += nl_mask.count_ones() as u64;

            // Whitespace : single-shuffle classifier.
            let ws_lookup = _mm256_shuffle_epi8(ws_table, data);
            let is_ws = _mm256_cmpeq_epi8(ws_lookup, data);
            let ws_mask = _mm256_movemask_epi8(is_ws) as u32;

            let prev_was_ws = (ws_mask << 1) | prev_ws_bit;
            let word_starts = !ws_mask & prev_was_ws;
            words += word_starts.count_ones() as u64;
            prev_ws_bit = (ws_mask >> 31) & 1;

            i += 32;
        }

        *in_word = prev_ws_bit == 0;
        let (tw, tl) = count_scalar(&chunk[i..], in_word);
        (words + tw, lines + tl)
    }
}

fn count_scalar(chunk: &[u8], in_word: &mut bool) -> (u64, u64) {
    let mut words: u64 = 0;
    let mut lines: u64 = 0;
    let mut iw = *in_word;
    for &b in chunk {
        if b == b'\n' {
            lines += 1;
        }
        if b.is_ascii_whitespace() {
            iw = false;
        } else if !iw {
            iw = true;
            words += 1;
        }
    }
    *in_word = iw;
    (words, lines)
}

fn count(chunk: &[u8], in_word: &mut bool) -> (u64, u64) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { count_avx2(chunk, in_word) };
        }
    }
    count_scalar(chunk, in_word)
}

fn main() -> io::Result<()> {
    let filename = env::args().nth(1).expect("Usage: wcrs <filename>");
    let file = File::open(&filename)?;
    let mmap = unsafe { Mmap::map(&file)? };
    let data: &[u8] = &mmap;
    let bytes = data.len() as u64;
    let mut in_word = false;

    let (words, lines) = count(data, &mut in_word);

    println!("  {lines}  {words} {bytes} {filename}");
    Ok(())
}

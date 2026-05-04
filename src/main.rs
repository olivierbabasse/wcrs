use memmap2::{Advice, MmapMut, MmapOptions};
use rayon::prelude::*;
use std::env;
use std::fs::File;
use std::io::{self};
use std::os::unix::fs::FileExt;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_u64_lanes(v: __m256i) -> u64 {
    unsafe {
        let mut tmp = [0u64; 4];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, v);
        tmp[0] + tmp[1] + tmp[2] + tmp[3]
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn count_avx2(chunk: &[u8], in_word: &mut bool) -> (u64, u64) {
    unsafe {
        // Number of inner iterations per accumulator flush. Each per-byte u8 lane
        // accumulates at most BATCH +1s before overflow, so we pick BATCH = 255.
        const BATCH: usize = 255;
        const BATCH_BYTES: usize = BATCH * 32;

        let mut words: u64 = 0;
        let mut lines: u64 = 0;

        // prev_ws holds the previous 32-byte block's "is_ws" mask. Only byte 31
        // (the last byte of the block) is meaningful for cross-iter carry — it
        // bridges into byte 0 of the next iter via _mm256_alignr_epi8.
        // Initial value : if we're entering "in a word", the byte before the chunk
        // was non-whitespace → 0x00. Otherwise → 0xFF.
        let mut prev_ws: __m256i = _mm256_set1_epi8(if *in_word { 0 } else { -1i8 });

        let newline = _mm256_set1_epi8(b'\n' as i8);
        // Single-shuffle whitespace classifier table : each whitespace byte sits at
        // the table position equal to its own value. _mm256_shuffle_epi8 + cmpeq
        // produces the WS mask in 2 SIMD ops. Table duplicated across both lanes.
        let ws_table = _mm256_setr_epi8(
            0x20, 0, 0, 0, 0, 0, 0, 0, 0, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0, 0, 0x20, 0, 0, 0, 0, 0,
            0, 0, 0, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0, 0,
        );
        let zero = _mm256_setzero_si256();

        let mut i = 0;
        let len = chunk.len();

        // Software prefetch lookahead.
        // _mm_prefetch is a hint and never faults, so it's safe to prefetch past
        // the end of the mmap region.
        const PREFETCH_DIST: usize = 768;

        // Hot loop : process full 8160-byte (255 × 32) batches, accumulating
        // counts in per-byte u8 lanes. Flush via _mm256_sad_epu8 to u64 totals
        // before the lanes can overflow.
        while i + BATCH_BYTES <= len {
            let mut nl_acc = zero;
            let mut ws_acc = zero;
            let batch_end = i + BATCH_BYTES;
            while i < batch_end {
                _mm_prefetch::<_MM_HINT_T0>(chunk.as_ptr().add(i + PREFETCH_DIST) as *const i8);

                let data = _mm256_loadu_si256(chunk.as_ptr().add(i) as *const __m256i);

                // Newline counting : cmpeq → -1 per match → subtract from accumulator (= +1).
                let nl_cmp = _mm256_cmpeq_epi8(data, newline);
                nl_acc = _mm256_sub_epi8(nl_acc, nl_cmp);

                // Whitespace : single-shuffle classifier.
                let ws_lookup = _mm256_shuffle_epi8(ws_table, data);
                let is_ws = _mm256_cmpeq_epi8(ws_lookup, data);

                // Word-start detection in vector space : shift is_ws left by 1 byte
                // (with byte 31 of prev iter feeding in), then AND with !is_ws.
                // _mm256_alignr_epi8 is intra-lane only, so we permute lanes first.
                let bridge = _mm256_permute2x128_si256::<0x21>(prev_ws, is_ws);
                let prev_was_ws_vec = _mm256_alignr_epi8::<15>(is_ws, bridge);
                let word_starts = _mm256_andnot_si256(is_ws, prev_was_ws_vec);
                ws_acc = _mm256_sub_epi8(ws_acc, word_starts);

                prev_ws = is_ws;
                i += 32;
            }
            lines += sum_u64_lanes(_mm256_sad_epu8(nl_acc, zero));
            words += sum_u64_lanes(_mm256_sad_epu8(ws_acc, zero));
        }

        // Tail SIMD loop : <BATCH 32-byte iters left, accumulators fresh.
        let mut nl_acc = zero;
        let mut ws_acc = zero;
        while i + 32 <= len {
            let data = _mm256_loadu_si256(chunk.as_ptr().add(i) as *const __m256i);

            let nl_cmp = _mm256_cmpeq_epi8(data, newline);
            nl_acc = _mm256_sub_epi8(nl_acc, nl_cmp);

            let ws_lookup = _mm256_shuffle_epi8(ws_table, data);
            let is_ws = _mm256_cmpeq_epi8(ws_lookup, data);

            let bridge = _mm256_permute2x128_si256::<0x21>(prev_ws, is_ws);
            let prev_was_ws_vec = _mm256_alignr_epi8::<15>(is_ws, bridge);
            let word_starts = _mm256_andnot_si256(is_ws, prev_was_ws_vec);
            ws_acc = _mm256_sub_epi8(ws_acc, word_starts);

            prev_ws = is_ws;
            i += 32;
        }
        lines += sum_u64_lanes(_mm256_sad_epu8(nl_acc, zero));
        words += sum_u64_lanes(_mm256_sad_epu8(ws_acc, zero));

        // Recover in_word state from byte 31 of prev_ws (last SIMD byte's WS status).
        // 0xFF means the byte was whitespace, so we end NOT in a word.
        let mut tmp = [0u8; 32];
        _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, prev_ws);
        *in_word = tmp[31] == 0;

        // Remaining <32 bytes go through the scalar path.
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

struct ChunkResult {
    words: u64,
    lines: u64,
    starts_nonws: bool,
    ends_nonws: bool,
}

// Per-thread pread buffer size. 128 KB matches fastlwc — fits in L2 with margin
// for code/state, big enough that pread overhead amortises, small enough that
// the SIMD scan keeps its working set in cache.
const PREAD_BUFSIZE: usize = 128 * 1024;
// Anonymous huge-page allocation. Must be at least 2 MB and 2 MB-aligned for the
// kernel to back it with a single huge page (with MADV_HUGEPAGE in effect).
// We use only PREAD_BUFSIZE of it ; the rest sits unused but huge-page-mapped.
const HUGE_BUF_SIZE: usize = 2 * 1024 * 1024;

// Allocate a big huge-page-backed buffer covering n_threads * HUGE_BUF_SIZE.
// One mmap call instead of N parallel ones.
fn alloc_big_huge_buf(n_threads: usize) -> MmapMut {
    let total = n_threads * HUGE_BUF_SIZE;
    let mut m = MmapOptions::new().len(total).map_anon().expect("map_anon");
    let _ = m.advise(Advice::HugePage);
    // Touch one byte per huge-page slot so each gets faulted-in as a huge page
    // up-front (synchronously here, before any worker thread starts scanning).
    // MmapMut derefs to [u8], so this is bounds-checked indexing.
    let mut o = 0;
    while o < total {
        m[o] = 0;
        o += HUGE_BUF_SIZE;
    }
    m
}

fn count_range(file: &File, start: u64, end: u64, buf: &mut [u8]) -> ChunkResult {
    if start >= end {
        return ChunkResult {
            words: 0,
            lines: 0,
            starts_nonws: false,
            ends_nonws: false,
        };
    }
    let mut total_words: u64 = 0;
    let mut total_lines: u64 = 0;
    // count() carries word-boundary state across calls via &mut bool — we just
    // thread it through every pread chunk in this thread's range.
    let mut in_word = false;

    let mut first_byte: Option<u8> = None;
    let mut last_byte: u8 = 0;

    let mut offset = start;
    while offset < end {
        let to_read = ((end - offset) as usize).min(PREAD_BUFSIZE).min(buf.len());
        let n = file.read_at(&mut buf[..to_read], offset).expect("read_at");
        if n == 0 {
            panic!("unexpected EOF at offset {offset}");
        }
        let slice = &buf[..n];
        if first_byte.is_none() {
            first_byte = Some(slice[0]);
        }
        last_byte = slice[n - 1];
        let (w, l) = count(slice, &mut in_word);
        total_words += w;
        total_lines += l;
        offset += n as u64;
    }

    ChunkResult {
        words: total_words,
        lines: total_lines,
        starts_nonws: !first_byte.unwrap().is_ascii_whitespace(),
        ends_nonws: !last_byte.is_ascii_whitespace(),
    }
}

fn main() -> io::Result<()> {
    let filename = env::args().nth(1).expect("Usage: wcrs <filename>");
    let file = File::open(&filename)?;
    let bytes = file.metadata()?.len();

    // Split the file into one contiguous byte-range per worker thread.
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = (bytes as usize).div_ceil(n_threads).max(1);
    let ranges: Vec<(u64, u64)> = (0..n_threads)
        .map(|i| {
            let s = (i * chunk_size) as u64;
            let e = ((i + 1) * chunk_size).min(bytes as usize) as u64;
            (s, e)
        })
        .filter(|(s, e)| s < e)
        .collect();

    // Allocate one big huge-page-backed buffer up-front and hand each worker
    // its own non-overlapping slice via par_chunks_mut
    let mut big_buf = alloc_big_huge_buf(ranges.len());

    let results: Vec<ChunkResult> = big_buf
        .par_chunks_mut(HUGE_BUF_SIZE)
        .zip(ranges.par_iter())
        .map(|(buf, &(s, e))| count_range(&file, s, e, buf))
        .collect();

    let mut lines: u64 = 0;
    let mut words: u64 = 0;
    for r in &results {
        lines += r.lines;
        words += r.words;
    }
    // Cross-thread fix-up : if range N ends in non-whitespace and range N+1 starts
    // in non-whitespace, the word straddling the boundary was counted twice.
    for i in 1..results.len() {
        if results[i - 1].ends_nonws && results[i].starts_nonws {
            words -= 1;
        }
    }

    println!("  {lines}  {words} {bytes} {filename}");
    Ok(())
}

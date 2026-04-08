use std::env;
use std::fs::File;
use std::io::{self, Read};

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
        for &b in chunk {
            if b == b'\n' {
                lines += 1;
            }
            if b.is_ascii_whitespace() {
                in_word = false;
            } else if !in_word {
                in_word = true;
                words += 1;
            }
        }
    }

    println!("  {lines}  {words} {bytes} {filename}");
    Ok(())
}

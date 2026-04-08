use std::env;
use std::fs;

fn main() {
    let filename = env::args().nth(1).expect("Usage: wcrs <filename>");
    let data = fs::read(&filename).expect("Could not read file");

    let bytes = data.len();
    let lines = data.iter().filter(|&&b| b == b'\n').count();
    let words = data
        .split(|b| b.is_ascii_whitespace())
        .filter(|w| !w.is_empty())
        .count();

    println!("  {lines}  {words} {bytes} {filename}");
}

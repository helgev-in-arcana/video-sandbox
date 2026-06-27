//! `Frame::transposed()` の単独ベンチ。1920x1080 を多数回転置して所要時間と実効帯域を出す。
//! 実行: cargo run --release -p video-pipeline --example transpose_bench

use std::hint::black_box;
use std::time::Instant;
use video_pipeline::Frame;

fn main() {
    let w = 1920u32;
    let h = 1080u32;
    let f = Frame::black(w, h, 0);
    let iters = 200u32;

    // ウォームアップ。
    for _ in 0..10 {
        black_box(f.transposed());
    }

    let t = Instant::now();
    for _ in 0..iters {
        black_box(f.transposed());
    }
    let el = t.elapsed();

    let per = el / iters;
    // 1 回あたり read + write で w*h*4 バイトずつ。
    let bytes = w as f64 * h as f64 * 4.0 * 2.0;
    let gbps = bytes / per.as_secs_f64() / 1e9;
    println!(
        "transpose {w}x{h}: {:.3} ms/op, {:.2} GB/s  ({} iters, total {:.2}s)",
        per.as_secs_f64() * 1000.0,
        gbps,
        iters,
        el.as_secs_f64(),
    );
}

//! ビルド時に同梱 FFmpeg 8.1 の DLL を実行ファイルの隣（target/<profile>/）へコピーする。
//! Windows は exe と同じディレクトリを最優先で探すため、これで `cargo run` が
//! PATH 設定なしでそのまま動く（このプロジェクト内でしか動かさない前提の雑な対応）。

use std::{env, fs, path::PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let bin_dir = manifest.join("ffmpeg-8.1").join("bin");

    // OUT_DIR = target/<profile>/build/<pkg>-<hash>/out → 3 つ上が target/<profile>/
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let Some(dest) = out_dir.ancestors().nth(3) else { return };

    if let Ok(entries) = fs::read_dir(&bin_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "dll") {
                let _ = fs::copy(&path, dest.join(path.file_name().unwrap()));
            }
        }
    }

    println!("cargo:rerun-if-changed=ffmpeg-8.1/bin");
}

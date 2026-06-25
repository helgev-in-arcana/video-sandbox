//! 実験用 動画ピクセルパイプライン。
//!
//! 動画ファイルを入力し、各フレームを RGBA8 のピクセルバッファ（[`Frame`]）として
//! 受け取り、ユーザが書いた [`Process`] ノードを [`Pipeline::pipe`] で連結して通し、
//! 動画ファイルに出力する小規模なフレームワーク。
//!
//! # 設計（確定事項）
//!
//! - 実行モデルは **push（ストリーミング）**。変換は **1 入力 → 1 出力** に限定。
//! - 中間フレーム表現は **RGBA8 単一形式**（[`Frame::data`]）。入出力境界でのみ
//!   FFmpeg の `sws_scale` により変換する。
//! - ディスパッチは動的（`Box<dyn Process>`）。デコード／エンコードは `rsmpeg` を
//!   [`VideoFile::new`]（ソース）と [`Pipeline::encode_to`]（シンク）の内部に閉じ込め、
//!   中間の [`Process`] ノードは FFmpeg に一切依存しない。
//!
//! # 使用例
//!
//! ```no_run
//! use video_pipeline::{VideoFile, Invert, MotionBlur, EncodeSettings};
//!
//! VideoFile::new("video.mp4")
//!     .pipe(Invert)
//!     .pipe(MotionBlur { prev: None, alpha: 0.7 })
//!     .map(|f, ctx| { /* 使い捨ての実験。ctx.index / ctx.pts を参照できる */ f })
//!     .encode_to("out.mp4", EncodeSettings::default())?;
//! # Ok::<(), anyhow::Error>(())
//! ```

#![warn(missing_docs)]

mod ffmpeg;
mod frame;
mod pipeline;
mod process;

pub use frame::Frame;
pub use pipeline::{EncodeSettings, Pipeline, VideoFile};
pub use process::{Invert, MotionBlur, Process, ProcessCtx};

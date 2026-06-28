//! 実験用 動画ピクセルパイプライン。
//!
//! 動画ファイルを入力し、各フレームを RGBA8 のピクセルバッファ（[`Frame`]）として
//! 受け取り、ユーザが書いた [`Process`] ノードを [`Pipeline::pipe`] で連結して通し、
//! 動画ファイルに出力する小規模なフレームワーク。
//!
//! # 設計
//!
//! - 実行モデルは **pull（Iterator 型）**。各アダプタが上流 [`Pipeline`] をジェネリクスで所有する。
//! - スレッド境界は [`.buffered(cap)`](Pipeline::buffered) で任意の位置に挿せる。
//! - 中間フレーム表現は **RGBA8 単一形式**（[`Frame::data`]）。入出力境界でのみ
//!   FFmpeg の `sws_scale` により変換する。
//! - 処理ノードは [`Process`] トレイトを実装し、[`Pipeline::pipe`] に渡す。複数入力の合成
//!   （Mix）は、2 つの `impl Pipeline` を受け取りそれ自身が [`Pipeline`] を実装する構造体として
//!   利用側（ノード）に置く。

#![warn(missing_docs)]

mod ffmpeg;
mod frame;
mod pipeline;
mod pixel;
mod process;

pub use frame::{Frame, FrameCtx};
pub use pipeline::{
    Buffered, Decode, DecodeSettings, EncodeSettings, HwAccel, Pipeline, Piped, VideoEncoder,
    VideoFile,
};
pub use pixel::{Hsv, Pixel};
pub use process::Process;

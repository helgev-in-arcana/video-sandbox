//! 動画ストリーム（[`Pipeline`](video_pipeline::Pipeline)）を生成するソース。
//!
//! 単一フレームを繰り返す [`StillVideo`]（[`crate::framegen`] で作った静止フレームを
//! 動画化する）と、入力フレームを持たずゼロからフレーム列を生み出す手続き的ノイズ源
//! [`FractalNoise`]（時間方向に発展するフラクタルノイズ。基底ノイズ [`NoiseKind`]、
//! フラクタル合成方式 [`FractalKind`]、時間の連続性 [`TimeMode`] を [`FractalNoiseDescriptor`]
//! で選べる）を提供する。
//!
//! いずれも各フレームの [`FrameCtx`](video_pipeline::FrameCtx) を連番に振り直すので、
//! 時間依存の処理ノードや後段の処理ノード（例 [`Displace`](crate::nodes::Displace) の
//! map 入力）へそのまま流せる。

mod fractal_noise;
mod noise;
mod still_video;

pub use fractal_noise::*;
pub use noise::*;
pub use still_video::*;

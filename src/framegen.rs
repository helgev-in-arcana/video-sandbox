//! フレーム生成ソース。映像処理の入力となる単一 [`Frame`](video_pipeline::Frame) を、
//! 外部リソースやプログラムから作り出す。
//!
//! ここで作った [`Frame`](video_pipeline::Frame) は [`crate::videogen`] に渡して動画
//! ストリーム（[`Pipeline`](video_pipeline::Pipeline)）化し、以降の処理ノードへ流す。

mod image_frame;

pub use image_frame::*;

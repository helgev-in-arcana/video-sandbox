//! [`Process`] トレイトとフレーム文脈、および同梱ノード。
//!
//! 実行モデルは push・1 入力 → 1 出力に確定。[`Process::process`] は無謬
//! （ピクセル演算は失敗しない前提）なので `Result` を返さない。`&mut self` に状態を
//! 持てるため、時間方向の畳み込み（前フレーム参照）も素直に書ける。
//!
//! 具体ノードの実装は [`nodes`] サブモジュールにまとめ、ここから re-export する。

mod nodes;

pub use nodes::{Invert, MotionBlur};

use crate::frame::Frame;

/// 各フレームに付随する文脈。ノードがフレーム番号やタイムスタンプを参照できる。
#[derive(Clone, Copy, Debug)]
pub struct ProcessCtx {
    /// 0 始まりのフレーム番号（ソースから取り出した順）。
    pub index: u64,
    /// presentation timestamp（入力ストリームの time_base 基準）。[`Frame::pts`] と同値。
    pub pts: i64,
}

/// 1 枚のフレームを受け取り、1 枚を返す変換ノード。
///
/// パイプラインに [`Pipeline::pipe`](crate::Pipeline::pipe) で連結される。状態を持つ
/// ノードは `&mut self` のフィールドに保持する（例: [`MotionBlur`]）。
pub trait Process {
    /// `frame` を変換して返す。`ctx` は当該フレームのフレーム番号とタイムスタンプ。
    fn process(&mut self, frame: Frame, ctx: ProcessCtx) -> Frame;
}

/// [`Pipeline::map`](crate::Pipeline::map) 用。クロージャを [`Process`] として挿す無名ノード。
pub(crate) struct MapNode<F>(pub F);

impl<F: FnMut(Frame, ProcessCtx) -> Frame> Process for MapNode<F> {
    fn process(&mut self, frame: Frame, ctx: ProcessCtx) -> Frame {
        (self.0)(frame, ctx)
    }
}

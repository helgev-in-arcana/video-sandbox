use crate::frame::{Frame, FrameCtx};

/// 1 枚のフレームを受け取り、1 枚を返す変換ノード。
///
/// パイプラインに [`Pipeline::pipe`](crate::Pipeline::pipe) で連結される。状態を持つ
/// ノードは `&mut self` のフィールドに保持する（例: [`MotionBlur`]）。
pub trait Process {
    /// `frame` を変換して返す。`ctx` は当該フレームのフレーム番号とタイムスタンプ。
    fn process(&mut self, frame: Frame, ctx: FrameCtx) -> Frame;
}

impl<F: FnMut(Frame, FrameCtx) -> Frame> Process for F {
    fn process(&mut self, frame: Frame, ctx: FrameCtx) -> Frame {
        (self)(frame, ctx)
    }
}

impl Process for () {
    fn process(&mut self, frame: Frame, _ctx: FrameCtx) -> Frame {
        frame
    }
}

//! 同梱の [`Process`](super::Process) ノード実装。

use super::{Process, ProcessCtx};
use crate::frame::Frame;

/// 純粋ピクセル変換（1:1, 状態なし）。RGB を反転し、alpha はそのまま残す。
pub struct Invert;

impl Process for Invert {
    fn process(&mut self, mut frame: Frame, _ctx: ProcessCtx) -> Frame {
        for px in frame.data.chunks_exact_mut(4) {
            px[0] = 255 - px[0];
            px[1] = 255 - px[1];
            px[2] = 255 - px[2];
            // px[3] (alpha) はそのまま
        }
        frame
    }
}

/// 時間方向の畳み込み。直前フレームと現フレームをブレンドする（`&mut self` に状態保持）。
pub struct MotionBlur {
    /// 直前フレームの RGBA8 バッファ。初回は `None`。
    pub prev: Option<Vec<u8>>,
    /// 現フレームの重み（0.0〜1.0）。`1.0 - alpha` が直前フレームの重み。
    pub alpha: f32,
}

impl Process for MotionBlur {
    fn process(&mut self, mut frame: Frame, _ctx: ProcessCtx) -> Frame {
        if let Some(prev) = &self.prev {
            if prev.len() == frame.data.len() {
                for (c, p) in frame.data.iter_mut().zip(prev) {
                    *c = (*c as f32 * self.alpha + *p as f32 * (1.0 - self.alpha)) as u8;
                }
            }
        }
        self.prev = Some(frame.data.clone());
        frame
    }
}

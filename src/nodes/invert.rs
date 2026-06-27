use video_pipeline::{Frame, FrameCtx, Process};

/// 純粋ピクセル変換（1:1, 状態なし）。RGB を反転し、alpha はそのまま残す。
pub struct Invert;

impl Process for Invert {
    fn process(&mut self, mut frame: Frame, _ctx: FrameCtx) -> Frame {
        for y in 0..frame.height() {
            for x in 0..frame.width() {
                let px = frame.get_pixel(x, y);
                frame.set_pixel(x, y, px.invert());
            }
        }
        frame
    }
}

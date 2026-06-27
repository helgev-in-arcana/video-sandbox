use video_pipeline::{Frame, FrameCtx, Process};

/// 時間方向の畳み込み。直前フレームと現フレームをブレンドする（`&mut self` に状態保持）。
pub struct MotionBlur {
    /// 直前フレームの RGBA8 バッファ。初回は `None`。
    pub prev: Option<Frame>,
    /// 現フレームの重み（0.0〜1.0）。`1.0 - alpha` が直前フレームの重み。
    pub alpha: f32,
}

impl Process for MotionBlur {
    fn process(&mut self, mut frame: Frame, _ctx: FrameCtx) -> Frame {
        // if let Some(prev) = &self.prev {
        //     if prev.len() == frame.data.len() {
        //         for (c, p) in frame.data.iter_mut().zip(prev) {
        //             *c = (*c as f32 * self.alpha + *p as f32 * (1.0 - self.alpha)) as u8;
        //         }
        //     }
        // }
        // self.prev = Some(frame.data.clone());
        // frame

        let Some(prev) = self.prev.as_ref() else {
            self.prev = Some(frame.clone());
            return frame;
        };
        let w = frame.width().min(prev.width());
        let h = frame.height().min(prev.height());
        for y in 0..h {
            for x in 0..w {
                let prev_px = prev.get_pixel(x, y);
                let frame_px = frame.get_pixel(x, y);

                // prev*(1-alpha) + frame*alpha = lerp(prev, frame, alpha)
                frame.set_pixel(x, y, prev_px.lerp(frame_px, self.alpha));
            }
        }
        self.prev = Some(frame.clone());
        frame
    }
}

use video_pipeline::{Frame, FrameCtx, Pipeline};

/// 1 枚の静止 [`Frame`] を、指定フレーム数ぶん繰り返す動画ソース。
///
/// 取り出すたびに元フレームを複製し、[`FrameCtx`] を `index` 0,1,2,… と振り直す。
/// `pts` も `index` と同値、`seconds` は `index / fps` で算出する。`fps` は既定 30 で、
/// [`with_fps`](Self::with_fps) で変更できる（ここでの `fps` は秒数計算用であり、
/// 実際の出力フレームレートはエンコード時の [`EncodeSettings`](video_pipeline::EncodeSettings)
/// で決まる。両者を揃えると時間軸が一致する）。
///
/// # 使用例
///
/// ```no_run
/// use video_pipeline::{EncodeSettings, Pipeline};
/// use video_sandbox::framegen::ImageFrame;
/// use video_sandbox::videogen::StillVideo;
///
/// let frame = ImageFrame::load("logo.png")?;
/// // 画像を 90 フレーム（30fps なら 3 秒）の動画にする。
/// StillVideo::new(frame, 90)
///     .encode_to("out.mp4", EncodeSettings::default())?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct StillVideo {
    /// 複製元の静止フレーム。
    frame: Frame,
    /// 生成する総フレーム数。
    frames: u64,
    /// 秒数計算に使うフレームレート。
    fps: f32,
    /// 次に発行するフレーム番号。
    index: u64,
}

impl StillVideo {
    /// `frame` を `frames` 枚ぶん繰り返す動画ソースを構築する（既定 30fps）。
    pub fn new(frame: Frame, frames: u64) -> Self {
        Self { frame, frames, fps: 30.0, index: 0 }
    }

    /// 秒数計算に使うフレームレートを設定する。
    pub fn with_fps(mut self, fps: f32) -> Self {
        self.fps = fps;
        self
    }
}

impl Pipeline for StillVideo {
    fn next_frame(&mut self) -> Option<Frame> {
        if self.index >= self.frames {
            return None;
        }
        let mut frame = self.frame.clone();
        frame.set_ctx(FrameCtx {
            index: self.index,
            pts: self.index as i64,
            seconds: self.index as f32 / self.fps,
        });
        self.index += 1;
        Some(frame)
    }

    fn size_hint(&self) -> Option<u64> {
        Some(self.frames)
    }
}

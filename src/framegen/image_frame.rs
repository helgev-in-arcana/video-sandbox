use anyhow::{Context, Result};
use video_pipeline::{Frame, FrameCtx};

/// 画像ファイルを読み込んで 1 枚の [`Frame`] にデコードするローダ。
///
/// `image` クレートが対応する形式（PNG・JPEG・BMP・GIF の先頭フレームなど）を、
/// 内部表現の RGBA8 に正規化して取り込む。生成される [`Frame`] の [`FrameCtx`] は
/// 先頭フレーム相当（`index = 0`, `pts = 0`, `seconds = 0.0`）で初期化される。連番の
/// 動画にするときは [`crate::videogen`] 側で文脈を振り直す。
///
/// # 使用例
///
/// ```no_run
/// use video_sandbox::framegen::ImageFrame;
///
/// let frame = ImageFrame::load("logo.png")?;
/// assert!(frame.width() > 0);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct ImageFrame;

impl ImageFrame {
    /// `path` の画像を読み込み、RGBA8 の [`Frame`] にして返す。
    ///
    /// 読み込みやデコードに失敗した場合は [`Err`] を返す（ソースが無謬な動画デコードと
    /// 異なり、画像はパスが明示指定されるため呼び出し側でハンドリングできるようにする）。
    pub fn load(path: &str) -> Result<Frame> {
        let img = image::open(path)
            .with_context(|| format!("画像 '{path}' を開けませんでした"))?
            .to_rgba8();
        let (width, height) = img.dimensions();
        let ctx = FrameCtx { index: 0, pts: 0, seconds: 0.0 };
        Ok(Frame::from_rgba_bytes(width, height, img.as_raw(), ctx))
    }
}

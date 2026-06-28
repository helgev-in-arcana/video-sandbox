use anyhow::{bail, Result};
use std::sync::mpsc::sync_channel;

use crate::ffmpeg::{decode_iter, Encoder};
use crate::frame::Frame;
use crate::process::Process;

// ─── エンコード／デコード設定 ────────────────────────────────────────────────

/// 使用する H.264 エンコーダ。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VideoEncoder {
    /// libx264（ソフト）。既定。
    #[default]
    Software,
    /// NVIDIA NVENC（`h264_nvenc`）。
    Nvenc,
    /// Intel Quick Sync（`h264_qsv`）。
    Qsv,
    /// AMD AMF（`h264_amf`）。
    Amf,
}

impl VideoEncoder {
    pub(crate) fn codec_name(self) -> &'static str {
        match self {
            VideoEncoder::Software => "libx264",
            VideoEncoder::Nvenc => "h264_nvenc",
            VideoEncoder::Qsv => "h264_qsv",
            VideoEncoder::Amf => "h264_amf",
        }
    }
}

/// HW デコードの種類。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HwAccel {
    /// Direct3D 11 VA（Windows 汎用・全ベンダ）。
    D3d11va,
    /// DirectX VA 2（Windows・旧）。
    Dxva2,
    /// NVIDIA CUDA。
    Cuda,
    /// Intel Quick Sync。
    Qsv,
    /// VA-API（Linux・Intel/AMD）。
    Vaapi,
}

/// エンコード設定。実験用に必要十分な最小限。
pub struct EncodeSettings {
    /// 出力フレームレート（fps）。
    pub fps: i32,
    /// 目標ビットレート（bit/s）。
    pub bit_rate: i64,
    /// 使用するエンコーダ。既定は [`VideoEncoder::Software`]（libx264）。
    pub encoder: VideoEncoder,
}

impl Default for EncodeSettings {
    /// 30fps・8 Mbps・ソフトエンコード（libx264）。
    fn default() -> Self {
        EncodeSettings { fps: 30, bit_rate: 8_000_000, encoder: VideoEncoder::default() }
    }
}

/// デコード設定。
pub struct DecodeSettings {
    /// HW デコードの種類。`None` ならソフトデコード。
    pub hwaccel: Option<HwAccel>,
}

impl Default for DecodeSettings {
    /// ソフトデコード。
    fn default() -> Self {
        DecodeSettings { hwaccel: None }
    }
}

// ─── Pipeline トレイト ───────────────────────────────────────────────────────

/// フレームの pull 型ストリーム。
///
/// [`std::iter::Iterator`] と同じ構造で、アダプタが上流ソースをジェネリクスで所有する。
/// この設計により複数入力の合成・分岐・スレッド境界を型として自然に表現できる。複数入力を
/// 合成する Mix ノードは、2 つの `impl Pipeline` を受け取りそれ自身が `Pipeline` を実装する
/// 形でユーザ側（ノード）に実装できる。
///
/// # 使用例
///
/// ```no_run
/// use video_pipeline::{VideoFile, EncodeSettings, Pipeline};
///
/// VideoFile::new("a.mp4")
///     .buffered(4)                         // デコードを別スレッドへ
///     .pipe(|f, _ctx| f)                   // 処理ノード（恒等）
///     .encode_to("out.mp4", EncodeSettings::default())?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub trait Pipeline {
    /// 次のフレームを取り出す。枯渇したら `None`。
    fn next_frame(&mut self) -> Option<Frame>;

    /// 総フレーム数のヒント（不明なら `None`。進捗表示用）。
    fn size_hint(&self) -> Option<u64> {
        None
    }

    /// [`Process`] ノードを 1 段連結する。`Iterator::map` 相当。
    fn pipe<P: Process>(self, p: P) -> Piped<Self, P>
    where
        Self: Sized,
    {
        Piped { source: self, process: p }
    }

    /// 上流をバックグラウンドスレッドに移し、容量 `cap` の有界チャネルで繋ぐ。
    ///
    /// デコードや重い処理段を別スレッドに退避して、処理とエンコードをオーバーラップさせる。
    /// 複数の `.buffered()` を挟むことで 3 段以上のパイプライン並列も組める。
    fn buffered(self, cap: usize) -> Buffered
    where
        Self: Sized + Send + 'static,
    {
        let total = self.size_hint();
        let (tx, rx) = sync_channel(cap);
        std::thread::spawn(move || {
            let mut src = self;
            while let Some(frame) = src.next_frame() {
                if tx.send(frame).is_err() {
                    break;
                }
            }
        });
        Buffered { rx, total }
    }

    /// `Box<dyn Pipeline>` へ型消去する。型名が長くなりすぎた時の逃げ道。
    fn boxed(self) -> Box<dyn Pipeline>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }

    /// ソースを消費し、全フレームを `path` に H.264 動画として書き出す。
    ///
    /// エンコードは専用スレッドで並列実行される。上流に `.buffered()` を挟むと
    /// デコード・処理・エンコードを 3 スレッドで重ねられる。
    fn encode_to(mut self, path: &str, settings: EncodeSettings) -> Result<()>
    where
        Self: Sized,
    {
        use std::io::Write;

        let (tx, rx) = sync_channel::<Frame>(4);
        let path = path.to_string();

        let enc_handle = std::thread::spawn(move || -> Result<u64> {
            let mut enc: Option<Encoder> = None;
            let mut count = 0u64;
            for frame in rx {
                let enc = match &mut enc {
                    Some(e) => e,
                    None => {
                        enc.insert(Encoder::new(&path, frame.width(), frame.height(), &settings)?)
                    }
                };
                enc.encode(frame)?;
                count += 1;
            }
            match enc {
                Some(e) => e.finish()?,
                None => bail!("ソースからフレームが 1 枚も得られませんでした"),
            }
            Ok(count)
        });

        let total = self.size_hint();
        let mut index = 0u64;
        while let Some(frame) = self.next_frame() {
            if tx.send(frame).is_err() {
                break;
            }
            index += 1;
            let mut err = std::io::stderr().lock();
            match total {
                Some(t) => {
                    let _ = write!(
                        err,
                        "\r[進捗] {index}/{t} ({:.1}%)",
                        index as f64 / t as f64 * 100.0
                    );
                }
                None => {
                    let _ = write!(err, "\r[進捗] {index} フレーム");
                }
            }
            let _ = err.flush();
        }
        eprintln!();
        drop(tx);

        enc_handle
            .join()
            .map_err(|_| anyhow::anyhow!("エンコードスレッドが panic しました"))?
            .map(|_| ())
    }
}

impl Pipeline for Box<dyn Pipeline> {
    fn next_frame(&mut self) -> Option<Frame> {
        (**self).next_frame()
    }
    fn size_hint(&self) -> Option<u64> {
        (**self).size_hint()
    }
}

// ─── アダプタ構造体 ──────────────────────────────────────────────────────────

/// [`Pipeline::pipe`] で生成される 1 段変換アダプタ。
pub struct Piped<S, P> {
    source: S,
    process: P,
}

impl<S: Pipeline, P: Process> Pipeline for Piped<S, P> {
    fn next_frame(&mut self) -> Option<Frame> {
        let frame = self.source.next_frame()?;
        let ctx = frame.ctx();
        Some(self.process.process(frame, ctx))
    }
    fn size_hint(&self) -> Option<u64> {
        self.source.size_hint()
    }
}

/// [`Pipeline::buffered`] で生成されるスレッド境界アダプタ。上流はバックグラウンドスレッドで走る。
pub struct Buffered {
    rx: std::sync::mpsc::Receiver<Frame>,
    total: Option<u64>,
}

impl Pipeline for Buffered {
    fn next_frame(&mut self) -> Option<Frame> {
        self.rx.recv().ok()
    }
    fn size_hint(&self) -> Option<u64> {
        self.total
    }
}

// ─── デコードソース ──────────────────────────────────────────────────────────

/// 動画ファイルをデコードする [`Pipeline`]。[`VideoFile`] で生成する。
pub struct Decode {
    inner: Box<dyn Iterator<Item = Frame> + Send>,
    total: Option<u64>,
    index: u64,
}

impl Pipeline for Decode {
    fn next_frame(&mut self) -> Option<Frame> {
        let mut frame = self.inner.next()?;
        frame.set_index(self.index);
        self.index += 1;
        Some(frame)
    }
    fn size_hint(&self) -> Option<u64> {
        self.total
    }
}

/// 動画ファイルソース。[`Decode`] を生成するファクトリ。
pub struct VideoFile;

impl VideoFile {
    /// `path` をソフトデコードで開く。
    pub fn new(path: &str) -> Decode {
        Self::open(path, DecodeSettings::default())
    }

    /// `path` を [`DecodeSettings`] で開く。HW デコードはここで選択する。
    pub fn open(path: &str, settings: DecodeSettings) -> Decode {
        let (inner, total) = decode_iter(path, settings.hwaccel);
        Decode { inner, total, index: 0 }
    }
}

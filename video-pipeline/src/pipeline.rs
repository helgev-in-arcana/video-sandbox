//! パイプラインのビルダーと駆動ループ。
//!
//! `VideoFile::new(...).pipe(A).pipe(B).encode_to(...)` の形で組む。ディスパッチは
//! 動的（`Box<dyn Process>`）。1:1 に確定しているので駆動ループは単なる `fold`。
//! プール・ping-pong・flush 伝播・Send 境界はいずれも持たない。

use anyhow::{anyhow, bail, Result};

use crate::ffmpeg::{decode_iter, Encoder};
use crate::frame::{Frame, FrameCtx};
use crate::process::Process;

/// エンコード設定。実験用に必要十分な最小限。
pub struct EncodeSettings {
    /// 出力フレームレート（fps）。出力 pts はフレーム番号で振り直す。
    pub fps: i32,
    /// 目標ビットレート（bit/s）。
    pub bit_rate: i64,
    /// 使用する H.264 エンコーダ名。`None` なら `libx264`（ソフト）。
    /// 例: `Some("h264_amf")`（AMD HW）/ `Some("h264_nvenc")` / `Some("h264_qsv")`。
    pub encoder: Option<String>,
}

impl Default for EncodeSettings {
    /// 30fps・8 Mbps・ソフトエンコード（libx264）。
    fn default() -> Self {
        EncodeSettings { fps: 30, bit_rate: 8_000_000, encoder: None }
    }
}

/// デコード設定。
pub struct DecodeSettings {
    /// HW デコードの種類。`None` ならソフトデコード。
    /// 例: `Some("d3d11va")`（AMD/Windows 汎用）/ `Some("dxva2")`。
    pub hwaccel: Option<String>,
}

impl Default for DecodeSettings {
    /// ソフトデコード。
    fn default() -> Self {
        DecodeSettings { hwaccel: None }
    }
}

/// ソース（フレーム列）と連結された [`Process`] ノード列を保持するビルダー兼実行器。
///
/// [`VideoFile::new`] で作り、[`Pipeline::pipe`] / [`Pipeline::map`] で組み立て、
/// [`Pipeline::encode_to`] で消費する。
pub struct Pipeline {
    source: Box<dyn Iterator<Item = Frame> + Send>,
    stages: Vec<Box<dyn Process>>,
}

impl Pipeline {
    /// [`Process`] ノードを末尾に連結する。
    pub fn pipe(mut self, p: impl Process + 'static) -> Self {
        self.stages.push(Box::new(p));
        self
    }

    /// ソースを消費し、全ステージを通して `path` に動画として書き出す。
    ///
    /// 3 段パイプライン: **デコード**（専用スレッド）→ **処理**（このスレッド + 内部 rayon）
    /// → **エンコード**（専用スレッド）を有界チャネルで繋ぎ、ffmpeg の I/O を処理と重ねる。
    /// 処理段は順序を保った単一消費者なので、状態付きノード（`Feedback` 等）の結果は
    /// 逐次実行と一致する。エンコーダは最初のフレームの寸法で開かれ、フレームが 1 枚も
    /// 得られなければエラーを返す。
    pub fn encode_to(self, path: &str, settings: EncodeSettings) -> Result<()> {
        use std::sync::mpsc::sync_channel;
        use std::time::{Duration, Instant};

        // 有界チャネルの容量。小さめにしてバックプレッシャーとメモリ上限を効かせる。
        const CAP: usize = 4;

        let Pipeline { mut source, mut stages } = self;
        let path = path.to_string();

        let (dec_tx, dec_rx) = sync_channel::<Frame>(CAP);
        let (enc_tx, enc_rx) = sync_channel::<Frame>(CAP);

        // --- デコード専用スレッド: source.next() の結果を流すだけ ---
        let dec_handle = std::thread::spawn(move || {
            let mut busy = Duration::ZERO;
            loop {
                let t = Instant::now();
                let frame = source.next();
                busy += t.elapsed();
                match frame {
                    // 送信ブロック（バックプレッシャー）は busy に含めない。
                    Some(f) => {
                        if dec_tx.send(f).is_err() {
                            break; // 下流が落ちた
                        }
                    }
                    None => break,
                }
            }
            busy
        });

        // --- エンコード専用スレッド: 最初のフレームでエンコーダを開き、流れてくる順に書く ---
        let enc_handle = std::thread::spawn(move || -> Result<(Duration, u64)> {
            let mut enc: Option<Encoder> = None;
            let mut busy = Duration::ZERO;
            let mut frames = 0u64;
            for frame in enc_rx {
                let enc = match &mut enc {
                    Some(e) => e,
                    None => {
                        enc.insert(Encoder::new(&path, frame.width(), frame.height(), &settings)?)
                    }
                };
                let t = Instant::now();
                enc.encode(frame)?;
                busy += t.elapsed();
                frames += 1;
            }
            match enc {
                Some(e) => e.finish()?,
                None => bail!("ソースからフレームが 1 枚も得られませんでした"),
            }
            Ok((busy, frames))
        });

        // --- 処理段（このスレッド）: デコード結果を順に受け、全ステージを通して送る ---
        let mut t_process = Duration::ZERO;
        let mut index = 0u64;
        for frame in dec_rx {
            let ctx = FrameCtx { index, pts: frame.pts() };
            let t = Instant::now();
            let out = stages.iter_mut().fold(frame, |f, s| s.process(f, ctx));
            t_process += t.elapsed();
            if enc_tx.send(out).is_err() {
                break; // エンコードスレッドが落ちた
            }
            index += 1;
        }
        drop(enc_tx); // チャネルを閉じてエンコードスレッドを終了させる

        let t_decode = dec_handle
            .join()
            .map_err(|_| anyhow!("デコードスレッドが panic しました"))?;
        let (t_encode, frames) = enc_handle
            .join()
            .map_err(|_| anyhow!("エンコードスレッドが panic しました"))??;

        eprintln!(
            "[計測] {frames} フレーム（各段の busy 時間）: \
             デコード {:.2}s / 処理 {:.2}s / エンコード {:.2}s",
            t_decode.as_secs_f32(),
            t_process.as_secs_f32(),
            t_encode.as_secs_f32(),
        );

        Ok(())
    }
}

/// 動画ファイルソース。
///
/// `rsmpeg` は [`VideoFile::new`]（ソース）と [`Pipeline::encode_to`]（シンク）の内部に
/// 閉じ込められ、中間の [`Process`] は `rsmpeg` に一切依存しない。
pub struct VideoFile;

impl VideoFile {
    /// `path` をソフトデコードで開く（[`DecodeSettings::default`]）。
    ///
    /// 開けなかった場合でも panic せず、フレームを 1 枚も yield しない空のソースになる
    /// （実際のエラーは標準エラー出力に表示され、[`Pipeline::encode_to`] がエラーを返す）。
    pub fn new(path: &str) -> Pipeline {
        Self::open(path, DecodeSettings::default())
    }

    /// `path` のデコーダを [`DecodeSettings`] で開き、各フレームを RGBA8 化して yield する
    /// [`Pipeline`] を作る。HW デコードの選択はここで行う。
    pub fn open(path: &str, settings: DecodeSettings) -> Pipeline {
        Pipeline { source: decode_iter(path, settings.hwaccel.as_deref()), stages: vec![] }
    }
}

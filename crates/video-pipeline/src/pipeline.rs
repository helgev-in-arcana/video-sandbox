//! パイプラインのビルダーと駆動ループ。
//!
//! `VideoFile::new(...).pipe(A).pipe(B).encode_to(...)` の形で組む。ディスパッチは
//! 動的（`Box<dyn Process>`）。1:1 に確定しているので駆動ループは単なる `fold`。
//! プール・ping-pong・flush 伝播・Send 境界はいずれも持たない。

use anyhow::{bail, Result};

use crate::ffmpeg::{decode_iter, Encoder};
use crate::frame::Frame;
use crate::process::{MapNode, Process, ProcessCtx};

/// エンコード設定。実験用に必要十分な最小限。
pub struct EncodeSettings {
    /// 出力フレームレート（fps）。出力 pts はフレーム番号で振り直す。
    pub fps: i32,
    /// 目標ビットレート（bit/s）。
    pub bit_rate: i64,
}

impl Default for EncodeSettings {
    /// 30fps・8 Mbps。
    fn default() -> Self {
        EncodeSettings { fps: 30, bit_rate: 8_000_000 }
    }
}

/// ソース（フレーム列）と連結された [`Process`] ノード列を保持するビルダー兼実行器。
///
/// [`VideoFile::new`] で作り、[`Pipeline::pipe`] / [`Pipeline::map`] で組み立て、
/// [`Pipeline::encode_to`] で消費する。
pub struct Pipeline {
    source: Box<dyn Iterator<Item = Frame>>,
    stages: Vec<Box<dyn Process>>,
}

impl Pipeline {
    /// [`Process`] ノードを末尾に連結する。
    pub fn pipe(mut self, p: impl Process + 'static) -> Self {
        self.stages.push(Box::new(p));
        self
    }

    /// 使い捨てのインライン実験用ヘルパ。クロージャを [`Process`] として挿す。
    ///
    /// クロージャは `(Frame, ProcessCtx) -> Frame` で、[`ProcessCtx`] からフレーム番号
    /// やタイムスタンプを参照できる。
    pub fn map(self, f: impl FnMut(Frame, ProcessCtx) -> Frame + 'static) -> Self {
        self.pipe(MapNode(f))
    }

    /// ソースを消費し、全ステージを通して `path` に動画として書き出す。
    ///
    /// 駆動ループの本体は `fold` 一行で、所有権がステージ間を move で流れていく。
    /// エンコーダは最初のフレームの寸法で開かれる。フレームが 1 枚も得られなければ
    /// エラーを返す。
    pub fn encode_to(mut self, path: &str, settings: EncodeSettings) -> Result<()> {
        let mut enc: Option<Encoder> = None;
        for (index, frame) in self.source.enumerate() {
            let ctx = ProcessCtx { index: index as u64, pts: frame.pts };
            let out = self.stages.iter_mut().fold(frame, |f, s| s.process(f, ctx));
            // エンコーダは最初のフレームの寸法で開く（resize ノードにも追従できる）。
            let enc = match &mut enc {
                Some(e) => e,
                None => enc.insert(Encoder::new(path, out.width, out.height, &settings)?),
            };
            enc.encode(out)?;
        }
        match enc {
            Some(e) => e.finish(),
            None => bail!("ソースからフレームが 1 枚も得られませんでした"),
        }
    }
}

/// 動画ファイルソース。
///
/// `rsmpeg` は [`VideoFile::new`]（ソース）と [`Pipeline::encode_to`]（シンク）の内部に
/// 閉じ込められ、中間の [`Process`] は `rsmpeg` に一切依存しない。
pub struct VideoFile;

impl VideoFile {
    /// `path` のデコーダを開き、各フレームを RGBA8 化して yield する [`Pipeline`] を作る。
    ///
    /// 開けなかった場合でも panic せず、フレームを 1 枚も yield しない空のソースになる
    /// （実際のエラーは標準エラー出力に表示され、[`Pipeline::encode_to`] がエラーを返す）。
    pub fn new(path: &str) -> Pipeline {
        Pipeline { source: decode_iter(path), stages: vec![] }
    }
}

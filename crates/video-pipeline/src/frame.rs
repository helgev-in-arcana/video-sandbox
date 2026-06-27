//! 中間フレーム表現。入出力境界で RGBA8 に正規化され、内部は常にこの単一形式。

use rayon::prelude::*;

use crate::pixel::Pixel;

/// 1 枚のフレーム。
///
/// 内部バッファは行優先（x 軸方向に連続）の [`Pixel`] 列で、len = `width * height`。
/// ピクセル `(x, y)` は `data[y * width + x]` に対応する。添字を直接触る代わりに
/// [`Frame::get_pixel`] / [`Frame::set_pixel`] / [`Frame::per_iter_row`] を使う。
/// I/O 境界向けにバイト列としては [`Frame::data`] で `&[u8]`（RGBA8）を取り出せる。
#[derive(Clone)]
pub struct Frame {
    /// 幅（ピクセル）。
    width: u32,
    /// 高さ（ピクセル）。
    height: u32,
    /// 行優先のピクセル列。`data[y * width + x]` でアクセスする。
    data: Vec<Pixel>,
    /// `rsmpeg` から引き継ぐ presentation timestamp（入力ストリームの time_base 基準）。
    pts: i64,
}

impl Frame {
    /// 幅（ピクセル）。
    pub fn width(&self) -> u32 {
        self.width
    }

    /// 高さ（ピクセル）。
    pub fn height(&self) -> u32 {
        self.height
    }

    /// presentation timestamp（入力ストリームの time_base 基準）。
    pub fn pts(&self) -> i64 {
        self.pts
    }

    /// 内部バッファを RGBA8 のバイト列として借りる（I/O 境界向け）。
    ///
    /// `Vec<Pixel>` を [`bytemuck`] でコピーなしに `&[u8]`（len = `width * height * 4`）へ
    /// 再解釈する。
    pub fn data(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data)
    }
}

impl Frame {
    /// 既存のピクセル列からフレームを構築する。
    ///
    /// `data` は行優先で len = `width * height` でなければならない。
    pub(crate) fn from_rgba(width: u32, height: u32, data: Vec<Pixel>, pts: i64) -> Self {
        debug_assert_eq!(data.len(), (width as usize) * (height as usize));
        Frame { width, height, data, pts }
    }

    /// 指定サイズの黒（RGB=0, alpha=255）フレームを確保する。
    pub fn black(width: u32, height: u32, pts: i64) -> Self {
        let data = vec![Pixel::BLACK; (width as usize) * (height as usize)];
        Frame { width, height, data, pts }
    }

    /// `(x, y)` のピクセル添字。範囲外はデバッグビルドで panic。
    #[inline]
    fn offset(&self, x: u32, y: u32) -> usize {
        debug_assert!(x < self.width && y < self.height, "ピクセル ({x}, {y}) は範囲外です");
        (y as usize * self.width as usize) + x as usize
    }

    /// `(x, y)` の RGBA ピクセルを読む。
    ///
    /// 範囲外アクセスはデバッグビルドで panic（リリースビルドでは `Vec` の
    /// 境界チェックに委ねる）。
    #[inline]
    pub fn get_pixel(&self, x: u32, y: u32) -> Pixel {
        self.data[self.offset(x, y)]
    }

    /// `(x, y)` に RGBA ピクセルを書く。
    ///
    /// 範囲外アクセスはデバッグビルドで panic（リリースビルドでは `Vec` の
    /// 境界チェックに委ねる）。
    #[inline]
    pub fn set_pixel(&mut self, x: u32, y: u32, px: Pixel) {
        let i = self.offset(x, y);
        self.data[i] = px;
    }

    /// 行と列を入れ替えた新しいフレームを返す（`width` と `height` が入れ替わる）。
    ///
    /// `dst[x][y] = src[y][x]`。ピクセル単位でコピーする愚直な実装で、大きなフレームでは
    /// キャッシュ効率が悪い（タイル化は将来の最適化対象）。`pts` は引き継ぐ。
    pub fn transposed(&self) -> Frame {
        let w = self.width as usize;
        let h = self.height as usize;
        let mut data = vec![Pixel::default(); w * h];
        for y in 0..h {
            for x in 0..w {
                data[x * h + y] = self.data[y * w + x];
            }
        }
        // 転置後は幅と高さが入れ替わる。
        Frame { width: self.height, height: self.width, data, pts: self.pts }
    }

    /// 各行（x 軸方向に連続するピクセル列）を並列に走査し、行ごとに `f` を適用する。
    ///
    /// `f` には当該フレームの [`FrameCtx`]、行番号 `y`、その行のピクセル列全体への
    /// 可変参照 `&mut [Pixel]`（長さ `width`）が渡る。行どうしは独立に別スレッドで
    /// 処理され得るため、`f` は `Fn + Sync` であり、行内で完結する変更のみ行える
    /// （行内ソートなど）。行をまたぐ参照や共有可変状態は扱えない。
    ///
    /// 列方向に処理したい場合は [`Frame::transposed`] してから本メソッドを呼び、
    /// 必要なら再度転置して戻す。
    pub fn per_iter_row<F>(&mut self, ctx: &FrameCtx, f: F)
    where
        F: Fn(&FrameCtx, usize, &mut [Pixel]) + Sync,
    {
        let width = self.width as usize;
        self.data
            .par_chunks_mut(width)
            .enumerate()
            .for_each(|(y, row)| f(ctx, y, row));
    }
}

/// 各フレームに付随する文脈。ノードがフレーム番号やタイムスタンプを参照できる。
#[derive(Clone, Copy, Debug)]
pub struct FrameCtx {
    /// 0 始まりのフレーム番号（ソースから取り出した順）。
    pub index: u64,
    /// presentation timestamp（入力ストリームの time_base 基準）。[`Frame::pts`] と同値。
    pub pts: i64,
}

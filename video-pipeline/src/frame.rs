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
    /// フレームに付随する文脈（タイムスタンプ・秒数・フレーム番号）。
    ctx: FrameCtx,
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

    /// フレームに付随する文脈（タイムスタンプ・秒数・フレーム番号）。
    pub fn ctx(&self) -> FrameCtx {
        self.ctx
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
    pub(crate) fn from_rgba(width: u32, height: u32, data: Vec<Pixel>, ctx: FrameCtx) -> Self {
        debug_assert_eq!(data.len(), (width as usize) * (height as usize));
        Frame { width, height, data, ctx }
    }

    /// 指定サイズの黒（RGB=0, alpha=255）フレームを確保する。
    pub fn black(width: u32, height: u32, ctx: FrameCtx) -> Self {
        let data = vec![Pixel::BLACK; (width as usize) * (height as usize)];
        Frame { width, height, data, ctx }
    }

    /// RGBA8 のバイト列（行優先・パック済み、len = `width * height * 4`）からフレームを構築する。
    ///
    /// 画像デコーダ等、フレームワーク外で得た生バイトを取り込むための公開コンストラクタ。
    ///
    /// # パニック
    ///
    /// `bytes.len()` が `width * height * 4` でないとき。
    pub fn from_rgba_bytes(width: u32, height: u32, bytes: &[u8], ctx: FrameCtx) -> Self {
        let count = (width as usize) * (height as usize);
        assert_eq!(
            bytes.len(),
            count * 4,
            "バイト列の長さ {} が width*height*4 = {} と一致しません",
            bytes.len(),
            count * 4
        );
        let data = bytemuck::cast_slice(bytes).to_vec();
        Frame { width, height, data, ctx }
    }

    /// フレームに付随する文脈を差し替える。
    ///
    /// 1 枚の静止フレームから連番の動画を生成する場合など、同じピクセルのまま
    /// [`FrameCtx`]（フレーム番号・秒数）だけを更新したいときに使う。
    pub fn set_ctx(&mut self, ctx: FrameCtx) {
        self.ctx = ctx;
    }

    /// フレーム番号を設定する（デコーダは 0 を埋め、パイプラインが後から確定させる）。
    pub(crate) fn set_index(&mut self, index: u64) {
        self.ctx.index = index;
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
    /// `dst[x][y] = src[y][x]`。出力を `T` 行ずつの帯に分割して **rayon で並列化**し、
    /// 各帯の中は `T×T` の **タイル単位**で処理してキャッシュミスを抑える。`pts` は引き継ぐ。
    pub fn transposed(&self) -> Frame {
        // タイルの一辺（ピクセル）。T×T×4B（T=64 で 16KB）で読み書きとも L1 に収まる。
        // 実測で 16/32/64 のうち 64 が最速だったため採用。
        const T: usize = 64;
        let w = self.width as usize; // src 幅（= dst の 1 行の長さ／高さ方向）
        let h = self.height as usize; // src 高さ（= dst の 1 行の長さ）
        let src = &self.data;
        // dst は width=h, height=w。行 x（src の列）、列 y（src の行）で dst[x*h + y]。
        // 全画素をちょうど 1 回ずつ書くので、ゼロ初期化は無駄。未初期化で確保する。
        // SAFETY: Pixel は Copy（Drop なし）。下のループが全 w*h 要素を読む前に書く。
        let mut data: Vec<Pixel> = Vec::with_capacity(w * h);
        unsafe { data.set_len(w * h) };

        // dst を「T 行ぶん」の帯に分割。帯どうしは互いに素な可変スライスなので並列で安全。
        data.par_chunks_mut(T * h).enumerate().for_each(|(b, band)| {
            let x0 = b * T; // この帯の先頭 dst 行 = src 列の開始
            let rows = band.len() / h; // 端の帯では < T
            // 列方向（src 行 = dst 列）も T ずつタイル化し、T×T ブロックを順に埋める。
            for yb in (0..h).step_by(T) {
                let y_end = (yb + T).min(h);
                for dx in 0..rows {
                    let x = x0 + dx; // src 列
                    let dst_row = &mut band[dx * h..dx * h + h];
                    for y in yb..y_end {
                        dst_row[y] = src[y * w + x];
                    }
                }
            }
        });

        // 転置後は幅と高さが入れ替わる。
        Frame { width: self.height, height: self.width, data, ctx: self.ctx }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// タイル化転置が `T[a][b] == F[b][a]` を満たすことを、タイル境界を跨ぐ非正方・
    /// 非 32 倍数サイズで検証する（端の帯 `rows < T` と部分タイル `y_end` を踏む）。
    #[test]
    fn transposed_matches_definition() {
        let (w, h) = (37u32, 53u32);
        let ctx = FrameCtx { index: 0, pts: 7, seconds: 0.7 };
        let mut f = Frame::black(w, h, ctx);
        for y in 0..h {
            for x in 0..w {
                f.set_pixel(x, y, Pixel::new(x as u8, y as u8, (x ^ y) as u8, 255));
            }
        }
        let t = f.transposed();
        assert_eq!(t.width(), h);
        assert_eq!(t.height(), w);
        assert_eq!(t.ctx().pts, f.ctx().pts);
        for y in 0..h {
            for x in 0..w {
                // 転置: T.get_pixel(y, x) == F.get_pixel(x, y)
                assert_eq!(t.get_pixel(y, x), f.get_pixel(x, y), "({x}, {y}) で不一致");
            }
        }
    }
}

/// 各フレームに付随する文脈。ノードがフレーム番号やタイムスタンプを参照できる。
#[derive(Clone, Copy, Debug)]
pub struct FrameCtx {
    /// 0 始まりのフレーム番号（ソースから取り出した順）。
    pub index: u64,
    /// presentation timestamp（入力ストリームの time_base 基準）。
    pub pts: i64,
    /// 動画先頭からの経過時間（秒）。`pts * time_base` で算出。
    pub seconds: f32,
}

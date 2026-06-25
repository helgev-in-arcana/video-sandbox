//! 中間フレーム表現。入出力境界で RGBA8 に正規化され、内部は常にこの単一形式。

/// 1 枚のフレーム。
///
/// `data` は RGBA8 で、stride = `width * 4`、len = `width * height * 4`。
/// ピクセル `(x, y)` のチャンネル `c` は `data[(y * width + x) * 4 + c]` に対応する。
/// 添字を直接触る代わりに [`Frame::get_pixel`] / [`Frame::set_pixel`] も使える。
pub struct Frame {
    /// 幅（ピクセル）。
    pub width: u32,
    /// 高さ（ピクセル）。
    pub height: u32,
    /// RGBA8 ピクセル列。`data[(y * width + x) * 4 + c]` でアクセスする。
    pub data: Vec<u8>,
    /// `rsmpeg` から引き継ぐ presentation timestamp（入力ストリームの time_base 基準）。
    pub pts: i64,
}

impl Frame {
    /// 指定サイズの黒（RGB=0, alpha=255）フレームを確保する。
    pub fn black(width: u32, height: u32, pts: i64) -> Self {
        let mut data = vec![0u8; (width as usize) * (height as usize) * 4];
        for px in data.chunks_exact_mut(4) {
            px[3] = 255;
        }
        Frame { width, height, data, pts }
    }

    /// `(x, y)` のバイトオフセット。範囲外はデバッグビルドで panic。
    #[inline]
    fn offset(&self, x: u32, y: u32) -> usize {
        debug_assert!(x < self.width && y < self.height, "ピクセル ({x}, {y}) は範囲外です");
        ((y as usize * self.width as usize) + x as usize) * 4
    }

    /// `(x, y)` の RGBA ピクセルを読む。
    ///
    /// 範囲外アクセスはデバッグビルドで panic（リリースビルドでは `Vec` の
    /// 境界チェックに委ねる）。
    #[inline]
    pub fn get_pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = self.offset(x, y);
        [self.data[i], self.data[i + 1], self.data[i + 2], self.data[i + 3]]
    }

    /// `(x, y)` に RGBA ピクセルを書く。
    ///
    /// 範囲外アクセスはデバッグビルドで panic（リリースビルドでは `Vec` の
    /// 境界チェックに委ねる）。
    #[inline]
    pub fn set_pixel(&mut self, x: u32, y: u32, px: [u8; 4]) {
        let i = self.offset(x, y);
        self.data[i..i + 4].copy_from_slice(&px);
    }
}

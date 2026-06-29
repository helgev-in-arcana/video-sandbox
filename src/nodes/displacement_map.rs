use video_pipeline::{Frame, FrameCtx, Pipeline, Process};

/// ディスプレイスメントマップ。`map` 側のフレームを変位量として読み、主入力フレームを
/// ピクセルごとにずらしてサンプリングする map 駆動の [`Process`]。
///
/// 各出力ピクセル `(x, y)` について、`map` の同位置のピクセルから変位 `(dx, dy)` を求め、
/// 主入力の `(x + dx, y + dy)` を読んで出力する。変位はマップの **赤チャンネルが X
/// 方向**、**緑チャンネルが Y 方向**に対応し、`0..=255` を `-1.0..=1.0` に正規化したうえで
/// [`scale_x`](Self::with_scale) / `scale_y`（ピクセル）を掛ける。よって中央値 `128` が変位
/// ゼロ、`0` で `-scale`、`255` で `+scale` となる。サンプリングは最近傍で、参照先が画面外に
/// はみ出す場合は端にクランプする。
///
/// 主入力は [`Process::process`] で受け、副入力 `map` を内部に [`Pipeline`] として所有して毎フレーム
/// `next_frame()` で引く。`map` が主入力より先に枯渇したら直近フレームを据え置く（コピー無し）。
/// 一度も map が来ない場合のみ主入力を素通しする。`map` を `.buffered()` しておけばデコードを
/// 別スレッドに重ねられる。
///
/// # 使用例
///
/// ```no_run
/// use video_pipeline::{VideoFile, EncodeSettings, Pipeline};
/// use video_sandbox::nodes::Displace;
///
/// // map.mp4 の RG チャンネルで source.mp4 を最大 ±30px ずらす。
/// VideoFile::new("source.mp4")
///     .buffered(4)
///     .pipe(Displace::new(VideoFile::new("map.mp4").buffered(4)).with_scale(30.0, 30.0))
///     .encode_to("out.mp4", EncodeSettings::default())?;
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct Displace<M> {
    map: M,
    /// 直近に取れた map フレーム（枯渇後はこれを据え置いて使う）。
    last_map: Option<Frame>,
    /// X 方向の最大変位（ピクセル）。マップ赤チャンネルに掛かる。
    scale_x: f32,
    /// Y 方向の最大変位（ピクセル）。マップ緑チャンネルに掛かる。
    scale_y: f32,
}

impl<M: Pipeline> Displace<M> {
    /// `map`（変位量を与える側）から変位ノードを構築する。
    ///
    /// 変位の強さは既定で X・Y とも 16px。[`with_scale`](Self::with_scale) で変更できる。
    pub fn new(map: M) -> Self {
        Self { map, last_map: None, scale_x: 16.0, scale_y: 16.0 }
    }

    /// 最大変位量（ピクセル）を X・Y 個別に設定する。
    pub fn with_scale(mut self, scale_x: f32, scale_y: f32) -> Self {
        self.scale_x = scale_x;
        self.scale_y = scale_y;
        self
    }
}

impl<M: Pipeline> Process for Displace<M> {
    fn process(&mut self, src: Frame, ctx: FrameCtx) -> Frame {
        // map が生きていれば前進、尽きていれば直近フレームを据え置く（ムーブのみ、コピー無し）。
        if let Some(m) = self.map.next_frame() {
            self.last_map = Some(m);
        }
        let Some(map) = &self.last_map else {
            return src; // 一度も map が来ていない時だけ素通し。
        };

        let w = src.width().min(map.width());
        let h = src.height().min(map.height());
        let mut out = Frame::black(w, h, ctx);
        if w == 0 || h == 0 {
            return out;
        }

        let (sx, sy) = (self.scale_x, self.scale_y);
        let (max_x, max_y) = ((w - 1) as f32, (h - 1) as f32);
        out.per_iter_row(&ctx, |_ctx, y, row| {
            let y = y as u32;
            for x in 0..row.len() as u32 {
                let m = map.get_pixel(x, y);
                // 0..=255 → -1.0..=1.0（128 で変位ゼロ）にして scale を掛ける。
                let dx = (m.r as f32 / 127.5 - 1.0) * sx;
                let dy = (m.g as f32 / 127.5 - 1.0) * sy;
                let sxp = (x as f32 + dx).round().clamp(0.0, max_x) as u32;
                let syp = (y as f32 + dy).round().clamp(0.0, max_y) as u32;
                row[x as usize] = src.get_pixel(sxp, syp);
            }
        });

        out
    }
}

// 変位の正規化が中央値で 0 になることを確認する軽いテスト。
#[cfg(test)]
mod tests {
    #[test]
    fn neutral_gray_is_zero_displacement() {
        let center = 128u8;
        let d = center as f32 / 127.5 - 1.0;
        assert!(d.abs() < 0.01, "中央値で変位がほぼ 0 でない: {d}");
    }
}

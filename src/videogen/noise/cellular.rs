use super::perm::Perm;

/// セルラーノイズの距離計量。特徴点までの「近さ」をどの距離で測るか。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CellDistance {
    /// ユークリッド距離 `sqrt(dx^2 + dy^2 + dz^2)`。丸い細胞状。
    #[default]
    Euclidean,
    /// マンハッタン距離 `|dx| + |dy| + |dz|`。菱形に尖る。
    Manhattan,
    /// チェビシェフ距離 `max(|dx|, |dy|, |dz|)`。矩形のセル。
    Chebyshev,
}

/// セルラーノイズが返す特徴量。最近傍距離 F1・第 2 近傍 F2 の組み合わせ。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum CellFeature {
    /// 最近傍距離 F1。点状の細胞パターン。
    #[default]
    F1,
    /// 第 2 近傍距離 F2。
    F2,
    /// `F2 - F1`。セル境界が稜線として浮き出る（ひび割れ／結晶状）。
    F2MinusF1,
}

/// セルラー（Worley）ノイズの設定。距離計量と返す特徴量を選ぶ。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Cellular {
    pub distance: CellDistance,
    pub feature: CellFeature,
}

impl Cellular {
    /// 既定（ユークリッド距離・F1）の設定を作る。
    pub const fn new() -> Self {
        Self { distance: CellDistance::Euclidean, feature: CellFeature::F1 }
    }

    /// 距離計量を差し替える。
    pub const fn distance(mut self, d: CellDistance) -> Self {
        self.distance = d;
        self
    }

    /// 返す特徴量を差し替える。
    pub const fn feature(mut self, f: CellFeature) -> Self {
        self.feature = f;
        self
    }
}

impl Perm {
    /// セルラー（Worley）3D ノイズ。概ね `[-1, 1]` に正規化して返す。
    ///
    /// 各整数セルに 1 個の特徴点をハッシュで散らし、評価点の周囲 3×3×3 セルから最近傍
    /// （F1）・第 2 近傍（F2）距離を求める。[`Cellular`] で距離計量と返す特徴量を選べる。
    pub fn cellular3(&self, x: f32, y: f32, z: f32, cfg: Cellular) -> f32 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let zi = z.floor() as i32;

        let mut f1 = f32::INFINITY;
        let mut f2 = f32::INFINITY;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (cx, cy, cz) = (xi + dx, yi + dy, zi + dz);
                    // セルをハッシュして特徴点を [0, 1)^3 のジッタで配置。
                    let (ox, oy, oz) = self.feature_point(cx, cy, cz);
                    let px = cx as f32 + ox;
                    let py = cy as f32 + oy;
                    let pz = cz as f32 + oz;
                    let d = distance(cfg.distance, px - x, py - y, pz - z);
                    if d < f1 {
                        f2 = f1;
                        f1 = d;
                    } else if d < f2 {
                        f2 = d;
                    }
                }
            }
        }

        let v = match cfg.feature {
            CellFeature::F1 => f1,
            CellFeature::F2 => f2,
            CellFeature::F2MinusF1 => f2 - f1,
        };
        // 距離（おおむね [0, 1] 強）を [-1, 1] に写す。clamp で外れ値を抑える。
        (v.clamp(0.0, 1.0) * 2.0 - 1.0).clamp(-1.0, 1.0)
    }

    /// セル `(cx, cy, cz)` の特徴点ジッタ `[0, 1)^3` を決定的に引く。
    #[inline]
    fn feature_point(&self, cx: i32, cy: i32, cz: i32) -> (f32, f32, f32) {
        let h = self.hash3(cx, cy, cz) as i32;
        // 同じセルから 3 軸ぶんの独立した擬似乱数を引く。
        let ox = self.hash(h) as f32 / 256.0;
        let oy = self.hash(h + 1) as f32 / 256.0;
        let oz = self.hash(h + 2) as f32 / 256.0;
        (ox, oy, oz)
    }
}

/// 2 点間の距離を計量に従って計算する。
#[inline]
fn distance(metric: CellDistance, dx: f32, dy: f32, dz: f32) -> f32 {
    match metric {
        CellDistance::Euclidean => (dx * dx + dy * dy + dz * dz).sqrt(),
        CellDistance::Manhattan => dx.abs() + dy.abs() + dz.abs(),
        CellDistance::Chebyshev => dx.abs().max(dy.abs()).max(dz.abs()),
    }
}

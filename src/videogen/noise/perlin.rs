use super::perm::{fade, lerp, Perm};

/// 3D 勾配（標準 12 方向 + 重複 4）とのドット積。古典 improved Perlin と同じ。
#[inline]
fn grad3(hash: usize, x: f32, y: f32, z: f32) -> f32 {
    match hash & 15 {
        0 => x + y,
        1 => -x + y,
        2 => x - y,
        3 => -x - y,
        4 => x + z,
        5 => -x + z,
        6 => x - z,
        7 => -x - z,
        8 => y + z,
        9 => -y + z,
        10 => y - z,
        11 => -y - z,
        12 => x + y,
        13 => -y + z,
        14 => -x + y,
        _ => -y - z,
    }
}

impl Perm {
    /// Perlin 3D ノイズ。格子点に擬似乱数勾配を置きドット積を補間する。概ね `[-1, 1]`。
    pub fn perlin3(&self, x: f32, y: f32, z: f32) -> f32 {
        let xi = x.floor() as i32;
        let yi = y.floor() as i32;
        let zi = z.floor() as i32;
        let xf = x - x.floor();
        let yf = y - y.floor();
        let zf = z - z.floor();
        let u = fade(xf);
        let v = fade(yf);
        let w = fade(zf);

        let aaa = self.hash3(xi, yi, zi);
        let aba = self.hash3(xi, yi + 1, zi);
        let aab = self.hash3(xi, yi, zi + 1);
        let abb = self.hash3(xi, yi + 1, zi + 1);
        let baa = self.hash3(xi + 1, yi, zi);
        let bba = self.hash3(xi + 1, yi + 1, zi);
        let bab = self.hash3(xi + 1, yi, zi + 1);
        let bbb = self.hash3(xi + 1, yi + 1, zi + 1);

        let x1 = lerp(grad3(aaa, xf, yf, zf), grad3(baa, xf - 1.0, yf, zf), u);
        let x2 = lerp(grad3(aba, xf, yf - 1.0, zf), grad3(bba, xf - 1.0, yf - 1.0, zf), u);
        let y1 = lerp(x1, x2, v);
        let x3 = lerp(grad3(aab, xf, yf, zf - 1.0), grad3(bab, xf - 1.0, yf, zf - 1.0), u);
        let x4 =
            lerp(grad3(abb, xf, yf - 1.0, zf - 1.0), grad3(bbb, xf - 1.0, yf - 1.0, zf - 1.0), u);
        let y2 = lerp(x3, x4, v);
        lerp(y1, y2, w)
    }
}

use crate::rgb::Rgb;
use crate::xyz::Xyz;

/// A CIELAB color.
#[derive(Copy, Clone, Debug, Default, PartialOrd, PartialEq)]
pub struct Lab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

impl Lab {
    /// Create a new CIELAB color.
    ///
    /// `l`: lightness component (0 to 100)
    ///
    /// `a`: green (negative) and red (positive) component.
    ///
    /// `b`: blue (negative) and yellow (positive) component.
    #[inline]
    pub fn new(l: f32, a: f32, b: f32) -> Self {
        Self { l, a, b }
    }
}

impl Lab {
    pub fn from_rgb(rgb: &Rgb<u8>) -> Self {
        let xyz = Xyz::from_srgb(rgb);
        let x = xyz.x * 100f32 / 95.047f32;
        let y = xyz.y * 100f32 / 100f32;
        let z = xyz.z * 100f32 / 108.883f32;
        let x = if x > 0.008856f32 {
            x.cbrt()
        } else {
            7.787f32 * x + 16f32 / 116f32
        };
        let y = if y > 0.008856f32 {
            y.cbrt()
        } else {
            7.787f32 * y + 16f32 / 116f32
        };
        let z = if z > 0.008856f32 {
            z.cbrt()
        } else {
            7.787f32 * z + 16f32 / 116f32
        };
        Self::new((116f32 * y) - 16f32, 500f32 * (x - y), 200f32 * (y - z))
    }
}

impl Lab {
    pub fn to_rgb8(&self) -> Rgb<u8> {
        let y = (self.l + 16.0) / 116.0;
        let x = self.a * (1f32 / 500f32) + y;
        let z = y - self.b * (1f32 / 200f32);
        let x3 = x.powf(3.0);
        let y3 = y.powf(3.0);
        let z3 = z.powf(3.0);
        let x = 95.047
            * if x3 > 0.008856 {
                x3
            } else {
                (x - 16.0 / 116.0) / 7.787
            };
        let y = 100.0
            * if y3 > 0.008856 {
                y3
            } else {
                (y - 16.0 / 116.0) / 7.787
            };
        let z = 108.883
            * if z3 > 0.008856 {
                z3
            } else {
                (z - 16.0 / 116.0) / 7.787
            };
        Xyz::new(x / 100f32, y / 100f32, z / 100f32).to_srgb()
    }

    pub fn to_rgb(&self) -> Rgb<u8> {
        self.to_rgb8()
    }
}

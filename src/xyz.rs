use crate::rgb::Rgb;
use crate::{srgb_from_linear, srgb_to_linear};

/// A CIE 1931 XYZ color.
#[derive(Copy, Clone, Debug, Default)]
pub struct Xyz {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Xyz {
    #[inline]
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

static XYZ_SCALE_U8: f32 = 1f32 / 255f32;

impl Xyz {
    pub fn from_rgb(rgb: &Rgb<u8>) -> Self {
        let r = srgb_to_linear(rgb.r as f32 * XYZ_SCALE_U8);
        let g = srgb_to_linear(rgb.g as f32 * XYZ_SCALE_U8);
        let b = srgb_to_linear(rgb.b as f32 * XYZ_SCALE_U8);
        Self::new(
            (0.4124 * r + 0.3576 * g + 0.1805 * b) * 100.0,
            (0.2126 * r + 0.7152 * g + 0.0722 * b) * 100.0,
            (0.0193 * r + 0.1192 * g + 0.9505 * b) * 100.0,
        )
    }
}

impl Xyz {
    pub(crate) fn to_rgb(&self) -> Rgb<u8> {
        let x = self.x / 100.0;
        let y = self.y / 100.0;
        let z = self.z / 100.0;
        let r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
        let g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
        let b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;
        let r = 255f32 * srgb_from_linear(r);
        let g = 255f32 * srgb_from_linear(g);
        let b = 255f32 * srgb_from_linear(b);
        Rgb::new(r as u8, g as u8, b as u8)
    }
}
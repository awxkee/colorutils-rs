use crate::Rgb;

#[derive(Debug, PartialOrd, PartialEq, Copy, Clone)]
/// Represents color as sigmoid function: `y = 1 / (1 + exp(-x))`
/// and it's inverse
/// `x = ln(y / (1 - y))`
pub struct Sigmoidal {
    pub sr: f32,
    pub sg: f32,
    pub sb: f32,
}

#[inline(always)]
fn to_sigmoidal(x: f32) -> f32 {
    let den = 1f32 + (-x).exp();
    if den == 0f32 {
        return 0f32;
    }
    return 1f32 / den;
}

#[inline(always)]
fn inverse_sigmoidal(x: f32) -> f32 {
    let den = 1f32 - x;
    if den == 0f32 {
        return 0f32;
    }
    let k = x / den;
    if k <= 0f32 {
        return 0f32;
    }
    return k.ln();
}

impl Sigmoidal {
    pub fn new(sr: f32, sg: f32, sb: f32) -> Self {
        Sigmoidal { sr, sg, sb }
    }

    #[inline(always)]
    pub fn from_rgb(rgb: &Rgb<u8>) -> Self {
        let normalized = rgb.to_rgb_f32();
        Sigmoidal::new(
            to_sigmoidal(normalized.r),
            to_sigmoidal(normalized.g),
            to_sigmoidal(normalized.b),
        )
    }

    #[inline(always)]
    pub fn to_rgb(&self) -> Rgb<u8> {
        let rgb_normalized = Rgb::new(
            inverse_sigmoidal(self.sr),
            inverse_sigmoidal(self.sg),
            inverse_sigmoidal(self.sb),
        );
        return rgb_normalized.into();
    }
}

impl From<Rgb<u8>> for Sigmoidal {
    #[inline(always)]
    fn from(value: Rgb<u8>) -> Self {
        Sigmoidal::from_rgb(&value)
    }
}

impl From<Rgb<f32>> for Sigmoidal {
    #[inline(always)]
    fn from(value: Rgb<f32>) -> Self {
        Sigmoidal::new(
            to_sigmoidal(value.r),
            to_sigmoidal(value.g),
            to_sigmoidal(value.b),
        )
    }
}

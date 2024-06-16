use crate::hsv::Hsv;
use crate::lab::Lab;
use crate::luv::Luv;
use crate::{Hsl, LCh, Sigmoidal};

#[derive(Debug, PartialOrd, PartialEq, Clone, Copy)]
pub struct Rgb<T> {
    pub r: T,
    pub g: T,
    pub b: T,
}

impl Rgb<u8> {
    #[inline]
    pub fn to_hsl(&self) -> Hsl {
        Hsl::from_rgb(self)
    }

    #[inline]
    pub fn to_hsv(&self) -> Hsv {
        Hsv::from(self)
    }

    #[inline]
    pub fn to_lab(&self) -> Lab {
        Lab::from_rgb(self)
    }

    #[inline]
    pub fn to_luv(&self) -> Luv {
        Luv::from_rgb(self)
    }

    #[inline]
    pub fn to_lch(&self) -> LCh {
        LCh::from_rgb(self)
    }

    #[inline(always)]
    pub fn to_rgb_f32(&self) -> Rgb<f32> {
        const SCALE: f32 = 1f32 / 255f32;
        Rgb::<f32>::new(
            self.r as f32 * SCALE,
            self.g as f32 * SCALE,
            self.b as f32 * SCALE,
        )
    }

    #[inline(always)]
    pub fn to_sigmoidal(&self) -> Sigmoidal {
        Sigmoidal::from_rgb(self)
    }
}

impl From<Rgb<f32>> for Rgb<u8> {
    #[inline(always)]
    fn from(value: Rgb<f32>) -> Self {
        value.to_u8()
    }
}

impl Rgb<f32> {
    #[inline(always)]
    pub fn apply(&self, gen: fn(f32) -> f32) -> Self {
        Self {
            r: gen(self.r),
            g: gen(self.g),
            b: gen(self.b),
        }
    }

    #[inline(always)]
    pub fn to_u8(&self) -> Rgb<u8> {
        Rgb::<u8>::new(
            (self.r * 255f32).max(0f32).round().min(255f32) as u8,
            (self.g * 255f32).max(0f32).round().min(255f32) as u8,
            (self.b * 255f32).max(0f32).round().min(255f32) as u8,
        )
    }
}

impl From<Sigmoidal> for Rgb<u8> {
    #[inline(always)]
    fn from(value: Sigmoidal) -> Self {
        value.to_rgb()
    }
}

impl<T> Rgb<T> {
    pub fn new(r: T, g: T, b: T) -> Rgb<T> {
        Rgb { r, g, b }
    }
}

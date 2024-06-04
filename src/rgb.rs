use crate::hsv::Hsv;
use crate::lab::Lab;
use crate::luv::Luv;
use crate::{Hsl, LCh};

pub struct Rgb<T> {
    pub r: T,
    pub g: T,
    pub b: T,
}

impl Rgb<u8> {
    #[allow(dead_code)]
    #[inline]
    pub fn to_hsl(&self) -> Hsl {
        Hsl::from_rgb(self)
    }

    #[allow(dead_code)]
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

    #[inline]
    pub fn to_rgb_f32(&self) -> Rgb<f32> {
        const SCALE: f32 = 1f32 / 255f32;
        Rgb::<f32>::new(
            self.r as f32 * SCALE,
            self.g as f32 * SCALE,
            self.b as f32 * SCALE,
        )
    }
}

impl<T> Rgb<T> {
    pub fn new(r: T, g: T, b: T) -> Rgb<T> {
        Rgb { r, g, b }
    }
}

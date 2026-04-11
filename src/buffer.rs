/*
 * // Copyright 2026 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::err::ColorError;
use std::borrow::Cow;
use std::fmt::Debug;
use std::ops::{Index, Range, RangeFrom};

pub struct ImageBuffer<'a, F>
where
    [F]: ToOwned<Owned = Vec<F>>,
{
    pub data: std::borrow::Cow<'a, [F]>,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub channels: u32,
}

impl<'a, F: Sized> ImageBuffer<'a, F>
where
    [F]: ToOwned<Owned = Vec<F>>,
{
    pub fn new(data: &'a [F], width: u32, height: u32, channels: u32) -> Result<Self, ColorError> {
        let min_stride = width * channels;
        let required = min_stride as usize * height as usize;
        if data.len() < required {
            return Err(ColorError::BufferTooSmall {
                expected: required,
                got: data.len(),
            });
        }
        Ok(Self {
            data: Cow::Borrowed(data),
            width,
            height,
            stride: min_stride,
            channels,
        })
    }

    pub fn from_vec(
        data: Vec<F>,
        width: u32,
        height: u32,
        channels: u32,
    ) -> Result<Self, ColorError> {
        let required = width as usize * channels as usize * height as usize;
        if data.len() < required {
            return Err(ColorError::BufferTooSmall {
                expected: required,
                got: data.len(),
            });
        }
        Ok(Self {
            data: Cow::Owned(data),
            width,
            height,
            stride: width * channels,
            channels,
        })
    }

    pub fn try_match(&self, other: &ImageBuffer<'_, F>) -> Result<(), ColorError> {
        if self.width != other.width || self.height != other.height {
            return Err(ColorError::DimensionMismatch {
                expected: (self.width, self.height),
                got: (other.width, other.height),
            });
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), ColorError> {
        let min_stride = self
            .width
            .checked_mul(self.channels)
            .ok_or(ColorError::DimensionOverflow)?;

        if self.stride < min_stride {
            return Err(ColorError::StrideTooNarrow {
                min: min_stride,
                given: self.stride,
            });
        }

        let required = self
            .stride()
            .checked_mul(self.height as usize)
            .ok_or(ColorError::DimensionOverflow)?;

        if required == 0 {
            return Err(ColorError::ZeroImageSize);
        }

        if self.data.len() < required {
            return Err(ColorError::BufferTooSmall {
                expected: required,
                got: self.data.len(),
            });
        }

        Ok(())
    }

    pub fn stride(&self) -> usize {
        if self.stride == 0 {
            return self.width as usize * self.channels as usize;
        }
        self.stride as usize
    }
}

#[derive(Debug)]
pub enum BufferStore<'a, T: Debug> {
    Borrowed(&'a mut [T]),
    Owned(Vec<T>),
}

impl<T: Copy + Debug> BufferStore<'_, T> {
    #[allow(clippy::should_implement_trait)]
    pub fn borrow(&self) -> &[T] {
        match self {
            Self::Borrowed(p_ref) => p_ref,
            Self::Owned(vec) => vec,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn borrow_mut(&mut self) -> &mut [T] {
        match self {
            Self::Borrowed(p_ref) => p_ref,
            Self::Owned(vec) => vec,
        }
    }
}

impl<T: Copy + Debug> Index<usize> for BufferStore<'_, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            Self::Borrowed(p_ref) => &p_ref[index],
            Self::Owned(vec) => &vec[index],
        }
    }
}

impl<T: Copy + Debug> Index<Range<usize>> for BufferStore<'_, T> {
    type Output = [T];

    fn index(&self, index: Range<usize>) -> &Self::Output {
        match self {
            Self::Borrowed(p_ref) => &p_ref[index],
            Self::Owned(vec) => &vec[index],
        }
    }
}

impl<T: Copy + Debug> Index<RangeFrom<usize>> for BufferStore<'_, T> {
    type Output = [T];

    fn index(&self, index: RangeFrom<usize>) -> &Self::Output {
        match self {
            Self::Borrowed(p_ref) => &p_ref[index],
            Self::Owned(vec) => &vec[index],
        }
    }
}

#[derive(Debug)]
pub struct ImageBufferMut<'a, F: Copy + Debug> {
    pub data: BufferStore<'a, F>,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub channels: u32,
}

impl<'a, F: Copy + Debug + Sized> ImageBufferMut<'a, F> {
    pub fn new(
        data: BufferStore<'a, F>,
        width: u32,
        height: u32,
        channels: u32,
    ) -> Result<Self, ColorError> {
        let buf = Self {
            data,
            width,
            height,
            stride: width * channels,
            channels,
        };
        buf.validate()?;
        Ok(buf)
    }

    pub fn validate(&self) -> Result<(), ColorError> {
        let min_stride = self
            .width
            .checked_mul(self.channels)
            .ok_or(ColorError::DimensionOverflow)?;

        if self.stride < min_stride {
            return Err(ColorError::StrideTooNarrow {
                min: min_stride,
                given: self.stride,
            });
        }

        let required = self
            .stride()
            .checked_mul(self.height as usize)
            .ok_or(ColorError::DimensionOverflow)?;

        if required == 0 {
            return Err(ColorError::ZeroImageSize);
        }

        if self.data.borrow().len() < required {
            return Err(ColorError::BufferTooSmall {
                expected: required,
                got: self.data.borrow().len(),
            });
        }

        Ok(())
    }

    pub fn try_match(&self, other: &ImageBufferMut<'_, F>) -> Result<(), ColorError> {
        if self.width != other.width || self.height != other.height {
            return Err(ColorError::DimensionMismatch {
                expected: (self.width, self.height),
                got: (other.width, other.height),
            });
        }
        Ok(())
    }

    pub fn try_match_immutable<Z>(&self, other: &ImageBuffer<'_, Z>) -> Result<(), ColorError>
    where
        [Z]: ToOwned<Owned = Vec<Z>>,
    {
        if self.width != other.width || self.height != other.height {
            return Err(ColorError::DimensionMismatch {
                expected: (self.width, self.height),
                got: (other.width, other.height),
            });
        }
        Ok(())
    }

    pub fn try_match_immutable_with_channels<Z>(
        &self,
        other: &ImageBuffer<'_, Z>,
    ) -> Result<(), ColorError>
    where
        [Z]: ToOwned<Owned = Vec<Z>>,
    {
        if self.width != other.width || self.height != other.height {
            return Err(ColorError::DimensionMismatch {
                expected: (self.width, self.height),
                got: (other.width, other.height),
            });
        }
        if self.channels != other.channels {
            return Err(ColorError::ChannelCountMismatch {
                expected: self.channels,
                got: other.channels,
            });
        }
        Ok(())
    }

    pub fn stride(&self) -> usize {
        if self.stride == 0 {
            return self.width as usize * self.channels as usize;
        }
        self.stride as usize
    }
}

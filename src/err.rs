/*
 * // Copyright 2026 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use std::fmt;

#[derive(Debug)]
pub enum ColorError {
    BufferTooSmall {
        expected: usize,
        got: usize,
    },
    StrideTooNarrow {
        min: u32,
        given: u32,
    },
    DimensionMismatch {
        expected: (u32, u32),
        got: (u32, u32),
    },
    DimensionOverflow,
    ChannelCountMismatch {
        expected: u32,
        got: u32,
    },
    Generic(String),
    ZeroImageSize,
    UnsupportedChannelsCount(u32),
}

impl fmt::Display for ColorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ColorError::BufferTooSmall { expected, got } => {
                write!(f, "buffer too small: expected {expected} bytes, got {got}")
            }
            ColorError::StrideTooNarrow { min, given } => {
                write!(
                    f,
                    "stride {given} is too narrow for width \u{00d7} channels (minimum {min})"
                )
            }
            ColorError::DimensionMismatch { expected, got } => f.write_fmt(format_args!(
                "source and destination dimensions differ expected {:?} got {:?}",
                expected, got
            )),
            ColorError::DimensionOverflow => {
                f.write_str("source and destination dimensions overflow")
            }
            ColorError::ChannelCountMismatch { expected, got } => f.write_fmt(format_args!(
                "channel count mismatch: expected {expected}, got {got}"
            )),
            ColorError::Generic(msg) => f.write_str(msg),
            ColorError::ZeroImageSize => f.write_str("Image may not have zero based size"),
            ColorError::UnsupportedChannelsCount(provided) => {
                f.write_fmt(format_args!("Channels count {} is not supported", provided))
            }
        }
    }
}

impl std::error::Error for ColorError {}

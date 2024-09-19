/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum XyzTarget {
    Lab = 0,
    Xyz = 1,
    Luv = 2,
    Lch = 3,
}

impl From<u8> for XyzTarget {
    fn from(value: u8) -> Self {
        match value {
            0 => XyzTarget::Lab,
            1 => XyzTarget::Xyz,
            2 => XyzTarget::Luv,
            3 => XyzTarget::Lch,
            _ => {
                panic!("Not implemented")
            }
        }
    }
}

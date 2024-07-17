/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum XyzTarget {
    LAB = 0,
    XYZ = 1,
    LUV = 2,
    LCH = 3,
}

impl From<u8> for XyzTarget {
    fn from(value: u8) -> Self {
        match value {
            0 => XyzTarget::LAB,
            1 => XyzTarget::XYZ,
            2 => XyzTarget::LUV,
            3 => XyzTarget::LCH,
            _ => {
                panic!("Not implemented")
            }
        }
    }
}

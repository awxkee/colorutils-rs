/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum HsvTarget {
    HSV,
    HSL,
}

impl From<u8> for HsvTarget {
    fn from(value: u8) -> Self {
        return match value {
            0 => HsvTarget::HSV,
            1 => HsvTarget::HSL,
            _ => panic!("Unknown target was request"),
        };
    }
}

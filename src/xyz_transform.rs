/// sRGB to XYZ transformation matrix, D65 White point
pub const SRGB_TO_XYZ_D65: [[f32; 3]; 3] = [
    [0.4124564f32, 0.3575761f32, 0.1804375f32],
    [0.2126729f32, 0.7151522f32, 0.0721750f32],
    [0.0193339f32, 0.1191920f32, 0.9503041f32],
];

/// XYZ to sRGB transformation matrix, D65 White point
pub const XYZ_TO_SRGB_D65: [[f32; 3]; 3] = [
    [3.2404542f32, -1.5371385f32, -0.4985314f32],
    [-0.9692660f32, 1.8760108f32, 0.0415560f32],
    [0.0556434f32, -0.2040259f32, 1.0572252f32],
];

/// sRGB to XYZ transformation matrix, D50 White point
pub const SRGB_TO_XYZ_D50: [[f32; 3]; 3] = [
    [0.4360747f32, 0.3850649f32, 0.1430804f32],
    [0.2225045f32, 0.7168786f32, 0.0606169f32],
    [0.0139322f32, 0.0971045f32, 0.7141733f32],
];

/// XYZ to sRGB transformation matrix, D50 White point
pub const XYZ_TO_SRGB_D50: [[f32; 3]; 3] = [
    [3.1338561f32, -1.6168667f32, -0.4906146f32],
    [-0.9787684f32, 1.9161415f32, 0.0334540f32],
    [0.0719453f32, -0.2289914f32, 1.4052427f32],
];

pub fn srgb_to_linear(gamma: f32) -> f32 {
    return if gamma < 0f32 {
        0f32
    } else if gamma < 12.92f32 * 0.0030412825601275209f32 {
        gamma / 12.92f32
    } else if gamma < 1.0f32 {
        ((gamma + 0.0550107189475866f32) / 1.0550107189475866f32).powf(2.4f32)
    } else {
        1.0f32
    };
}

pub fn srgb_from_linear(linear: f32) -> f32 {
    return if linear < 0.0f32 {
        0.0f32
    } else if linear < 0.0030412825601275209f32 {
        linear * 12.92f32
    } else if linear < 1.0f32 {
        1.0550107189475866f32 * linear.powf(1.0f32 / 2.4f32) - 0.0550107189475866f32
    } else {
        1.0f32
    };
}

pub fn rec709_to_linear(gamma: f32) -> f32 {
    return if gamma < 0.0f32 {
        0.0f32
    } else if gamma < 4.5f32 * 0.018053968510807f32 {
        gamma / 4.5f32
    } else if gamma < 1.0f32 {
        ((gamma + 0.09929682680944f32) / 1.09929682680944f32).powf(1.0f32 / 0.45f32)
    } else {
        1.0f32
    };
}

pub fn rec709_from_linear(linear: f32) -> f32 {
    return if linear < 0.0f32 {
        0.0f32
    } else if linear < 0.018053968510807f32 {
        linear * 4.5f32
    } else if linear < 1.0f32 {
        1.09929682680944f32 * linear.powf(0.45f32) - 0.09929682680944f32
    } else {
        1.0f32
    };
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum TransferFunction {
    Srgb,
    Rec709,
}

impl TransferFunction {
    #[inline(always)]
    pub fn get_linearize_function(&self) -> fn(f32) -> f32 {
        match self {
            TransferFunction::Srgb => srgb_to_linear,
            TransferFunction::Rec709 => rec709_to_linear,
        }
    }

    #[inline(always)]
    pub fn get_gamma_function(&self) -> fn(f32) -> f32 {
        match self {
            TransferFunction::Srgb => srgb_from_linear,
            TransferFunction::Rec709 => rec709_from_linear,
        }
    }
}

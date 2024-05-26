#[derive(Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum ImageConfiguration {
    Rgb = 0,
    Rgba = 1,
    Bgra = 2,
    Bgr = 3,
}

impl ImageConfiguration {
    #[inline(always)]
    pub fn get_channels_count(&self) -> usize {
        match self {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => 3,
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => 4,
        }
    }

    #[inline(always)]
    pub fn has_alpha(&self) -> bool {
        match self {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => false,
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => true,
        }
    }

    #[inline(always)]
    pub fn get_r_channel_offset(&self) -> usize {
        match self {
            ImageConfiguration::Rgb => 0,
            ImageConfiguration::Rgba => 0,
            ImageConfiguration::Bgra | ImageConfiguration::Bgr => 2,
        }
    }

    #[inline(always)]
    pub fn get_g_channel_offset(&self) -> usize {
        match self {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => 1,
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => 1,
        }
    }

    #[inline(always)]
    pub fn get_b_channel_offset(&self) -> usize {
        match self {
            ImageConfiguration::Rgb => 2,
            ImageConfiguration::Rgba => 2,
            ImageConfiguration::Bgra | ImageConfiguration::Bgr => 0,
        }
    }
    #[inline(always)]
    pub fn get_a_channel_offset(&self) -> usize {
        match self {
            ImageConfiguration::Rgb | ImageConfiguration::Bgr => 0,
            ImageConfiguration::Rgba | ImageConfiguration::Bgra => 3,
        }
    }
}

impl From<u8> for ImageConfiguration {
    #[inline(always)]
    fn from(value: u8) -> Self {
        match value {
            0 => ImageConfiguration::Rgb,
            1 => ImageConfiguration::Rgba,
            2 => ImageConfiguration::Bgra,
            3 => ImageConfiguration::Bgr,
            _ => {
                panic!("Unknown value")
            }
        }
    }
}

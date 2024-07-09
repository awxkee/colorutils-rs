#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::linear_to_planar::neon_linear_plane_to_gamma;
use crate::TransferFunction;

#[inline(always)]
fn linear_to_gamma_channels(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    let mut src_offset = 0usize;
    let mut dst_offset = 0usize;

    let transfer = transfer_function.get_gamma_function();

    for _ in 0..height as usize {
        let mut _cx = 0usize;

        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "arm"),
            target_feature = "neon"
        ))]
        unsafe {
            _cx = neon_linear_plane_to_gamma(
                _cx,
                src.as_ptr(),
                src_offset as u32,
                dst.as_mut_ptr(),
                dst_offset as u32,
                width,
                transfer_function,
            );
        }

        let src_ptr = unsafe { (src.as_ptr() as *const u8).add(src_offset) as *const f32 };
        let dst_ptr = unsafe { dst.as_mut_ptr().add(dst_offset) };

        for x in _cx..width as usize {
            let px = x;
            let src_slice = unsafe { src_ptr.add(px) };
            let pixel = unsafe { src_slice.read_unaligned() }.min(1f32).max(0f32);

            let dst = unsafe { dst_ptr.add(px) };
            let transferred = transfer(pixel);
            let rgb8 = (transferred * 255f32).min(255f32).max(0f32) as u8;

            unsafe {
                dst.write_unaligned(rgb8);
            }
        }

        src_offset += src_stride as usize;
        dst_offset += dst_stride as usize;
    }
}

/// This function converts Linear to Plane. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains Linear plane data
/// * `src_stride` - Bytes per row for src data.
/// * `dst` - A mutable slice to receive Gamma plane data
/// * `dst_stride` - Bytes per row for dst data
/// * `width` - Image width
/// * `height` - Image height
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn linear_to_plane(
    src: &[f32],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    linear_to_gamma_channels(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

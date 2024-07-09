#[cfg(all(
    any(target_arch = "aarch64", target_arch = "arm"),
    target_feature = "neon"
))]
use crate::neon::planar_to_linear::neon_plane_to_linear;
use crate::TransferFunction;

#[inline(always)]
fn channels_to_linear(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    let mut src_offset = 0usize;
    let mut dst_offset = 0usize;

    let transfer = transfer_function.get_linearize_function();
    for _ in 0..height as usize {
        let mut _cx = 0usize;

        let src_ptr = unsafe { src.as_ptr().add(src_offset) };
        let dst_ptr = unsafe { (dst.as_mut_ptr() as *mut u8).add(dst_offset) as *mut f32 };

        #[cfg(all(
            any(target_arch = "aarch64", target_arch = "arm"),
            target_feature = "neon"
        ))]
        unsafe {
            _cx = neon_plane_to_linear(
                _cx,
                src.as_ptr(),
                src_offset,
                width,
                dst.as_mut_ptr(),
                dst_offset,
                transfer_function,
            )
        }

        for x in _cx..width as usize {
            let px = x;
            let dst = unsafe { dst_ptr.add(px) };
            let src = unsafe { src_ptr.add(px) };
            let pixel_f = unsafe { src.read_unaligned() as f32 } * (1. / 255.);
            let transferred = transfer(pixel_f);

            unsafe {
                dst.write_unaligned(transferred);
            }
        }

        src_offset += src_stride as usize;
        dst_offset += dst_stride as usize;
    }
}

/// This function converts Plane to Linear. This is much more effective than naive direct transformation
///
/// # Arguments
/// * `src` - A slice contains RGB data
/// * `src_stride` - Bytes per row for src data.
/// * `width` - Image width
/// * `height` - Image height
/// * `dst` - A mutable slice to receive linear data
/// * `dst_stride` - Bytes per row for dst data
/// * `transfer_function` - Transfer function from gamma to linear space. If you don't have specific pick `Srgb`
pub fn plane_to_linear(
    src: &[u8],
    src_stride: u32,
    dst: &mut [f32],
    dst_stride: u32,
    width: u32,
    height: u32,
    transfer_function: TransferFunction,
) {
    channels_to_linear(
        src,
        src_stride,
        dst,
        dst_stride,
        width,
        height,
        transfer_function,
    );
}

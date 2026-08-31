//! IEEE 754 binary16 (`f16`) storage, as raw `u16` bits.
//!
//! Rust's `f16` is still unstable, and this crate's MSRV is 1.82, so the type
//! is carried as its bit pattern and converted explicitly. That is not purely a
//! workaround: every conversion site being visible is what makes the error
//! analysis in [`crate::tolerance_for_spmv_f16`] auditable.
//!
//! # This is a storage format, not an accumulator
//!
//! binary16 has an 11-bit significand, so its unit roundoff is `2^-11`
//! (9.8e-4) against f32's `2^-23` (1.2e-7) — a factor of 8192. Accumulating a
//! 500-term row sum in binary16 admits a relative error near `500 * 4.9e-4`,
//! about 24%. It also overflows at 65504, which a row sum reaches easily.
//!
//! So nothing here accumulates in binary16. Weights are *stored* narrow, then
//! widened to f32 on load and summed in f32. That buys the bandwidth and keeps
//! the error bound tractable; see [`crate::tolerance_for_spmv_f16`].

/// Machine epsilon of IEEE binary16: `2^-10`, the ulp at 1.0.
///
/// Deliberately the same convention as [`f32::EPSILON`] (`2^-23`), not the
/// unit roundoff `2^-11`. [`crate::tolerance_for_spmv`] is written in terms of
/// `f32::EPSILON`, and its binary16 counterpart sits next to it; a reader
/// comparing the two should not have to notice a silent factor-of-two change
/// of convention between them.
///
/// It is 8192x `f32::EPSILON`. That ratio is the whole reason binary16 is a
/// storage format here and never an accumulator.
pub const HALF_EPSILON: f32 = 9.765_625e-4;

/// Largest finite binary16 value.
pub const HALF_MAX: f32 = 65504.0;

/// Smallest positive *normal* binary16 value, `2^-14`.
pub const HALF_MIN_POSITIVE: f32 = 6.103_515_6e-5;

/// Widen binary16 bits to `f32`. Exact: every binary16 is an f32.
#[inline]
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mantissa = u32::from(bits & 0x03ff);

    let out = if exp == 0 {
        if mantissa == 0 {
            // Signed zero.
            sign
        } else {
            // Subnormal: renormalise into f32's exponent range. Shifting the
            // leading one out of the mantissa is what makes this exact rather
            // than a scaled approximation.
            // A binary16 subnormal is `mantissa * 2^-24`. With the leading one
            // at bit `k`, that is `(mantissa / 2^k) * 2^(k-24)`, so the f32
            // exponent is `127 + k - 24`. In terms of `shift` (= 10 - k) that
            // is `127 - 14 - shift`. Using -15 here — the exponent bias rather
            // than this derivation — puts every subnormal out by a factor of 2.
            let shift = mantissa.leading_zeros() - 21;
            let exp32 = 127 - 14 - shift;
            // Shift the leading one onto bit 10, then mask it off: what
            // remains is the 10-bit fraction. `shift + 1` pushes it to bit 11
            // and the mask then discards the whole significand — 0x0003 became
            // 0x0002, losing the fraction entirely.
            let man32 = (mantissa << shift) & 0x03ff;
            sign | (exp32 << 23) | (man32 << 13)
        }
    } else if exp == 0x1f {
        // Inf or NaN. The mantissa is carried across so a quiet NaN stays quiet
        // and a signalling payload is not silently rewritten.
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mantissa << 13)
    };
    f32::from_bits(out)
}

/// Narrow `f32` to binary16 bits, round-to-nearest-even.
///
/// Ties-to-even because that is what both the CPU and Metal's `half` do;
/// anything else would make the host-encoded weights disagree with a GPU that
/// re-rounded them, and the differential suite would report a kernel bug.
///
/// Values beyond [`HALF_MAX`] become infinity rather than saturating.
/// Saturation would silently turn an overflow into a plausible large number,
/// and this crate would rather the caller see an infinity propagate.
#[inline]
pub fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x007f_ffff;

    if exp == 0xff {
        // Inf or NaN. A NaN whose payload is entirely in the low bits would
        // become an infinity if the mantissa were merely truncated, so a
        // non-zero payload is forced to set at least one surviving bit.
        let man16 = (mantissa >> 13) as u16;
        let man16 = if mantissa != 0 && man16 == 0 {
            1
        } else {
            man16
        };
        return sign | 0x7c00 | man16;
    }

    let unbiased = exp - 127;
    if unbiased >= 16 {
        // Overflow, including the case where rounding would carry into it.
        return sign | 0x7c00;
    }
    if unbiased < -24 {
        // Below half the smallest subnormal: rounds to zero.
        return sign;
    }

    // `implicit` is the significand with its hidden bit restored, so normal and
    // subnormal share one rounding path.
    let (mut significand, shift) = if unbiased < -14 {
        (mantissa | 0x0080_0000, (-14 - unbiased) as u32 + 13)
    } else {
        (mantissa, 13)
    };
    if unbiased < -14 {
        // Subnormal: the exponent field is zero and the value is carried
        // entirely in the mantissa.
        let round_bit = 1u32 << (shift - 1);
        let sticky = significand & (round_bit - 1);
        let mut out = significand >> shift;
        if significand & round_bit != 0 && (sticky != 0 || out & 1 != 0) {
            out += 1;
        }
        return sign | out as u16;
    }

    significand &= 0x007f_ffff;
    let round_bit = 1u32 << (shift - 1);
    let sticky = significand & (round_bit - 1);
    let mut man16 = significand >> shift;
    let mut exp16 = (unbiased + 15) as u32;
    if significand & round_bit != 0 && (sticky != 0 || man16 & 1 != 0) {
        man16 += 1;
        if man16 == 0x400 {
            // Rounding carried out of the significand.
            man16 = 0;
            exp16 += 1;
            if exp16 >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((exp16 as u16) << 10) | man16 as u16
}

/// Narrow a slice, allocating the bit pattern array a GPU buffer is filled from.
pub fn f32_slice_to_f16(values: &[f32]) -> Vec<u16> {
    values.iter().copied().map(f32_to_f16_bits).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exhaustive: all 65536 binary16 values widen and narrow back to
    /// themselves.
    ///
    /// This is a proof rather than a sample — the domain is small enough to
    /// enumerate, so there is no reason to settle for random cases. NaNs are
    /// compared by class, since a payload is free to change.
    #[test]
    fn every_f16_round_trips_through_f32() {
        for bits in 0u16..=u16::MAX {
            let wide = f16_bits_to_f32(bits);
            let back = f32_to_f16_bits(wide);
            let is_nan = (bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) != 0;
            if is_nan {
                assert!(wide.is_nan(), "0x{bits:04X} widened to a non-NaN");
                assert_eq!(back & 0x7c00, 0x7c00, "0x{bits:04X} lost its NaN class");
                assert_ne!(back & 0x03ff, 0, "0x{bits:04X} became an infinity");
            } else {
                assert_eq!(back, bits, "0x{bits:04X} -> {wide} -> 0x{back:04X}");
            }
        }
    }

    #[test]
    fn the_documented_constants_are_the_values_they_name() {
        assert_eq!(f16_bits_to_f32(0x7bff), HALF_MAX);
        assert_eq!(f16_bits_to_f32(0x0400), HALF_MIN_POSITIVE);
        // Same convention as `f32::EPSILON`: the ulp at 1.0, not unit roundoff.
        assert_eq!(HALF_EPSILON, 2.0f32.powi(-10));
        assert_eq!(HALF_EPSILON / f32::EPSILON, 8192.0);
    }

    #[test]
    fn overflow_becomes_infinity_rather_than_saturating() {
        // Saturation would turn an overflow into a plausible large number.
        assert_eq!(f32_to_f16_bits(70000.0), 0x7c00);
        assert_eq!(f32_to_f16_bits(-70000.0), 0xfc00);
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        // The largest f32 that still rounds *down* to HALF_MAX must not tip.
        assert_eq!(f32_to_f16_bits(65504.0), 0x7bff);
        // Halfway between HALF_MAX and the next binary16 rounds to infinity.
        assert_eq!(f32_to_f16_bits(65520.0), 0x7c00);
    }

    #[test]
    fn rounding_is_ties_to_even() {
        // 2049 sits exactly halfway between 2048 and 2050 in binary16, whose
        // ulp is 2 at that magnitude. Ties-to-even picks 2048.
        assert_eq!(f16_bits_to_f32(f32_to_f16_bits(2049.0)), 2048.0);
        // 2051 is halfway between 2050 and 2052; even is 2052.
        assert_eq!(f16_bits_to_f32(f32_to_f16_bits(2051.0)), 2052.0);
        // Not a tie: rounds to nearest.
        assert_eq!(f16_bits_to_f32(f32_to_f16_bits(2050.4)), 2050.0);
    }

    #[test]
    fn subnormals_survive_both_directions() {
        // Smallest positive subnormal, 2^-24.
        let tiny = f16_bits_to_f32(0x0001);
        assert_eq!(tiny, 2.0f32.powi(-24));
        assert_eq!(f32_to_f16_bits(tiny), 0x0001);
        // Just under half of it rounds to zero, keeping the sign.
        assert_eq!(f32_to_f16_bits(tiny * 0.4), 0x0000);
        assert_eq!(f32_to_f16_bits(-tiny * 0.4), 0x8000);
    }
}

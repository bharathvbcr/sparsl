//! Bitpacked spike vectors: 32 spikes to a `u32`.
//!
//! A spike is a boolean — a cell fired this tick or it did not. Carrying that
//! as an `f32` spends 32 bits to say one, and in an SpMV the spike vector is
//! the operand that gets *gathered*: `x[col[i]]` is a random read, and its cost
//! is set by whether `x` fits in cache rather than by how much of it is
//! streamed.
//!
//! That is the difference from narrow weights. Halving the weights cut streamed
//! traffic by 25% and bought a few percent; this shrinks the gathered operand
//! 32-fold, and at 50,000 cells takes it from 200 KB — past L2 — to 6.25 KB.
//!
//! # Exact, not approximate
//!
//! Unlike [`crate::half`], this loses nothing. A spike is 0 or 1, both exactly
//! representable in `f32`, and `w * 0.0` and `w * 1.0` are exact. The kernels
//! multiply by a decoded `0.0`/`1.0` rather than branching on the bit, so the
//! arithmetic performed is *identical* to the dense path — same operations,
//! same order, same roundings. There is no `tolerance_for_spmv_spikes` because
//! there is nothing for it to bound.
//!
//! It goes further than that, and the reason is worth stating. Every other
//! cross-backend comparison in this crate needs a tolerance partly because
//! Metal contracts `a * b + c` into a single `fma`, rounding once where the CPU
//! rounds twice — see `tests/fma_contraction.rs`. That difference cannot arise
//! here: with the multiplier exactly 0 or 1 the product is exact, so there is
//! no intermediate rounding for contraction to skip. CPU and GPU spike SpMV
//! agree *bit for bit*, which the dense one does not.
//!
//! # Layout
//!
//! Spike `i` is bit `i % 32` of word `i / 32`, least-significant bit first.
//! Trailing bits of the final word are zero, which is why an unpacked round
//! trip needs the original length: the packing alone cannot say whether the
//! last word's high bits are absent cells or silent ones.

/// Spikes carried per packed word.
pub const SPIKES_PER_WORD: usize = 32;

/// Words needed to hold `n` spikes.
#[inline]
pub const fn packed_len(n: usize) -> usize {
    n.div_ceil(SPIKES_PER_WORD)
}

/// Pack a boolean spike vector, least-significant bit first.
pub fn pack_spikes(spikes: &[bool]) -> Vec<u32> {
    let mut out = vec![0u32; packed_len(spikes.len())];
    for (i, &fired) in spikes.iter().enumerate() {
        if fired {
            out[i / SPIKES_PER_WORD] |= 1u32 << (i % SPIKES_PER_WORD);
        }
    }
    out
}

/// Unpack `n` spikes from `words`.
///
/// # Panics
///
/// If `words` is too short for `n` spikes. A short buffer would silently
/// produce a vector of quiet non-spikes, which is indistinguishable from a
/// cell that genuinely did not fire.
pub fn unpack_spikes(words: &[u32], n: usize) -> Vec<bool> {
    assert!(
        words.len() >= packed_len(n),
        "sparsl: {} words cannot hold {n} spikes",
        words.len()
    );
    (0..n)
        .map(|i| (words[i / SPIKES_PER_WORD] >> (i % SPIKES_PER_WORD)) & 1 == 1)
        .collect()
}

/// The dense `f32` vector a packed spike train stands in for.
///
/// The reference the bitpacked kernels must agree with, bit for bit.
pub fn spikes_to_f32(words: &[u32], n: usize) -> Vec<f32> {
    unpack_spikes(words, n)
        .into_iter()
        .map(|fired| if fired { 1.0 } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;

    #[test]
    fn packing_round_trips_at_every_word_boundary() {
        // The interesting lengths are around multiples of 32, where a partial
        // final word either exists or does not.
        for n in [0usize, 1, 31, 32, 33, 63, 64, 65, 1000] {
            let mut rng = Rng::new(0x5217 + n as u64);
            let spikes: Vec<bool> = (0..n).map(|_| rng.next_f32() < 0.5).collect();
            let packed = pack_spikes(&spikes);
            assert_eq!(packed.len(), packed_len(n), "word count for n={n}");
            assert_eq!(unpack_spikes(&packed, n), spikes, "round trip at n={n}");
        }
    }

    #[test]
    fn the_layout_is_least_significant_bit_first() {
        // Pinned because the kernels index it independently; if the host and
        // the GPU disagreed about bit order every spike would land on the wrong
        // cell, and the result would still look like a plausible spike train.
        let mut spikes = vec![false; 40];
        spikes[0] = true;
        spikes[31] = true;
        spikes[32] = true;
        let packed = pack_spikes(&spikes);
        assert_eq!(packed[0], 0x8000_0001, "bits 0 and 31 of the first word");
        assert_eq!(packed[1], 0x0000_0001, "bit 0 of the second word");
    }

    #[test]
    fn trailing_bits_of_a_partial_word_are_zero() {
        // Not cosmetic: a kernel reading past `n` must see silence, not
        // whatever the allocation happened to contain.
        let packed = pack_spikes(&[true; 33]);
        assert_eq!(packed[0], u32::MAX);
        assert_eq!(packed[1], 1, "only bit 0 is a real cell");
    }

    #[test]
    fn an_undersized_buffer_is_refused_rather_than_read_as_silence() {
        let packed = pack_spikes(&[true, false, true]);
        let result = std::panic::catch_unwind(|| unpack_spikes(&packed, 1000));
        assert!(
            result.is_err(),
            "a short buffer must not decode as no-spikes"
        );
    }

    #[test]
    fn the_dense_form_is_exactly_zero_and_one() {
        let packed = pack_spikes(&[true, false, true, true]);
        assert_eq!(spikes_to_f32(&packed, 4), vec![1.0, 0.0, 1.0, 1.0]);
    }
}

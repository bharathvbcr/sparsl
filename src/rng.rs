//! Seeded ChaCha RNG (U01).

use rand::RngCore;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

/// Deterministic ChaCha12 random stream.
///
/// Constructed from a `u64` seed; identical seeds always yield identical
/// byte/integer sequences (GC3).
#[derive(Clone, Debug)]
pub struct Rng {
    inner: ChaCha12Rng,
}

impl Rng {
    /// Create a new stream from `seed`.
    #[inline]
    pub fn new(seed: u64) -> Self {
        Self {
            inner: ChaCha12Rng::seed_from_u64(seed),
        }
    }

    /// Next `u32` from the stream.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }

    /// Next `u64` from the stream.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }

    /// Fill `dest` with the next bytes from the stream.
    #[inline]
    pub fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.inner.fill_bytes(dest);
    }

    /// Uniform `f32` in `[0, 1)`.
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        // Standard `[0, 1)` mapping from 24 random mantissa bits.
        (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }

    /// Uniform `f64` in `[0, 1)`.
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in `0..max` (panics if `max == 0`).
    #[inline]
    pub fn gen_index(&mut self, max: usize) -> usize {
        assert!(max > 0, "gen_index requires max > 0");
        (self.next_u64() % (max as u64)) as usize
    }
}

impl RngCore for Rng {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        Rng::next_u32(self)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        Rng::next_u64(self)
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        Rng::fill_bytes(self, dest);
    }

    #[inline]
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        Rng::fill_bytes(self, dest);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;
    use rand::RngCore;
    use rand::SeedableRng;
    use rand_chacha::ChaCha12Rng;

    /// Golden seed for the fixed ChaCha12 sequence.
    pub const GOLDEN_SEED: u64 = 0xB177_C0DE_0000_0001;

    /// First eight `u64` draws from `Rng::new(GOLDEN_SEED)` (ChaCha12 /
    /// `SeedableRng::seed_from_u64`). Regenerate only if the backend changes.
    const GOLDEN_U64: [u64; 8] = [
        0x1222_A135_52E1_0174,
        0x8B17_5BF2_F28F_BA44,
        0xF54A_185E_BBA1_12D0,
        0x7D8F_4BE8_3AD9_42E8,
        0x1F2D_8534_B623_3BED,
        0xCD5A_C9D5_E29B_24AA,
        0x0D6B_7C14_6022_4661,
        0xD50E_3383_116C_E713,
    ];

    #[test]
    fn rng_reproduces_golden_sequence() {
        let mut rng = Rng::new(GOLDEN_SEED);
        let got: [u64; 8] = std::array::from_fn(|_| rng.next_u64());
        assert_eq!(
            got, GOLDEN_U64,
            "RNG golden sequence mismatch — ChaCha12 backend may have changed"
        );

        // Also must match the underlying ChaCha12 backend directly.
        let mut backend = ChaCha12Rng::seed_from_u64(GOLDEN_SEED);
        let from_backend: [u64; 8] = std::array::from_fn(|_| backend.next_u64());
        assert_eq!(got, from_backend, "Rng must wrap ChaCha12Rng faithfully");

        // Fresh construction must replay the same stream.
        let mut rng2 = Rng::new(GOLDEN_SEED);
        let got2: [u64; 8] = std::array::from_fn(|_| rng2.next_u64());
        assert_eq!(got2, GOLDEN_U64);
    }

    #[test]
    fn rng_same_seed_is_deterministic() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn rng_different_seeds_diverge() {
        let mut a = Rng::new(1);
        let mut b = Rng::new(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }
}

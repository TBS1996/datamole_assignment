use core::ops::Div;
use rand::rngs::SmallRng;
use rand::Rng;
use rand::SeedableRng;

enum Endian {
    Little,
    Big,
}

const I32_SIZE: usize = (i32::BITS / 8) as usize;

/// Keeps tracks of values within a rolling window and can perform statistical operations on them.
pub struct RollingStats<const WINDOW_SIZE: usize> {
    // The rolling window.
    data: [i32; WINDOW_SIZE],
    // the current index of the rolling window.
    index: usize,
    // Keeps track of an incomplete i32-value.
    buffer: Buffer,
    // Ensures we don't use uninitialized values in the window for statistical operations.
    window_filled: bool,
}

impl<const WINDOW_SIZE: usize> RollingStats<WINDOW_SIZE> {
    const ASSERT: () = assert!(WINDOW_SIZE != 0, "Window-size must be non-zero");

    /// Creates a new instance of [`RollingStats`] where bytes are represented as big-endian.
    pub fn new_big_endian() -> Self {
        Self::new(Endian::Big)
    }

    /// Creates a new instance of [`RollingStats`] where bytes are represented as small-endian.
    pub fn new_little_endian() -> Self {
        Self::new(Endian::Little)
    }

    /// Creates a new instance of [`RollingStats`].
    fn new(endian: Endian) -> Self {
        let _ = Self::ASSERT;
        Self {
            data: [0; WINDOW_SIZE],
            index: 0,
            buffer: Buffer::new(endian),
            window_filled: false,
        }
    }

    fn data(&self) -> &[i32] {
        match self.window_filled {
            true => &self.data,
            false => &self.data[..self.index],
        }
    }

    /// Clears the buffer.
    pub fn clear_buffer(&mut self) {
        self.buffer.index = 0;
    }

    /// Calculates a sample based on the standard deviation and mean of the dataset
    /// using the box-mueller trarnsform.
    pub fn std_sample(&self) -> f32 {
        let mean = self.mean() as f64;
        let std_dev = self.standard_deviation() as f64;
        let pi = 3.14159265358979323846264338327950288_f64;

        let mut rng = SmallRng::from_entropy();
        let u1: f64 = rng.gen::<f64>().max(1e-10);
        let u2: f64 = rng.gen::<f64>();
        let z0 = libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * pi * u2);
        (mean + z0 * std_dev) as f32
    }

    /// Calculates the standard deviation of the values within the current window of [`RollingStats`].
    pub fn standard_deviation(&self) -> f32 {
        let mean = self.mean();
        let len = self.data().len();

        if len == 0 {
            return 0.;
        }

        let variation = self
            .data()
            .iter()
            .map(|num| {
                let diff = *num as f64 - mean as f64;
                diff * diff
            })
            .sum::<f64>()
            .div(len as f64);

        libm::sqrt(variation) as f32
    }

    /// Calculates the current mean of the values within the current window.
    pub fn mean(&self) -> f32 {
        if self.data().len() == 0 {
            return 0.;
        }
        self.data().iter().sum::<i32>() as f32 / self.data().len() as f32
    }

    /// Pushes the bytes into the rolling window.
    pub fn extend(&mut self, bytes: &[u8]) {
        let max_accom = WINDOW_SIZE * I32_SIZE - self.buffer.index();
        let start_index = bytes.len().saturating_sub(max_accom);

        for byte in &bytes[start_index..] {
            self.push(*byte);
        }
    }

    fn push(&mut self, byte: u8) {
        if let Some(num) = self.buffer.push(byte) {
            self.data[self.index] = num;
            self.index = (self.index + 1) % WINDOW_SIZE;
            self.window_filled |= self.index == 0; // Set window_filled when it wraps around.
        }
    }
}

struct Buffer {
    data: [u8; I32_SIZE],
    index: usize,
    endian: Endian,
}

impl Buffer {
    fn new(endian: Endian) -> Self {
        Self {
            data: [0; I32_SIZE],
            index: 0,
            endian,
        }
    }

    fn index(&self) -> usize {
        self.index
    }

    fn push(&mut self, byte: u8) -> Option<i32> {
        self.data[self.index] = byte;
        self.index = (self.index + 1) % I32_SIZE;

        let is_buffer_full = self.index == 0;

        match (is_buffer_full, &self.endian) {
            (true, Endian::Little) => Some(i32::from_le_bytes(self.data)),
            (true, Endian::Big) => Some(i32::from_be_bytes(self.data)),
            (false, _) => None,
        }
    }
}

#[cfg(feature = "std")]
impl<const WINDOW_SIZE: usize> std::io::Write for RollingStats<WINDOW_SIZE> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.extend(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test initializing and calculating mean with big endian
    #[test]
    fn test_mean_big_endian() {
        let mut stats = RollingStats::<3>::new_big_endian();
        stats.extend(&[0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3]);
        assert_eq!(stats.mean(), 2.0); // (1+2+3)/3
    }

    // Test initializing and calculating mean with little endian
    #[test]
    fn test_mean_little_endian() {
        let mut stats = RollingStats::<3>::new_little_endian();
        stats.extend(&[1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
        assert_eq!(stats.mean(), 2.0); // (1+2+3)/3
    }

    // Test the behavior when the buffer is not yet filled (less data than window size)
    #[test]
    fn test_partial_fill() {
        let mut stats = RollingStats::<5>::new_big_endian();
        stats.extend(&[0, 0, 0, 1, 0, 0, 0, 2]);
        assert_eq!(stats.mean(), 1.5); // (1+2)/2
    }

    // Test handling of incomplete byte sequences
    #[test]
    fn test_incomplete_bytes() {
        let mut stats = RollingStats::<3>::new_big_endian();
        stats.extend(&[0, 0]); // Incomplete byte sequence for an i32
        stats.extend(&[0, 1]); // Completes the first number
        stats.extend(&[0, 0, 0, 2, 0, 0, 0, 3]); // More complete numbers
        assert_eq!(stats.mean(), 2.0); // (1+2+3)/3
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_standard_deviation() {
        let mut stats = RollingStats::<5>::new_big_endian();
        let data: Vec<i32> = vec![1, 2, 3, 4, 5];
        let data: Vec<u8> = data.iter().map(|num| num.to_be_bytes()).flatten().collect();
        stats.extend(&data);
        assert_eq!(stats.standard_deviation(), 1.4142135623730951);
    }

    #[test]
    fn test_index_wrapping() {
        let mut stats = RollingStats::<3>::new_big_endian();
        for i in 0..10 {
            let bytes = (i as i32).to_be_bytes();
            stats.extend(&bytes);
        }
        // Window should only contain the last three elements: 7, 8, 9
        assert_eq!(stats.mean(), 8.0); // (7+8+9)/3
    }

    /// The test from the assignment example.
    #[cfg(feature = "std")]
    #[test]
    fn test_write() {
        use std::io::Write;

        let mut stats = RollingStats::<3>::new_big_endian();
        assert!(stats
            .write(&[0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4])
            .is_ok());
        assert_eq!(stats.mean(), 3.0);
    }
}

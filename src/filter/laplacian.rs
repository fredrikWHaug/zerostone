//! Surface Laplacian (Current Source Density) spatial filter for EEG signal processing.
//!
//! The Surface Laplacian is a spatial high-pass filter that estimates the second spatial
//! derivative of scalp potentials, effectively computing the current source density at each
//! electrode location. This filter reduces volume conduction effects and improves spatial
//! resolution, making it particularly useful for:
//!
//! - Motor imagery brain-computer interfaces (BCI)
//! - Reducing spatial smearing from volume conduction
//! - Localizing cortical activity
//! - Improving signal-to-noise ratio for focal sources
//!
//! # Mathematical Background
//!
//! The Surface Laplacian applies the Hjorth algorithm, which approximates the second
//! spatial derivative using neighboring electrodes:
//!
//! ```text
//! V_lap[i] = V[i] - Σ(g_ij × V[j])
//! ```
//!
//! Where:
//! - `V[i]` = voltage at electrode i
//! - `V[j]` = voltage at neighboring electrode j
//! - `g_ij` = normalized weight (inverse distance or equal weights)
//!
//! For equidistant neighbors, this simplifies to:
//! ```text
//! V_lap[i] = V[i] - mean(neighbors)
//! ```
//!
//! # Neighbor Configuration
//!
//! The filter requires specifying which electrodes are neighbors for each channel.
//! Neighbors are represented as fixed-size arrays with sentinel values (`u16::MAX`)
//! marking unused slots:
//!
//! ```rust
//! use zerostone::SurfaceLaplacian;
//!
//! // 3-channel linear array: 0-1-2
//! let neighbors = [
//!     [1, u16::MAX],      // Channel 0: only right neighbor
//!     [0, 2],             // Channel 1: neighbors 0, 2
//!     [1, u16::MAX],      // Channel 2: only left neighbor
//! ];
//!
//! let laplacian: SurfaceLaplacian<3, 2> = SurfaceLaplacian::unweighted(neighbors);
//! ```
//!
//! # Weighting Schemes
//!
//! Two weighting schemes are supported:
//!
//! - **Unweighted**: Equal weights (1/N) for N neighbors. Suitable for regular montages
//!   where electrodes are approximately equidistant.
//!
//! - **Weighted**: Inverse distance weights. Provides better accuracy for irregular
//!   electrode placements. Weights are normalized to sum to 1.0 per channel.
//!
//! # Performance
//!
//! The filter is optimized for real-time processing with:
//! - Stack-only allocation (no heap usage)
//! - Compile-time size calculations
//! - Single-pass processing with minimal branching
//! - Typical performance: <500 ns/sample for 32 channels with 4 neighbors
//!
//! # Memory Layout
//!
//! For `SurfaceLaplacian<128, 8>`:
//! ```text
//! neighbor_indices: 128 × 8 × 2 bytes = 2,048 bytes
//! neighbor_weights: 128 × 8 × 4 bytes = 4,096 bytes
//! neighbor_counts:  128 × 1 byte      =   128 bytes
//! Total: ~6.3 KB (stack allocated)
//! ```

/// Surface Laplacian spatial filter.
///
/// Computes the second spatial derivative of scalp potentials using neighboring
/// electrodes to reduce volume conduction effects and improve spatial resolution.
///
/// # Type Parameters
///
/// - `C`: Number of channels (electrodes)
/// - `MAX_N`: Maximum number of neighbors per channel (typically 4-8)
///
/// # Examples
///
/// ```
/// use zerostone::SurfaceLaplacian;
///
/// // 5-channel linear array: 0-1-2-3-4
/// let neighbors = [
///     [1, u16::MAX],      // Channel 0: edge, only right neighbor
///     [0, 2],             // Channel 1: neighbors 0, 2
///     [1, 3],             // Channel 2: neighbors 1, 3 (center)
///     [2, 4],             // Channel 3: neighbors 2, 4
///     [3, u16::MAX],      // Channel 4: edge, only left neighbor
/// ];
///
/// let laplacian: SurfaceLaplacian<5, 2> = SurfaceLaplacian::unweighted(neighbors);
///
/// let samples = [1.0, 2.0, 5.0, 2.0, 1.0];
/// let filtered = laplacian.process(&samples);
///
/// // Channel 2 (center): 5.0 - (2.0 + 2.0)/2 = 3.0
/// assert!((filtered[2] - 3.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceLaplacian<const C: usize, const MAX_N: usize> {
    /// Neighbor channel indices for each channel. INVALID_INDEX marks unused slots.
    neighbor_indices: [[u16; MAX_N]; C],
    /// Normalized weights for each neighbor (sum to 1.0 per channel).
    neighbor_weights: [[f32; MAX_N]; C],
    /// Number of valid neighbors for each channel.
    neighbor_counts: [u8; C],
}

/// Sentinel value marking unused neighbor slots.
const INVALID_INDEX: u16 = u16::MAX;

impl<const C: usize, const MAX_N: usize> SurfaceLaplacian<C, MAX_N> {
    /// Compile-time assertion: C must be at least 1
    const _ASSERT_C: () = assert!(C >= 1, "C (channels) must be at least 1");

    /// Compile-time assertion: MAX_N must be in valid range
    const _ASSERT_MAX_N: () = assert!(
        MAX_N >= 1 && MAX_N <= 16,
        "MAX_N (max neighbors) must be between 1 and 16"
    );

    /// Creates a Surface Laplacian filter with equal weights for all neighbors.
    ///
    /// Each neighbor receives equal weight (1/N for N neighbors). Suitable for
    /// regular montages where electrodes are approximately equidistant.
    ///
    /// # Arguments
    ///
    /// - `neighbors`: Array of neighbor indices for each channel. Use `u16::MAX`
    ///   to mark unused slots.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - Any neighbor index is >= C (out of bounds)
    /// - A channel includes itself as a neighbor (self-reference)
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::SurfaceLaplacian;
    ///
    /// // 3-channel linear configuration
    /// let neighbors = [
    ///     [1, u16::MAX],
    ///     [0, 2],
    ///     [1, u16::MAX],
    /// ];
    ///
    /// let laplacian: SurfaceLaplacian<3, 2> = SurfaceLaplacian::unweighted(neighbors);
    /// assert_eq!(laplacian.neighbor_count(1), 2); // Middle channel has 2 neighbors
    /// ```
    pub const fn unweighted(neighbors: [[u16; MAX_N]; C]) -> Self {
        // Trigger compile-time assertions
        #[allow(clippy::let_unit_value)]
        let () = Self::_ASSERT_C;
        #[allow(clippy::let_unit_value)]
        let () = Self::_ASSERT_MAX_N;

        let neighbor_indices = neighbors;
        let mut neighbor_weights = [[0.0f32; MAX_N]; C];
        let mut neighbor_counts = [0u8; C];

        // Process each channel
        let mut ch = 0;
        while ch < C {
            let mut count = 0u8;

            // Count valid neighbors and validate indices
            let mut n = 0;
            while n < MAX_N {
                let idx = neighbor_indices[ch][n];

                if idx != INVALID_INDEX {
                    // Validate neighbor index
                    if idx as usize >= C {
                        panic!("Neighbor index out of bounds");
                    }

                    // Check for self-reference
                    if idx as usize == ch {
                        panic!("Channel cannot be its own neighbor");
                    }

                    count += 1;
                }

                n += 1;
            }

            // Store count
            neighbor_counts[ch] = count;

            // Compute equal weights: 1/N for N neighbors
            if count > 0 {
                let weight = 1.0 / (count as f32);
                let mut n = 0;
                while n < MAX_N {
                    if neighbor_indices[ch][n] != INVALID_INDEX {
                        neighbor_weights[ch][n] = weight;
                    }
                    n += 1;
                }
            }

            ch += 1;
        }

        Self {
            neighbor_indices,
            neighbor_weights,
            neighbor_counts,
        }
    }

    /// Creates a Surface Laplacian filter with inverse distance weights.
    ///
    /// Neighbors are weighted by inverse distance: w_ij = (1/d_ij) / Σ(1/d_ij).
    /// Provides better accuracy for irregular electrode placements.
    ///
    /// # Arguments
    ///
    /// - `neighbors`: Array of neighbor indices for each channel. Use `u16::MAX`
    ///   to mark unused slots.
    /// - `distances`: Array of distances to each neighbor (same structure as neighbors).
    ///   Distances for unused slots (INVALID_INDEX) are ignored.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - Any neighbor index is >= C (out of bounds)
    /// - A channel includes itself as a neighbor (self-reference)
    /// - Any distance is <= 0.0 or non-finite
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::SurfaceLaplacian;
    ///
    /// // 3-channel irregular configuration with distances
    /// let neighbors = [
    ///     [1, u16::MAX],
    ///     [0, 2],
    ///     [1, u16::MAX],
    /// ];
    ///
    /// let distances = [
    ///     [2.0, 0.0],      // Distance from 0 to 1 is 2.0
    ///     [2.0, 3.0],      // Distance from 1 to 0 is 2.0, to 2 is 3.0
    ///     [3.0, 0.0],      // Distance from 2 to 1 is 3.0
    /// ];
    ///
    /// let laplacian: SurfaceLaplacian<3, 2> = SurfaceLaplacian::weighted(neighbors, distances);
    /// ```
    pub fn weighted(neighbors: [[u16; MAX_N]; C], distances: [[f32; MAX_N]; C]) -> Self {
        // Trigger compile-time assertions
        #[allow(clippy::let_unit_value)]
        let () = Self::_ASSERT_C;
        #[allow(clippy::let_unit_value)]
        let () = Self::_ASSERT_MAX_N;

        let neighbor_indices = neighbors;
        let mut neighbor_weights = [[0.0f32; MAX_N]; C];
        let mut neighbor_counts = [0u8; C];

        // Process each channel
        for ch in 0..C {
            let mut count = 0u8;
            let mut inverse_distance_sum = 0.0f32;

            // First pass: validate and compute sum of inverse distances
            for n in 0..MAX_N {
                let idx = neighbor_indices[ch][n];

                if idx != INVALID_INDEX {
                    // Validate neighbor index
                    assert!(
                        (idx as usize) < C,
                        "Neighbor index {} out of bounds for channel {}",
                        idx,
                        ch
                    );

                    // Check for self-reference
                    assert!(
                        idx as usize != ch,
                        "Channel {} cannot be its own neighbor",
                        ch
                    );

                    // Validate distance
                    let dist = distances[ch][n];
                    assert!(
                        dist > 0.0 && dist.is_finite(),
                        "Invalid distance {} for channel {} neighbor {}",
                        dist,
                        ch,
                        n
                    );

                    inverse_distance_sum += 1.0 / dist;
                    count += 1;
                }
            }

            // Store count
            neighbor_counts[ch] = count;

            // Second pass: compute normalized weights
            if count > 0 {
                for n in 0..MAX_N {
                    let idx = neighbor_indices[ch][n];
                    if idx != INVALID_INDEX {
                        let dist = distances[ch][n];
                        neighbor_weights[ch][n] = (1.0 / dist) / inverse_distance_sum;
                    }
                }
            }
        }

        Self {
            neighbor_indices,
            neighbor_weights,
            neighbor_counts,
        }
    }

    /// Processes a single sample through the Surface Laplacian filter.
    ///
    /// Computes the spatial derivative for each channel by subtracting the
    /// weighted average of neighboring channels from the channel's own value.
    ///
    /// # Arguments
    ///
    /// - `samples`: Array of sample values for all channels
    ///
    /// # Returns
    ///
    /// Filtered sample array with the same dimensions as input
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::SurfaceLaplacian;
    ///
    /// let neighbors = [
    ///     [1, u16::MAX],
    ///     [0, 2],
    ///     [1, u16::MAX],
    /// ];
    ///
    /// let laplacian: SurfaceLaplacian<3, 2> = SurfaceLaplacian::unweighted(neighbors);
    ///
    /// let samples = [1.0, 2.0, 3.0];
    /// let filtered = laplacian.process(&samples);
    ///
    /// // Channel 1: 2.0 - (1.0 + 3.0)/2 = 0.0
    /// assert!((filtered[1] - 0.0).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn process(&self, samples: &[f32; C]) -> [f32; C] {
        let mut output = [0.0; C];

        for ch in 0..C {
            let mut neighbor_sum = 0.0;

            // Only iterate valid neighbors (cached count)
            let n_count = self.neighbor_counts[ch] as usize;
            for n in 0..n_count {
                let neighbor_idx = self.neighbor_indices[ch][n] as usize;
                let weight = self.neighbor_weights[ch][n];
                neighbor_sum += weight * samples[neighbor_idx];
            }

            // Laplacian: V[i] - weighted_mean(neighbors)
            output[ch] = samples[ch] - neighbor_sum;
        }

        output
    }

    /// Returns the number of neighbors for a given channel.
    ///
    /// # Arguments
    ///
    /// - `channel`: Channel index
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::SurfaceLaplacian;
    ///
    /// let neighbors = [
    ///     [1, u16::MAX],
    ///     [0, 2],
    ///     [1, u16::MAX],
    /// ];
    ///
    /// let laplacian: SurfaceLaplacian<3, 2> = SurfaceLaplacian::unweighted(neighbors);
    ///
    /// assert_eq!(laplacian.neighbor_count(0), 1);
    /// assert_eq!(laplacian.neighbor_count(1), 2);
    /// assert_eq!(laplacian.neighbor_count(2), 1);
    /// ```
    #[inline]
    pub const fn neighbor_count(&self, channel: usize) -> usize {
        self.neighbor_counts[channel] as usize
    }

    /// Returns the weight for a specific neighbor of a channel.
    ///
    /// # Arguments
    ///
    /// - `channel`: Channel index
    /// - `neighbor`: Neighbor index (0 to neighbor_count-1)
    ///
    /// # Panics
    ///
    /// Panics if `neighbor` >= `neighbor_count(channel)`
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::SurfaceLaplacian;
    ///
    /// let neighbors = [
    ///     [1, u16::MAX],
    ///     [0, 2],
    ///     [1, u16::MAX],
    /// ];
    ///
    /// let laplacian: SurfaceLaplacian<3, 2> = SurfaceLaplacian::unweighted(neighbors);
    ///
    /// // Channel 1 has 2 neighbors, each with weight 0.5
    /// assert!((laplacian.neighbor_weight(1, 0) - 0.5).abs() < 1e-6);
    /// assert!((laplacian.neighbor_weight(1, 1) - 0.5).abs() < 1e-6);
    /// ```
    #[inline]
    pub fn neighbor_weight(&self, channel: usize, neighbor: usize) -> f32 {
        assert!(
            neighbor < self.neighbor_count(channel),
            "Neighbor index out of bounds"
        );
        self.neighbor_weights[channel][neighbor]
    }

    /// Returns the channel index for a specific neighbor of a channel.
    ///
    /// # Arguments
    ///
    /// - `channel`: Channel index
    /// - `neighbor`: Neighbor index (0 to neighbor_count-1)
    ///
    /// # Panics
    ///
    /// Panics if `neighbor` >= `neighbor_count(channel)`
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::SurfaceLaplacian;
    ///
    /// let neighbors = [
    ///     [1, u16::MAX],
    ///     [0, 2],
    ///     [1, u16::MAX],
    /// ];
    ///
    /// let laplacian: SurfaceLaplacian<3, 2> = SurfaceLaplacian::unweighted(neighbors);
    ///
    /// // Channel 1's neighbors are channels 0 and 2
    /// assert_eq!(laplacian.neighbor_index(1, 0), 0);
    /// assert_eq!(laplacian.neighbor_index(1, 1), 2);
    /// ```
    #[inline]
    pub fn neighbor_index(&self, channel: usize, neighbor: usize) -> usize {
        assert!(
            neighbor < self.neighbor_count(channel),
            "Neighbor index out of bounds"
        );
        self.neighbor_indices[channel][neighbor] as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_channel_no_neighbors() {
        // A single channel with no neighbors should output the input unchanged
        let neighbors = [[INVALID_INDEX]];
        let laplacian: SurfaceLaplacian<1, 1> = SurfaceLaplacian::unweighted(neighbors);

        let samples = [5.0];
        let filtered = laplacian.process(&samples);

        assert_eq!(filtered[0], 5.0);
    }

    #[test]
    fn test_three_channel_linear() {
        // Linear array: 0-1-2
        let neighbors = [[1, INVALID_INDEX], [0, 2], [1, INVALID_INDEX]];

        let laplacian: SurfaceLaplacian<3, 2> = SurfaceLaplacian::unweighted(neighbors);

        let samples = [1.0, 2.0, 3.0];
        let filtered = laplacian.process(&samples);

        // Channel 0: 1.0 - 2.0 = -1.0
        assert!((filtered[0] - (-1.0)).abs() < 1e-6);

        // Channel 1: 2.0 - (1.0 + 3.0)/2 = 0.0
        assert!(filtered[1].abs() < 1e-6);

        // Channel 2: 3.0 - 2.0 = 1.0
        assert!((filtered[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_five_channel_linear_with_peak() {
        // Linear array: 0-1-2-3-4 with a peak at channel 2
        let neighbors = [
            [1, INVALID_INDEX],
            [0, 2],
            [1, 3],
            [2, 4],
            [3, INVALID_INDEX],
        ];

        let laplacian: SurfaceLaplacian<5, 2> = SurfaceLaplacian::unweighted(neighbors);

        let samples = [1.0, 2.0, 5.0, 2.0, 1.0];
        let filtered = laplacian.process(&samples);

        // Channel 2 (center): 5.0 - (2.0 + 2.0)/2 = 3.0
        assert!((filtered[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_weight_normalization_unweighted() {
        // Verify that weights sum to 1.0 for each channel
        let neighbors = [[1, INVALID_INDEX], [0, 2], [1, INVALID_INDEX]];

        let laplacian: SurfaceLaplacian<3, 2> = SurfaceLaplacian::unweighted(neighbors);

        for ch in 0..3 {
            let mut weight_sum = 0.0;
            for n in 0..laplacian.neighbor_count(ch) {
                weight_sum += laplacian.neighbor_weight(ch, n);
            }
            assert!(
                (weight_sum - 1.0).abs() < 1e-6,
                "Channel {} weights don't sum to 1.0",
                ch
            );
        }
    }

    #[test]
    fn test_weighted_with_distances() {
        // 3-channel linear with non-uniform distances
        let neighbors = [[1, INVALID_INDEX], [0, 2], [1, INVALID_INDEX]];

        let distances = [
            [2.0, 0.0], // Distance from 0 to 1 is 2.0
            [2.0, 4.0], // Distance from 1 to 0 is 2.0, to 2 is 4.0
            [4.0, 0.0], // Distance from 2 to 1 is 4.0
        ];

        let laplacian: SurfaceLaplacian<3, 2> = SurfaceLaplacian::weighted(neighbors, distances);

        // Verify weight normalization
        for ch in 0..3 {
            let mut weight_sum = 0.0;
            for n in 0..laplacian.neighbor_count(ch) {
                weight_sum += laplacian.neighbor_weight(ch, n);
            }
            assert!(
                (weight_sum - 1.0).abs() < 1e-6,
                "Channel {} weights don't sum to 1.0",
                ch
            );
        }

        // Channel 1: closer neighbor (0) should have higher weight than farther neighbor (2)
        let weight_0 = laplacian.neighbor_weight(1, 0);
        let weight_1 = laplacian.neighbor_weight(1, 1);
        assert!(
            weight_0 > weight_1,
            "Closer neighbor should have higher weight"
        );

        // Verify specific weights: 1/2 / (1/2 + 1/4) = 0.5 / 0.75 = 2/3
        assert!((weight_0 - 2.0 / 3.0).abs() < 1e-6);
        assert!((weight_1 - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_reference_independence() {
        // Surface Laplacian is second derivative, so should be reference-independent
        let neighbors = [[1, INVALID_INDEX], [0, 2], [1, INVALID_INDEX]];

        let laplacian: SurfaceLaplacian<3, 2> = SurfaceLaplacian::unweighted(neighbors);

        let samples1 = [1.0, 2.0, 3.0];
        let samples2 = [101.0, 102.0, 103.0]; // Same pattern, different reference

        let filtered1 = laplacian.process(&samples1);
        let filtered2 = laplacian.process(&samples2);

        // Results should be identical (second derivative cancels constant offset)
        for ch in 0..3 {
            assert!((filtered1[ch] - filtered2[ch]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_negative_values() {
        let neighbors = [[1, INVALID_INDEX], [0, 2], [1, INVALID_INDEX]];

        let laplacian: SurfaceLaplacian<3, 2> = SurfaceLaplacian::unweighted(neighbors);

        let samples = [-1.0, -2.0, -3.0];
        let filtered = laplacian.process(&samples);

        // Channel 1: -2.0 - (-1.0 + -3.0)/2 = -2.0 - (-2.0) = 0.0
        assert!(filtered[1].abs() < 1e-6);
    }

    #[test]
    fn test_zero_input() {
        let neighbors = [[1, INVALID_INDEX], [0, 2], [1, INVALID_INDEX]];

        let laplacian: SurfaceLaplacian<3, 2> = SurfaceLaplacian::unweighted(neighbors);

        let samples = [0.0, 0.0, 0.0];
        let filtered = laplacian.process(&samples);

        for &val in &filtered {
            assert!(val.abs() < 1e-6);
        }
    }

    #[test]
    fn test_single_neighbor() {
        // Channel with only one neighbor
        let neighbors = [[1, INVALID_INDEX], [INVALID_INDEX, INVALID_INDEX]];

        let laplacian: SurfaceLaplacian<2, 2> = SurfaceLaplacian::unweighted(neighbors);

        assert_eq!(laplacian.neighbor_count(0), 1);
        assert_eq!(laplacian.neighbor_count(1), 0);

        let samples = [5.0, 3.0];
        let filtered = laplacian.process(&samples);

        // Channel 0: 5.0 - 3.0 = 2.0
        assert!((filtered[0] - 2.0).abs() < 1e-6);

        // Channel 1: 3.0 - 0.0 = 3.0 (no neighbors)
        assert!((filtered[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_varying_neighbor_counts() {
        // Channels with 1, 2, and 3 neighbors
        let neighbors = [
            [1, INVALID_INDEX, INVALID_INDEX],
            [0, 2, INVALID_INDEX],
            [0, 1, 3],
            [2, INVALID_INDEX, INVALID_INDEX],
        ];

        let laplacian: SurfaceLaplacian<4, 3> = SurfaceLaplacian::unweighted(neighbors);

        assert_eq!(laplacian.neighbor_count(0), 1);
        assert_eq!(laplacian.neighbor_count(1), 2);
        assert_eq!(laplacian.neighbor_count(2), 3);
        assert_eq!(laplacian.neighbor_count(3), 1);

        // Verify weight normalization
        assert!((laplacian.neighbor_weight(0, 0) - 1.0).abs() < 1e-6);
        assert!((laplacian.neighbor_weight(1, 0) - 0.5).abs() < 1e-6);
        assert!((laplacian.neighbor_weight(2, 0) - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_channel_independence() {
        // Changing an unrelated channel shouldn't affect output
        let neighbors = [[1, INVALID_INDEX], [0, 2], [1, INVALID_INDEX]];

        let laplacian: SurfaceLaplacian<3, 2> = SurfaceLaplacian::unweighted(neighbors);

        let samples1 = [1.0, 2.0, 3.0];
        let samples2 = [100.0, 2.0, 3.0]; // Change channel 0

        let filtered1 = laplacian.process(&samples1);
        let filtered2 = laplacian.process(&samples2);

        // Channel 2 output should be the same (doesn't depend on channel 0)
        assert!((filtered1[2] - filtered2[2]).abs() < 1e-6);
    }

    #[test]
    fn test_large_channel_count() {
        // Test with 128 channels (typical EEG cap size)
        let mut neighbors = [[INVALID_INDEX; 4]; 128];

        // Create a ring topology: each channel connected to neighbors
        for (i, neighbor) in neighbors.iter_mut().enumerate() {
            neighbor[0] = ((i + 127) % 128) as u16; // Previous
            neighbor[1] = ((i + 1) % 128) as u16; // Next
        }

        let laplacian: SurfaceLaplacian<128, 4> = SurfaceLaplacian::unweighted(neighbors);

        let mut samples = [0.0f32; 128];
        samples[64] = 10.0; // Peak in middle

        let filtered = laplacian.process(&samples);

        // Peak should have positive Laplacian
        assert!(filtered[64] > 0.0);
    }

    #[test]
    fn test_helper_methods() {
        let neighbors = [[1, INVALID_INDEX], [0, 2], [1, INVALID_INDEX]];

        let laplacian: SurfaceLaplacian<3, 2> = SurfaceLaplacian::unweighted(neighbors);

        // Test neighbor_count
        assert_eq!(laplacian.neighbor_count(0), 1);
        assert_eq!(laplacian.neighbor_count(1), 2);
        assert_eq!(laplacian.neighbor_count(2), 1);

        // Test neighbor_index
        assert_eq!(laplacian.neighbor_index(1, 0), 0);
        assert_eq!(laplacian.neighbor_index(1, 1), 2);

        // Test neighbor_weight
        assert!((laplacian.neighbor_weight(1, 0) - 0.5).abs() < 1e-6);
        assert!((laplacian.neighbor_weight(1, 1) - 0.5).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "Neighbor index out of bounds")]
    fn test_invalid_neighbor_index() {
        let neighbors = [[5, INVALID_INDEX]]; // Index 5 is out of bounds for 1 channel
        let _laplacian: SurfaceLaplacian<1, 2> = SurfaceLaplacian::unweighted(neighbors);
    }

    #[test]
    #[should_panic(expected = "Channel cannot be its own neighbor")]
    fn test_self_reference() {
        let neighbors = [[0, INVALID_INDEX]]; // Channel 0 references itself
        let _laplacian: SurfaceLaplacian<1, 2> = SurfaceLaplacian::unweighted(neighbors);
    }

    #[test]
    #[should_panic(expected = "Invalid distance")]
    fn test_negative_distance() {
        let neighbors = [[1, INVALID_INDEX], [0, INVALID_INDEX]];
        let distances = [[-1.0, 0.0], [1.0, 0.0]]; // Negative distance
        let _laplacian: SurfaceLaplacian<2, 2> = SurfaceLaplacian::weighted(neighbors, distances);
    }

    #[test]
    #[should_panic(expected = "Invalid distance")]
    fn test_zero_distance() {
        let neighbors = [[1, INVALID_INDEX], [0, INVALID_INDEX]];
        let distances = [[0.0, 0.0], [1.0, 0.0]]; // Zero distance
        let _laplacian: SurfaceLaplacian<2, 2> = SurfaceLaplacian::weighted(neighbors, distances);
    }

    #[test]
    #[should_panic(expected = "Invalid distance")]
    fn test_infinite_distance() {
        let neighbors = [[1, INVALID_INDEX], [0, INVALID_INDEX]];
        let distances = [[f32::INFINITY, 0.0], [1.0, 0.0]];
        let _laplacian: SurfaceLaplacian<2, 2> = SurfaceLaplacian::weighted(neighbors, distances);
    }

    #[test]
    fn test_equidistant_weighted_matches_unweighted() {
        // With equal distances, weighted should match unweighted
        let neighbors = [[1, INVALID_INDEX], [0, 2], [1, INVALID_INDEX]];

        let distances = [[1.0, 0.0], [1.0, 1.0], [1.0, 0.0]];

        let unweighted: SurfaceLaplacian<3, 2> = SurfaceLaplacian::unweighted(neighbors);
        let weighted: SurfaceLaplacian<3, 2> = SurfaceLaplacian::weighted(neighbors, distances);

        let samples = [1.0, 2.0, 3.0];
        let filtered_unweighted = unweighted.process(&samples);
        let filtered_weighted = weighted.process(&samples);

        for ch in 0..3 {
            assert!(
                (filtered_unweighted[ch] - filtered_weighted[ch]).abs() < 1e-6,
                "Channel {} mismatch",
                ch
            );
        }
    }

    #[test]
    fn test_max_neighbors() {
        // Test with MAX_N neighbors per channel
        let neighbors = [
            [1, 2, 3, 4],
            [0, 2, 3, 4],
            [0, 1, 3, 4],
            [0, 1, 2, 4],
            [0, 1, 2, 3],
        ];

        let laplacian: SurfaceLaplacian<5, 4> = SurfaceLaplacian::unweighted(neighbors);

        // Each channel should have 4 neighbors
        for ch in 0..5 {
            assert_eq!(laplacian.neighbor_count(ch), 4);
        }

        // Each weight should be 0.25
        for ch in 0..5 {
            for n in 0..4 {
                assert!((laplacian.neighbor_weight(ch, n) - 0.25).abs() < 1e-6);
            }
        }
    }
}

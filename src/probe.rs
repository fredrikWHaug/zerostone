//! Probe geometry model for multi-channel electrode arrays.
//!
//! Stores 2D positions for each channel on a probe (Neuropixels, tetrodes,
//! high-density arrays) and provides spatial queries used by spike sorting:
//! neighbor channels within a radius, nearest-k channels, pairwise distance,
//! spatial extent, and channel density.
//!
//! All data lives in fixed-size arrays with const generics -- no heap, no alloc,
//! `#![no_std]` compatible.

/// 2D probe geometry with `C` channels.
///
/// Each channel has an (x, y) position in micrometers. Factory functions
/// produce common electrode layouts (linear, tetrode, polytrode).
///
/// # Type Parameters
///
/// * `C` - Number of channels (must be >= 1)
///
/// # Example
///
/// ```
/// use zerostone::probe::ProbeLayout;
///
/// // 4-channel linear probe with 25 um pitch
/// let probe = ProbeLayout::<4>::linear(25.0);
/// let d = probe.channel_distance(0, 3);
/// assert!((d - 75.0).abs() < 1e-9);
/// ```
pub struct ProbeLayout<const C: usize> {
    /// (x, y) positions in micrometers for each channel.
    positions: [[f64; 2]; C],
}

/// Compile-time assertion helper: C must be at least 1.
impl<const C: usize> ProbeLayout<C> {
    const _ASSERT_C: () = assert!(C >= 1, "ProbeLayout requires at least 1 channel");
}

impl<const C: usize> ProbeLayout<C> {
    /// Create a probe from explicit channel positions.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::probe::ProbeLayout;
    ///
    /// let positions = [[0.0, 0.0], [0.0, 25.0], [0.0, 50.0]];
    /// let probe = ProbeLayout::new(positions);
    /// assert_eq!(probe.positions()[1], [0.0, 25.0]);
    /// ```
    pub fn new(positions: [[f64; 2]; C]) -> Self {
        let () = Self::_ASSERT_C;
        Self { positions }
    }

    /// Create a linear probe with channels spaced `pitch` micrometers apart
    /// along the y-axis (x = 0 for all channels).
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::probe::ProbeLayout;
    ///
    /// let probe = ProbeLayout::<8>::linear(20.0);
    /// assert_eq!(probe.positions()[0], [0.0, 0.0]);
    /// assert_eq!(probe.positions()[7], [0.0, 140.0]);
    /// ```
    pub fn linear(pitch: f64) -> Self {
        let () = Self::_ASSERT_C;
        let mut positions = [[0.0; 2]; C];
        let mut i = 0;
        while i < C {
            positions[i] = [0.0, i as f64 * pitch];
            i += 1;
        }
        Self { positions }
    }

    /// Create a multi-column polytrode layout.
    ///
    /// Channels are arranged row by row from left to right. Each row has
    /// `columns` channels separated by `pitch_x` horizontally, centered on
    /// x = 0. Rows are separated by `pitch_y` vertically. If `C` is not
    /// divisible by `columns`, the last row has fewer channels.
    ///
    /// # Panics
    ///
    /// Panics if `columns` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::probe::ProbeLayout;
    ///
    /// // 2-column polytrode, 8 channels, 32 um x-pitch, 25 um y-pitch
    /// let probe = ProbeLayout::<8>::polytrode(2, 32.0, 25.0);
    /// // Row 0: channels 0,1 at y=0; Row 1: channels 2,3 at y=25; ...
    /// assert!((probe.positions()[0][0] - (-16.0)).abs() < 1e-9);
    /// assert!((probe.positions()[1][0] - 16.0).abs() < 1e-9);
    /// ```
    pub fn polytrode(columns: usize, pitch_x: f64, pitch_y: f64) -> Self {
        let () = Self::_ASSERT_C;
        assert!(columns >= 1, "columns must be >= 1");
        let mut positions = [[0.0; 2]; C];
        // Center x-positions so the midpoint of the row is at x=0
        let x_offset = (columns as f64 - 1.0) * pitch_x * 0.5;
        let mut i = 0;
        while i < C {
            let col = i % columns;
            let row = i / columns;
            positions[i] = [col as f64 * pitch_x - x_offset, row as f64 * pitch_y];
            i += 1;
        }
        Self { positions }
    }

    /// Read the channel positions.
    pub fn positions(&self) -> &[[f64; 2]; C] {
        &self.positions
    }

    /// Euclidean distance between two channels.
    ///
    /// Returns 0.0 if both indices are equal. Returns `f64::NAN` if either
    /// index is out of range (>= C), so this function never panics.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::probe::ProbeLayout;
    ///
    /// let probe = ProbeLayout::<4>::linear(10.0);
    /// let d = probe.channel_distance(0, 2);
    /// assert!((d - 20.0).abs() < 1e-9);
    /// ```
    pub fn channel_distance(&self, ch_a: usize, ch_b: usize) -> f64 {
        if ch_a >= C || ch_b >= C {
            return f64::NAN;
        }
        let dx = self.positions[ch_a][0] - self.positions[ch_b][0];
        let dy = self.positions[ch_a][1] - self.positions[ch_b][1];
        libm::sqrt(dx * dx + dy * dy)
    }

    /// Find all channels within `radius` micrometers of `channel`.
    ///
    /// Writes neighbor indices into `output` (excluding `channel` itself).
    /// Returns the number of neighbors found. If `channel >= C`, returns 0.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::probe::ProbeLayout;
    ///
    /// let probe = ProbeLayout::<8>::linear(10.0);
    /// let mut neighbors = [0usize; 8];
    /// let n = probe.neighbor_channels(3, 15.0, &mut neighbors);
    /// // Channels 2 and 4 are within 15 um of channel 3 (distance 10 each)
    /// assert_eq!(n, 2);
    /// ```
    pub fn neighbor_channels(&self, channel: usize, radius: f64, output: &mut [usize]) -> usize {
        if channel >= C {
            return 0;
        }
        let r_sq = radius * radius;
        let mut count = 0;
        let mut i = 0;
        while i < C {
            if i != channel {
                let dx = self.positions[i][0] - self.positions[channel][0];
                let dy = self.positions[i][1] - self.positions[channel][1];
                let dist_sq = dx * dx + dy * dy;
                if dist_sq <= r_sq && count < output.len() {
                    output[count] = i;
                    count += 1;
                }
            }
            i += 1;
        }
        count
    }

    /// Find the `k` nearest channels to `channel` (excluding itself).
    ///
    /// Writes neighbor indices into `output`, sorted by ascending distance.
    /// Returns the number of neighbors found (min of `k`, `C - 1`, and
    /// `output.len()`). If `channel >= C`, returns 0.
    ///
    /// Uses an insertion-sort approach -- efficient for small k typical of
    /// electrode neighborhoods (6-32 channels).
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::probe::ProbeLayout;
    ///
    /// let probe = ProbeLayout::<8>::linear(10.0);
    /// let mut nearest = [0usize; 3];
    /// let n = probe.nearest_channels(3, 3, &mut nearest);
    /// assert_eq!(n, 3);
    /// // Nearest to channel 3: channels 2 and 4 (dist 10), then 1 or 5 (dist 20)
    /// assert!(nearest[0] == 2 || nearest[0] == 4);
    /// ```
    pub fn nearest_channels(&self, channel: usize, k: usize, output: &mut [usize]) -> usize {
        if channel >= C || k == 0 {
            return 0;
        }
        let max_k = if k < C - 1 { k } else { C - 1 };
        let max_k = if max_k < output.len() {
            max_k
        } else {
            output.len()
        };

        // Collect (distance_sq, index) for all other channels.
        // We store up to C-1 candidates. Since C is const-generic and small
        // enough to be on the stack, this is fine.
        let mut dists = [0.0f64; C];
        let mut indices = [0usize; C];
        let mut n_candidates = 0;

        let mut i = 0;
        while i < C {
            if i != channel {
                let dx = self.positions[i][0] - self.positions[channel][0];
                let dy = self.positions[i][1] - self.positions[channel][1];
                dists[n_candidates] = dx * dx + dy * dy;
                indices[n_candidates] = i;
                n_candidates += 1;
            }
            i += 1;
        }

        // Insertion sort by distance (stable, O(C^2) but C is small)
        let mut j = 1;
        while j < n_candidates {
            let key_d = dists[j];
            let key_i = indices[j];
            let mut pos = j;
            while pos > 0 && dists[pos - 1] > key_d {
                dists[pos] = dists[pos - 1];
                indices[pos] = indices[pos - 1];
                pos -= 1;
            }
            dists[pos] = key_d;
            indices[pos] = key_i;
            j += 1;
        }

        // Copy the first max_k results
        let mut out_i = 0;
        while out_i < max_k {
            output[out_i] = indices[out_i];
            out_i += 1;
        }
        max_k
    }

    /// Spatial extent of the probe: `(x_range, y_range)`.
    ///
    /// Returns the difference between maximum and minimum coordinates on
    /// each axis. For a single-channel probe both values are 0.0.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::probe::ProbeLayout;
    ///
    /// let probe = ProbeLayout::<8>::linear(20.0);
    /// let (xr, yr) = probe.spatial_extent();
    /// assert!((xr - 0.0).abs() < 1e-9);
    /// assert!((yr - 140.0).abs() < 1e-9);
    /// ```
    pub fn spatial_extent(&self) -> (f64, f64) {
        let mut x_min = self.positions[0][0];
        let mut x_max = x_min;
        let mut y_min = self.positions[0][1];
        let mut y_max = y_min;
        let mut i = 1;
        while i < C {
            let x = self.positions[i][0];
            let y = self.positions[i][1];
            if x < x_min {
                x_min = x;
            }
            if x > x_max {
                x_max = x;
            }
            if y < y_min {
                y_min = y;
            }
            if y > y_max {
                y_max = y;
            }
            i += 1;
        }
        (x_max - x_min, y_max - y_min)
    }

    /// Estimate channel density: `C / bounding_area` in channels per um^2.
    ///
    /// If the bounding area is zero (e.g., all channels collinear or a single
    /// channel), returns `f64::INFINITY`.
    ///
    /// # Example
    ///
    /// ```
    /// use zerostone::probe::ProbeLayout;
    ///
    /// let probe = ProbeLayout::<8>::polytrode(2, 32.0, 25.0);
    /// let density = probe.channel_density();
    /// assert!(density > 0.0);
    /// assert!(density.is_finite());
    /// ```
    pub fn channel_density(&self) -> f64 {
        let (xr, yr) = self.spatial_extent();
        let area = xr * yr;
        if area <= 0.0 {
            return f64::INFINITY;
        }
        C as f64 / area
    }

    /// Number of channels.
    pub fn n_channels(&self) -> usize {
        C
    }
}

/// Create a 4-channel tetrode geometry.
///
/// Channels are placed at the corners of a square with side length `pitch`
/// (in micrometers), centered on the origin.
///
/// # Example
///
/// ```
/// use zerostone::probe::tetrode;
///
/// let probe = tetrode(25.0);
/// let d01 = probe.channel_distance(0, 1);
/// assert!((d01 - 25.0).abs() < 1e-9);
/// ```
pub fn tetrode(pitch: f64) -> ProbeLayout<4> {
    let half = pitch * 0.5;
    ProbeLayout::new([[-half, -half], [half, -half], [-half, half], [half, half]])
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove that `channel_distance` never panics for arbitrary channel indices.
    #[kani::proof]
    #[kani::unwind(6)]
    fn channel_distance_no_panic() {
        let probe = ProbeLayout::<4>::linear(25.0);
        let a: usize = kani::any();
        let b: usize = kani::any();
        kani::assume(a <= 8);
        kani::assume(b <= 8);
        let d = probe.channel_distance(a, b);
        // For valid indices: d >= 0 and finite. For invalid: NAN.
        if a < 4 && b < 4 {
            assert!(d >= 0.0 && d.is_finite());
        } else {
            assert!(d.is_nan());
        }
    }

    /// Prove that `neighbor_channels` returns only valid channel indices.
    #[kani::proof]
    #[kani::unwind(6)]
    fn neighbor_channels_valid_indices() {
        let probe = ProbeLayout::<4>::linear(25.0);
        let ch: usize = kani::any();
        let radius: f64 = kani::any();
        kani::assume(ch <= 8);
        kani::assume(radius.is_finite() && radius >= 0.0 && radius <= 1000.0);

        let mut output = [0usize; 4];
        let n = probe.neighbor_channels(ch, radius, &mut output);
        assert!(n <= 4);
        let mut i = 0;
        while i < n {
            assert!(output[i] < 4, "neighbor index must be < C");
            assert!(output[i] != ch, "neighbor must not be the query channel");
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_probe_creation() {
        let probe = ProbeLayout::<4>::linear(25.0);
        assert_eq!(probe.positions()[0], [0.0, 0.0]);
        assert_eq!(probe.positions()[1], [0.0, 25.0]);
        assert_eq!(probe.positions()[2], [0.0, 50.0]);
        assert_eq!(probe.positions()[3], [0.0, 75.0]);
    }

    #[test]
    fn test_linear_distances() {
        let probe = ProbeLayout::<4>::linear(10.0);
        let d01 = probe.channel_distance(0, 1);
        let d02 = probe.channel_distance(0, 2);
        let d03 = probe.channel_distance(0, 3);
        assert!((d01 - 10.0).abs() < 1e-9);
        assert!((d02 - 20.0).abs() < 1e-9);
        assert!((d03 - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_distance_symmetry() {
        let probe = ProbeLayout::<4>::linear(10.0);
        let d12 = probe.channel_distance(1, 2);
        let d21 = probe.channel_distance(2, 1);
        assert!((d12 - d21).abs() < 1e-12);
    }

    #[test]
    fn test_distance_self_zero() {
        let probe = ProbeLayout::<4>::linear(10.0);
        for i in 0..4 {
            assert!((probe.channel_distance(i, i)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_distance_out_of_range() {
        let probe = ProbeLayout::<4>::linear(10.0);
        assert!(probe.channel_distance(0, 4).is_nan());
        assert!(probe.channel_distance(4, 0).is_nan());
        assert!(probe.channel_distance(100, 200).is_nan());
    }

    #[test]
    fn test_tetrode_geometry() {
        let probe = tetrode(20.0);
        // All 4 channels at corners of a 20x20 square centered at origin
        assert_eq!(probe.positions()[0], [-10.0, -10.0]);
        assert_eq!(probe.positions()[1], [10.0, -10.0]);
        assert_eq!(probe.positions()[2], [-10.0, 10.0]);
        assert_eq!(probe.positions()[3], [10.0, 10.0]);

        // Side length should be pitch
        let side = probe.channel_distance(0, 1);
        assert!((side - 20.0).abs() < 1e-9);

        // Diagonal should be pitch * sqrt(2)
        let diag = probe.channel_distance(0, 3);
        assert!((diag - 20.0 * libm::sqrt(2.0)).abs() < 1e-9);
    }

    #[test]
    fn test_polytrode_2col() {
        let probe = ProbeLayout::<8>::polytrode(2, 32.0, 25.0);
        // Row 0: channels 0,1 at y=0
        assert!((probe.positions()[0][1] - 0.0).abs() < 1e-9);
        assert!((probe.positions()[1][1] - 0.0).abs() < 1e-9);
        // Row 1: channels 2,3 at y=25
        assert!((probe.positions()[2][1] - 25.0).abs() < 1e-9);
        // x should be symmetric: -16 and 16
        assert!((probe.positions()[0][0] - (-16.0)).abs() < 1e-9);
        assert!((probe.positions()[1][0] - 16.0).abs() < 1e-9);
    }

    #[test]
    fn test_polytrode_3col() {
        let probe = ProbeLayout::<6>::polytrode(3, 10.0, 20.0);
        // Row 0: channels 0,1,2 at y=0 with x = -10, 0, 10
        assert!((probe.positions()[0][0] - (-10.0)).abs() < 1e-9);
        assert!((probe.positions()[1][0] - 0.0).abs() < 1e-9);
        assert!((probe.positions()[2][0] - 10.0).abs() < 1e-9);
        // Row 1: channels 3,4,5 at y=20
        assert!((probe.positions()[3][1] - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_neighbor_channels_linear() {
        let probe = ProbeLayout::<8>::linear(10.0);
        let mut neighbors = [0usize; 8];

        // Channel 3 with radius 15: should find channels 2 and 4 (distance 10)
        let n = probe.neighbor_channels(3, 15.0, &mut neighbors);
        assert_eq!(n, 2);
        assert!(neighbors[..n].contains(&2));
        assert!(neighbors[..n].contains(&4));
    }

    #[test]
    fn test_neighbor_channels_large_radius() {
        let probe = ProbeLayout::<4>::linear(10.0);
        let mut neighbors = [0usize; 4];
        // Radius covers entire probe
        let n = probe.neighbor_channels(0, 1000.0, &mut neighbors);
        assert_eq!(n, 3); // all except self
    }

    #[test]
    fn test_neighbor_channels_zero_radius() {
        let probe = ProbeLayout::<4>::linear(10.0);
        let mut neighbors = [0usize; 4];
        let n = probe.neighbor_channels(0, 0.0, &mut neighbors);
        // Zero radius: only channels at the exact same position qualify
        // Since it is a linear probe with distinct positions, no neighbors
        assert_eq!(n, 0);
    }

    #[test]
    fn test_neighbor_channels_out_of_range() {
        let probe = ProbeLayout::<4>::linear(10.0);
        let mut neighbors = [0usize; 4];
        let n = probe.neighbor_channels(10, 100.0, &mut neighbors);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_nearest_channels_linear() {
        let probe = ProbeLayout::<8>::linear(10.0);
        let mut nearest = [0usize; 3];
        let n = probe.nearest_channels(3, 3, &mut nearest);
        assert_eq!(n, 3);
        // Two nearest at distance 10 (channels 2,4), then one at 20 (channel 1 or 5)
        // The first two should be 2 and 4 (in some order)
        assert!((nearest[0] == 2 || nearest[0] == 4));
        assert!((nearest[1] == 2 || nearest[1] == 4));
        assert!(nearest[0] != nearest[1]);
    }

    #[test]
    fn test_nearest_channels_k_exceeds_c() {
        let probe = ProbeLayout::<3>::linear(10.0);
        let mut nearest = [0usize; 10];
        let n = probe.nearest_channels(0, 10, &mut nearest);
        // Can return at most C-1 = 2 neighbors
        assert_eq!(n, 2);
    }

    #[test]
    fn test_nearest_channels_out_of_range() {
        let probe = ProbeLayout::<4>::linear(10.0);
        let mut nearest = [0usize; 4];
        let n = probe.nearest_channels(10, 3, &mut nearest);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_nearest_channels_k_zero() {
        let probe = ProbeLayout::<4>::linear(10.0);
        let mut nearest = [0usize; 4];
        let n = probe.nearest_channels(0, 0, &mut nearest);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_spatial_extent_linear() {
        let probe = ProbeLayout::<8>::linear(20.0);
        let (xr, yr) = probe.spatial_extent();
        assert!((xr - 0.0).abs() < 1e-9);
        assert!((yr - 140.0).abs() < 1e-9);
    }

    #[test]
    fn test_spatial_extent_polytrode() {
        let probe = ProbeLayout::<8>::polytrode(2, 32.0, 25.0);
        let (xr, yr) = probe.spatial_extent();
        assert!((xr - 32.0).abs() < 1e-9);
        assert!((yr - 75.0).abs() < 1e-9); // 4 rows: 0, 25, 50, 75
    }

    #[test]
    fn test_channel_density() {
        let probe = ProbeLayout::<8>::polytrode(2, 32.0, 25.0);
        let density = probe.channel_density();
        // area = 32 * 75 = 2400; density = 8/2400
        let expected = 8.0 / (32.0 * 75.0);
        assert!((density - expected).abs() < 1e-12);
    }

    #[test]
    fn test_channel_density_collinear() {
        let probe = ProbeLayout::<4>::linear(10.0);
        let density = probe.channel_density();
        assert!(density.is_infinite());
    }

    #[test]
    fn test_single_channel() {
        let probe = ProbeLayout::<1>::new([[5.0, 10.0]]);
        assert_eq!(probe.n_channels(), 1);
        let (xr, yr) = probe.spatial_extent();
        assert!((xr).abs() < 1e-12);
        assert!((yr).abs() < 1e-12);
        assert!(probe.channel_density().is_infinite());

        let mut neighbors = [0usize; 1];
        let n = probe.neighbor_channels(0, 1000.0, &mut neighbors);
        assert_eq!(n, 0);

        let mut nearest = [0usize; 1];
        let n = probe.nearest_channels(0, 1, &mut nearest);
        assert_eq!(n, 0);
    }

    #[test]
    fn test_all_channels_same_position() {
        let positions = [[0.0, 0.0]; 4];
        let probe = ProbeLayout::new(positions);

        // All distances should be 0
        for i in 0..4 {
            for j in 0..4 {
                assert!((probe.channel_distance(i, j)).abs() < 1e-12);
            }
        }

        // All other channels are neighbors at any radius (including 0)
        let mut neighbors = [0usize; 4];
        let n = probe.neighbor_channels(0, 0.0, &mut neighbors);
        assert_eq!(n, 3);

        // nearest-k should return all others
        let mut nearest = [0usize; 4];
        let n = probe.nearest_channels(0, 3, &mut nearest);
        assert_eq!(n, 3);
    }

    #[test]
    fn test_n_channels() {
        let probe = ProbeLayout::<16>::linear(10.0);
        assert_eq!(probe.n_channels(), 16);
    }

    #[test]
    fn test_tetrode_spatial_extent() {
        let probe = tetrode(30.0);
        let (xr, yr) = probe.spatial_extent();
        assert!((xr - 30.0).abs() < 1e-9);
        assert!((yr - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_tetrode_neighbors() {
        let probe = tetrode(20.0);
        let mut neighbors = [0usize; 4];
        // radius 25: side length is 20, so adjacent corners are within range
        // diagonal is ~28.28, so not within 25
        let n = probe.neighbor_channels(0, 25.0, &mut neighbors);
        assert_eq!(n, 2); // channels 1 and 2 are adjacent sides
        assert!(neighbors[..n].contains(&1));
        assert!(neighbors[..n].contains(&2));
    }

    #[test]
    fn test_polytrode_single_column_equals_linear() {
        let linear = ProbeLayout::<4>::linear(25.0);
        let poly = ProbeLayout::<4>::polytrode(1, 0.0, 25.0);
        for i in 0..4 {
            assert!((linear.positions()[i][0] - poly.positions()[i][0]).abs() < 1e-9);
            assert!((linear.positions()[i][1] - poly.positions()[i][1]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_neighbor_output_buffer_smaller_than_c() {
        let probe = ProbeLayout::<8>::linear(10.0);
        let mut neighbors = [0usize; 2];
        // Channel 4 has many neighbors at radius 1000 but buffer only holds 2
        let n = probe.neighbor_channels(4, 1000.0, &mut neighbors);
        assert_eq!(n, 2);
        // All returned indices should be valid
        for &idx in &neighbors[..n] {
            assert!(idx < 8);
        }
    }

    #[test]
    fn test_nearest_output_buffer_smaller_than_k() {
        let probe = ProbeLayout::<8>::linear(10.0);
        let mut nearest = [0usize; 2];
        // Ask for 5 nearest but buffer only holds 2
        let n = probe.nearest_channels(4, 5, &mut nearest);
        assert_eq!(n, 2);
        for &idx in &nearest[..n] {
            assert!(idx < 8);
        }
    }
}

//! Inter-Spike Interval (ISI) analysis for extracellular electrophysiology.
//!
//! Provides streaming ISI histogram computation, firing regularity metrics,
//! burst detection, and autocorrelogram construction from spike times.

/// Fixed-bin ISI histogram with streaming updates.
///
/// Accumulates inter-spike intervals into fixed-width bins. Intervals
/// exceeding the histogram range are counted in an overflow bin.
///
/// # Type Parameters
///
/// * `BINS` - Number of histogram bins
///
/// # Example
///
/// ```
/// use zerostone::IsiHistogram;
///
/// let mut hist = IsiHistogram::<50>::new(1.0); // 1ms bin width
/// let spike_times = [0.100, 0.120, 0.155, 0.200];
/// hist.add_train(&spike_times);
/// assert_eq!(hist.total_count(), 3);
/// ```
pub struct IsiHistogram<const BINS: usize> {
    bins: [u64; BINS],
    bin_width: f64,
    overflow: u64,
    sum: f64,
    sum_sq: f64,
    count: u64,
}

impl<const BINS: usize> IsiHistogram<BINS> {
    /// Create a new ISI histogram.
    ///
    /// `bin_width` is in the same units as the spike times (typically seconds or milliseconds).
    /// The histogram covers the range `[0, BINS * bin_width)`.
    pub fn new(bin_width: f64) -> Self {
        assert!(bin_width > 0.0, "bin_width must be positive");
        Self {
            bins: [0; BINS],
            bin_width,
            overflow: 0,
            sum: 0.0,
            sum_sq: 0.0,
            count: 0,
        }
    }

    /// Add a single ISI value to the histogram.
    pub fn add_interval(&mut self, isi: f64) {
        if isi < 0.0 {
            return;
        }
        self.sum += isi;
        self.sum_sq += isi * isi;
        self.count += 1;

        let bin = (isi / self.bin_width) as usize;
        if bin < BINS {
            self.bins[bin] += 1;
        } else {
            self.overflow += 1;
        }
    }

    /// Compute ISIs from a sorted spike train and add them all.
    ///
    /// Spike times must be sorted in ascending order.
    pub fn add_train(&mut self, spike_times: &[f64]) {
        if spike_times.len() < 2 {
            return;
        }
        for i in 1..spike_times.len() {
            let isi = spike_times[i] - spike_times[i - 1];
            self.add_interval(isi);
        }
    }

    /// Compute ISIs from sorted spike time indices and sample rate.
    ///
    /// Converts sample indices to time intervals using `1.0 / sample_rate`.
    pub fn add_train_samples(&mut self, spike_indices: &[usize], sample_rate: f64) {
        if spike_indices.len() < 2 {
            return;
        }
        let inv_sr = 1.0 / sample_rate;
        for i in 1..spike_indices.len() {
            let isi = (spike_indices[i] as f64 - spike_indices[i - 1] as f64) * inv_sr;
            self.add_interval(isi);
        }
    }

    /// Total number of intervals recorded (excluding overflow).
    pub fn binned_count(&self) -> u64 {
        let mut total = 0u64;
        for &b in &self.bins {
            total += b;
        }
        total
    }

    /// Total number of intervals recorded (including overflow).
    pub fn total_count(&self) -> u64 {
        self.count
    }

    /// Number of overflow intervals (ISI >= BINS * bin_width).
    pub fn overflow_count(&self) -> u64 {
        self.overflow
    }

    /// Mean ISI computed from all intervals.
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum / self.count as f64
    }

    /// Variance of ISI computed from all intervals.
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f64;
        let mean = self.sum / n;
        self.sum_sq / n - mean * mean
    }

    /// Standard deviation of ISI.
    pub fn std_dev(&self) -> f64 {
        let v = self.variance();
        if v <= 0.0 {
            return 0.0;
        }
        libm::sqrt(v)
    }

    /// Coefficient of variation: std / mean.
    ///
    /// CV = 0 for perfectly regular firing. CV = 1 for Poisson firing.
    /// CV > 1 indicates bursty firing.
    pub fn coefficient_of_variation(&self) -> f64 {
        let m = self.mean();
        if m <= 0.0 {
            return 0.0;
        }
        self.std_dev() / m
    }

    /// Burst index: fraction of ISIs below the given threshold.
    ///
    /// Returns value in [0, 1]. Higher values indicate more bursty firing.
    /// Typical burst threshold: 5-10 ms for cortical neurons.
    pub fn burst_index(&self, threshold: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        let max_bin = (threshold / self.bin_width) as usize;
        let mut burst_count = 0u64;
        let limit = if max_bin < BINS { max_bin } else { BINS };
        for i in 0..limit {
            burst_count += self.bins[i];
        }
        burst_count as f64 / self.count as f64
    }

    /// Read the histogram bins.
    pub fn bins(&self) -> &[u64; BINS] {
        &self.bins
    }

    /// Bin width.
    pub fn bin_width(&self) -> f64 {
        self.bin_width
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        self.bins = [0; BINS];
        self.overflow = 0;
        self.sum = 0.0;
        self.sum_sq = 0.0;
        self.count = 0;
    }

    /// Find the mode bin (most frequent ISI range).
    ///
    /// Returns `(bin_index, count)` or `None` if empty.
    pub fn mode_bin(&self) -> Option<(usize, u64)> {
        if self.count == 0 {
            return None;
        }
        let mut max_idx = 0;
        let mut max_val = self.bins[0];
        for i in 1..BINS {
            if self.bins[i] > max_val {
                max_val = self.bins[i];
                max_idx = i;
            }
        }
        Some((max_idx, max_val))
    }
}

/// Compute an autocorrelogram from spike times.
///
/// For each spike, counts how many other spikes fall into each time lag bin,
/// up to `max_lag`. This reveals refractory periods and rhythmic firing patterns.
///
/// # Arguments
///
/// * `spike_times` - Sorted spike times (ascending order)
/// * `bin_width` - Width of each lag bin (same units as spike times)
/// * `max_lag` - Maximum lag to compute (same units as spike times)
/// * `output` - Output buffer for the autocorrelogram (length >= max_lag / bin_width)
///
/// Returns the number of bins filled in `output`.
pub fn autocorrelogram(
    spike_times: &[f64],
    bin_width: f64,
    max_lag: f64,
    output: &mut [u64],
) -> usize {
    let n_bins = (max_lag / bin_width) as usize;
    let n_bins = if n_bins < output.len() {
        n_bins
    } else {
        output.len()
    };

    for b in output.iter_mut().take(n_bins) {
        *b = 0;
    }

    let n = spike_times.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let dt = spike_times[j] - spike_times[i];
            if dt >= max_lag {
                break;
            }
            let bin = (dt / bin_width) as usize;
            if bin < n_bins {
                output[bin] += 1;
            }
        }
    }

    n_bins
}

/// Compute the coefficient of variation directly from spike times.
///
/// CV = std(ISI) / mean(ISI). Returns 0.0 if fewer than 2 spikes.
pub fn isi_cv(spike_times: &[f64]) -> f64 {
    let n = spike_times.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let count = (n - 1) as f64;
    for i in 1..n {
        let isi = spike_times[i] - spike_times[i - 1];
        sum += isi;
        sum_sq += isi * isi;
    }
    let mean = sum / count;
    if mean <= 0.0 {
        return 0.0;
    }
    let var = sum_sq / count - mean * mean;
    if var <= 0.0 {
        return 0.0;
    }
    libm::sqrt(var) / mean
}

/// Local variation (Lv) of ISIs -- measures local firing irregularity.
///
/// Lv = (3 / (n-1)) * sum( ((ISI_i - ISI_{i+1}) / (ISI_i + ISI_{i+1}))^2 )
///
/// Lv = 1 for Poisson process, Lv < 1 for regular firing, Lv > 1 for bursty firing.
/// More robust than CV because it compares adjacent intervals rather than global stats.
///
/// Returns 0.0 if fewer than 3 spikes (need at least 2 consecutive ISIs).
pub fn local_variation(spike_times: &[f64]) -> f64 {
    let n = spike_times.len();
    if n < 3 {
        return 0.0;
    }
    let n_isi = n - 1;
    let mut sum = 0.0;
    for i in 0..n_isi - 1 {
        let isi_a = spike_times[i + 1] - spike_times[i];
        let isi_b = spike_times[i + 2] - spike_times[i + 1];
        let denom = isi_a + isi_b;
        if denom > 0.0 {
            let diff = isi_a - isi_b;
            sum += (diff * diff) / (denom * denom);
        }
    }
    3.0 * sum / (n_isi - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_basic() {
        let mut hist = IsiHistogram::<10>::new(10.0); // 10ms bins, using ms units
                                                      // Spike times in ms -- all exactly representable
        let spike_times = [0.0, 20.0, 40.0, 60.0, 80.0]; // 20ms intervals
        hist.add_train(&spike_times);
        assert_eq!(hist.total_count(), 4);
        assert_eq!(hist.bins()[2], 4); // bin 2 covers [20, 30)
    }

    #[test]
    fn test_histogram_overflow() {
        let mut hist = IsiHistogram::<5>::new(0.01); // covers [0, 0.05)
        hist.add_interval(0.06); // overflow
        assert_eq!(hist.overflow_count(), 1);
        assert_eq!(hist.binned_count(), 0);
        assert_eq!(hist.total_count(), 1);
    }

    #[test]
    fn test_cv_constant_intervals() {
        let spike_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let cv = isi_cv(&spike_times);
        assert!(
            cv < 1e-10,
            "CV of constant intervals should be ~0, got {}",
            cv
        );
    }

    #[test]
    fn test_cv_via_histogram() {
        let mut hist = IsiHistogram::<100>::new(0.001);
        let spike_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        hist.add_train(&spike_times);
        let cv = hist.coefficient_of_variation();
        assert!(
            cv < 1e-10,
            "CV of constant intervals should be ~0, got {}",
            cv
        );
    }

    #[test]
    fn test_burst_index_no_bursts() {
        let mut hist = IsiHistogram::<100>::new(0.001); // 1ms bins
                                                        // Regular 100ms intervals -- no bursts at 10ms threshold
        let spike_times = [0.0, 0.1, 0.2, 0.3, 0.4];
        hist.add_train(&spike_times);
        let bi = hist.burst_index(0.010); // 10ms threshold
        assert!(bi < 1e-10, "No bursts expected, got {}", bi);
    }

    #[test]
    fn test_burst_index_all_bursts() {
        let mut hist = IsiHistogram::<100>::new(0.001); // 1ms bins
                                                        // All intervals are 3ms -- all below 10ms threshold
        let spike_times = [0.0, 0.003, 0.006, 0.009, 0.012];
        hist.add_train(&spike_times);
        let bi = hist.burst_index(0.010);
        assert!((bi - 1.0).abs() < 1e-10, "All bursts expected, got {}", bi);
    }

    #[test]
    fn test_burst_index_mixed() {
        let mut hist = IsiHistogram::<200>::new(0.001);
        // 2 short (5ms) + 2 long (100ms) intervals
        let spike_times = [0.0, 0.005, 0.010, 0.110, 0.210];
        hist.add_train(&spike_times);
        let bi = hist.burst_index(0.010);
        assert!((bi - 0.5).abs() < 1e-10, "50% bursts expected, got {}", bi);
    }

    #[test]
    fn test_autocorrelogram_basic() {
        // Spikes at regular 10ms intervals
        let spike_times = [0.0, 0.010, 0.020, 0.030, 0.040];
        let mut output = [0u64; 10];
        let n = autocorrelogram(&spike_times, 0.005, 0.050, &mut output);
        assert_eq!(n, 10);
        // Bin at lag ~10ms (bin index 2) should have most counts
        assert!(output[2] > 0, "Expected counts at 10ms lag");
    }

    #[test]
    fn test_autocorrelogram_empty() {
        let spike_times: [f64; 0] = [];
        let mut output = [0u64; 10];
        let n = autocorrelogram(&spike_times, 0.001, 0.050, &mut output);
        for &v in &output[..n] {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn test_local_variation_regular() {
        // Perfectly regular firing: Lv should be 0
        let spike_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        let lv = local_variation(&spike_times);
        assert!(lv < 1e-10, "Regular firing should have Lv ~0, got {}", lv);
    }

    #[test]
    fn test_local_variation_too_few_spikes() {
        assert_eq!(local_variation(&[0.0, 0.1]), 0.0);
        assert_eq!(local_variation(&[0.0]), 0.0);
        assert_eq!(local_variation(&[]), 0.0);
    }

    #[test]
    fn test_histogram_from_samples() {
        let mut hist = IsiHistogram::<100>::new(0.001); // 1ms bins
        let indices = [0, 30, 60, 90]; // at 30kHz, 30 samples = 1ms
        hist.add_train_samples(&indices, 30000.0);
        assert_eq!(hist.total_count(), 3);
        // 30 samples / 30000 Hz = 0.001s = exactly 1ms -> bin 1
        assert_eq!(hist.bins()[1], 3);
    }

    #[test]
    fn test_mean_and_std() {
        let mut hist = IsiHistogram::<100>::new(0.001);
        hist.add_interval(0.010);
        hist.add_interval(0.010);
        hist.add_interval(0.010);
        assert!((hist.mean() - 0.010).abs() < 1e-12);
        assert!(hist.std_dev() < 1e-12);
    }

    #[test]
    fn test_mode_bin() {
        let mut hist = IsiHistogram::<10>::new(0.01);
        hist.add_interval(0.025); // bin 2
        hist.add_interval(0.026); // bin 2
        hist.add_interval(0.055); // bin 5
        let (mode_idx, mode_count) = hist.mode_bin().unwrap();
        assert_eq!(mode_idx, 2);
        assert_eq!(mode_count, 2);
    }

    #[test]
    fn test_reset() {
        let mut hist = IsiHistogram::<10>::new(0.01);
        hist.add_interval(0.005);
        hist.add_interval(0.015);
        assert_eq!(hist.total_count(), 2);
        hist.reset();
        assert_eq!(hist.total_count(), 0);
        assert_eq!(hist.mean(), 0.0);
    }

    #[test]
    fn test_negative_interval_ignored() {
        let mut hist = IsiHistogram::<10>::new(0.01);
        hist.add_interval(-0.005);
        assert_eq!(hist.total_count(), 0);
    }
}

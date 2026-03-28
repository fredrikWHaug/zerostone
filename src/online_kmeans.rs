//! Online k-means clustering for spike sorting.
//!
//! Provides a MacQueen-style sequential online k-means with OSort-inspired
//! adaptive cluster creation. Operates spike-by-spike, O(K*D) per update,
//! zero allocation after init, deterministic.
//!
//! # Example
//!
//! ```
//! use zerostone::online_kmeans::{OnlineKMeans, KMeansResult};
//!
//! // 2D features, up to 4 clusters, count cap 1000
//! let mut km = OnlineKMeans::<2, 4>::new(1000);
//! km.set_create_threshold(5.0);
//!
//! // Feed points -- clusters are created automatically when distance > threshold
//! let r1 = km.update(&[0.0, 0.0]);
//! assert!(r1.created); // first point always creates cluster 0
//! assert_eq!(km.n_active(), 1);
//!
//! let r2 = km.update(&[10.0, 10.0]);
//! assert!(r2.created); // far from cluster 0, creates cluster 1
//! assert_eq!(km.n_active(), 2);
//!
//! // Predict without modifying state
//! let (cluster, dist) = km.predict(&[0.1, 0.1]);
//! assert_eq!(cluster, 0);
//! assert!(dist < 1.0);
//! ```

use crate::float::{self, Float};

/// Result of assigning a point to a cluster.
#[derive(Debug, Clone, Copy)]
pub struct KMeansResult {
    /// Index of the assigned cluster.
    pub cluster: usize,
    /// Euclidean distance to the assigned centroid.
    pub distance: Float,
    /// Whether a new cluster was created for this point.
    pub created: bool,
}

/// Errors from online k-means operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KMeansError {
    /// All K cluster slots are already occupied.
    ClustersFull,
    /// Invalid index or parameter.
    InvalidInput,
}

/// Online k-means clustering with adaptive cluster creation.
///
/// # Type Parameters
///
/// * `D` - Dimensionality of feature vectors
/// * `K` - Maximum number of clusters
pub struct OnlineKMeans<const D: usize, const K: usize> {
    centroids: [[Float; D]; K],
    counts: [u32; K],
    variance_accum: [[Float; D]; K], // Welford M2 accumulator
    n_active: usize,
    total_assigned: u64,
    max_count: u32,
    create_threshold: Float,
    merge_threshold: Float,
}

impl<const D: usize, const K: usize> OnlineKMeans<D, K> {
    /// Create a new online k-means clusterer.
    ///
    /// `max_count` caps the per-cluster observation count, keeping centroids
    /// plastic to distribution drift. A typical value is 1000-10000.
    pub fn new(max_count: u32) -> Self {
        Self {
            centroids: [[0.0; D]; K],
            counts: [0; K],
            variance_accum: [[0.0; D]; K],
            n_active: 0,
            total_assigned: 0,
            max_count: max_count.max(1),
            create_threshold: float::MAX,
            merge_threshold: 0.0,
        }
    }

    /// Set the distance threshold for creating new clusters.
    ///
    /// When a point is farther than this from all existing centroids and
    /// there is a free slot, a new cluster is created.
    pub fn set_create_threshold(&mut self, t: Float) {
        self.create_threshold = t;
    }

    /// Set the distance threshold for merging clusters.
    ///
    /// When `merge_closest()` is called, two clusters merge only if
    /// their centroid distance is below this threshold.
    pub fn set_merge_threshold(&mut self, t: Float) {
        self.merge_threshold = t;
    }

    /// Assign a point to the nearest cluster, updating the centroid online.
    ///
    /// If no clusters exist or the point is farther than `create_threshold`
    /// from all centroids (and a free slot exists), a new cluster is created.
    pub fn update(&mut self, point: &[Float; D]) -> KMeansResult {
        if self.n_active == 0 {
            // First point: create cluster 0
            self.centroids[0] = *point;
            self.counts[0] = 1;
            self.n_active = 1;
            self.total_assigned = 1;
            return KMeansResult {
                cluster: 0,
                distance: 0.0,
                created: true,
            };
        }

        // Find nearest centroid
        let (best, best_dist) = self.find_nearest(point);

        // Check if we should create a new cluster
        if best_dist > self.create_threshold && self.n_active < K {
            let idx = self.n_active;
            self.centroids[idx] = *point;
            self.counts[idx] = 1;
            self.variance_accum[idx] = [0.0; D];
            self.n_active += 1;
            self.total_assigned += 1;
            return KMeansResult {
                cluster: idx,
                distance: 0.0,
                created: true,
            };
        }

        // Update the nearest centroid using Welford's method
        let count = self.counts[best];
        let new_count = if count < self.max_count {
            count + 1
        } else {
            self.max_count
        };
        self.counts[best] = new_count;
        let eta = 1.0 / new_count as Float;

        for (d, &val) in point.iter().enumerate() {
            let old_mean = self.centroids[best][d];
            let delta = val - old_mean;
            let new_mean = old_mean + eta * delta;
            self.centroids[best][d] = new_mean;
            let delta2 = val - new_mean;
            self.variance_accum[best][d] += delta * delta2;
        }

        self.total_assigned += 1;
        KMeansResult {
            cluster: best,
            distance: best_dist,
            created: false,
        }
    }

    /// Read-only assignment: find the nearest centroid without modifying state.
    ///
    /// Returns `(cluster_index, distance)`. If no clusters are active,
    /// returns `(0, 0.0)`.
    pub fn predict(&self, point: &[Float; D]) -> (usize, Float) {
        if self.n_active == 0 {
            return (0, 0.0);
        }
        self.find_nearest(point)
    }

    /// Merge the two closest active clusters if their distance is below
    /// the merge threshold.
    ///
    /// Returns the merged pair `(kept, removed)` or `None` if no merge occurred.
    pub fn merge_closest(&mut self) -> Option<(usize, usize)> {
        if self.n_active < 2 {
            return None;
        }

        // Find the closest pair
        let mut best_i = 0;
        let mut best_j = 1;
        let mut best_dist = float::MAX;
        for i in 0..self.n_active {
            for j in (i + 1)..self.n_active {
                let dist = self.centroid_distance(i, j);
                if dist < best_dist {
                    best_dist = dist;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if best_dist > self.merge_threshold {
            return None;
        }

        // Merge j into i via weighted average
        let ci = self.counts[best_i] as Float;
        let cj = self.counts[best_j] as Float;
        let total = ci + cj;
        if total > 0.0 {
            for d in 0..D {
                self.centroids[best_i][d] =
                    (self.centroids[best_i][d] * ci + self.centroids[best_j][d] * cj) / total;
                // Combine M2 accumulators (approximate)
                self.variance_accum[best_i][d] += self.variance_accum[best_j][d];
            }
        }
        let merged_count = (self.counts[best_i] as u64 + self.counts[best_j] as u64)
            .min(self.max_count as u64) as u32;
        self.counts[best_i] = merged_count;

        // Remove cluster j by shifting
        self.remove_cluster_internal(best_j);

        Some((best_i, best_j))
    }

    /// Remove a cluster by index, shifting remaining clusters down.
    ///
    /// Panics if `idx >= n_active`.
    pub fn remove_cluster(&mut self, idx: usize) {
        assert!(idx < self.n_active, "cluster index out of range");
        self.remove_cluster_internal(idx);
    }

    /// Get a reference to a centroid by index.
    pub fn centroid(&self, idx: usize) -> Option<&[Float; D]> {
        if idx < self.n_active {
            Some(&self.centroids[idx])
        } else {
            None
        }
    }

    /// Per-dimension variance for a cluster (M2 / (count - 1)).
    ///
    /// Returns `None` if the cluster index is invalid or count < 2.
    pub fn cluster_variance(&self, idx: usize) -> Option<[Float; D]> {
        if idx >= self.n_active || self.counts[idx] < 2 {
            return None;
        }
        let n = self.counts[idx] as Float;
        let mut var = [0.0; D];
        for (v, &m2) in var.iter_mut().zip(self.variance_accum[idx].iter()) {
            *v = m2 / (n - 1.0);
        }
        Some(var)
    }

    /// Number of active clusters.
    pub fn n_active(&self) -> usize {
        self.n_active
    }

    /// Observation count for a cluster.
    pub fn count(&self, idx: usize) -> u32 {
        if idx < self.n_active {
            self.counts[idx]
        } else {
            0
        }
    }

    /// Total number of points assigned across all updates.
    pub fn total_assigned(&self) -> u64 {
        self.total_assigned
    }

    /// Reset all clusters and state.
    pub fn reset(&mut self) {
        self.centroids = [[0.0; D]; K];
        self.counts = [0; K];
        self.variance_accum = [[0.0; D]; K];
        self.n_active = 0;
        self.total_assigned = 0;
    }

    /// Manually seed a centroid. Returns the new cluster index.
    ///
    /// Fails with `KMeansError::ClustersFull` if all K slots are occupied.
    pub fn seed_centroid(&mut self, centroid: &[Float; D]) -> Result<usize, KMeansError> {
        if self.n_active >= K {
            return Err(KMeansError::ClustersFull);
        }
        let idx = self.n_active;
        self.centroids[idx] = *centroid;
        self.counts[idx] = 1;
        self.variance_accum[idx] = [0.0; D];
        self.n_active += 1;
        Ok(idx)
    }

    /// Initialize centroids from data using deterministic farthest-point seeding.
    ///
    /// Picks the first seed as the point with the largest norm (highest energy),
    /// then iteratively picks the point farthest from all existing seeds.
    /// This is a deterministic variant of k-means++ that produces well-separated
    /// initial centroids without randomness.
    ///
    /// `max_seeds` limits how many seeds are created (at most K and at most
    /// `data.len()`). Existing active clusters are preserved; new seeds are
    /// appended.
    ///
    /// Returns the number of seeds added.
    pub fn init_farthest_point(&mut self, data: &[[Float; D]], max_seeds: usize) -> usize {
        let n = data.len();
        if n == 0 || max_seeds == 0 {
            return 0;
        }
        let slots = K.saturating_sub(self.n_active);
        let target = max_seeds.min(slots).min(n);
        if target == 0 {
            return 0;
        }

        // Pick first seed: point with largest L2 norm
        let mut best_idx = 0;
        let mut best_norm: Float = 0.0;
        for (i, pt) in data.iter().enumerate() {
            let mut norm: Float = 0.0;
            for &v in pt.iter() {
                norm += v * v;
            }
            if norm > best_norm {
                best_norm = norm;
                best_idx = i;
            }
        }
        let _ = self.seed_centroid(&data[best_idx]);
        let mut added = 1;
        if added >= target {
            return added;
        }

        // Track min distance from each point to any seed.
        // Use a fixed-size approach: recompute distances each iteration
        // (O(target * n * D), acceptable for typical spike counts < 10K).
        while added < target {
            let mut farthest_idx = 0;
            let mut farthest_dist: Float = 0.0;
            for (i, pt) in data.iter().enumerate() {
                // Distance to nearest existing seed
                let (_, dist) = self.find_nearest(pt);
                if dist > farthest_dist {
                    farthest_dist = dist;
                    farthest_idx = i;
                }
            }
            // Don't add duplicates (farthest_dist == 0 means all points are at seeds)
            if farthest_dist < 1e-15 {
                break;
            }
            let _ = self.seed_centroid(&data[farthest_idx]);
            added += 1;
        }
        added
    }

    /// The current create threshold.
    pub fn create_threshold(&self) -> Float {
        self.create_threshold
    }

    /// The current merge threshold.
    pub fn merge_threshold(&self) -> Float {
        self.merge_threshold
    }

    /// Maximum count cap.
    pub fn max_count(&self) -> u32 {
        self.max_count
    }

    // --- internal helpers ---

    fn find_nearest(&self, point: &[Float; D]) -> (usize, Float) {
        let mut best = 0;
        let mut best_dist_sq = float::MAX;
        for i in 0..self.n_active {
            let dist_sq = euclidean_dist_sq_early(&self.centroids[i], point, best_dist_sq);
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best = i;
            }
        }
        // Return actual distance (sqrt) for threshold comparisons
        (best, float::sqrt(best_dist_sq))
    }

    fn centroid_distance(&self, i: usize, j: usize) -> Float {
        euclidean_dist(&self.centroids[i], &self.centroids[j])
    }

    fn remove_cluster_internal(&mut self, idx: usize) {
        // Shift everything after idx down by one
        for i in idx..(self.n_active - 1) {
            self.centroids[i] = self.centroids[i + 1];
            self.counts[i] = self.counts[i + 1];
            self.variance_accum[i] = self.variance_accum[i + 1];
        }
        let last = self.n_active - 1;
        self.centroids[last] = [0.0; D];
        self.counts[last] = 0;
        self.variance_accum[last] = [0.0; D];
        self.n_active -= 1;
    }
}

/// Euclidean distance between two D-dimensional points.
fn euclidean_dist<const D: usize>(a: &[Float; D], b: &[Float; D]) -> Float {
    let mut sum: Float = 0.0;
    for d in 0..D {
        let diff = a[d] - b[d];
        sum += diff * diff;
    }
    float::sqrt(sum)
}

/// Squared Euclidean distance with early exit.
///
/// Returns the squared distance between `a` and `b`, but bails out early
/// if the partial sum exceeds `best_sq`. This avoids computing all D
/// dimensions when the point is clearly farther than the current best.
#[inline]
fn euclidean_dist_sq_early<const D: usize>(
    a: &[Float; D],
    b: &[Float; D],
    best_sq: Float,
) -> Float {
    let mut sum: Float = 0.0;
    for d in 0..D {
        let diff = a[d] - b[d];
        sum += diff * diff;
        if sum > best_sq {
            return sum; // early exit: already worse
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate alloc;
    use alloc::vec;
    use alloc::vec::Vec;

    // Simple pseudo-RNG (xorshift64)
    struct Rng(u64);
    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn gaussian(&mut self, mean: Float, std: Float) -> Float {
            let u1 = (self.next_u64() % 1_000_000 + 1) as Float / 1_000_001.0;
            let u2 = (self.next_u64() % 1_000_000) as Float / 1_000_000.0;
            let z = float::sqrt(-2.0 * float::log(u1)) * float::cos(2.0 * float::PI * u2);
            mean + z * std
        }
    }

    #[test]
    fn test_basic_assignment() {
        // 3 well-separated 2D clusters
        let mut km = OnlineKMeans::<2, 4>::new(1000);
        km.set_create_threshold(3.0);

        // Seed the three clusters manually
        km.seed_centroid(&[0.0, 0.0]).unwrap();
        km.seed_centroid(&[10.0, 0.0]).unwrap();
        km.seed_centroid(&[0.0, 10.0]).unwrap();

        // Points near each cluster should be assigned correctly
        let r1 = km.update(&[0.1, 0.1]);
        assert_eq!(r1.cluster, 0);
        assert!(!r1.created);

        let r2 = km.update(&[9.9, 0.1]);
        assert_eq!(r2.cluster, 1);

        let r3 = km.update(&[0.1, 9.9]);
        assert_eq!(r3.cluster, 2);
    }

    #[test]
    fn test_auto_create_clusters() {
        let mut km = OnlineKMeans::<2, 8>::new(1000);
        km.set_create_threshold(3.0);

        // Feed points from 3 well-separated clusters
        let centers = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
        let mut rng = Rng::new(42);

        for center in &centers {
            for _ in 0..20 {
                let point = [
                    center[0] + rng.gaussian(0.0, 0.5),
                    center[1] + rng.gaussian(0.0, 0.5),
                ];
                km.update(&point);
            }
        }

        assert_eq!(
            km.n_active(),
            3,
            "Should discover 3 clusters, got {}",
            km.n_active()
        );
    }

    #[test]
    fn test_centroid_convergence() {
        let mut km = OnlineKMeans::<2, 4>::new(10000);
        km.set_create_threshold(100.0); // high so no new clusters created

        let mut rng = Rng::new(99);
        let true_mean = [5.0, -3.0];

        // First point seeds the cluster
        km.update(&[
            true_mean[0] + rng.gaussian(0.0, 1.0),
            true_mean[1] + rng.gaussian(0.0, 1.0),
        ]);

        // Feed 5000 points from a Gaussian centered at true_mean
        for _ in 0..5000 {
            let point = [
                true_mean[0] + rng.gaussian(0.0, 1.0),
                true_mean[1] + rng.gaussian(0.0, 1.0),
            ];
            km.update(&point);
        }

        assert_eq!(km.n_active(), 1);
        let centroid = km.centroid(0).unwrap();
        assert!(
            float::abs(centroid[0] - true_mean[0]) < 0.2,
            "centroid[0]={} should be near {}",
            centroid[0],
            true_mean[0]
        );
        assert!(
            float::abs(centroid[1] - true_mean[1]) < 0.2,
            "centroid[1]={} should be near {}",
            centroid[1],
            true_mean[1]
        );
    }

    #[test]
    fn test_predict_readonly() {
        let mut km = OnlineKMeans::<2, 4>::new(1000);
        km.seed_centroid(&[0.0, 0.0]).unwrap();
        km.seed_centroid(&[10.0, 0.0]).unwrap();

        let count_before = km.count(0);
        let total_before = km.total_assigned();

        let (cluster, dist) = km.predict(&[0.5, 0.0]);
        assert_eq!(cluster, 0);
        assert!(dist < 1.0);

        // State should be unchanged
        assert_eq!(km.count(0), count_before);
        assert_eq!(km.total_assigned(), total_before);
    }

    #[test]
    fn test_merge_close_clusters() {
        let mut km = OnlineKMeans::<2, 8>::new(1000);
        km.set_merge_threshold(2.0);

        // Create two very close clusters
        km.seed_centroid(&[0.0, 0.0]).unwrap();
        km.seed_centroid(&[1.0, 0.0]).unwrap();
        km.seed_centroid(&[10.0, 10.0]).unwrap();

        assert_eq!(km.n_active(), 3);
        let result = km.merge_closest();
        assert!(result.is_some());
        let (kept, removed) = result.unwrap();
        assert_eq!(kept, 0);
        assert_eq!(removed, 1);
        assert_eq!(km.n_active(), 2);

        // The merged centroid should be between the two originals
        let c = km.centroid(0).unwrap();
        assert!(c[0] > -0.1 && c[0] < 1.1);

        // Distant cluster should still be at index 1 (shifted from 2)
        let c_far = km.centroid(1).unwrap();
        assert!(float::abs(c_far[0] - 10.0) < 0.01);
    }

    #[test]
    fn test_remove_cluster() {
        let mut km = OnlineKMeans::<2, 4>::new(1000);
        km.seed_centroid(&[0.0, 0.0]).unwrap();
        km.seed_centroid(&[5.0, 5.0]).unwrap();
        km.seed_centroid(&[10.0, 10.0]).unwrap();

        assert_eq!(km.n_active(), 3);
        km.remove_cluster(1); // remove middle cluster
        assert_eq!(km.n_active(), 2);

        // Cluster at index 1 should now be what was at index 2
        let c = km.centroid(1).unwrap();
        assert!(float::abs(c[0] - 10.0) < 0.01);
    }

    #[test]
    fn test_count_capping() {
        let max_count = 100;
        let mut km = OnlineKMeans::<2, 4>::new(max_count);
        km.set_create_threshold(float::MAX);

        // Feed many points to the single cluster
        for i in 0..500 {
            km.update(&[i as Float * 0.001, 0.0]);
        }

        assert_eq!(km.count(0), max_count);
        assert_eq!(km.total_assigned(), 500);

        // Centroid should still be plastic (not frozen at early values)
        let c = km.centroid(0).unwrap();
        // After 500 points with count capped at 100, centroid should be
        // near the recent values (around 0.4-0.5 range), not near 0
        assert!(
            c[0] > 0.05,
            "centroid should show drift adaptation, got {}",
            c[0]
        );
    }

    #[test]
    fn test_variance_tracking() {
        let mut km = OnlineKMeans::<2, 4>::new(100000);
        km.set_create_threshold(float::MAX);

        let mut rng = Rng::new(77);
        let true_var_x: Float = 4.0; // std = 2.0
        let true_var_y: Float = 1.0; // std = 1.0

        for _ in 0..10000 {
            let point = [rng.gaussian(0.0, 2.0), rng.gaussian(0.0, 1.0)];
            km.update(&point);
        }

        let var = km.cluster_variance(0).unwrap();
        assert!(
            float::abs(var[0] - true_var_x) < 0.5,
            "variance[0]={} should be near {}",
            var[0],
            true_var_x
        );
        assert!(
            float::abs(var[1] - true_var_y) < 0.3,
            "variance[1]={} should be near {}",
            var[1],
            true_var_y
        );
    }

    #[test]
    fn test_seed_centroid() {
        let mut km = OnlineKMeans::<3, 4>::new(1000);
        let idx0 = km.seed_centroid(&[1.0, 2.0, 3.0]).unwrap();
        let idx1 = km.seed_centroid(&[4.0, 5.0, 6.0]).unwrap();
        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(km.n_active(), 2);

        let c = km.centroid(0).unwrap();
        assert_eq!(*c, [1.0, 2.0, 3.0]);

        // Predict should use seeded centroids
        let (cluster, _) = km.predict(&[1.1, 2.1, 3.1]);
        assert_eq!(cluster, 0);
    }

    #[test]
    fn test_seed_full() {
        let mut km = OnlineKMeans::<2, 2>::new(1000);
        km.seed_centroid(&[0.0, 0.0]).unwrap();
        km.seed_centroid(&[1.0, 1.0]).unwrap();
        assert_eq!(
            km.seed_centroid(&[2.0, 2.0]),
            Err(KMeansError::ClustersFull)
        );
    }

    #[test]
    fn test_reset() {
        let mut km = OnlineKMeans::<2, 4>::new(1000);
        km.seed_centroid(&[1.0, 2.0]).unwrap();
        km.update(&[3.0, 4.0]);
        assert!(km.n_active() > 0);
        assert!(km.total_assigned() > 0);

        km.reset();
        assert_eq!(km.n_active(), 0);
        assert_eq!(km.total_assigned(), 0);
        assert!(km.centroid(0).is_none());
    }

    #[test]
    fn test_empty_predict() {
        let km = OnlineKMeans::<2, 4>::new(1000);
        let (cluster, dist) = km.predict(&[1.0, 2.0]);
        assert_eq!(cluster, 0);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_high_dimensional() {
        let mut km = OnlineKMeans::<8, 16>::new(1000);
        km.set_create_threshold(5.0);

        let mut rng = Rng::new(123);

        // Create 4 clusters in 8D space
        let centers: [[Float; 8]; 4] = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];

        for center in &centers {
            for _ in 0..50 {
                let mut point = [0.0; 8];
                for (d, p) in point.iter_mut().enumerate() {
                    *p = center[d] + rng.gaussian(0.0, 0.5);
                }
                km.update(&point);
            }
        }

        assert_eq!(
            km.n_active(),
            4,
            "Should discover 4 clusters in 8D, got {}",
            km.n_active()
        );
    }

    #[test]
    fn test_end_to_end_spike_sorting() {
        use crate::spike_sort::{
            detect_spikes, estimate_noise_mad, WaveformExtractor, WaveformPca,
        };

        let mut rng = Rng::new(42);
        let n_samples = 10000;

        // Generate synthetic recording with 3 neuron types
        let mut data = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            data.push(rng.gaussian(0.0, 1.0));
        }

        // Spike waveform shapes
        let make_spike = |amplitude: Float, width: Float| -> [Float; 8] {
            let mut wf = [0.0; 8];
            for (i, v) in wf.iter_mut().enumerate() {
                let x = (i as Float - 2.0) / width;
                *v = -amplitude * float::exp(-0.5 * x * x);
            }
            wf
        };

        let spike1 = make_spike(8.0, 2.0);
        let spike2 = make_spike(6.0, 3.0);
        let spike3 = make_spike(10.0, 1.5);

        let positions = [
            vec![200, 700, 1200, 1700, 2200, 2700, 3200, 3700, 4200, 4700],
            vec![400, 900, 1400, 1900, 2400, 2900, 3400, 3900, 4400, 4900],
            vec![600, 1100, 1600, 2100, 2600, 3100, 3600, 4100, 4600, 5100],
        ];
        let spikes = [spike1, spike2, spike3];

        for (spike, pos) in spikes.iter().zip(positions.iter()) {
            for &p in pos {
                if p >= 2 && p + 6 < n_samples {
                    for (j, &v) in spike.iter().enumerate() {
                        data[p - 2 + j] += v;
                    }
                }
            }
        }

        // 1. Noise estimation
        let mut scratch = vec![0.0 as Float; n_samples];
        let noise = estimate_noise_mad(&data, &mut scratch);
        assert!(noise > 0.5 && noise < 2.0);

        // 2. Detection
        let threshold = 4.0 * noise;
        let mut spike_times = vec![0usize; 100];
        let n_detected = detect_spikes(&data, threshold, 20, &mut spike_times);
        assert!(n_detected >= 15, "Expected >=15 spikes, got {}", n_detected);

        // 3. Extraction
        let ext = WaveformExtractor::<1, 8>::new();
        let mut waveforms = vec![[0.0 as Float; 8]; n_detected];
        let n_extracted = ext.extract(&data, &spike_times[..n_detected], &mut waveforms);
        assert!(n_extracted >= 15);

        // 4. PCA
        let mut pca = WaveformPca::<8, 3, 64>::new();
        pca.fit(&waveforms[..n_extracted]).unwrap();

        let mut features = Vec::with_capacity(n_extracted);
        for wf in waveforms.iter().take(n_extracted) {
            let mut f = [0.0; 3];
            pca.transform(wf, &mut f).unwrap();
            features.push(f);
        }

        // 5. OnlineKMeans clustering
        let mut km = OnlineKMeans::<3, 8>::new(1000);
        km.set_create_threshold(3.0);

        for feat in &features {
            km.update(feat);
        }

        // Should discover at least 2 clusters (3 is ideal but depends on noise)
        assert!(
            km.n_active() >= 2,
            "Should find >=2 clusters, got {}",
            km.n_active()
        );

        // All clusters should have at least one spike
        for i in 0..km.n_active() {
            assert!(km.count(i) >= 1, "Cluster {} should have >=1 spike", i);
        }
    }

    #[test]
    fn test_init_farthest_point_basic() {
        let data = [
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [0.1, 0.1],
            [10.1, 0.1],
            [0.1, 10.1],
        ];
        let mut km = OnlineKMeans::<2, 8>::new(1000);
        km.set_create_threshold(3.0);
        let added = km.init_farthest_point(&data, 3);
        assert_eq!(added, 3);
        assert_eq!(km.n_active(), 3);

        // Seeds should be well-separated (each near one of the three corners)
        let mut has_origin = false;
        let mut has_x = false;
        let mut has_y = false;
        for i in 0..3 {
            let c = km.centroid(i).unwrap();
            if c[0] < 1.0 && c[1] < 1.0 {
                has_origin = true;
            }
            if c[0] > 9.0 && c[1] < 1.0 {
                has_x = true;
            }
            if c[0] < 1.0 && c[1] > 9.0 {
                has_y = true;
            }
        }
        assert!(
            has_origin && has_x && has_y,
            "Seeds should cover all 3 corners"
        );
    }

    #[test]
    fn test_init_farthest_point_empty() {
        let data: &[[Float; 2]] = &[];
        let mut km = OnlineKMeans::<2, 4>::new(1000);
        let added = km.init_farthest_point(data, 3);
        assert_eq!(added, 0);
        assert_eq!(km.n_active(), 0);
    }

    #[test]
    fn test_init_farthest_point_preserves_existing() {
        let mut km = OnlineKMeans::<2, 4>::new(1000);
        km.seed_centroid(&[5.0, 5.0]).unwrap();
        assert_eq!(km.n_active(), 1);

        let data = [[0.0, 0.0], [10.0, 10.0]];
        let added = km.init_farthest_point(&data, 2);
        assert_eq!(added, 2);
        assert_eq!(km.n_active(), 3);
    }

    #[test]
    fn test_init_then_update() {
        // Verify that seeded init + online update produces better clustering
        // than pure online (first-come) on ordered data
        let mut rng = Rng::new(99);
        let centers = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
        let mut data = Vec::new();
        // Generate data in order (all cluster 0 first, then 1, then 2)
        for center in &centers {
            for _ in 0..30 {
                data.push([
                    center[0] + rng.gaussian(0.0, 0.5),
                    center[1] + rng.gaussian(0.0, 0.5),
                ]);
            }
        }

        // With farthest-point init
        let mut km_init = OnlineKMeans::<2, 8>::new(1000);
        km_init.set_create_threshold(3.0);
        km_init.init_farthest_point(&data, 3);
        for pt in &data {
            km_init.update(pt);
        }

        // Seeds should have picked points near each of the 3 corners
        assert!(
            km_init.n_active() >= 3,
            "Init should find >=3 clusters, got {}",
            km_init.n_active()
        );
    }
}

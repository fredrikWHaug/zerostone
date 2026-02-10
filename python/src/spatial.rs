//! Python bindings for spatial filter primitives.

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zerostone::{
    ChannelRouter as ZsChannelRouter,
    CommonAverageReference as ZsCommonAverageReference,
    SurfaceLaplacian as ZsSurfaceLaplacian,
};

// ============================================================================
// CommonAverageReference (CAR)
// ============================================================================

/// Internal enum for handling different channel counts.
enum CarInner {
    Ch4(ZsCommonAverageReference<4>),
    Ch8(ZsCommonAverageReference<8>),
    Ch16(ZsCommonAverageReference<16>),
    Ch32(ZsCommonAverageReference<32>),
    Ch64(ZsCommonAverageReference<64>),
    /// Dynamic implementation for non-standard channel counts
    Dynamic { channels: usize },
}

/// Common Average Reference (CAR) spatial filter.
///
/// Subtracts the mean of all channels from each channel, removing common-mode
/// noise and making recordings reference-independent. This is widely used in
/// EEG, ECoG, and other multi-channel neural recordings.
///
/// # Example
/// ```python
/// import npyci as npy
/// import numpy as np
///
/// # Create CAR filter for 8 channels
/// car = npy.CAR(channels=8)
///
/// # Process multi-channel data (samples x channels)
/// data = np.random.randn(1000, 8).astype(np.float32)
/// filtered = car.process(data)
///
/// # Output has zero mean across channels for each sample
/// assert np.allclose(filtered.mean(axis=1), 0, atol=1e-6)
/// ```
#[pyclass]
pub struct CAR {
    inner: CarInner,
    channels: usize,
}

#[pymethods]
impl CAR {
    /// Create a new Common Average Reference filter.
    ///
    /// Args:
    ///     channels (int): Number of channels. Common values (4, 8, 16, 32, 64)
    ///         use optimized implementations; other values use dynamic fallback.
    ///
    /// Returns:
    ///     CAR: A new CAR filter instance.
    ///
    /// Example:
    ///     >>> car = CAR(channels=8)
    #[new]
    fn new(channels: usize) -> PyResult<Self> {
        if channels == 0 {
            return Err(PyValueError::new_err("channels must be at least 1"));
        }

        let inner = match channels {
            4 => CarInner::Ch4(ZsCommonAverageReference::new()),
            8 => CarInner::Ch8(ZsCommonAverageReference::new()),
            16 => CarInner::Ch16(ZsCommonAverageReference::new()),
            32 => CarInner::Ch32(ZsCommonAverageReference::new()),
            64 => CarInner::Ch64(ZsCommonAverageReference::new()),
            _ => CarInner::Dynamic { channels },
        };

        Ok(Self { inner, channels })
    }

    /// Process multi-channel data through the CAR filter.
    ///
    /// Args:
    ///     input (np.ndarray): Input data as 2D float32 array with shape (samples, channels).
    ///
    /// Returns:
    ///     np.ndarray: Filtered data as 2D float32 array with same shape.
    ///
    /// Raises:
    ///     ValueError: If input is not 2D or channel count doesn't match.
    ///
    /// Example:
    ///     >>> data = np.random.randn(1000, 8).astype(np.float32)
    ///     >>> filtered = car.process(data)
    fn process<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = input.shape();
        let (n_samples, n_channels) = (shape[0], shape[1]);

        if n_channels != self.channels {
            return Err(PyValueError::new_err(format!(
                "Channel count mismatch: CAR configured for {} channels, got {}",
                self.channels, n_channels
            )));
        }

        let input_array = input.as_array();
        let mut output = vec![0.0f32; n_samples * n_channels];

        match &self.inner {
            CarInner::Ch4(car) => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let samples: [f32; 4] = [row[0], row[1], row[2], row[3]];
                    let filtered = car.process(&samples);
                    for (j, &val) in filtered.iter().enumerate() {
                        output[i * n_channels + j] = val;
                    }
                }
            }
            CarInner::Ch8(car) => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let mut samples = [0.0f32; 8];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let filtered = car.process(&samples);
                    for (j, &val) in filtered.iter().enumerate() {
                        output[i * n_channels + j] = val;
                    }
                }
            }
            CarInner::Ch16(car) => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let mut samples = [0.0f32; 16];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let filtered = car.process(&samples);
                    for (j, &val) in filtered.iter().enumerate() {
                        output[i * n_channels + j] = val;
                    }
                }
            }
            CarInner::Ch32(car) => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let mut samples = [0.0f32; 32];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let filtered = car.process(&samples);
                    for (j, &val) in filtered.iter().enumerate() {
                        output[i * n_channels + j] = val;
                    }
                }
            }
            CarInner::Ch64(car) => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let mut samples = [0.0f32; 64];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let filtered = car.process(&samples);
                    for (j, &val) in filtered.iter().enumerate() {
                        output[i * n_channels + j] = val;
                    }
                }
            }
            CarInner::Dynamic { channels } => {
                // Dynamic implementation: compute mean and subtract
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let mean: f32 = row.iter().sum::<f32>() / *channels as f32;
                    for (j, &val) in row.iter().enumerate() {
                        output[i * n_channels + j] = val - mean;
                    }
                }
            }
        }

        // Create 2D output array
        let output_array = Array2::from_shape_vec((n_samples, n_channels), output)
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape output: {}", e)))?;
        Ok(PyArray2::from_owned_array(py, output_array))
    }

    /// Reset the filter state.
    ///
    /// CAR is stateless, so this is a no-op. Provided for API consistency.
    fn reset(&self) {
        // CAR is stateless, nothing to reset
    }

    /// Get the number of channels.
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    fn __repr__(&self) -> String {
        format!("CAR(channels={})", self.channels)
    }
}

// ============================================================================
// SurfaceLaplacian
// ============================================================================

/// Internal enum for handling different channel/neighbor configurations.
enum LaplacianInner {
    Ch4N2(ZsSurfaceLaplacian<4, 2>),
    Ch8N2(ZsSurfaceLaplacian<8, 2>),
    Ch16N2(ZsSurfaceLaplacian<16, 2>),
    Ch32N2(ZsSurfaceLaplacian<32, 2>),
    Ch64N2(ZsSurfaceLaplacian<64, 2>),
    /// Dynamic implementation for non-standard configurations
    Dynamic {
        #[allow(dead_code)]
        channels: usize,
        neighbors: Vec<Vec<usize>>,
    },
}

/// Surface Laplacian (Hjorth) spatial filter.
///
/// Computes the second spatial derivative to reduce volume conduction effects
/// and improve spatial resolution. Particularly useful for motor imagery BCI.
///
/// # Example
/// ```python
/// import npyci as npy
/// import numpy as np
///
/// # Create linear Laplacian for 8-channel electrode strip
/// lap = npy.SurfaceLaplacian.linear(channels=8)
///
/// # Process multi-channel data (samples x channels)
/// data = np.random.randn(1000, 8).astype(np.float32)
/// filtered = lap.process(data)
/// ```
#[pyclass]
pub struct SurfaceLaplacian {
    inner: LaplacianInner,
    channels: usize,
}

#[pymethods]
impl SurfaceLaplacian {
    /// Create a Surface Laplacian filter for a linear electrode array.
    ///
    /// In a linear array, each electrode has at most 2 neighbors (left and right),
    /// with edge electrodes having only 1 neighbor.
    ///
    /// Args:
    ///     channels (int): Number of channels in the linear array.
    ///
    /// Returns:
    ///     SurfaceLaplacian: A new Surface Laplacian filter.
    ///
    /// Example:
    ///     >>> lap = SurfaceLaplacian.linear(channels=8)
    #[staticmethod]
    fn linear(channels: usize) -> PyResult<Self> {
        if channels < 2 {
            return Err(PyValueError::new_err(
                "Linear Laplacian requires at least 2 channels",
            ));
        }

        // Build linear neighbor configuration
        let inner = match channels {
            4 => {
                let neighbors = build_linear_neighbors_4();
                LaplacianInner::Ch4N2(ZsSurfaceLaplacian::unweighted(neighbors))
            }
            8 => {
                let neighbors = build_linear_neighbors_8();
                LaplacianInner::Ch8N2(ZsSurfaceLaplacian::unweighted(neighbors))
            }
            16 => {
                let neighbors = build_linear_neighbors_16();
                LaplacianInner::Ch16N2(ZsSurfaceLaplacian::unweighted(neighbors))
            }
            32 => {
                let neighbors = build_linear_neighbors_32();
                LaplacianInner::Ch32N2(ZsSurfaceLaplacian::unweighted(neighbors))
            }
            64 => {
                let neighbors = build_linear_neighbors_64();
                LaplacianInner::Ch64N2(ZsSurfaceLaplacian::unweighted(neighbors))
            }
            _ => {
                // Build dynamic neighbor list
                let neighbors = build_linear_neighbors_dynamic(channels);
                LaplacianInner::Dynamic {
                    channels,
                    neighbors,
                }
            }
        };

        Ok(Self { inner, channels })
    }

    /// Create a Surface Laplacian filter with custom neighbor configuration.
    ///
    /// Args:
    ///     channels (int): Number of channels.
    ///     neighbors (list[list[int]]): List of neighbor indices for each channel.
    ///         Each inner list contains the indices of neighboring channels.
    ///
    /// Returns:
    ///     SurfaceLaplacian: A new Surface Laplacian filter.
    ///
    /// Raises:
    ///     ValueError: If neighbor indices are invalid.
    ///
    /// Example:
    ///     >>> # Custom 4-channel configuration
    ///     >>> neighbors = [[1], [0, 2], [1, 3], [2]]
    ///     >>> lap = SurfaceLaplacian.custom(channels=4, neighbors=neighbors)
    #[staticmethod]
    fn custom(channels: usize, neighbors: Vec<Vec<usize>>) -> PyResult<Self> {
        if channels == 0 {
            return Err(PyValueError::new_err("channels must be at least 1"));
        }

        if neighbors.len() != channels {
            return Err(PyValueError::new_err(format!(
                "neighbors list length ({}) must match channels ({})",
                neighbors.len(),
                channels
            )));
        }

        // Validate neighbor indices
        for (ch, ch_neighbors) in neighbors.iter().enumerate() {
            for &neighbor_idx in ch_neighbors {
                if neighbor_idx >= channels {
                    return Err(PyValueError::new_err(format!(
                        "Invalid neighbor index {} for {} channels",
                        neighbor_idx, channels
                    )));
                }
                if neighbor_idx == ch {
                    return Err(PyValueError::new_err(format!(
                        "Channel {} cannot be its own neighbor",
                        ch
                    )));
                }
            }
        }

        let inner = LaplacianInner::Dynamic {
            channels,
            neighbors,
        };

        Ok(Self { inner, channels })
    }

    /// Process multi-channel data through the Surface Laplacian filter.
    ///
    /// Args:
    ///     input (np.ndarray): Input data as 2D float32 array with shape (samples, channels).
    ///
    /// Returns:
    ///     np.ndarray: Filtered data as 2D float32 array with same shape.
    ///
    /// Raises:
    ///     ValueError: If input is not 2D or channel count doesn't match.
    ///
    /// Example:
    ///     >>> data = np.random.randn(1000, 8).astype(np.float32)
    ///     >>> filtered = lap.process(data)
    fn process<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = input.shape();
        let (n_samples, n_channels) = (shape[0], shape[1]);

        if n_channels != self.channels {
            return Err(PyValueError::new_err(format!(
                "Channel count mismatch: SurfaceLaplacian configured for {} channels, got {}",
                self.channels, n_channels
            )));
        }

        let input_array = input.as_array();
        let mut output = vec![0.0f32; n_samples * n_channels];

        match &self.inner {
            LaplacianInner::Ch4N2(lap) => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let samples: [f32; 4] = [row[0], row[1], row[2], row[3]];
                    let filtered = lap.process(&samples);
                    for (j, &val) in filtered.iter().enumerate() {
                        output[i * n_channels + j] = val;
                    }
                }
            }
            LaplacianInner::Ch8N2(lap) => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let mut samples = [0.0f32; 8];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let filtered = lap.process(&samples);
                    for (j, &val) in filtered.iter().enumerate() {
                        output[i * n_channels + j] = val;
                    }
                }
            }
            LaplacianInner::Ch16N2(lap) => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let mut samples = [0.0f32; 16];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let filtered = lap.process(&samples);
                    for (j, &val) in filtered.iter().enumerate() {
                        output[i * n_channels + j] = val;
                    }
                }
            }
            LaplacianInner::Ch32N2(lap) => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let mut samples = [0.0f32; 32];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let filtered = lap.process(&samples);
                    for (j, &val) in filtered.iter().enumerate() {
                        output[i * n_channels + j] = val;
                    }
                }
            }
            LaplacianInner::Ch64N2(lap) => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let mut samples = [0.0f32; 64];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let filtered = lap.process(&samples);
                    for (j, &val) in filtered.iter().enumerate() {
                        output[i * n_channels + j] = val;
                    }
                }
            }
            LaplacianInner::Dynamic { neighbors, .. } => {
                // Dynamic implementation
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    for (ch, ch_neighbors) in neighbors.iter().enumerate() {
                        if ch_neighbors.is_empty() {
                            // No neighbors: output = input
                            output[i * n_channels + ch] = row[ch];
                        } else {
                            // Compute weighted average of neighbors
                            let neighbor_mean: f32 = ch_neighbors
                                .iter()
                                .map(|&n| row[n])
                                .sum::<f32>()
                                / ch_neighbors.len() as f32;
                            output[i * n_channels + ch] = row[ch] - neighbor_mean;
                        }
                    }
                }
            }
        }

        // Create 2D output array
        let output_array = Array2::from_shape_vec((n_samples, n_channels), output)
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape output: {}", e)))?;
        Ok(PyArray2::from_owned_array(py, output_array))
    }

    /// Reset the filter state.
    ///
    /// SurfaceLaplacian is stateless, so this is a no-op. Provided for API consistency.
    fn reset(&self) {
        // SurfaceLaplacian is stateless, nothing to reset
    }

    /// Get the number of channels.
    #[getter]
    fn channels(&self) -> usize {
        self.channels
    }

    fn __repr__(&self) -> String {
        format!("SurfaceLaplacian(channels={})", self.channels)
    }
}

// ============================================================================
// Helper functions for building linear neighbor configurations
// ============================================================================

const INVALID: u16 = u16::MAX;

fn build_linear_neighbors_4() -> [[u16; 2]; 4] {
    [
        [1, INVALID],     // Channel 0: only right neighbor
        [0, 2],           // Channel 1: neighbors 0, 2
        [1, 3],           // Channel 2: neighbors 1, 3
        [2, INVALID],     // Channel 3: only left neighbor
    ]
}

fn build_linear_neighbors_8() -> [[u16; 2]; 8] {
    [
        [1, INVALID],     // Channel 0
        [0, 2],           // Channel 1
        [1, 3],           // Channel 2
        [2, 4],           // Channel 3
        [3, 5],           // Channel 4
        [4, 6],           // Channel 5
        [5, 7],           // Channel 6
        [6, INVALID],     // Channel 7
    ]
}

fn build_linear_neighbors_16() -> [[u16; 2]; 16] {
    let mut neighbors = [[INVALID; 2]; 16];
    for i in 0..16 {
        if i > 0 {
            neighbors[i][0] = (i - 1) as u16;
        }
        if i < 15 {
            neighbors[i][1] = (i + 1) as u16;
        }
    }
    // Fix first channel
    neighbors[0] = [1, INVALID];
    neighbors
}

fn build_linear_neighbors_32() -> [[u16; 2]; 32] {
    let mut neighbors = [[INVALID; 2]; 32];
    for i in 0..32 {
        if i > 0 {
            neighbors[i][0] = (i - 1) as u16;
        }
        if i < 31 {
            neighbors[i][1] = (i + 1) as u16;
        }
    }
    neighbors[0] = [1, INVALID];
    neighbors
}

fn build_linear_neighbors_64() -> [[u16; 2]; 64] {
    let mut neighbors = [[INVALID; 2]; 64];
    for i in 0..64 {
        if i > 0 {
            neighbors[i][0] = (i - 1) as u16;
        }
        if i < 63 {
            neighbors[i][1] = (i + 1) as u16;
        }
    }
    neighbors[0] = [1, INVALID];
    neighbors
}

fn build_linear_neighbors_dynamic(channels: usize) -> Vec<Vec<usize>> {
    let mut neighbors = Vec::with_capacity(channels);
    for i in 0..channels {
        let mut ch_neighbors = Vec::new();
        if i > 0 {
            ch_neighbors.push(i - 1);
        }
        if i < channels - 1 {
            ch_neighbors.push(i + 1);
        }
        neighbors.push(ch_neighbors);
    }
    neighbors
}

// ============================================================================
// ChannelRouter
// ============================================================================

/// Internal enum for handling different channel configurations.
enum RouterInner {
    R4to4(ZsChannelRouter<4, 4>),
    R8to8(ZsChannelRouter<8, 8>),
    R8to4(ZsChannelRouter<8, 4>),
    R16to16(ZsChannelRouter<16, 16>),
    R16to8(ZsChannelRouter<16, 8>),
    R32to32(ZsChannelRouter<32, 32>),
    R64to64(ZsChannelRouter<64, 64>),
    /// Dynamic implementation for non-standard configurations
    Dynamic {
        #[allow(dead_code)]
        in_channels: usize,
        #[allow(dead_code)]
        out_channels: usize,
        indices: Vec<usize>,
    },
}

/// Channel router for selecting, permuting, or duplicating channels.
///
/// Routes channels between pipeline stages by mapping output channels to input channels.
/// Useful for:
/// - Selecting a subset of channels (e.g., frontal EEG electrodes)
/// - Reordering channels (e.g., group spatially-adjacent channels)
/// - Creating identity mappings for pipeline compatibility
///
/// # Example
/// ```python
/// import npyci as npy
/// import numpy as np
///
/// # Select channels 0, 2, 4 from 8-channel input
/// router = npy.ChannelRouter.select(in_channels=8, indices=[0, 2, 4])
/// data = np.random.randn(100, 8).astype(np.float32)
/// selected = router.process(data)  # Shape: (100, 3)
///
/// # Permute channel order
/// router = npy.ChannelRouter.permute(channels=4, indices=[3, 2, 1, 0])
/// reversed_data = router.process(data[:, :4])  # Reverse channel order
///
/// # Identity (pass-through)
/// router = npy.ChannelRouter.identity(channels=8)
/// same_data = router.process(data)  # No change
/// ```
#[pyclass]
pub struct ChannelRouter {
    inner: RouterInner,
    in_channels: usize,
    out_channels: usize,
}

#[pymethods]
impl ChannelRouter {
    /// Create a channel router that selects a subset of input channels.
    ///
    /// Args:
    ///     in_channels (int): Number of input channels.
    ///     indices (list[int]): Indices of input channels to select.
    ///         The length of this list determines the number of output channels.
    ///
    /// Returns:
    ///     ChannelRouter: A new channel router.
    ///
    /// Raises:
    ///     ValueError: If any index is >= in_channels.
    ///
    /// Example:
    ///     >>> router = ChannelRouter.select(in_channels=8, indices=[0, 2, 4])
    ///     >>> data = np.random.randn(100, 8).astype(np.float32)
    ///     >>> selected = router.process(data)  # Shape: (100, 3)
    #[staticmethod]
    fn select(in_channels: usize, indices: Vec<usize>) -> PyResult<Self> {
        let out_channels = indices.len();

        if out_channels == 0 {
            return Err(PyValueError::new_err("indices must not be empty"));
        }

        // Validate indices
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= in_channels {
                return Err(PyValueError::new_err(format!(
                    "Index {} out of bounds: indices[{}] = {} (must be < {})",
                    i, i, idx, in_channels
                )));
            }
        }

        let inner = Self::create_inner(in_channels, out_channels, &indices);

        Ok(Self {
            inner,
            in_channels,
            out_channels,
        })
    }

    /// Create a channel router that permutes (reorders) channels.
    ///
    /// Args:
    ///     channels (int): Number of channels (input and output are the same).
    ///     indices (list[int]): Permutation indices. Must have length == channels.
    ///         indices[i] = j means output channel i comes from input channel j.
    ///
    /// Returns:
    ///     ChannelRouter: A new channel router.
    ///
    /// Raises:
    ///     ValueError: If indices length != channels or any index is invalid.
    ///
    /// Example:
    ///     >>> # Reverse channel order
    ///     >>> router = ChannelRouter.permute(channels=4, indices=[3, 2, 1, 0])
    #[staticmethod]
    fn permute(channels: usize, indices: Vec<usize>) -> PyResult<Self> {
        if indices.len() != channels {
            return Err(PyValueError::new_err(format!(
                "indices length ({}) must match channels ({})",
                indices.len(),
                channels
            )));
        }

        // Validate indices
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= channels {
                return Err(PyValueError::new_err(format!(
                    "Index {} out of bounds: indices[{}] = {} (must be < {})",
                    i, i, idx, channels
                )));
            }
        }

        let inner = Self::create_inner(channels, channels, &indices);

        Ok(Self {
            inner,
            in_channels: channels,
            out_channels: channels,
        })
    }

    /// Create an identity channel router (pass-through).
    ///
    /// Args:
    ///     channels (int): Number of channels.
    ///
    /// Returns:
    ///     ChannelRouter: A new identity router.
    ///
    /// Example:
    ///     >>> router = ChannelRouter.identity(channels=8)
    #[staticmethod]
    fn identity(channels: usize) -> PyResult<Self> {
        if channels == 0 {
            return Err(PyValueError::new_err("channels must be at least 1"));
        }

        let indices: Vec<usize> = (0..channels).collect();
        let inner = Self::create_inner(channels, channels, &indices);

        Ok(Self {
            inner,
            in_channels: channels,
            out_channels: channels,
        })
    }

    /// Process multi-channel data through the channel router.
    ///
    /// Args:
    ///     input (np.ndarray): Input data as 2D float32 array with shape (samples, in_channels).
    ///
    /// Returns:
    ///     np.ndarray: Routed data as 2D float32 array with shape (samples, out_channels).
    ///
    /// Raises:
    ///     ValueError: If input channel count doesn't match.
    ///
    /// Example:
    ///     >>> data = np.random.randn(100, 8).astype(np.float32)
    ///     >>> routed = router.process(data)
    fn process<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let shape = input.shape();
        let (n_samples, n_channels) = (shape[0], shape[1]);

        if n_channels != self.in_channels {
            return Err(PyValueError::new_err(format!(
                "Channel count mismatch: ChannelRouter configured for {} input channels, got {}",
                self.in_channels, n_channels
            )));
        }

        let input_array = input.as_array();
        let mut output = vec![0.0f32; n_samples * self.out_channels];

        match &self.inner {
            RouterInner::R4to4(router) => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let samples: [f32; 4] = [row[0], row[1], row[2], row[3]];
                    let routed = router.process(&samples);
                    for (j, &val) in routed.iter().enumerate() {
                        output[i * self.out_channels + j] = val;
                    }
                }
            }
            RouterInner::R8to8(router) => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let mut samples = [0.0f32; 8];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let routed = router.process(&samples);
                    for (j, &val) in routed.iter().enumerate() {
                        output[i * self.out_channels + j] = val;
                    }
                }
            }
            RouterInner::R8to4(router) => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let mut samples = [0.0f32; 8];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let routed = router.process(&samples);
                    for (j, &val) in routed.iter().enumerate() {
                        output[i * self.out_channels + j] = val;
                    }
                }
            }
            RouterInner::R16to16(router) => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let mut samples = [0.0f32; 16];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let routed = router.process(&samples);
                    for (j, &val) in routed.iter().enumerate() {
                        output[i * self.out_channels + j] = val;
                    }
                }
            }
            RouterInner::R16to8(router) => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let mut samples = [0.0f32; 16];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let routed = router.process(&samples);
                    for (j, &val) in routed.iter().enumerate() {
                        output[i * self.out_channels + j] = val;
                    }
                }
            }
            RouterInner::R32to32(router) => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let mut samples = [0.0f32; 32];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let routed = router.process(&samples);
                    for (j, &val) in routed.iter().enumerate() {
                        output[i * self.out_channels + j] = val;
                    }
                }
            }
            RouterInner::R64to64(router) => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    let mut samples = [0.0f32; 64];
                    for (j, val) in row.iter().enumerate() {
                        samples[j] = *val;
                    }
                    let routed = router.process(&samples);
                    for (j, &val) in routed.iter().enumerate() {
                        output[i * self.out_channels + j] = val;
                    }
                }
            }
            RouterInner::Dynamic { indices, .. } => {
                for (i, row) in input_array.rows().into_iter().enumerate() {
                    for (j, &idx) in indices.iter().enumerate() {
                        output[i * self.out_channels + j] = row[idx];
                    }
                }
            }
        }

        let output_array = Array2::from_shape_vec((n_samples, self.out_channels), output)
            .map_err(|e| PyValueError::new_err(format!("Failed to reshape output: {}", e)))?;
        Ok(PyArray2::from_owned_array(py, output_array))
    }

    /// Reset the router state.
    ///
    /// ChannelRouter is stateless, so this is a no-op. Provided for API consistency.
    fn reset(&self) {
        // ChannelRouter is stateless, nothing to reset
    }

    /// Get the number of input channels.
    #[getter]
    fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Get the number of output channels.
    #[getter]
    fn out_channels(&self) -> usize {
        self.out_channels
    }

    fn __repr__(&self) -> String {
        format!(
            "ChannelRouter(in_channels={}, out_channels={})",
            self.in_channels, self.out_channels
        )
    }
}

impl ChannelRouter {
    fn create_inner(in_channels: usize, out_channels: usize, indices: &[usize]) -> RouterInner {
        // Try to use optimized implementations for common configurations
        match (in_channels, out_channels) {
            (4, 4) => {
                let arr: [usize; 4] = indices.try_into().unwrap();
                RouterInner::R4to4(ZsChannelRouter::new(arr))
            }
            (8, 8) => {
                let arr: [usize; 8] = indices.try_into().unwrap();
                RouterInner::R8to8(ZsChannelRouter::new(arr))
            }
            (8, 4) => {
                let arr: [usize; 4] = indices.try_into().unwrap();
                RouterInner::R8to4(ZsChannelRouter::new(arr))
            }
            (16, 16) => {
                let arr: [usize; 16] = indices.try_into().unwrap();
                RouterInner::R16to16(ZsChannelRouter::new(arr))
            }
            (16, 8) => {
                let arr: [usize; 8] = indices.try_into().unwrap();
                RouterInner::R16to8(ZsChannelRouter::new(arr))
            }
            (32, 32) => {
                let arr: [usize; 32] = indices.try_into().unwrap();
                RouterInner::R32to32(ZsChannelRouter::new(arr))
            }
            (64, 64) => {
                let arr: [usize; 64] = indices.try_into().unwrap();
                RouterInner::R64to64(ZsChannelRouter::new(arr))
            }
            _ => RouterInner::Dynamic {
                in_channels,
                out_channels,
                indices: indices.to_vec(),
            },
        }
    }
}

//! Python bindings for Pipeline composition.

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Dynamic processing pipeline that chains multiple processors.
///
/// Processors are called sequentially, with the output of each stage
/// becoming the input to the next. All processors must have compatible
/// channel counts (output channels of stage N = input channels of stage N+1).
///
/// # Example
/// ```python
/// import npyci as npy
/// import numpy as np
///
/// # Create processors
/// car = npy.CAR(channels=8)
/// lap = npy.SurfaceLaplacian.linear(channels=8)
///
/// # Create pipeline
/// pipeline = npy.Pipeline([car, lap])
///
/// # Process data through all stages
/// data = np.random.randn(1000, 8).astype(np.float32)
/// filtered = pipeline.process(data)
/// ```
#[pyclass]
pub struct Pipeline {
    /// List of processor objects (stored as Python objects for dynamic dispatch)
    processors: Vec<PyObject>,
    /// Number of stages in the pipeline
    n_stages: usize,
}

#[pymethods]
impl Pipeline {
    /// Create a new pipeline from a list of processors.
    ///
    /// Each processor must have a `.process(input)` method that takes
    /// a 2D numpy array and returns a 2D numpy array.
    ///
    /// Args:
    ///     processors (list): List of processor objects (CAR, SurfaceLaplacian,
    ///         ChannelRouter, etc.). Must have at least one processor.
    ///
    /// Returns:
    ///     Pipeline: A new pipeline.
    ///
    /// Raises:
    ///     ValueError: If the processor list is empty.
    ///
    /// Example:
    ///     >>> car = npy.CAR(channels=8)
    ///     >>> lap = npy.SurfaceLaplacian.linear(channels=8)
    ///     >>> pipeline = Pipeline([car, lap])
    #[new]
    fn new(processors: Vec<PyObject>) -> PyResult<Self> {
        if processors.is_empty() {
            return Err(PyValueError::new_err(
                "Pipeline requires at least one processor",
            ));
        }

        let n_stages = processors.len();

        Ok(Self {
            processors,
            n_stages,
        })
    }

    /// Process data through all pipeline stages.
    ///
    /// The input is passed through each processor in sequence. The output
    /// of stage N becomes the input to stage N+1.
    ///
    /// Args:
    ///     input (np.ndarray): Input data as 2D float32 array with shape (samples, channels).
    ///
    /// Returns:
    ///     np.ndarray: Output from the final pipeline stage.
    ///
    /// Raises:
    ///     ValueError: If channel counts are incompatible between stages.
    ///
    /// Example:
    ///     >>> data = np.random.randn(1000, 8).astype(np.float32)
    ///     >>> filtered = pipeline.process(data)
    fn process<'py>(
        &self,
        py: Python<'py>,
        input: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        // Convert input to owned array for processing
        let shape = input.shape();
        let (n_samples, n_channels) = (shape[0], shape[1]);

        let input_array = input.as_array();
        let mut data: Vec<f32> = input_array.iter().copied().collect();
        let mut current_channels = n_channels;

        // Process through each stage
        for (stage_idx, processor) in self.processors.iter().enumerate() {
            // Create a numpy array from current data
            let current_array = Array2::from_shape_vec((n_samples, current_channels), data)
                .map_err(|e| PyValueError::new_err(format!("Failed to create array: {}", e)))?;
            let py_array = PyArray2::from_owned_array(py, current_array);

            // Call the processor's process method
            let result = processor
                .call_method1(py, "process", (py_array,))
                .map_err(|e| {
                    PyValueError::new_err(format!("Pipeline stage {} failed: {}", stage_idx, e))
                })?;

            // Extract the result as a numpy array
            let result_array: PyReadonlyArray2<f32> = result.extract(py).map_err(|e| {
                PyValueError::new_err(format!(
                    "Pipeline stage {} did not return a 2D float32 array: {}",
                    stage_idx, e
                ))
            })?;

            // Update data for next stage
            let result_shape = result_array.shape();
            if result_shape[0] != n_samples {
                return Err(PyValueError::new_err(format!(
                    "Pipeline stage {} changed sample count: expected {}, got {}",
                    stage_idx, n_samples, result_shape[0]
                )));
            }

            current_channels = result_shape[1];
            data = result_array.as_array().iter().copied().collect();
        }

        // Create final output array
        let output_array = Array2::from_shape_vec((n_samples, current_channels), data)
            .map_err(|e| PyValueError::new_err(format!("Failed to create output: {}", e)))?;
        Ok(PyArray2::from_owned_array(py, output_array))
    }

    /// Reset all processors in the pipeline.
    ///
    /// Calls `.reset()` on each processor if the method exists.
    fn reset(&self, py: Python<'_>) -> PyResult<()> {
        for processor in &self.processors {
            // Try to call reset, ignore if method doesn't exist
            let _ = processor.call_method0(py, "reset");
        }
        Ok(())
    }

    /// Get the number of stages in the pipeline.
    #[getter]
    fn n_stages(&self) -> usize {
        self.n_stages
    }

    fn __repr__(&self) -> String {
        format!("Pipeline(n_stages={})", self.n_stages)
    }

    fn __len__(&self) -> usize {
        self.n_stages
    }
}

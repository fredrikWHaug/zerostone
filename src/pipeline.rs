//! Pipeline infrastructure for composing signal processing primitives.
//!
//! This module provides traits and utilities for building real-time processing
//! chains similar to BCI2000, OpenEphys, or VST plugin architectures.
//!
//! # Overview
//!
//! After building 30+ signal processing primitives, this module enables composing
//! them into processing chains for real-world BCI and audio applications. The core
//! abstraction is [`BlockProcessor`], a trait for processing blocks of multi-channel
//! time-series data.
//!
//! # Design Philosophy
//!
//! ## Why Block-Based?
//!
//! Real-time signal processing operates on **blocks** (chunks) of samples rather than
//! individual samples because:
//!
//! - **Latency vs Throughput:** Smaller blocks reduce latency, larger blocks increase
//!   throughput. Typical BCI systems use 32-256 sample blocks (16-128 ms at 250 Hz).
//! - **Cache Efficiency:** Processing contiguous blocks improves CPU cache utilization
//!   compared to sample-by-sample processing.
//! - **Hardware Alignment:** Audio interfaces and ADCs deliver data in blocks matching
//!   DMA buffer sizes.
//!
//! See: <https://trtc.io/blog/details/throughput-latency>
//!
//! ## Why Not Iterator-Based?
//!
//! While Rust's iterator ecosystem (e.g., dasp's Signal trait) is elegant for
//! pull-based processing, real-time audio/BCI is **push-based**:
//!
//! - Hardware delivers blocks at fixed intervals (interrupt-driven)
//! - Processors must handle data immediately without backpressure
//! - Buffer ownership must be explicit (no hidden allocations)
//!
//! Block-based APIs make these constraints explicit and match industry standards
//! (VST, AudioUnit, BCI2000).
//!
//! ## Why Not Callback-Based?
//!
//! Callback-based APIs (common in C/C++) are harder to compose and reason about:
//!
//! ```ignore
//! // Callback style (harder to compose)
//! filter.process(input, |filtered| {
//!     detector.process(filtered, |detected| {
//!         // Nested callbacks get unwieldy
//!     });
//! });
//!
//! // Block style (easier to compose)
//! filter.process_block(input, &mut temp);
//! detector.process_block(&temp, output);
//! ```
//!
//! # Block Layout
//!
//! Blocks use **interleaved layout** where each element is a multi-channel sample:
//!
//! ```text
//! [[ch0, ch1, ch2], [ch0, ch1, ch2], [ch0, ch1, ch2], ...]
//!  ^-- sample 0 --^  ^-- sample 1 --^  ^-- sample 2 --^
//! ```
//!
//! This matches:
//! - **Hardware acquisition:** ADCs deliver samples with all channels simultaneously
//! - **Time-domain processing:** Algorithms iterate over time, accessing all channels
//! - **Cache locality:** Samples that are processed together are stored together
//!
//! Alternative **planar layout** (one buffer per channel) can be more efficient for
//! frequency-domain processing or SIMD operations, but requires conversion utilities.
//!
//! # Industry Patterns
//!
//! This design draws from established real-time processing frameworks:
//!
//! - **BCI2000 GenericFilter:** Separate input/output with `Process(Input&, Output&)`
//!   - Channel-major storage: `signal(channel, element)`
//!   - Copy-on-write semantics for efficiency
//!   - See: <https://www.bci2000.org/mediawiki/index.php/Programming_Reference:GenericFilter_Class>
//!
//! - **OpenEphys Processors:** Plugin architecture with `process()` callback
//!   - SOURCE/FILTER/SINK processor types
//!   - Real-time multi-channel continuous data
//!   - See: <https://open-ephys.github.io/gui-docs/Developer-Guide/Creating-a-new-plugin.html>
//!
//! - **dasp (Rust Audio):** Signal trait with Frame abstraction
//!   - Iterator-like API for streams
//!   - Zero-allocation, trait-generic design
//!   - See: <https://github.com/RustAudio/dasp>
//!
//! # Processing Modes
//!
//! [`BlockProcessor`] supports both **in-place** and **out-of-place** processing:
//!
//! ## In-Place Processing
//!
//! Modifies the input buffer directly, saving memory:
//!
//! ```ignore
//! let mut samples = [[1.0, 2.0], [3.0, 4.0]];
//! filter.process_block_inplace(&mut samples);
//! // samples now contains filtered values
//! ```
//!
//! **Advantages:**
//! - No temporary buffers needed
//! - Better cache locality (single buffer)
//! - Lower memory bandwidth (no copying)
//!
//! **Disadvantages:**
//! - Overwrites original data
//! - Some algorithms require separate buffers (FFT, median filter)
//!
//! ## Out-of-Place Processing
//!
//! Reads from input, writes to separate output:
//!
//! ```ignore
//! let input = [[1.0, 2.0], [3.0, 4.0]];
//! let mut output = [[0.0; 2]; 4];
//! let n_written = filter.process_block(&input, &mut output);
//! ```
//!
//! **Advantages:**
//! - Preserves original data
//! - Easier to reason about (pure function)
//! - Required for rate-changing processors (decimation/interpolation)
//!
//! **Disadvantages:**
//! - Requires extra memory
//! - Extra copy overhead (unless algorithm requires it anyway)
//!
//! See: <https://www.tutorialspoint.com/digital_signal_processing/dsp_in_place_computation.htm>
//!
//! # Future Additions
//!
//! Planned for Week 1 (Jan 27 - Feb 1):
//!
//! - **`Pipeline<P1, P2, ...>`:** Type-safe processor chaining
//! - **`ChannelRouter`:** Channel selection, splitting, and merging
//! - **Implementations:** BlockProcessor for all existing primitives (filters, FFT, etc.)
//!
//! # Examples
//!
//! ```
//! use zerostone::{BlockProcessor, IirFilter, BiquadCoeffs};
//!
//! // Create 4th-order Butterworth lowpass (2 cascaded biquads)
//! let coeffs = BiquadCoeffs::butterworth_lowpass(1000.0, 100.0);
//! let mut filter: IirFilter<2> = IirFilter::new([coeffs, coeffs]);
//!
//! // Process a block of single-channel data
//! let input = [[1.0], [3.0], [5.0]];  // 3 samples × 1 channel
//! let mut output = [[0.0]; 3];
//! let n_written = BlockProcessor::process_block(&mut filter, &input, &mut output);
//! assert_eq!(n_written, 3);
//! ```

use core::default::Default;

/// Block processor for multi-channel real-time signal processing.
///
/// This trait provides a unified interface for processing blocks of multi-channel data,
/// supporting both in-place and out-of-place operation modes. Implementations handle
/// streaming audio, neural signals, or other time-series data.
///
/// # Type Parameters
///
/// - `Sample`: The sample data type (f32, f64, Complex, etc.)
/// - `CHANNELS`: The number of channels (compile-time constant)
///
/// # Processing Modes
///
/// - **In-place:** [`process_block_inplace`](BlockProcessor::process_block_inplace)
///   modifies the input buffer directly
/// - **Out-of-place:** [`process_block`](BlockProcessor::process_block) reads from
///   input and writes to separate output
///
/// Implementations should provide an efficient version of at least one method and use
/// the default implementation for the other. For example, filters typically optimize
/// `process_block_inplace` while decimators optimize `process_block`.
///
/// # Block Layout
///
/// Blocks use **interleaved layout** where each element is a multi-channel sample:
///
/// ```text
/// [[ch0, ch1, ch2], [ch0, ch1, ch2], ...]
///  ^-- sample 0 --^  ^-- sample 1 --^
/// ```
///
/// This layout is cache-friendly for time-domain algorithms and matches hardware
/// acquisition patterns (samples arrive with all channels simultaneously).
///
/// # Channel Count
///
/// The channel count is a **compile-time constant** (const generic parameter `CHANNELS`), which:
///
/// - Enables zero-allocation processing (stack arrays only)
/// - Provides type-level validation (can't mix 8-channel and 16-channel processors)
/// - Allows LLVM to optimize channel loops (unrolling, vectorization)
///
/// # Rate-Changing Processors
///
/// Some processors produce different numbers of output samples than input samples:
///
/// - **Decimators:** Output length = input length / decimation factor
/// - **Interpolators:** Output length = input length × interpolation factor
/// - **Event detectors:** Variable output length (only samples where events occur)
///
/// These processors should return the actual number of output samples written from
/// [`process_block`](BlockProcessor::process_block). See [`RateChangingProcessor`]
/// for the marker trait.
///
/// # Examples
///
/// ```ignore
/// use zerostone::{BlockProcessor, IirFilter, BiquadCoeffs};
///
/// // Create a 4th-order Butterworth lowpass filter at 100 Hz (1000 Hz sample rate)
/// let coeffs = BiquadCoeffs::butterworth_lowpass(1000.0, 100.0);
/// let mut filter: IirFilter<2> = IirFilter::new([coeffs, coeffs]);
///
/// // Out-of-place processing
/// let input = [[1.0, 2.0], [3.0, 4.0]];  // 2 samples × 2 channels
/// let mut output = [[0.0; 2]; 2];
/// let n_written = filter.process_block(&input, &mut output);
/// assert_eq!(n_written, 2);
///
/// // In-place processing (more efficient)
/// let mut samples = [[1.0, 2.0], [3.0, 4.0]];
/// filter.process_block_inplace(&mut samples);
/// ```
///
/// # See Also
///
/// - BCI2000 GenericFilter: <https://www.bci2000.org/mediawiki/index.php/Programming_Reference:GenericFilter_Class>
/// - OpenEphys Processors: <https://open-ephys.github.io/gui-docs/Developer-Guide/Creating-a-new-plugin.html>
/// - dasp Signal trait: <https://github.com/RustAudio/dasp>
pub trait BlockProcessor<const IN_CHANNELS: usize, const OUT_CHANNELS: usize = IN_CHANNELS> {
    /// Sample data type (f32, f64, Complex, i16, etc.).
    ///
    /// Must implement [`Copy`] for efficient block processing without allocations.
    type Sample: Copy + Default;

    /// Process a block of samples out-of-place (input → output).
    ///
    /// Reads from `input` and writes to `output`. The output length may differ from
    /// input length (e.g., decimation produces fewer samples, interpolation produces more).
    ///
    /// # Default Implementation
    ///
    /// The default implementation copies `input` to `output` and then calls
    /// [`process_block_inplace`](BlockProcessor::process_block_inplace). This is
    /// inefficient and should be overridden by processors that naturally operate
    /// out-of-place (decimators, interpolators, median filters).
    ///
    /// # Arguments
    ///
    /// * `input` - Input block (each element is a multi-channel sample)
    /// * `output` - Output buffer to write processed samples
    ///
    /// # Returns
    ///
    /// Number of valid output samples written to `output`. This may be less than
    /// `output.len()` for variable-rate processors (decimators, event detectors).
    ///
    /// For fixed-rate processors (filters, FFT), this should equal `input.len().min(output.len())`.
    ///
    /// # Panics
    ///
    /// May panic if output buffer is too small for the processor's requirements.
    /// Check documentation for specific processor constraints.
    ///
    /// # Performance
    ///
    /// - **Fixed-rate processors:** O(n) where n = block length
    /// - **Rate-changing processors:** O(n × rate) where rate is decimation/interpolation factor
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut filter = IirFilter::new(...);
    /// let input = [[1.0, 2.0], [3.0, 4.0]];  // 2 samples × 2 channels
    /// let mut output = [[0.0; 2]; 2];
    /// let n_written = filter.process_block(&input, &mut output);
    /// assert_eq!(n_written, 2);
    /// ```
    ///
    /// Rate-changing example:
    ///
    /// ```ignore
    /// let mut decimator = Decimator::new(4);  // Decimate by 4
    /// let input = [[1.0], [2.0], [3.0], [4.0]];  // 4 samples × 1 channel
    /// let mut output = [[0.0; 1]; 1];
    /// let n_written = decimator.process_block(&input, &mut output);
    /// assert_eq!(n_written, 1);  // 4 inputs → 1 output
    /// ```
    fn process_block(
        &mut self,
        input: &[[Self::Sample; IN_CHANNELS]],
        output: &mut [[Self::Sample; OUT_CHANNELS]],
    ) -> usize {
        // Default implementation only works when IN_CHANNELS == OUT_CHANNELS
        // Processors where channels differ MUST override this method
        if IN_CHANNELS != OUT_CHANNELS {
            panic!(
                "Default process_block requires IN_CHANNELS == OUT_CHANNELS. \
                 This processor has IN={}, OUT={} and must override process_block.",
                IN_CHANNELS, OUT_CHANNELS
            );
        }

        // Copy input to output manually (since IN == OUT at runtime)
        let len = input.len().min(output.len());
        for i in 0..len {
            for j in 0..IN_CHANNELS.min(OUT_CHANNELS) {
                output[i][j] = input[i][j];
            }
        }

        // Call in-place processing (safe because IN == OUT)
        // We use unsafe to cast the slice since we've runtime-checked equality
        unsafe {
            let output_as_in = core::slice::from_raw_parts_mut(
                output.as_mut_ptr() as *mut [Self::Sample; IN_CHANNELS],
                len,
            );
            self.process_block_inplace(output_as_in);
        }
        len
    }

    /// Process a block of samples in-place (modifies buffer).
    ///
    /// Modifies `block` directly, overwriting input values with processed output.
    /// This is more memory-efficient than [`process_block`](BlockProcessor::process_block)
    /// but requires mutable input.
    ///
    /// # Implementation Requirements
    ///
    /// Processors must implement at least one of [`process_block`](BlockProcessor::process_block)
    /// or `process_block_inplace`. The default implementation of each calls the other, but
    /// implementors should override at least one to avoid infinite recursion.
    ///
    /// **Recommended approach:**
    /// - Override the method that's most natural for your algorithm
    /// - Let the other use the default implementation (copy + process)
    ///
    /// # Arguments
    ///
    /// * `block` - Buffer to process in-place
    ///
    /// # Performance
    ///
    /// In-place processing is typically faster than out-of-place because:
    ///
    /// - No memory allocation for output buffer
    /// - Better cache locality (single buffer)
    /// - Lower memory bandwidth (no copying)
    ///
    /// However, some algorithms inherently require separate buffers (FFT, median filter)
    /// and won't benefit from in-place processing.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut filter = IirFilter::new(...);
    /// let mut samples = [[1.0, 2.0], [3.0, 4.0]];
    /// filter.process_block_inplace(&mut samples);
    /// // samples now contains filtered values
    /// ```
    fn process_block_inplace(&mut self, block: &mut [[Self::Sample; IN_CHANNELS]]) {
        // Default implementation only works when IN_CHANNELS == OUT_CHANNELS
        // Processors where channels differ cannot use in-place processing
        if IN_CHANNELS != OUT_CHANNELS {
            panic!(
                "In-place processing requires IN_CHANNELS == OUT_CHANNELS. \
                 This processor has IN={}, OUT={} and cannot use process_block_inplace.",
                IN_CHANNELS, OUT_CHANNELS
            );
        }

        // Copy to temporary, process out-of-place, copy back
        let len = block.len();
        if len == 0 {
            return;
        }

        // For small blocks, use stack allocation (up to 64 samples)
        if len <= 64 {
            let mut temp = [[Self::Sample::default(); IN_CHANNELS]; 64];
            temp[..len].copy_from_slice(block);

            // Cast to output type (safe because IN == OUT)
            let block_as_out = unsafe {
                core::slice::from_raw_parts_mut(
                    block.as_mut_ptr() as *mut [Self::Sample; OUT_CHANNELS],
                    len,
                )
            };
            let n = self.process_block(&temp[..len], block_as_out);
            debug_assert_eq!(n, len, "process_block changed length unexpectedly");
        } else {
            // For large blocks, process in chunks to avoid stack overflow
            const CHUNK: usize = 64;
            for chunk in block.chunks_mut(CHUNK) {
                let mut temp = [[Self::Sample::default(); IN_CHANNELS]; CHUNK];
                let chunk_len = chunk.len();
                temp[..chunk_len].copy_from_slice(chunk);

                // Cast to output type (safe because IN == OUT)
                let chunk_as_out = unsafe {
                    core::slice::from_raw_parts_mut(
                        chunk.as_mut_ptr() as *mut [Self::Sample; OUT_CHANNELS],
                        chunk_len,
                    )
                };
                let n = self.process_block(&temp[..chunk_len], chunk_as_out);
                debug_assert_eq!(n, chunk_len, "process_block changed length unexpectedly");
            }
        }
    }

    /// Reset the processor state to initial conditions.
    ///
    /// For stateful processors (filters, detectors), this clears internal buffers
    /// and counters. For stateless processors (CAR, Laplacian), this is a no-op.
    ///
    /// # When to Reset
    ///
    /// - **Between recordings:** Clear state when starting a new session
    /// - **After artifacts:** Reset filters after detected artifacts to prevent ringing
    /// - **Pipeline reconfiguration:** Reset all processors when changing parameters
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let mut filter = IirFilter::new(...);
    /// // ... process some data ...
    /// filter.reset();  // Clear filter state for new recording
    /// ```
    ///
    /// Pipeline reset:
    ///
    /// ```ignore
    /// // Reset all processors in a pipeline
    /// pipeline.reset();  // Calls reset() on all constituent processors
    /// ```
    fn reset(&mut self) {
        // Default: no-op (stateless processors)
    }

    /// Get a human-readable name for this processor.
    ///
    /// Used for debugging, logging, and pipeline visualization. The default
    /// implementation returns the Rust type name (e.g., `"zerostone::filter::IirFilter<2>"`).
    ///
    /// Implementations may override this to provide more user-friendly names:
    ///
    /// ```ignore
    /// fn name(&self) -> &str {
    ///     "IirFilter (4th-order Butterworth lowpass @ 100 Hz)"
    /// }
    /// ```
    ///
    /// # Examples
    ///
    /// ```ignore
    /// println!("Running: {}", filter.name());
    /// // Output: "Running: IirFilter (4th-order Butterworth lowpass @ 100 Hz)"
    /// ```
    ///
    /// Pipeline debugging:
    ///
    /// ```ignore
    /// for processor in pipeline.processors() {
    ///     println!("  - {}", processor.name());
    /// }
    /// ```
    fn name(&self) -> &str {
        core::any::type_name::<Self>() // Default: Rust type name
    }
}

/// Processor that changes sample rate (decimators, interpolators).
///
/// This marker trait indicates that the processor produces a different number
/// of output samples than input samples. Useful for pipeline validation and
/// automatic buffer sizing.
///
/// # Rate-Changing Processors
///
/// - **Decimators:** Reduce sample rate (M input samples → 1 output sample)
/// - **Interpolators:** Increase sample rate (1 input sample → M output samples)
/// - **Event detectors:** Variable rate (only output when events occur)
///
/// # Buffer Sizing
///
/// When chaining rate-changing processors, the pipeline must allocate appropriately
/// sized intermediate buffers. The [`output_length`](RateChangingProcessor::output_length)
/// method enables automatic buffer sizing:
///
/// ```ignore
/// let input_len = 1024;
/// let output_len = decimator.output_length(input_len).unwrap();
/// let mut buffer = vec![[0.0; CHANNELS]; output_len];
/// ```
///
/// # Examples
///
/// ```ignore
/// use zerostone::{BlockProcessor, RateChangingProcessor, Decimator};
///
/// let mut decimator: Decimator<8, 4> = Decimator::new();  // Decimate by 4
/// assert_eq!(decimator.output_length(1024), Some(256));   // 1024 / 4 = 256
/// ```
///
/// Variable-rate processor:
///
/// ```ignore
/// let mut detector = ThresholdDetector::new(3.0, 100);
/// assert_eq!(detector.output_length(1024), None);  // Variable (depends on data)
/// ```
pub trait RateChangingProcessor<const IN_CHANNELS: usize, const OUT_CHANNELS: usize = IN_CHANNELS>:
    BlockProcessor<IN_CHANNELS, OUT_CHANNELS>
{
    /// Expected output length for a given input length.
    ///
    /// Returns `None` if output length is variable (event-triggered processing).
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Fixed-rate decimator
    /// let decimator = Decimator::new(4);
    /// assert_eq!(decimator.output_length(1024), Some(256));
    ///
    /// // Fixed-rate interpolator
    /// let interpolator = Interpolator::new(4);
    /// assert_eq!(interpolator.output_length(256), Some(1024));
    ///
    /// // Variable-rate detector
    /// let detector = ThresholdDetector::new(3.0, 100);
    /// assert_eq!(detector.output_length(1024), None);  // Depends on signal
    /// ```
    fn output_length(&self, input_length: usize) -> Option<usize>;
}

/// Processor that can be cloned for parallel processing.
///
/// This marker trait enables SIMD or multi-threaded pipelines where each thread
/// has its own processor instance. Only processors with cloneable state can
/// implement this trait.
///
/// # Use Cases
///
/// - **Parallel processing:** Process multiple blocks concurrently on different cores
/// - **SIMD lanes:** Duplicate processor state across SIMD lanes
/// - **Pipeline branching:** Split signal into multiple paths (e.g., multi-band analysis)
///
/// # Stateful vs Stateless
///
/// - **Stateless processors** (CAR, Laplacian) can always be cloned safely
/// - **Stateful processors** (IIR, FIR) require careful handling of filter state
///
/// # Examples
///
/// ```ignore
/// use zerostone::{BlockProcessor, CloneableProcessor, MedianFilter};
///
/// let filter: MedianFilter<8, 5> = MedianFilter::new();
/// let filter_copy = filter.clone();  // OK: MedianFilter is Clone
///
/// // Parallel processing
/// let mut filters: Vec<_> = (0..4).map(|_| filter.clone()).collect();
/// for (thread_id, filter) in filters.iter_mut().enumerate() {
///     // Each thread has its own filter instance
/// }
/// ```
pub trait CloneableProcessor<const IN_CHANNELS: usize, const OUT_CHANNELS: usize = IN_CHANNELS>:
    BlockProcessor<IN_CHANNELS, OUT_CHANNELS> + Clone
{
}

// Automatic implementation: any BlockProcessor that's also Clone is CloneableProcessor
impl<T, const IN_CHANNELS: usize, const OUT_CHANNELS: usize>
    CloneableProcessor<IN_CHANNELS, OUT_CHANNELS> for T
where
    T: BlockProcessor<IN_CHANNELS, OUT_CHANNELS> + Clone,
{
}

/// Type-safe processor pipeline with compile-time channel validation.
///
/// Chains multiple processors together, ensuring that each processor's output
/// channels match the next processor's input channels at compile time.
///
/// # Type Parameters
///
/// - `P`: The current processor in the chain
/// - `Next`: The next processor in the chain (defaults to `Terminal` for single-stage pipelines)
///
/// # Compile-Time Validation
///
/// The `.chain()` method enforces that the output channels of the current processor
/// match the input channels of the next processor. Mismatches are caught at compile time:
///
/// ```compile_fail
/// # use zerostone::{Pipeline, IirFilter, Decimator, BiquadCoeffs};
/// let filter8ch: IirFilter<2> = IirFilter::new([BiquadCoeffs::identity(); 2]); // 8 channels
/// let decimator16ch: Decimator<16> = Decimator::new(4); // 16 channels
/// let pipeline = Pipeline::new(filter8ch).chain(decimator16ch); // ERROR: 8 != 16
/// ```
///
/// # Examples
///
/// Single-stage pipeline:
/// ```
/// use zerostone::{Pipeline, IirFilter, BiquadCoeffs, BlockProcessor};
///
/// let coeffs = BiquadCoeffs::butterworth_lowpass(1000.0, 100.0);
/// let filter: IirFilter<2> = IirFilter::new([coeffs, coeffs]);
/// let mut pipeline = Pipeline::new(filter);
///
/// let input = [[1.0]; 10];
/// let mut output = [[0.0]; 10];
/// pipeline.process_block(&input, &mut output);
/// ```
///
/// Multi-stage pipeline construction:
/// ```
/// use zerostone::{Pipeline, IirFilter, Decimator, BiquadCoeffs};
///
/// // Create processors
/// let coeffs = BiquadCoeffs::butterworth_lowpass(1000.0, 100.0);
/// let filter: IirFilter<2> = IirFilter::new([coeffs, coeffs]);
/// let decimator: Decimator<1> = Decimator::new(4);
///
/// // Chain them together - compile-time channel validation!
/// let _pipeline = Pipeline::new(filter).chain(decimator);
///
/// // Note: Multi-stage pipelines don't currently implement BlockProcessor
/// // due to Rust const generic limitations. Only single-stage pipelines
/// // (Pipeline<P, Terminal>) can be used with the BlockProcessor trait.
/// ```
pub struct Pipeline<P, Next = Terminal> {
    current: P,
    next: Next,
}

/// Terminal marker for end of pipeline chain.
///
/// This is the default `Next` parameter for `Pipeline<P>`, indicating
/// a single-stage pipeline with no subsequent processors.
pub struct Terminal;

impl<P> Pipeline<P, Terminal> {
    /// Create a new single-stage pipeline.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::{Pipeline, IirFilter, BiquadCoeffs};
    ///
    /// let coeffs = BiquadCoeffs::butterworth_lowpass(1000.0, 100.0);
    /// let filter: IirFilter<2> = IirFilter::new([coeffs, coeffs]);
    /// let pipeline = Pipeline::new(filter);
    /// ```
    pub fn new(processor: P) -> Self {
        Self {
            current: processor,
            next: Terminal,
        }
    }
}

// Chain method for single-stage pipelines (Terminal)
impl<P> Pipeline<P, Terminal> {
    /// Chain another processor to this single-stage pipeline.
    ///
    /// The compiler enforces that the first processor's output channels match the second processor's input channels.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::{Pipeline, IirFilter, Decimator, BiquadCoeffs};
    ///
    /// let coeffs = BiquadCoeffs::butterworth_lowpass(1000.0, 100.0);
    /// let filter: IirFilter<2> = IirFilter::new([coeffs, coeffs]);
    /// let decimator: Decimator<1> = Decimator::new(4);
    ///
    /// // Chain processors - compiler verifies channels match
    /// let pipeline = Pipeline::new(filter).chain(decimator);
    /// ```
    ///
    /// Compile-time error if channels don't match:
    /// ```compile_fail
    /// # use zerostone::{Pipeline, Decimator};
    /// let dec1: Decimator<8> = Decimator::new(2);
    /// let dec2: Decimator<16> = Decimator::new(2);
    /// let pipeline = Pipeline::new(dec1).chain(dec2); // ERROR: 8 != 16
    /// ```
    pub fn chain<P2, const IN: usize, const OUT: usize, const OUT2: usize>(
        self,
        next_processor: P2,
    ) -> Pipeline<P, Pipeline<P2, Terminal>>
    where
        P: BlockProcessor<IN, OUT>,
        P2: BlockProcessor<OUT, OUT2, Sample = P::Sample>,
    {
        Pipeline {
            current: self.current,
            next: Pipeline {
                current: next_processor,
                next: Terminal,
            },
        }
    }
}

// Chain method for multi-stage pipelines
impl<P1, P2> Pipeline<P1, Pipeline<P2, Terminal>> {
    /// Chain another processor to this multi-stage pipeline.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::{Pipeline, IirFilter, Decimator, BiquadCoeffs};
    ///
    /// let coeffs = BiquadCoeffs::butterworth_lowpass(1000.0, 100.0);
    /// let filter1: IirFilter<2> = IirFilter::new([coeffs, coeffs]);
    /// let filter2: IirFilter<2> = IirFilter::new([coeffs, coeffs]);
    /// let decimator: Decimator<1> = Decimator::new(4);
    ///
    /// // Chain three processors
    /// let pipeline = Pipeline::new(filter1)
    ///     .chain(filter2)
    ///     .chain(decimator);
    /// ```
    pub fn chain<P3, const IN: usize, const MID: usize, const OUT: usize, const OUT2: usize>(
        self,
        next_processor: P3,
    ) -> Pipeline<P1, Pipeline<P2, Pipeline<P3, Terminal>>>
    where
        P1: BlockProcessor<IN, MID>,
        P2: BlockProcessor<MID, OUT, Sample = P1::Sample>,
        P3: BlockProcessor<OUT, OUT2, Sample = P1::Sample>,
    {
        Pipeline {
            current: self.current,
            next: Pipeline {
                current: self.next.current,
                next: Pipeline {
                    current: next_processor,
                    next: Terminal,
                },
            },
        }
    }
}

// BlockProcessor implementation for single-stage pipeline
impl<P, const IN: usize, const OUT: usize> BlockProcessor<IN, OUT> for Pipeline<P, Terminal>
where
    P: BlockProcessor<IN, OUT>,
{
    type Sample = P::Sample;

    fn process_block(
        &mut self,
        input: &[[Self::Sample; IN]],
        output: &mut [[Self::Sample; OUT]],
    ) -> usize {
        self.current.process_block(input, output)
    }

    fn process_block_inplace(&mut self, block: &mut [[Self::Sample; IN]]) {
        self.current.process_block_inplace(block)
    }

    fn reset(&mut self) {
        self.current.reset();
    }

    fn name(&self) -> &str {
        self.current.name()
    }
}

// Note: Multi-stage pipelines (Pipeline<P, Pipeline<P2, ...>>) don't currently implement
// BlockProcessor due to Rust const generic limitations. For now, only single-stage pipelines
// (Pipeline<P, Terminal>) implement the trait. Multi-stage pipelines can still be constructed
// with chain(), but processing must be done manually through the nested structure.
//
// Future improvements could use associated types instead of const generics to enable
// BlockProcessor for arbitrary-length pipelines.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AcCoupler, BiquadCoeffs, Decimator, IirFilter};

    #[test]
    fn test_single_stage_pipeline() {
        // Create a simple single-stage pipeline
        let filter: IirFilter<2> = IirFilter::new([BiquadCoeffs::passthrough(); 2]);
        let mut pipeline = Pipeline::new(filter);

        let input = [[1.0], [2.0], [3.0]];
        let mut output = [[0.0]; 3];

        let n = pipeline.process_block(&input, &mut output);
        assert_eq!(n, 3);
    }

    #[test]
    fn test_two_stage_pipeline_construction() {
        // Test that we can construct a two-stage pipeline
        let filter1: IirFilter<2> = IirFilter::new([BiquadCoeffs::passthrough(); 2]);
        let filter2: IirFilter<2> = IirFilter::new([BiquadCoeffs::passthrough(); 2]);

        let _pipeline = Pipeline::new(filter1).chain(filter2);
        // Successfully constructs - this proves type-safe chaining works
    }

    #[test]
    fn test_three_stage_pipeline_construction() {
        // Test that we can construct a three-stage pipeline
        let filter1: IirFilter<2> = IirFilter::new([BiquadCoeffs::passthrough(); 2]);
        let filter2: IirFilter<2> = IirFilter::new([BiquadCoeffs::passthrough(); 2]);
        let filter3: IirFilter<2> = IirFilter::new([BiquadCoeffs::passthrough(); 2]);

        let _pipeline = Pipeline::new(filter1).chain(filter2).chain(filter3);
        // Successfully constructs
    }

    #[test]
    fn test_mixed_processor_pipeline() {
        // Test chaining different processor types
        let filter: IirFilter<2> = IirFilter::new([BiquadCoeffs::passthrough(); 2]);
        let decimator: Decimator<1> = Decimator::new(4);

        let _pipeline = Pipeline::new(filter).chain(decimator);
        // Successfully constructs with different processor types
    }

    #[test]
    fn test_pipeline_with_ac_coupler() {
        // Test chaining with AC coupler
        let coupler: AcCoupler<1> = AcCoupler::new(100.0, 0.1);
        let mut pipeline = Pipeline::new(coupler);

        let input = [[1.0], [1.0], [1.0]];
        let mut output = [[0.0]; 3];

        pipeline.process_block(&input, &mut output);
        // Verifies it works
    }

    #[test]
    fn test_pipeline_reset() {
        // Test that reset works on single-stage pipeline
        let coeffs = BiquadCoeffs::butterworth_lowpass(1000.0, 100.0);
        let filter: IirFilter<2> = IirFilter::new([coeffs, coeffs]);
        let mut pipeline = Pipeline::new(filter);

        let input = [[1.0], [2.0], [3.0]];
        let mut output1 = [[0.0]; 3];
        let mut output2 = [[0.0]; 3];

        // Process once
        pipeline.process_block(&input, &mut output1);

        // Reset
        pipeline.reset();

        // Process again - should match a fresh pipeline
        pipeline.process_block(&input, &mut output2);

        // Results should be the same after reset
        for i in 0..3 {
            assert!((output1[i][0] - output2[i][0]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_pipeline_name() {
        let filter: IirFilter<2> = IirFilter::new([BiquadCoeffs::passthrough(); 2]);
        let pipeline = Pipeline::new(filter);

        assert_eq!(pipeline.name(), "IirFilter");
    }

    // Compile-time test: This should fail to compile if uncommented
    // #[test]
    // fn test_mismatched_channels_compile_error() {
    //     let filter8: IirFilter<8> = IirFilter::new(...);
    //     let filter16: IirFilter<16> = IirFilter::new(...);
    //     let _pipeline = Pipeline::new(filter8).chain(filter16); // ERROR: channels don't match
    // }
}

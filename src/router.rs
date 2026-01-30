//! Channel routing for pipeline composition
//!
//! The `ChannelRouter` enables flexible channel selection, reordering, and duplication
//! between pipeline stages. This is essential for:
//! - Selecting channel subsets (e.g., process only frontal EEG channels)
//! - Reordering channels (e.g., group spatially-adjacent channels)
//! - Broadcasting channels (e.g., duplicate reference channel)
//!
//! # Examples
//!
//! ```
//! use zerostone::ChannelRouter;
//!
//! // Select channels 0, 2, 4 from 8-channel input
//! let router: ChannelRouter<8, 3> = ChannelRouter::select([0, 2, 4]);
//! let input = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
//! let output = router.process(&input);
//! assert_eq!(output, [0.0, 2.0, 4.0]);
//!
//! // Permute channel order
//! let router: ChannelRouter<3, 3> = ChannelRouter::permute([2, 0, 1]);
//! let input = [10.0, 20.0, 30.0];
//! let output = router.process(&input);
//! assert_eq!(output, [30.0, 10.0, 20.0]);
//!
//! // Duplicate single channel to all outputs
//! let router: ChannelRouter<1, 4> = ChannelRouter::duplicate();
//! let input = [5.0];
//! let output = router.process(&input);
//! assert_eq!(output, [5.0, 5.0, 5.0, 5.0]);
//! ```

#![allow(clippy::module_name_repetitions)]

use crate::pipeline::BlockProcessor;

/// Routes channels between pipeline stages by selecting, reordering, or duplicating them.
///
/// The router uses a compile-time sized index array that maps output channels to input channels.
/// For example, `indices[0] = 2` means the first output channel comes from the third input channel.
///
/// # Type Parameters
///
/// - `IN_CHANNELS`: Number of input channels (compile-time constant)
/// - `OUT_CHANNELS`: Number of output channels (compile-time constant)
///
/// # Performance
///
/// - **Space**: O(OUT_CHANNELS) for index array (stack-allocated)
/// - **Time**: O(OUT_CHANNELS) per sample (simple array indexing)
/// - **Latency**: Deterministic, zero-allocation, real-time safe
///
/// # Examples
///
/// ```
/// use zerostone::{ChannelRouter, BlockProcessor};
///
/// // Create an 8→3 channel selector
/// let mut router: ChannelRouter<8, 3> = ChannelRouter::select([0, 3, 7]);
///
/// // Process a block of samples
/// let input = [
///     [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
///     [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
/// ];
/// let mut output = [[0.0; 3]; 2];
///
/// let n = router.process_block(&input, &mut output);
/// assert_eq!(n, 2);
/// assert_eq!(output[0], [1.0, 4.0, 8.0]);
/// assert_eq!(output[1], [9.0, 12.0, 16.0]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChannelRouter<const IN_CHANNELS: usize, const OUT_CHANNELS: usize> {
    /// Maps output channel index → input channel index
    /// E.g., indices[0] = 2 means output[0] comes from input[2]
    indices: [usize; OUT_CHANNELS],
}

impl<const IN: usize, const OUT: usize> ChannelRouter<IN, OUT> {
    // Compile-time validation
    const _ASSERT_IN: () = assert!(IN >= 1, "IN_CHANNELS must be at least 1");
    const _ASSERT_OUT: () = assert!(OUT >= 1, "OUT_CHANNELS must be at least 1");

    /// Creates a new channel router with explicit index mapping.
    ///
    /// # Arguments
    ///
    /// - `indices`: Array mapping output channel indices to input channel indices
    ///
    /// # Panics
    ///
    /// Panics if any index in `indices` is >= `IN_CHANNELS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::ChannelRouter;
    ///
    /// // Map output channels [0, 1] to input channels [3, 1]
    /// let router: ChannelRouter<8, 2> = ChannelRouter::new([3, 1]);
    /// let input = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
    /// let output = router.process(&input);
    /// assert_eq!(output, [3.0, 1.0]);
    /// ```
    #[must_use]
    pub fn new(indices: [usize; OUT]) -> Self {
        // Validate all indices are in bounds
        for (out_idx, &in_idx) in indices.iter().enumerate() {
            assert!(
                in_idx < IN,
                "Channel routing index out of bounds: indices[{}] = {} (must be < {})",
                out_idx,
                in_idx,
                IN
            );
        }

        Self { indices }
    }

    /// Creates a router that selects a subset of input channels.
    ///
    /// This is an alias for `new()` with clearer intent for channel selection use cases.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::ChannelRouter;
    ///
    /// // Select frontal channels from 8-channel EEG
    /// let router: ChannelRouter<8, 3> = ChannelRouter::select([0, 1, 2]);
    /// ```
    #[must_use]
    pub fn select(indices: [usize; OUT]) -> Self {
        Self::new(indices)
    }

    /// Creates a router that permutes (reorders) channels.
    ///
    /// This is an alias for `new()` with clearer intent for channel reordering use cases.
    /// Typically used when `IN_CHANNELS == OUT_CHANNELS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::ChannelRouter;
    ///
    /// // Reverse channel order
    /// let router: ChannelRouter<4, 4> = ChannelRouter::permute([3, 2, 1, 0]);
    /// let input = [1.0, 2.0, 3.0, 4.0];
    /// let output = router.process(&input);
    /// assert_eq!(output, [4.0, 3.0, 2.0, 1.0]);
    /// ```
    #[must_use]
    pub fn permute(indices: [usize; OUT]) -> Self {
        Self::new(indices)
    }

    /// Creates a router that duplicates a single input channel to all outputs.
    ///
    /// Only valid when `IN_CHANNELS == 1`. Broadcasts the single input channel
    /// to all `OUT_CHANNELS` output channels.
    ///
    /// # Panics
    ///
    /// Panics if `IN_CHANNELS != 1`.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::ChannelRouter;
    ///
    /// // Broadcast reference channel to 8 outputs
    /// let router: ChannelRouter<1, 8> = ChannelRouter::duplicate();
    /// let input = [42.0];
    /// let output = router.process(&input);
    /// assert_eq!(output, [42.0; 8]);
    /// ```
    #[must_use]
    pub fn duplicate() -> Self {
        assert!(IN == 1, "duplicate() requires IN_CHANNELS == 1, got {}", IN);
        Self { indices: [0; OUT] }
    }

    /// Creates an identity router that passes all channels through unchanged.
    ///
    /// Only valid when `IN_CHANNELS == OUT_CHANNELS`. Maps each output channel
    /// to the corresponding input channel (`output[i] = input[i]`).
    ///
    /// # Panics
    ///
    /// Panics if `IN_CHANNELS != OUT_CHANNELS`.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::ChannelRouter;
    ///
    /// let router: ChannelRouter<4, 4> = ChannelRouter::identity();
    /// let input = [1.0, 2.0, 3.0, 4.0];
    /// let output = router.process(&input);
    /// assert_eq!(output, input);
    /// ```
    #[must_use]
    pub fn identity() -> Self {
        assert!(
            IN == OUT,
            "identity() requires IN_CHANNELS == OUT_CHANNELS, got IN={}, OUT={}",
            IN,
            OUT
        );

        // Create identity mapping: [0, 1, 2, ..., OUT-1]
        let mut indices = [0; OUT];
        let mut i = 0;
        while i < OUT {
            indices[i] = i;
            i += 1;
        }
        Self { indices }
    }

    /// Returns the index mapping array.
    ///
    /// Each element `indices()[i]` indicates which input channel maps to output channel `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::ChannelRouter;
    ///
    /// let router: ChannelRouter<8, 3> = ChannelRouter::select([0, 2, 4]);
    /// assert_eq!(router.indices(), &[0, 2, 4]);
    /// ```
    #[must_use]
    pub const fn indices(&self) -> &[usize; OUT] {
        &self.indices
    }

    /// Processes a single sample by routing channels according to the index mapping.
    ///
    /// # Examples
    ///
    /// ```
    /// use zerostone::ChannelRouter;
    ///
    /// let router: ChannelRouter<4, 2> = ChannelRouter::select([1, 3]);
    /// let input = [10.0, 20.0, 30.0, 40.0];
    /// let output = router.process(&input);
    /// assert_eq!(output, [20.0, 40.0]);
    /// ```
    #[must_use]
    pub fn process(&self, input: &[f32; IN]) -> [f32; OUT] {
        core::array::from_fn(|i| input[self.indices[i]])
    }
}

impl<const IN: usize, const OUT: usize> BlockProcessor<IN, OUT> for ChannelRouter<IN, OUT> {
    type Sample = f32;

    fn process_block(&mut self, input: &[[f32; IN]], output: &mut [[f32; OUT]]) -> usize {
        let len = input.len().min(output.len());
        for i in 0..len {
            output[i] = self.process(&input[i]);
        }
        len
    }

    fn name(&self) -> &str {
        "ChannelRouter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_channels() {
        let router: ChannelRouter<8, 3> = ChannelRouter::select([0, 2, 4]);
        let input = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let output = router.process(&input);
        assert_eq!(output, [0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_select_non_sequential() {
        let router: ChannelRouter<8, 4> = ChannelRouter::select([7, 3, 1, 5]);
        let input = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let output = router.process(&input);
        assert_eq!(output, [7.0, 3.0, 1.0, 5.0]);
    }

    #[test]
    fn test_permute_channels() {
        let router: ChannelRouter<3, 3> = ChannelRouter::permute([2, 0, 1]);
        let input = [10.0, 20.0, 30.0];
        let output = router.process(&input);
        assert_eq!(output, [30.0, 10.0, 20.0]);
    }

    #[test]
    fn test_permute_reverse() {
        let router: ChannelRouter<4, 4> = ChannelRouter::permute([3, 2, 1, 0]);
        let input = [1.0, 2.0, 3.0, 4.0];
        let output = router.process(&input);
        assert_eq!(output, [4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_duplicate_channel() {
        let router: ChannelRouter<1, 4> = ChannelRouter::duplicate();
        let input = [5.0];
        let output = router.process(&input);
        assert_eq!(output, [5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_duplicate_to_many() {
        let router: ChannelRouter<1, 8> = ChannelRouter::duplicate();
        let input = [42.0];
        let output = router.process(&input);
        assert_eq!(output, [42.0; 8]);
    }

    #[test]
    fn test_identity() {
        let router: ChannelRouter<4, 4> = ChannelRouter::identity();
        let input = [1.0, 2.0, 3.0, 4.0];
        let output = router.process(&input);
        assert_eq!(output, input);
    }

    #[test]
    fn test_indices_accessor() {
        let router: ChannelRouter<8, 3> = ChannelRouter::select([0, 2, 4]);
        assert_eq!(router.indices(), &[0, 2, 4]);

        let router2: ChannelRouter<4, 4> = ChannelRouter::permute([3, 2, 1, 0]);
        assert_eq!(router2.indices(), &[3, 2, 1, 0]);
    }

    #[test]
    fn test_copy_trait() {
        let router: ChannelRouter<4, 2> = ChannelRouter::select([0, 3]);
        let copied = router; // Copy trait allows implicit copy
        let copied2 = router; // Can use original after copy

        assert_eq!(router.indices(), copied.indices());
        assert_eq!(copied.indices(), copied2.indices());
    }

    #[test]
    fn test_clone_trait() {
        // Verify Clone is implemented (compile-time check)
        fn assert_clone<T: Clone>() {}
        assert_clone::<ChannelRouter<4, 2>>();
    }

    #[test]
    fn test_equality() {
        let router1: ChannelRouter<4, 2> = ChannelRouter::select([0, 3]);
        let router2: ChannelRouter<4, 2> = ChannelRouter::select([0, 3]);
        let router3: ChannelRouter<4, 2> = ChannelRouter::select([1, 2]);

        assert_eq!(router1, router2);
        assert_ne!(router1, router3);
    }

    #[test]
    fn test_debug_impl_exists() {
        // Just verify Debug is implemented (compile-time check)
        fn assert_debug<T: core::fmt::Debug>() {}
        assert_debug::<ChannelRouter<4, 2>>();
    }

    #[test]
    fn test_identity_single_channel() {
        let router: ChannelRouter<1, 1> = ChannelRouter::identity();
        let input = [42.0];
        let output = router.process(&input);
        assert_eq!(output, [42.0]);
    }

    #[test]
    fn test_single_channel_input() {
        let router: ChannelRouter<1, 1> = ChannelRouter::select([0]);
        let input = [99.0];
        let output = router.process(&input);
        assert_eq!(output, [99.0]);
    }

    #[test]
    fn test_single_channel_output() {
        let router: ChannelRouter<8, 1> = ChannelRouter::select([5]);
        let input = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let output = router.process(&input);
        assert_eq!(output, [5.0]);
    }

    #[test]
    fn test_duplicate_selection() {
        // Selecting the same channel multiple times is allowed
        let router: ChannelRouter<4, 3> = ChannelRouter::new([1, 1, 1]);
        let input = [10.0, 20.0, 30.0, 40.0];
        let output = router.process(&input);
        assert_eq!(output, [20.0, 20.0, 20.0]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_invalid_index() {
        let _router: ChannelRouter<4, 2> = ChannelRouter::new([0, 5]); // 5 >= 4
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_invalid_index_in_middle() {
        let _router: ChannelRouter<8, 4> = ChannelRouter::new([0, 2, 9, 7]); // 9 >= 8
    }

    #[test]
    fn test_block_processor_out_of_place() {
        let mut router: ChannelRouter<4, 2> = ChannelRouter::select([0, 3]);
        let input = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let mut output = [[0.0; 2]; 2];

        let n = router.process_block(&input, &mut output);
        assert_eq!(n, 2);
        assert_eq!(output[0], [1.0, 4.0]);
        assert_eq!(output[1], [5.0, 8.0]);
    }

    #[test]
    fn test_block_processor_with_permute() {
        let mut router: ChannelRouter<3, 3> = ChannelRouter::permute([2, 0, 1]);
        let input = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let mut output = [[0.0; 3]; 3];

        let n = router.process_block(&input, &mut output);
        assert_eq!(n, 3);
        assert_eq!(output[0], [3.0, 1.0, 2.0]);
        assert_eq!(output[1], [6.0, 4.0, 5.0]);
        assert_eq!(output[2], [9.0, 7.0, 8.0]);
    }

    #[test]
    fn test_block_processor_with_duplicate() {
        let mut router: ChannelRouter<1, 3> = ChannelRouter::duplicate();
        let input = [[1.0], [2.0], [3.0]];
        let mut output = [[0.0; 3]; 3];

        let n = router.process_block(&input, &mut output);
        assert_eq!(n, 3);
        assert_eq!(output[0], [1.0, 1.0, 1.0]);
        assert_eq!(output[1], [2.0, 2.0, 2.0]);
        assert_eq!(output[2], [3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_block_processor_partial_block() {
        let mut router: ChannelRouter<4, 2> = ChannelRouter::select([1, 2]);
        let input = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let mut output = [[0.0; 2]; 1]; // Output smaller than input

        let n = router.process_block(&input, &mut output);
        assert_eq!(n, 1);
        assert_eq!(output[0], [2.0, 3.0]);
    }

    #[test]
    fn test_block_processor_name() {
        let router: ChannelRouter<4, 2> = ChannelRouter::select([0, 1]);
        assert_eq!(router.name(), "ChannelRouter");
    }

    #[test]
    #[should_panic]
    fn test_block_processor_inplace_panics_when_channels_differ() {
        let mut router: ChannelRouter<4, 2> = ChannelRouter::select([0, 1]);
        let mut block = [[1.0, 2.0, 3.0, 4.0]];

        // Should panic - can't do in-place when IN != OUT
        router.process_block_inplace(&mut block);
    }

    #[test]
    fn test_block_processor_inplace_works_for_permute() {
        let mut router: ChannelRouter<3, 3> = ChannelRouter::permute([2, 0, 1]);
        let mut block = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        router.process_block_inplace(&mut block);

        assert_eq!(block[0], [3.0, 1.0, 2.0]);
        assert_eq!(block[1], [6.0, 4.0, 5.0]);
    }

    #[test]
    fn test_block_processor_inplace_identity() {
        let mut router: ChannelRouter<4, 4> = ChannelRouter::identity();
        let mut block = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let original = block;

        router.process_block_inplace(&mut block);

        assert_eq!(block, original);
    }

    #[test]
    fn test_pipeline_single_stage_select() {
        use crate::pipeline::Pipeline;

        // Create a single-stage pipeline with a router
        let router: ChannelRouter<8, 3> = ChannelRouter::select([0, 2, 4]);
        let mut pipeline = Pipeline::new(router);

        // Process some data
        let input = [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        ];
        let mut output = [[0.0; 3]; 2];

        let n = pipeline.process_block(&input, &mut output);
        assert_eq!(n, 2);

        // Verify correct channels selected
        assert_eq!(output[0], [1.0, 3.0, 5.0]);
        assert_eq!(output[1], [9.0, 11.0, 13.0]);
    }

    #[test]
    fn test_pipeline_single_stage_duplicate() {
        use crate::pipeline::Pipeline;

        // Duplicate single channel to 4 channels
        let duplicator: ChannelRouter<1, 4> = ChannelRouter::duplicate();
        let mut pipeline = Pipeline::new(duplicator);

        // Process data
        let input = [[1.0], [2.0], [3.0]];
        let mut output = [[0.0; 4]; 3];

        let n = pipeline.process_block(&input, &mut output);
        assert_eq!(n, 3);

        // All 4 output channels should be identical
        assert_eq!(output[0], [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(output[1], [2.0, 2.0, 2.0, 2.0]);
        assert_eq!(output[2], [3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_pipeline_type_safe_chaining() {
        use crate::decimate::Decimator;
        use crate::pipeline::Pipeline;

        // Test that type-safe composition compiles
        let router: ChannelRouter<8, 3> = ChannelRouter::select([0, 2, 4]);
        let decimator: Decimator<3> = Decimator::new(2);

        // This should compile - channels match (3 → 3)
        let _pipeline = Pipeline::new(router).chain(decimator);

        // The following would NOT compile (channels don't match):
        // let bad_decimator: Decimator<5> = Decimator::new(2);
        // let _bad_pipeline = Pipeline::new(router).chain(bad_decimator); // ERROR: 3 != 5
    }

    #[test]
    fn test_pipeline_multi_router_composition() {
        use crate::pipeline::Pipeline;

        // Test chaining two routers
        let router1: ChannelRouter<4, 3> = ChannelRouter::select([0, 1, 2]);
        let router2: ChannelRouter<3, 3> = ChannelRouter::permute([2, 0, 1]);

        // This should compile - channels match (3 → 3)
        let _pipeline = Pipeline::new(router1).chain(router2);

        // Verify type-checking works at compile time
    }
}

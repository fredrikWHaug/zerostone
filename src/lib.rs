#![no_std]

mod buffer;
mod detector;
mod filter;
mod stats;

// Re-export at crate root for convenience
pub use buffer::CircularBuffer;
pub use detector::{DetectorState, SpikeEvent, ThresholdDetector};
pub use filter::{BiquadCoeffs, FirFilter, IirFilter};
pub use stats::OnlineStats;

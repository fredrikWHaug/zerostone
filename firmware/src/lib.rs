//! Zerostone firmware library.
//!
//! Contains hardware drivers and signal processing primitives for the
//! Zerostone neural recording platform. The library is `no_std` and
//! suitable for embedded ARM Cortex-M targets.

#![no_std]

pub mod ble;
pub mod classifier;
pub mod intan;
pub mod pipeline;
pub mod ring_buffer;

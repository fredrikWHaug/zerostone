//! Zerostone firmware library.
//!
//! Contains hardware drivers and signal processing primitives for the
//! Zerostone neural recording platform. The library is `no_std` and
//! suitable for embedded ARM Cortex-M targets.

#![cfg_attr(not(test), no_std)]

pub mod ble;
pub mod classifier;
pub mod dsp;
pub mod intan;
pub mod online_learn;
pub mod pipeline;
pub mod ring_buffer;
pub mod stats;
pub mod watchdog;

#[cfg(test)]
mod integration;

#[cfg(test)]
mod synth;

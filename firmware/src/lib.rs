//! Zerostone firmware library.
//!
//! Contains hardware drivers and signal processing primitives for the
//! Zerostone neural recording platform. The library is `no_std` and
//! suitable for embedded ARM Cortex-M targets.

#![cfg_attr(not(test), no_std)]

pub mod ble;
pub mod ble_server;
pub mod classifier;
pub mod dsp;
pub mod fault;
pub mod flash_log;
pub mod intan;
pub mod online_learn;
pub mod pipeline;
pub mod power;
pub mod ring_buffer;
pub mod stats;
pub mod stim;
pub mod watchdog;

#[cfg(test)]
mod integration;

#[cfg(test)]
mod sim;

#[cfg(test)]
mod synth;

//! Real-time spike detection pipeline.
//!
//! Consumes multi-channel ADC frames from the Intan SPI driver, runs
//! per-channel amplitude thresholding with online noise estimation,
//! and emits [`SpikeEvent`]s into a fixed-size [`EventQueue`].
//!
//! This is a lightweight first-stage detector. Template matching and
//! cluster assignment will be layered on top in a later stage.

use zerostone::float::Float;

// ---------------------------------------------------------------------------
// SpikeEvent
// ---------------------------------------------------------------------------

/// A detected spike event.
#[derive(Clone, Copy, Debug)]
pub struct SpikeEvent {
    /// Global sample index at detection time.
    pub sample_idx: u32,
    /// Channel on which the spike was detected (0-indexed).
    pub channel: u8,
    /// Cluster ID (0 = unclassified in the amplitude-only stage).
    pub cluster_id: u8,
    /// Peak amplitude in normalized units ([-1, 1]).
    pub amplitude: f32,
}

// ---------------------------------------------------------------------------
// EventQueue
// ---------------------------------------------------------------------------

/// Fixed-size circular buffer of [`SpikeEvent`]s.
///
/// `N` is the maximum number of events the queue can hold.
pub struct EventQueue<const N: usize> {
    buf: [Option<SpikeEvent>; N],
    head: usize,
    tail: usize,
    count: usize,
}

impl<const N: usize> EventQueue<N> {
    /// Creates a new empty event queue.
    pub const fn new() -> Self {
        Self {
            // Option<SpikeEvent> is not Copy in const context, so we
            // initialize with a const array of None via a helper.
            buf: [None; N],
            head: 0,
            tail: 0,
            count: 0,
        }
    }

    /// Pushes a spike event into the queue.
    ///
    /// Returns `true` on success, `false` if the queue is full.
    pub fn push(&mut self, event: SpikeEvent) -> bool {
        if self.count == N {
            return false;
        }
        self.buf[self.head] = Some(event);
        self.head = (self.head + 1) % N;
        self.count += 1;
        true
    }

    /// Pops the oldest spike event from the queue.
    pub fn pop(&mut self) -> Option<SpikeEvent> {
        if self.count == 0 {
            return None;
        }
        let event = self.buf[self.tail].take();
        self.tail = (self.tail + 1) % N;
        self.count -= 1;
        event
    }

    /// Returns the number of events in the queue.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns `true` if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

// ---------------------------------------------------------------------------
// Per-channel state
// ---------------------------------------------------------------------------

/// Online noise estimator and detector state for a single channel.
///
/// Noise is estimated via an exponential moving average (EMA) of
/// `|sample - running_median|`. The running median is itself tracked
/// with an EMA for simplicity (exact median requires sorting, which is
/// too expensive per-sample on an MCU). The MAD-like estimate is then
/// scaled by the configured threshold factor to set the detection
/// threshold.
#[derive(Clone, Copy)]
struct ChannelState {
    /// EMA of the signal (approximates median for zero-mean signals).
    level: Float,
    /// EMA of |sample - level| (approximates MAD).
    noise: Float,
    /// Whether the channel is currently inside a spike (to avoid
    /// re-triggering on the same excursion).
    in_spike: bool,
}

impl ChannelState {
    const fn new() -> Self {
        Self {
            level: 0.0,
            noise: 0.0,
            in_spike: false,
        }
    }
}

/// EMA smoothing factor for the level (median proxy) and noise
/// estimates. Smaller values = more smoothing. 0.002 at 30 kHz gives
/// an effective window of ~500 samples (~17 ms), long enough to
/// average over individual spikes but short enough to track baseline
/// drift.
const ALPHA_LEVEL: Float = 0.002;

/// EMA smoothing factor for the noise (MAD proxy). Slightly slower
/// than the level tracker so that individual spikes do not inflate the
/// noise estimate too quickly.
const ALPHA_NOISE: Float = 0.001;

/// Minimum noise floor to prevent division by zero or threshold
/// collapse on silent channels. Corresponds to ~1 LSB in the
/// normalized [-1, 1] range for a 16-bit ADC.
const MIN_NOISE: Float = 1.0 / 32768.0;

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Maximum number of channels supported by the pipeline.
///
/// The const generic `C` must not exceed this value.
const MAX_CHANNELS: usize = 128;

/// Real-time spike detection pipeline for `C` channels.
///
/// Call [`Pipeline::process_frame`] once per ADC sample (one sample per
/// channel) to detect spikes via simple amplitude thresholding with
/// online noise estimation.
pub struct Pipeline<const C: usize> {
    /// Per-channel detection state.
    channels: [ChannelState; MAX_CHANNELS],
    /// MAD multiplier for the detection threshold.
    threshold_factor: Float,
}

impl<const C: usize> Pipeline<C> {
    /// Creates a new pipeline.
    ///
    /// `threshold_factor` is the multiplier applied to the per-channel
    /// noise estimate (MAD proxy) to form the detection threshold.
    /// Typical values are 4.0--6.0; 5.0 is a common default.
    pub fn new(threshold_factor: Float) -> Self {
        assert!(C <= MAX_CHANNELS, "C exceeds MAX_CHANNELS");
        Self {
            channels: [ChannelState::new(); MAX_CHANNELS],
            threshold_factor,
        }
    }

    /// Processes one multi-channel frame and detects spikes.
    ///
    /// - `frame`: raw i16 ADC values, one per channel.
    /// - `sample_idx`: the global sample counter for this frame.
    /// - `events`: output queue where detected [`SpikeEvent`]s are pushed.
    ///
    /// Returns the number of spike events emitted for this frame.
    pub fn process_frame<const N: usize>(
        &mut self,
        frame: &[i16; C],
        sample_idx: u32,
        events: &mut EventQueue<N>,
    ) -> usize {
        let mut count = 0usize;

        for ch in 0..C {
            let raw = frame[ch];
            let x: Float = raw as Float / 32768.0;

            let st = &mut self.channels[ch];

            // Update running level (median proxy) via EMA.
            st.level += ALPHA_LEVEL * (x - st.level);

            // Update noise estimate (MAD proxy) via EMA of |residual|.
            let residual = x - st.level;
            let abs_residual = if residual < 0.0 { -residual } else { residual };
            st.noise += ALPHA_NOISE * (abs_residual - st.noise);

            // Enforce minimum noise floor.
            let noise = if st.noise > MIN_NOISE { st.noise } else { MIN_NOISE };

            // Threshold detection with hysteresis (in_spike flag).
            let threshold = self.threshold_factor * noise;
            let abs_x = if residual < 0.0 { -residual } else { residual };

            if abs_x > threshold {
                if !st.in_spike {
                    st.in_spike = true;
                    let pushed = events.push(SpikeEvent {
                        sample_idx,
                        channel: ch as u8,
                        cluster_id: 0,
                        amplitude: x,
                    });
                    if pushed {
                        count += 1;
                    }
                }
            } else {
                st.in_spike = false;
            }
        }

        count
    }
}

// ---------------------------------------------------------------------------
// Tests (host-only)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- EventQueue tests ---------------------------------------------------

    #[test]
    fn event_queue_new_is_empty() {
        let q = EventQueue::<8>::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn event_queue_push_pop() {
        let mut q = EventQueue::<4>::new();
        let ev = SpikeEvent {
            sample_idx: 42,
            channel: 3,
            cluster_id: 0,
            amplitude: -0.5,
        };
        assert!(q.push(ev));
        assert_eq!(q.len(), 1);

        let out = q.pop().unwrap();
        assert_eq!(out.sample_idx, 42);
        assert_eq!(out.channel, 3);
        assert!(q.is_empty());
    }

    #[test]
    fn event_queue_full_rejects() {
        let mut q = EventQueue::<2>::new();
        let ev = SpikeEvent {
            sample_idx: 0,
            channel: 0,
            cluster_id: 0,
            amplitude: 0.0,
        };
        assert!(q.push(ev));
        assert!(q.push(ev));
        assert!(!q.push(ev));
    }

    #[test]
    fn event_queue_empty_pop_returns_none() {
        let mut q = EventQueue::<4>::new();
        assert!(q.pop().is_none());
    }

    #[test]
    fn event_queue_wrap_around() {
        let mut q = EventQueue::<3>::new();
        let make = |idx| SpikeEvent {
            sample_idx: idx,
            channel: 0,
            cluster_id: 0,
            amplitude: 0.0,
        };

        // Fill.
        assert!(q.push(make(1)));
        assert!(q.push(make(2)));
        assert!(q.push(make(3)));

        // Drain two.
        assert_eq!(q.pop().unwrap().sample_idx, 1);
        assert_eq!(q.pop().unwrap().sample_idx, 2);

        // Push two more (wraps head).
        assert!(q.push(make(4)));
        assert!(q.push(make(5)));

        // Drain all.
        assert_eq!(q.pop().unwrap().sample_idx, 3);
        assert_eq!(q.pop().unwrap().sample_idx, 4);
        assert_eq!(q.pop().unwrap().sample_idx, 5);
        assert!(q.is_empty());
    }

    // -- Pipeline tests -----------------------------------------------------

    #[test]
    fn zeros_produce_no_spikes() {
        let mut pipe = Pipeline::<4>::new(5.0);
        let mut q = EventQueue::<16>::new();
        let frame = [0i16; 4];

        // Feed many zero frames — no spikes should appear.
        for i in 0..1000 {
            let n = pipe.process_frame(&frame, i, &mut q);
            assert_eq!(n, 0, "unexpected spike at frame {i}");
        }
        assert!(q.is_empty());
    }

    #[test]
    fn large_value_triggers_spike() {
        let mut pipe = Pipeline::<4>::new(5.0);
        let mut q = EventQueue::<16>::new();

        // Warm up with quiet frames so noise estimate is low.
        let quiet = [0i16; 4];
        for i in 0..2000 {
            pipe.process_frame(&quiet, i, &mut q);
        }
        assert!(q.is_empty());

        // Inject a large spike on channel 2.
        let mut spike_frame = [0i16; 4];
        spike_frame[2] = -20000; // large negative excursion
        let n = pipe.process_frame(&spike_frame, 2000, &mut q);

        assert_eq!(n, 1);
        let ev = q.pop().unwrap();
        assert_eq!(ev.channel, 2);
        assert_eq!(ev.sample_idx, 2000);
        assert!(ev.amplitude < 0.0);
    }

    #[test]
    fn noise_estimation_converges() {
        let mut pipe = Pipeline::<1>::new(5.0);
        let mut q = EventQueue::<16>::new();

        // Feed constant non-zero signal. Noise should converge toward
        // zero (since there is no variation around the level).
        let frame = [1000i16; 1];
        for i in 0..5000 {
            pipe.process_frame(&frame, i, &mut q);
        }

        // After convergence, level should be close to 1000/32768.
        let expected_level: Float = 1000.0 / 32768.0;
        let level = pipe.channels[0].level;
        let diff = if level > expected_level {
            level - expected_level
        } else {
            expected_level - level
        };
        assert!(
            diff < 0.001,
            "level {level} not close to expected {expected_level}"
        );

        // Noise (MAD proxy) should be very small since signal is constant.
        let noise = pipe.channels[0].noise;
        assert!(
            noise < 0.001,
            "noise {noise} should be near zero for constant signal"
        );
    }

    #[test]
    fn hysteresis_prevents_double_trigger() {
        let mut pipe = Pipeline::<1>::new(5.0);
        let mut q = EventQueue::<16>::new();

        // Warm up.
        let quiet = [0i16; 1];
        for i in 0..2000 {
            pipe.process_frame(&quiet, i, &mut q);
        }

        // Two consecutive large-amplitude frames should trigger only once
        // (the in_spike flag suppresses the second).
        let loud = [20000i16; 1];
        let n1 = pipe.process_frame(&loud, 2000, &mut q);
        let n2 = pipe.process_frame(&loud, 2001, &mut q);
        assert_eq!(n1, 1);
        assert_eq!(n2, 0);
    }
}

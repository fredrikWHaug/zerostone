//! Synthetic neural data generator for testing.
//!
//! Provides [`SynthStream`] which generates deterministic multi-channel
//! ADC frames with injected spike waveforms at known times. Used to
//! validate the detection pipeline, waveform extractor, and classifier
//! against ground-truth spike times and labels.

#[cfg(test)]
mod tests {
    use crate::classifier::{Classifier, WaveformExtractor};
    use crate::pipeline::{EventQueue, Pipeline, SpikeEvent};
    use zerostone::float::Float;

    // -----------------------------------------------------------------------
    // SynthSpike
    // -----------------------------------------------------------------------

    /// A ground-truth spike to inject into the synthetic stream.
    #[derive(Clone, Debug)]
    #[allow(dead_code)]
    struct SynthSpike {
        /// Which channel (0-indexed).
        channel: usize,
        /// Sample index at which the spike peak occurs.
        sample_idx: u32,
        /// Ground-truth cluster label.
        cluster_id: u8,
        /// Peak amplitude in ADC units (i16).
        amplitude: i16,
    }

    // -----------------------------------------------------------------------
    // SynthStream
    // -----------------------------------------------------------------------

    /// Deterministic synthetic neural data stream.
    #[allow(dead_code)]
    struct SynthStream {
        num_channels: usize,
        /// Spikes sorted by sample_idx.
        spikes: Vec<SynthSpike>,
        /// Background noise amplitude (uniform random in [-noise, +noise]).
        noise_amplitude: i16,
        /// Current sample counter (incremented each call to next_frame).
        current_sample: u32,
        /// Cursor into `spikes` for efficient scanning.
        spike_cursor: usize,
        /// xorshift32 PRNG state.
        rng_state: u32,
    }

    /// Deterministic xorshift32 PRNG. Period 2^32 - 1.
    fn xorshift32(state: &mut u32) -> u32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        x
    }

    impl SynthStream {
        /// Creates a new synthetic stream.
        fn new(num_channels: usize, noise_amplitude: i16) -> Self {
            Self {
                num_channels,
                spikes: Vec::new(),
                noise_amplitude,
                current_sample: 0,
                spike_cursor: 0,
                rng_state: 0xDEAD_BEEF,
            }
        }

        /// Adds a spike at the given time, channel, and amplitude.
        /// Spikes should be added in order of sample_idx for best performance.
        fn add_spike(&mut self, channel: usize, sample_idx: u32, cluster_id: u8, amplitude: i16) {
            self.spikes.push(SynthSpike {
                channel,
                sample_idx,
                cluster_id,
                amplitude,
            });
            // Keep sorted by sample_idx.
            self.spikes.sort_by_key(|s| s.sample_idx);
        }

        /// Generates one multi-channel frame.
        ///
        /// Background is uniform random noise in [-noise_amplitude, +noise_amplitude].
        /// Spike waveform is a triangular pulse: ramp from 0 to peak over 5 samples,
        /// then ramp back from peak to 0 over 5 samples (total duration 10 samples,
        /// peak at sample_idx).
        fn next_frame<const C: usize>(&mut self) -> [i16; C] {
            assert!(C >= self.num_channels);
            let mut frame = [0i16; C];

            // Fill with noise.
            if self.noise_amplitude > 0 {
                let range = (self.noise_amplitude as i32) * 2 + 1;
                for ch in 0..C {
                    let r = xorshift32(&mut self.rng_state);
                    let noise = ((r % range as u32) as i32) - self.noise_amplitude as i32;
                    frame[ch] = noise as i16;
                }
            }

            // Add spike waveforms. A spike with peak at sample_idx S affects
            // samples [S-5, S+4]. At sample t, the contribution is:
            //   t in [S-5, S-1]: amplitude * (t - (S-5)) / 5  (ramp up)
            //   t == S:          amplitude                      (peak)
            //   t in [S+1, S+4]: amplitude * (S+5 - t) / 5    (ramp down)
            let t = self.current_sample;
            for spike in &self.spikes {
                // Skip spikes too far in the past.
                if spike.sample_idx + 5 < t {
                    continue;
                }
                // Stop scanning if spikes are too far in the future.
                if t + 6 <= spike.sample_idx.saturating_sub(5) {
                    break;
                }

                let s = spike.sample_idx;
                let ch = spike.channel;
                if ch >= C {
                    continue;
                }

                // Check if current sample is within the spike's waveform window.
                let s_i64 = s as i64;
                let t_i64 = t as i64;
                let offset = t_i64 - s_i64; // negative = before peak

                if offset >= -5 && offset <= 4 {
                    let scale = if offset <= 0 {
                        // Ramp up: at offset=-5 -> 0, offset=0 -> 1.0
                        (5 + offset) as f32 / 5.0
                    } else {
                        // Ramp down: at offset=1 -> 4/5, offset=4 -> 1/5
                        (5 - offset) as f32 / 5.0
                    };
                    let contribution = (spike.amplitude as f32 * scale) as i32;
                    let current = frame[ch] as i32;
                    // Saturating add.
                    let sum = current + contribution;
                    frame[ch] = sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
                }
            }

            self.current_sample += 1;
            frame
        }
    }

    // -----------------------------------------------------------------------
    // Helper: drain all events from queue
    // -----------------------------------------------------------------------

    fn drain_events<const N: usize>(q: &mut EventQueue<N>) -> Vec<SpikeEvent> {
        let mut out = Vec::new();
        while let Some(ev) = q.pop() {
            out.push(ev);
        }
        out
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    const WARMUP: u32 = 2000;

    // -- Test 1: Detection rate with 100 large spikes ----------------------

    #[test]
    fn test_detection_rate_100_spikes() {
        const C: usize = 4;
        let mut stream = SynthStream::new(C, 50);
        let mut pipeline = Pipeline::<C>::new(5.0);
        let mut events = EventQueue::<256>::new();

        // Inject 100 spikes spaced 300 samples apart (10 Hz at 30 kHz)
        // on rotating channels, large amplitude.
        for i in 0..100u32 {
            let sample = WARMUP + 100 + i * 300;
            let ch = (i as usize) % C;
            stream.add_spike(ch, sample, 1, -20000);
        }

        // Run the full stream. Total duration: warmup + 100*300 + margin.
        let total_frames = WARMUP + 100 * 300 + 200;
        let mut detected = 0usize;
        for _ in 0..total_frames {
            let frame: [i16; C] = stream.next_frame();
            let n = pipeline.process_frame(&frame, stream.current_sample - 1, &mut events);
            detected += n;
            // Drain to avoid queue overflow.
            while events.pop().is_some() {}
        }

        let rate = detected as f64 / 100.0;
        assert!(
            rate >= 0.90,
            "detection rate {:.1}% is below 90% (detected {detected}/100)",
            rate * 100.0
        );
    }

    // -- Test 2: False positive rate with pure noise -----------------------

    #[test]
    fn test_false_positive_rate() {
        const C: usize = 4;
        let mut pipeline = Pipeline::<C>::new(5.0);
        let mut events = EventQueue::<256>::new();

        // Warm up with the SAME noise level so the EMA noise estimator
        // converges to the actual noise floor before we start counting.
        // 10,000 frames at ALPHA_NOISE=0.001 gives ~10 time constants.
        let mut stream = SynthStream::new(C, 50);
        let warmup_frames = 10_000u32;
        for _ in 0..warmup_frames {
            let frame: [i16; C] = stream.next_frame();
            pipeline.process_frame(&frame, stream.current_sample - 1, &mut events);
            while events.pop().is_some() {} // discard warmup FPs
        }

        // Count FPs over 1 second (30,000 frames) of pure noise.
        let test_frames = 30_000u32;
        let mut false_positives = 0usize;
        for _ in 0..test_frames {
            let frame: [i16; C] = stream.next_frame();
            let n = pipeline.process_frame(&frame, stream.current_sample - 1, &mut events);
            false_positives += n;
            while events.pop().is_some() {}
        }

        // After convergence, threshold = 5 * noise_ema. With uniform
        // noise in [-50, +50], MAD ~ 25/32768. Threshold ~ 5*25/32768
        // = 0.0038. Max noise value = 50/32768 = 0.0015. So threshold
        // is well above noise -- should be 0 FPs in steady state.
        assert!(
            false_positives < 10,
            "too many false positives: {false_positives} (expected < 10)"
        );
    }

    // -- Test 3: Classification accuracy -----------------------------------

    #[test]
    fn test_classification_accuracy() {
        // Strategy: Use two clearly distinct spike polarities (negative
        // vs positive) which produce waveforms with opposite sign in the
        // extractor. First, run one example of each through the pipeline
        // to capture the actual extracted waveform as a template. Then
        // inject many more spikes and classify them.
        const C: usize = 4;
        const W: usize = 16;

        // -- Phase 1: Learn templates from example spikes --

        // Template 1: negative spike.
        {
            let mut learn_stream = SynthStream::new(C, 0);
            let mut learn_pipe = Pipeline::<C>::new(5.0);
            let mut learn_ext = WaveformExtractor::<C, W>::new();
            let mut learn_q = EventQueue::<16>::new();

            learn_stream.add_spike(0, WARMUP + 50, 1, -20000);

            let total = WARMUP + 100;
            for _ in 0..total {
                let frame: [i16; C] = learn_stream.next_frame();
                let s = learn_stream.current_sample - 1;
                learn_ext.push_frame(&frame);
                learn_pipe.process_frame(&frame, s, &mut learn_q);
                if let Some(_ev) = learn_q.pop() {
                    // Capture the template right at detection.
                    let t1 = learn_ext.extract(0);

                    // Template 2: positive spike.
                    let mut learn_stream2 = SynthStream::new(C, 0);
                    let mut learn_pipe2 = Pipeline::<C>::new(5.0);
                    let mut learn_ext2 = WaveformExtractor::<C, W>::new();
                    let mut learn_q2 = EventQueue::<16>::new();

                    learn_stream2.add_spike(1, WARMUP + 50, 2, 20000);
                    for _ in 0..(WARMUP + 100) {
                        let f2: [i16; C] = learn_stream2.next_frame();
                        let s2 = learn_stream2.current_sample - 1;
                        learn_ext2.push_frame(&f2);
                        learn_pipe2.process_frame(&f2, s2, &mut learn_q2);
                        if let Some(_ev2) = learn_q2.pop() {
                            let t2 = learn_ext2.extract(1);

                            // -- Phase 2: Run classification with learned templates --
                            run_classification_with_templates(t1, t2);
                            return;
                        }
                    }
                    panic!("positive spike not detected during template learning");
                }
            }
            panic!("negative spike not detected during template learning");
        }
    }

    /// Helper for test_classification_accuracy. Runs the full classification
    /// test using pre-learned templates.
    fn run_classification_with_templates(
        t1: [Float; 16],
        t2: [Float; 16],
    ) {
        const C: usize = 4;
        const W: usize = 16;
        const MAX_T: usize = 4;

        let mut classifier = Classifier::<W, MAX_T>::new(0.3);
        classifier.add_template(&t1, 1); // negative spike template
        classifier.add_template(&t2, 2); // positive spike template

        let mut pipeline = Pipeline::<C>::new(5.0);
        let mut extractor = WaveformExtractor::<C, W>::new();
        let mut events = EventQueue::<64>::new();
        let mut stream = SynthStream::new(C, 10);

        // Ground truth: (sample_idx, expected_cluster_id).
        let mut ground_truth: Vec<(u32, u8)> = Vec::new();

        // 30 negative spikes on channel 0.
        for i in 0..30u32 {
            let s = WARMUP + 100 + i * 400;
            stream.add_spike(0, s, 1, -20000);
            ground_truth.push((s, 1));
        }
        // 30 positive spikes on channel 0 (after the negatives).
        for i in 0..30u32 {
            let s = WARMUP + 100 + 30 * 400 + i * 400;
            stream.add_spike(0, s, 2, 20000);
            ground_truth.push((s, 2));
        }

        let total_frames = WARMUP + 100 + 60 * 400 + 200;
        let mut detected_events: Vec<SpikeEvent> = Vec::new();

        for _ in 0..total_frames {
            let frame: [i16; C] = stream.next_frame();
            let sample = stream.current_sample - 1;
            extractor.push_frame(&frame);
            pipeline.process_frame(&frame, sample, &mut events);

            while let Some(mut ev) = events.pop() {
                if ev.channel == 0 {
                    let wf = extractor.extract(0);
                    ev.cluster_id = classifier.classify(&wf);
                    detected_events.push(ev);
                }
            }
        }

        // Match detected events to ground truth.
        let mut correct = 0usize;
        let mut total_matched = 0usize;
        for ev in &detected_events {
            let mut best_dist = 100u32;
            let mut best_gt_id: Option<u8> = None;
            for (gt_s, gt_id) in &ground_truth {
                let dist = if ev.sample_idx >= *gt_s {
                    ev.sample_idx - *gt_s
                } else {
                    *gt_s - ev.sample_idx
                };
                if dist < best_dist {
                    best_dist = dist;
                    best_gt_id = Some(*gt_id);
                }
            }
            if let Some(gt_id) = best_gt_id {
                if best_dist <= 10 {
                    total_matched += 1;
                    if ev.cluster_id == gt_id {
                        correct += 1;
                    }
                }
            }
        }

        assert!(
            total_matched >= 20,
            "too few matched events: {total_matched}"
        );

        let accuracy = correct as f64 / total_matched as f64;
        assert!(
            accuracy >= 0.70,
            "classification accuracy {:.1}% is below 70% ({correct}/{total_matched})",
            accuracy * 100.0
        );
    }

    // -- Test 4: Multi-channel detection -----------------------------------

    #[test]
    fn test_multi_channel_detection() {
        const C: usize = 32;
        let mut stream = SynthStream::new(C, 30);
        let mut pipeline = Pipeline::<C>::new(5.0);
        let mut events = EventQueue::<64>::new();

        // Spikes on channels 0, 7, 15, 31 at different times.
        let channels = [0usize, 7, 15, 31];
        for (i, &ch) in channels.iter().enumerate() {
            let s = WARMUP + 200 + (i as u32) * 500;
            stream.add_spike(ch, s, 1, -22000);
        }

        let total_frames = WARMUP + 200 + 4 * 500 + 200;
        let mut detected_channels: Vec<u8> = Vec::new();

        for _ in 0..total_frames {
            let frame: [i16; C] = stream.next_frame();
            pipeline.process_frame(&frame, stream.current_sample - 1, &mut events);
            for ev in drain_events(&mut events) {
                detected_channels.push(ev.channel);
            }
        }

        // Each target channel should have at least one detection.
        for &ch in &channels {
            assert!(
                detected_channels.contains(&(ch as u8)),
                "no spike detected on channel {ch}, detected channels: {:?}",
                detected_channels
            );
        }
    }

    // -- Test 5: Latency (sample_idx accuracy) -----------------------------

    #[test]
    fn test_latency_zero_sample() {
        const C: usize = 4;
        let mut stream = SynthStream::new(C, 0); // no noise
        let mut pipeline = Pipeline::<C>::new(5.0);
        let mut events = EventQueue::<16>::new();

        let spike_time: u32 = WARMUP + 500;
        stream.add_spike(0, spike_time, 1, -25000);

        let total_frames = WARMUP + 600;
        let mut detected: Vec<SpikeEvent> = Vec::new();

        for _ in 0..total_frames {
            let frame: [i16; C] = stream.next_frame();
            pipeline.process_frame(&frame, stream.current_sample - 1, &mut events);
            detected.extend(drain_events(&mut events));
        }

        // Find the detection for channel 0.
        let ch0_events: Vec<&SpikeEvent> =
            detected.iter().filter(|e| e.channel == 0).collect();
        assert!(
            !ch0_events.is_empty(),
            "no spike detected on channel 0"
        );

        // The detected sample_idx should be within the spike's waveform
        // window. The triangular pulse ramps up over 5 samples before peak,
        // so the threshold crossing happens on the rising edge. Allow a
        // tolerance of 6 samples (the ramp-up duration).
        let ev = ch0_events[0];
        let diff = if ev.sample_idx >= spike_time {
            ev.sample_idx - spike_time
        } else {
            spike_time - ev.sample_idx
        };
        assert!(
            diff <= 6,
            "detected at sample {} but spike peak was at {} (offset {})",
            ev.sample_idx, spike_time, diff
        );
    }

    // -- Test 6: xorshift32 determinism ------------------------------------

    #[test]
    fn test_xorshift_determinism() {
        let mut s1 = 42u32;
        let mut s2 = 42u32;
        let seq1: Vec<u32> = (0..100).map(|_| xorshift32(&mut s1)).collect();
        let seq2: Vec<u32> = (0..100).map(|_| xorshift32(&mut s2)).collect();
        assert_eq!(seq1, seq2, "same seed must produce same sequence");

        // Different seed produces different sequence.
        let mut s3 = 99u32;
        let seq3: Vec<u32> = (0..100).map(|_| xorshift32(&mut s3)).collect();
        assert_ne!(seq1, seq3, "different seeds should produce different sequences");
    }

    // -- Test 7: Spike ordering --------------------------------------------

    #[test]
    fn test_spike_ordering() {
        const C: usize = 4;
        let mut stream = SynthStream::new(C, 20);
        let mut pipeline = Pipeline::<C>::new(5.0);
        let mut events = EventQueue::<128>::new();

        // 20 spikes at increasing times on channel 0.
        let mut expected_times: Vec<u32> = Vec::new();
        for i in 0..20u32 {
            let s = WARMUP + 200 + i * 400;
            stream.add_spike(0, s, 1, -20000);
            expected_times.push(s);
        }

        let total_frames = WARMUP + 200 + 20 * 400 + 200;
        let mut detected_times: Vec<u32> = Vec::new();

        for _ in 0..total_frames {
            let frame: [i16; C] = stream.next_frame();
            pipeline.process_frame(&frame, stream.current_sample - 1, &mut events);
            for ev in drain_events(&mut events) {
                if ev.channel == 0 {
                    detected_times.push(ev.sample_idx);
                }
            }
        }

        // Verify monotonically increasing.
        for i in 1..detected_times.len() {
            assert!(
                detected_times[i] > detected_times[i - 1],
                "events not in order: {} then {} at indices {}/{}",
                detected_times[i - 1],
                detected_times[i],
                i - 1,
                i
            );
        }

        // Should detect most of the 20 spikes.
        assert!(
            detected_times.len() >= 15,
            "only detected {}/20 spikes",
            detected_times.len()
        );
    }

    // -- Test 8: High firing rate (100 Hz) ---------------------------------

    #[test]
    fn test_high_firing_rate() {
        const C: usize = 4;
        let mut stream = SynthStream::new(C, 30);
        let mut pipeline = Pipeline::<C>::new(5.0);
        let mut events = EventQueue::<256>::new();

        // 100 Hz = 1 spike every 300 samples at 30 kHz.
        // 1 second = 100 spikes.
        let num_spikes = 100u32;
        for i in 0..num_spikes {
            let s = WARMUP + 50 + i * 300;
            stream.add_spike(0, s, 1, -20000);
        }

        let total_frames = WARMUP + 50 + num_spikes * 300 + 200;
        let mut detected = 0usize;

        for _ in 0..total_frames {
            let frame: [i16; C] = stream.next_frame();
            let n = pipeline.process_frame(&frame, stream.current_sample - 1, &mut events);
            detected += n;
            while events.pop().is_some() {}
        }

        let rate = detected as f64 / num_spikes as f64;
        assert!(
            rate >= 0.80,
            "high firing rate detection {:.1}% is below 80% ({detected}/{num_spikes})",
            rate * 100.0
        );
    }
}

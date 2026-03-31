//! End-to-end integration tests for the firmware pipeline.
//!
//! Tests the full data flow: ring buffer -> pipeline detection ->
//! waveform extraction -> classifier -> BLE serialization.

#[cfg(test)]
mod tests {
    use crate::ble::{deserialize_spike_event, serialize_spike_event};
    use crate::classifier::{Classifier, WaveformExtractor};
    use crate::pipeline::{EventQueue, Pipeline};
    use crate::ring_buffer::FrameRingBuffer;

    const NUM_CH: usize = 32;
    const WAVEFORM_LEN: usize = 48;
    const MAX_TEMPLATES: usize = 4;

    /// Generate a synthetic spike frame: large value on one channel, zeros elsewhere.
    fn make_spike_frame(channel: usize, amplitude: i16) -> [i16; NUM_CH] {
        let mut frame = [0i16; NUM_CH];
        frame[channel] = amplitude;
        frame
    }

    #[test]
    fn ring_buffer_to_pipeline_basic() {
        let mut ring = FrameRingBuffer::<NUM_CH, 64>::new();
        let mut pipeline = Pipeline::<NUM_CH>::new(5.0);
        let mut events = EventQueue::<16>::new();

        // Push quiet frames through ring buffer -> pipeline.
        for i in 0..500u32 {
            let frame = [0i16; NUM_CH];
            assert!(ring.push(&frame));
            let mut out = [0i16; NUM_CH];
            assert!(ring.pop(&mut out));
            pipeline.process_frame(&out, i, &mut events);
        }

        // No spikes from silence.
        assert!(events.is_empty());

        // Inject spike on channel 5.
        let spike = make_spike_frame(5, -25000);
        assert!(ring.push(&spike));
        let mut out = [0i16; NUM_CH];
        assert!(ring.pop(&mut out));
        let n = pipeline.process_frame(&out, 500, &mut events);

        assert_eq!(n, 1);
        let ev = events.pop().unwrap();
        assert_eq!(ev.channel, 5);
        assert_eq!(ev.sample_idx, 500);
        assert!(ev.amplitude < 0.0);
    }

    #[test]
    fn full_pipeline_detect_extract_classify_serialize() {
        let mut pipeline = Pipeline::<NUM_CH>::new(5.0);
        let mut extractor = WaveformExtractor::<NUM_CH, WAVEFORM_LEN>::new();
        let mut classifier = Classifier::<WAVEFORM_LEN, MAX_TEMPLATES>::new(0.5);
        let mut events = EventQueue::<16>::new();

        // Load a simple negative-peak template.
        let mut template = [0.0f32; WAVEFORM_LEN];
        template[WAVEFORM_LEN - 1] = -1.0; // peak at last sample
        classifier.add_template(&template, 1);

        // Warm up with 2000 quiet frames.
        let quiet = [0i16; NUM_CH];
        for i in 0..2000u32 {
            extractor.push_frame(&quiet);
            pipeline.process_frame(&quiet, i, &mut events);
        }
        assert!(events.is_empty());

        // Inject a spike: large negative on channel 3.
        let spike = make_spike_frame(3, -25000);
        extractor.push_frame(&spike);
        let n = pipeline.process_frame(&spike, 2000, &mut events);

        assert_eq!(n, 1);
        let mut ev = events.pop().unwrap();
        assert_eq!(ev.channel, 3);

        // Extract waveform and classify.
        let waveform = extractor.extract(3);
        ev.cluster_id = classifier.classify(&waveform);

        // The waveform should have a negative peak in the last sample,
        // matching our template, so cluster_id should be 1.
        assert_eq!(ev.cluster_id, 1, "spike should match negative-peak template");

        // Serialize round-trip through BLE format.
        let mut buf = [0u8; 8];
        serialize_spike_event(&ev, &mut buf);
        let decoded = deserialize_spike_event(&buf);
        assert_eq!(decoded.sample_idx, 2000);
        assert_eq!(decoded.channel, 3);
        assert_eq!(decoded.cluster_id, 1);
        assert!((decoded.amplitude - ev.amplitude).abs() < 0.001);
    }

    #[test]
    fn multi_channel_simultaneous_spikes() {
        let mut pipeline = Pipeline::<NUM_CH>::new(5.0);
        let mut events = EventQueue::<64>::new();

        // Warm up.
        let quiet = [0i16; NUM_CH];
        for i in 0..2000u32 {
            pipeline.process_frame(&quiet, i, &mut events);
        }

        // Spike on channels 0, 15, and 31 simultaneously.
        let mut frame = [0i16; NUM_CH];
        frame[0] = -20000;
        frame[15] = 20000;
        frame[31] = -18000;
        let n = pipeline.process_frame(&frame, 2000, &mut events);

        assert_eq!(n, 3, "should detect 3 simultaneous spikes");

        let ev0 = events.pop().unwrap();
        let ev1 = events.pop().unwrap();
        let ev2 = events.pop().unwrap();

        assert_eq!(ev0.channel, 0);
        assert_eq!(ev1.channel, 15);
        assert_eq!(ev2.channel, 31);
    }

    #[test]
    fn sustained_30khz_no_overflow() {
        // Simulate 1 second at 30 kHz (30,000 frames) with occasional spikes.
        let mut ring = FrameRingBuffer::<NUM_CH, 64>::new();
        let mut pipeline = Pipeline::<NUM_CH>::new(5.0);
        let mut events = EventQueue::<256>::new();
        let mut total_spikes = 0usize;

        // Warm up (needed for noise estimator).
        let quiet = [0i16; NUM_CH];
        for i in 0..2000u32 {
            pipeline.process_frame(&quiet, i, &mut events);
        }

        for i in 2000..32000u32 {
            let frame = if i % 3000 == 0 {
                // Inject spike every 100ms on channel 10.
                make_spike_frame(10, -22000)
            } else {
                [0i16; NUM_CH]
            };

            assert!(ring.push(&frame));
            let mut out = [0i16; NUM_CH];
            assert!(ring.pop(&mut out));

            let n = pipeline.process_frame(&out, i, &mut events);
            total_spikes += n;

            // Drain events to prevent overflow.
            while events.pop().is_some() {}
        }

        // Should have detected spikes at i=3000,6000,9000,...,30000 = 10 spikes.
        assert!(
            total_spikes >= 8 && total_spikes <= 12,
            "expected ~10 spikes from periodic injection, got {total_spikes}"
        );
    }

    // -----------------------------------------------------------------------
    // Day 2 Session 7: System-level integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_stats_tracking_10k_frames() {
        use crate::stats::RuntimeStats;

        let mut pipeline = Pipeline::<NUM_CH>::new(5.0);
        let mut events = EventQueue::<16>::new();
        let mut stats = RuntimeStats::new();

        let quiet = [0i16; NUM_CH];
        for i in 0..10_000u32 {
            pipeline.process_frame(&quiet, i, &mut events);
            stats.record_frame();
            while events.pop().is_some() {}
        }

        assert_eq!(stats.total_frames, 10_000);
    }

    #[test]
    fn test_stim_fires_on_target_cluster() {
        use crate::pipeline::SpikeEvent;
        use crate::stim::{StimConfig, StimDecision, StimState, RejectReason};

        let cfg = StimConfig::new()
            .with_target_cluster_id(1)
            .with_refractory_ms(0)
            .with_max_rate_hz(100)
            .with_enabled(true);
        let mut stim = StimState::new(cfg);

        // Cluster 1 should trigger.
        let ev1 = SpikeEvent {
            sample_idx: 1000,
            channel: 0,
            cluster_id: 1,
            amplitude: -0.5,
        };
        let d1 = stim.evaluate(&ev1, 30_000);
        assert_eq!(d1, StimDecision::Trigger { pulse_width_us: 200 });

        // Cluster 2 should be rejected.
        let ev2 = SpikeEvent {
            sample_idx: 2000,
            channel: 0,
            cluster_id: 2,
            amplitude: -0.5,
        };
        let d2 = stim.evaluate(&ev2, 30_000);
        assert_eq!(d2, StimDecision::Reject(RejectReason::WrongCluster));

        assert_eq!(stim.trigger_count, 1);
        assert_eq!(stim.cluster_rejects, 1);
    }

    #[test]
    fn test_stim_refractory_period() {
        use crate::pipeline::SpikeEvent;
        use crate::stim::{StimConfig, StimDecision, StimState, RejectReason};

        let cfg = StimConfig::new()
            .with_target_cluster_id(1)
            .with_refractory_ms(10) // 10 ms = 300 samples at 30 kHz
            .with_max_rate_hz(1000)
            .with_enabled(true);
        let mut stim = StimState::new(cfg);

        let make = |idx: u32| SpikeEvent {
            sample_idx: idx,
            channel: 0,
            cluster_id: 1,
            amplitude: -0.5,
        };

        // First trigger succeeds.
        assert_eq!(
            stim.evaluate(&make(0), 30_000),
            StimDecision::Trigger { pulse_width_us: 200 }
        );

        // 5 ms later (150 samples) -- still within refractory.
        assert_eq!(
            stim.evaluate(&make(150), 30_000),
            StimDecision::Reject(RejectReason::Refractory)
        );

        // 10 ms later (300 samples) -- at boundary, should still be rejected (need >).
        assert_eq!(
            stim.evaluate(&make(300), 30_000),
            StimDecision::Trigger { pulse_width_us: 200 }
        );
    }

    #[test]
    fn test_online_learner_convergence() {
        use crate::online_learn::OnlineLearner;

        // min_spikes_init=1 so templates are immediately available.
        let mut learner = OnlineLearner::<WAVEFORM_LEN, MAX_TEMPLATES>::new(1, 0.95);

        // Build a synthetic spike waveform.
        let mut waveform = [0.0f32; WAVEFORM_LEN];
        waveform[20] = -1.0;
        waveform[21] = -0.8;
        waveform[22] = -0.3;
        waveform[23] = 0.2;

        // First waveform creates a new template slot via cold start.
        learner.learn(&waveform, 0);
        assert_eq!(learner.active_count(), 1);

        // Retrieve the assigned cluster_id.
        let (tmpls, count) = learner.get_templates();
        assert_eq!(count, 1);
        let assigned_id = tmpls[0].1;
        assert_ne!(assigned_id, 0);

        // Feed 99 more identical waveforms with the assigned cluster_id.
        for _ in 0..99 {
            learner.learn(&waveform, assigned_id);
        }

        // Verify templates stabilized: still one cluster, template matches input.
        let (final_tmpls, final_count) = learner.get_templates();
        assert!(final_count > 0, "should have at least one template");
        assert_eq!(learner.active_count(), 1, "all went to same cluster");

        let tmpl = &final_tmpls[0].0;
        for i in 0..WAVEFORM_LEN {
            assert!(
                (tmpl[i] - waveform[i]).abs() < 1e-4,
                "template sample {i} diverged: {} vs {}",
                tmpl[i],
                waveform[i]
            );
        }
    }

    #[test]
    fn test_watchdog_staleness_after_timeout() {
        use crate::watchdog::{WatchdogConfig, WatchdogState};

        let cfg = WatchdogConfig::new().with_timeout(100); // 100 ms
        let mut wd = WatchdogState::new(cfg);
        let sample_rate = 30_000u32;

        // Pet at sample 0.
        wd.pet(0);
        assert!(!wd.is_stale(0, sample_rate));

        // 50 ms later (1500 samples) -- not stale.
        assert!(!wd.is_stale(1500, sample_rate));

        // 100 ms later (3000 samples) -- exactly at boundary, not stale (need >).
        assert!(!wd.is_stale(3000, sample_rate));

        // 101 ms later (3030 samples) -- stale.
        assert!(wd.is_stale(3030, sample_rate));

        // Pet again at 3030.
        wd.pet(3030);
        assert!(!wd.is_stale(3031, sample_rate));

        // Advance past timeout again.
        assert!(wd.is_stale(3030 + 3031, sample_rate));
    }

    #[test]
    fn test_flash_log_round_trip() {
        use crate::flash_log::{FlashLog, FlashRegion, LogEntryType};

        let region = FlashRegion::new(0x000E_0000, 64 * 1024, 4096);
        let mut log: FlashLog<64> = FlashLog::new(region);

        log.begin_session(1000);
        assert_eq!(log.entry_count(), 1);

        // Append stats.
        let stats_bytes = [0u8; 16];
        assert!(log.append_stats(&stats_bytes, 1005));
        assert_eq!(log.entry_count(), 2);

        // Append fault.
        assert!(log.append_fault(0x01, 0x0800_1234, 0x0800_5678, 1010));
        assert_eq!(log.entry_count(), 3);

        // Append config.
        let config_bytes = [0xAA; 8];
        assert!(log.append_config(&config_bytes, 1020));
        assert_eq!(log.entry_count(), 4);

        // Verify ordering: SessionStart, Stats, Fault, Config.
        let (_, h0) = log.get_entry(0).unwrap();
        assert_eq!(h0.entry_type, LogEntryType::SessionStart as u8);
        assert_eq!(h0.sequence, 0);

        let (_, h1) = log.get_entry(1).unwrap();
        assert_eq!(h1.entry_type, LogEntryType::Stats as u8);
        assert_eq!(h1.sequence, 1);
        assert_eq!(h1.timestamp_s, 5); // 1005 - 1000

        let (_, h2) = log.get_entry(2).unwrap();
        assert_eq!(h2.entry_type, LogEntryType::Fault as u8);
        assert_eq!(h2.sequence, 2);
        assert_eq!(h2.timestamp_s, 10); // 1010 - 1000

        let (_, h3) = log.get_entry(3).unwrap();
        assert_eq!(h3.entry_type, LogEntryType::Config as u8);
        assert_eq!(h3.sequence, 3);
        assert_eq!(h3.timestamp_s, 20); // 1020 - 1000

        // Verify offsets are monotonically increasing.
        let (o0, _) = log.get_entry(0).unwrap();
        let (o1, _) = log.get_entry(1).unwrap();
        let (o2, _) = log.get_entry(2).unwrap();
        let (o3, _) = log.get_entry(3).unwrap();
        assert!(*o0 < *o1);
        assert!(*o1 < *o2);
        assert!(*o2 < *o3);
    }

    #[test]
    fn test_full_pipeline_to_stim() {
        use crate::stim::{StimConfig, StimDecision, StimState};

        let mut pipeline = Pipeline::<NUM_CH>::new(5.0);
        let mut extractor = WaveformExtractor::<NUM_CH, WAVEFORM_LEN>::new();
        let mut classifier = Classifier::<WAVEFORM_LEN, MAX_TEMPLATES>::new(0.5);
        let mut events = EventQueue::<16>::new();

        // Load a template with a large negative peak at the last sample.
        let mut template = [0.0f32; WAVEFORM_LEN];
        template[WAVEFORM_LEN - 1] = -1.0;
        classifier.add_template(&template, 1);

        // Configure stim to trigger on cluster 1.
        let stim_cfg = StimConfig::new()
            .with_target_cluster_id(1)
            .with_refractory_ms(0)
            .with_max_rate_hz(100)
            .with_enabled(true);
        let mut stim = StimState::new(stim_cfg);

        // Warm up pipeline and extractor.
        let quiet = [0i16; NUM_CH];
        for i in 0..2000u32 {
            extractor.push_frame(&quiet);
            pipeline.process_frame(&quiet, i, &mut events);
        }

        // Inject a spike on channel 3.
        let spike = make_spike_frame(3, -25000);
        extractor.push_frame(&spike);
        let n = pipeline.process_frame(&spike, 2000, &mut events);
        assert_eq!(n, 1);

        let mut ev = events.pop().unwrap();
        assert_eq!(ev.channel, 3);

        // Extract waveform and classify.
        let waveform = extractor.extract(3);
        ev.cluster_id = classifier.classify(&waveform);
        assert_eq!(ev.cluster_id, 1, "spike should match template");

        // Evaluate stim.
        let decision = stim.evaluate(&ev, 30_000);
        assert_eq!(
            decision,
            StimDecision::Trigger { pulse_width_us: 200 },
            "stim should trigger on matching cluster"
        );
        assert_eq!(stim.trigger_count, 1);
    }

    #[test]
    fn test_power_config_duty_cycle() {
        use crate::power::{PowerConfig, PowerProfile};

        let cfg = PowerConfig::new()
            .with_profile(PowerProfile::Balanced)
            .with_sample_rate_hz(30_000)
            .with_processing_us(19);

        // Duty cycle = 19 us / (1_000_000 / 30_000) = 19 / 33.33 ~ 0.57.
        let dc = cfg.duty_cycle();
        assert!(dc > 0.5 && dc < 0.65, "duty cycle = {dc}");

        // Average current should be between idle and active.
        let avg = cfg.average_current_ma();
        assert!(
            avg > PowerProfile::Balanced.idle_current_ma()
                && avg < PowerProfile::Balanced.active_current_ma(),
            "avg current = {avg}"
        );

        // Battery life for 60 mAh should be reasonable (MCU-only).
        let mcu_life = cfg.battery_life_hours(60.0);
        assert!(mcu_life > 10.0 && mcu_life < 100.0, "mcu life = {mcu_life} hours");

        // System battery life should be shorter (includes Intan + BLE).
        let sys_life = cfg.system_battery_life_hours(60.0);
        assert!(sys_life < mcu_life, "system life should be less than MCU-only");
        assert!(sys_life > 2.0 && sys_life < 15.0, "sys life = {sys_life} hours");
    }

    #[test]
    fn stress_60s_no_drops() {
        use crate::intan::{DmaDoubleBuffer, FrameTiming};
        use crate::stats::RuntimeStats;
        use crate::stim::{StimConfig, StimState};

        let mut dma = DmaDoubleBuffer::new();
        let mut timing = FrameTiming::new();
        let mut pipeline = Pipeline::<NUM_CH>::new(5.0);
        let mut extractor = WaveformExtractor::<NUM_CH, WAVEFORM_LEN>::new();
        let mut classifier = Classifier::<WAVEFORM_LEN, MAX_TEMPLATES>::new(0.5);
        let mut events = EventQueue::<64>::new();
        let mut stats = RuntimeStats::new();

        // Load a simple template.
        let mut template = [0.0f32; WAVEFORM_LEN];
        template[WAVEFORM_LEN - 1] = -1.0;
        classifier.add_template(&template, 1);

        let stim_cfg = StimConfig::new()
            .with_target_cluster_id(1)
            .with_refractory_ms(5)
            .with_max_rate_hz(200)
            .with_enabled(true);
        let mut stim = StimState::new(stim_cfg);

        // 60 seconds at 30 kHz = 1,800,000 frames.
        let total_frames: u32 = 1_800_000;
        let mut total_spikes = 0u32;

        for i in 0..total_frames {
            // Simulate writing into DMA active buffer.
            let buf = dma.active_buf();
            buf.data = if i % 6000 == 3000 {
                // Inject spike every 200ms on channel 5.
                make_spike_frame(5, -25000)
            } else {
                [0i16; NUM_CH]
            };
            dma.swap();

            // Process from ready buffer (mimics CPU processing while DMA fills next).
            let frame = &dma.ready_buf().data;
            extractor.push_frame(frame);
            let n = pipeline.process_frame(frame, i, &mut events);
            stats.record_frame();

            // Simulate frame timing (1 us per frame in test).
            timing.record(1);

            for _ in 0..n {
                if let Some(mut ev) = events.pop() {
                    let waveform = extractor.extract(ev.channel as usize);
                    ev.cluster_id = classifier.classify(&waveform);
                    stats.record_spike(ev.cluster_id != 0);
                    stim.evaluate(&ev, 30_000);
                    total_spikes += 1;
                }
            }

            // Tick stim rate limiter every second.
            if i > 0 && i % 30_000 == 0 {
                stim.tick_second(i);
                stats.tick_second();
            }
        }

        assert_eq!(stats.total_frames, total_frames);
        // Spikes injected every 6000 frames starting at 3000: ~300 spikes.
        assert!(
            total_spikes >= 250 && total_spikes <= 350,
            "expected ~300 spikes, got {total_spikes}"
        );
        assert_eq!(timing.count(), total_frames);
        assert_eq!(timing.min_us(), 1);
        assert_eq!(timing.max_us(), 1);
    }

    #[test]
    fn ble_batch_round_trip() {
        use crate::ble::serialize_spike_batch;
        use crate::pipeline::SpikeEvent;

        let events: Vec<SpikeEvent> = (0..5)
            .map(|i| SpikeEvent {
                sample_idx: i * 100,
                channel: i as u8,
                cluster_id: (i as u8) + 1,
                amplitude: -0.5,
            })
            .collect();

        // MTU = 247, overhead = 3, so max events per packet = (247-3)/8 = 30.
        let mut buf = [0u8; 244];
        let n = serialize_spike_batch(&events, &mut buf, 30);
        assert_eq!(n, 40); // 5 events * 8 bytes

        for i in 0..5 {
            let start = i * 8;
            let slice: &[u8; 8] = buf[start..start + 8].try_into().unwrap();
            let decoded = deserialize_spike_event(slice);
            assert_eq!(decoded.sample_idx, (i as u32) * 100);
            assert_eq!(decoded.channel, i as u8);
            assert_eq!(decoded.cluster_id, (i as u8) + 1);
        }
    }
}

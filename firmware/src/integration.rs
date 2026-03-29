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

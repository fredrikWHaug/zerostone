#![no_std]
#![no_main]

//! Zerostone firmware entry point.
//!
//! Orchestrates the full spike sorting pipeline as Embassy async tasks:
//!
//! ```text
//! [30kHz Timer] -> [SPI Task] -> [Channel] -> [Processing Task] -> [Channel] -> [BLE Task]
//!                   reads Intan    frames       detect + classify     events      serialize
//!                                                    |
//!                                              [Stats Channel] -> [Stats Task]
//! ```

use embassy_executor::Spawner;
use embassy_nrf::gpio::{Level, Output, OutputDrive};
use embassy_nrf::spim::{self, Spim};
use embassy_nrf::{bind_interrupts, peripherals};
use embassy_sync::blocking_mutex::raw::NoopRawMutex;
use embassy_sync::channel::Channel;
use embassy_time::{Ticker, Timer};
use embedded_hal_bus::spi::ExclusiveDevice;
use embassy_time::Delay;
use static_cell::StaticCell;
use {defmt_rtt as _, panic_probe as _};

use zerostone_firmware::ble::serialize_spike_event;
use zerostone_firmware::ble_server::BleServer;
use zerostone_firmware::classifier::{Classifier, WaveformExtractor};
use zerostone_firmware::fault::FaultLog;
use zerostone_firmware::intan::{DmaDoubleBuffer, IntanDriver, NUM_CHANNELS};
use zerostone_firmware::online_learn::OnlineLearner;
use zerostone_firmware::pipeline::{EventQueue, Pipeline, SpikeEvent};
use zerostone_firmware::stats::RuntimeStats;
use zerostone_firmware::watchdog::WatchdogConfig;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Waveform snippet length in samples (~1.6 ms at 30 kHz).
const WAVEFORM_LEN: usize = 48;

/// Maximum number of templates in the classifier.
const MAX_TEMPLATES: usize = 8;

/// MAD multiplier for spike detection threshold.
const THRESHOLD_FACTOR: f32 = 5.0;

/// Minimum normalized cross-correlation for template match.
const MIN_CORRELATION: f32 = 0.7;

/// Frame channel capacity between SPI and processing tasks.
const FRAME_CHANNEL_DEPTH: usize = 64;

/// Event channel capacity between processing and BLE tasks.
const EVENT_CHANNEL_DEPTH: usize = 32;

/// Stats channel capacity between processing and stats tasks.
const STATS_CHANNEL_DEPTH: usize = 4;

/// Number of frames between stats snapshots (~1 second at 30 kHz).
const STATS_INTERVAL_FRAMES: u32 = 30_000;

/// Number of frames between online learner merge attempts.
const LEARN_MERGE_INTERVAL: u32 = 10_000;

/// Minimum spike count before a learned template is used.
const MIN_SPIKES_INIT: u32 = 50;

/// NCC threshold for merging similar learned templates.
const LEARN_MERGE_THRESHOLD: f32 = 0.90;

// ---------------------------------------------------------------------------
// Shared channels (static lifetime for Embassy tasks)
// ---------------------------------------------------------------------------

/// Channel for ADC frames: SPI task -> Processing task.
static FRAME_CHANNEL: StaticCell<Channel<NoopRawMutex, [i16; NUM_CHANNELS], FRAME_CHANNEL_DEPTH>> =
    StaticCell::new();

/// Channel for classified spike events: Processing task -> BLE task.
static EVENT_CHANNEL: StaticCell<Channel<NoopRawMutex, SpikeEvent, EVENT_CHANNEL_DEPTH>> =
    StaticCell::new();

/// Channel for serialized stats snapshots: Processing task -> Stats task.
/// Each snapshot is 16 bytes produced by RuntimeStats::serialize().
static STATS_CHANNEL: StaticCell<Channel<NoopRawMutex, [u8; 16], STATS_CHANNEL_DEPTH>> =
    StaticCell::new();

// ---------------------------------------------------------------------------
// Interrupt bindings
// ---------------------------------------------------------------------------

bind_interrupts!(struct Irqs {
    SERIAL0 => spim::InterruptHandler<peripherals::SERIAL0>;
});

// ---------------------------------------------------------------------------
// Type alias for the SPI device (SpiBus + CS pin + delay)
// ---------------------------------------------------------------------------

type SpiDev = ExclusiveDevice<Spim<'static, peripherals::SERIAL0>, Output<'static>, Delay>;

// ---------------------------------------------------------------------------
// SPI sampling task — runs at 30 kHz
// ---------------------------------------------------------------------------

#[embassy_executor::task]
async fn spi_task(
    spi_dev: SpiDev,
    frame_tx: &'static Channel<NoopRawMutex, [i16; NUM_CHANNELS], FRAME_CHANNEL_DEPTH>,
) {
    let mut intan = IntanDriver::new(spi_dev);

    // Initialize the Intan RHD2132 (bandwidth config + ADC calibration).
    match intan.init().await {
        Ok(()) => defmt::info!("intan: initialized"),
        Err(_e) => {
            defmt::error!("intan: init failed");
            return;
        }
    }

    // DMA double buffer: DMA fills one while CPU processes the other.
    let mut dma_buf = DmaDoubleBuffer::new();

    // 30 kHz = 33.333... us per sample.
    let mut ticker = Ticker::every(embassy_time::Duration::from_micros(33));
    let mut dropped: u32 = 0;

    loop {
        ticker.next().await;

        match intan.read_frame_dma(dma_buf.active_buf()).await {
            Ok(()) => {
                dma_buf.swap();
                // Send the just-completed buffer's data to the processing task.
                if frame_tx.try_send(dma_buf.ready_buf().data).is_err() {
                    dropped += 1;
                    if dropped % 1000 == 0 {
                        defmt::warn!("spi: {} frames dropped (channel full)", dropped);
                    }
                }
            }
            Err(_e) => {
                defmt::error!("spi: read_frame_dma failed");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Processing task — spike detection + classification + online learning
// ---------------------------------------------------------------------------

#[embassy_executor::task]
async fn processing_task(
    frame_rx: &'static Channel<NoopRawMutex, [i16; NUM_CHANNELS], FRAME_CHANNEL_DEPTH>,
    event_tx: &'static Channel<NoopRawMutex, SpikeEvent, EVENT_CHANNEL_DEPTH>,
    stats_tx: &'static Channel<NoopRawMutex, [u8; 16], STATS_CHANNEL_DEPTH>,
) {
    // Stack-allocated pipeline and classifier.
    let mut pipeline = Pipeline::<NUM_CHANNELS>::new(THRESHOLD_FACTOR);
    let mut extractor = WaveformExtractor::<NUM_CHANNELS, WAVEFORM_LEN>::new();
    let mut classifier = Classifier::<WAVEFORM_LEN, MAX_TEMPLATES>::new(MIN_CORRELATION);

    // Runtime statistics tracker.
    let mut stats = RuntimeStats::new();

    // Online template learner.
    let mut learner =
        OnlineLearner::<WAVEFORM_LEN, MAX_TEMPLATES>::new(MIN_SPIKES_INIT, LEARN_MERGE_THRESHOLD);

    // Load placeholder templates so classification has something to match.
    // Template 1: canonical negative-first biphasic spike.
    let mut biphasic_neg: [f32; WAVEFORM_LEN] = [0.0; WAVEFORM_LEN];
    biphasic_neg[16] = -0.25;
    biphasic_neg[17] = -0.50;
    biphasic_neg[18] = -0.75;
    biphasic_neg[19] = -0.90;
    biphasic_neg[20] = -1.00;
    biphasic_neg[21] = -0.80;
    biphasic_neg[22] = -0.50;
    biphasic_neg[23] = -0.20;
    biphasic_neg[24] = 0.10;
    biphasic_neg[25] = 0.25;
    biphasic_neg[26] = 0.40;
    biphasic_neg[27] = 0.50;
    biphasic_neg[28] = 0.40;
    biphasic_neg[29] = 0.25;
    biphasic_neg[30] = 0.10;
    classifier.add_template(&biphasic_neg, 1);

    // Template 2: positive-first biphasic spike (inverted template 1).
    let mut biphasic_pos: [f32; WAVEFORM_LEN] = [0.0; WAVEFORM_LEN];
    let mut i = 0;
    while i < WAVEFORM_LEN {
        biphasic_pos[i] = -biphasic_neg[i];
        i += 1;
    }
    classifier.add_template(&biphasic_pos, 2);

    defmt::info!("processing: pipeline ready, {} templates loaded", 2);

    let mut sample_idx: u32 = 0;
    let mut event_queue = EventQueue::<64>::new();

    loop {
        // Block until a frame arrives from the SPI task.
        let frame = frame_rx.receive().await;

        // Record frame for statistics.
        stats.record_frame();

        // Feed the waveform extractor (must happen before detection so the
        // buffer contains the spike waveform when we extract it).
        extractor.push_frame(&frame);

        // Run spike detection.
        let n_spikes = pipeline.process_frame(&frame, sample_idx, &mut event_queue);

        // For each detected spike, extract waveform and classify.
        for _ in 0..n_spikes {
            if let Some(mut event) = event_queue.pop() {
                let waveform = extractor.extract(event.channel as usize);
                event.cluster_id = classifier.classify(&waveform);

                let classified = event.cluster_id != 0;
                stats.record_spike(classified);

                // Feed waveform to the online learner.
                learner.learn(&waveform, event.cluster_id);

                // Forward classified event to BLE task.
                if event_tx.try_send(event).is_err() {
                    stats.record_dropped_event();
                    defmt::warn!("processing: event channel full, spike dropped");
                }

                defmt::info!(
                    "spike: ch={} idx={} cluster={} amp={}",
                    event.channel,
                    event.sample_idx,
                    event.cluster_id,
                    event.amplitude,
                );
            }
        }

        // Periodically attempt template merge and reload.
        if sample_idx > 0 && sample_idx % LEARN_MERGE_INTERVAL == 0 {
            learner.try_merge();

            let (templates, n_templates) = learner.get_templates();
            if n_templates > 0 {
                // Rebuild classifier with learned templates.
                let mut new_clf =
                    Classifier::<WAVEFORM_LEN, MAX_TEMPLATES>::new(MIN_CORRELATION);
                let mut loaded = 0usize;
                let mut t = 0;
                while t < n_templates {
                    new_clf.add_template(&templates[t].0, templates[t].1);
                    loaded += 1;
                    t += 1;
                }
                classifier = new_clf;
                defmt::info!(
                    "learning: reloaded classifier with {} learned templates",
                    loaded,
                );
            }
        }

        // Every STATS_INTERVAL_FRAMES, tick the second counter and send snapshot.
        if sample_idx > 0 && sample_idx % STATS_INTERVAL_FRAMES == 0 {
            stats.tick_second();
            let mut buf = [0u8; 16];
            stats.serialize(&mut buf);
            // Best-effort send; drop if stats_task is behind.
            let _ = stats_tx.try_send(buf);
        }

        sample_idx = sample_idx.wrapping_add(1);
    }
}

// ---------------------------------------------------------------------------
// BLE output task — serialize + notify via BleServer state machine
// ---------------------------------------------------------------------------

#[embassy_executor::task]
async fn ble_task(
    event_rx: &'static Channel<NoopRawMutex, SpikeEvent, EVENT_CHANNEL_DEPTH>,
) {
    let mut server = BleServer::<8>::new();
    server.start_advertising();
    defmt::info!(
        "ble: server initialized, {} characteristics, state=advertising",
        server.table.count(),
    );

    let mut total_sent: u32 = 0;

    loop {
        let event = event_rx.receive().await;

        // Write event into the GATT table via BleServer.
        server.push_spike_event(&event);

        // Also serialize for logging.
        let mut buf = [0u8; 8];
        serialize_spike_event(&event, &mut buf);

        total_sent += 1;

        defmt::debug!(
            "ble: tx event #{} ch={} cluster={} [{=[u8]:x}]",
            total_sent,
            event.channel,
            event.cluster_id,
            buf,
        );

        // TODO: When BLE radio is connected, issue a GATT notification
        // if server.is_notify_enabled().
    }
}

// ---------------------------------------------------------------------------
// Stats reporting task — receives serialized snapshots, logs via defmt
// ---------------------------------------------------------------------------

#[embassy_executor::task]
async fn stats_task(
    stats_rx: &'static Channel<NoopRawMutex, [u8; 16], STATS_CHANNEL_DEPTH>,
) {
    defmt::info!("stats: task started");

    loop {
        let buf = stats_rx.receive().await;

        let total_frames = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
        let total_spikes = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
        let dropped_frames = u16::from_le_bytes([buf[8], buf[9]]);
        let dropped_events = u16::from_le_bytes([buf[10], buf[11]]);
        let peak_spike_rate = u16::from_le_bytes([buf[12], buf[13]]);
        let uptime_s = u16::from_le_bytes([buf[14], buf[15]]);

        defmt::info!(
            "stats: frames={} spikes={} dropped_f={} dropped_e={} peak_rate={} uptime={}s",
            total_frames,
            total_spikes,
            dropped_frames,
            dropped_events,
            peak_spike_rate,
            uptime_s,
        );
    }
}

// ---------------------------------------------------------------------------
// LED heartbeat task — 1 Hz blink
// ---------------------------------------------------------------------------

#[embassy_executor::task]
async fn heartbeat_task(mut led: Output<'static>) {
    loop {
        led.set_high();
        Timer::after_millis(500).await;
        led.set_low();
        Timer::after_millis(500).await;
    }
}

// ---------------------------------------------------------------------------
// Main — initialize peripherals and spawn tasks
// ---------------------------------------------------------------------------

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_nrf::init(Default::default());

    defmt::info!("zerostone firmware starting");

    // --- Log watchdog configuration at boot ---
    let wdt_cfg = WatchdogConfig::new();
    defmt::info!(
        "watchdog: timeout={}ms pause_sleep={} pause_debug={}",
        wdt_cfg.timeout_ms,
        wdt_cfg.pause_on_sleep,
        wdt_cfg.pause_on_debug,
    );

    // --- Initialize fault log (deferred HardFault handler integration) ---
    let _fault_log: FaultLog<8> = FaultLog::new();
    defmt::info!("fault: log initialized, capacity=8");

    // --- LED heartbeat on P0.28 ---
    let led = Output::new(p.P0_28, Level::Low, OutputDrive::Standard);
    spawner.must_spawn(heartbeat_task(led));

    // --- SPI for Intan RHD2132 ---
    // Pin assignment: SCK=P0.13, MOSI=P0.14, MISO=P0.15, CS=P0.16
    let mut spi_config = spim::Config::default();
    spi_config.frequency = spim::Frequency::M8; // 8 MHz SPI clock
    spi_config.mode = spim::MODE_0; // CPOL=0, CPHA=0 (Intan default)

    let spi_bus = Spim::new(
        p.SERIAL0,
        Irqs,
        p.P0_13, // SCK
        p.P0_15, // MISO
        p.P0_14, // MOSI
        spi_config,
    );

    // Wrap SpiBus + CS pin into a SpiDevice (manages CS assertion).
    let cs_pin = Output::new(p.P0_16, Level::High, OutputDrive::Standard);
    let spi_dev = ExclusiveDevice::new(spi_bus, cs_pin, Delay).unwrap();

    // --- Initialize shared channels ---
    let frame_channel = FRAME_CHANNEL.init(Channel::new());
    let event_channel = EVENT_CHANNEL.init(Channel::new());
    let stats_channel = STATS_CHANNEL.init(Channel::new());

    // --- Spawn tasks ---
    spawner.must_spawn(spi_task(spi_dev, frame_channel));
    spawner.must_spawn(processing_task(frame_channel, event_channel, stats_channel));
    spawner.must_spawn(ble_task(event_channel));
    spawner.must_spawn(stats_task(stats_channel));

    defmt::info!("zerostone: all tasks spawned");
}

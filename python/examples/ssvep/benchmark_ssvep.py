"""SSVEP Detection Benchmark on Real Physionet Data.

Validates zpybci.ssvep_detect on the MAMEM SSVEP Database (Dataset 3)
from Physionet. Dataset 3 uses the Emotiv EPOC headset (14 channels,
128 Hz) with 5 simultaneous flickering frequencies.

Dataset: https://physionet.org/content/mssvepdb/1.0.0/
Frequencies: 6.66, 7.50, 8.57, 10.00, 12.00 Hz

Trial boundaries and target frequencies are read from the .win
annotation files provided with each record.

Usage:
    python examples/ssvep/benchmark_ssvep.py
"""
import sys

import numpy as np

try:
    import wfdb
except ImportError:
    print("ERROR: wfdb is required. Install with: pip install wfdb")
    sys.exit(1)

import zpybci as zbci

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FS = 128.0
STIM_FREQS = [6.66, 7.50, 8.57, 10.00, 12.00]
N_CLASSES = len(STIM_FREQS)
CHANCE = 1.0 / N_CLASSES  # 20%
PN_DIR = "mssvepdb/1.0.0/dataset3"

# Emotiv EPOC channel order (14 channels):
# 0:AF3 1:F7 2:F3 3:FC5 4:T7 5:P7 6:O1 7:O2 8:P8 9:T8 10:FC6 11:F4 12:F8 13:AF4
#
# For SSVEP, occipital and parietal channels are most relevant.
# Select 8 posterior/temporal channels: P7, O1, O2, P8, T7, T8, FC5, FC6
SSVEP_CH = [5, 6, 7, 8, 4, 9, 3, 10]
N_CHANNELS = len(SSVEP_CH)

# Records: sessions a-e, repetitions i and ii
SESSIONS = ["a", "b", "c", "d", "e"]
REPS = ["i", "ii"]

# Skip the first 0.5s of each trial to avoid onset transient
ONSET_SKIP_S = 0.5
ONSET_SKIP = int(ONSET_SKIP_S * FS)

# Bandpass filter range (covers all SSVEP frequencies + harmonics)
BP_LOW = 5.0
BP_HIGH = 30.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trials(subject_id, session, rep):
    """Load trials from a single WFDB record using .win annotations.

    Returns list of (segment, target_freq) pairs, or empty list on error.
    """
    record_name = f"U{subject_id:03d}{session}{rep}"
    try:
        rec = wfdb.rdrecord(record_name, pn_dir=PN_DIR)
        ann = wfdb.rdann(record_name, "win", pn_dir=PN_DIR)
    except Exception:
        return []

    signal_raw = rec.p_signal[:, :14].astype(np.float32)  # (n_samples, 14)

    # Bandpass filter each channel to isolate SSVEP band
    filters = [zbci.IirFilter.butterworth_bandpass(FS, BP_LOW, BP_HIGH)
               for _ in range(14)]
    signal_filt = np.empty_like(signal_raw)
    for ch in range(14):
        signal_filt[:, ch] = filters[ch].process(
            np.ascontiguousarray(signal_raw[:, ch]))
    signal = signal_filt.astype(np.float64)

    n_annotations = len(ann.sample)
    trials = []

    # Annotations come in pairs: '(' = trial start, ')' = trial end
    for i in range(0, n_annotations - 1, 2):
        if ann.symbol[i] != "(" or ann.symbol[i + 1] != ")":
            continue

        start = ann.sample[i] + ONSET_SKIP
        end = ann.sample[i + 1]
        if start >= end or end > signal.shape[0]:
            continue

        freq_str = ann.aux_note[i]
        try:
            target_freq = float(freq_str)
        except ValueError:
            continue

        # Check that this frequency is one we expect
        if not any(abs(target_freq - f) < 0.1 for f in STIM_FREQS):
            continue

        # Find closest matching frequency
        target_freq = min(STIM_FREQS, key=lambda f: abs(f - target_freq))

        segment = signal[start:end, :][:, SSVEP_CH]
        trials.append((segment, target_freq))

    return trials


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("SSVEP Detection Benchmark -- MAMEM Dataset 3 (Physionet)")
    print(f"Channels: {N_CHANNELS}, Sample rate: {FS} Hz")
    print(f"Frequencies: {STIM_FREQS} Hz")
    print(f"Onset skip: {ONSET_SKIP_S}s per trial")
    print()

    all_detected = []
    all_targets = []
    subject_accuracies = []

    for sid in range(1, 12):
        subject_detected = []
        subject_targets = []
        n_loaded = 0

        print(f"Subject {sid:03d}:", end=" ", flush=True)

        for session in SESSIONS:
            for rep in REPS:
                trials = load_trials(sid, session, rep)
                n_loaded += len(trials)

                for segment, target_freq in trials:
                    target_idx = STIM_FREQS.index(target_freq)
                    detected_idx, correlation = zbci.ssvep_detect(
                        segment, FS, STIM_FREQS, n_harmonics=2
                    )
                    subject_detected.append(detected_idx)
                    subject_targets.append(target_idx)

        if not subject_detected:
            print("no data loaded")
            continue

        det = np.array(subject_detected)
        tgt = np.array(subject_targets)
        acc = float(np.mean(det == tgt))
        subject_accuracies.append(acc)

        all_detected.extend(subject_detected)
        all_targets.extend(subject_targets)

        print(f"{n_loaded} trials, accuracy {acc*100:.1f}%")

    if not all_detected:
        print("\nNo data could be loaded. Check network connection.")
        return

    # -----------------------------------------------------------------------
    # Overall results
    # -----------------------------------------------------------------------
    det = np.array(all_detected)
    tgt = np.array(all_targets)
    overall_acc = float(np.mean(det == tgt))

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print()

    # Per-frequency accuracy
    print(f"{'Frequency':<12} {'Correct':<10} {'Total':<10} {'Accuracy'}")
    print("-" * 45)
    for i, freq in enumerate(STIM_FREQS):
        mask = tgt == i
        total = int(np.sum(mask))
        if total > 0:
            correct = int(np.sum((det == i) & mask))
            acc = correct / total
            print(f"{freq:<12.2f} {correct:<10} {total:<10} {acc*100:.1f}%")
        else:
            print(f"{freq:<12.2f} {'--':<10} {0:<10} {'N/A'}")

    print("-" * 45)
    total = len(tgt)
    correct = int(np.sum(det == tgt))
    print(f"{'Overall':<12} {correct:<10} {total:<10} {overall_acc*100:.1f}%")
    print()

    # Confusion matrix
    print("Confusion Matrix (rows=true, cols=predicted):")
    print()
    header = "         " + "".join(f"{f:>8.2f}" for f in STIM_FREQS)
    print(header)
    for i, freq in enumerate(STIM_FREQS):
        row = f"{freq:>8.2f} "
        mask = tgt == i
        for j in range(N_CLASSES):
            count = int(np.sum(det[mask] == j))
            row += f"{count:>8}"
        print(row)

    print()
    print(f"Chance level     : {CHANCE*100:.0f}%")
    print(f"Overall accuracy : {overall_acc*100:.1f}%")
    above = "YES" if overall_acc > CHANCE else "NO"
    print(f"Above chance     : {above}")

    if subject_accuracies:
        mean_subj = float(np.mean(subject_accuracies))
        n_above = sum(1 for a in subject_accuracies if a > CHANCE)
        print(f"Mean subject acc : {mean_subj*100:.1f}%")
        print(f"Subjects > chance: {n_above}/{len(subject_accuracies)}")

    print()
    print("Note: MAMEM-3 uses a consumer-grade Emotiv EPOC headset (14 channels,")
    print("128 Hz) with closely-spaced frequencies (6.66-12 Hz). Standard CCA")
    print("typically achieves near-chance accuracy on this dataset. Higher accuracy")
    print("requires filter-bank CCA (FBCCA) or task-related component analysis (TRCA).")


if __name__ == "__main__":
    main()

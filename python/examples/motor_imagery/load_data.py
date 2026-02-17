"""Load and epoch Physionet EEG Motor Movement/Imagery Dataset.

Dataset: EEGMMIDB — 109 subjects, 64 channels, 160 Hz.
Files are downloaded per-subject to a temp dir, read, then discarded.
No persistent disk storage.

Usage:
    python load_data.py
"""
import os
import tempfile
import numpy as np
import wfdb
import pyedflib

# Imagined left (T1) / right (T2) fist runs
IMAGERY_RUNS = [4, 8, 12]

SAMPLES_PER_TRIAL = 640  # 4 s * 160 Hz
CHANNELS = 64
PN_DB = "eegmmidb"
PN_DIR = "eegmmidb/1.0.0/"


def _load_edf_signals(edf_path: str) -> np.ndarray:
    """Read all channels from an EDF file. Returns float32 (n_samples, 64)."""
    f = pyedflib.EdfReader(edf_path)
    n_ch = f.signals_in_file
    n_samp = f.getNSamples()[0]
    signals = np.empty((n_samp, n_ch), dtype=np.float32)
    for i in range(n_ch):
        signals[:, i] = f.readSignal(i)
    f._close()
    return signals


def _load_annotations(ann_path: str) -> list:
    """Read .edf.event file; returns list of (sample, label_str) for T1/T2 events."""
    ann = wfdb.rdann(ann_path, "event")
    events = []
    for sample, note in zip(ann.sample, ann.aux_note):
        if note.startswith("T1"):
            events.append((sample, "T1"))
        elif note.startswith("T2"):
            events.append((sample, "T2"))
    return events


def stream_subject(subject_id: int) -> list:
    """Download and epoch all imagery trials for one subject.

    Returns a list of (epoch, label) tuples where:
      epoch : np.ndarray, float32, shape (640, 64)
      label : int — 0 = left fist (T1), 1 = right fist (T2)

    Files are downloaded to a temp dir and deleted when done.
    """
    trials = []

    subj_tag = f"S{subject_id:03d}"
    with tempfile.TemporaryDirectory() as tmpdir:
        # Batch-download all runs in a single dl_files call
        all_files = []
        for run in IMAGERY_RUNS:
            stem = f"{subj_tag}R{run:02d}"
            all_files.append(f"{subj_tag}/{stem}.edf")
            all_files.append(f"{subj_tag}/{stem}.edf.event")
        try:
            wfdb.dl_files(PN_DB, tmpdir, all_files)
        except Exception as exc:
            print(f"  Warning: could not download {subj_tag}: {exc}")
            return trials

        subj_dir = os.path.join(tmpdir, subj_tag)
        for run in IMAGERY_RUNS:
            stem = f"{subj_tag}R{run:02d}"
            edf_path = os.path.join(subj_dir, f"{stem}.edf")
            ann_path = os.path.join(subj_dir, f"{stem}.edf")

            try:
                signal = _load_edf_signals(edf_path)
                events = _load_annotations(ann_path)
            except Exception as exc:
                print(f"  Warning: could not read {stem}: {exc}")
                continue

            n_samples = signal.shape[0]
            for sample_idx, label_str in events:
                end = sample_idx + SAMPLES_PER_TRIAL
                if end > n_samples:
                    continue
                epoch = signal[sample_idx:end].copy()
                label = 0 if label_str == "T1" else 1
                trials.append((epoch, label))

    return trials


def load_subjects(subject_ids: list) -> tuple:
    """Load and epoch trials for a list of subject IDs.

    Returns:
      X : np.ndarray, float32, shape (n_trials, 640, 64)
      y : np.ndarray, int32,   shape (n_trials,)
    """
    all_epochs = []
    all_labels = []
    for sid in subject_ids:
        trials = stream_subject(sid)
        for epoch, label in trials:
            all_epochs.append(epoch)
            all_labels.append(label)

    X = (
        np.stack(all_epochs, axis=0)
        if all_epochs
        else np.empty((0, SAMPLES_PER_TRIAL, CHANNELS), dtype=np.float32)
    )
    y = np.array(all_labels, dtype=np.int32)
    return X, y


if __name__ == "__main__":
    print("Streaming subject S001 (runs 4, 8, 12) as a smoke test ...")
    trials = stream_subject(1)
    print(f"  Loaded {len(trials)} trials")
    if trials:
        epoch, label = trials[0]
        print(f"  First trial shape: {epoch.shape}, dtype: {epoch.dtype}, label: {label}")
        print(f"  Signal range: [{epoch.min():.4f}, {epoch.max():.4f}]")
    print("Done.")

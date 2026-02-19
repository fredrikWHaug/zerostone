"""Validate zpybci pipeline on real Physionet EEG data.

Streams EEGMMIDB for all 109 subjects, runs each trial through:
  bandpass (8-30 Hz) -> notch (60 Hz) -> CAR

Checks for NaN/Inf and saves a comparison plot of one trial.

Downloads are parallelized across subjects (I/O-bound) while pipeline
processing is sequential (filter state is per-pipeline).

Usage:
    python validate_pipeline.py
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import zpybci as zbci
from load_data import stream_subject, CHANNELS, SAMPLES_PER_TRIAL

FS = 160.0
N_SUBJECTS = 109
MAX_WORKERS = 8
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")
PLOT_PATH = os.path.join(OUTPUT_DIR, "validate_eeg.png")

PLOT_CHANNEL = 0


def build_pipeline():
    bandpass_filters = [
        zbci.IirFilter.butterworth_bandpass(FS, 8.0, 30.0) for _ in range(CHANNELS)
    ]
    notch = zbci.NotchFilter.powerline_60hz(FS, channels=CHANNELS)
    car = zbci.CAR(channels=CHANNELS)
    return bandpass_filters, notch, car


def reset_pipeline(bandpass_filters, notch):
    for f in bandpass_filters:
        f.reset()
    notch.reset()


def process_trial(trial: np.ndarray, bandpass_filters, notch, car) -> dict:
    """Run one trial through the full pipeline.

    Returns dict with keys: bandpass, notch, car — each shape (640, 64)
    """
    assert trial.shape == (SAMPLES_PER_TRIAL, CHANNELS), f"Unexpected shape: {trial.shape}"

    # Step 1: per-channel bandpass (column slices are non-contiguous; copy before process)
    bandpassed = np.empty_like(trial)
    for ch in range(CHANNELS):
        bandpassed[:, ch] = bandpass_filters[ch].process(np.ascontiguousarray(trial[:, ch]))

    # Step 2: notch (operates on full 2D block)
    denoised = notch.process(bandpassed)

    # Step 3: CAR (stateless, operates on full 2D block)
    referenced = car.process(denoised)

    return {"bandpass": bandpassed, "notch": denoised, "car": referenced}


def sanity_check(stages: dict) -> tuple:
    """Return (ok: bool, reason: str)."""
    for name, arr in stages.items():
        if arr.shape != (SAMPLES_PER_TRIAL, CHANNELS):
            return False, f"{name} wrong shape {arr.shape}"
        if not np.isfinite(arr).all():
            return False, f"{name} contains NaN or Inf"
        rms_per_ch = np.sqrt(np.mean(arr ** 2, axis=0))
        if not np.isfinite(rms_per_ch).all():
            return False, f"{name} RMS has NaN/Inf"
        if (rms_per_ch == 0).any():
            return False, f"{name} has zero-RMS channel"
    return True, "ok"


def save_plot(raw: np.ndarray, stages: dict, channel: int = PLOT_CHANNEL):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t = np.arange(SAMPLES_PER_TRIAL) / FS

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    plot_data = [
        ("Raw EEG", raw[:, channel]),
        ("After Bandpass (8-30 Hz)", stages["bandpass"][:, channel]),
        ("After Notch (60 Hz)", stages["notch"][:, channel]),
        ("After CAR", stages["car"][:, channel]),
    ]
    for ax, (title, signal) in zip(axes, plot_data):
        ax.plot(t, signal, linewidth=0.7, color="steelblue")
        ax.set_ylabel("uV")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"EEG Pipeline Validation — S001, Channel {channel}", fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {os.path.abspath(PLOT_PATH)}")


def main():
    bandpass_filters, notch, car = build_pipeline()
    subject_ids = list(range(1, N_SUBJECTS + 1))

    total_trials = 0
    total_pass = 0
    total_fail = 0
    fail_details = []
    plot_saved = False

    print(f"Validating pipeline on {N_SUBJECTS} subjects ({MAX_WORKERS} parallel downloads) ...")
    print(f"{'Subject':<10} {'Trials':<8} {'Pass':<8} {'Fail':<8}")
    print("-" * 40)
    sys.stdout.flush()

    # Phase 1: parallel download + epoch (I/O-bound)
    subject_trials = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(stream_subject, sid): sid for sid in subject_ids}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                subject_trials[sid] = future.result()
            except Exception as exc:
                print(f"  S{sid:03d} download error: {exc}")
                sys.stdout.flush()
                subject_trials[sid] = []

    # Phase 2: sequential pipeline processing (filter state is shared)
    for sid in subject_ids:
        trials = subject_trials.get(sid, [])
        if not trials:
            print(f"S{sid:03d}      {'0':<8} {'0':<8} {'0':<8}  (no data)")
            sys.stdout.flush()
            continue

        subj_pass = 0
        subj_fail = 0

        for i, (epoch, label) in enumerate(trials):
            reset_pipeline(bandpass_filters, notch)
            stages = process_trial(epoch, bandpass_filters, notch, car)
            ok, reason = sanity_check(stages)

            if ok:
                subj_pass += 1
                if not plot_saved:
                    save_plot(epoch.copy(), {k: v.copy() for k, v in stages.items()})
                    plot_saved = True
            else:
                subj_fail += 1
                fail_details.append((sid, i, reason))

        total_trials += len(trials)
        total_pass += subj_pass
        total_fail += subj_fail
        status = "OK" if subj_fail == 0 else f"FAIL({subj_fail})"
        print(f"S{sid:03d}      {len(trials):<8} {subj_pass:<8} {subj_fail:<8}  {status}")
        sys.stdout.flush()

    print()
    print("=" * 40)
    print(f"Total trials : {total_trials}")
    print(f"Passed       : {total_pass}")
    print(f"Failed       : {total_fail}")
    if fail_details:
        print("\nFailure details:")
        for sid, trial_idx, reason in fail_details[:10]:
            print(f"  S{sid:03d} trial {trial_idx}: {reason}")
        if len(fail_details) > 10:
            print(f"  ... and {len(fail_details) - 10} more")
    else:
        print("All trials passed sanity checks.")

    assert total_fail == 0, f"{total_fail} trials failed sanity checks — see above"
    print("\nValidation complete. Pipeline is clean on real EEG data.")


if __name__ == "__main__":
    main()

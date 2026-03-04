"""Tests for EDF/EDF+ file reader."""

import struct
import tempfile
import os

import numpy as np
import pytest

import zpybci as zbci


# ---- Helpers ----


def write_str_field(buf, offset, length, text):
    """Write a space-padded ASCII string field."""
    encoded = text.encode("ascii")
    data = encoded[:length].ljust(length, b" ")
    buf[offset : offset + length] = data


def write_int_field(buf, offset, length, value):
    """Write an integer as left-justified ASCII in a space-padded field."""
    s = str(value).encode("ascii")
    data = s[:length].ljust(length, b" ")
    buf[offset : offset + length] = data


def write_float_field(buf, offset, length, value):
    """Write a float as left-justified ASCII in a space-padded field."""
    # Use integer representation for clean values
    if value == int(value):
        s = str(int(value)).encode("ascii")
    else:
        s = f"{value}".encode("ascii")
    data = s[:length].ljust(length, b" ")
    buf[offset : offset + length] = data


def make_edf_bytes(
    labels,
    samples_per_record,
    n_records,
    record_duration,
    physical_range=(-3200.0, 3200.0),
    digital_range=(-32768, 32767),
    data=None,
    patient_id="Test Patient",
    recording_id="Test Recording",
    reserved="",
):
    """Build a synthetic EDF file as bytes.

    Args:
        labels: list of signal labels
        samples_per_record: list of samples per record for each signal
        n_records: number of data records
        record_duration: duration of each record in seconds
        physical_range: (min, max) physical values
        digital_range: (min, max) digital values
        data: optional list of 1D arrays of int16 per channel (all records concatenated)
        patient_id: patient identification string
        recording_id: recording identification string
        reserved: reserved field (e.g. "EDF+C" for EDF+)
    """
    ns = len(labels)
    header_bytes = 256 + ns * 256
    rec_size = sum(s * 2 for s in samples_per_record)
    total = header_bytes + n_records * rec_size
    buf = bytearray(b" " * total)

    # Main header
    buf[0:1] = b"0"  # version
    write_str_field(buf, 8, 80, patient_id)
    write_str_field(buf, 88, 80, recording_id)
    write_str_field(buf, 168, 8, "01.01.26")
    write_str_field(buf, 176, 8, "12.00.00")
    write_int_field(buf, 184, 8, header_bytes)
    write_str_field(buf, 192, 44, reserved)
    write_int_field(buf, 236, 8, n_records)
    write_int_field(buf, 244, 8, int(record_duration))
    write_int_field(buf, 252, 4, ns)

    base = 256
    for i, label in enumerate(labels):
        # Label (16 bytes each)
        write_str_field(buf, base + i * 16, 16, label)
        # Transducer (80 bytes each)
        off = base + ns * 16 + i * 80
        write_str_field(buf, off, 80, "AgAgCl")
        # Physical dimension (8 bytes each)
        off = base + ns * 96 + i * 8
        write_str_field(buf, off, 8, "uV")
        # Physical min (8 bytes each)
        off = base + ns * 104 + i * 8
        write_float_field(buf, off, 8, physical_range[0])
        # Physical max (8 bytes each)
        off = base + ns * 112 + i * 8
        write_float_field(buf, off, 8, physical_range[1])
        # Digital min (8 bytes each)
        off = base + ns * 120 + i * 8
        write_int_field(buf, off, 8, digital_range[0])
        # Digital max (8 bytes each)
        off = base + ns * 128 + i * 8
        write_int_field(buf, off, 8, digital_range[1])
        # Prefiltering (80 bytes each)
        off = base + ns * 136 + i * 80
        write_str_field(buf, off, 80, "HP:0.1Hz")
        # N samples per record (8 bytes each)
        off = base + ns * 216 + i * 8
        write_int_field(buf, off, 8, samples_per_record[i])
        # Reserved per signal (32 bytes each) -- already spaces

    # Data records
    for r in range(n_records):
        rec_offset = header_bytes + r * rec_size
        for ch, spr in enumerate(samples_per_record):
            for s in range(spr):
                if data is not None:
                    val = int(data[ch][r * spr + s])
                else:
                    val = 0
                buf[rec_offset : rec_offset + 2] = struct.pack("<h", val)
                rec_offset += 2

    return bytes(buf)


def write_temp_edf(**kwargs):
    """Write synthetic EDF to a temp file and return its path."""
    data = make_edf_bytes(**kwargs)
    fd, path = tempfile.mkstemp(suffix=".edf")
    os.write(fd, data)
    os.close(fd)
    return path


# ---- Header tests ----


class TestHeader:
    def test_basic_header(self):
        path = write_temp_edf(
            labels=["Fp1", "Fp2"],
            samples_per_record=[256, 256],
            n_records=10,
            record_duration=1,
        )
        try:
            rec = zbci.read_edf(path)
            h = rec.header
            assert h["n_signals"] == 2
            assert h["n_records"] == 10
            assert h["record_duration"] == 1.0
            assert h["patient_id"] == "Test Patient"
            assert h["recording_id"] == "Test Recording"
            assert h["is_edf_plus"] is False
            assert h["is_continuous"] is True
        finally:
            os.unlink(path)

    def test_duration(self):
        path = write_temp_edf(
            labels=["Ch1"],
            samples_per_record=[256],
            n_records=5,
            record_duration=2,
        )
        try:
            rec = zbci.read_edf(path)
            assert rec.duration == 10.0
        finally:
            os.unlink(path)

    def test_n_signals(self):
        path = write_temp_edf(
            labels=["A", "B", "C"],
            samples_per_record=[100, 100, 100],
            n_records=1,
            record_duration=1,
        )
        try:
            rec = zbci.read_edf(path)
            assert rec.n_signals == 3
        finally:
            os.unlink(path)

    def test_edf_plus_continuous(self):
        path = write_temp_edf(
            labels=["Ch1"],
            samples_per_record=[256],
            n_records=1,
            record_duration=1,
            reserved="EDF+C",
        )
        try:
            rec = zbci.read_edf(path)
            assert rec.header["is_edf_plus"] is True
            assert rec.header["is_continuous"] is True
        finally:
            os.unlink(path)

    def test_edf_plus_discontinuous(self):
        path = write_temp_edf(
            labels=["Ch1"],
            samples_per_record=[256],
            n_records=1,
            record_duration=1,
            reserved="EDF+D",
        )
        try:
            rec = zbci.read_edf(path)
            assert rec.header["is_edf_plus"] is True
            assert rec.header["is_continuous"] is False
        finally:
            os.unlink(path)


# ---- Signal header tests ----


class TestSignals:
    def test_signal_labels(self):
        path = write_temp_edf(
            labels=["EEG Fp1", "EEG Fp2", "EOG"],
            samples_per_record=[256, 256, 128],
            n_records=1,
            record_duration=1,
        )
        try:
            rec = zbci.read_edf(path)
            sigs = rec.signals
            assert len(sigs) == 3
            assert sigs[0]["label"] == "EEG Fp1"
            assert sigs[1]["label"] == "EEG Fp2"
            assert sigs[2]["label"] == "EOG"
        finally:
            os.unlink(path)

    def test_sample_rate(self):
        path = write_temp_edf(
            labels=["EEG", "EOG"],
            samples_per_record=[256, 128],
            n_records=1,
            record_duration=1,
        )
        try:
            rec = zbci.read_edf(path)
            sigs = rec.signals
            assert sigs[0]["sample_rate"] == 256.0
            assert sigs[1]["sample_rate"] == 128.0
            # Top-level sample_rate is first signal's rate
            assert rec.sample_rate == 256.0
        finally:
            os.unlink(path)

    def test_physical_digital_ranges(self):
        path = write_temp_edf(
            labels=["Ch1"],
            samples_per_record=[256],
            n_records=1,
            record_duration=1,
            physical_range=(-500.0, 500.0),
            digital_range=(-2048, 2047),
        )
        try:
            rec = zbci.read_edf(path)
            sig = rec.signals[0]
            assert sig["physical_min"] == -500.0
            assert sig["physical_max"] == 500.0
            assert sig["digital_min"] == -2048
            assert sig["digital_max"] == 2047
            assert sig["physical_dimension"] == "uV"
        finally:
            os.unlink(path)


# ---- Channel data tests ----


class TestChannelData:
    def test_get_channel_by_index(self):
        # Simple identity mapping: physical = digital (range 0-100)
        spr = 4
        ch_data = [np.array([0, 25, 50, 100], dtype=np.int16)]
        path = write_temp_edf(
            labels=["Ch1"],
            samples_per_record=[spr],
            n_records=1,
            record_duration=1,
            physical_range=(0.0, 100.0),
            digital_range=(0, 100),
            data=ch_data,
        )
        try:
            rec = zbci.read_edf(path)
            ch = rec.get_channel(0)
            assert ch.shape == (4,)
            np.testing.assert_allclose(ch, [0.0, 25.0, 50.0, 100.0], atol=0.01)
        finally:
            os.unlink(path)

    def test_get_channel_by_label(self):
        spr = 4
        ch0 = np.array([10, 20, 30, 40], dtype=np.int16)
        ch1 = np.array([50, 60, 70, 80], dtype=np.int16)
        path = write_temp_edf(
            labels=["Fp1", "Fp2"],
            samples_per_record=[spr, spr],
            n_records=1,
            record_duration=1,
            physical_range=(0.0, 100.0),
            digital_range=(0, 100),
            data=[ch0, ch1],
        )
        try:
            rec = zbci.read_edf(path)
            ch = rec.get_channel("Fp2")
            np.testing.assert_allclose(ch, [50.0, 60.0, 70.0, 80.0], atol=0.01)
        finally:
            os.unlink(path)

    def test_physical_scaling(self):
        # digital -32768 -> physical -3200
        # digital 32767 -> physical 3200
        # digital 0 -> physical ~0
        spr = 3
        ch_data = [np.array([-32768, 0, 32767], dtype=np.int16)]
        path = write_temp_edf(
            labels=["Ch1"],
            samples_per_record=[spr],
            n_records=1,
            record_duration=1,
            physical_range=(-3200.0, 3200.0),
            digital_range=(-32768, 32767),
            data=ch_data,
        )
        try:
            rec = zbci.read_edf(path)
            ch = rec.get_channel(0)
            assert abs(ch[0] - (-3200.0)) < 0.1
            assert abs(ch[1]) < 1.0  # slight asymmetry
            assert abs(ch[2] - 3200.0) < 0.1
        finally:
            os.unlink(path)

    def test_multiple_records(self):
        spr = 4
        n_records = 3
        # 3 records x 4 samples = 12 total
        ch_data = [np.arange(12, dtype=np.int16)]
        path = write_temp_edf(
            labels=["Ch1"],
            samples_per_record=[spr],
            n_records=n_records,
            record_duration=1,
            physical_range=(0.0, 100.0),
            digital_range=(0, 100),
            data=ch_data,
        )
        try:
            rec = zbci.read_edf(path)
            ch = rec.get_channel(0)
            assert ch.shape == (12,)
            np.testing.assert_allclose(ch, np.arange(12, dtype=np.float64), atol=0.01)
        finally:
            os.unlink(path)

    def test_multi_channel_different_rates(self):
        # Channel 0: 256 spr, Channel 1: 128 spr
        ch0 = np.zeros(256, dtype=np.int16)
        ch1 = np.ones(128, dtype=np.int16) * 100
        path = write_temp_edf(
            labels=["EEG", "EOG"],
            samples_per_record=[256, 128],
            n_records=1,
            record_duration=1,
            physical_range=(-3200.0, 3200.0),
            digital_range=(-32768, 32767),
            data=[ch0, ch1],
        )
        try:
            rec = zbci.read_edf(path)
            eeg = rec.get_channel(0)
            eog = rec.get_channel(1)
            assert eeg.shape == (256,)
            assert eog.shape == (128,)
            # All EEG values should be near 0
            assert np.max(np.abs(eeg)) < 1.0
            # All EOG values should be consistent
            assert np.std(eog) < 1e-10
        finally:
            os.unlink(path)

    def test_get_all_channels(self):
        spr = 4
        ch0 = np.array([0, 10, 20, 30], dtype=np.int16)
        ch1 = np.array([40, 50, 60, 70], dtype=np.int16)
        path = write_temp_edf(
            labels=["A", "B"],
            samples_per_record=[spr, spr],
            n_records=1,
            record_duration=1,
            physical_range=(0.0, 100.0),
            digital_range=(0, 100),
            data=[ch0, ch1],
        )
        try:
            rec = zbci.read_edf(path)
            all_ch = rec.get_all_channels()
            assert all_ch.shape == (2, 4)
            np.testing.assert_allclose(all_ch[0], [0.0, 10.0, 20.0, 30.0], atol=0.01)
            np.testing.assert_allclose(all_ch[1], [40.0, 50.0, 60.0, 70.0], atol=0.01)
        finally:
            os.unlink(path)

    def test_get_all_channels_different_rates(self):
        # When channels have different sample rates, shorter ones are zero-padded
        ch0 = np.array([10, 20, 30, 40], dtype=np.int16)
        ch1 = np.array([50, 60], dtype=np.int16)
        path = write_temp_edf(
            labels=["Fast", "Slow"],
            samples_per_record=[4, 2],
            n_records=1,
            record_duration=1,
            physical_range=(0.0, 100.0),
            digital_range=(0, 100),
            data=[ch0, ch1],
        )
        try:
            rec = zbci.read_edf(path)
            all_ch = rec.get_all_channels()
            # Shape should be (2, 4) -- max samples
            assert all_ch.shape == (2, 4)
            np.testing.assert_allclose(all_ch[0], [10.0, 20.0, 30.0, 40.0], atol=0.01)
            # Slow channel: 2 values + 2 zeros
            np.testing.assert_allclose(all_ch[1, :2], [50.0, 60.0], atol=0.01)
            np.testing.assert_allclose(all_ch[1, 2:], [0.0, 0.0], atol=0.01)
        finally:
            os.unlink(path)


# ---- Error handling tests ----


class TestErrors:
    def test_invalid_file_path(self):
        with pytest.raises(ValueError, match="Cannot read"):
            zbci.read_edf("/nonexistent/path.edf")

    def test_invalid_channel_index(self):
        path = write_temp_edf(
            labels=["Ch1"],
            samples_per_record=[4],
            n_records=1,
            record_duration=1,
        )
        try:
            rec = zbci.read_edf(path)
            with pytest.raises(ValueError, match="out of range"):
                rec.get_channel(5)
        finally:
            os.unlink(path)

    def test_invalid_channel_label(self):
        path = write_temp_edf(
            labels=["Ch1"],
            samples_per_record=[4],
            n_records=1,
            record_duration=1,
        )
        try:
            rec = zbci.read_edf(path)
            with pytest.raises(ValueError, match="No signal"):
                rec.get_channel("Nonexistent")
        finally:
            os.unlink(path)

    def test_truncated_header(self):
        # Write only 100 bytes
        fd, path = tempfile.mkstemp(suffix=".edf")
        os.write(fd, b"0" + b" " * 99)
        os.close(fd)
        try:
            with pytest.raises(ValueError, match="Invalid EDF"):
                zbci.read_edf(path)
        finally:
            os.unlink(path)


# ---- Repr test ----


class TestRepr:
    def test_repr(self):
        path = write_temp_edf(
            labels=["Fp1", "Fp2"],
            samples_per_record=[256, 256],
            n_records=10,
            record_duration=1,
        )
        try:
            rec = zbci.read_edf(path)
            r = repr(rec)
            assert "EdfRecording" in r
            assert "Fp1" in r
            assert "Fp2" in r
        finally:
            os.unlink(path)

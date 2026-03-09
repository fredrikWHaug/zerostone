"""Tests for XDF file reader."""

import struct
import tempfile
import os

import numpy as np
import pytest

import zpybci as zbci


# ---- Helpers ----


def varlen_bytes(val):
    """Encode a variable-length integer in XDF format."""
    if val <= 255:
        return struct.pack("<BB", 1, val)
    elif val <= 0xFFFFFFFF:
        return struct.pack("<BI", 4, val)
    else:
        return struct.pack("<BQ", 8, val)


def make_chunk(tag, content):
    """Build a raw XDF chunk: varlen(length) + uint16(tag) + content."""
    chunk_len = 2 + len(content)
    return varlen_bytes(chunk_len) + struct.pack("<H", tag) + content


def make_file_header(version="1.0"):
    """Build a FileHeader chunk (tag=1)."""
    xml = f'<?xml version="1.0"?><info><version>{version}</version></info>'
    return make_chunk(1, xml.encode("utf-8"))


def make_stream_header(stream_id, name, stype, ch_count, srate, fmt):
    """Build a StreamHeader chunk (tag=2)."""
    xml = (
        f'<?xml version="1.0"?><info>'
        f"<name>{name}</name>"
        f"<type>{stype}</type>"
        f"<channel_count>{ch_count}</channel_count>"
        f"<nominal_srate>{srate}</nominal_srate>"
        f"<channel_format>{fmt}</channel_format>"
        f"</info>"
    )
    content = struct.pack("<I", stream_id) + xml.encode("utf-8")
    return make_chunk(2, content)


def make_samples_numeric(stream_id, ch_count, samples, fmt_char, fmt_size):
    """Build a Samples chunk (tag=3) for numeric data.

    samples: list of (timestamp_or_nan, [values...])
    fmt_char: struct format char ('f' for f32, 'd' for f64, 'h' for i16, 'i' for i32)
    """
    content = struct.pack("<I", stream_id)
    for ts, values in samples:
        assert len(values) == ch_count
        if np.isnan(ts):
            content += struct.pack("B", 0)  # no timestamp
        else:
            content += struct.pack("B", 8)  # has timestamp
            content += struct.pack("<d", ts)
        for v in values:
            content += struct.pack(f"<{fmt_char}", v)
    return make_chunk(3, content)


def make_samples_f32(stream_id, ch_count, samples):
    return make_samples_numeric(stream_id, ch_count, samples, "f", 4)


def make_samples_f64(stream_id, ch_count, samples):
    return make_samples_numeric(stream_id, ch_count, samples, "d", 8)


def make_samples_i16(stream_id, ch_count, samples):
    return make_samples_numeric(stream_id, ch_count, samples, "h", 2)


def make_samples_i32(stream_id, ch_count, samples):
    return make_samples_numeric(stream_id, ch_count, samples, "i", 4)


def make_samples_string(stream_id, ch_count, samples):
    """Build a Samples chunk for string data.

    samples: list of (timestamp_or_nan, [str_values...])
    Each string is prefixed with a varlen length.
    """
    content = struct.pack("<I", stream_id)
    for ts, values in samples:
        assert len(values) == ch_count
        if np.isnan(ts):
            content += struct.pack("B", 0)
        else:
            content += struct.pack("B", 8)
            content += struct.pack("<d", ts)
        for s in values:
            encoded = s.encode("utf-8")
            content += varlen_bytes(len(encoded))
            content += encoded
    return make_chunk(3, content)


def make_clock_offset(stream_id, collection_time, offset_value):
    """Build a ClockOffset chunk (tag=4)."""
    content = struct.pack("<I", stream_id)
    content += struct.pack("<d", collection_time)
    content += struct.pack("<d", offset_value)
    return make_chunk(4, content)


def make_stream_footer(stream_id):
    """Build a StreamFooter chunk (tag=6)."""
    xml = '<?xml version="1.0"?><info><first_timestamp>0</first_timestamp></info>'
    content = struct.pack("<I", stream_id) + xml.encode("utf-8")
    return make_chunk(6, content)


def make_xdf_bytes(chunks):
    """Build a complete XDF file from a list of chunk byte strings."""
    data = b"XDF:" + make_file_header()
    for chunk in chunks:
        data += chunk
    return data


def write_temp_xdf(chunks):
    """Write synthetic XDF to a temp file and return its path."""
    data = make_xdf_bytes(chunks)
    fd, path = tempfile.mkstemp(suffix=".xdf")
    os.write(fd, data)
    os.close(fd)
    return path


# ---- Magic and header tests ----


class TestMagicAndHeader:
    def test_valid_file(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 2, 256.0, "float32"),
        ])
        try:
            rec = zbci.read_xdf(path)
            assert rec.stream_count == 1
        finally:
            os.unlink(path)

    def test_invalid_magic(self):
        fd, path = tempfile.mkstemp(suffix=".xdf")
        os.write(fd, b"NOT_XDF_DATA")
        os.close(fd)
        try:
            with pytest.raises(ValueError, match="Invalid XDF"):
                zbci.read_xdf(path)
        finally:
            os.unlink(path)

    def test_file_header_version(self):
        path = write_temp_xdf([
            make_stream_header(1, "Test", "EEG", 1, 256.0, "float32"),
        ])
        try:
            rec = zbci.read_xdf(path)
            h = rec.header
            assert h["version"] == "1.0"
        finally:
            os.unlink(path)


# ---- Stream metadata tests ----


class TestStreamMetadata:
    def test_stream_name_and_type(self):
        path = write_temp_xdf([
            make_stream_header(1, "MyEEG", "EEG", 8, 512.0, "float32"),
        ])
        try:
            rec = zbci.read_xdf(path)
            s = rec.streams[0]
            assert s.name == "MyEEG"
            assert s.stream_type == "EEG"
        finally:
            os.unlink(path)

    def test_channel_count(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 32, 256.0, "float32"),
        ])
        try:
            rec = zbci.read_xdf(path)
            assert rec.streams[0].channel_count == 32
        finally:
            os.unlink(path)

    def test_sample_rate(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 1, 1000.0, "float32"),
        ])
        try:
            rec = zbci.read_xdf(path)
            assert rec.streams[0].sample_rate == 1000.0
        finally:
            os.unlink(path)

    def test_channel_format(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 1, 256.0, "double64"),
        ])
        try:
            rec = zbci.read_xdf(path)
            assert rec.streams[0].channel_format == "double64"
        finally:
            os.unlink(path)

    def test_stream_id(self):
        path = write_temp_xdf([
            make_stream_header(42, "Test", "EEG", 1, 256.0, "float32"),
        ])
        try:
            rec = zbci.read_xdf(path)
            assert rec.streams[0].stream_id == 42
        finally:
            os.unlink(path)


# ---- Numeric data tests ----


class TestNumericData:
    def test_float32_stream(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 2, 256.0, "float32"),
            make_samples_f32(1, 2, [
                (1.0, [0.5, -0.5]),
                (2.0, [1.0, -1.0]),
                (3.0, [1.5, -1.5]),
            ]),
        ])
        try:
            rec = zbci.read_xdf(path)
            s = rec.streams[0]
            data = s.data
            assert data.shape == (3, 2)
            np.testing.assert_allclose(data[0], [0.5, -0.5], atol=1e-5)
            np.testing.assert_allclose(data[1], [1.0, -1.0], atol=1e-5)
            np.testing.assert_allclose(data[2], [1.5, -1.5], atol=1e-5)
        finally:
            os.unlink(path)

    def test_double64_stream(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 1, 256.0, "double64"),
            make_samples_f64(1, 1, [
                (0.0, [3.141592653589793]),
                (0.004, [2.718281828459045]),
            ]),
        ])
        try:
            rec = zbci.read_xdf(path)
            data = rec.streams[0].data
            assert data.shape == (2, 1)
            np.testing.assert_allclose(data[0, 0], 3.141592653589793, atol=1e-15)
            np.testing.assert_allclose(data[1, 0], 2.718281828459045, atol=1e-15)
        finally:
            os.unlink(path)

    def test_int16_stream(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 2, 256.0, "int16"),
            make_samples_i16(1, 2, [
                (1.0, [1000, -2000]),
                (2.0, [3000, -4000]),
            ]),
        ])
        try:
            rec = zbci.read_xdf(path)
            data = rec.streams[0].data
            assert data.shape == (2, 2)
            np.testing.assert_allclose(data[0], [1000.0, -2000.0])
            np.testing.assert_allclose(data[1], [3000.0, -4000.0])
        finally:
            os.unlink(path)

    def test_int32_stream(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 1, 256.0, "int32"),
            make_samples_i32(1, 1, [
                (0.0, [100000]),
                (0.004, [-200000]),
            ]),
        ])
        try:
            rec = zbci.read_xdf(path)
            data = rec.streams[0].data
            assert data.shape == (2, 1)
            np.testing.assert_allclose(data[0, 0], 100000.0)
            np.testing.assert_allclose(data[1, 0], -200000.0)
        finally:
            os.unlink(path)

    def test_multiple_sample_chunks(self):
        """Test that samples split across multiple chunks are concatenated."""
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 1, 256.0, "float32"),
            make_samples_f32(1, 1, [(1.0, [10.0])]),
            make_samples_f32(1, 1, [(2.0, [20.0])]),
            make_samples_f32(1, 1, [(3.0, [30.0])]),
        ])
        try:
            rec = zbci.read_xdf(path)
            data = rec.streams[0].data
            assert data.shape == (3, 1)
            np.testing.assert_allclose(data[:, 0], [10.0, 20.0, 30.0], atol=1e-5)
        finally:
            os.unlink(path)


# ---- Timestamp tests ----


class TestTimestamps:
    def test_explicit_timestamps(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 1, 256.0, "float32"),
            make_samples_f32(1, 1, [
                (100.0, [1.0]),
                (100.004, [2.0]),
                (100.008, [3.0]),
            ]),
        ])
        try:
            rec = zbci.read_xdf(path)
            ts = rec.streams[0].timestamps
            assert ts.shape == (3,)
            np.testing.assert_allclose(ts, [100.0, 100.004, 100.008], atol=1e-10)
        finally:
            os.unlink(path)

    def test_deduced_timestamps(self):
        """Samples without timestamps have NaN."""
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 1, 256.0, "float32"),
            make_samples_f32(1, 1, [
                (float("nan"), [1.0]),
                (float("nan"), [2.0]),
            ]),
        ])
        try:
            rec = zbci.read_xdf(path)
            ts = rec.streams[0].timestamps
            assert ts.shape == (2,)
            assert np.isnan(ts[0])
            assert np.isnan(ts[1])
        finally:
            os.unlink(path)

    def test_mixed_timestamps(self):
        """First sample has timestamp, rest are deduced."""
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 1, 256.0, "float32"),
            make_samples_f32(1, 1, [
                (100.0, [1.0]),
                (float("nan"), [2.0]),
                (float("nan"), [3.0]),
            ]),
        ])
        try:
            rec = zbci.read_xdf(path)
            ts = rec.streams[0].timestamps
            assert ts.shape == (3,)
            np.testing.assert_allclose(ts[0], 100.0, atol=1e-10)
            assert np.isnan(ts[1])
            assert np.isnan(ts[2])
        finally:
            os.unlink(path)


# ---- Multi-stream tests ----


class TestMultiStream:
    def test_two_streams(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 2, 256.0, "float32"),
            make_stream_header(2, "Markers", "Markers", 1, 0.0, "float32"),
            make_samples_f32(1, 2, [(1.0, [0.1, 0.2])]),
            make_samples_f32(2, 1, [(1.5, [99.0])]),
        ])
        try:
            rec = zbci.read_xdf(path)
            assert rec.stream_count == 2
            eeg = rec.streams[0]
            markers = rec.streams[1]
            assert eeg.name == "EEG"
            assert markers.name == "Markers"
            assert eeg.data.shape == (1, 2)
            assert markers.data.shape == (1, 1)
        finally:
            os.unlink(path)

    def test_get_stream_by_name(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 2, 256.0, "float32"),
            make_stream_header(2, "Markers", "Markers", 1, 0.0, "float32"),
            make_samples_f32(1, 2, [(1.0, [0.1, 0.2])]),
        ])
        try:
            rec = zbci.read_xdf(path)
            eeg = rec.get_stream("EEG")
            assert eeg.name == "EEG"
            assert eeg.channel_count == 2
        finally:
            os.unlink(path)

    def test_get_stream_by_index(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 2, 256.0, "float32"),
            make_stream_header(2, "Markers", "Markers", 1, 0.0, "float32"),
        ])
        try:
            rec = zbci.read_xdf(path)
            s = rec.get_stream(1)
            assert s.name == "Markers"
        finally:
            os.unlink(path)

    def test_get_stream_invalid_name(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 1, 256.0, "float32"),
        ])
        try:
            rec = zbci.read_xdf(path)
            with pytest.raises(ValueError, match="No stream"):
                rec.get_stream("Nonexistent")
        finally:
            os.unlink(path)

    def test_get_stream_invalid_index(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 1, 256.0, "float32"),
        ])
        try:
            rec = zbci.read_xdf(path)
            with pytest.raises(ValueError, match="out of range"):
                rec.get_stream(5)
        finally:
            os.unlink(path)


# ---- String stream tests ----


class TestStringStream:
    def test_marker_stream(self):
        path = write_temp_xdf([
            make_stream_header(1, "Markers", "Markers", 1, 0.0, "string"),
            make_samples_string(1, 1, [
                (10.0, ["stimulus_onset"]),
                (12.5, ["response"]),
                (15.0, ["stimulus_offset"]),
            ]),
        ])
        try:
            rec = zbci.read_xdf(path)
            s = rec.streams[0]
            assert s.channel_format == "string"
            data = s.data
            assert len(data) == 3
            assert data[0] == ["stimulus_onset"]
            assert data[1] == ["response"]
            assert data[2] == ["stimulus_offset"]
            ts = s.timestamps
            np.testing.assert_allclose(ts, [10.0, 12.5, 15.0], atol=1e-10)
        finally:
            os.unlink(path)

    def test_multi_channel_string(self):
        path = write_temp_xdf([
            make_stream_header(1, "Events", "Markers", 2, 0.0, "string"),
            make_samples_string(1, 2, [
                (1.0, ["event_type", "event_value"]),
            ]),
        ])
        try:
            rec = zbci.read_xdf(path)
            data = rec.streams[0].data
            assert len(data) == 1
            assert data[0] == ["event_type", "event_value"]
        finally:
            os.unlink(path)


# ---- Clock offset tests ----


class TestClockOffsets:
    def test_clock_offsets(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 1, 256.0, "float32"),
            make_clock_offset(1, 100.0, -0.003),
            make_clock_offset(1, 200.0, -0.002),
            make_clock_offset(1, 300.0, -0.001),
        ])
        try:
            rec = zbci.read_xdf(path)
            offsets = rec.streams[0].clock_offsets
            assert len(offsets) == 3
            assert abs(offsets[0][0] - 100.0) < 1e-10
            assert abs(offsets[0][1] - (-0.003)) < 1e-10
            assert abs(offsets[1][0] - 200.0) < 1e-10
            assert abs(offsets[2][0] - 300.0) < 1e-10
        finally:
            os.unlink(path)

    def test_no_clock_offsets(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 1, 256.0, "float32"),
        ])
        try:
            rec = zbci.read_xdf(path)
            assert len(rec.streams[0].clock_offsets) == 0
        finally:
            os.unlink(path)


# ---- Edge case tests ----


class TestEdgeCases:
    def test_empty_stream(self):
        """Stream with header but no samples."""
        path = write_temp_xdf([
            make_stream_header(1, "Empty", "EEG", 4, 256.0, "float32"),
        ])
        try:
            rec = zbci.read_xdf(path)
            s = rec.streams[0]
            data = s.data
            assert data.shape == (0, 4)
            assert s.timestamps.shape == (0,)
        finally:
            os.unlink(path)

    def test_single_sample(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 1, 256.0, "float32"),
            make_samples_f32(1, 1, [(42.0, [7.5])]),
        ])
        try:
            rec = zbci.read_xdf(path)
            data = rec.streams[0].data
            assert data.shape == (1, 1)
            np.testing.assert_allclose(data[0, 0], 7.5, atol=1e-5)
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        with pytest.raises(ValueError, match="Cannot read"):
            zbci.read_xdf("/nonexistent/path/file.xdf")

    def test_stream_footer_ignored(self):
        """StreamFooter chunks should not cause errors."""
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 1, 256.0, "float32"),
            make_samples_f32(1, 1, [(1.0, [1.0])]),
            make_stream_footer(1),
        ])
        try:
            rec = zbci.read_xdf(path)
            assert rec.streams[0].data.shape == (1, 1)
        finally:
            os.unlink(path)


# ---- Repr tests ----


class TestRepr:
    def test_recording_repr(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 2, 256.0, "float32"),
            make_stream_header(2, "Markers", "Markers", 1, 0.0, "string"),
        ])
        try:
            rec = zbci.read_xdf(path)
            r = repr(rec)
            assert "XdfRecording" in r
            assert "EEG" in r
            assert "Markers" in r
        finally:
            os.unlink(path)

    def test_stream_repr(self):
        path = write_temp_xdf([
            make_stream_header(1, "EEG", "EEG", 32, 256.0, "float32"),
            make_samples_f32(1, 32, [(1.0, [0.0] * 32)]),
        ])
        try:
            rec = zbci.read_xdf(path)
            r = repr(rec.streams[0])
            assert "XdfStream" in r
            assert "EEG" in r
            assert "32" in r
        finally:
            os.unlink(path)

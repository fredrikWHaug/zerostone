"""Tests for MDA (MountainSort Data Array) file reader bindings."""

import struct
import tempfile
import os

import numpy as np
import pytest

import zpybci as zbci


# MDA dtype codes
MDA_UINT8 = -2
MDA_FLOAT32 = -3
MDA_INT16 = -4
MDA_INT32 = -5
MDA_UINT16 = -6
MDA_FLOAT64 = -7
MDA_UINT32 = -8


def write_mda(path, data, dtype_code):
    """Write a simple MDA file.

    Data is written in column-major (Fortran) order as per MDA spec.
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<i", dtype_code))
        f.write(struct.pack("<i", data.ndim))
        for d in data.shape:
            f.write(struct.pack("<i", d))
        f.write(data.flatten(order="F").tobytes())


class TestReadMdaFloat32:
    """Test reading float32 MDA files."""

    def test_1d_float32(self, tmp_path):
        arr = np.array([1.0, 2.5, -3.0], dtype=np.float32)
        path = str(tmp_path / "test.mda")
        write_mda(path, arr, MDA_FLOAT32)

        result = zbci.read_mda(path)
        assert result["dtype"] == "float32"
        assert result["shape"] == [3]
        np.testing.assert_allclose(result["data"], arr.astype(np.float64), atol=1e-6)

    def test_2d_float32(self, tmp_path):
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        path = str(tmp_path / "test.mda")
        write_mda(path, arr, MDA_FLOAT32)

        result = zbci.read_mda(path)
        assert result["dtype"] == "float32"
        assert result["shape"] == [2, 3]
        # Data is column-major flat
        expected_flat = arr.flatten(order="F").astype(np.float64)
        np.testing.assert_allclose(result["data"], expected_flat, atol=1e-6)

    def test_reshape_2d(self, tmp_path):
        """Verify data can be reshaped back to original 2D array."""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        path = str(tmp_path / "test.mda")
        write_mda(path, arr, MDA_FLOAT32)

        result = zbci.read_mda(path)
        shape = tuple(result["shape"])
        reshaped = result["data"].reshape(shape, order="F")
        np.testing.assert_allclose(reshaped, arr.astype(np.float64), atol=1e-6)


class TestReadMdaFloat64:
    """Test reading float64 MDA files."""

    def test_1d_float64(self, tmp_path):
        arr = np.array([np.pi, -np.e, 0.0, 1e300], dtype=np.float64)
        path = str(tmp_path / "test.mda")
        write_mda(path, arr, MDA_FLOAT64)

        result = zbci.read_mda(path)
        assert result["dtype"] == "float64"
        assert result["shape"] == [4]
        np.testing.assert_allclose(result["data"], arr, atol=1e-15)


class TestReadMdaInt16:
    """Test reading int16 MDA files."""

    def test_1d_int16(self, tmp_path):
        arr = np.array([0, 100, -200, 32767, -32768], dtype=np.int16)
        path = str(tmp_path / "test.mda")
        write_mda(path, arr, MDA_INT16)

        result = zbci.read_mda(path)
        assert result["dtype"] == "int16"
        assert result["shape"] == [5]
        np.testing.assert_allclose(result["data"], arr.astype(np.float64))

    def test_2d_int16(self, tmp_path):
        arr = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int16)
        path = str(tmp_path / "test.mda")
        write_mda(path, arr, MDA_INT16)

        result = zbci.read_mda(path)
        assert result["shape"] == [3, 2]
        reshaped = result["data"].reshape(tuple(result["shape"]), order="F")
        np.testing.assert_allclose(reshaped, arr.astype(np.float64))


class TestReadMdaInt32:
    """Test reading int32 MDA files."""

    def test_1d_int32(self, tmp_path):
        arr = np.array([0, 1000000, -999], dtype=np.int32)
        path = str(tmp_path / "test.mda")
        write_mda(path, arr, MDA_INT32)

        result = zbci.read_mda(path)
        assert result["dtype"] == "int32"
        assert result["shape"] == [3]
        np.testing.assert_allclose(result["data"], arr.astype(np.float64))


class TestReadMdaUint8:
    """Test reading uint8 MDA files."""

    def test_1d_uint8(self, tmp_path):
        arr = np.array([0, 127, 255], dtype=np.uint8)
        path = str(tmp_path / "test.mda")
        write_mda(path, arr, MDA_UINT8)

        result = zbci.read_mda(path)
        assert result["dtype"] == "uint8"
        assert result["shape"] == [3]
        np.testing.assert_allclose(result["data"], arr.astype(np.float64))


class TestReadMdaUint16:
    """Test reading uint16 MDA files."""

    def test_1d_uint16(self, tmp_path):
        arr = np.array([0, 1000, 65535], dtype=np.uint16)
        path = str(tmp_path / "test.mda")
        write_mda(path, arr, MDA_UINT16)

        result = zbci.read_mda(path)
        assert result["dtype"] == "uint16"
        assert result["shape"] == [3]
        np.testing.assert_allclose(result["data"], arr.astype(np.float64))


class TestReadMdaUint32:
    """Test reading uint32 MDA files."""

    def test_1d_uint32(self, tmp_path):
        arr = np.array([0, 4000000000], dtype=np.uint32)
        path = str(tmp_path / "test.mda")
        write_mda(path, arr, MDA_UINT32)

        result = zbci.read_mda(path)
        assert result["dtype"] == "uint32"
        assert result["shape"] == [2]
        np.testing.assert_allclose(result["data"], arr.astype(np.float64), rtol=1e-6)


class TestReadMdaErrors:
    """Test error handling."""

    def test_nonexistent_file(self):
        with pytest.raises(ValueError, match="Failed to read"):
            zbci.read_mda("/nonexistent/path/file.mda")

    def test_invalid_header(self, tmp_path):
        path = str(tmp_path / "bad.mda")
        with open(path, "wb") as f:
            f.write(b"\x00\x00\x00\x00")  # invalid dtype code 0
        with pytest.raises(ValueError, match="Invalid MDA header"):
            zbci.read_mda(path)

    def test_truncated_data(self, tmp_path):
        """Header claims 10 float32 elements but file has none."""
        path = str(tmp_path / "trunc.mda")
        with open(path, "wb") as f:
            f.write(struct.pack("<i", MDA_FLOAT32))
            f.write(struct.pack("<i", 1))
            f.write(struct.pack("<i", 10))
            # No data bytes
        with pytest.raises(ValueError, match="Failed to read MDA data"):
            zbci.read_mda(path)

    def test_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.mda")
        with open(path, "wb") as f:
            pass  # 0 bytes
        with pytest.raises(ValueError, match="Invalid MDA header"):
            zbci.read_mda(path)


class TestReadMdaShape:
    """Test that shape tuple is correct for various dimensionalities."""

    def test_1d_shape(self, tmp_path):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        path = str(tmp_path / "test.mda")
        write_mda(path, arr, MDA_FLOAT32)

        result = zbci.read_mda(path)
        assert result["shape"] == [5]
        assert len(result["data"]) == 5

    def test_2d_shape(self, tmp_path):
        arr = np.zeros((4, 100), dtype=np.float64)
        path = str(tmp_path / "test.mda")
        write_mda(path, arr, MDA_FLOAT64)

        result = zbci.read_mda(path)
        assert result["shape"] == [4, 100]
        assert len(result["data"]) == 400

    def test_3d_shape(self, tmp_path):
        arr = np.zeros((2, 3, 4), dtype=np.float32)
        path = str(tmp_path / "test.mda")
        write_mda(path, arr, MDA_FLOAT32)

        result = zbci.read_mda(path)
        assert result["shape"] == [2, 3, 4]
        assert len(result["data"]) == 24

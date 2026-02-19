"""Tests for OasisDeconvolution Python bindings."""
import numpy as np
import pytest


class TestOasisDeconvolution:
    """Tests for OasisDeconvolution."""

    def test_import(self):
        import zpybci as zbci
        assert hasattr(zbci, 'OasisDeconvolution')

    def test_create(self):
        import zpybci as zbci
        deconv = zbci.OasisDeconvolution(channels=1, gamma=0.95, lambda_=0.01)
        assert np.isclose(deconv.gamma, 0.95)
        assert np.isclose(deconv.lambda_, 0.01)
        assert deconv.channels == 1
        assert deconv.sample_count == 0

    def test_create_all_sizes(self):
        import zpybci as zbci
        for ch in [1, 4, 8, 16, 32, 64]:
            deconv = zbci.OasisDeconvolution(channels=ch, gamma=0.95, lambda_=0.01)
            assert deconv.channels == ch

    def test_invalid_channels(self):
        import zpybci as zbci
        with pytest.raises(ValueError):
            zbci.OasisDeconvolution(channels=3, gamma=0.95, lambda_=0.01)

    def test_from_tau(self):
        import zpybci as zbci
        deconv = zbci.OasisDeconvolution.from_tau(
            channels=1, sample_rate=30.0, tau=1.0, lambda_=0.01
        )
        assert deconv.channels == 1
        assert deconv.gamma > 0.0

    def test_update_single_channel(self):
        import zpybci as zbci
        deconv = zbci.OasisDeconvolution(channels=1, gamma=0.95, lambda_=0.01)
        fluor = np.array([1.0], dtype=np.float32)
        baseline = np.array([0.0], dtype=np.float32)
        calcium, spike = deconv.update(fluor, baseline)
        assert calcium.shape == (1,)
        assert spike.shape == (1,)
        assert calcium.dtype == np.float32
        assert spike.dtype == np.float32
        assert deconv.sample_count == 1

    def test_update_multi_channel(self):
        import zpybci as zbci
        deconv = zbci.OasisDeconvolution(channels=4, gamma=0.95, lambda_=0.01)
        fluor = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        baseline = np.zeros(4, dtype=np.float32)
        calcium, spike = deconv.update(fluor, baseline)
        assert calcium.shape == (4,)
        assert spike.shape == (4,)

    def test_update_wrong_size(self):
        import zpybci as zbci
        deconv = zbci.OasisDeconvolution(channels=4, gamma=0.95, lambda_=0.01)
        with pytest.raises(ValueError):
            deconv.update(
                np.array([1.0], dtype=np.float32),
                np.zeros(4, dtype=np.float32)
            )

    def test_set_lambda(self):
        import zpybci as zbci
        deconv = zbci.OasisDeconvolution(channels=1, gamma=0.95, lambda_=0.01)
        deconv.set_lambda(0.05)
        assert np.isclose(deconv.lambda_, 0.05)

    def test_reset(self):
        import zpybci as zbci
        deconv = zbci.OasisDeconvolution(channels=1, gamma=0.95, lambda_=0.01)
        fluor = np.array([1.0], dtype=np.float32)
        baseline = np.array([0.0], dtype=np.float32)
        deconv.update(fluor, baseline)
        assert deconv.sample_count == 1
        deconv.reset()
        assert deconv.sample_count == 0

    def test_sequence(self):
        """Test processing a sequence of fluorescence data."""
        import zpybci as zbci
        deconv = zbci.OasisDeconvolution(channels=1, gamma=0.95, lambda_=0.01)
        for i in range(100):
            fluor = np.array([float(i % 10) * 0.1], dtype=np.float32)
            baseline = np.array([0.0], dtype=np.float32)
            calcium, spike = deconv.update(fluor, baseline)
            assert calcium.shape == (1,)
        assert deconv.sample_count == 100

    def test_repr(self):
        import zpybci as zbci
        deconv = zbci.OasisDeconvolution(channels=1, gamma=0.95, lambda_=0.01)
        r = repr(deconv)
        assert 'OasisDeconvolution' in r
        assert 'channels=1' in r


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

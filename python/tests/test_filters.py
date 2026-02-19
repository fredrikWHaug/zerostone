"""Tests for filter Python bindings (FirFilter, AcCoupler, MedianFilter)."""
import numpy as np
import pytest


class TestFirFilter:
    """Tests for FirFilter."""

    def test_import(self):
        """Test that FirFilter can be imported."""
        import zpybci as zbci
        assert hasattr(zbci, 'FirFilter')

    def test_create_with_taps(self):
        """Test creating FirFilter with custom taps."""
        import zpybci as zbci

        fir = zbci.FirFilter(taps=[0.2, 0.2, 0.2, 0.2, 0.2])
        assert fir.num_taps == 5

    def test_moving_average(self):
        """Test creating a moving average filter."""
        import zpybci as zbci

        fir = zbci.FirFilter.moving_average(5)
        assert fir.num_taps == 5

    def test_moving_average_invalid_size(self):
        """Test that invalid sizes raise errors."""
        import zpybci as zbci

        with pytest.raises(ValueError):
            zbci.FirFilter.moving_average(0)
        with pytest.raises(ValueError):
            zbci.FirFilter.moving_average(65)

    def test_empty_taps_error(self):
        """Test that empty taps raise error."""
        import zpybci as zbci

        with pytest.raises(ValueError):
            zbci.FirFilter(taps=[])

    def test_process_output_shape(self):
        """Test that process returns correct shape."""
        import zpybci as zbci

        fir = zbci.FirFilter.moving_average(5)
        signal = np.random.randn(100).astype(np.float32)
        filtered = fir.process(signal)

        assert isinstance(filtered, np.ndarray)
        assert filtered.dtype == np.float32
        assert len(filtered) == len(signal)

    def test_moving_average_convergence(self):
        """Test that moving average converges to DC value."""
        import zpybci as zbci

        fir = zbci.FirFilter.moving_average(5)
        signal = np.ones(20, dtype=np.float32)
        filtered = fir.process(signal)

        # After 5 samples, moving average of ones should be 1.0
        assert np.allclose(filtered[4:], 1.0, atol=1e-5)

    def test_reset(self):
        """Test that reset clears filter state."""
        import zpybci as zbci

        fir = zbci.FirFilter.moving_average(5)

        # Process some data
        signal = np.ones(10, dtype=np.float32) * 5.0
        fir.process(signal)

        # Reset and process same signal
        fir.reset()
        out1 = fir.process(signal)

        # Create fresh filter
        fir2 = zbci.FirFilter.moving_average(5)
        out2 = fir2.process(signal)

        assert np.allclose(out1, out2)

    def test_repr(self):
        """Test string representation."""
        import zpybci as zbci

        fir = zbci.FirFilter.moving_average(8)
        assert 'FirFilter' in repr(fir)
        assert '8' in repr(fir)

    def test_optimized_sizes(self):
        """Test that optimized sizes (8, 16, 32, 64) work."""
        import zpybci as zbci

        for size in [8, 16, 32, 64]:
            fir = zbci.FirFilter.moving_average(size)
            signal = np.ones(100, dtype=np.float32)
            filtered = fir.process(signal)
            # Should converge to 1.0
            assert np.allclose(filtered[-10:], 1.0, atol=1e-5)

    def test_dynamic_size(self):
        """Test non-optimized sizes use dynamic implementation."""
        import zpybci as zbci

        # Size 10 should use dynamic implementation
        fir = zbci.FirFilter.moving_average(10)
        assert fir.num_taps == 10

        signal = np.ones(50, dtype=np.float32)
        filtered = fir.process(signal)
        assert np.allclose(filtered[-10:], 1.0, atol=1e-5)


class TestAcCoupler:
    """Tests for AcCoupler."""

    def test_import(self):
        """Test that AcCoupler can be imported."""
        import zpybci as zbci
        assert hasattr(zbci, 'AcCoupler')

    def test_create(self):
        """Test creating AcCoupler."""
        import zpybci as zbci

        ac = zbci.AcCoupler(1000.0, 0.1)
        assert ac.sample_rate == 1000.0
        assert np.isclose(ac.cutoff, 0.1)

    def test_invalid_sample_rate(self):
        """Test that invalid sample rate raises error."""
        import zpybci as zbci

        with pytest.raises(ValueError):
            zbci.AcCoupler(0.0, 0.1)
        with pytest.raises(ValueError):
            zbci.AcCoupler(-100.0, 0.1)

    def test_invalid_cutoff(self):
        """Test that invalid cutoff raises error."""
        import zpybci as zbci

        # Cutoff above Nyquist
        with pytest.raises(ValueError):
            zbci.AcCoupler(1000.0, 600.0)
        # Negative cutoff
        with pytest.raises(ValueError):
            zbci.AcCoupler(1000.0, -1.0)

    def test_process_output_shape(self):
        """Test that process returns correct shape."""
        import zpybci as zbci

        ac = zbci.AcCoupler(1000.0, 0.1)
        signal = np.random.randn(100).astype(np.float32)
        filtered = ac.process(signal)

        assert isinstance(filtered, np.ndarray)
        assert filtered.dtype == np.float32
        assert len(filtered) == len(signal)

    def test_removes_dc(self):
        """Test that AC coupler removes DC offset."""
        import zpybci as zbci

        ac = zbci.AcCoupler(250.0, 0.5)

        # Constant DC signal
        signal = np.ones(1000, dtype=np.float32) * 5.0
        filtered = ac.process(signal)

        # DC should be attenuated (last samples should be near zero)
        assert abs(filtered[-1]) < 0.1

    def test_preserves_ac(self):
        """Test that AC coupler preserves high-frequency content."""
        import zpybci as zbci

        sample_rate = 250.0
        ac = zbci.AcCoupler(sample_rate, 0.1)

        # 10 Hz sine wave (well above 0.1 Hz cutoff)
        t = np.arange(0, 2, 1/sample_rate, dtype=np.float32)
        signal = np.sin(2 * np.pi * 10 * t).astype(np.float32)
        filtered = ac.process(signal)

        # After settling, amplitude should be preserved (>90%)
        max_filtered = np.max(np.abs(filtered[250:]))
        assert max_filtered > 0.9

    def test_reset(self):
        """Test that reset clears filter state."""
        import zpybci as zbci

        ac = zbci.AcCoupler(1000.0, 0.1)

        # Process some data
        ac.process(np.ones(100, dtype=np.float32))

        # Reset
        ac.reset()

        # First sample after reset should pass through (like fresh filter)
        out = ac.process(np.array([5.0], dtype=np.float32))
        assert np.allclose(out[0], 5.0, atol=1e-5)

    def test_repr(self):
        """Test string representation."""
        import zpybci as zbci

        ac = zbci.AcCoupler(1000.0, 0.1)
        assert 'AcCoupler' in repr(ac)
        assert '1000' in repr(ac)


class TestMedianFilter:
    """Tests for MedianFilter."""

    def test_import(self):
        """Test that MedianFilter can be imported."""
        import zpybci as zbci
        assert hasattr(zbci, 'MedianFilter')

    def test_create(self):
        """Test creating MedianFilter."""
        import zpybci as zbci

        for size in [3, 5, 7]:
            mf = zbci.MedianFilter(size)
            assert mf.window_size == size

    def test_invalid_window_size(self):
        """Test that invalid window sizes raise error."""
        import zpybci as zbci

        for size in [1, 2, 4, 6, 8, 9]:
            with pytest.raises(ValueError):
                zbci.MedianFilter(size)

    def test_process_output_shape(self):
        """Test that process returns correct shape."""
        import zpybci as zbci

        mf = zbci.MedianFilter(5)
        signal = np.random.randn(100).astype(np.float32)
        filtered = mf.process(signal)

        assert isinstance(filtered, np.ndarray)
        assert filtered.dtype == np.float32
        assert len(filtered) == len(signal)

    def test_spike_rejection(self):
        """Test that median filter rejects spikes."""
        import zpybci as zbci

        mf = zbci.MedianFilter(5)

        # Prime filter with baseline
        baseline = np.ones(5, dtype=np.float32) * 10.0
        mf.process(baseline)

        # Inject spike
        spike_signal = np.array([100.0], dtype=np.float32)
        out = mf.process(spike_signal)

        # Spike should be rejected (median of [10,10,10,10,100] = 10)
        assert np.allclose(out[0], 10.0, atol=1e-5)

    def test_median_correctness(self):
        """Test that median is computed correctly."""
        import zpybci as zbci

        mf = zbci.MedianFilter(5)

        # Process 5 samples to fill window
        signal = np.array([5.0, 1.0, 3.0, 4.0, 2.0], dtype=np.float32)
        out = mf.process(signal)

        # After processing [5,1,3,4,2], next sample gives median of that window
        next_out = mf.process(np.array([6.0], dtype=np.float32))
        # Window is [1,3,4,2,6], sorted [1,2,3,4,6], median = 3
        assert np.allclose(next_out[0], 3.0, atol=1e-5)

    def test_reset(self):
        """Test that reset clears filter state."""
        import zpybci as zbci

        mf = zbci.MedianFilter(5)

        # Process some data
        mf.process(np.array([5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32))

        # Reset
        mf.reset()

        # Should be back to zero-padding phase
        out = mf.process(np.array([10.0], dtype=np.float32))
        # median([0,0,0,0,10]) = 0
        assert np.allclose(out[0], 0.0, atol=1e-5)

    def test_repr(self):
        """Test string representation."""
        import zpybci as zbci

        mf = zbci.MedianFilter(5)
        assert 'MedianFilter' in repr(mf)
        assert '5' in repr(mf)

    def test_constant_signal(self):
        """Test that constant signal passes through unchanged."""
        import zpybci as zbci

        mf = zbci.MedianFilter(5)
        signal = np.ones(20, dtype=np.float32) * 7.0
        filtered = mf.process(signal)

        # After window fills, should output constant value
        assert np.allclose(filtered[4:], 7.0, atol=1e-5)


class TestLmsFilter:
    """Tests for LmsFilter."""

    def test_import(self):
        """Test that LmsFilter can be imported."""
        import zpybci as zbci
        assert hasattr(zbci, 'LmsFilter')

    def test_create(self):
        """Test creating LmsFilter."""
        import zpybci as zbci

        lms = zbci.LmsFilter(taps=32, mu=0.01)
        assert lms.num_taps == 32
        assert np.isclose(lms.mu, 0.01)

    def test_invalid_params(self):
        """Test that invalid params raise errors."""
        import zpybci as zbci

        with pytest.raises(ValueError):
            zbci.LmsFilter(taps=0, mu=0.01)
        with pytest.raises(ValueError):
            zbci.LmsFilter(taps=32, mu=0.0)
        with pytest.raises(ValueError):
            zbci.LmsFilter(taps=32, mu=-0.01)

    def test_process_output_shape(self):
        """Test that process returns correct shapes."""
        import zpybci as zbci

        lms = zbci.LmsFilter(taps=32, mu=0.01)
        reference = np.random.randn(100).astype(np.float32)
        desired = np.random.randn(100).astype(np.float32)

        output, error = lms.process(reference, desired)

        assert isinstance(output, np.ndarray)
        assert isinstance(error, np.ndarray)
        assert output.dtype == np.float32
        assert error.dtype == np.float32
        assert len(output) == 100
        assert len(error) == 100

    def test_adaptation(self):
        """Test that LMS filter adapts to minimize error."""
        import zpybci as zbci

        lms = zbci.LmsFilter(taps=16, mu=0.05)

        # Input and desired (scaled input)
        reference = np.sin(np.arange(500) * 0.1).astype(np.float32)
        desired = reference * 2.0

        output, error = lms.process(reference, desired)

        # Error should decrease over time
        initial_error = np.mean(np.abs(error[:50]))
        final_error = np.mean(np.abs(error[-50:]))
        assert final_error < initial_error * 0.5

    def test_weights_accessible(self):
        """Test that weights are accessible."""
        import zpybci as zbci

        lms = zbci.LmsFilter(taps=16, mu=0.01)
        weights = lms.weights

        assert isinstance(weights, np.ndarray)
        assert len(weights) == 16
        # Initially all zeros
        assert np.allclose(weights, 0.0)

    def test_reset(self):
        """Test that reset clears state."""
        import zpybci as zbci

        lms = zbci.LmsFilter(taps=16, mu=0.01)

        # Process some data
        reference = np.random.randn(100).astype(np.float32)
        desired = np.random.randn(100).astype(np.float32)
        lms.process(reference, desired)

        # Weights should be non-zero
        assert not np.allclose(lms.weights, 0.0)

        # Reset
        lms.reset()

        # Weights should be zero again
        assert np.allclose(lms.weights, 0.0)

    def test_optimized_sizes(self):
        """Test that optimized sizes work."""
        import zpybci as zbci

        for taps in [8, 16, 32, 64]:
            lms = zbci.LmsFilter(taps=taps, mu=0.01)
            assert lms.num_taps == taps

            reference = np.random.randn(50).astype(np.float32)
            desired = np.random.randn(50).astype(np.float32)
            output, error = lms.process(reference, desired)
            assert len(output) == 50

    def test_dynamic_size(self):
        """Test non-optimized sizes use dynamic implementation."""
        import zpybci as zbci

        lms = zbci.LmsFilter(taps=10, mu=0.01)
        assert lms.num_taps == 10

        reference = np.random.randn(50).astype(np.float32)
        desired = np.random.randn(50).astype(np.float32)
        output, error = lms.process(reference, desired)
        assert len(output) == 50

    def test_repr(self):
        """Test string representation."""
        import zpybci as zbci

        lms = zbci.LmsFilter(taps=32, mu=0.01)
        assert 'LmsFilter' in repr(lms)
        assert '32' in repr(lms)


class TestNlmsFilter:
    """Tests for NlmsFilter."""

    def test_import(self):
        """Test that NlmsFilter can be imported."""
        import zpybci as zbci
        assert hasattr(zbci, 'NlmsFilter')

    def test_create(self):
        """Test creating NlmsFilter."""
        import zpybci as zbci

        nlms = zbci.NlmsFilter(taps=32, mu=0.5, epsilon=0.01)
        assert nlms.num_taps == 32
        assert np.isclose(nlms.mu, 0.5)
        assert np.isclose(nlms.epsilon, 0.01)

    def test_invalid_params(self):
        """Test that invalid params raise errors."""
        import zpybci as zbci

        with pytest.raises(ValueError):
            zbci.NlmsFilter(taps=0, mu=0.5, epsilon=0.01)
        with pytest.raises(ValueError):
            zbci.NlmsFilter(taps=32, mu=0.0, epsilon=0.01)
        with pytest.raises(ValueError):
            zbci.NlmsFilter(taps=32, mu=0.5, epsilon=0.0)

    def test_process_output_shape(self):
        """Test that process returns correct shapes."""
        import zpybci as zbci

        nlms = zbci.NlmsFilter(taps=32, mu=0.5, epsilon=0.01)
        reference = np.random.randn(100).astype(np.float32)
        desired = np.random.randn(100).astype(np.float32)

        output, error = nlms.process(reference, desired)

        assert isinstance(output, np.ndarray)
        assert isinstance(error, np.ndarray)
        assert len(output) == 100
        assert len(error) == 100

    def test_faster_convergence_than_lms(self):
        """Test that NLMS converges faster than LMS with varying amplitude."""
        import zpybci as zbci

        lms = zbci.LmsFilter(taps=16, mu=0.01)
        nlms = zbci.NlmsFilter(taps=16, mu=0.5, epsilon=0.01)

        # Varying amplitude signal
        t = np.arange(300)
        amplitude = 1.0 + 0.5 * np.sin(t * 0.05)
        reference = (amplitude * np.sin(t * 0.1)).astype(np.float32)
        desired = (reference * 2.0).astype(np.float32)

        _, lms_error = lms.process(reference, desired)
        _, nlms_error = nlms.process(reference, desired)

        # NLMS should have lower error in the latter half
        lms_final = np.mean(np.abs(lms_error[-50:]))
        nlms_final = np.mean(np.abs(nlms_error[-50:]))
        assert nlms_final < lms_final

    def test_reset(self):
        """Test that reset clears state."""
        import zpybci as zbci

        nlms = zbci.NlmsFilter(taps=16, mu=0.5, epsilon=0.01)

        reference = np.random.randn(100).astype(np.float32)
        desired = np.random.randn(100).astype(np.float32)
        nlms.process(reference, desired)

        assert not np.allclose(nlms.weights, 0.0)

        nlms.reset()

        assert np.allclose(nlms.weights, 0.0)

    def test_repr(self):
        """Test string representation."""
        import zpybci as zbci

        nlms = zbci.NlmsFilter(taps=32, mu=0.5, epsilon=0.01)
        assert 'NlmsFilter' in repr(nlms)
        assert '32' in repr(nlms)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

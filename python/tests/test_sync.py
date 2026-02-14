"""Tests for clock synchronization Python bindings."""
import numpy as np
import pytest


class TestClockOffset:
    """Tests for ClockOffset."""

    def test_import(self):
        import npyci as npy
        assert hasattr(npy, 'ClockOffset')

    def test_create(self):
        import npyci as npy
        offset = npy.ClockOffset(offset=0.001, local_time=10.0, rtt=0.02)
        assert np.isclose(offset.offset, 0.001)
        assert np.isclose(offset.local_time, 10.0)
        assert np.isclose(offset.rtt, 0.02)

    def test_quality(self):
        import npyci as npy
        offset = npy.ClockOffset(offset=0.001, local_time=10.0, rtt=0.02)
        assert offset.quality > 0.0
        # Quality is 1/rtt
        assert np.isclose(offset.quality, 1.0 / 0.02)

    def test_from_ntp(self):
        import npyci as npy
        # Symmetric NTP exchange
        offset = npy.ClockOffset.from_ntp(t1=1.0, t2=1.005, t3=1.006, t4=1.010)
        assert isinstance(offset.offset, float)
        assert isinstance(offset.rtt, float)

    def test_repr(self):
        import npyci as npy
        offset = npy.ClockOffset(offset=0.001, local_time=10.0, rtt=0.02)
        r = repr(offset)
        assert 'ClockOffset' in r


class TestSampleClock:
    """Tests for SampleClock."""

    def test_import(self):
        import npyci as npy
        assert hasattr(npy, 'SampleClock')

    def test_create(self):
        import npyci as npy
        clock = npy.SampleClock(start_time=0.0, sample_rate=250.0)
        assert np.isclose(clock.start_time, 0.0)
        assert np.isclose(clock.sample_rate, 250.0)

    def test_sample_to_time(self):
        import npyci as npy
        clock = npy.SampleClock(start_time=0.0, sample_rate=250.0)
        # 250 samples at 250 Hz = 1 second
        assert np.isclose(clock.sample_to_time(250), 1.0)
        assert np.isclose(clock.sample_to_time(0), 0.0)
        assert np.isclose(clock.sample_to_time(1000), 4.0)

    def test_time_to_sample(self):
        import npyci as npy
        clock = npy.SampleClock(start_time=0.0, sample_rate=250.0)
        assert clock.time_to_sample(1.0) == 250
        assert clock.time_to_sample(0.0) == 0

    def test_time_to_sample_frac(self):
        import npyci as npy
        clock = npy.SampleClock(start_time=0.0, sample_rate=250.0)
        assert np.isclose(clock.time_to_sample_frac(1.0), 250.0)

    def test_with_offset_start(self):
        import npyci as npy
        clock = npy.SampleClock(start_time=5.0, sample_rate=100.0)
        assert np.isclose(clock.sample_to_time(0), 5.0)
        assert np.isclose(clock.sample_to_time(100), 6.0)

    def test_repr(self):
        import npyci as npy
        clock = npy.SampleClock(start_time=0.0, sample_rate=250.0)
        assert 'SampleClock' in repr(clock)


class TestLinearDrift:
    """Tests for LinearDrift."""

    def test_import(self):
        import npyci as npy
        assert hasattr(npy, 'LinearDrift')

    def test_create(self):
        import npyci as npy
        drift = npy.LinearDrift()
        assert drift.count == 0

    def test_add_measurement(self):
        import npyci as npy
        drift = npy.LinearDrift()
        drift.add_measurement(0.0, 0.0)
        drift.add_measurement(10.0, 0.001)
        assert drift.count == 2

    def test_add_offset(self):
        import npyci as npy
        drift = npy.LinearDrift()
        offset = npy.ClockOffset(offset=0.001, local_time=10.0, rtt=0.02)
        drift.add_offset(offset)
        assert drift.count == 1

    def test_correct(self):
        import npyci as npy
        drift = npy.LinearDrift()
        # Perfect linear drift: 0.001s per 10s
        drift.add_measurement(0.0, 0.0)
        drift.add_measurement(10.0, 0.001)
        drift.add_measurement(20.0, 0.002)
        corrected = drift.correct(10.0)
        assert isinstance(corrected, float)

    def test_slope_intercept(self):
        import npyci as npy
        drift = npy.LinearDrift()
        drift.add_measurement(0.0, 0.0)
        drift.add_measurement(10.0, 0.01)
        assert isinstance(drift.slope, float)
        assert isinstance(drift.intercept, float)

    def test_reset(self):
        import npyci as npy
        drift = npy.LinearDrift()
        drift.add_measurement(0.0, 0.0)
        drift.add_measurement(10.0, 0.001)
        assert drift.count == 2
        drift.reset()
        assert drift.count == 0

    def test_repr(self):
        import npyci as npy
        drift = npy.LinearDrift()
        assert 'LinearDrift' in repr(drift)


class TestOffsetBuffer:
    """Tests for OffsetBuffer."""

    def test_import(self):
        import npyci as npy
        assert hasattr(npy, 'OffsetBuffer')

    def test_create_default(self):
        import npyci as npy
        buf = npy.OffsetBuffer()
        assert buf.count == 0

    def test_create_sizes(self):
        import npyci as npy
        for size in [8, 16, 32, 64, 128]:
            buf = npy.OffsetBuffer(capacity=size)
            assert buf.count == 0

    def test_invalid_size(self):
        import npyci as npy
        with pytest.raises(ValueError):
            npy.OffsetBuffer(capacity=10)

    def test_add_and_retrieve(self):
        import npyci as npy
        buf = npy.OffsetBuffer(capacity=8)
        offset = npy.ClockOffset(offset=0.001, local_time=10.0, rtt=0.02)
        buf.add(offset)
        assert buf.count == 1

        best = buf.best_offset()
        assert best is not None
        assert np.isclose(best.offset, 0.001)

        latest = buf.latest_offset()
        assert latest is not None
        assert np.isclose(latest.offset, 0.001)

    def test_best_offset_selects_lowest_rtt(self):
        import npyci as npy
        buf = npy.OffsetBuffer(capacity=8)
        buf.add(npy.ClockOffset(offset=0.001, local_time=1.0, rtt=0.10))
        buf.add(npy.ClockOffset(offset=0.002, local_time=2.0, rtt=0.01))  # Best
        buf.add(npy.ClockOffset(offset=0.003, local_time=3.0, rtt=0.05))
        best = buf.best_offset()
        assert best is not None
        assert np.isclose(best.rtt, 0.01)

    def test_latest_offset(self):
        import npyci as npy
        buf = npy.OffsetBuffer(capacity=8)
        buf.add(npy.ClockOffset(offset=0.001, local_time=1.0, rtt=0.02))
        buf.add(npy.ClockOffset(offset=0.005, local_time=2.0, rtt=0.02))
        latest = buf.latest_offset()
        assert latest is not None
        assert np.isclose(latest.offset, 0.005)

    def test_median_offset(self):
        import npyci as npy
        buf = npy.OffsetBuffer(capacity=8)
        for i in range(5):
            buf.add(npy.ClockOffset(offset=float(i) * 0.001, local_time=float(i), rtt=0.02))
        median = buf.median_offset()
        assert median is not None

    def test_empty_buffer_returns_none(self):
        import npyci as npy
        buf = npy.OffsetBuffer(capacity=8)
        assert buf.best_offset() is None
        assert buf.latest_offset() is None
        assert buf.median_offset() is None

    def test_reset(self):
        import npyci as npy
        buf = npy.OffsetBuffer(capacity=8)
        buf.add(npy.ClockOffset(offset=0.001, local_time=1.0, rtt=0.02))
        assert buf.count == 1
        buf.reset()
        assert buf.count == 0

    def test_repr(self):
        import npyci as npy
        buf = npy.OffsetBuffer(capacity=8)
        assert 'OffsetBuffer' in repr(buf)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

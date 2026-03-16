"""Tests for template subtraction bindings."""

import numpy as np
import pytest

import zpybci as zbci


class TestTemplateSubtractorCreation:
    def test_basic_creation(self):
        sub = zbci.TemplateSubtractor(window_len=8, max_templates=4)
        assert sub.n_templates == 0

    def test_with_max_iter(self):
        sub = zbci.TemplateSubtractor(window_len=16, max_templates=8, max_iter=100)
        assert sub.n_templates == 0

    def test_all_supported_sizes(self):
        for w, n in [(8, 4), (8, 8), (16, 4), (16, 8), (16, 16),
                     (32, 8), (32, 16), (64, 16), (64, 32)]:
            sub = zbci.TemplateSubtractor(window_len=w, max_templates=n)
            assert sub.n_templates == 0

    def test_invalid_window_len(self):
        with pytest.raises(ValueError):
            zbci.TemplateSubtractor(window_len=7, max_templates=4)

    def test_invalid_max_templates(self):
        with pytest.raises(ValueError):
            zbci.TemplateSubtractor(window_len=8, max_templates=3)

    def test_repr(self):
        sub = zbci.TemplateSubtractor(window_len=8, max_templates=4)
        r = repr(sub)
        assert "TemplateSubtractor" in r
        assert "8" in r


class TestAddTemplate:
    def test_add_one(self):
        sub = zbci.TemplateSubtractor(window_len=8, max_templates=4)
        template = np.array([-0.5, -2.0, -5.0, -3.0, -1.0, 0.5, 1.0, 0.3])
        idx = sub.add_template(template, 0.5, 2.0)
        assert idx == 0
        assert sub.n_templates == 1

    def test_add_multiple(self):
        sub = zbci.TemplateSubtractor(window_len=8, max_templates=4)
        for i in range(4):
            t = np.ones(8) * (i + 1.0)
            idx = sub.add_template(t, 0.5, 2.0)
            assert idx == i
        assert sub.n_templates == 4

    def test_wrong_length(self):
        sub = zbci.TemplateSubtractor(window_len=8, max_templates=4)
        with pytest.raises(ValueError):
            sub.add_template(np.ones(4), 0.5, 2.0)

    def test_template_full(self):
        sub = zbci.TemplateSubtractor(window_len=8, max_templates=4)
        for i in range(4):
            sub.add_template(np.ones(8) * (i + 1.0), 0.5, 2.0)
        with pytest.raises(ValueError):
            sub.add_template(np.ones(8), 0.5, 2.0)


class TestPeel:
    def test_single_spike(self):
        sub = zbci.TemplateSubtractor(window_len=8, max_templates=4)
        template = np.array([-0.5, -2.0, -5.0, -3.0, -1.0, 0.5, 1.0, 0.3])
        sub.add_template(template, 0.5, 2.0)

        data = np.zeros(24)
        amp = 1.5
        data[4:12] = amp * template
        spike_times = np.array([6], dtype=np.int64)
        results = sub.peel(data, spike_times, pre_samples=2)
        assert len(results) >= 1
        assert results[0]["template_id"] == 0
        assert abs(results[0]["amplitude"] - 1.5) < 0.1

    def test_no_templates_returns_empty(self):
        sub = zbci.TemplateSubtractor(window_len=8, max_templates=4)
        data = np.zeros(24)
        results = sub.peel(data, np.array([6], dtype=np.int64), pre_samples=2)
        assert len(results) == 0

    def test_amplitude_rejection(self):
        sub = zbci.TemplateSubtractor(window_len=8, max_templates=4)
        template = np.array([-1.0, -3.0, -5.0, -3.0, -1.0, 0.5, 1.0, 0.3])
        sub.add_template(template, 0.8, 1.2)
        data = np.zeros(24)
        data[4:12] = 2.0 * template
        results = sub.peel(data, np.array([6], dtype=np.int64), pre_samples=2)
        assert len(results) == 0

    def test_set_amplitude_bounds(self):
        sub = zbci.TemplateSubtractor(window_len=8, max_templates=4)
        template = np.array([-1.0, -3.0, -5.0, -3.0, -1.0, 0.5, 1.0, 0.3])
        sub.add_template(template, 0.8, 1.2)
        sub.set_amplitude_bounds(0, 0.5, 3.0)
        data = np.zeros(24)
        data[4:12] = 1.5 * template
        results = sub.peel(data, np.array([6], dtype=np.int64), pre_samples=2)
        assert len(results) >= 1

    def test_larger_window(self):
        sub = zbci.TemplateSubtractor(window_len=16, max_templates=8)
        template = np.zeros(16)
        template[4:8] = [-1.0, -5.0, -3.0, -1.0]
        sub.add_template(template, 0.5, 2.0)
        data = np.zeros(48)
        data[10:26] = 1.0 * template
        results = sub.peel(data, np.array([14], dtype=np.int64), pre_samples=4)
        assert len(results) >= 1

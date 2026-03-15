"""Comprehensive tests for gpu_top.py."""

import threading
import time
from collections import deque
from unittest.mock import patch

import pytest

import gpu_top as gt


# ── parse_field ────────────────────────────────────────────────────────────────

class TestParseField:
    def test_empty_string(self):
        assert gt.parse_field("") is None

    def test_whitespace_only(self):
        assert gt.parse_field("   ") is None

    def test_na_lowercase(self):
        assert gt.parse_field("n/a") is None

    def test_na_uppercase(self):
        assert gt.parse_field("N/A") is None

    def test_na_bracketed(self):
        assert gt.parse_field("[N/A]") is None

    def test_not_found_bracketed(self):
        assert gt.parse_field("[Not Found]") is None

    def test_not_supported(self):
        assert gt.parse_field("Not Supported") is None

    def test_not_supported_bracketed(self):
        assert gt.parse_field("[Not Supported]") is None

    def test_valid_float(self):
        assert gt.parse_field("3.14") == pytest.approx(3.14)

    def test_valid_int(self):
        assert gt.parse_field("42", cast=int) == 42

    def test_percent_suffix(self):
        assert gt.parse_field("75 %") == pytest.approx(75.0)

    def test_percent_suffix_nospace(self):
        assert gt.parse_field("75%") == pytest.approx(75.0)

    def test_watt_suffix(self):
        assert gt.parse_field("33.25 W") == pytest.approx(33.25)

    def test_mhz_suffix(self):
        assert gt.parse_field("337 MHz") == pytest.approx(337.0)

    def test_mib_suffix(self):
        assert gt.parse_field("23746 MiB") == pytest.approx(23746.0)

    def test_celsius_suffix(self):
        assert gt.parse_field("36 C") == pytest.approx(36.0)

    def test_custom_default(self):
        assert gt.parse_field("N/A", default=-1) == -1

    def test_invalid_string_returns_default(self):
        assert gt.parse_field("not_a_number") is None

    def test_invalid_string_custom_default(self):
        assert gt.parse_field("garbage", default=0.0) == 0.0

    def test_leading_trailing_whitespace(self):
        assert gt.parse_field("  42.0  ") == pytest.approx(42.0)

    def test_zero_is_valid(self):
        assert gt.parse_field("0") == pytest.approx(0.0)

    def test_negative_value(self):
        assert gt.parse_field("-5.0") == pytest.approx(-5.0)


# ── make_bar ───────────────────────────────────────────────────────────────────

class TestMakeBar:
    def test_none_value_shows_dashes(self):
        bar = gt.make_bar(None, 100, 10, "green")
        assert bar.plain == "━" * 10

    def test_zero_total_shows_dashes(self):
        bar = gt.make_bar(50, 0, 10, "green")
        assert bar.plain == "━" * 10

    def test_negative_total_shows_dashes(self):
        bar = gt.make_bar(50, -1, 10, "green")
        assert bar.plain == "━" * 10

    def test_full_bar(self):
        bar = gt.make_bar(100, 100, 10, "green")
        assert bar.plain == "█" * 10

    def test_empty_bar(self):
        bar = gt.make_bar(0, 100, 10, "green")
        assert bar.plain == "░" * 10

    def test_half_bar_length(self):
        bar = gt.make_bar(50, 100, 10, "green")
        assert len(bar.plain) == 10

    def test_over_max_clamped(self):
        bar = gt.make_bar(200, 100, 10, "green")
        assert bar.plain == "█" * 10

    def test_negative_value_clamped(self):
        bar = gt.make_bar(-10, 100, 10, "green")
        assert bar.plain == "░" * 10

    def test_width_respected(self):
        for w in [1, 5, 20, 50]:
            bar = gt.make_bar(50, 100, w, "green")
            assert len(bar.plain) == w

    def test_fractional_block_included(self):
        # 33% of 30 = 10 cells filled; the remaining fraction should use a
        # sub-block character, so the bar should not be all full or all empty
        bar = gt.make_bar(33, 100, 30, "green")
        plain = bar.plain
        assert "█" in plain or any(c in plain for c in "▏▎▍▌▋▊▉")


# ── color helpers ──────────────────────────────────────────────────────────────

class TestUtilColor:
    def test_none(self):
        assert gt._util_color(None) == "white"

    def test_zero(self):
        assert gt._util_color(0) == "bright_green"

    def test_boundary_low_mid(self):
        assert gt._util_color(59) == "bright_green"
        assert gt._util_color(60) == "yellow"

    def test_boundary_mid_high(self):
        assert gt._util_color(84) == "yellow"
        assert gt._util_color(85) == "bright_red"

    def test_max(self):
        assert gt._util_color(100) == "bright_red"


class TestTempColor:
    def test_none(self):
        assert gt._temp_color(None) == "white"

    def test_cool(self):
        assert gt._temp_color(30) == "bright_green"

    def test_boundary_cool_warm(self):
        assert gt._temp_color(59) == "bright_green"
        assert gt._temp_color(60) == "yellow"    # 60 is no longer < 60

    def test_warm(self):
        assert gt._temp_color(70) == "yellow"

    def test_boundary_warm_hot(self):
        assert gt._temp_color(79) == "yellow"
        assert gt._temp_color(80) == "bright_red"

    def test_hot(self):
        assert gt._temp_color(95) == "bright_red"


class TestPowerColor:
    def test_none_draw(self):
        assert gt._power_color(None, 575) == "bright_blue"

    def test_none_limit(self):
        assert gt._power_color(100, None) == "bright_blue"

    def test_zero_limit(self):
        assert gt._power_color(100, 0) == "bright_blue"

    def test_low_ratio(self):
        assert gt._power_color(100, 575) == "bright_blue"   # ~17%

    def test_boundary_low_mid(self):
        assert gt._power_color(344, 575) == "bright_blue"   # 59.8%
        assert gt._power_color(345, 575) == "yellow"        # 60%

    def test_boundary_mid_high(self):
        assert gt._power_color(488, 575) == "yellow"        # ~84.9%
        assert gt._power_color(490, 575) == "bright_red"    # ~85.2%

    def test_at_limit(self):
        assert gt._power_color(575, 575) == "bright_red"


# ── _fmt ───────────────────────────────────────────────────────────────────────

class TestFmt:
    def test_none(self):
        assert gt._fmt(None) == "---"

    def test_zero(self):
        assert gt._fmt(0.0) == "0"

    def test_float(self):
        assert gt._fmt(3.7) == "4"

    def test_with_unit(self):
        assert gt._fmt(42.0, ".0f", "W") == "42W"

    def test_decimal_spec(self):
        assert gt._fmt(3.14159, ".2f") == "3.14"


# ── _sparkline_rows ────────────────────────────────────────────────────────────

class TestSparklineRows:
    def test_returns_correct_row_count(self):
        rows = gt._sparkline_rows([50], 1, 100, "green", rows=4)
        assert len(rows) == 4

    def test_empty_values_all_padding(self):
        rows = gt._sparkline_rows([], 10, 100, "green", rows=2)
        assert len(rows) == 2
        for row in rows:
            assert len(row.plain) == 10
            assert all(c == "░" for c in row.plain)

    def test_none_values_show_filler(self):
        rows = gt._sparkline_rows([None, None, None], 3, 100, "green", rows=2)
        for row in rows:
            assert all(c == "░" for c in row.plain)

    def test_zero_max_val_shows_filler(self):
        rows = gt._sparkline_rows([50, 100], 2, 0, "green", rows=2)
        for row in rows:
            assert all(c == "░" for c in row.plain)

    def test_full_value_fills_bottom_row(self):
        # 100% fills all rows; bottom row should be all full blocks
        rows = gt._sparkline_rows([100] * 10, 10, 100, "green", rows=4)
        bottom = rows[-1].plain
        assert all(c == "█" for c in bottom)

    def test_zero_value_all_empty(self):
        rows = gt._sparkline_rows([0] * 10, 10, 100, "green", rows=4)
        for row in rows:
            assert all(c == "░" for c in row.plain)

    def test_width_respected(self):
        rows = gt._sparkline_rows([50] * 5, 20, 100, "green", rows=3)
        for row in rows:
            assert len(row.plain) == 20   # 15 padding + 5 samples

    def test_only_latest_width_samples_used(self):
        # 15 values, width=10: only last 10 should appear
        values = [0] * 5 + [100] * 10
        rows = gt._sparkline_rows(values, 10, 100, "green", rows=1)
        bottom = rows[0].plain
        assert all(c == "█" for c in bottom)

    def test_vblocks_index_in_range(self):
        # Many fractional values should never produce an IndexError
        import math
        values = [50 + 49 * math.sin(i * 0.3) for i in range(40)]
        rows = gt._sparkline_rows(values, 40, 100, "green", rows=4)
        for row in rows:
            for ch in row.plain:
                assert ch in gt._VBLOCKS or ch == "░"

    def test_over_max_clamped(self):
        rows = gt._sparkline_rows([999], 1, 100, "green", rows=4)
        bottom = rows[-1].plain
        assert bottom == "█"


# ── query_gpus / query_procs (mocked subprocess) ──────────────────────────────

class TestQueryGpus:
    def _csv(self, *fields):
        return ", ".join(str(f) for f in fields) + "\n"

    @patch("gpu_top.subprocess.check_output")
    def test_normal_parse(self, mock_sub):
        mock_sub.return_value = self._csv(
            0, "NVIDIA RTX 5090", "GPU-uuid-abc",
            2, 8, 23746, 32607, 36, "33.25", "575.00", 0, 337, 405, "595.79",
        )
        gpus = gt.query_gpus()
        assert len(gpus) == 1
        g = gpus[0]
        assert g.index == 0
        assert g.name == "NVIDIA RTX 5090"
        assert g.util_gpu == pytest.approx(2.0)
        assert g.mem_used == pytest.approx(23746.0)
        assert g.mem_total == pytest.approx(32607.0)
        assert g.temp == pytest.approx(36.0)
        assert g.power_draw == pytest.approx(33.25)
        assert g.power_limit == pytest.approx(575.0)
        assert g.driver == "595.79"

    @patch("gpu_top.subprocess.check_output")
    def test_na_fields_return_none(self, mock_sub):
        mock_sub.return_value = self._csv(
            0, "NVIDIA GPU", "GPU-uuid",
            "N/A", "N/A", "N/A", 32607, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A", "595.79",
        )
        gpus = gt.query_gpus()
        assert len(gpus) == 1
        g = gpus[0]
        assert g.util_gpu is None
        assert g.util_mem is None
        assert g.mem_used is None
        assert g.temp is None
        assert g.power_draw is None
        assert g.fan is None

    @patch("gpu_top.subprocess.check_output")
    def test_not_found_fields(self, mock_sub):
        mock_sub.return_value = self._csv(
            0, "NVIDIA GPU", "GPU-uuid",
            "[Not Found]", 8, 23746, 32607, 36, 33.25, 575, "[N/A]", 337, 405, "595.79",
        )
        gpus = gt.query_gpus()
        assert gpus[0].util_gpu is None
        assert gpus[0].fan is None

    @patch("gpu_top.subprocess.check_output")
    def test_subprocess_failure_returns_empty(self, mock_sub):
        mock_sub.side_effect = Exception("nvidia-smi not found")
        assert gt.query_gpus() == []

    @patch("gpu_top.subprocess.check_output")
    def test_empty_output_returns_empty(self, mock_sub):
        mock_sub.return_value = ""
        assert gt.query_gpus() == []

    @patch("gpu_top.subprocess.check_output")
    def test_short_line_skipped(self, mock_sub):
        mock_sub.return_value = "0, NVIDIA GPU\n"   # too few fields
        assert gt.query_gpus() == []

    @patch("gpu_top.subprocess.check_output")
    def test_multiple_gpus(self, mock_sub):
        def row(i):
            return self._csv(i, f"GPU {i}", f"GPU-uuid-{i}",
                             10, 5, 1000, 8000, 50, 100, 250, 30, 1000, 2000, "595.79")
        mock_sub.return_value = row(0) + row(1)
        gpus = gt.query_gpus()
        assert len(gpus) == 2
        assert gpus[0].index == 0
        assert gpus[1].index == 1


class TestQueryProcs:
    @patch("gpu_top.subprocess.check_output")
    def test_normal_parse(self, mock_sub):
        mock_sub.return_value = "GPU-uuid, 1234, my_process, 512\n"
        procs = gt.query_procs()
        assert len(procs) == 1
        p = procs[0]
        assert p.gpu_uuid == "GPU-uuid"
        assert p.pid == 1234
        assert p.name == "my_process"
        assert p.mem_used == pytest.approx(512.0)

    @patch("gpu_top.subprocess.check_output")
    def test_not_found_name(self, mock_sub):
        mock_sub.return_value = "GPU-uuid, 1171, [Not Found], [N/A]\n"
        procs = gt.query_procs()
        assert len(procs) == 1
        assert procs[0].name == "?"
        assert procs[0].mem_used is None

    @patch("gpu_top.subprocess.check_output")
    def test_na_memory(self, mock_sub):
        mock_sub.return_value = "GPU-uuid, 42, myapp, N/A\n"
        procs = gt.query_procs()
        assert procs[0].mem_used is None

    @patch("gpu_top.subprocess.check_output")
    def test_subprocess_failure_returns_empty(self, mock_sub):
        mock_sub.side_effect = OSError("no nvidia-smi")
        assert gt.query_procs() == []

    @patch("gpu_top.subprocess.check_output")
    def test_empty_output(self, mock_sub):
        mock_sub.return_value = ""
        assert gt.query_procs() == []

    @patch("gpu_top.subprocess.check_output")
    def test_blank_lines_skipped(self, mock_sub):
        mock_sub.return_value = "\n\nGPU-uuid, 1, app, 100\n\n"
        procs = gt.query_procs()
        assert len(procs) == 1

    @patch("gpu_top.subprocess.check_output")
    def test_short_line_skipped(self, mock_sub):
        mock_sub.return_value = "GPU-uuid, 1\n"
        assert gt.query_procs() == []


# ── State ──────────────────────────────────────────────────────────────────────

class TestState:
    def test_initial_values(self):
        s = gt.State(1.0)
        assert s.gpus == []
        assert s.procs == []
        assert s.history == {}
        assert s.tick == 0
        assert not s.stale
        assert s.poll_interval == pytest.approx(1.0)

    def test_snapshot_returns_copies(self):
        s = gt.State(1.0)
        s.gpus = [gt.GPUInfo(index=0)]
        gpus, procs, hist, cpu, tick, stale, pi = s.snapshot()
        gpus.append(gt.GPUInfo(index=99))
        assert len(s.gpus) == 1   # original unchanged

    def test_snapshot_history_is_copy(self):
        s = gt.State(1.0)
        s.history[0] = deque([gt.HistorySample(util_gpu=50, power_w=100, mem_pct=30, ts=0.0)])
        _, _, hist, _, _, _, _ = s.snapshot()
        hist[0].append(gt.HistorySample(util_gpu=99, power_w=0, mem_pct=0, ts=0.0))
        assert len(s.history[0]) == 1   # original unchanged

    def test_poll_interval_in_snapshot(self):
        s = gt.State(2.5)
        _, _, _, _, _, _, pi = s.snapshot()
        assert pi == pytest.approx(2.5)

    def test_snapshot_cpu_initially_none(self):
        s = gt.State(1.0)
        _, _, _, cpu, _, _, _ = s.snapshot()
        assert cpu is None

    def test_snapshot_thread_safe(self):
        """Concurrent reads and writes should not raise."""
        s = gt.State(0.1)
        errors = []

        def writer():
            for _ in range(100):
                with s.lock:
                    s.tick += 1
                    s.poll_interval = 1.0

        def reader():
            for _ in range(100):
                try:
                    s.snapshot()
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ── History accumulation in _collect_loop ──────────────────────────────────────

class TestCollectLoopHistory:
    @patch("gpu_top.query_procs", return_value=[])
    @patch("gpu_top.query_gpus")
    def test_history_appended_each_tick(self, mock_gpus, mock_procs):
        gpu = gt.GPUInfo(
            index=0, util_gpu=42.0, power_draw=100.0,
            mem_used=1000.0, mem_total=4000.0,
        )
        mock_gpus.return_value = [gpu]

        state = gt.State(poll_interval=0.05)
        stop = threading.Event()
        t = threading.Thread(target=gt._collect_loop, args=(state, stop), daemon=True)
        t.start()
        time.sleep(0.3)
        stop.set()
        t.join(timeout=1.0)

        with state.lock:
            hist = list(state.history.get(0, []))
        assert len(hist) >= 2
        sample = hist[0]
        assert sample.util_gpu == pytest.approx(42.0)
        assert sample.power_w == pytest.approx(100.0)
        assert sample.mem_pct == pytest.approx(25.0)   # 1000/4000*100

    @patch("gpu_top.query_procs", return_value=[])
    @patch("gpu_top.query_gpus")
    def test_history_mem_pct_none_when_mem_total_zero(self, mock_gpus, _):
        gpu = gt.GPUInfo(index=0, mem_used=500.0, mem_total=0.0)
        mock_gpus.return_value = [gpu]

        state = gt.State(poll_interval=0.05)
        stop = threading.Event()
        t = threading.Thread(target=gt._collect_loop, args=(state, stop), daemon=True)
        t.start()
        time.sleep(0.15)
        stop.set()
        t.join(timeout=1.0)

        with state.lock:
            hist = list(state.history.get(0, []))
        assert any(s.mem_pct is None for s in hist)

    @patch("gpu_top.query_procs", return_value=[])
    @patch("gpu_top.query_gpus")
    def test_history_mem_pct_none_when_mem_total_none(self, mock_gpus, _):
        gpu = gt.GPUInfo(index=0, mem_used=500.0, mem_total=None)
        mock_gpus.return_value = [gpu]

        state = gt.State(poll_interval=0.05)
        stop = threading.Event()
        t = threading.Thread(target=gt._collect_loop, args=(state, stop), daemon=True)
        t.start()
        time.sleep(0.15)
        stop.set()
        t.join(timeout=1.0)

        with state.lock:
            hist = list(state.history.get(0, []))
        assert all(s.mem_pct is None for s in hist)

    @patch("gpu_top.query_procs", return_value=[])
    @patch("gpu_top.query_gpus", return_value=[])
    def test_stale_set_on_empty_result(self, mock_gpus, _):
        mock_gpus.return_value = []
        # Simulate a query failure by making it return None via side_effect
        mock_gpus.side_effect = Exception("smi error")

        state = gt.State(poll_interval=0.05)
        stop = threading.Event()
        t = threading.Thread(target=gt._collect_loop, args=(state, stop), daemon=True)
        t.start()
        time.sleep(0.15)
        stop.set()
        t.join(timeout=1.0)

        with state.lock:
            assert state.stale

    @patch("gpu_top.query_procs", return_value=[])
    @patch("gpu_top.query_gpus")
    def test_history_deque_maxlen(self, mock_gpus, _):
        mock_gpus.return_value = [gt.GPUInfo(index=0)]
        state = gt.State(poll_interval=0.01)
        stop = threading.Event()
        t = threading.Thread(target=gt._collect_loop, args=(state, stop), daemon=True)
        t.start()
        # Run long enough to exceed HISTORY_LEN at 0.01s interval (120 * 0.01 = 1.2s)
        # Just check it doesn't grow beyond maxlen
        time.sleep(0.2)
        stop.set()
        t.join(timeout=1.0)

        with state.lock:
            hist = state.history.get(0)
        assert hist is not None
        assert len(hist) <= gt.HISTORY_LEN


# ── build_history_panel ────────────────────────────────────────────────────────

class TestBuildHistoryPanel:
    def _gpu(self, power_limit=575.0):
        return gt.GPUInfo(index=0, name="Test GPU", power_limit=power_limit)

    def _hist(self, n=30, base_ts=0.0):
        return [
            gt.HistorySample(util_gpu=float(i % 100), power_w=float(i * 5 % 575),
                             mem_pct=float(i % 100), ts=base_ts + i)
            for i in range(n)
        ]

    def test_renders_without_error(self):
        panel = gt.build_history_panel(self._gpu(), self._hist(), console_width=120)
        assert panel is not None

    def test_single_sample_no_crash(self):
        hist = [gt.HistorySample(util_gpu=50, power_w=100, mem_pct=30, ts=1000.0)]
        panel = gt.build_history_panel(self._gpu(), hist, console_width=120)
        assert panel is not None

    def test_all_none_fields_no_crash(self):
        hist = [
            gt.HistorySample(util_gpu=None, power_w=None, mem_pct=None, ts=float(i))
            for i in range(10)
        ]
        panel = gt.build_history_panel(self._gpu(), hist, console_width=120)
        assert panel is not None

    def test_span_label_seconds(self):
        hist = self._hist(n=10, base_ts=1000.0)   # span = 9s
        panel = gt.build_history_panel(self._gpu(), hist, console_width=120)
        from rich.console import Console
        from io import StringIO
        buf = StringIO()
        Console(file=buf, width=200).print(panel)
        output = buf.getvalue()
        assert "9s ago" in output

    def test_span_label_minutes(self):
        hist = self._hist(n=90, base_ts=1000.0)   # span = 89s → 1m29s
        panel = gt.build_history_panel(self._gpu(), hist, console_width=120)
        from rich.console import Console
        from io import StringIO
        buf = StringIO()
        Console(file=buf, width=200).print(panel)
        output = buf.getvalue()
        assert "1m" in output

    def test_power_limit_none_falls_back(self):
        # When power_limit is None, max_power should fall back to actual values
        hist = [
            gt.HistorySample(util_gpu=50, power_w=200.0, mem_pct=50, ts=float(i))
            for i in range(5)
        ]
        panel = gt.build_history_panel(self._gpu(power_limit=None), hist, console_width=120)
        assert panel is not None

    def test_narrow_console_no_crash(self):
        panel = gt.build_history_panel(self._gpu(), self._hist(), console_width=30)
        assert panel is not None


# ── build_gpu_panel ────────────────────────────────────────────────────────────

class TestBuildGpuPanel:
    def test_all_none_fields(self):
        g = gt.GPUInfo(index=0, name="Test GPU")
        panel = gt.build_gpu_panel(g, bar_width=20)
        from io import StringIO
        from rich.console import Console
        buf = StringIO()
        Console(file=buf, width=120).print(panel)
        assert "---" in buf.getvalue()

    def test_normal_render(self):
        g = gt.GPUInfo(
            index=0, name="RTX 5090", util_gpu=75.0, mem_used=16000.0,
            mem_total=32000.0, temp=65.0, power_draw=300.0, power_limit=575.0,
            fan=50.0, clk_gfx=2500.0, clk_mem=12000.0,
        )
        panel = gt.build_gpu_panel(g, bar_width=20)
        from io import StringIO
        from rich.console import Console
        buf = StringIO()
        Console(file=buf, width=120).print(panel)
        output = buf.getvalue()
        assert "75%" in output or "75" in output
        assert "65" in output    # temp


# ── build_proc_panel ───────────────────────────────────────────────────────────

class TestBuildProcPanel:
    def test_no_procs_shows_placeholder(self):
        panel = gt.build_proc_panel([], [])
        from io import StringIO
        from rich.console import Console
        buf = StringIO()
        Console(file=buf, width=120).print(panel)
        assert "No active compute processes" in buf.getvalue()

    def test_proc_with_na_memory(self):
        gpu = gt.GPUInfo(index=0, uuid="GPU-abc")
        proc = gt.ProcInfo(gpu_uuid="GPU-abc", pid=123, name="myapp", mem_used=None)
        panel = gt.build_proc_panel([gpu], [proc])
        from io import StringIO
        from rich.console import Console
        buf = StringIO()
        Console(file=buf, width=120).print(panel)
        output = buf.getvalue()
        assert "123" in output
        assert "N/A" in output

    def test_proc_unknown_gpu_shows_question(self):
        gpu = gt.GPUInfo(index=0, uuid="GPU-known")
        proc = gt.ProcInfo(gpu_uuid="GPU-unknown", pid=1, name="app", mem_used=512.0)
        panel = gt.build_proc_panel([gpu], [proc])
        from io import StringIO
        from rich.console import Console
        buf = StringIO()
        Console(file=buf, width=120).print(panel)
        assert "?" in buf.getvalue()

    def test_unknown_process_name(self):
        gpu = gt.GPUInfo(index=0, uuid="GPU-abc")
        proc = gt.ProcInfo(gpu_uuid="GPU-abc", pid=1, name="?", mem_used=100.0)
        panel = gt.build_proc_panel([gpu], [proc])
        from io import StringIO
        from rich.console import Console
        buf = StringIO()
        Console(file=buf, width=120).print(panel)
        assert "unknown" in buf.getvalue()


# ── build_renderable ───────────────────────────────────────────────────────────

class TestBuildRenderable:
    def test_empty_gpus_no_crash(self):
        r = gt.build_renderable([], [], {}, None, False, 1.0, 30, 120)
        assert r is not None

    def test_with_history(self):
        gpu = gt.GPUInfo(index=0, name="Test GPU", power_limit=575.0)
        hist = {0: [
            gt.HistorySample(util_gpu=50.0, power_w=100.0, mem_pct=25.0, ts=float(i))
            for i in range(10)
        ]}
        r = gt.build_renderable([gpu], [], hist, None, False, 1.0, 30, 120)
        assert r is not None

    def test_stale_flag_in_header(self):
        from io import StringIO
        from rich.console import Console
        r = gt.build_renderable([], [], {}, None, stale=True, poll_interval=1.0, bar_width=20, console_width=100)
        buf = StringIO()
        Console(file=buf, width=100).print(r)
        assert "stale" in buf.getvalue()

    def test_history_panel_omitted_when_empty(self):
        """History panel should not appear if there's no history for a GPU."""
        from io import StringIO
        from rich.console import Console
        gpu = gt.GPUInfo(index=0, name="Test GPU")
        r = gt.build_renderable([gpu], [], {}, None, False, 1.0, 30, 120)
        buf = StringIO()
        Console(file=buf, width=120).print(r)
        assert "History" not in buf.getvalue()

    def test_cpu_panel_shown_for_gpu0(self):
        from io import StringIO
        from rich.console import Console
        gpu = gt.GPUInfo(index=0, name="Test GPU")
        cpu = gt.CPUInfo(util_pct=42.0, mem_used_mb=8192.0, mem_total_mb=16384.0)
        r = gt.build_renderable([gpu], [], {}, cpu, False, 1.0, 30, 120)
        buf = StringIO()
        Console(file=buf, width=120).print(r)
        assert "CPU" in buf.getvalue()

    def test_cpu_panel_omitted_when_none(self):
        from io import StringIO
        from rich.console import Console
        gpu = gt.GPUInfo(index=0, name="Test GPU")
        r = gt.build_renderable([gpu], [], {}, None, False, 1.0, 30, 120)
        buf = StringIO()
        Console(file=buf, width=120).print(r)
        # CPU Util label should not appear when cpu is None
        assert "CPU Util" not in buf.getvalue()


# ── _read_proc_stat ────────────────────────────────────────────────────────────

class TestReadProcStat:
    def test_returns_list_of_ints(self):
        result = gt._read_proc_stat()
        # On Linux this should succeed; on other platforms may return None
        if result is not None:
            assert isinstance(result, list)
            assert all(isinstance(x, int) for x in result)
            assert len(result) >= 4   # user, nice, system, idle minimum

    def test_bad_file_returns_none(self):
        with patch("builtins.open", side_effect=OSError("no file")):
            assert gt._read_proc_stat() is None

    def test_wrong_first_field_returns_none(self):
        import io
        fake = "notcpu 100 200 300 400\n"
        with patch("builtins.open", return_value=io.StringIO(fake)):
            assert gt._read_proc_stat() is None

    def test_parses_values_correctly(self):
        import io
        fake = "cpu 100 200 300 400 50 0 0 0\n"
        with patch("builtins.open", return_value=io.StringIO(fake)):
            result = gt._read_proc_stat()
        assert result == [100, 200, 300, 400, 50, 0, 0, 0]


# ── _read_meminfo ──────────────────────────────────────────────────────────────

class TestReadMeminfo:
    def test_returns_tuple_of_floats_on_linux(self):
        used, total = gt._read_meminfo()
        if used is not None:
            assert used > 0
            assert total > used

    def test_bad_file_returns_none_none(self):
        with patch("builtins.open", side_effect=OSError("no file")):
            used, total = gt._read_meminfo()
        assert used is None
        assert total is None

    def test_parses_values_correctly(self):
        import io
        fake = "MemTotal:       16384 kB\nMemAvailable:    8192 kB\n"
        with patch("builtins.open", return_value=io.StringIO(fake)):
            used, total = gt._read_meminfo()
        assert used == pytest.approx(8192.0 / 1024)   # 8 MiB used
        assert total == pytest.approx(16384.0 / 1024)  # 16 MiB total

    def test_missing_keys_returns_none_none(self):
        import io
        fake = "SomeOtherKey:    1234 kB\n"
        with patch("builtins.open", return_value=io.StringIO(fake)):
            used, total = gt._read_meminfo()
        assert used is None
        assert total is None


# ── _read_sensors ──────────────────────────────────────────────────────────────

class TestReadSensors:
    def test_sensors_not_installed_returns_none(self):
        with patch("gpu_top.subprocess.check_output", side_effect=FileNotFoundError):
            temp, fan = gt._read_sensors()
        assert temp is None
        assert fan is None

    def test_invalid_json_returns_none(self):
        with patch("gpu_top.subprocess.check_output", return_value="not json"):
            temp, fan = gt._read_sensors()
        assert temp is None
        assert fan is None

    def test_parses_package_temp_and_fan(self):
        sensors_json = '''{
            "coretemp-isa-0000": {
                "Adapter": "ISA adapter",
                "Package id 0": {
                    "temp1_input": 55.0,
                    "temp1_max": 100.0
                },
                "fan1": {
                    "fan1_input": 1200.0
                }
            }
        }'''
        with patch("gpu_top.subprocess.check_output", return_value=sensors_json):
            temp, fan = gt._read_sensors()
        assert temp == pytest.approx(55.0)
        assert fan == pytest.approx(1200.0)

    def test_falls_back_to_any_temp_when_no_package(self):
        sensors_json = '''{
            "some-chip": {
                "Core 0": {
                    "temp1_input": 48.0
                }
            }
        }'''
        with patch("gpu_top.subprocess.check_output", return_value=sensors_json):
            temp, fan = gt._read_sensors()
        assert temp == pytest.approx(48.0)

    def test_prefers_package_over_core_temp(self):
        sensors_json = '''{
            "coretemp-isa-0000": {
                "Core 0": {
                    "temp1_input": 45.0
                },
                "Package id 0": {
                    "temp2_input": 60.0
                }
            }
        }'''
        with patch("gpu_top.subprocess.check_output", return_value=sensors_json):
            temp, fan = gt._read_sensors()
        assert temp == pytest.approx(60.0)

    def test_fan_none_when_absent(self):
        sensors_json = '{"chip": {"Core 0": {"temp1_input": 50.0}}}'
        with patch("gpu_top.subprocess.check_output", return_value=sensors_json):
            _, fan = gt._read_sensors()
        assert fan is None


# ── query_cpu ──────────────────────────────────────────────────────────────────

class TestQueryCpu:
    def _fake_proc_stat(self, times):
        import io
        line = "cpu " + " ".join(str(x) for x in times) + "\n"
        return io.StringIO(line)

    def test_first_call_util_none(self):
        with patch("gpu_top._read_proc_stat", return_value=[100, 0, 50, 850, 0, 0, 0, 0]), \
             patch("gpu_top._read_meminfo", return_value=(4096.0, 16384.0)), \
             patch("gpu_top._read_sensors", return_value=(55.0, 1200.0)):
            cpu_info, curr_times = gt.query_cpu(None)
        assert cpu_info.util_pct is None
        assert cpu_info.mem_used_mb == pytest.approx(4096.0)
        assert cpu_info.mem_total_mb == pytest.approx(16384.0)
        assert cpu_info.temp == pytest.approx(55.0)
        assert cpu_info.fan_rpm == pytest.approx(1200.0)

    def test_second_call_computes_util(self):
        # prev: idle=850, total=1000 → next tick adds 100 total, 50 idle → 50% busy
        prev = [100, 0, 50, 850]
        curr = [150, 0, 75, 875]  # delta total=100, idle delta=25 → 75% util
        with patch("gpu_top._read_proc_stat", return_value=curr), \
             patch("gpu_top._read_meminfo", return_value=(4096.0, 16384.0)), \
             patch("gpu_top._read_sensors", return_value=(None, None)):
            cpu_info, _ = gt.query_cpu(prev)
        assert cpu_info.util_pct == pytest.approx(75.0)

    def test_util_clamped_to_0_100(self):
        prev = [0, 0, 0, 0]
        curr = [0, 0, 0, 0]   # zero delta → total=0 → util stays None
        with patch("gpu_top._read_proc_stat", return_value=curr), \
             patch("gpu_top._read_meminfo", return_value=(None, None)), \
             patch("gpu_top._read_sensors", return_value=(None, None)):
            cpu_info, _ = gt.query_cpu(prev)
        assert cpu_info.util_pct is None

    def test_returns_curr_times_for_next_call(self):
        times = [200, 10, 80, 900]
        with patch("gpu_top._read_proc_stat", return_value=times), \
             patch("gpu_top._read_meminfo", return_value=(None, None)), \
             patch("gpu_top._read_sensors", return_value=(None, None)):
            _, curr_times = gt.query_cpu(None)
        assert curr_times == times


# ── build_cpu_panel ────────────────────────────────────────────────────────────

class TestBuildCpuPanel:
    def _render(self, cpu):
        from io import StringIO
        from rich.console import Console
        panel = gt.build_cpu_panel(cpu, bar_width=20)
        buf = StringIO()
        Console(file=buf, width=120).print(panel)
        return buf.getvalue()

    def test_renders_without_error(self):
        cpu = gt.CPUInfo(util_pct=42.0, mem_used_mb=8192.0, mem_total_mb=16384.0)
        assert gt.build_cpu_panel(cpu, bar_width=20) is not None

    def test_all_none_fields_no_crash(self):
        cpu = gt.CPUInfo()
        output = self._render(cpu)
        assert "---" in output

    def test_util_shown(self):
        cpu = gt.CPUInfo(util_pct=55.0, mem_used_mb=4096.0, mem_total_mb=8192.0)
        output = self._render(cpu)
        assert "55" in output

    def test_memory_gib_shown(self):
        cpu = gt.CPUInfo(util_pct=10.0, mem_used_mb=8192.0, mem_total_mb=16384.0)
        output = self._render(cpu)
        assert "GiB" in output

    def test_temp_and_fan_shown_when_present(self):
        cpu = gt.CPUInfo(util_pct=10.0, mem_used_mb=4096.0, mem_total_mb=8192.0,
                         temp=55.0, fan_rpm=1200.0)
        output = self._render(cpu)
        assert "55" in output
        assert "1200" in output

    def test_temp_fan_section_omitted_when_both_none(self):
        cpu = gt.CPUInfo(util_pct=10.0, mem_used_mb=4096.0, mem_total_mb=8192.0,
                         temp=None, fan_rpm=None)
        output = self._render(cpu)
        assert "Temp" not in output
        assert "Fan" not in output

    def test_fan_shows_dash_when_none(self):
        cpu = gt.CPUInfo(util_pct=10.0, mem_used_mb=4096.0, mem_total_mb=8192.0,
                         temp=60.0, fan_rpm=None)
        output = self._render(cpu)
        assert "---" in output

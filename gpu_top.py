#!/usr/bin/env python3
"""gpu-top — a terminal GPU monitor powered by nvidia-smi and rich."""

import argparse
import select
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Field parsing ──────────────────────────────────────────────────────────────

_NA_VALUES = {"n/a", "[n/a]", "[not found]", "", "not supported", "[not supported]",
              "no data", "[no data]"}

_UNIT_SUFFIXES = (" %", " W", " MHz", " MiB", " C", "%", "W", "MHz", "MiB", "°C")


def parse_field(raw: str, cast=float, default=None):
    v = raw.strip()
    for suffix in _UNIT_SUFFIXES:
        if v.endswith(suffix):
            v = v[: -len(suffix)].strip()
            break
    if v.lower() in _NA_VALUES:
        return default
    try:
        return cast(v)
    except (ValueError, TypeError):
        return default


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class GPUInfo:
    index: int = 0
    name: str = "Unknown"
    uuid: str = ""
    util_gpu: Optional[float] = None
    util_mem: Optional[float] = None
    mem_used: Optional[float] = None
    mem_total: Optional[float] = None
    temp: Optional[float] = None
    power_draw: Optional[float] = None
    power_limit: Optional[float] = None
    fan: Optional[float] = None
    clk_gfx: Optional[float] = None
    clk_mem: Optional[float] = None
    driver: str = ""


@dataclass
class ProcInfo:
    gpu_uuid: str = ""
    pid: int = 0
    name: str = "?"
    mem_used: Optional[float] = None


# ── nvidia-smi queries ─────────────────────────────────────────────────────────

_GPU_FIELDS = ",".join([
    "index", "name", "uuid",
    "utilization.gpu", "utilization.memory",
    "memory.used", "memory.total",
    "temperature.gpu",
    "power.draw", "power.limit",
    "fan.speed",
    "clocks.current.graphics", "clocks.current.memory",
    "driver_version",
])

_PROC_FIELDS = "gpu_uuid,pid,process_name,used_memory"


def _run(cmd: list) -> Optional[str]:
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, text=True, timeout=5
        )
    except Exception:
        return None


def query_gpus() -> List[GPUInfo]:
    out = _run(["nvidia-smi", f"--query-gpu={_GPU_FIELDS}", "--format=csv,noheader,nounits"])
    if not out:
        return []
    gpus = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 13:
            continue
        gpus.append(GPUInfo(
            index=parse_field(p[0], int, 0),
            name=p[1],
            uuid=p[2],
            util_gpu=parse_field(p[3]),
            util_mem=parse_field(p[4]),
            mem_used=parse_field(p[5]),
            mem_total=parse_field(p[6]),
            temp=parse_field(p[7]),
            power_draw=parse_field(p[8]),
            power_limit=parse_field(p[9]),
            fan=parse_field(p[10]),
            clk_gfx=parse_field(p[11]),
            clk_mem=parse_field(p[12]),
            driver=p[13] if len(p) > 13 else "",
        ))
    return gpus


def query_procs() -> List[ProcInfo]:
    out = _run(["nvidia-smi", f"--query-compute-apps={_PROC_FIELDS}", "--format=csv,noheader,nounits"])
    if not out:
        return []
    procs = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        p = [x.strip() for x in line.split(",")]
        if len(p) < 4:
            continue
        name = p[2]
        if name.lower() in _NA_VALUES:
            name = "?"
        procs.append(ProcInfo(
            gpu_uuid=p[0],
            pid=parse_field(p[1], int, 0),
            name=name,
            mem_used=parse_field(p[3]),
        ))
    return procs


# ── Collector thread ───────────────────────────────────────────────────────────

HISTORY_LEN = 120  # samples retained (2 min at 1 s poll)


@dataclass
class HistorySample:
    util_gpu: Optional[float]   # 0–100 %
    power_w: Optional[float]    # watts
    mem_pct: Optional[float]    # 0–100 %
    ts: float = 0.0             # time.monotonic() at collection


class State:
    def __init__(self, poll_interval: float):
        self.lock = threading.Lock()
        self.gpus: List[GPUInfo] = []
        self.procs: List[ProcInfo] = []
        self.history: Dict[int, deque] = {}   # gpu_index → deque[HistorySample]
        self.tick: int = 0
        self.stale: bool = False
        self.poll_interval: float = poll_interval

    def snapshot(self):
        with self.lock:
            hist = {k: list(v) for k, v in self.history.items()}
            return list(self.gpus), list(self.procs), hist, self.tick, self.stale, self.poll_interval


def _collect_loop(state: State, stop: threading.Event):
    while not stop.is_set():
        t0 = time.monotonic()
        results = [None, None]

        def _g(): results[0] = query_gpus()
        def _p(): results[1] = query_procs()

        t1 = threading.Thread(target=_g, daemon=True)
        t2 = threading.Thread(target=_p, daemon=True)
        t1.start(); t2.start()
        t1.join(); t2.join()

        with state.lock:
            if results[0] is not None:
                state.gpus = results[0]
                state.procs = results[1] or []
                state.stale = False
                for g in state.gpus:
                    if g.index not in state.history:
                        state.history[g.index] = deque(maxlen=HISTORY_LEN)
                    mem_pct = (
                        g.mem_used / g.mem_total * 100
                        if g.mem_used is not None and g.mem_total
                        else None
                    )
                    state.history[g.index].append(HistorySample(
                        util_gpu=g.util_gpu,
                        power_w=g.power_draw,
                        mem_pct=mem_pct,
                        ts=time.monotonic(),
                    ))
            else:
                state.stale = True
            state.tick += 1

        elapsed = time.monotonic() - t0
        remaining = state.poll_interval - elapsed
        if remaining > 0:
            stop.wait(remaining)


# ── Rendering helpers ──────────────────────────────────────────────────────────

# 9-step smooth block characters
_BLOCKS = " ▏▎▍▌▋▊▉█"


def make_bar(value: Optional[float], total: float, width: int, color: str,
             empty_color: str = "grey42") -> Text:
    """Render a smooth fractional block progress bar."""
    bar = Text(no_wrap=True, overflow="ignore")
    if value is None or total <= 0:
        bar.append("━" * width, style=f"{empty_color} dim")
        return bar
    pct = max(0.0, min(1.0, value / total))
    filled = pct * width
    full = int(filled)
    frac = filled - full
    empty = width - full - (1 if frac > 0 and full < width else 0)

    bar.append("█" * full, style=f"bold {color}")
    if frac > 0 and full < width:
        idx = max(1, int(frac * 8))
        bar.append(_BLOCKS[idx], style=color)
    if empty > 0:
        bar.append("░" * empty, style=empty_color)
    return bar


def _util_color(pct: Optional[float]) -> str:
    if pct is None: return "white"
    if pct < 60:    return "bright_green"
    if pct < 85:    return "yellow"
    return "bright_red"


def _temp_color(t: Optional[float]) -> str:
    if t is None:  return "white"
    if t < 60:     return "bright_green"
    if t < 80:     return "yellow"
    return "bright_red"


def _power_color(draw: Optional[float], limit: Optional[float]) -> str:
    if draw is None or limit is None or limit == 0:
        return "bright_blue"
    ratio = draw / limit
    if ratio < 0.6:  return "bright_blue"
    if ratio < 0.85: return "yellow"
    return "bright_red"


def _fmt(val: Optional[float], spec: str = ".0f", unit: str = "") -> str:
    return "---" if val is None else f"{val:{spec}}{unit}"


# ── Sparkline / history graph ──────────────────────────────────────────────────

# 9 vertical fill levels (space = 0, █ = full)
_VBLOCKS = " ▁▂▃▄▅▆▇█"


def _sparkline_rows(values: List[Optional[float]], width: int,
                    max_val: float, color: str, rows: int = 4) -> List[Text]:
    """
    Build `rows` lines of Text that together form a vertically-stacked
    block-character sparkline.  values[-width:] are used; fewer samples
    are left-padded with dim filler.
    """
    samples = list(values)[-width:]
    pad = width - len(samples)

    row_texts = []
    for row_idx in range(rows):           # row 0 = top, row rows-1 = bottom
        rt = Text(no_wrap=True, overflow="ignore")
        # left-pad with filler if not enough history yet
        rt.append("░" * pad, style="grey19 dim")
        for val in samples:
            if val is None or max_val <= 0:
                rt.append("░", style="grey19 dim")
                continue
            pct = max(0.0, min(1.0, val / max_val))
            # fraction of total height covered by this row's band
            band_lo = (rows - 1 - row_idx) / rows
            band_hi = (rows - row_idx) / rows
            if pct >= band_hi:
                rt.append("█", style=f"bold {color}")
            elif pct > band_lo:
                frac = (pct - band_lo) * rows   # 0.0–1.0 within band
                idx = max(1, min(8, int(frac * 8 + 0.5)))
                rt.append(_VBLOCKS[idx], style=color)
            else:
                rt.append("░", style="grey19 dim")
        row_texts.append(rt)
    return row_texts


def _append_graph(body: Text, label: str, values: List[Optional[float]],
                  max_val: float, color: str, unit: str,
                  graph_w: int, rows: int = 4) -> None:
    """Append a labeled sparkline section into an existing Text object."""
    spark = _sparkline_rows(values, graph_w, max_val, color, rows)
    max_str = f"{max_val:.0f}{unit}"

    # ── header: "  label ──…── max_str" fits exactly in graph_w + 2 cols ──────
    # cols: 2("  ") + len(label) + 1(" ") + dash_len + 1(" ") + len(max_str)
    dash_len = max(0, graph_w - len(label) - len(max_str) - 2)
    body.append(f"  {label} ", style="bold white")
    body.append("─" * dash_len, style="dim")
    body.append(f" {max_str}", style=f"dim {color}")
    body.append("\n")

    # ── sparkline rows ─────────────────────────────────────────────────────────
    for row in spark:
        body.append("  ")
        body.append_text(row)
        body.append("\n")

    # ── footer: time span on left, 0 on right ─────────────────────────────────
    zero_str = f"0{unit}"
    body.append("  ", style="")
    body.append("─" * (graph_w - len(zero_str)), style="dim")
    body.append(zero_str, style=f"dim {color}")
    body.append("\n")


def build_history_panel(g: GPUInfo, history: List[HistorySample],
                        console_width: int) -> Panel:
    # graph fills panel interior: width - 2 (borders) - 2 (padding each side)
    graph_w = max(10, console_width - 6)

    util_vals  = [s.util_gpu for s in history]
    power_vals = [s.power_w  for s in history]
    mem_vals   = [s.mem_pct  for s in history]
    max_power  = g.power_limit or max((v for v in power_vals if v is not None), default=100.0)

    body = Text(no_wrap=True, overflow="ignore")
    _append_graph(body, "GPU Util", util_vals,  100.0,     "bright_green", "%", graph_w)
    body.append("\n")
    _append_graph(body, "MEM Used", mem_vals,   100.0,     "cyan",         "%", graph_w)
    body.append("\n")
    _append_graph(body, "Power",    power_vals, max_power, "bright_blue",  "W", graph_w)

    # time-span legend using actual elapsed time from sample timestamps
    span_s = history[-1].ts - history[0].ts if len(history) > 1 else 0.0
    if span_s < 60:
        span_lbl = f"← {span_s:.0f}s ago"
    else:
        m, s = divmod(int(span_s), 60)
        span_lbl = f"← {m}m{s:02d}s ago"
    now_lbl    = "now →"
    mid_dashes = max(0, graph_w - len(span_lbl) - len(now_lbl) - 2)
    body.append(f"\n  {span_lbl} ", style="dim")
    body.append("─" * mid_dashes, style="dim")
    body.append(f" {now_lbl}", style="dim")

    title = Text()
    title.append(f"  GPU {g.index}  ", style="bold bright_white on dark_green")
    title.append(f" {g.name} — History ", style="bold bright_green")

    return Panel(
        body,
        title=title,
        title_align="left",
        border_style="dark_green",
        box=box.ROUNDED,
        padding=(0, 1),
    )


# ── GPU panel ──────────────────────────────────────────────────────────────────

def build_gpu_panel(g: GPUInfo, bar_width: int) -> Panel:
    lines = Text(no_wrap=True, overflow="ignore")

    # ── GPU utilization ────────────────────────────────────────────────────────
    gpu_col = _util_color(g.util_gpu)
    lines.append("  GPU Util  ", style="bold white")
    lines.append(make_bar(g.util_gpu, 100, bar_width, gpu_col))
    lines.append(f"  {_fmt(g.util_gpu, '.0f', '%'):>5}", style=f"bold {gpu_col}")
    lines.append("\n")

    # ── Memory usage ───────────────────────────────────────────────────────────
    mem_pct = (g.mem_used / g.mem_total * 100) if (g.mem_used and g.mem_total) else None
    mem_col = _util_color(mem_pct)
    lines.append("  MEM Used  ", style="bold white")
    lines.append(make_bar(g.mem_used, g.mem_total or 1, bar_width, "cyan"))
    lines.append(f"  {_fmt(mem_pct, '.0f', '%'):>5}", style="bold cyan")
    if g.mem_used is not None and g.mem_total is not None:
        lines.append(f"  {g.mem_used:.0f} / {g.mem_total:.0f} MiB", style="dim cyan")
    lines.append("\n")

    # ── Blank separator ────────────────────────────────────────────────────────
    lines.append("\n")

    # ── Temperature ────────────────────────────────────────────────────────────
    t_col = _temp_color(g.temp)
    lines.append("  Temp      ", style="bold white")
    temp_txt = _fmt(g.temp, ".0f", "°C")
    lines.append(f"{temp_txt:>8}", style=f"bold {t_col}")

    # ── Power ──────────────────────────────────────────────────────────────────
    p_col = _power_color(g.power_draw, g.power_limit)
    lines.append("      Power  ", style="bold white")
    lines.append(make_bar(g.power_draw, g.power_limit or 1, 20, p_col))
    lines.append(f"  {_fmt(g.power_draw, '.0f', 'W'):>6}", style=f"bold {p_col}")
    if g.power_limit is not None:
        lines.append(f" / {g.power_limit:.0f}W", style="dim")
    lines.append("\n")

    # ── Fan + clocks ───────────────────────────────────────────────────────────
    lines.append("  Fan       ", style="bold white")
    lines.append(f"{_fmt(g.fan, '.0f', '%'):>8}", style="white")
    lines.append("      Clocks  ", style="bold white")
    lines.append(f"GFX {_fmt(g.clk_gfx, '.0f', ' MHz')}  MEM {_fmt(g.clk_mem, '.0f', ' MHz')}", style="dim")
    lines.append("\n")

    title = Text()
    title.append(f"  GPU {g.index}  ", style="bold bright_white on dark_green")
    title.append(f" {g.name} ", style="bold bright_green")

    return Panel(
        lines,
        title=title,
        title_align="left",
        border_style="green",
        box=box.ROUNDED,
        padding=(0, 1),
    )


# ── Process table ──────────────────────────────────────────────────────────────

def build_proc_panel(gpus: List[GPUInfo], procs: List[ProcInfo]) -> Panel:
    uuid_to_idx = {g.uuid: g.index for g in gpus}

    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold white",
        expand=True,
        pad_edge=False,
    )
    table.add_column("GPU", style="bright_green", width=5, justify="right")
    table.add_column("PID", style="cyan", width=8, justify="right")
    table.add_column("Process", style="white", ratio=1)
    table.add_column("MEM Used", style="yellow", width=12, justify="right")

    if not procs:
        table.add_row("—", "—", "[dim]No active compute processes[/dim]", "—")
    else:
        for p in procs:
            gpu_idx = str(uuid_to_idx.get(p.gpu_uuid, "?"))
            mem_str = f"{p.mem_used:.0f} MiB" if p.mem_used is not None else "[dim]N/A[/dim]"
            name = p.name if p.name not in ("?", "") else "[dim italic]unknown[/dim italic]"
            table.add_row(gpu_idx, str(p.pid), name, mem_str)

    return Panel(
        table,
        title="[bold white]  Processes  [/bold white]",
        title_align="left",
        border_style="bright_blue",
        box=box.ROUNDED,
        padding=(0, 1),
    )


# ── Header & footer ────────────────────────────────────────────────────────────

def build_header(gpus: List[GPUInfo], stale: bool, poll_interval: float) -> Panel:
    t = Text(no_wrap=True, overflow="ellipsis")
    t.append("  ⚡ gpu-top", style="bold bright_green")
    if gpus and gpus[0].driver:
        t.append(f"  │  Driver {gpus[0].driver}", style="dim green")
    t.append(f"  │  refresh {poll_interval:.1f}s", style="dim")
    if stale:
        t.append("  ⚠ stale", style="bold red")
    t.append(f"  │  {time.strftime('%H:%M:%S')}", style="dim white")
    t.append("  ", style="")
    return Panel(t, box=box.ROUNDED, border_style="dim green", padding=(0, 0))


def build_footer() -> Panel:
    t = Text()
    t.append("  [q]", style="bold yellow"); t.append(" quit", style="dim")
    t.append("   [+]", style="bold yellow"); t.append(" slower poll", style="dim")
    t.append("   [-]", style="bold yellow"); t.append(" faster poll", style="dim")
    t.append("   [Ctrl-C]", style="bold yellow"); t.append(" exit  ", style="dim")
    return Panel(t, box=box.ROUNDED, border_style="dim", padding=(0, 0))


# ── Full renderable ────────────────────────────────────────────────────────────

def build_renderable(gpus, procs, history, stale, poll_interval, bar_width, console_width):
    parts = [build_header(gpus, stale, poll_interval)]
    for g in gpus:
        parts.append(build_gpu_panel(g, bar_width))
        gpu_hist = history.get(g.index, [])
        if gpu_hist:
            parts.append(build_history_panel(g, gpu_hist, console_width))
    parts.append(build_proc_panel(gpus, procs))
    parts.append(build_footer())
    return Group(*parts)


# ── Key input (raw/cbreak mode) ────────────────────────────────────────────────

try:
    import termios, tty as _tty
    _HAS_TERMIOS = True
except ImportError:
    _HAS_TERMIOS = False


def _read_key_nb() -> Optional[str]:
    """Non-blocking single-character read from stdin."""
    if not _HAS_TERMIOS or not sys.stdin.isatty():
        return None
    try:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
    except Exception:
        pass
    return None


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="gpu-top: a terminal GPU monitor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-d", "--delay", type=float, default=1.0, metavar="SEC",
                        help="poll interval in seconds")
    parser.add_argument("--bar-width", type=int, default=30, metavar="N",
                        help="width of progress bars")
    args = parser.parse_args()

    # Verify nvidia-smi exists
    try:
        subprocess.check_output(["nvidia-smi", "--version"], stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("Error: nvidia-smi not found in PATH.", file=sys.stderr)
        sys.exit(1)

    state = State(poll_interval=max(0.1, args.delay))
    stop = threading.Event()
    collector = threading.Thread(target=_collect_loop, args=(state, stop), daemon=True)
    collector.start()

    # Wait for first data (up to 3s)
    for _ in range(60):
        if state.tick > 0:
            break
        time.sleep(0.05)

    console = Console(highlight=False)

    # Set cbreak so we can read keys without Enter
    old_tty = None
    if _HAS_TERMIOS and sys.stdin.isatty():
        try:
            old_tty = termios.tcgetattr(sys.stdin)
            _tty.setcbreak(sys.stdin)
        except Exception:
            old_tty = None

    try:
        with Live(
            console=console,
            refresh_per_second=8,
            screen=True,   # alternate buffer — clean full-screen like nvtop
            transient=False,
        ) as live:
            while True:
                key = _read_key_nb()
                if key in ("q", "Q"):
                    break
                elif key == "+":
                    with state.lock:
                        step = 0.1 if state.poll_interval < 1.0 else 0.5
                        state.poll_interval = round(min(state.poll_interval + step, 10.0), 1)
                elif key == "-":
                    with state.lock:
                        step = 0.1 if state.poll_interval <= 1.0 else 0.5
                        state.poll_interval = round(max(state.poll_interval - step, 0.1), 1)

                gpus, procs, hist, tick, stale, poll_interval = state.snapshot()
                live.update(build_renderable(
                    gpus, procs, hist, stale, poll_interval,
                    args.bar_width, console.width,
                ))
                time.sleep(0.05)   # 20 fps key-check; rich auto-refreshes at 8 fps

    except KeyboardInterrupt:
        pass
    finally:
        if old_tty is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
            except Exception:
                pass
        stop.set()


if __name__ == "__main__":
    main()

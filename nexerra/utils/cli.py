# ---------------------------------------
#     _   __
#    / | / /__  _  _____  ______________ _
#   /  |/ / _ \| |/_/ _ \/ ___/ ___/ __ `/
#  / /|  /  __/>  </  __/ /  / /  / /_/ /
# /_/ |_/\___/_/|_|\___/_/  /_/   \__,_/
#
#  Terminal UI for the inference entrypoints.
#  Presentation only: wraps nexerra.inference.FlowDesign.main and restyles its
#  logs, prints and progress bar. No model, sampling or filtering code is touched,
#  so outputs are identical to running FlowDesign.py directly.
#  Example usage:
#   nexerra-flow-design --alpha 0.9 --num-samples 1000 --batch-size 128 \
#   --reward gas --threshold 0.5 --filters [--export-svg assets/cli.svg]
#  Author: Dhruv Menon (dm958[at]cam[dot]ac[dot]uk)
#
#  MIT License. See LICENSE in the repo root.
#  Copyright (c) 2025 Dhruv Menon
# ---------------------------------------

import argparse
import contextlib
import io
import logging
import os
import random
import sys
import time
import types
import warnings
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pyfiglet
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

# --- Palette ---
ACCENT = "#8b7cf6"
GLOW = "#ece8ff"
CYAN = "#38bdf8"
OK = "#4ade80"
WARN = "#fbbf24"
ERR = "#f87171"
DIM = "grey50"
LOGO = ["#c4b5fd", "#a78bfa", "#8b7cf6", "#7c83f0", "#5b9ae0", "#38bdf8"]
GLYPHS = ["·", "✢", "✳", "✶", "✻", "✽", "✻", "✶", "✳", "✢"]

# --- Things to say while the model works ---
VERBS = {
    "boot":     ["Warming up the reactor", "Dusting off the glovebox", "Calibrating the balance",
                 "Degassing the solvent", "Labelling the vials"],
    "load":     ["Unpacking checkpoints", "Recalling 120 epochs of chemistry", "Untangling attention heads",
                 "Reloading the latent space", "Remembering every linker it has met"],
    "flow":     ["Charting the latent landscape", "Tuning the vector field", "Reading the flow map"],
    "generate": ["Reticulating", "Nucleating", "Integrating the vector field", "Crystallizing", "Marinating",
                 "Riding the flow", "Self-assembling", "Annealing", "Percolating", "Templating",
                 "Brewing linkers", "Stirring the flask", "Coordinating", "Wandering latent space"],
    "place":    ["Placing connection points", "Finding coordination sites", "Docking onto metal nodes",
                 "Plugging in the nodes"],
    "filter":   ["Sieving", "Checking ring strain", "Scrubbing azides", "Consulting SCScore",
                 "Asking a synthetic chemist", "Pruning the impossible"],
    "reward":   ["Weighing the candidates", "Scoring pore chemistry", "Ranking linkers",
                 "Polishing the shortlist", "Tasting the reward"],
}
DONE = ["Brewed", "Crystallized", "Reticulated", "Simmered", "Distilled", "Assembled", "Cooked up"]
TIPS = [
    "Use [bold]--filters[/] to drop strained rings, reactive groups and hard-to-make linkers.",
    "Lower [bold]--alpha[/] to wander further from the seed; raise it to stay close.",
    "Edit [bold]designed/linker/run/input.txt[/] to change the seed linker.",
    "Every decoded linker is kept in [bold]output_all.txt[/], even ones that miss the reward.",
    "Switch objectives with [bold]--reward gas[/] or [bold]--reward grav[/].",
]


def _elapsed(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds}s" if seconds < 60 else f"{seconds // 60}m {seconds % 60:02d}s"

def _plural(n, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"

def _home(path: str | Path) -> str:
    path = os.path.abspath(path); cwd = os.getcwd(); home = str(Path.home())
    if path.startswith(cwd + os.sep): return os.path.relpath(path, cwd)
    return "~" + path[len(home):] if path.startswith(home) else path

def _version() -> str:
    try: return version("nexerra-r1")
    except PackageNotFoundError: return "dev"

def _shimmer(text: str, now: float) -> Text:
    '''Bright highlight sweeping across the status verb.'''
    out = Text()
    pos = int(now * 16) % (len(text) + 12) - 6
    for i, ch in enumerate(text): out.append(ch, style = f"bold {GLOW}" if abs(i - pos) <= 1 else f"bold {ACCENT}")
    return out

def logo() -> Text:
    art = pyfiglet.figlet_format("Nexerra", font = "slant").rstrip("\n").split("\n")
    out = Text()
    for i, line in enumerate(art): out.append(line + "\n", style = f"bold {LOGO[min(i, len(LOGO) - 1)]}")
    return out


class _LineSink(io.TextIOBase):
    '''Stand-in for stdout that hands complete lines to the UI.'''
    def __init__(self, on_line):
        self.on_line = on_line; self.buf = ""
    def writable(self): return True
    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            if line.strip(): self.on_line(line)
        return len(s)


class _UIHandler(logging.Handler):
    def __init__(self, ui):
        super().__init__(level = logging.INFO); self.ui = ui
    def emit(self, record):
        try: self.ui.on_log(record)
        except Exception: self.handleError(record)


class NexerraUI:
    '''Live status line + styled transcript, driven by FlowDesign's own logs.'''
    def __init__(self, console: Console, argv: list[str]):
        self.console = console
        self.argv = argv
        self.t0 = time.monotonic()
        self.args = None
        self.progress = None                # [done, total] batches while sampling
        self.counts = {}                    # design funnel
        self.output = None
        self.tau_z = None
        self.lr_failures = 0
        self.warned = set()
        self.transcript = []                # renderables, for --export-svg
        self.live = Live(self, console = console, refresh_per_second = 12, redirect_stdout = False, transient = True)
        self.set_phase("boot")

    # --- Transcript ---
    def print(self, renderable = ""):
        self.transcript.append(renderable); self.console.print(renderable)

    def step(self, title: str, *details):
        line = Text.assemble(("● ", ACCENT), (title, "bold"))
        rows = [line]
        for i, d in enumerate(details):
            rows.append(Text.assemble(("  └ " if i == 0 else "    ", DIM), d if isinstance(d, Text) else Text(str(d), style = DIM)))
        self.print(Group(*rows)); self.print()

    def warn(self, msg: str):
        if msg in self.warned: return
        self.warned.add(msg); self.print(Text.assemble(("  ! ", f"bold {WARN}"), (msg, WARN)))

    def set_phase(self, phase: str):
        self.phase = phase; self.phase_t0 = time.monotonic()
        self.verbs = random.sample(VERBS[phase], len(VERBS[phase]))

    # --- Live status line ---
    def __rich__(self):
        now = time.monotonic()
        verb = self.verbs[int((now - self.phase_t0) / 3.0) % len(self.verbs)]
        line = Text.assemble((GLYPHS[int(now * 8) % len(GLYPHS)] + " ", ACCENT), _shimmer(verb + "…", now),
                             (f"  ({_elapsed(now - self.t0)} · ctrl+c to interrupt)", DIM))
        if self.progress is None: return line
        done, total = self.progress
        bar = Table.grid(padding = (0, 1))
        bar.add_row(Text("  "), ProgressBar(total = total, completed = done, width = 32, complete_style = ACCENT,
                                            finished_style = OK, style = "grey23"),
                    Text(f"{done}/{total} batches", style = DIM))
        return Group(line, bar)

    # --- Header ---
    def header(self, args):
        import torch
        self.args = args
        try: seed = Path(args.input).read_text().strip() or "—"
        except OSError: seed = "—"
        device = f"cuda · {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
        sampling = f"{args.num_samples} samples · α {args.alpha} · batch {args.batch_size}"
        if args.filters: sampling += " · filters on"

        grid = Table.grid(padding = (0, 2))
        grid.add_column(style = DIM); grid.add_column()
        grid.add_row("mode:", "flow-guided linker design")
        grid.add_row("seed:", Text(seed, style = CYAN))
        grid.add_row("objective:", f"{args.reward} reward · threshold {args.threshold}")
        grid.add_row("sampling:", sampling)
        grid.add_row("flow:", Path(args.flow_ckpt).name)
        grid.add_row("device:", device)
        grid.add_row("directory:", _home(os.getcwd()))
        title = Text.assemble((">_ ", DIM), ("Nexerra-R1", "bold"), (f" (v{_version()})", DIM))
        self.print(Panel(Group(title, Text(""), grid), box = box.ROUNDED, border_style = "grey37",
                         expand = False, padding = (0, 1)))
        self.print(Padding(Text.from_markup(f"[bold]Tip:[/] {random.choice(TIPS)}"), (1, 1, 1, 1)))
        prompt = Text.assemble(("› ", DIM), ("nexerra-flow-design " + " ".join(self.argv)).rstrip())
        self.print(Padding(prompt, (1, 1), style = "on #24242c", expand = True)); self.print()
        self.set_phase("load")

    # --- Event routing (FlowDesign's log lines + prints) ---
    def on_print(self, line: str):
        if line.startswith("Restored variables from"): return     # SCScore weights
        if line.startswith("Input SMILES:"):
            self.set_phase("generate"); return
        if line.startswith("Warning:"): self.warn(line[len("Warning:"):].strip()); return
        self.print(Text(line, style = DIM))

    def on_log(self, record: logging.LogRecord):
        fmt, args = str(record.msg), record.args or ()
        if not (record.name.startswith("nexerra") or record.name == "__main__") and record.levelno < logging.WARNING: return
        if fmt.startswith(("Synthetic complexity", "Model initialized", "Tokenizer initialized", "Latent bank loaded",
                           "Moving to reward", "Stored in decreasing")): return
        if fmt.startswith("Failed to optimize [Lr]"): self.lr_failures += 1; return
        if fmt.startswith("Model state loaded"):
            self.step("Loaded HT-VAE", Path(args[0]).name); self.set_phase("flow"); return
        if fmt.startswith(("Setting tau_z", "Overriding tau_z")):
            self.tau_z = args[-1] if fmt.startswith("Setting") else args[0]; return
        if fmt.startswith("Flow model loaded"):
            detail = Path(args[0]).name + (f" · τ_z = {self.tau_z:.4f}" if self.tau_z is not None else "")
            self.step("Loaded OT-CFM flow", detail); return
        if fmt.startswith("All generated SMILES saved"):
            try: n = sum(1 for line in open(args[0]) if line.strip())
            except OSError: n = None
            if n is not None: self.counts["unique, [Lr] placed"] = n
            details = [f"{_plural(n, 'unique linker')} → {_home(args[0])}"]
            if self.lr_failures: details.append(f"{self.lr_failures} could not take [Lr] placement")
            self.step("Placed [Lr] connection points", *details)
            self.set_phase("filter" if self.args is not None and self.args.filters else "reward"); return
        if fmt.startswith("%d molecules remain after ring"): self.counts["ring filter"] = args[0]; return
        if fmt.startswith("%d molecules remain after SCScore"): self.counts["SCScore ≤ 4.5"] = args[0]; return
        if fmt.startswith("%d molecules remain after functional"):
            self.counts["functional groups"] = args[0]
            keys = ["ring filter", "SCScore ≤ 4.5", "functional groups"]
            self.step("Applied hard-chemistry filters", *[f"{k:<18} {self.counts[k]:>6} left" for k in keys])
            return
        if fmt.startswith("%d UNIQUE molecules generated"): self.set_phase("reward"); return
        if fmt.startswith("%d UNIQUE molecules remain after reward"):
            self.counts[f"reward ≥ {self.args.threshold if self.args else '?'}"] = args[0]
            self.step(f"Scored with the {self.args.reward if self.args else ''} reward",
                      f"{_plural(args[0], 'linker')} cleared threshold {self.args.threshold if self.args else ''}"); return
        if fmt.startswith("Generated SMILES saved"): self.output = args[0]; return
        if record.levelno >= logging.WARNING: self.warn(record.getMessage()); return
        self.print(Text("  " + record.getMessage(), style = DIM))

    # --- Patched tqdm: drives the progress bar instead ---
    def track(self, iterable, desc = None, total = None, **_):
        total = total if total is not None else len(iterable)
        self.progress = [0, total]; t = time.monotonic()
        for item in iterable:
            yield item
            self.progress[0] += 1
        self.progress = None
        n = self.args.num_samples if self.args else "?"
        alpha = f" · α = {self.args.alpha}" if self.args else ""
        self.step(f"Sampled {n} latents along the flow", f"{total} batches{alpha} · {_elapsed(time.monotonic() - t)}")
        self.set_phase("place")

    # --- Summary ---
    def summary(self):
        rows = [("sampled", self.args.num_samples if self.args else None)] + list(self.counts.items())
        rows = [(k, v) for k, v in rows if v is not None]
        top = max([v for _, v in rows] + [1])
        funnel = Table.grid(padding = (0, 2))
        for i, (k, v) in enumerate(rows):
            colour = OK if i == len(rows) - 1 else ACCENT
            funnel.add_row(Text(k, style = DIM), Text(f"{v:>6}"), Text("━" * (max(1, round(28 * v / top)) if v else 0), style = colour))
        self.print(Group(Text.assemble(("● ", ACCENT), ("Design funnel", "bold")), Padding(funnel, (0, 0, 0, 4))))
        self.print()

        designed = []
        if self.output and os.path.isfile(self.output):
            designed = [l.strip() for l in open(self.output) if l.strip()]
        if designed:
            lines = [Text.assemble((f"    {i + 1}. ", DIM), (smi, CYAN)) for i, smi in enumerate(designed[:5])]
            more = len(designed) - 5
            if more > 0: lines.append(Text(f"    … and {more} more", style = DIM))
            self.print(Group(Text.assemble(("● ", ACCENT), ("Designed linkers", "bold")), *lines)); self.print()
        else:
            self.warn("No linkers cleared the reward threshold. Try a lower --threshold or more --num-samples.")
            self.print()
        self.print(Text.assemble(("✻ ", ACCENT), (f"{random.choice(DONE)} {_plural(len(designed), 'linker')} in "
                                                  f"{_elapsed(time.monotonic() - self.t0)}", "bold"),
                                 (f"  → {_home(self.output)}" if self.output else "", DIM)))


def _quiet_warnings():
    for category in (FutureWarning, DeprecationWarning): warnings.filterwarnings("ignore", category = category)
    warnings.filterwarnings("ignore", message = ".*nested tensors.*")
    warnings.filterwarnings("ignore", message = ".*to-Python converter.*")


def flow_design(argv = None):
    '''Console entrypoint: `nexerra-flow-design`.'''
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help = False)
    pre.add_argument("--export-svg", type = str, default = None)
    pre.add_argument("--plain", action = "store_true")
    ui_args, argv = pre.parse_known_args(argv)

    if ui_args.plain:                       # escape hatch: the original, unstyled CLI
        from nexerra.inference import FlowDesign as fd
        return fd.main(argv)

    console = Console(file = sys.stdout, highlight = False)
    console.print(logo())
    if "-h" in argv or "--help" in argv:
        _quiet_warnings(); warnings.filterwarnings("ignore", category = RuntimeWarning)
        logging.getLogger().handlers[:] = [logging.NullHandler()]
        with contextlib.redirect_stdout(io.StringIO()): from nexerra.inference import FlowDesign as fd
        fd.display_banner = lambda: None
        return fd.main(argv)

    _quiet_warnings()
    ui = NexerraUI(console, argv)
    ui.transcript.append(logo())
    root = logging.getLogger()
    root.handlers[:] = [_UIHandler(ui)]; root.setLevel(logging.INFO)

    class _Parser(argparse.ArgumentParser):
        def parse_args(self, *a, **kw):
            args = super().parse_args(*a, **kw); ui.header(args); return args

    status = 0
    try:
        with ui.live, contextlib.redirect_stdout(_LineSink(ui.on_print)):
            from nexerra.inference import FlowDesign as fd
            fd.display_banner = lambda: None
            fd.tqdm = ui.track
            fd.argparse = types.SimpleNamespace(ArgumentParser = _Parser)
            fd.main(argv)
            ui.progress = None
        ui.summary()
    except KeyboardInterrupt:
        console.print(Text("\n✗ Interrupted. The flask is safe.", style = f"bold {WARN}")); status = 130
    except AssertionError as e:
        console.print(Text(f"✗ {e}", style = f"bold {ERR}")); status = 1

    if ui_args.export_svg:
        rec = Console(record = True, width = min(console.width, 110), file = io.StringIO(), highlight = False)
        for r in ui.transcript: rec.print(r)
        rec.save_svg(ui_args.export_svg, title = "nexerra-flow-design")
        console.print(Text(f"  saved terminal snapshot → {ui_args.export_svg}", style = DIM))
    return status


if __name__ == "__main__":
    sys.exit(flow_design())
# -----------------------------------------------------------------

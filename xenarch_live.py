"""
Live run status for the Mission Control dashboard.

Scripts call it while they run; it keeps dashboard/live.json up to date, and the page
(dashboard/index.html) polls that file every 2 s.

    live = LiveRun("xenarch_mk18_script.py", data="data/xenarch-imagery", steps=["Chips", "Memory", ...])
    live.step("Memory")                       # current step
    live.progress(120, 8000, "chips")         # progress bar inside the step
    live.metric(epoch=3, loss=1383.9)         # one row of the live metrics table / chart
    live.log("rung 0: keep 9")                # recent log lines
    live.samples(list_of_2d_arrays)           # a few training chips, shown as thumbnails
    live.done() / live.fail(exc)
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

LIVE_DIR = Path("dashboard")
LIVE_FILE = LIVE_DIR / "live.json"
SAMPLE_DIR = LIVE_DIR / "live"


class LiveRun:
    def __init__(self, script, data="", steps=(), min_interval=1.0):
        self.min_interval, self._last = min_interval, 0.0
        self.state = {
            "script": Path(script).name, "data": str(data), "status": "running",
            "started": datetime.now().isoformat(timespec="seconds"), "updated": None,
            "steps": list(steps), "step": None, "progress": None,
            "metrics": [], "log": [], "samples": [], "error": None,
        }
        self._write(force=True)

    # --- updates -----------------------------------------------------------
    def step(self, name):
        self.state["step"], self.state["progress"] = name, None
        self.log(f"step: {name}")

    def progress(self, done, total, unit=""):
        self.state["progress"] = {"done": int(done), "total": int(total), "unit": unit}
        self._write()

    def metric(self, **row):
        self.state["metrics"].append({k: _num(v) for k, v in row.items()})
        self._write(force=True)

    def log(self, line):
        self.state["log"] = (self.state["log"] + [f"{datetime.now():%H:%M:%S}  {line}"])[-12:]
        self._write(force=True)

    def samples(self, chips, n=12, size=96):
        """Save up to n chips (2-D arrays in [0, 1]) as thumbnails for the page."""
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        names = []
        for i, c in enumerate(chips[:n]):
            img = Image.fromarray((np.clip(c, 0, 1) * 255).astype(np.uint8)).resize((size, size))
            img.save(SAMPLE_DIR / f"sample_{i:02d}.png")
            names.append(f"live/sample_{i:02d}.png")
        self.state["samples"] = names
        self._write(force=True)

    def done(self):
        self.state["status"], self.state["progress"] = "done", None
        self.log("finished")

    def fail(self, exc):
        self.state["status"], self.state["error"] = "failed", f"{type(exc).__name__}: {exc}"
        self.log(f"failed: {self.state['error']}")

    # --- io ------------------------------------------------------------------
    def _write(self, force=False):
        now = time.time()
        if not force and now - self._last < self.min_interval:
            return
        self._last = now
        self.state["updated"] = datetime.now().isoformat(timespec="seconds")
        LIVE_DIR.mkdir(exist_ok=True)
        tmp = LIVE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        os.replace(tmp, LIVE_FILE)            # atomic: the page never reads a half-written file


def _num(v):
    try:
        return float(v) if not isinstance(v, (str, bool, type(None))) else v
    except (TypeError, ValueError):
        return str(v)

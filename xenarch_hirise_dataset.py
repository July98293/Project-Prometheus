"""
Xenarch HiRISE set — natural Mars terrain grouped by landform, for the PatchCore memory bank.

The Mars side of xenarch-imagery already has ~330 HiRISE observations; what it lacks is variety.
This script picks HiRISE observations by what the HiRISE team imaged them for (RATIONALE_DESC in the
archive index, e.g. "Gullies in crater in Terra Cimmeria") and spreads them over sun angles:

  aeolian          dunes, ripples, transverse aeolian ridges, dust-devil tracks, yardangs
  slopes           gullies, recurring slope lineae / slope streaks, landslides, rockfall, boulders
  periglacial_ice  polygons, brain terrain, scalloped depressions, debris aprons, valley fill, glaciers
  polar            spiders / CO2 features, polar layered deposits, residual cap
  impacts          new impact sites, dark blast zones, fresh craters
  layers_volcanic  layered deposits, lava flows, channels, inverted channels, deltas

Only 2×2-binned observations (0.5 m/pixel) are used: the raw full-resolution CCD channels carry a column
pattern that binning averages away, and the existing hirise_png tiles are bin 2 too. From each, four 1024×512 tiles are
cut at the image centre from the raw EDR channels of the most central CCDs available (RED5, RED4, then
RED3 / RED6 — RED4 is missing from many recent observations): one HTTP range request of ~0.5 MB per tile. Observations whose rationale mentions hardware or landing sites, or that lie
within 15 km of a lander, rover or crash site, are skipped, as are observations already in xenarch-imagery.

  .venv/bin/python xenarch_hirise_dataset.py [--out data/hirise-set] [--per-category 16]
  .venv/bin/python xenarch_hirise_dataset.py --upload <user>/<repo>     (after `hf auth login`)
"""
import csv, math, re, sys, time, urllib.error
from pathlib import Path

import numpy as np
from PIL import Image

from xenarch_live import LiveRun
from xenarch_lroc_nac_dataset import http, upload

INDEX = "https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX"
EDR = "https://hirise-pds.lpl.arizona.edu/PDS/EDR"
TILE_LINES, BINNING = 1024, 2        # bin 2: 512 samples per channel, like the hirise_png tiles
MARS_R = 3389.5
EXCLUDE_KM = 15
INCIDENCE_BINS = [(0, 40), (40, 60), (60, 75), (75, 90)]
CHANNELS = ["RED5_0", "RED5_1", "RED4_0", "RED4_1", "RED3_0", "RED3_1", "RED6_0", "RED6_1"]   # centre outwards
PER_OBS = 4

# Landers, rovers, heat shields / crash sites (lat, east lon).
HARDWARE = [(22.27, 312.05), (48.27, 134.26), (19.13, 326.78), (-14.57, 175.47), (-1.95, 354.47),
            (68.22, 234.25), (-4.59, 137.44), (-4.75, 137.38), (4.50, 135.62), (18.44, 77.45),
            (25.07, 109.93), (-2.05, 353.79), (11.53, 90.43), (-44.9, 199.8), (-76.57, 165.2)]
HARDWARE_WORDS = re.compile(
    r"rover|lander|curiosity|opportunity|spirit|phoenix|insight|perseverance|ingenuity|hardware|heat ?shield|"
    r"parachute|backshell|schiaparelli|beagle|zhurong|tianwen|viking|pathfinder|\bmsl\b|mars 2020|landing|"
    r"spacecraft|debris", re.I)

# category -> {sub-landform: rationale pattern}; sub-landforms are taken in turn so none dominates
CATEGORIES = {
    "aeolian": {"dunes": r"\bdunes?\b", "ripples_tars": r"ripple|\btars?\b|transverse aeolian",
                "dust_devil_tracks": r"dust devil", "yardangs": r"yardang"},
    "slopes": {"gullies": r"gull", "rsl_slope_streaks": r"recurring slope|\brsl\b|slope streak|slope lineae",
               "landslides_rockfall": r"landslide|rock ?fall|mass wasting|boulder"},
    "periglacial_ice": {"polygons": r"polygon|patterned ground", "brain_terrain": r"brain",
                        "scalloped": r"scallop",
                        "debris_aprons_fill": r"lobate debris|lineated valley|concentric crater fill|glacier|viscous flow"},
    "polar": {"spiders_co2": r"spider|araneiform|\bco2\b|swiss cheese",
              "polar_layers_cap": r"polar layered|\b[ns]pld\b|residual cap|polar cap"},
    "impacts": {"new_impacts": r"new impact|recent impact|new crater|new dated|blast", "fresh_craters": r"fresh crater"},
    "layers_volcanic": {"layered": r"layer", "lava": r"lava|volcan", "channels": r"channel|inverted|delta"},
}

live = None


def _log(msg):
    print(msg, flush=True)
    if live:
        live.log(msg)


def km_between(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    c = math.sin(la1) * math.sin(la2) + math.cos(la1) * math.cos(la2) * math.cos(lo1 - lo2)
    return MARS_R * math.acos(max(-1.0, min(1.0, c)))


def read_index(cache):
    """RED observations from the HiRISE RDR cumulative index (fixed-width PDS table)."""
    cache.mkdir(parents=True, exist_ok=True)
    for ext in ("LBL", "TAB"):
        if not (cache / f"RDRCUMINDEX.{ext}").exists():
            (cache / f"RDRCUMINDEX.{ext}").write_bytes(http(f"{INDEX}.{ext}", deadline=1800))
    lbl = (cache / "RDRCUMINDEX.LBL").read_text()
    cols = {m[0]: (int(m[1]) - 1, int(m[1]) - 1 + int(m[2])) for m in
            re.findall(r"NAME\s*=\s*(\w+).*?START_BYTE\s*=\s*(\d+).*?BYTES\s*=\s*(\d+)", lbl, re.S)}
    rows = []
    for line in (cache / "RDRCUMINDEX.TAB").open("rb"):
        s = line.decode("latin1")
        g = lambda k: s[cols[k][0]:cols[k][1]].strip().strip('"').strip()
        if not g("PRODUCT_ID").endswith("_RED"):
            continue
        try:
            rows.append(dict(obs=g("OBSERVATION_ID"), why=g("RATIONALE_DESC"), inc=float(g("INCIDENCE_ANGLE")),
                             emi=float(g("EMISSION_ANGLE")), scale=float(g("MAP_SCALE")),
                             lat=(float(g("MINIMUM_LATITUDE")) + float(g("MAXIMUM_LATITUDE"))) / 2,
                             lon=(float(g("MINIMUM_LONGITUDE")) + float(g("MAXIMUM_LONGITUDE"))) / 2,
                             ls=float(g("SOLAR_LONGITUDE")), time=g("START_TIME")))
        except ValueError:
            continue
    return rows


def pick(rows, per_category, known):
    """Balanced choice: sub-landforms in turn, each cycling through incidence bins; seeded, so repeatable."""
    rng = np.random.default_rng(11)
    ok = [r for r in rows if r["scale"] == 0.5 and r["emi"] < 20 and r["obs"] not in known
          and not HARDWARE_WORDS.search(r["why"]) and min(km_between((r["lat"], r["lon"]), h) for h in HARDWARE) > EXCLUDE_KM]
    used, out = set(), {}
    for cat, subs in CATEGORIES.items():
        pools = {}
        for sub, pat in subs.items():
            rx = re.compile(pat, re.I)
            m = [r for r in ok if rx.search(r["why"]) and r["obs"] not in used]
            if cat == "layers_volcanic" and sub == "layered":
                m = [r for r in m if abs(r["lat"]) < 70]          # polar layers belong to `polar`
            pools[sub] = {}
            for b in INCIDENCE_BINS:
                inbin = [r for r in m if b[0] <= r["inc"] < b[1]]
                pools[sub][b] = [inbin[i] for i in rng.permutation(len(inbin))]
        chosen, turn = [], 0
        while len(chosen) < per_category and any(any(p.values()) for p in pools.values()):
            sub = list(subs)[turn % len(subs)]
            b = INCIDENCE_BINS[(turn // len(subs)) % len(INCIDENCE_BINS)]
            turn += 1
            for bb in [b] + INCIDENCE_BINS:                        # fall back to any bin if this one is empty
                while pools[sub][bb] and pools[sub][bb][-1]["obs"] in used:
                    pools[sub][bb].pop()
                if pools[sub][bb]:
                    r = pools[sub][bb].pop(); used.add(r["obs"]); chosen.append({**r, "sub": sub}); break
        out[cat] = chosen
    return out


def edr_url(obs, channel):
    orbit = int(obs.split("_")[1])
    lo = orbit // 100 * 100
    return f"{EDR}/{obs.split('_')[0]}/ORB_{lo:06d}_{lo + 99:06d}/{obs}/{obs}_{channel}.IMG"


def destripe(t):
    """Repair the odd single columns of the raw CCD (reading well off their neighbours): columns whose median
    stands out from a 15-column running median by > 4 MAD get the mean of their nearest good neighbours."""
    col = np.median(t, 0)
    smooth = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(col, 7, mode="edge"), 15), 1)
    dev = col - smooth
    odd = np.abs(dev) > 4 * np.median(np.abs(dev)) + 0.5
    t = t.copy()
    good = np.flatnonzero(~odd)
    for c in np.flatnonzero(odd):
        left, right = good[good < c], good[good > c]
        nb = [x for x in (left[-1] if left.size else None, right[0] if right.size else None) if x is not None]
        t[:, c] = t[:, nb].mean(1)
    return t


def fetch_tile(url):
    """Centre 1024 lines of one EDR channel (bin 2 -> 512 samples) as uint8, or None if unusable."""
    head = http(url, "bytes=0-32767").decode("latin1")
    image = head[re.search(r"^\s*OBJECT\s*=\s*IMAGE\s*$", head, re.M).start():]
    g = lambda k: int(re.search(rf"{k}\s*=\s*(\d+)", image).group(1))
    lines, samples, pre, suf = g("LINES"), g("LINE_SAMPLES"), g("LINE_PREFIX_BYTES"), g("LINE_SUFFIX_BYTES")
    start = int(re.search(r"\^IMAGE\s*=\s*(\d+)", head).group(1)) - 1
    binning = int(re.search(r"MRO:BINNING\s*=\s*(\d+)", head).group(1))
    if binning != BINNING or lines < TILE_LINES:
        return None, dict(lines=lines, binning=binning)
    rec = pre + samples + suf
    l0 = lines // 2 - TILE_LINES // 2
    raw = np.frombuffer(http(url, f"bytes={start + l0 * rec}-{start + (l0 + TILE_LINES) * rec - 1}"), np.uint8)
    t = raw.reshape(TILE_LINES, rec)[:, pre:pre + samples].astype(np.float32)
    bad = (t == 0) | (t == 255)
    if bad.mean() > 0.01:
        return None, dict(lines=lines, binning=binning)
    t = destripe(t)
    lo, hi = np.percentile(t[~bad], [0.5, 99.5])
    if hi - lo < 2:
        return None, dict(lines=lines, binning=binning)
    return (np.clip((t - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8), dict(lines=lines, binning=binning, line0=l0)


FIELDS = ["file", "category", "landform", "observation_id", "rationale", "channel", "lat", "lon",
          "incidence_deg", "emission_deg", "solar_longitude", "utc_start", "line0", "source_url"]


def build(out, per_category):
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.csv"
    rows = list(csv.DictReader(meta_path.open())) if meta_path.exists() else []
    done = {(r["observation_id"], r["channel"]) for r in rows}
    known = {re.sub(r"_(RED|IR|BG).*", "", p.stem) for p in Path("data/xenarch-imagery/hirise_png").glob("*.png")}

    live.step("Choosing observations")
    chosen = pick(read_index(Path("data/cache/hirise")), per_category, known)
    for cat, obs in chosen.items():
        _log(f"   {cat:<16} {len(obs)} observations: " + ", ".join(sorted({o['sub'] for o in obs})))
    jobs = [(cat, o) for cat, obs in chosen.items() for o in obs]

    live.step("Downloading")
    samples = []
    for k, (cat, o) in enumerate(jobs):
        live.progress(k, len(jobs), "observations")
        have = sum((o["obs"], ch) in done for ch in CHANNELS)
        for ch in CHANNELS:
            if have >= PER_OBS:
                break
            if (o["obs"], ch) in done:
                continue
            url = edr_url(o["obs"], ch)
            try:
                tile, info = fetch_tile(url)
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    _log(f"   download failed {o['obs']}_{ch}: {e}")
                continue                                    # 404: this CCD was off for the observation
            except Exception as e:
                _log(f"   download failed {o['obs']}_{ch}: {e}"); continue
            if tile is None:
                _log(f"   skipped {o['obs']}_{ch} (bin {info['binning']}, {info['lines']} lines, or gaps)"); continue
            have += 1
            save_tile(out, rows, cat, o, ch, url, tile, info)
            if have == 1:
                _log(f"   {cat:<16} {o['sub']:<20} {o['obs']}  i={o['inc']:.0f}°  {o['why'][:50]}")
                samples.append(tile / 255.0)
                live.samples(samples[-12:])
            time.sleep(0.5)
        with meta_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, FIELDS); w.writeheader(); w.writerows(rows)
    write_card(out, rows)
    counts = {cat: sum(r["category"] == cat for r in rows) for cat in CATEGORIES}
    live.metric(tiles=len(rows), **counts)
    _log(f"Done: {len(rows)} tiles in {out}  " + "  ".join(f"{k} {v}" for k, v in counts.items()))


def save_tile(out, rows, cat, o, ch, url, tile, info):
    f = f"{cat}/{o['obs']}_{ch}.png"
    (out / cat).mkdir(exist_ok=True)
    Image.fromarray(tile).save(out / f, optimize=True)
    rows.append(dict(file=f, category=cat, landform=o["sub"], observation_id=o["obs"], rationale=o["why"],
                     channel=ch, lat=f"{o['lat']:.4f}", lon=f"{o['lon']:.4f}", incidence_deg=o["inc"],
                     emission_deg=o["emi"], solar_longitude=o["ls"], utc_start=o["time"],
                     line0=info["line0"], source_url=url))


def write_card(out, rows):
    counts = {cat: sum(r["category"] == cat for r in rows) for cat in CATEGORIES}
    subs = {cat: sorted({r["landform"] for r in rows if r["category"] == cat}) for cat in CATEGORIES}
    inc = np.array([float(r["incidence_deg"]) for r in rows]) if rows else np.zeros(0)
    bins = "  ".join(f"{lo}-{hi}°: {int(((inc >= lo) & (inc < hi)).sum())}" for lo, hi in INCIDENCE_BINS)
    table = "\n".join(f"| {cat} | {', '.join(subs[cat]).replace('_', ' ')} | {counts[cat]} |" for cat in CATEGORIES)
    (out / "README.md").write_text(f"""---
license: cc0-1.0
task_categories: [image-classification]
tags: [mars, hirise, planetary-science, anomaly-detection, remote-sensing]
pretty_name: Xenarch HiRISE natural terrain
---
# Xenarch HiRISE natural terrain

{len(rows)} grayscale 1024×512 tiles (lines × samples) from MRO HiRISE raw RED images (EDR, bin 2, ~0.5 m/pixel),
chosen as *natural* Mars terrain for the memory bank of an unsupervised anomaly detector
(Xenarch, PatchCore). Each tile is the centre 1024 lines of one CCD channel (RED4_0, RED4_1, RED5_0,
RED5_1, or the next CCDs out when one is missing) of an observation, so the four tiles of an observation sit side by side at its centre.

| Folder | Landforms | Tiles |
|---|---|---|
{table}

Sun incidence (0° = sun overhead): {bins}

Observations were chosen by the HiRISE team's own description of why each was taken (`rationale`
in `metadata.csv`), taking sub-landforms in turn and spreading them over sun angles. The label says what
the observation was targeted at; a single central tile may show only part of it.

Skipped: observations whose description mentions hardware or landing sites, observations within
{EXCLUDE_KM} km of a known lander, rover or crash site, and observations already in xenarch-imagery.
The tiles have not been checked one by one for hardware.

Raw EDR values (8-bit, companded, uncalibrated) with odd single columns repaired (columns whose median stands out from
their neighbours by > 4 MAD are replaced by the mean of their good neighbours); each tile maps its 0.5-99.5 percentile to 0-255.
Tiles with more than 1% gap or saturated pixels were dropped.
Some tiles keep blocky noise in the first few dozen columns (the CCD channel edge).

Credit: NASA / JPL / University of Arizona (HiRISE). HiRISE data are public domain.
""")


def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    out = Path(_arg("--out", "data/hirise-set"))
    if "--upload" in sys.argv:
        upload(_arg("--upload", ""), out); sys.exit()
    live = LiveRun(__file__, str(out), ["Choosing observations", "Downloading"])
    try:
        build(out, int(_arg("--per-category", 16)))
        live.done()
    except BaseException as e:
        live.fail(e); raise

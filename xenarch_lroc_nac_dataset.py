"""
Xenarch LROC NAC set — high-resolution natural lunar terrain for the PatchCore memory bank.

The Moon side of xenarch-imagery is almost all Clementine (~100-200 m/pixel); only 6 LROC NAC
images (~0.5-1.5 m/pixel) exist, the scale where hardware would show. This script builds a set of
1024×1024 NAC tiles of *natural* terrain, grouped by landform and spread over sun angles:

  fresh_craters    small fresh craters, bright ejecta, boulder fields
  mass_wasting     crater walls: boulder tracks, rockfalls, landslides
  tectonic         wrinkle ridges (dorsa) and scarps (rupes)
  volcanic         rilles, lava-tube pits, domes, cones, irregular mare patches
  chains_terrain   crater chains (catenae), mare plains, farside highlands
  polar            polar terrain under very low sun and long shadows

How: target coordinates come from the IAU nomenclature (USGS) plus a few hand-listed features;
NASA ODE finds NAC CDR images over each target and one image per incidence bin is kept. Only a
1024-line strip through the target is downloaded (HTTP range, ~10 MB instead of ~530 MB) and cut into
a tile centred on the target plus up to three neighbours across the strip. Targets within 10 km of a landing or impact site are skipped so no hardware reaches
the "natural" memory.

  .venv/bin/python xenarch_lroc_nac_dataset.py [--out data/lroc-nac-set] [--per-category 12] [--suns 2]
  .venv/bin/python xenarch_lroc_nac_dataset.py --upload <user>/<repo>     (after `hf auth login`)
"""
import csv, io, json, math, re, struct, subprocess, sys, time, urllib.error, urllib.parse, urllib.request, zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from xenarch_live import LiveRun

ODE = "https://oderest.rsl.wustl.edu/live2/"
NAMES_ZIP = "https://planetarynames.wr.usgs.gov/shapefiles/MOON_nomenclature_center_pts.zip"
TILE, TILES_PER_IMAGE = 1024, 4
MOON_KM_PER_DEG = 1737.4 * math.pi / 180
INCIDENCE_BINS = [(0, 45), (45, 65), (65, 80), (80, 90)]   # high sun ... grazing sun
EXCLUDE_KM = 10

# Landing, rover and impact sites (lat, east lon). IAU "Statio" / astronaut-named features are added too.
HARDWARE = [
    (0.674, 23.473), (-3.012, 336.578), (-3.646, 342.528), (26.132, 3.634), (-8.973, 15.500), (20.191, 30.772),
    (-2.474, 316.66), (-3.016, 336.58), (1.461, 23.195), (0.474, 358.573), (-40.981, 348.51),
    (7.08, 295.63), (18.87, 297.95), (-0.513, 56.364), (38.32, 324.99), (3.787, 56.625), (25.85, 30.45),
    (25.83, 30.92), (12.714, 62.213), (44.12, 340.49), (-45.44, 177.59), (43.06, 308.08), (-41.64, 206.0),
    (-69.37, 32.32), (-13.32, 25.25), (-80.13, 1.44), (18.56, 61.81), (-84.79, 29.2), (32.6, 19.3),
    (47.58, 44.09), (-57.86, 61.36), (-84.73, 310.4), (75.6, 333.4), (11.85, 266.8), (-10.35, 339.33),
    (2.64, 24.79), (-12.83, 357.63), (-34.26, 313.8), (-3.94, 338.79), (-2.55, 332.1), (-8.09, 333.99),
    (-1.51, 348.12), (1.3, 336.2), (-4.21, 347.69), (-3.42, 340.33), (26.36, 0.25), (19.96, 30.5),
    (-1.5, 52.36), (-65.5, 80.4), (-87.7, 44.0),
]

# (name looked up in the IAU list, where to aim) — aim: "center", "ejecta" (0.7 D out) or "rim" (0.5 D out)
CATEGORIES = {
    "fresh_craters": [(n, "ejecta") for n in [
        "Linné", "Giordano Bruno", "Censorinus", "Dawes", "Bessel", "Moltke", "Proclus", "Byrgius A",
        "Necho", "Hell Q", "Stevinus A", "Messier A", "Kepler", "Aristarchus"]],
    "mass_wasting": [(n, "rim") for n in [
        "Vitello", "Tsiolkovskiy", "Schrödinger", "Jackson", "King", "Tycho", "Copernicus", "Aristillus",
        "Theophilus", "Eratosthenes", "Aristoteles", "Plato", "Censorinus", "Giordano Bruno"]],
    "tectonic": [("@Dorsum, dorsa", "center"), ("@Rupes, rupēs", "center")],
    "volcanic": [("Vallis Schröteri", "center"), ("Rima Hadley", "center"), ("Rima Sharp", "center"),
                 ("Rima Marius", "center"), ("Rimae Prinz", "center"), ("Mons Gruithuisen Gamma", "center"),
                 ("Mons Rümker", "center"),
                 ("Marius Hills pit", (14.09, 303.23)), ("Mare Tranquillitatis pit", (8.335, 33.222)),
                 ("Mare Ingenii pit", (-35.95, 166.06)), ("Ina", (18.65, 5.30)),
                 ("Sosigenes IMP", (8.335, 19.071)), ("Maskelyne IMP", (4.33, 33.75)),
                 ("Marius Hills domes", (13.3, 305.0)), ("Hortensius domes", (7.5, 332.3))],
    "chains_terrain": [("@Catena, catenae", "center"), ("Mare Imbrium", "center"), ("Mare Serenitatis", "center"),
                       ("Mare Crisium", "center"), ("Mare Nubium", "center"), ("Mare Moscoviense", "center"),
                       ("Farside highlands 1", (10.0, 160.0)), ("Farside highlands 2", (-20.0, 200.0)),
                       ("Farside highlands 3", (30.0, 190.0)), ("Farside highlands 4", (-5.0, 120.0))],
    "polar": [(n, "rim") for n in [
        "Shackleton", "de Gerlache", "Haworth", "Amundsen", "Faustini", "Sverdrup", "Shoemaker",
        "Peary", "Hermite", "Rozhdestvenskiy", "Plaskett", "Byrd", "Whipple", "Nansen"]],
}

live = None


def _log(msg):
    print(msg, flush=True)
    if live:
        live.log(msg)


OPENER = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())   # the NASA store redirects via a cookie


def http(url, rng=None, tries=8, deadline=90):
    """GET with polite retries: the NASA store answers 429 when pushed and sometimes stalls mid-transfer,
    so each attempt also has a hard `deadline` in seconds (the socket timeout alone misses a trickle)."""
    req = urllib.request.Request(url, headers={"User-Agent": "xenarch-dataset/1.0", **({"Range": rng} if rng else {})})
    for k in range(tries):
        try:
            t0, buf = time.monotonic(), bytearray()
            with OPENER.open(req, timeout=30) as r:
                while chunk := r.read(1 << 16):
                    buf += chunk
                    if time.monotonic() - t0 > deadline:
                        raise TimeoutError("stalled transfer")
            return bytes(buf)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) and not (e.code == 302 and k < tries - 1):
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            pass
        time.sleep(min(300, 5 * 2 ** k))
    raise RuntimeError(f"gave up on {url}")


def http_range(url, start, n, piece=2 << 20):
    """Bytes [start, start + n) in 2 MB pieces, each retried on its own."""
    return b"".join(http(url, f"bytes={a}-{min(start + n, a + piece) - 1}") for a in range(start, start + n, piece))


# ── geometry ────────────────────────────────────────────────────────────────
def unit(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


def km_between(a, b):
    return MOON_KM_PER_DEG * math.degrees(math.acos(float(np.clip(unit(*a) @ unit(*b), -1, 1))))


def offset(lat, lon, km, bearing_deg):
    """Point `km` away from (lat, lon) along a bearing (0 = north)."""
    d, b, la, lo = km / 1737.4, math.radians(bearing_deg), math.radians(lat), math.radians(lon)
    la2 = math.asin(math.sin(la) * math.cos(d) + math.cos(la) * math.sin(d) * math.cos(b))
    lo2 = lo + math.atan2(math.sin(b) * math.sin(d) * math.cos(la), math.cos(d) - math.sin(la) * math.sin(la2))
    return math.degrees(la2), math.degrees(lo2) % 360


def locate(footprint, lat, lon):
    """Fractional (sample, line) of a point inside a NAC footprint.
    ODE footprints run v0 = (last sample, line 0) -> v1 = (last sample, last line) -> v2 -> v3 = (sample 0, line 0);
    checked on Linné (line axis) and the Mare Tranquillitatis pit (both axes, within ~60 pixels).
    Solved on the tangent plane at the target."""
    t = unit(lat, lon)
    east = np.cross([0, 0, 1], t); east = east / (np.linalg.norm(east) or 1) if np.linalg.norm(east) > 1e-9 else np.array([1., 0, 0])
    north = np.cross(t, east)
    P = [np.array([(unit(la, lo) @ east), (unit(la, lo) @ north)]) / (unit(la, lo) @ t) for la, lo in footprint[:4]]
    u = v = 0.5
    for _ in range(30):     # Newton on the bilinear patch
        f = (1 - u) * (1 - v) * P[0] + (1 - u) * v * P[1] + u * v * P[2] + u * (1 - v) * P[3]
        du = -(1 - v) * P[0] - v * P[1] + v * P[2] + (1 - v) * P[3]
        dv = -(1 - u) * P[0] + (1 - u) * P[1] + u * P[2] - u * P[3]
        J = np.stack([du, dv], 1)
        try:
            step = np.linalg.solve(J, -f)
        except np.linalg.LinAlgError:
            return None
        u, v = u + step[0], v + step[1]
        if np.abs(step).max() < 1e-7:
            break
    return (1 - u, v) if -0.02 <= u <= 1.02 and -0.02 <= v <= 1.02 else None


# ── targets ────────────────────────────────────────────────────────────────
def iau_names(cache):
    path = cache / "MOON_nomenclature_center_pts.dbf"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(io.BytesIO(http(NAMES_ZIP))).extractall(cache)
    raw = path.read_bytes()
    n, hl, rl = struct.unpack("<IHH", raw[4:12])
    fields, p = [], 32
    while raw[p] != 0x0D:
        fields.append((raw[p:p + 11].split(b"\0")[0].decode(), raw[p + 16])); p += 32
    rows = []
    for i in range(n):
        r, q, row = raw[hl + i * rl: hl + (i + 1) * rl], 1, {}
        for name, l in fields:
            row[name] = r[q:q + l].decode("utf8", "replace").strip(); q += l
        rows.append(row)
    return rows


def build_targets(names, per_category):
    by_name = {r["clean_name"].lower(): r for r in names}
    by_name.update({r["name"].lower(): r for r in names})
    exclude = HARDWARE + [(float(r["center_lat"]), float(r["center_lon"])) for r in names
                          if r["type"] in ("Astronaut-named features", "Statio")]
    rng = np.random.default_rng(7)
    out = {}
    for cat, items in CATEGORIES.items():
        rows = []
        for name, aim in items:
            if name.startswith("@"):     # every IAU feature of that type, spread evenly
                feats = sorted((r for r in names if r["type"] == name[1:]), key=lambda r: r["name"])
                rows += [(r["name"], float(r["center_lat"]), float(r["center_lon"]) % 360) for r in feats]
                continue
            if isinstance(aim, tuple):
                rows.append((name, *aim)); continue
            r = by_name.get(name.lower())
            if not r:
                _log(f"   not in the IAU list, skipped: {name}"); continue
            lat, lon, d = float(r["center_lat"]), float(r["center_lon"]) % 360, float(r["diameter"] or 0)
            if aim != "center" and d > 4:
                lat, lon = offset(lat, lon, d * (0.7 if aim == "ejecta" else 0.5), rng.uniform(0, 360))
            rows.append((r["name"], lat, lon))
        kept = []
        for name, lat, lon in rows:
            near = min(km_between((lat, lon), h) for h in exclude)
            if near < EXCLUDE_KM:
                _log(f"   {near:.1f} km from a landing/impact site, skipped: {name}")
            else:
                kept.append((name, lat, lon))
        if len(kept) > per_category:
            kept = [kept[int(i)] for i in np.linspace(0, len(kept) - 1, per_category)]
        out[cat] = kept
    return out


# ── NAC search and download ────────────────────────────────────────────────
def find_images(lat, lon, suns):
    """One NAC CDR per incidence bin over the point, sharpest and most nadir first, up to `suns`."""
    d = 0.01
    dl = min(179, d / max(math.cos(math.radians(lat)), 0.02))
    q = dict(query="product", target="moon", ihid="LRO", iid="LROC", pt="CDRNAC4", results="mf",
             output="JSON", limit=300, minlat=max(-90, lat - d), maxlat=min(90, lat + d),
             westlon=max(0, lon - dl), eastlon=min(360, lon + dl))
    res = json.loads(http(ODE + "?" + urllib.parse.urlencode(q)))["ODEResults"]
    prods = (res.get("Products") or {}).get("Product") or []
    prods = prods if isinstance(prods, list) else [prods]
    cands = []
    for p in prods:
        try:
            inc, emi, mres = float(p["Incidence_angle"]), float(p["Emission_angle"]), float(p["Map_resolution"])
            url = next(f["URL"] for f in p["Product_files"]["Product_file"] if f["FileName"].upper().endswith(".IMG"))
            poly = [tuple(map(float, xy.split()))[::-1] for xy in
                    re.search(r"\(\(([^()]+)\)\)", p["Footprint_geometry"]).group(1).split(",")]
        except (KeyError, ValueError, AttributeError, StopIteration):
            continue
        if mres > 2.0 or emi > 25 or len(poly) != 5:
            continue
        uv = locate(poly, lat, lon)
        if uv:
            cands.append(dict(product=p["Product_name"][:-4], url=url, inc=inc, emi=emi, res=mres,
                              phase=float(p.get("Phase_angle") or "nan"), time=p.get("UTC_start_time", ""), uv=uv))
    picked = []
    for lo, hi in INCIDENCE_BINS:
        inbin = sorted((c for c in cands if lo <= c["inc"] < hi), key=lambda c: (c["res"], c["emi"]))
        if inbin:
            picked.append(inbin[0])
    order = sorted(picked, key=lambda c: -c["inc"] if lat and abs(lat) > 75 else abs(c["inc"] - 55))
    return order[:suns]


def fetch_tiles(c):
    """Download the 1024-line strip through the target; return the tile centred on it (index 0) and its
    neighbours across the strip, as ((line offset, sample offset, index), uint8 tile)."""
    lbl = http(c["url"][:-4] + ".xml").decode()
    lines = int(re.search(r"<axis_name>Line</axis_name>\s*<elements>(\d+)", lbl).group(1))
    samples = int(re.search(r"<axis_name>Sample</axis_name>\s*<elements>(\d+)", lbl).group(1))
    offset_b = int(re.findall(r'<offset unit="byte">(\d+)</offset>', lbl)[-1])
    u, v = c["uv"]
    l0 = int(np.clip(v * lines - TILE / 2, 0, max(0, lines - TILE)))
    if lines - l0 < TILE:
        return []
    a = offset_b + l0 * samples * 2
    img = np.frombuffer(http_range(c["url"], a, TILE * samples * 2), "<i2").reshape(TILE, samples)
    img = img.astype(np.float32)
    c.update(line0=l0, sample0=0, lines=lines, samples=samples)
    s_mid = int(np.clip(u * samples - TILE / 2, 0, samples - TILE))
    starts = [s_mid] + [s for k in range(1, 5) for s in (s_mid - k * TILE, s_mid + k * TILE) if 0 <= s <= samples - TILE]
    tiles = []
    for k, j in enumerate(starts[:TILES_PER_IMAGE]):
        i, t = 0, img[:, j:j + TILE]
        bad = (t <= -32768) | (t >= 32767)
        if bad.mean() > 0.01:
            continue
        lo, hi = np.percentile(t[~bad], [0.5, 99.5])
        if hi - lo < 1e-6 or np.percentile(t[~bad], 90) <= 0:     # blank or fully shadowed
            continue
        tiles.append(((i, j, k), (np.clip((t - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)))
    return tiles


FIELDS = ["file", "category", "target", "target_lat", "target_lon", "product_id", "incidence_deg", "emission_deg",
          "phase_deg", "resolution_m", "utc_start", "line0", "sample0", "source_url"]


def build(out, per_category, suns):
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.csv"
    rows = list(csv.DictReader(meta_path.open())) if meta_path.exists() else []
    done = {(r["product_id"], r["target"]) for r in rows}

    live.step("Finding images")
    jobs_path = out / f"jobs_{per_category}x{suns}.json"      # the ODE search is slow: reuse it on a restart
    if jobs_path.exists():
        jobs = [tuple(j) for j in json.loads(jobs_path.read_text())]
        _log(f"   reusing {len(jobs)} images found earlier ({jobs_path.name})")
    else:
        targets = build_targets(iau_names(Path("data/cache/iau_moon")), per_category)
        total, jobs, seen = sum(map(len, targets.values())), [], 0
        for cat, items in targets.items():
            for name, lat, lon in items:
                live.progress(seen, total, "targets"); seen += 1
                try:
                    imgs = find_images(lat, lon, suns)
                except Exception as e:
                    _log(f"   search failed for {name}: {e}"); continue
                _log(f"   {cat:<15} {name:<28} {lat:7.2f} {lon:7.2f}  sun incidence: "
                     + (", ".join(f"{c['inc']:.0f}°" for c in imgs) or "no NAC image"))
                jobs += [(cat, name, lat, lon, c) for c in imgs]
                time.sleep(0.5)
        jobs_path.write_text(json.dumps(jobs))

    live.step("Downloading")
    samples = []
    for k, (cat, name, lat, lon, c) in enumerate(jobs):
        live.progress(k, len(jobs), "strips")
        if (c["product"], name) in done:
            continue
        try:
            tiles = fetch_tiles(c)
        except Exception as e:
            _log(f"   download failed {c['product']} ({name}): {e}"); continue
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        (out / cat).mkdir(exist_ok=True)
        for (i, j, k), t in tiles:
            f = f"{cat}/{c['product']}_{slug}_{k}.png"
            Image.fromarray(t).save(out / f, optimize=True)
            rows.append(dict(file=f, category=cat, target=name, target_lat=f"{lat:.4f}", target_lon=f"{lon:.4f}",
                             product_id=c["product"], incidence_deg=c["inc"], emission_deg=c["emi"],
                             phase_deg=c["phase"], resolution_m=c["res"], utc_start=c["time"],
                             line0=c["line0"] + i, sample0=c["sample0"] + j, source_url=c["url"]))
        samples += [t / 255.0 for _, t in tiles[:1]]
        with meta_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, FIELDS); w.writeheader(); w.writerows(rows)
        _log(f"   {cat:<15} {name:<28} {c['product']}  i={c['inc']:.0f}°  {c['res']:.2f} m/px  -> {len(tiles)} tiles")
        if len(samples) % 6 == 0:
            live.samples(samples[-12:])
        time.sleep(3)
    write_card(out, rows)
    counts = {cat: sum(r["category"] == cat for r in rows) for cat in CATEGORIES}
    live.metric(tiles=len(rows), **counts)
    _log(f"Done: {len(rows)} tiles in {out}  " + "  ".join(f"{k} {v}" for k, v in counts.items()))


def write_card(out, rows):
    counts = {cat: sum(r["category"] == cat for r in rows) for cat in CATEGORIES}
    inc = np.array([float(r["incidence_deg"]) for r in rows]) if rows else np.zeros(0)
    bins = "  ".join(f"{lo}-{hi}°: {int(((inc >= lo) & (inc < hi)).sum())}" for lo, hi in INCIDENCE_BINS)
    (out / "README.md").write_text(f"""---
license: cc0-1.0
task_categories: [image-classification]
tags: [moon, lroc, planetary-science, anomaly-detection, remote-sensing]
pretty_name: Xenarch LROC NAC natural terrain
---
# Xenarch LROC NAC natural terrain

{len(rows)} grayscale 1024×1024 tiles cut from Lunar Reconnaissance Orbiter Camera Narrow Angle Camera
calibrated images (CDR, ~0.5-2 m/pixel), chosen as *natural* lunar terrain for the memory bank of an
unsupervised anomaly detector (Xenarch, PatchCore). Each target was imaged under several sun angles.

| Folder | Landform | Tiles |
|---|---|---|
| fresh_craters | small fresh craters, bright ejecta, boulder fields | {counts['fresh_craters']} |
| mass_wasting | crater walls: boulder tracks, rockfalls, landslides | {counts['mass_wasting']} |
| tectonic | wrinkle ridges (dorsa), scarps (rupes) | {counts['tectonic']} |
| volcanic | rilles, lava-tube pits, domes, cones, irregular mare patches | {counts['volcanic']} |
| chains_terrain | crater chains, mare plains, farside highlands | {counts['chains_terrain']} |
| polar | polar terrain under very low sun | {counts['polar']} |

Sun incidence (0° = sun overhead): {bins}

`metadata.csv` gives, per tile: target name and coordinates, source product, incidence / emission / phase
angles, resolution, the tile's first line and sample in the source image and the source URL.

Targets come from the IAU Gazetteer of Planetary Nomenclature (USGS) plus a few hand-listed features
(lava-tube pits, irregular mare patches, dome fields); images were found with NASA ODE. Tiles are
located from the archive footprints (rounded to 0.01°, ~300 m); tile index 0 is centred on the target,
1-3 are its neighbours across the image strip. A small feature can still sit off-centre in tile 0. Targets within {EXCLUDE_KM} km of a known landing or impact site were skipped,
but the set has not been checked tile by tile for hardware.

Stretch: each tile maps its 0.5-99.5 percentile of I/F to 0-255; tiles with >1% missing pixels or fully
in shadow were dropped.

Credit: NASA / GSFC / Arizona State University (LROC). LRO data are public domain.
""")


def upload(repo, out):
    """Create the dataset repo (private) and upload the folder with the logged-in `hf` CLI."""
    hf = str(Path(sys.executable).with_name("hf"))
    subprocess.run([hf, "repos", "create", repo, "--repo-type", "dataset", "--private", "--exist-ok"], check=True)
    subprocess.run([hf, "upload-large-folder", repo, str(out), "--repo-type", "dataset",
                    "--exclude", "jobs_*.json", "--exclude", ".cache/**"], check=True)
    print(f"https://huggingface.co/datasets/{repo}")


def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    out = Path(_arg("--out", "data/lroc-nac-set"))
    if "--upload" in sys.argv:
        upload(_arg("--upload", ""), out); sys.exit()
    live = LiveRun(__file__, str(out), ["Finding images", "Downloading"])
    try:
        build(out, int(_arg("--per-category", 12)), int(_arg("--suns", 2)))
        live.done()
    except BaseException as e:
        live.fail(e); raise

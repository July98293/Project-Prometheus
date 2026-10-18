"""
Xenarch Chandrayaan-2 OHRC set — the sharpest natural lunar terrain available (0.23-0.26 m/px) for the memory bank.

ISRO's Orbiter High Resolution Camera takes 12,000-sample panchromatic strips (3 km x ~22 km) at ~0.25 m/px,
twice as sharp as LROC NAC. ISRO's own portals (pradan / chmapbrowse.issdc.gov.in) need a registered ISSDC
login to download; the calibrated PDS4 products of three scenes are mirrored on the Internet Archive
(login-free), and that is what this script uses by default. Each scene zip (~800 MB) holds the 8-bit
calibrated image (.img + PDS4 .xml label), a geometry grid (.csv: lon, lat, scan, pix) and a browse PNG.

Per scene the strip is cut into 1024x1024 tiles on a grid; `--per-scene` tiles are kept, spread evenly along
the strip, skipping tiles that are mostly saturated, black or featureless. The tile centre's lat / lon is
interpolated from the geometry grid. Scenes within 10 km of a landing / impact site are refused (none of the
mirrored scenes is).

  .venv/bin/python xenarch_ohrc_dataset.py [--out data/ohrc-set] [--per-scene 100] [--cache data/cache/ohrc]
  .venv/bin/python xenarch_ohrc_dataset.py --scenes-from DIR     use scene zips already downloaded from ISSDC
  .venv/bin/python xenarch_ohrc_dataset.py --upload jules1235813/xenarch   (-> natural-terrain_training/moon_chandrayaan2-ohrc)

Landform: all three mirrored scenes are cratered southern highlands (62-74 S), so the single landform folder is
`cratered_highlands`; `target` is the scene's product id. incidence / emission / phase are not in the OHRC
label and are left empty.
"""
import csv, math, re, sys, time, urllib.request, zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.interpolate import LinearNDInterpolator

from xenarch_hf import split_of
from xenarch_live import LiveRun
from xenarch_lroc_nac_dataset import HARDWARE, EXCLUDE_KM, MOON_KM_PER_DEG

MIRROR = ("https://archive.org/download/chandrayaan-2-high-resolution-images-of-the-moon/"
          "Optical%20High%20Resolution%20Camera%20%28OHRC%29/")
SCENES = ["ch2_ohr_ncp_20200229T0739312111_d_img_d18", "ch2_ohr_ncp_20200229T0938004033_d_img_d32",
          "ch2_ohr_ncp_20200824T1003365280_d_img_d18"]
TILE, LANDFORM, INSTRUMENT = 1024, "cratered_highlands", "Chandrayaan-2 OHRC"
FIELDS = ["file_name", "split", "body", "instrument", "landform", "target", "frame_id", "lat", "lon_east", "incidence_deg",
          "emission_deg", "phase_deg", "resolution_m_px", "utc_start", "line0", "sample0", "source_url"]

live = None


def _log(msg):
    print(msg, flush=True)
    if live:
        live.log(msg)


def download(url, dest):
    """Resumable download (archive.org drops long transfers now and then)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    for attempt in range(20):
        have = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "xenarch-dataset/1.0", **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, tmp.open("ab") as f:
                total = have + int(r.headers.get("Content-Length", 0))
                while chunk := r.read(1 << 20):
                    f.write(chunk)
                    if f.tell() % (50 << 20) < (1 << 20):
                        live.progress(f.tell(), total, "bytes") if live else None
            tmp.rename(dest)
            return dest
        except Exception as e:  # noqa: BLE001
            _log(f"   retry {attempt + 1}: {e}")
            time.sleep(15)
    raise RuntimeError(f"could not download {url}")


def label(xml_bytes):
    """Scene facts from the PDS4 label."""
    t = ET.fromstring(xml_bytes)
    g = lambda tag: next((el.text.strip() for el in t.iter() if el.tag.split("}")[-1] == tag), "")
    lines = samples = None
    for el in t.iter():
        if el.tag.split("}")[-1] == "Axis_Array":
            name = next(c.text for c in el if c.tag.split("}")[-1] == "axis_name")
            n = int(next(c.text for c in el if c.tag.split("}")[-1] == "elements"))
            lines, samples = (n, samples) if name == "Line" else (lines, n)
    return dict(lines=lines, samples=samples, utc=g("start_date_time"), res=float(g("pixel_resolution")),
                orbit=g("imaging_orbit_number"), alt=g("spacecraft_altitude"),
                lat=float(g("upper_left_latitude")), lon=float(g("upper_left_longitude")))


def geometry(csv_text):
    """(scan, pix) -> (lat, lon) interpolator from the geometry grid. The user guide says the columns are
    lon, lat, scan, pix; the files say `Longitude,Lattitude,Pixel,Scan` (sic), so match by prefix."""
    rows = list(csv.reader(csv_text.splitlines()))
    hdr = [h.strip().lower() for h in rows[0]]
    col = lambda *pre: next(i for i, h in enumerate(hdr) if h.startswith(pre))
    i_lon, i_lat, i_scan, i_pix = col("lon"), col("lat"), col("scan"), col("pix")
    pts = np.array([[float(r[i_scan]), float(r[i_pix])] for r in rows[1:] if len(r) > i_pix])
    val = np.array([[float(r[i_lat]), float(r[i_lon])] for r in rows[1:] if len(r) > i_pix])
    return LinearNDInterpolator(pts, val)


def near_hardware(lat, lon):
    return any(math.hypot((lat - a) * MOON_KM_PER_DEG, ((lon - b + 180) % 360 - 180) * MOON_KM_PER_DEG * math.cos(math.radians(lat))) < EXCLUDE_KM
               for a, b in HARDWARE)


def tiles_of(scene, zpath, out, per_scene, rows, done):
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        xml = next(n for n in names if n.endswith("_d_img_" + scene[-3:] + ".xml") or (n.startswith("data/") and n.endswith(".xml")))
        img = next(n for n in names if n.startswith("data/") and n.endswith(".img"))
        grd = next(n for n in names if n.endswith(".csv") and "geometry/" in n)
        L = label(z.read(xml))
        if near_hardware(L["lat"], L["lon"]):
            _log(f"   {scene}: within {EXCLUDE_KM} km of a landing / impact site, skipped"); return
        interp = geometry(z.read(grd).decode())
        raw = (out.parent / "cache" / "ohrc" / (scene + ".img"))
        if not raw.exists() or raw.stat().st_size != L["lines"] * L["samples"]:
            _log(f"   extracting {img} ({L['lines'] * L['samples'] / 1e9:.2f} GB)")
            with z.open(img) as src, raw.open("wb") as dst:
                while chunk := src.read(16 << 20):
                    dst.write(chunk)
    a = np.memmap(raw, np.uint8, "r", shape=(L["lines"], L["samples"]))
    nx, ny = L["samples"] // TILE, L["lines"] // TILE
    x0s = [round(i * (L["samples"] - TILE) / max(1, nx - 1)) for i in range(nx)]
    y0s = [round(j * (L["lines"] - TILE) / max(1, ny - 1)) for j in range(ny)]
    cands = [(y0, x0) for y0 in y0s for x0 in x0s]
    # spread the kept tiles along the strip: every k-th candidate in reading order, then fill with the rest
    order = cands[::max(1, len(cands) // per_scene)] + [c for k, c in enumerate(cands) if k % max(1, len(cands) // per_scene)]
    kept = 0
    for y0, x0 in order:
        if kept >= per_scene:
            break
        t = np.asarray(a[y0:y0 + TILE, x0:x0 + TILE])
        bad = ((t <= 1) | (t >= 254)).mean()
        if bad > 0.1 or t.std() < 6:
            continue
        name = f"{LANDFORM}/{scene[:31]}_{y0}_{x0}.png"
        if name in done:
            kept += 1; continue
        (out / LANDFORM).mkdir(parents=True, exist_ok=True)
        Image.fromarray(t).save(out / name, optimize=True)
        lat, lon = interp(y0 + TILE / 2, x0 + TILE / 2)
        lon = lon % 360
        rows.append(dict(file_name=name, split=split_of(scene), body="Moon", instrument=INSTRUMENT, landform=LANDFORM,
                         target=scene, frame_id=scene, lat=f"{lat:.4f}", lon_east=f"{lon:.4f}", incidence_deg="", emission_deg="",
                         phase_deg="", resolution_m_px=f"{L['res']:.3f}", utc_start=L["utc"], line0=y0, sample0=x0,
                         source_url=MIRROR + scene + ".zip"))
        kept += 1
        if live and kept % 10 == 0:
            live.samples([t[::8, ::8] / 255.0])
    _log(f"   {scene}: {kept} tiles, {L['res']:.3f} m/px, {L['lat']:.2f} {L['lon']:.2f}, orbit {L['orbit']}")


def build(out, per_scene, cache, scenes_from):
    out.mkdir(parents=True, exist_ok=True)
    meta = out / "metadata.csv"
    rows = list(csv.DictReader(meta.open())) if meta.exists() else []
    done = {r["file_name"] for r in rows}
    zips = sorted(Path(scenes_from).glob("ch2_ohr_ncp_*.zip")) if scenes_from else []
    if not zips:
        zips = []
        for k, s in enumerate(SCENES):
            live.step(f"Scene {k + 1}/{len(SCENES)}: download") if live else None
            dest = cache / (s + ".zip")
            if not dest.exists():
                _log(f"   downloading {s}.zip (~800 MB)")
                download(MIRROR + s + ".zip", dest)
            zips.append(dest)
    for k, zpath in enumerate(zips):
        live.step(f"Scene {k + 1}/{len(zips)}: tiles") if live else None
        tiles_of(zpath.name[:-4], zpath, out, per_scene, rows, done)
        with meta.open("w", newline="") as fh:
            w = csv.DictWriter(fh, FIELDS); w.writeheader(); w.writerows(rows)
    live.metric(tiles=len(rows), scenes=len(zips)) if live else None
    _log(f"Done: {len(rows)} tiles in {out}")


def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    out = Path(_arg("--out", "data/ohrc-set"))
    if "--upload" in sys.argv:
        import xenarch_hf
        xenarch_hf.publish_terrain("ohrc", out, _arg("--upload", xenarch_hf.REPO)); sys.exit()
    live = LiveRun(__file__, str(out), [f"Scene {k + 1}/{len(SCENES)}: download" for k in range(len(SCENES))] +
                   [f"Scene {k + 1}/{len(SCENES)}: tiles" for k in range(len(SCENES))]) if "--no-live" not in sys.argv else None
    try:
        build(out, int(_arg("--per-scene", 100)), Path(_arg("--cache", "data/cache/ohrc")), _arg("--scenes-from", None))
        live.done() if live else None
    except BaseException as e:
        live.fail(e) if live else None
        raise

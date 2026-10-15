"""
Xenarch artifact site images — 1024×1024 crops centred on each catalogued object, cut from the frames in
frames.csv / mars_mission_frames.csv (run build_catalog.py first).

  Moon  LROC NAC CDR: only the 1024 lines through the object are downloaded (HTTP range, ~10 MB per crop).
        Per object: the sharpest usable frame in each sun-incidence bin, plus one sunlit frame from before
        arrival when there is one (Chang'e 3-6). Objects that fall in an existing crop of the same frame
        are listed in it instead of getting their own (Apollo LM / ALSEP / LRV).
  Mars  HiRISE RED RDR (JPEG 2000, ~500 MB each): the file is position-ordered (PCRL) with packet lengths in
        PLT markers, so only the packets of the precincts around the object are downloaded (a few MB). They
        are rebuilt into a valid codestream with empty packets everywhere else and `opj_decompress -d`
        decodes just that region. Every usable frame, plus sunlit frames from before arrival.

Writes PNGs and metadata.csv (Hugging Face imagefolder layout) to data/artifact-site-images/.
Needs `opj_decompress` (brew install openjpeg).

  .venv/bin/python artifact-site-catalog/download_images.py [--out data/artifact-site-images] [--only moon|mars] [--no-live]
  .venv/bin/python artifact-site-catalog/download_images.py --upload jules1235813/xenarch   (into hardware/images/ of the single dataset)
"""
import csv, math, re, struct, subprocess, sys, tempfile, time
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from xenarch_live import LiveRun
from xenarch_lroc_nac_dataset import http, http_range

CROP = 2048                 # the object can sit ~300 m off the catalogue pixel (NAC footprints, HiRISE pointing)
INCIDENCE_BINS = [(0, 45), (45, 65), (65, 80), (80, 90)]
SAME_CROP_PX = 300          # an object this close to an existing crop centre of the same frame joins that crop
MARGIN = 16                 # extra wavelet coefficients fetched around the crop at every resolution level
OPJ = "opj_decompress"

FIELDS = ["file_name", "site_id", "object", "mission", "body", "label", "objects_in_crop", "obj_x", "obj_y",
          "frame_id", "instrument", "utc_start", "after_arrival", "days_since_arrival", "incidence_deg",
          "sun_elevation_deg", "emission_deg", "phase_deg", "resolution_m_px", "crop_x0", "crop_y0",
          "position_note", "source_url"]

live = None


def _log(msg):
    print(msg, flush=True)
    if live:
        live.log(msg)


def stretch(a, valid):
    lo, hi = np.percentile(a[valid], [0.5, 99.5]) if valid.any() else (0, 1)
    return (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


# ── Moon: LROC NAC CDR strip ───────────────────────────────────────────────
_nac_labels = {}


def nac_label(f):
    """(lines, samples, byte offset of the image) from the PDS4 label, cached per frame."""
    if f["frame_id"] not in _nac_labels:
        lbl = http(f["label_url"]).decode()
        assert "SignedLSB2" in lbl, "unexpected NAC data type"
        _nac_labels[f["frame_id"]] = (int(re.search(r"<axis_name>Line</axis_name>\s*<elements>(\d+)", lbl).group(1)),
                                      int(re.search(r"<axis_name>Sample</axis_name>\s*<elements>(\d+)", lbl).group(1)),
                                      int(re.findall(r'<offset unit="byte">(\d+)</offset>', lbl)[-1]))
    return _nac_labels[f["frame_id"]]


def nac_pixel(f):
    lines, samples, _ = nac_label(f)
    return float(f["obj_sample_frac"]) * samples, float(f["obj_line_frac"]) * lines


def nac_crop(f):
    lines, samples, start = nac_label(f)
    cs, cl = nac_pixel(f)
    y0 = int(np.clip(round(cl - CROP / 2), 0, lines - CROP))
    x0 = int(np.clip(round(cs - CROP / 2), 0, samples - CROP))
    raw = np.frombuffer(http_range(f["image_url"], start + y0 * samples * 2, CROP * samples * 2), "<i2")
    a = raw.reshape(CROP, samples)[:, x0:x0 + CROP].astype(np.float32)
    valid = a > -32752                                    # below that: missing / saturated special constants
    return stretch(a, valid), x0, y0, (cs - x0, cl - y0), 1 - valid.mean()


# ── Mars: HiRISE RDR region from a position-ordered JPEG 2000 ──────────────
def rdr_pixel(lbl, lat, lon):
    """(sample, line) of a point in a HiRISE RDR, from its map projection keywords."""
    g = lambda k: float(re.search(rf"\b{k}\s*=\s*(-?[\d.]+)", lbl).group(1))
    kind = re.search(r'MAP_PROJECTION_TYPE\s*=\s*"?([A-Z ]+)', lbl).group(1).strip()
    res, scale, clat, clon = g("MAP_RESOLUTION"), g("MAP_SCALE"), g("CENTER_LATITUDE"), g("CENTER_LONGITUDE")
    R = res * scale * 180 / math.pi       # the radius actually used: polar RDRs use the polar radius (3376.2 km)
                                          # although A_AXIS_RADIUS says 3396.19; checked on the frame corners
    soff, loff = g("SAMPLE_PROJECTION_OFFSET"), g("LINE_PROJECTION_OFFSET")
    if kind == "EQUIRECTANGULAR":
        dlon = (lon - clon + 180) % 360 - 180
        return soff + dlon * res * math.cos(math.radians(clat)), loff - lat * res
    if kind == "POLAR STEREOGRAPHIC":
        la, dl = math.radians(lat), math.radians(lon - clon)
        if clat > 0:
            rho = 2 * R * math.tan(math.pi / 4 - la / 2)
            x, y = rho * math.sin(dl), -rho * math.cos(dl)
        else:
            rho = 2 * R * math.tan(math.pi / 4 + la / 2)
            x, y = rho * math.sin(dl), rho * math.cos(dl)
        return soff + x / scale, loff - y / scale
    raise ValueError(f"projection {kind}")


def j2k_layout(url):
    """Main header, tile header and packet table of a single-tile PCRL codestream with PLT markers."""
    head = b""
    for size in (1 << 17, 1 << 19, 1 << 21):
        head = http(url, f"bytes=0-{size - 1}")
        if b"\xff\x93" in head[head.find(b"\xff\x4f\xff\x51"):]:
            break
    soc = head.find(b"\xff\x4f\xff\x51")
    p, main, siz, cod = soc + 2, [head[soc:soc + 2]], None, None
    while True:                                           # main header marker segments
        m, L = struct.unpack(">HH", head[p:p + 4])
        if m == 0xFF90:
            break
        seg = head[p:p + 2 + L]
        if m == 0xFF51:
            siz = struct.unpack(">HIIIIIIIIH", seg[4:40])
        if m == 0xFF52:
            cod = seg[4:]
        if m != 0xFF55:                                   # TLM lengths would be wrong after the rebuild
            main.append(seg)
        p += 2 + L
    _, X, Y, X0, Y0, XT, YT, XT0, YT0, C = siz
    scod, prog, layers, nl = cod[0], cod[1], struct.unpack(">H", cod[2:4])[0], cod[5]
    if (X0, Y0, XT0, YT0) != (0, 0, 0, 0) or XT < X or YT < Y or C != 1 or prog != 3 or layers != 1 or scod & 6:
        raise ValueError(f"layout not supported (prog {prog}, layers {layers}, comps {C}, tiles {XT}x{YT})")
    pp = [(b & 15, b >> 4) for b in cod[10:10 + nl + 1]] if scod & 1 else [(15, 15)] * (nl + 1)
    isot, psot, tps, tns = struct.unpack(">HIBB", head[p + 4:p + 12])
    q, tile_hdr, lengths = p + 12, [], []
    while True:                                           # tile-part header: keep all but PLT
        m, L = struct.unpack(">HH", head[q:q + 4])
        if m == 0xFF93:
            break
        seg = head[q:q + 2 + L]
        if m == 0xFF58:
            v = 0
            for byte in seg[5:]:
                v = (v << 7) | (byte & 0x7F)
                if not byte & 0x80:
                    lengths.append(v); v = 0
        else:
            tile_hdr.append(seg)
        q += 2 + L
    data0 = q + 2
    # PCRL packet order (ISO 15444-1 B.12.1.4), image and tiles anchored at the origin
    dims = [(-(-X // 2 ** (nl - r)), -(-Y // 2 ** (nl - r))) for r in range(nl + 1)]
    npre = [(-(-w // 2 ** pp[r][0]), -(-h // 2 ** pp[r][1])) for r, (w, h) in enumerate(dims)]
    sx = min(2 ** (pp[r][0] + nl - r) for r in range(nl + 1))
    sy = min(2 ** (pp[r][1] + nl - r) for r in range(nl + 1))
    order = []
    for y in range(0, Y, sy):
        for x in range(0, X, sx):
            for r in range(nl + 1):
                bx, by = pp[r][0] + nl - r, pp[r][1] + nl - r
                if x % 2 ** bx == 0 and y % 2 ** by == 0:
                    order.append((r, x >> bx, y >> by))
    if len(order) != len(lengths) or data0 - p + sum(lengths) != psot:
        raise ValueError(f"packet table mismatch: {len(order)} packets vs {len(lengths)} PLT entries")
    offs = np.concatenate([[0], np.cumsum(lengths)]) + data0
    return dict(main=b"".join(main), tile_hdr=b"".join(tile_hdr), order=order, offs=offs, lengths=lengths,
                X=X, Y=Y, nl=nl, pp=pp)


def jp2_region(url, x0, y0, w, h):
    """Decode the region [x0, x0+w) × [y0, y0+h) of a remote RDR, fetching only the packets it needs."""
    L = j2k_layout(url)
    nl, pp = L["nl"], L["pp"]
    want = set()
    for r in range(nl + 1):
        s = 2 ** (nl - r)
        a0, a1 = x0 // s - MARGIN, -(-(x0 + w) // s) + MARGIN
        b0, b1 = y0 // s - MARGIN, -(-(y0 + h) // s) + MARGIN
        for px in range(max(0, a0) >> pp[r][0], (max(0, a1) - 1 >> pp[r][0]) + 1):
            for py in range(max(0, b0) >> pp[r][1], (max(0, b1) - 1 >> pp[r][1]) + 1):
                want.add((r, px, py))
    idx = [k for k, key in enumerate(L["order"]) if key in want and L["lengths"][k]]
    ranges = []                                           # byte ranges to fetch, merged across small gaps
    for k in idx:
        a, b = int(L["offs"][k]), int(L["offs"][k + 1])
        if ranges and a - ranges[-1][1] < 1 << 20:       # each request costs ~9 s on this server: fewer, larger
            ranges[-1][1] = b
        else:
            ranges.append([a, b])
    with ThreadPoolExecutor(4) as pool:
        got = dict(zip((a for a, _ in ranges), pool.map(lambda ab: http(url, f"bytes={ab[0]}-{ab[1] - 1}", deadline=300), ranges)))
    def packet(k):
        a = int(L["offs"][k])
        for start, blob in got.items():
            if start <= a < start + len(blob):
                return blob[a - start:a - start + L["lengths"][k]]
    keep = set(idx)
    body = b"".join(packet(k) if k in keep else b"\x00" for k in range(len(L["order"])))   # 0x00 = empty packet
    tile = L["tile_hdr"] + b"\xff\x93" + body
    cs = L["main"] + b"\xff\x90\x00\x0a" + struct.pack(">HIBB", 0, 12 + len(tile), 0, 1) + tile + b"\xff\xd9"
    with tempfile.TemporaryDirectory() as tmp:
        j, o = Path(tmp) / "r.j2k", Path(tmp) / "r.pgm"
        j.write_bytes(cs)
        subprocess.run([OPJ, "-i", str(j), "-o", str(o), "-d", f"{x0},{y0},{x0 + w},{y0 + h}", "-quiet"], cwd=tmp,
                       check=True, capture_output=True)
        raw = o.read_bytes()
        if not raw.startswith(b"P5"): raise ValueError(f"opj output {raw[:40]!r}")
    hdr = re.match(rb"P5\s+(?:#[^\n]*\n)*(\d+)\s+(\d+)\s+(\d+)\s", raw)
    if not hdr: raise ValueError(f"pgm header {raw[:60]!r}")
    W, H, maxv = map(int, hdr.groups())
    a = np.frombuffer(raw[hdr.end():], ">u2" if maxv > 255 else "u1").reshape(H, W)
    return a, sum(b - a for a, b in ranges)


def hirise_crop(f, lat, lon):
    lbl = http(f["label_url"]).decode("latin1")
    lines = int(re.search(r"\bLINES\s*=\s*(\d+)", lbl).group(1))
    samples = int(re.search(r"\bLINE_SAMPLES\s*=\s*(\d+)", lbl).group(1))
    s, l = rdr_pixel(lbl, lat, lon)
    if not (0 <= s < samples and 0 <= l < lines):
        raise ValueError(f"object outside the RDR ({s:.0f}, {l:.0f})")
    x0 = int(np.clip(round(s - CROP / 2), 0, max(0, samples - CROP)))
    y0 = int(np.clip(round(l - CROP / 2), 0, max(0, lines - CROP)))
    a, nbytes = jp2_region(f["image_url"], x0, y0, min(CROP, samples), min(CROP, lines))
    a = a.astype(np.float32)
    valid = (a > 2) & (a < 1022)                          # 0 null, 1-2 / 1022-1023 saturation codes
    return stretch(a, valid), x0, y0, (s - x0, l - y0), 1 - valid.mean(), nbytes


# ── selection ──────────────────────────────────────────────────────────────
def load(name):
    return list(csv.DictReader((HERE / name).open()))


def pick_moon(frames):
    jobs = []
    for sid in dict.fromkeys(f["site_id"] for f in frames):
        mine = [f for f in frames if f["site_id"] == sid]
        key = lambda f: (float(f["resolution_m_px"]), float(f["emission_deg"]))
        usable = [f for f in mine if f["usable"] == "True"]
        for lo, hi in INCIDENCE_BINS:
            inbin = sorted((f for f in usable if lo <= float(f["incidence_deg"]) < hi), key=key)
            jobs += inbin[:1]
        before = sorted((f for f in mine if f["after_arrival"] == "False" and float(f["incidence_deg"]) < 90
                         and float(f["resolution_m_px"]) <= 2.0), key=key)
        jobs += before[:1]
    return jobs


def pick_mars(frames, mission_frames, per_site=12, before_per_site=2):
    """Rovers first (every frame: each shows the rover somewhere new), then the fixed sites: up to `per_site`
    sunlit frames spread over sun angle and date, plus a few from before arrival. The HiRISE server gives
    ~57 KB/s, so every 6 MB crop costs ~2 minutes."""
    ok = lambda f: float(f["incidence_deg"]) < 90 and float(f["resolution_m_px"]) <= 2.0
    jobs = [f for f in mission_frames if ok(f) and f["object_lat"]]
    for sid in dict.fromkeys(f["site_id"] for f in frames):
        mine = sorted((f for f in frames if f["site_id"] == sid and ok(f)), key=lambda f: f["utc_start"])
        after = [f for f in mine if f["after_arrival"] == "True"]
        bins = [[f for f in after if lo <= float(f["incidence_deg"]) < hi] for lo, hi in INCIDENCE_BINS]
        chosen, turn = [], 0
        while len(chosen) < min(per_site, len(after)):          # one bin at a time, evenly spaced in time
            b = bins[turn % len(bins)]; turn += 1
            if b:
                chosen.append(b.pop(len(b) // 2 if turn <= len(bins) else (0 if turn % 2 else -1)))
        jobs += sorted(chosen, key=lambda f: f["utc_start"])
        before = [f for f in mine if f["after_arrival"] == "False"]
        jobs += [before[int(i)] for i in np.linspace(0, len(before) - 1, min(before_per_site, len(before)))] if before else []
    return jobs


def build(out, only):
    out.mkdir(parents=True, exist_ok=True)
    meta = out / f"metadata_{only or 'all'}.csv"           # one per process, merged by merge_metadata()
    rows = list(csv.DictReader(meta.open())) if meta.exists() else []
    done = {(r["frame_id"], r["site_id"]) for r in rows}
    sites = {s["site_id"]: s for s in load("sites.csv")}
    frames = load("frames.csv")
    jobs = []
    if only in (None, "moon"):
        jobs += pick_moon([f for f in frames if f["body"] == "Moon"])
    if only in (None, "mars"):
        jobs += pick_mars([f for f in frames if f["body"] == "Mars"], load("mars_mission_frames.csv"))
    _log(f"   {len(jobs)} crops to cut ({sum(j['body'] == 'Moon' for j in jobs)} Moon, "
         f"{sum(j['body'] == 'Mars' for j in jobs)} Mars), {len(done)} already done")
    # every catalogued object that each frame contains, to list all of them in its crops
    contained = {}
    for f in frames:
        contained.setdefault(f["frame_id"], []).append(f)
    centres = {}                                          # frame -> crops already cut from it
    for r in rows:
        centres.setdefault(r["frame_id"], []).append(r)
    samples = []
    for k, f in enumerate(jobs):
        live.progress(k, len(jobs), "crops")
        if (f["frame_id"], f["site_id"]) in done:
            continue
        try:
            if f["body"] == "Moon":
                gx, gy = nac_pixel(f)
                if any(abs(gx - (int(r["crop_x0"]) + CROP / 2)) < SAME_CROP_PX and
                       abs(gy - (int(r["crop_y0"]) + CROP / 2)) < SAME_CROP_PX for r in centres.get(f["frame_id"], [])):
                    continue                              # already listed in a crop of this frame
                img, x0, y0, (ox, oy), bad = nac_crop(f)
                note, nbytes = "footprint corners, about ±60 px", CROP * 5064 * 2
            else:
                lat = float(f.get("object_lat") or sites[f["site_id"]]["lat"])
                lon = float(f.get("object_lon") or sites[f["site_id"]]["lon_east"])
                img, x0, y0, (ox, oy), bad, nbytes = hirise_crop(f, lat, lon)
                unc = f.get("position_uncertainty_m") or ""
                note = "map projection" + (f", rover moved up to {unc} m that sol" if unc not in ("", "0", "0.0") else "")
        except Exception as e:
            _log(f"   failed {f['frame_id']} ({f['site_id']}): {e}"); continue
        if bad > 0.25:
            _log(f"   skipped {f['frame_id']} ({f['site_id']}): {bad:.0%} of the crop is missing or saturated"); continue
        label = "artifact" if f["after_arrival"] == "True" else "before_arrival"
        name = f"{f['body'].lower()}/{f['site_id']}/{f['frame_id']}.png"
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(out / name, optimize=True)
        others = []
        for g in contained.get(f["frame_id"], []):        # other catalogued objects inside this crop
            if g["site_id"] == f["site_id"] or g["body"] != "Moon":
                continue
            gx, gy = nac_pixel(g)
            gx, gy = gx - x0, gy - y0
            if 0 <= gx < CROP and 0 <= gy < CROP:
                others.append(f"{g['site_id']}@{gx:.0f},{gy:.0f}")
        row = dict(file_name=name, site_id=f["site_id"], object=f["object"], mission=f["mission"], body=f["body"],
                   label=label, objects_in_crop=";".join([f"{f['site_id']}@{ox:.0f},{oy:.0f}"] + others),
                   obj_x=round(ox), obj_y=round(oy), frame_id=f["frame_id"], instrument=f["instrument"],
                   utc_start=f["utc_start"], after_arrival=f["after_arrival"], days_since_arrival=f["days_since_arrival"],
                   incidence_deg=f["incidence_deg"], sun_elevation_deg=f["sun_elevation_deg"],
                   emission_deg=f["emission_deg"], phase_deg=f["phase_deg"], resolution_m_px=f["resolution_m_px"],
                   crop_x0=x0, crop_y0=y0, position_note=note, source_url=f["image_url"])
        rows.append(row)
        centres.setdefault(f["frame_id"], []).append(row)
        with meta.open("w", newline="") as fh:
            w = csv.DictWriter(fh, FIELDS, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
        _log(f"   {f['site_id']:<22} {f['frame_id']:<16} i={float(f['incidence_deg']):4.0f}°  "
             f"{float(f['resolution_m_px']):.2f} m/px  {label:<14} {nbytes / 1e6:5.1f} MB")
        samples.append(img[::8, ::8] / 255.0)
        if len(samples) % 6 == 0:
            live.samples(samples[-12:])
        time.sleep(1 if f["body"] == "Moon" else 0.3)
    live.metric(crops=len(rows), moon=sum(r["body"] == "Moon" for r in rows), mars=sum(r["body"] == "Mars" for r in rows))
    _log(f"Done: {len(rows)} crops in {out}")


def merge_metadata(out):
    rows = [r for p in sorted(out.glob("metadata_*.csv")) for r in csv.DictReader(p.open())]
    rows.sort(key=lambda r: (r["body"] != "Moon", r["file_name"]))
    with (out / "metadata.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS); w.writeheader(); w.writerows(rows)
    return rows


def upload(repo, out):
    merge_metadata(out)
    hf = str(Path(sys.executable).with_name("hf"))
    subprocess.run([hf, "upload", repo, str(out), "hardware/images", "--repo-type", "dataset", "--exclude", "metadata_*.csv",
                    "--exclude", "previews/**", "--commit-message", "hardware/images: crops centred on each catalogued object"], check=True)
    print(f"https://huggingface.co/datasets/{repo}/tree/main/hardware/images")


def _arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


if __name__ == "__main__":
    out = Path(_arg("--out", str(HERE.parent / "data/artifact-site-images")))
    if "--upload" in sys.argv:
        upload(_arg("--upload", ""), out); sys.exit()
    class _Quiet:                                         # --no-live: a second process must not fight over live.json
        def __getattr__(self, _):
            return lambda *a, **k: None
    live = _Quiet() if "--no-live" in sys.argv else LiveRun(__file__, str(out), ["Cutting crops"])
    try:
        live.step("Cutting crops")
        build(out, _arg("--only", None))
        merge_metadata(out)
        live.done()
    except BaseException as e:
        live.fail(e); raise

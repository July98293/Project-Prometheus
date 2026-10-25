"""
Xenarch human-trace labels: one 256x256 chip per labelled trace, sorted by class, for classification.

Every box of annotations/boxes_coco.json becomes a chip centred on it, cut from its 2048x2048 crop in
data/artifact-site-images/ (boxes larger than ~190 px are cut wider and resized to 256). The class comes from
the object's type in sites.csv; track boxes (object ids ending in _footpaths / _tracks, see boxtool.py --add)
are their own classes. Negatives: natural-terrain chips from the crops taken before the hardware arrived
(centre + four quadrants of each). The same object seen at the same spot through two crops of one frame
gives one chip.

  lander               crewed and robotic landers (Moon, Mars)
  rover                rovers and the Apollo LRVs
  rover-tracks         wheel tracks (LRV, Curiosity)
  astronaut-footpaths  trails of footprints (single footprints are ~0.3 m, below every camera here)
  surface-experiment   ALSEP, seismometers, laser retroreflectors
  impact-crater        spacecraft / rocket-stage / ascent-stage / retrorocket impact craters
  landing-hardware     heat shields, backshells with parachutes, the Perseverance descent stage
  landing-blast-zone   ground scoured by descent engines (Bradbury Landing, Octavia E. Butler Landing)
  natural-terrain      the same sites before arrival (no hardware)

`split` (train / validation) is fixed per landing area (objects within 10 km share it), so no terrain is in
both: the hash split of xenarch_hf, plus one validation area for every class seen in two or more areas. Writes data/artifact-site-labels/<body>/<class>/<object>_<frame>[_k].png, metadata.csv, one contact
sheet per class in contact-sheets/, and the dashboard page data (dashboard/labels.json + labels/*.jpg).

  .venv/bin/python artifact-site-catalog/build_labels.py [--out data/artifact-site-labels]
  .venv/bin/python xenarch_hf.py labels      publish to human-traces_labels/ of the HF dataset
"""
import csv, hashlib, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from xenarch_hf import split_of

IMG = ROOT / "data/artifact-site-images"
COCO = HERE / "annotations/boxes_coco.json"
DASH = ROOT / "dashboard"
CHIP, CROP = 256, 2048
GROW = 1.33                    # chip side >= 1.33 x the box, so the object never fills the chip edge to edge
SAME_PX = 4                    # boxes of one object this close in frame pixels (two crops of one frame) are one chip
AREA_KM = 10                   # objects closer than this share a location, hence a split
RADIUS_KM = {"Moon": 1737.4, "Mars": 3389.5}
NEGATIVE_AT = [(1024, 1024), (512, 512), (1536, 512), (512, 1536), (1536, 1536)]
NOT_NEGATIVE = {"mars/spirit/ESP_012787_1650.png"}     # labelled before_arrival but already shows Spirit (README)

TYPE_CLASS = {"crewed lander": "lander", "robotic lander": "lander", "rover": "rover", "helicopter": "rover",
              "surface experiment": "surface-experiment", "impact crater": "impact-crater",
              "landing hardware": "landing-hardware", "rover landing site": "landing-blast-zone"}
TRACK_CLASS = {"_footpaths": ("astronaut-footpaths", "astronaut footpaths"), "_tracks": ("rover-tracks", "rover tracks")}
CLASSES = ["lander", "rover", "rover-tracks", "astronaut-footpaths", "surface-experiment", "impact-crater",
           "landing-hardware", "landing-blast-zone", "natural-terrain"]

FIELDS = ["file_name", "split", "label", "body", "instrument", "object_id", "object", "object_type", "mission",
          "location_id", "frame_id", "crop_file", "chip_x0", "chip_y0", "chip_side_px", "chip_scale",
          "box_x_min", "box_y_min", "box_x_max", "box_y_max", "box_confidence", "includes_shadow",
          "resolution_m_px", "incidence_deg", "emission_deg", "phase_deg", "utc_start", "lat", "lon_east",
          "note", "source_url"]


def load(path):
    return list(csv.DictReader(open(path)))


def km(body, a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    c = math.sin(la1) * math.sin(la2) + math.cos(la1) * math.cos(la2) * math.cos(lo1 - lo2)
    return RADIUS_KM[body] * math.acos(max(-1.0, min(1.0, c)))


def locations(sites):
    """site_id -> location id: connected groups of objects closer than AREA_KM, named by their first site."""
    ids = sorted(sites)
    parent = {s: s for s in ids}
    find = lambda s: s if parent[s] == s else find(parent[s])
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            sa, sb = sites[a], sites[b]
            if sa["body"] == sb["body"] and km(sa["body"], (float(sa["lat"]), float(sa["lon_east"])),
                                                (float(sb["lat"]), float(sb["lon_east"]))) < AREA_KM:
                parent[find(b)] = find(a)
    return {s: find(s) for s in ids}


def classify(oid, crop_site, sites):
    """(label, object name, object type, mission) of a box's object id."""
    for suffix, (label, kind) in TRACK_CLASS.items():
        if oid.endswith(suffix):
            m = sites[crop_site]["mission"]
            return label, f"{m} {kind}", kind, m
    s = sites[oid]
    return TYPE_CLASS[s["object_type"]], s["object"], s["object_type"], s["mission"]


def assign_splits(rows):
    """location -> train / validation. The hash split of xenarch_hf (10 %), then every class found in two or more
    locations gets one validation location (the lowest hash) if it has none, and keeps at least one in train."""
    split = {r["location_id"]: split_of(r["location_id"]) for r in rows}
    locs = defaultdict(set)
    for r in rows:
        locs[r["label"]].add(r["location_id"])
    order = lambda l: hashlib.md5(l.encode()).hexdigest()
    for c in sorted(locs, key=lambda c: len(locs[c])):
        ls = sorted(locs[c], key=order)
        if len(ls) < 2:
            continue
        if all(split[l] == "train" for l in ls):
            split[ls[0]] = "validation"
        if all(split[l] == "validation" for l in ls):
            split[ls[-1]] = "train"
    return split


def cut(a, cx, cy, side):
    """Chip of `side` source px centred on (cx, cy), shifted inside the crop, resized to CHIP. -> (chip, x0, y0)."""
    side = min(side, CROP)
    x0 = int(np.clip(round(cx - side / 2), 0, CROP - side)); y0 = int(np.clip(round(cy - side / 2), 0, CROP - side))
    p = Image.fromarray(a[y0:y0 + side, x0:x0 + side])
    return (p if side == CHIP else p.resize((CHIP, CHIP), Image.LANCZOS)), x0, y0


def build(out):
    sites = {r["site_id"]: r for r in load(HERE / "sites.csv")}
    loc = locations(sites)
    meta = {r["file_name"]: r for r in load(IMG / "metadata.csv")}
    coco = json.load(COCO.open())
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    images = {i["id"]: i for i in coco["images"]}
    rows, seen, arrays = [], [], {}

    def crop(f):
        if f not in arrays:
            arrays.clear(); arrays[f] = np.asarray(Image.open(IMG / f).convert("L"))
        return arrays[f]

    def row(f, m, label, oid, obj, otype, mission, site, x0, y0, side, box=None, a=None):
        s = sites[site]
        scale = side / CHIP
        r = dict(split="", label=label, body=m["body"], instrument=m["instrument"], object_id=oid,
                 object=obj, object_type=otype, mission=mission, location_id=loc[site], frame_id=m["frame_id"],
                 crop_file=f, chip_x0=x0, chip_y0=y0, chip_side_px=side, chip_scale=round(scale, 4),
                 resolution_m_px=round(float(m["resolution_m_px"]) * scale, 4), incidence_deg=m["incidence_deg"],
                 emission_deg=m["emission_deg"], phase_deg=m["phase_deg"], utc_start=m["utc_start"],
                 lat=s["lat"], lon_east=s["lon_east"], source_url=m["source_url"])
        if box:
            bx, by, bw, bh = box
            r.update(box_x_min=round((bx - x0) / scale), box_y_min=round((by - y0) / scale),
                     box_x_max=round((bx + bw - 1 - x0) / scale), box_y_max=round((by + bh - 1 - y0) / scale),
                     box_confidence=a["confidence"], includes_shadow="yes" if a["includes_shadow"] else "no", note=a["note"])
        return r

    # positives: every box, one chip per object and frame position
    anns = sorted(coco["annotations"], key=lambda a: (images[a["image_id"]]["file_name"], a["id"]))
    for a in anns:
        f = images[a["image_id"]]["file_name"]; m = meta[f]; oid = cat[a["category_id"]]
        bx, by, bw, bh = a["bbox"]
        fx, fy = int(m["crop_x0"]) + bx, int(m["crop_y0"]) + by          # box corner in frame pixels
        if any(k[0] == m["frame_id"] and k[1] == oid and abs(k[2] - fx) <= SAME_PX and abs(k[3] - fy) <= SAME_PX for k in seen):
            continue
        seen.append((m["frame_id"], oid, fx, fy))
        label, obj, otype, mission = classify(oid, m["site_id"], sites)
        side = max(CHIP, math.ceil(GROW * max(bw, bh)))
        chip, x0, y0 = cut(crop(f), bx + bw / 2, by + bh / 2, side)
        site = oid if oid in sites else m["site_id"]
        r = row(f, m, label, oid, obj, otype, mission, site, x0, y0, min(side, CROP), a["bbox"], a)
        rows.append((r, chip))

    # negatives: natural terrain before arrival, five chips per crop, one per frame position
    taken = []
    for f, m in sorted(meta.items()):
        if m["label"] != "before_arrival" or f in NOT_NEGATIVE:
            continue
        for k, (cx, cy) in enumerate(NEGATIVE_AT):
            fx, fy = int(m["crop_x0"]) + cx, int(m["crop_y0"]) + cy
            if any(t[0] == m["frame_id"] and abs(t[1] - fx) < CHIP and abs(t[2] - fy) < CHIP for t in taken):
                continue
            taken.append((m["frame_id"], fx, fy))
            chip, x0, y0 = cut(crop(f), cx, cy, CHIP)
            s = sites[m["site_id"]]
            r = row(f, m, "natural-terrain", m["site_id"], f"{s['object']}: site before arrival",
                    "natural terrain (before arrival)", s["mission"], m["site_id"], x0, y0, CHIP)
            r["note"] = f"crop taken {abs(int(float(m['days_since_arrival'] or 0)))} days before arrival; chip {k + 1} of 5"
            rows.append((r, chip))

    splits = assign_splits([r for r, _ in rows])
    for r, _ in rows:
        r["split"] = splits[r["location_id"]]

    # file names: <object>_<frame>[_k].png under <body>/<class>/
    count = Counter((r["body"], r["label"], r["object_id"], r["frame_id"]) for r, _ in rows)
    k = Counter()
    for r, _ in rows:
        key = (r["body"], r["label"], r["object_id"], r["frame_id"])
        k[key] += 1
        stem = f"{r['object_id']}_{r['frame_id']}" + (f"_{k[key]}" if count[key] > 1 else "")
        r["file_name"] = f"{r['body'].lower()}/{r['label']}/{stem}.png"

    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*/*/*.png"):                     # rebuilt from scratch: boxes can move or disappear
        old.unlink()
    for r, chip in rows:
        (out / r["file_name"]).parent.mkdir(parents=True, exist_ok=True)
        chip.save(out / r["file_name"], optimize=True)
    rows.sort(key=lambda t: (CLASSES.index(t[0]["label"]), t[0]["file_name"]))
    with (out / "metadata.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS, lineterminator="\n"); w.writeheader(); w.writerows(r for r, _ in rows)
    sheets(out, rows)
    dashboard(rows)
    n = Counter((r["label"], r["split"]) for r, _ in rows)
    print(f"{len(rows)} chips in {out}")
    for c in CLASSES:
        print(f"  {c:<20} {n[(c, 'train')] + n[(c, 'validation')]:4d}  (train {n[(c, 'train')]}, validation {n[(c, 'validation')]})")


# ── contact sheets and dashboard data ─────────────────────────────────────────
NAVY, INK, MUTED, ORANGE = (12, 22, 40), (230, 233, 240), (140, 150, 170), (240, 136, 78)


def _font(size):
    for f in ("/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _tile(chip, r, t=160):
    im = chip.resize((t, t), Image.LANCZOS).convert("RGB")
    if r.get("box_x_min") != "" and r.get("box_x_min") is not None:
        s = t / CHIP
        ImageDraw.Draw(im).rectangle([r["box_x_min"] * s, r["box_y_min"] * s, r["box_x_max"] * s, r["box_y_max"] * s],
                                     outline=ORANGE, width=1)
    return im


def sheets(out, rows, t=160, cols=8, cap=64):
    """contact-sheets/<class>.jpg (up to `cap` chips, box outlined) and overview.jpg (first 8 chips per class)."""
    d = out / "contact-sheets"; d.mkdir(exist_ok=True)
    f13, f11 = _font(13), _font(11)
    by = defaultdict(list)
    for r, chip in rows:
        by[r["label"]].append((r, chip))
    for c in CLASSES:
        items = by[c][:cap]
        if not items:
            continue
        nr = math.ceil(len(items) / cols)
        im = Image.new("RGB", (cols * (t + 8) + 8, 40 + nr * (t + 26)), NAVY); dr = ImageDraw.Draw(im)
        dr.text((8, 12), f"{c.upper()}   {len(by[c])} chips" + (f", first {cap} shown" if len(by[c]) > cap else ""), fill=ORANGE, font=f13)
        for i, (r, chip) in enumerate(items):
            x, y = 8 + (i % cols) * (t + 8), 40 + (i // cols) * (t + 26)
            im.paste(_tile(chip, r, t), (x, y))
            dr.text((x, y + t + 4), f"{r['object_id'][:22]}", fill=MUTED, font=f11)
        im.save(d / f"{c}.jpg", quality=88)
    n = 8
    im = Image.new("RGB", (n * (t + 8) + 8 + 190, 16 + len(CLASSES) * (t + 12)), NAVY); dr = ImageDraw.Draw(im)
    for j, c in enumerate(CLASSES):
        y = 16 + j * (t + 12)
        dr.text((8, y + t // 2 - 16), c.upper(), fill=ORANGE, font=f13)
        dr.text((8, y + t // 2 + 4), f"{len(by[c])} chips", fill=MUTED, font=f11)
        step = max(1, len(by[c]) // n)
        for i, (r, chip) in enumerate(by[c][::step][:n]):
            im.paste(_tile(chip, r, t), (190 + i * (t + 8), y))
    im.save(d / "overview.jpg", quality=88)


def dashboard(rows, t=128):
    """dashboard/labels.json (every row) and dashboard/labels/<file>.jpg thumbnails with the box outlined."""
    d = DASH / "labels"; d.mkdir(exist_ok=True)
    keep = set()
    for r, chip in rows:
        name = r["file_name"][:-4].replace("/", "_") + ".jpg"; keep.add(name)
        _tile(chip, r, t).save(d / name, quality=82)
        r["thumb"] = f"labels/{name}"
    for old in d.glob("*.jpg"):
        if old.name not in keep:
            old.unlink()
    cols = ["thumb"] + FIELDS
    json.dump({"generated": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), "classes": CLASSES,
               "rows": [{c: r.get(c, "") for c in cols if c not in ("source_url",)} for r, _ in rows]},
              (DASH / "labels.json").open("w"), separators=(",", ":"))


if __name__ == "__main__":
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else ROOT / "data/artifact-site-labels"
    build(out)

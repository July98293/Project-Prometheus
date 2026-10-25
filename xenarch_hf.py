"""
Layout of the project's single HF dataset, jules1235813/xenarch, and the helpers that publish to it.

Every folder name says what it holds and which camera took it, so a folder copied out of the
dataset still explains itself. Local folders under data/ keep their short names; this module maps
them to the HF names and rewrites the path columns of the CSV / COCO files on the way up.

  natural-terrain_training/                    natural terrain only; metadata.csv `split`: train / validation
    moon_lroc-nac/moon_<landform>_lroc-nac/     data/lroc-nac-set/<landform>/
    mars_hirise/mars_<landform>_hirise/         data/hirise-set/<landform>/
    moon_chandrayaan2-ohrc/moon_<landform>_chandrayaan2-ohrc/   data/ohrc-set/<landform>/
    legacy_mixed-cameras/<body>_<camera>/       data/xenarch-imagery/<camera>_png/   (the old xenarch-imagery set)
  human-hardware_ground-truth/                 known human-made objects (test)
    catalog_moon-mars_lroc-nac_hirise.csv       artifact-site-catalog/catalog.csv: the main table, object x frame
    boxes-coco_…json                            artifact-site-catalog/annotations/boxes_coco.json (boxtool.py)
    crops_lroc-nac_hirise/<body>_<camera>/<site>_<camera>/<frame>_<site>.png
                                                data/artifact-site-images/<body>/<site>/<frame>.png
    derived/                                    sites, frames, rover-frames: rebuilt by build_catalog.py
  human-traces_labels/                         every labelled human trace as a 256x256 chip, one folder per class
    labels_moon-mars_lroc-nac_hirise.csv        data/artifact-site-labels/metadata.csv: the label table (build_labels.py)
    chips_lroc-nac_hirise/<body>_<camera>/<body>_<class>_<camera>/<object>_<frame>.png
                                                data/artifact-site-labels/<body>/<class>/<object>_<frame>.png
    contact-sheets_lroc-nac_hirise/<class>_lroc-nac_hirise.jpg   one preview sheet per class + overview
  raw/legacy_source-files/                     what the legacy set was made from: PDS originals (.IMG, JPEG),
                                               labels (JSON / XML), the old 'squares' tiles, manifest.csv

  .venv/bin/python xenarch_hf.py terrain moon|mars|ohrc|legacy [local dir]   publish a terrain set and its metadata.csv
  .venv/bin/python xenarch_hf.py legacy-meta                    build data/xenarch-imagery/metadata.csv from raw/ labels
  .venv/bin/python xenarch_hf.py crops [local dir]              publish the hardware crops
  .venv/bin/python xenarch_hf.py tables                         publish the catalog, boxes and derived tables
  .venv/bin/python xenarch_hf.py labels [local dir]             publish the human-trace label chips, table, contact sheets
  .venv/bin/python xenarch_hf.py readme                         publish docs/xenarch-dataset-README.md
  .venv/bin/python xenarch_hf.py migrate [--dry]                move files still under older names (server side)

Column names are the same in every table: file_name, split, body, instrument, landform, frame_id, lat, lon_east,
incidence_deg, emission_deg, phase_deg, resolution_m_px, utc_start, solar_longitude_ls.
"""
import csv, hashlib, io, json, re, sys
from pathlib import Path

from huggingface_hub import (CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi, hf_hub_download,
                             snapshot_download)

HERE = Path(__file__).resolve().parent
REPO = "jules1235813/xenarch"
TERRAIN = "natural-terrain_training"
HARDWARE = "human-hardware_ground-truth"
CROPS = f"{HARDWARE}/crops_lroc-nac_hirise"
DERIVED = f"{HARDWARE}/derived"
LABELS = "human-traces_labels"
CHIPS = f"{LABELS}/chips_lroc-nac_hirise"
SHEETS = f"{LABELS}/contact-sheets_lroc-nac_hirise"
LEGACY = f"{TERRAIN}/legacy_mixed-cameras"
RAW = "raw/legacy_source-files"
CAMERA = {"moon": "lroc-nac", "mars": "hirise"}                    # hardware crops: body -> camera
# terrain sets: set key -> (body, camera); a body can have several cameras
TERRAIN_SETS = {"moon": ("moon", "lroc-nac"), "mars": ("mars", "hirise"), "ohrc": ("moon", "chandrayaan2-ohrc")}
LOCAL_TERRAIN = {"moon": HERE / "data/lroc-nac-set", "mars": HERE / "data/hirise-set", "ohrc": HERE / "data/ohrc-set",
                 "legacy": HERE / "data/xenarch-imagery"}
LOCAL_CROPS = HERE / "data/artifact-site-images"
LOCAL_LABELS = HERE / "data/artifact-site-labels"
CATALOG = HERE / "artifact-site-catalog"

# local file -> path in the repo
TABLES = {
    CATALOG / "catalog.csv": f"{HARDWARE}/catalog_moon-mars_lroc-nac_hirise.csv",
    CATALOG / "annotations/boxes_coco.json": f"{HARDWARE}/boxes-coco_lroc-nac_hirise.json",
    CATALOG / "sites.csv": f"{DERIVED}/sites_moon-mars.csv",
    CATALOG / "frames.csv": f"{DERIVED}/frames_lroc-nac_hirise.csv",
    CATALOG / "mars_mission_frames.csv": f"{DERIVED}/rover-frames_mars_hirise.csv",
}
# legacy xenarch-imagery top folder (without _png / _meta) -> body_camera
LEGACY_CAMERAS = {"clementine": "moon_clementine", "lroc": "moon_lroc-nac-wac", "tmc2": "moon_chandrayaan2-tmc2",
                  "ctx": "mars_ctx", "hirise": "mars_hirise", "moc": "mars_moc"}
LEGACY_INSTRUMENT = {"clementine": "Clementine UVVIS", "tmc2": "Chandrayaan-2 TMC-2", "ctx": "MRO CTX"}
LEGACY_FIELDS = ["file_name", "split", "body", "instrument", "frame_id", "channel", "lat", "lon_east", "incidence_deg",
                 "emission_deg", "phase_deg", "resolution_m_px", "utc_start", "source_url", "label_url"]

VAL_PERCENT = 10   # like Mk18's val_frac


def split_of(source):
    """'train' or 'validation' for every tile of one source image / observation. Fixed by a hash of its id, so it
    is the same on every machine and does not move when tiles are added."""
    return "validation" if int(hashlib.md5(source.encode()).hexdigest(), 16) % 100 < VAL_PERCENT else "train"


def landform_dir(key, landform):
    body, camera = TERRAIN_SETS[key]
    return f"{body}_{landform}_{camera}"


def terrain_rel(body, rel):
    """'fresh_craters/x.png' (relative to the local set) -> path relative to the body's folder in the repo."""
    if body == "legacy":
        top, name = rel.split("/", 1)
        base = top.removesuffix("_png")
        return f"{LEGACY_CAMERAS[base]}/{name}" if top.endswith("_png") and base in LEGACY_CAMERAS else rel
    landform, name = rel.split("/", 1)
    return f"{landform_dir(body, landform)}/{name}"


def terrain_folder(key):
    if key == "legacy":
        return LEGACY
    body, camera = TERRAIN_SETS[key]
    return f"{TERRAIN}/{body}_{camera}"


def crop_rel(rel):
    """'moon/apollo11_lm/M1.png' -> 'moon_lroc-nac/apollo11_lm_lroc-nac/M1_apollo11_lm.png' (relative to CROPS).
    The site is in the file name too: one frame is often cropped around two sites."""
    if rel.split("/", 1)[0] not in CAMERA:
        return rel  # already renamed
    body, site, name = rel.split("/", 2)
    stem, ext = name.rsplit(".", 1)
    return f"{body}_{CAMERA[body]}/{site}_{CAMERA[body]}/{stem}_{site}.{ext}"


def chip_rel(rel):
    """'moon/lander/apollo11_lm_M1.png' -> 'moon_lroc-nac/moon_lander_lroc-nac/apollo11_lm_M1.png' (relative to CHIPS)."""
    if rel.split("/", 1)[0] not in CAMERA:
        return rel  # already renamed
    body, label, name = rel.split("/", 2)
    return f"{body}_{CAMERA[body]}/{body}_{label}_{CAMERA[body]}/{name}"


def _rewrite_csv(data, column=None, fn=None, drop=()):
    rows = list(csv.DictReader(io.StringIO(data.decode())))
    if not rows or (column and column not in rows[0]):
        return data
    for r in rows:
        if column:
            r[column] = fn(r[column]) if r[column] else r[column]
        for c in drop:
            r.pop(c, None)
    buf = io.StringIO()
    w = csv.DictWriter(buf, list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
    return buf.getvalue().encode()


def _rewrite_coco(data):
    d = json.loads(data)
    for im in d.get("images", []):
        im["file_name"] = crop_rel(im["file_name"])
    return json.dumps(d, indent=1).encode()


def _mars_renames():
    """Old Mars tile name (<obs>_<channel>.png) -> current one (with the sub-landform), from the local metadata."""
    meta = LOCAL_TERRAIN["mars"] / "metadata.csv"
    if not meta.exists():
        return {}
    return {f"{r['frame_id']}_{r['channel']}.png": Path(r["file_name"]).name for r in csv.DictReader(meta.open())}


def new_path(old, mars=None):
    """Repo path under an older layout -> (current path, content rewriter or None); (old, None) if already current."""
    p = old.split("/")
    if old.startswith("terrain/legacy/") and len(p) >= 3:
        if len(p) == 3:  # manifest.csv: drop the paths of the machine that downloaded it
            return f"{RAW}/{p[2]}", lambda b: _rewrite_csv(b, drop=("local_path",))
        top, name = p[2], "/".join(p[3:])
        if top == "squares":
            return f"{RAW}/squares_mixed-cameras/{name}", None
        base = re.sub(r"_(png|meta)$", "", top)
        cam = LEGACY_CAMERAS.get(base, base)
        if top.endswith("_png"):
            return f"{LEGACY}/{cam}/{name}", None
        return f"{RAW}/{cam}_{'pds-labels' if top.endswith('_meta') else 'originals'}/{name}", None
    if old.startswith(f"{TERRAIN}/mars_hirise/") and len(p) == 4 and p[3] in (mars or {}):
        return f"{'/'.join(p[:3])}/{mars[p[3]]}", None
    if old.startswith(f"{CROPS}/") and len(p) == 5 and old.endswith(".png"):
        site = p[3].rsplit("_", 1)[0]
        if not p[4].endswith(f"_{site}.png"):
            return f"{'/'.join(p[:4])}/{p[4][:-4]}_{site}.png", None
    return old, None


def _commit(ops, message, repo, batch=800):
    api = HfApi()
    for i in range(0, len(ops), batch):
        part = ops[i:i + batch]
        api.create_commit(repo, part, commit_message=message if len(ops) <= batch else f"{message} ({i // batch + 1})",
                          repo_type="dataset")
        print(f"  committed {i + len(part)}/{len(ops)} operations")


def migrate(repo=REPO, dry=False):
    """Rename every file still under an older path, server side for LFS files, re-uploaded for small ones."""
    api = HfApi()
    mars = _mars_renames()
    entries = [e for e in api.list_repo_tree(repo, repo_type="dataset", recursive=True) if hasattr(e, "lfs")]
    moves = []
    for e in entries:
        new, fix = new_path(e.path, mars)
        if new != e.path:
            moves.append((e, new, fix))
    print(f"{len(moves)} files to move")
    for e, new, _ in moves[:5] + moves[-5:]:
        print(f"  {e.path}  ->  {new}")
    if dry or not moves:
        return moves
    adds = []
    for e, new, fix in moves:
        if e.lfs is not None and fix is None:
            adds.append(CommitOperationCopy(e.path, new))
        else:  # regular files cannot be copied: download, rewrite if needed, re-add
            data = Path(hf_hub_download(repo, e.path, repo_type="dataset")).read_bytes()
            adds.append(CommitOperationAdd(new, fix(data) if fix else data))
    # copies and deletes go in the same commit so no batch leaves a file in both places or neither
    ops, batch = [], 400
    for i in range(0, len(moves), batch):
        chunk = list(zip(adds[i:i + batch], moves[i:i + batch]))
        ops.append([a for a, _ in chunk] + [CommitOperationDelete(m[0].path) for _, m in chunk])
    for n, part in enumerate(ops, 1):
        api.create_commit(repo, part, repo_type="dataset",
                          commit_message=f"Move files to the current layout ({n}/{len(ops)})")
        print(f"  commit {n}/{len(ops)}: {len(part) // 2} files")
    return moves


def legacy_metadata(local=None, repo=REPO):
    """metadata.csv for the legacy set, from the PDS labels kept in raw/ (the old set had one JSON per image).
    Values the archive marks missing (-1e32) stay empty. HiRISE position, sun angles and time come from the HiRISE
    RDR index instead: the old labels have them scrambled. HiRISE resolution is nominal (0.25 m x binning)."""
    from xenarch_hirise_dataset import read_index
    local = Path(local or LOCAL_TERRAIN["legacy"])
    labels = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{RAW}/*_pds-labels/*"])) / RAW
    index = {r["obs"]: r for r in read_index(HERE / "data/cache/hirise")}
    num = lambda v: v if isinstance(v, (int, float)) and abs(v) < 1e30 else ""
    rows = []
    for f in sorted(local.glob("*_png/*.png")):
        top = f.parent.name.removesuffix("_png")
        stem = re.sub(r"_browse$", "", f.stem)
        js = labels / f"{LEGACY_CAMERAS[top]}_pds-labels" / f"{stem}.json"
        d = json.loads(js.read_text()) if js.exists() else {}
        frame = d.get("PRODUCT_ID") or stem
        r = dict(file_name=str(f.relative_to(local)), body=LEGACY_CAMERAS[top].split("_")[0].capitalize(),
                 instrument=LEGACY_INSTRUMENT.get(top, ""), frame_id=frame, channel="",
                 lat=num(d.get("CENTER_LATITUDE")), lon_east=num(d.get("CENTER_LONGITUDE")),
                 incidence_deg=num(d.get("INCIDENCE_ANGLE")), emission_deg=num(d.get("EMISSION_ANGLE")),
                 phase_deg=num(d.get("PHASE_ANGLE")), resolution_m_px=num(d.get("RESOLUTION", d.get("SCALED_PIXEL_WIDTH"))),
                 utc_start=d.get("START_TIME") or "", source_url=d.get("ATLAS_DATA_URL", ""),
                 label_url=d.get("ATLAS_LABEL_URL", ""))
        if top == "lroc":
            r["instrument"] = "LROC NAC" if re.search(r"[LR][CE]$", frame) else "LROC WAC"
        if top == "hirise":
            m = re.match(r"(.+?)_((RED|IR|BG)\d+_\d)$", stem)
            obs = d.get("OBSERVATION_ID") or (m.group(1) if m else stem)
            ix = index.get(obs, {})
            r.update(frame_id=obs, channel=m.group(2) if m else "", instrument=f"HiRISE {m.group(3) if m else ''}".strip(),
                     lat=round(ix["lat"], 4) if ix else "", lon_east=round(ix["lon"], 4) if ix else "",
                     incidence_deg=ix.get("inc", ""), emission_deg=ix.get("emi", ""), phase_deg="",
                     utc_start=ix.get("time", r["utc_start"]),
                     resolution_m_px=0.25 * d["BINNING"] if num(d.get("BINNING")) else "")
        r["split"] = split_of(r["frame_id"])
        rows.append(r)
    with (local / "metadata.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, LEGACY_FIELDS); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows -> {local / 'metadata.csv'}")


def publish_terrain(body, local=None, repo=REPO):
    local = Path(local or LOCAL_TERRAIN[body])
    folder, rel = terrain_folder(body), lambda f: terrain_rel(body, f)
    pattern = "*_png/*.png" if body == "legacy" else "*/*.png"
    ops = [CommitOperationAdd(f"{folder}/{rel(str(f.relative_to(local)))}", str(f)) for f in sorted(local.glob(pattern))]
    meta = local / "metadata.csv"
    if meta.exists():
        ops.append(CommitOperationAdd(f"{folder}/metadata.csv", _rewrite_csv(meta.read_bytes(), "file_name", rel)))
    HfApi().create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    _commit(ops, f"{folder}: natural terrain tiles", repo)
    print(f"https://huggingface.co/datasets/{repo}/tree/main/{folder}")


def publish_crops(local=None, repo=REPO, metadata=None):
    local = Path(local or LOCAL_CROPS)
    ops = [CommitOperationAdd(f"{CROPS}/{crop_rel(str(f.relative_to(local)))}", str(f))
           for f in sorted(local.glob("*/*/*.png")) if f.parts[-3] in CAMERA]
    meta = Path(metadata or local / "metadata.csv")
    ops.append(CommitOperationAdd(f"{CROPS}/metadata.csv", _rewrite_csv(meta.read_bytes(), "file_name", crop_rel)))
    _commit(ops, f"{CROPS}: crops centred on each catalogued object", repo)
    print(f"https://huggingface.co/datasets/{repo}/tree/main/{CROPS}")


def publish_tables(repo=REPO):
    ops = []
    for f, path in TABLES.items():
        data = f.read_bytes()
        if f.name == "catalog.csv":
            data = _rewrite_csv(data, "crop_file", crop_rel)
        elif f.suffix == ".json":
            data = _rewrite_coco(data)
        ops.append(CommitOperationAdd(path, data))
    # tables that moved to derived/ leave no copy behind; the box CSVs were folded into the COCO file
    existing = set(HfApi().list_repo_files(repo, repo_type="dataset"))
    gone = [f"{HARDWARE}/{Path(p).name}" for p in TABLES.values() if p.startswith(DERIVED)]
    gone += [f"{HARDWARE}/boxes_lroc-nac_hirise.csv", f"{HARDWARE}/unboxed_lroc-nac_hirise.csv"]
    ops += [CommitOperationDelete(p) for p in gone if p in existing]
    _commit(ops, f"{HARDWARE}: site catalog, boxes, derived tables", repo)


def publish_labels(local=None, repo=REPO):
    """Chips under CHIPS (with their metadata.csv for the HF viewer), the label table at the top of LABELS, the
    contact sheets; files under LABELS that are no longer built are deleted."""
    local = Path(local or LOCAL_LABELS)
    ops = [CommitOperationAdd(f"{CHIPS}/{chip_rel(str(f.relative_to(local)))}", str(f))
           for f in sorted(local.glob("*/*/*.png")) if f.parts[-3] in CAMERA]
    meta = _rewrite_csv((local / "metadata.csv").read_bytes(), "file_name", chip_rel)
    meta = _rewrite_csv(meta, "crop_file", lambda p: f"{CROPS}/{crop_rel(p)}")
    ops.append(CommitOperationAdd(f"{CHIPS}/metadata.csv", meta))
    ops.append(CommitOperationAdd(f"{LABELS}/labels_moon-mars_lroc-nac_hirise.csv",
                                  _rewrite_csv(meta, "file_name", lambda p: f"{CHIPS.split('/', 1)[1]}/{p}")))
    ops += [CommitOperationAdd(f"{SHEETS}/{f.stem}_lroc-nac_hirise.jpg", str(f)) for f in sorted((local / "contact-sheets").glob("*.jpg"))]
    built = {o.path_in_repo for o in ops}
    existing = set(HfApi().list_repo_files(repo, repo_type="dataset"))
    ops += [CommitOperationDelete(p) for p in sorted(existing) if p.startswith(f"{LABELS}/") and p not in built]
    _commit(ops, f"{LABELS}: human-trace chips by class, label table, contact sheets", repo)
    print(f"https://huggingface.co/datasets/{repo}/tree/main/{LABELS}")


def publish_readme(repo=REPO):
    _commit([CommitOperationAdd("README.md", str(HERE / "docs/xenarch-dataset-README.md"))], "README", repo)


if __name__ == "__main__":
    cmd, rest = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    if cmd == "terrain":
        publish_terrain(rest[0], rest[1] if len(rest) > 1 else None)
    elif cmd == "legacy-meta":
        legacy_metadata(rest[0] if rest else None)
    elif cmd == "crops":
        publish_crops(rest[0] if rest else None)
    elif cmd == "tables":
        publish_tables()
    elif cmd == "labels":
        publish_labels(rest[0] if rest else None)
    elif cmd == "readme":
        publish_readme()
    elif cmd == "migrate":
        migrate(dry="--dry" in rest)
    else:
        print(__doc__)

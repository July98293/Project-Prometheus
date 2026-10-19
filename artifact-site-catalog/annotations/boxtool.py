#!/usr/bin/env python
"""Bounding-box annotation helper for the artifact-site crops.

  boxtool.py hunt FILE [--radius 500] [--top 12] [--at X,Y]   candidate sheet around the nominal position
  boxtool.py grid FILE X Y [--w 80]                            pixel grid zoom for manual box placement
  boxtool.py box FILE OBJECT X Y [--fit] [--w W --h H] [--conf C] [--note N]
                                                             write/replace one box (centre at X,Y)
  boxtool.py rect FILE OBJECT X0 Y0 X1 Y1 [--conf C] [--note N]  write/replace a box by corners
  boxtool.py drop FILE [OBJECT]                                remove box(es)
  boxtool.py unbox FILE [REASON]                               why this positive crop has no box (no REASON: clear it)
  boxtool.py apply PROPOSALS.json [--dry]                      merge a reviewer's proposals: {"boxes": [{file, object_id,
                                                             x_min, y_min, x_max, y_max, confidence, note, includes_shadow}],
                                                             "unboxed": {file: reason}, "drop": [[file, object_id]]}
  boxtool.py predict FILE                                      predicted pixels of the other catalogued objects,
                                                             from the first measured box (relative offsets)
  boxtool.py review [--site S]                                 review sheets (one PNG per frame) + dashboard page data

boxes_coco.json is the only annotation file: every crop is an image record (positives without a box carry
`unboxed_reason`), every box an annotation with `includes_shadow`, `confidence`, `annotator`, `note`.
"""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMG = os.path.join(ROOT, "data", "artifact-site-images")
HERE = os.path.dirname(os.path.abspath(__file__))
COCO = os.path.join(HERE, "boxes_coco.json")
REVIEW = os.path.join(HERE, "review")
SCRATCH = os.environ.get("BOX_SCRATCH", os.path.join(HERE, "scratch"))
ORANGE, BLUE, MAGENTA, RED = (240, 136, 78), (57, 135, 229), (213, 81, 129), (255, 40, 40)
CONF_COL = {"high": ORANGE, "medium": BLUE, "low": MAGENTA}


def meta():
    return {r["file_name"]: r for r in csv.DictReader(open(os.path.join(IMG, "metadata.csv")))}


def sites():
    return {r["site_id"]: r for r in csv.DictReader(open(os.path.join(HERE, "..", "sites.csv")))}


def nominal(m):
    """{object_id: (x, y)} nominal pixels of every catalogued object in this crop."""
    out = {}
    for tok in m["objects_in_crop"].split(";"):
        oid, xy = tok.split("@")
        x, y = xy.split(",")
        out[oid] = (int(x), int(y))
    return out


def load_coco():
    if not os.path.exists(COCO):
        return {"images": [], "annotations": [], "categories": []}
    return json.load(open(COCO))


def load_boxes():
    """COCO annotations as flat rows: file_name, site_id, object_id, label, x_min..y_max (inclusive), width_px, height_px, ..."""
    d = load_coco()
    ims = {i["id"]: i for i in d["images"]}
    cats = {c["id"]: c for c in d["categories"]}
    rows = []
    for a in d["annotations"]:
        im, c = ims[a["image_id"]], cats[a["category_id"]]
        x, y, w, h = a["bbox"]
        rows.append(dict(file_name=im["file_name"], site_id=im["site_id"], object_id=c["name"], label=c["label"],
                         x_min=x, y_min=y, x_max=x + w - 1, y_max=y + h - 1, width_px=w, height_px=h,
                         includes_shadow="yes" if a["includes_shadow"] else "no", confidence=a["confidence"],
                         annotator=a.get("annotator", ""), note=a["note"]))
    return rows


def load_unboxed():
    """{file_name: reason} for positive crops where the hardware could not be identified."""
    return {i["file_name"]: i["unboxed_reason"] for i in load_coco()["images"] if i.get("unboxed_reason")}


def save_boxes(rows, unboxed=None):
    """Write boxes_coco.json: every crop of metadata.csv as an image, `rows` as annotations. Category ids stay stable."""
    ms = meta()
    old = load_coco()
    if unboxed is None:
        unboxed = load_unboxed()
    rows.sort(key=lambda r: (r["file_name"], r["object_id"]))
    coco = {"info": {"description": "Xenarch artifact-site bounding boxes: human-made hardware in LROC NAC and HiRISE crops",
                     "version": "0.2", "date_created": __import__("datetime").date.today().isoformat()},
            "licenses": [{"id": 1, "name": "CC0-1.0", "url": "https://creativecommons.org/publicdomain/zero/1.0/"}],
            "images": [], "annotations": [], "categories": []}
    ids = {}
    for i, (file, m) in enumerate(ms.items(), 1):
        ids[file] = i
        im = {"id": i, "file_name": file, "width": 2048, "height": 2048, "license": 1,
              "site_id": m["site_id"], "body": m["body"], "frame_id": m["frame_id"], "instrument": m["instrument"],
              "resolution_m_px": float(m["resolution_m_px"]), "incidence_deg": float(m["incidence_deg"] or 0),
              "utc_start": m["utc_start"], "label": m["label"], "nominal_xy": [int(m["obj_x"]), int(m["obj_y"])]}
        if unboxed.get(file):
            im["unboxed_reason"] = unboxed[file]
        coco["images"].append(im)
    cats = {c["name"]: c for c in old["categories"]}
    used = {}
    for k, r in enumerate(rows, 1):
        oid = r["object_id"]
        if oid not in cats:
            cats[oid] = {"id": max([c["id"] for c in cats.values()] + [0]) + 1, "name": oid, "supercategory": "artifact", "label": r["label"]}
        used[oid] = cats[oid]
        x, y = int(r["x_min"]), int(r["y_min"])
        w, h = int(r["x_max"]) - x + 1, int(r["y_max"]) - y + 1
        coco["annotations"].append({"id": k, "image_id": ids[r["file_name"]], "category_id": cats[oid]["id"], "bbox": [x, y, w, h],
                                    "area": w * h, "iscrowd": 0, "includes_shadow": r["includes_shadow"] == "yes",
                                    "confidence": r["confidence"], "annotator": r.get("annotator", ""), "note": r["note"]})
    coco["categories"] = sorted(used.values(), key=lambda c: c["id"])
    json.dump(coco, open(COCO, "w"), indent=1)


def img(file):
    return Image.open(os.path.join(IMG, file)).convert("L")


def zmap(a, win=31):
    I = torch.tensor(a)[None, None]
    mu = F.avg_pool2d(I, win, 1, win // 2, count_include_pad=False)
    sd = (F.avg_pool2d(I * I, win, 1, win // 2, count_include_pad=False) - mu * mu).clamp_min(1e-6).sqrt()
    return ((I - mu) / sd)[0, 0].numpy()


def candidates(a, cx, cy, radius, top, small=False):
    """Bright-next-to-dark score, non-max suppressed, ranked inside `radius` of (cx, cy)."""
    k, w = (3, 9) if small else (7, 15)               # small: 2-5 m landers; default: 10 m LM / 30 m craters
    b = ndimage.uniform_filter(a, k)
    s = ndimage.maximum_filter(b, w) - ndimage.minimum_filter(b, w)   # bright area next to dark area
    yy, xx = np.mgrid[:a.shape[0], :a.shape[1]]
    s[(xx - cx) ** 2 + (yy - cy) ** 2 > radius ** 2] = 0
    peak = (s == ndimage.maximum_filter(s, 41)) & (s > 0)
    ys, xs = np.nonzero(peak)
    order = np.argsort(-s[ys, xs])
    out = []
    for i in order:                                   # greedy NMS: plateaus of equal score give many peaks
        if all((xs[i] - x) ** 2 + (ys[i] - y) ** 2 > 30 ** 2 for x, y, _ in out):
            out.append((int(xs[i]), int(ys[i]), float(s[ys[i], xs[i]])))
        if len(out) >= top:
            break
    return out


def font(size=14):
    for p in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def cmd_hunt(args):
    m = meta()[args.file]
    a = np.asarray(img(args.file), np.float32)
    cx, cy = map(int, args.at.split(",")) if args.at else (int(m["obj_x"]), int(m["obj_y"]))
    radius = args.radius if args.radius else (500 if m["body"] == "Moon" else 200)
    cands = candidates(a, cx, cy, radius, args.top, args.small)
    # overview: region around nominal, downsampled to fit 1000 px
    R = radius + 60
    x0, y0 = max(0, cx - R), max(0, cy - R)
    x1, y1 = min(a.shape[1], cx + R), min(a.shape[0], cy + R)
    ov = Image.fromarray(a[y0:y1, x0:x1].astype(np.uint8)).convert("RGB")
    sc = 1000 / max(ov.size)
    ov = ov.resize((round(ov.width * sc), round(ov.height * sc)), Image.LANCZOS)
    d = ImageDraw.Draw(ov); f = font(16)
    X, Y = (cx - x0) * sc, (cy - y0) * sc
    d.line((X - 15, Y, X + 15, Y), fill=RED, width=2); d.line((X, Y - 15, X, Y + 15), fill=RED, width=2)
    for oid, (nx, ny) in nominal(m).items():
        d.text(((nx - x0) * sc + 6, (ny - y0) * sc + 6), oid, fill=RED, font=f)
    for k, (x, y, s) in enumerate(cands):
        X, Y = (x - x0) * sc, (y - y0) * sc
        d.ellipse((X - 10, Y - 10, X + 10, Y + 10), outline=ORANGE, width=2)
        d.text((X + 12, Y - 10), str(k), fill=ORANGE, font=f)
    for r in load_boxes():
        if r["file_name"] == args.file:
            b = [(int(r["x_min"]) - x0) * sc, (int(r["y_min"]) - y0) * sc, (int(r["x_max"]) - x0) * sc, (int(r["y_max"]) - y0) * sc]
            d.rectangle(b, outline=BLUE, width=2); d.text((b[0], b[1] - 16), r["object_id"], fill=BLUE, font=f)
    # chips
    W, Z = (32, 5) if args.small else (48, 5)
    sheet = Image.new("RGB", (4 * (W * Z + 6), ((len(cands) + 3) // 4) * (W * Z + 22)), (12, 22, 40))
    ds = ImageDraw.Draw(sheet)
    for k, (x, y, s) in enumerate(cands):
        ch = Image.fromarray(a[max(0, y - W // 2):y + W // 2, max(0, x - W // 2):x + W // 2].astype(np.uint8)).resize((W * Z, W * Z), Image.NEAREST)
        px, py = (k % 4) * (W * Z + 6), (k // 4) * (W * Z + 22)
        sheet.paste(ch, (px, py + 20))
        ds.text((px + 4, py + 2), f"{k}: ({x},{y}) s={s:.1f} d={((x-cx)**2+(y-cy)**2)**.5:.0f}px", fill=ORANGE, font=f)
    os.makedirs(SCRATCH, exist_ok=True)
    stem = args.file.replace("/", "_")[:-4]
    ov2 = ov.resize((700, round(ov.height * 700 / ov.width)), Image.LANCZOS)
    comb = Image.new("RGB", (700 + sheet.width + 8, max(ov2.height, sheet.height)), (12, 22, 40))
    comb.paste(ov2, (0, 0)); comb.paste(sheet, (708, 0))
    comb.save(os.path.join(SCRATCH, stem + "_hunt.png"))
    print(f"{args.file}  {m['object']}  {m['resolution_m_px']} m/px  nominal=({cx},{cy})  incidence={m['incidence_deg']}  note={m['position_note']}")
    for k, (x, y, s) in enumerate(cands):
        print(f"  {k:2d} ({x:4d},{y:4d}) score={s:5.1f}  d={((x-cx)**2+(y-cy)**2)**.5:5.0f}px")
    print(os.path.join(SCRATCH, stem + "_hunt.png"))


def cmd_grid(args):
    a = img(args.file)
    w = args.w; z = max(2, 800 // w)
    x0, y0 = args.x - w // 2, args.y - w // 2
    c = a.crop((x0, y0, x0 + w, y0 + w)).resize((w * z, w * z), Image.NEAREST).convert("RGB")
    d = ImageDraw.Draw(c); f = font(12)
    for i in range(0, w, 10):
        d.line((i * z, 0, i * z, w * z), fill=(150, 0, 0)); d.line((0, i * z, w * z, i * z), fill=(150, 0, 0))
        d.text((i * z + 2, 2), str(x0 + i), fill=(255, 255, 0), font=f); d.text((2, i * z + 2), str(y0 + i), fill=(255, 255, 0), font=f)
    for r in load_boxes():
        if r["file_name"] == args.file:
            d.rectangle([(int(r["x_min"]) - x0) * z, (int(r["y_min"]) - y0) * z, (int(r["x_max"]) - x0 + 1) * z, (int(r["y_max"]) - y0 + 1) * z], outline=BLUE, width=2)
    os.makedirs(SCRATCH, exist_ok=True)
    p = os.path.join(SCRATCH, args.file.replace("/", "_")[:-4] + f"_grid_{args.x}_{args.y}.png")
    c.save(p); print(p)


def fit_box(a, x, y, w=30):
    """Bounding box of the |z|>2.5 blob connected to the strongest pixel near (x, y)."""
    z = zmap(a)
    x0, y0 = max(0, x - w), max(0, y - w)
    sub = np.abs(z[y0:y + w, x0:x + w]) > 2.5
    sub = ndimage.binary_dilation(sub, iterations=2)
    lab, n = ndimage.label(sub)
    cx, cy = x - x0, y - y0
    if n == 0:
        return None
    # label at/near centre
    r = 6
    win = lab[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
    ids = win[win > 0]
    if ids.size == 0:
        return None
    k = np.bincount(ids).argmax()
    ys, xs = np.nonzero(lab == k)
    return (x0 + xs.min() - 1, y0 + ys.min() - 1, x0 + xs.max() + 1, y0 + ys.max() + 1)


def put(file, oid, box, conf, note, shadow):
    m = meta()[file]
    s = sites()
    label = s[oid]["object"] if oid in s else oid
    rows = [r for r in load_boxes() if not (r["file_name"] == file and r["object_id"] == oid)]
    x0, y0, x1, y1 = [int(v) for v in box]
    rows.append(dict(file_name=file, site_id=m["site_id"], object_id=oid, label=label, x_min=x0, y_min=y0, x_max=x1, y_max=y1,
                     width_px=x1 - x0 + 1, height_px=y1 - y0 + 1, includes_shadow=shadow, confidence=conf, annotator="claude", note=note))
    save_boxes(rows)
    print(f"{file} {oid} box=({x0},{y0})-({x1},{y1}) {x1-x0+1}x{y1-y0+1}px conf={conf}")


def cmd_box(args):
    a = np.asarray(img(args.file), np.float32)
    if args.fit:
        b = fit_box(a, args.x, args.y)
        if b is None:
            sys.exit("no blob found; give --w/--h")
    else:
        b = (args.x - args.w // 2, args.y - args.h // 2, args.x + args.w // 2, args.y + args.h // 2)
    put(args.file, args.object, b, args.conf, args.note, args.shadow)


def cmd_rect(args):
    put(args.file, args.object, (args.x0, args.y0, args.x1, args.y1), args.conf, args.note, args.shadow)


def cmd_drop(args):
    rows = [r for r in load_boxes() if not (r["file_name"] == args.file and (args.object is None or r["object_id"] == args.object))]
    save_boxes(rows)


def cmd_unbox(args):
    if args.file not in meta():
        sys.exit(f"{args.file} is not in metadata.csv")
    u = load_unboxed()
    if args.reason:
        u[args.file] = args.reason
    else:
        u.pop(args.file, None)
    save_boxes(load_boxes(), u)


def cmd_apply(args):
    """Proposals replace any box of the same (file, object); an unboxed reason is only kept for crops left without a box."""
    P = json.load(open(args.file))
    ms, s = meta(), sites()
    rows, u = load_boxes(), load_unboxed()
    drop = {tuple(d) for d in P.get("drop", [])} | {(b["file"], b["object_id"]) for b in P.get("boxes", [])}
    rows = [r for r in rows if (r["file_name"], r["object_id"]) not in drop]
    for b in P.get("boxes", []):
        if b["file"] not in ms:
            sys.exit(f"{b['file']} is not in metadata.csv")
        x0, y0, x1, y1 = (int(b[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
        if not (0 <= x0 <= x1 < 2048 and 0 <= y0 <= y1 < 2048):
            sys.exit(f"box outside the crop: {b}")
        rows.append(dict(file_name=b["file"], site_id=ms[b["file"]]["site_id"], object_id=b["object_id"],
                         label=s[b["object_id"]]["object"] if b["object_id"] in s else b["object_id"],
                         x_min=x0, y_min=y0, x_max=x1, y_max=y1, width_px=x1 - x0 + 1, height_px=y1 - y0 + 1,
                         includes_shadow=b.get("includes_shadow", "yes"), confidence=b["confidence"],
                         annotator=b.get("annotator", "claude"), note=b["note"]))
        u.pop(b["file"], None)
    boxed = {r["file_name"] for r in rows}
    for f, reason in P.get("unboxed", {}).items():
        if f not in ms:
            sys.exit(f"{f} is not in metadata.csv")
        if f not in boxed:
            u[f] = reason
    print(f"{len(P.get('boxes', []))} boxes, {len(P.get('unboxed', {}))} unboxed reasons, {len(P.get('drop', []))} drops"
          + (" (dry run)" if args.dry else ""))
    if not args.dry:
        save_boxes(rows, u)


def cmd_predict(args):
    m = meta()[args.file]
    nom = nominal(m)
    mine = [r for r in load_boxes() if r["file_name"] == args.file and r["object_id"] in nom]
    if not mine:
        print("no measured box in this frame yet"); return
    r = mine[0]
    ax, ay = (int(r["x_min"]) + int(r["x_max"])) / 2, (int(r["y_min"]) + int(r["y_max"])) / 2
    nx, ny = nom[r["object_id"]]
    print(f"anchor {r['object_id']} measured=({ax:.0f},{ay:.0f}) nominal=({nx},{ny}) offset=({ax-nx:+.0f},{ay-ny:+.0f})")
    for oid, (x, y) in nom.items():
        if oid != r["object_id"]:
            print(f"  {oid}: predicted ({x + ax - nx:.0f},{y + ay - ny:.0f})")


def cmd_review(args):
    ms = meta(); boxes = load_boxes()
    by = {}
    for r in boxes:
        by.setdefault(r["file_name"], []).append(r)
    os.makedirs(REVIEW, exist_ok=True)
    f = font(14)
    index = []
    for file, m in ms.items():
        if args.site and m["site_id"] != args.site:
            continue
        rows = by.get(file, [])
        if not rows and m["label"] != "before_arrival":
            index.append((file, m, rows, None)); continue
        a = img(file).convert("RGB")
        if rows:
            xs = [int(r["x_min"]) for r in rows] + [int(r["x_max"]) for r in rows]
            ys = [int(r["y_min"]) for r in rows] + [int(r["y_max"]) for r in rows]
            cx, cy = (min(xs) + max(xs)) // 2, (min(ys) + max(ys)) // 2
            span = max(max(xs) - min(xs), max(ys) - min(ys))
        else:
            cx, cy, span = int(m["obj_x"]), int(m["obj_y"]), 0
        W = int(min(2048, max(120, span + 80)))        # tight zoom
        Z = max(1, 480 // W)
        x0, y0 = int(np.clip(cx - W // 2, 0, 2048 - W)), int(np.clip(cy - W // 2, 0, 2048 - W))
        zoom = a.crop((x0, y0, x0 + W, y0 + W)).resize((W * Z, W * Z), Image.NEAREST)
        dz = ImageDraw.Draw(zoom)
        for r in rows:
            c = CONF_COL[r["confidence"]]
            dz.rectangle([(int(r["x_min"]) - x0) * Z, (int(r["y_min"]) - y0) * Z, (int(r["x_max"]) - x0 + 1) * Z, (int(r["y_max"]) - y0 + 1) * Z], outline=c, width=2)
            dz.text(((int(r["x_min"]) - x0) * Z, (int(r["y_min"]) - y0) * Z - 16), f"{r['object_id']} ({r['confidence']})", fill=c, font=f)
        # context: 600 px window at 1x
        C = 600
        X0, Y0 = int(np.clip(cx - C // 2, 0, 2048 - C)), int(np.clip(cy - C // 2, 0, 2048 - C))
        ctx = a.crop((X0, Y0, X0 + C, Y0 + C))
        dc = ImageDraw.Draw(ctx)
        for r in rows:
            dc.rectangle([int(r["x_min"]) - X0 - 4, int(r["y_min"]) - Y0 - 4, int(r["x_max"]) - X0 + 4, int(r["y_max"]) - Y0 + 4], outline=CONF_COL[r["confidence"]], width=2)
        nx, ny = int(m["obj_x"]) - X0, int(m["obj_y"]) - Y0
        dc.line((nx - 10, ny, nx + 10, ny), fill=RED, width=2); dc.line((nx, ny - 10, nx, ny + 10), fill=RED, width=2)
        H = max(zoom.height, C) + 48
        sheet = Image.new("RGB", (zoom.width + C + 24, H), (12, 22, 40))
        sheet.paste(zoom, (8, 40)); sheet.paste(ctx, (zoom.width + 16, 40))
        ds = ImageDraw.Draw(sheet)
        title = f"{file}   {m['object']}   {m['instrument']} {m['resolution_m_px']} m/px   incidence {m['incidence_deg']}   {m['utc_start'][:10]}"
        if m["label"] == "before_arrival":
            title += "   BEFORE ARRIVAL (negative frame, no box)"
        ds.text((8, 8), title, fill=(230, 230, 230), font=f)
        ds.text((8, 24), f"left: {Z}x zoom   right: 600 px context, red cross = catalog nominal position", fill=(160, 160, 170), font=f)
        out = os.path.join(REVIEW, file.replace("/", "_")[:-4] + ".png")
        sheet.save(out)
        index.append((file, m, rows, os.path.relpath(out, HERE)))
    write_html(index)
    export_dashboard(ms, boxes, index)
    done = sum(1 for _, m, r, _ in index if r or m["label"] == "before_arrival")
    print(f"review: {done}/{len(index)} frames annotated or negative; {len(boxes)} boxes")


def export_dashboard(ms, boxes, index):
    """dashboard/boxes.json + dashboard/boxes/*.jpg for the Mission Control page (dashboard/boxes.html)."""
    dash = os.path.join(ROOT, "dashboard"); out = os.path.join(dash, "boxes"); os.makedirs(out, exist_ok=True)
    __import__("shutil").copyfile(COCO, os.path.join(dash, "boxes_coco.json"))   # for dashboard/coco.html
    frames = []
    for file, m, rows, png in index:
        if not rows:
            continue
        jpg = file.replace("/", "_")[:-4] + ".jpg"
        Image.open(os.path.join(HERE, png)).convert("RGB").save(os.path.join(out, jpg), quality=82)
        frames.append({"file": file, "site_id": m["site_id"], "object": m["object"], "body": m["body"], "instrument": m["instrument"],
                       "resolution_m_px": float(m["resolution_m_px"]), "incidence_deg": float(m["incidence_deg"] or 0), "utc": m["utc_start"][:10],
                       "nominal": [int(m["obj_x"]), int(m["obj_y"])], "sheet": "boxes/" + jpg,
                       "boxes": [{k: (int(r[k]) if k in ("x_min", "y_min", "x_max", "y_max", "width_px", "height_px") else r[k])
                                  for k in ("object_id", "label", "x_min", "y_min", "x_max", "y_max", "width_px", "height_px", "includes_shadow", "confidence", "note")} for r in rows]})
    for old in set(os.listdir(out)) - {f["sheet"][len("boxes/"):] for f in frames}:   # sheets of crops that lost their boxes
        os.remove(os.path.join(out, old))
    unboxed = [{"file_name": f, "site_id": ms[f]["site_id"], "reason": r} for f, r in sorted(load_unboxed().items())]
    # whole-crop thumbnails of the positives with no box, red cross = catalog position (dashboard/coco.html "to check")
    tdir = os.path.join(dash, "unboxed"); os.makedirs(tdir, exist_ok=True)
    keep = set()
    for u in unboxed:
        name = u["file_name"].replace("/", "_")[:-4] + ".jpg"; keep.add(name)
        dst = os.path.join(tdir, name)
        if os.path.exists(dst) and os.path.getmtime(dst) > os.path.getmtime(os.path.join(IMG, u["file_name"])):
            continue
        m = ms[u["file_name"]]; T = 640; s = T / 2048
        t = img(u["file_name"]).convert("RGB").resize((T, T), Image.LANCZOS); d = ImageDraw.Draw(t)
        x, y = int(m["obj_x"]) * s, int(m["obj_y"]) * s
        d.line((x - 12, y, x + 12, y), fill=RED, width=2); d.line((x, y - 12, x, y + 12), fill=RED, width=2)
        t.save(dst, quality=80)
    for old in set(os.listdir(tdir)) - keep:
        os.remove(os.path.join(tdir, old))
    pos = {b: sum(1 for m in ms.values() if m["body"] == b and m["label"] != "before_arrival") for b in ("Moon", "Mars")}
    boxed = {b: sum(1 for f in frames if f["body"] == b) for b in ("Moon", "Mars")}
    json.dump({"generated": __import__("datetime").datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"), "n_boxes": len(boxes), "positives": pos, "boxed": boxed,
               "confidence": {c: sum(1 for r in boxes if r["confidence"] == c) for c in ("high", "medium", "low")},
               "frames": frames, "unboxed": unboxed}, open(os.path.join(dash, "boxes.json"), "w"), indent=1)


def write_html(index):
    css = os.path.join(ROOT, "dashboard", "xenarch.css")
    parts = ["<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
             "<title>Artifact boxes review</title>",
             "<link href='https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&family=JetBrains+Mono:wght@400;500&display=swap' rel='stylesheet'>",
             "<style>body{background:#0c1628;color:#e6e9f0;font-family:Poppins,sans-serif;margin:0;padding:24px 16px;background-image:linear-gradient(rgba(255,255,255,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.04) 1px,transparent 1px);background-size:32px 32px}",
             "h1{font-weight:400;font-size:38px;margin:0 0 4px}h2{font-weight:400;font-size:22px;margin:40px 0 8px;color:#f0884e}",
             ".sub{font-weight:300;font-size:15px;color:#aab;margin-bottom:24px}.lab{font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:#8a93a6}",
             "table{border-collapse:collapse;font-family:'JetBrains Mono',monospace;font-size:12px;width:100%;max-width:1100px}td,th{border-bottom:1px solid rgba(255,255,255,.12);padding:4px 8px;text-align:left}",
             ".card{border:1px solid rgba(255,255,255,.12);background:rgba(12,22,40,.6);padding:12px;margin:12px 0;max-width:1120px}img{max-width:100%;height:auto;display:block}",
             ".flag{color:#f0884e}.miss{color:#d55181}</style></head><body>",
             "<h1>Artifact boxes</h1><div class='sub'>Bounding boxes around human-made hardware in every positive crop. Orange = high confidence, blue = medium, magenta = low. Red cross = catalog nominal position.</div>"]
    cur = None
    for file, m, rows, png in index:
        if m["site_id"] != cur:
            cur = m["site_id"]; parts.append(f"<h2>{m['object']} <span class='lab'>{m['site_id']}</span></h2>")
        parts.append("<div class='card'>")
        if png:
            parts.append(f"<img src='{png}' loading='lazy'>")
        else:
            parts.append(f"<div class='miss lab'>{file}: NOT YET ANNOTATED</div>")
        if rows:
            parts.append("<table><tr><th>object</th><th>x_min</th><th>y_min</th><th>x_max</th><th>y_max</th><th>w</th><th>h</th><th>shadow</th><th>conf</th><th>note</th></tr>")
            for r in rows:
                parts.append(f"<tr><td class='flag'>{r['object_id']}</td><td>{r['x_min']}</td><td>{r['y_min']}</td><td>{r['x_max']}</td><td>{r['y_max']}</td><td>{r['width_px']}</td><td>{r['height_px']}</td><td>{r['includes_shadow']}</td><td>{r['confidence']}</td><td>{r['note']}</td></tr>")
            parts.append("</table>")
        parts.append("</div>")
    parts.append("</body></html>")
    open(os.path.join(HERE, "review.html"), "w").write("\n".join(parts))


def cmd_site(args):
    """For every unboxed frame of a site: terrain-match from each boxed frame, print candidates, write a grid at the best."""
    import register as RG
    ms = meta(); boxes = load_boxes()
    frames = [f for f, m in ms.items() if m["site_id"] == args.site and m["label"] != "before_arrival"]
    done = {r["file_name"]: r for r in boxes if r["site_id"] == args.site and r["object_id"] == args.site}
    if not done:
        print("no anchor box yet for", args.site); return
    for f in frames:
        if f in done and not args.all:
            continue
        print("==", f, ms[f]["resolution_m_px"], "m/px incidence", ms[f]["incidence_deg"])
        best = []
        for rf, r in done.items():
            x, y = (int(r["x_min"]) + int(r["x_max"])) // 2, (int(r["y_min"]) + int(r["y_max"])) // 2
            res = RG.locate_points(rf, x, y, f)[:2]
            for name, px, py, pk, ratio, rx, ry, ang in res:
                best.append((pk, ratio, name, px, py, rx, ry, ang, rf))
        best.sort(reverse=True)
        for pk, ratio, name, px, py, rx, ry, ang, rf in best[:4]:
            print(f"   ({px:4d},{py:4d}) votes={pk:.0f} ratio={ratio:.2f} {name} rx={rx} ry={ry} ang={ang} from {os.path.basename(rf)}")
        pk, ratio, name, px, py, rx, ry, ang, rf = best[0]
        args2 = argparse.Namespace(file=f, x=px, y=py, w=args.w); cmd_grid(args2)


def cmd_transfer(args):
    """Copy boxes between crops cut from the same source frame (exact: crops differ by a known pixel offset)."""
    ms = meta(); boxes = load_boxes()
    have = {(r["file_name"], r["object_id"]) for r in boxes}
    new = []
    for r in list(boxes):
        src = ms[r["file_name"]]
        for f, m in ms.items():
            if f == r["file_name"] or m["frame_id"] != src["frame_id"]:
                continue
            dx, dy = int(src["crop_x0"]) - int(m["crop_x0"]), int(src["crop_y0"]) - int(m["crop_y0"])
            b = [int(r["x_min"]) + dx, int(r["y_min"]) + dy, int(r["x_max"]) + dx, int(r["y_max"]) + dy]
            if (f, r["object_id"]) in have or b[0] < 0 or b[1] < 0 or b[2] >= 2048 or b[3] >= 2048:
                continue
            n = dict(r, file_name=f, site_id=m["site_id"], x_min=b[0], y_min=b[1], x_max=b[2], y_max=b[3],
                     note=f"transferred from crop {r['file_name']} (same frame, offset {dx:+d},{dy:+d})")
            new.append(n); have.add((f, r["object_id"]))
            print(f"{f} {r['object_id']} <- {r['file_name']} ({dx:+d},{dy:+d})")
    save_boxes(boxes + new)
    print(len(new), "boxes transferred")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("hunt"); q.add_argument("file"); q.add_argument("--radius", type=int); q.add_argument("--top", type=int, default=12); q.add_argument("--at"); q.add_argument("--small", action="store_true")
    q = sp.add_parser("grid"); q.add_argument("file"); q.add_argument("x", type=int); q.add_argument("y", type=int); q.add_argument("--w", type=int, default=80)
    q = sp.add_parser("box"); q.add_argument("file"); q.add_argument("object"); q.add_argument("x", type=int); q.add_argument("y", type=int)
    q.add_argument("--fit", action="store_true"); q.add_argument("--w", type=int, default=20); q.add_argument("--h", type=int, default=20)
    q.add_argument("--conf", default="high"); q.add_argument("--note", default=""); q.add_argument("--shadow", default="yes")
    q = sp.add_parser("rect"); q.add_argument("file"); q.add_argument("object"); [q.add_argument(k, type=int) for k in ("x0", "y0", "x1", "y1")]
    q.add_argument("--conf", default="high"); q.add_argument("--note", default=""); q.add_argument("--shadow", default="yes")
    q = sp.add_parser("drop"); q.add_argument("file"); q.add_argument("object", nargs="?")
    q = sp.add_parser("unbox"); q.add_argument("file"); q.add_argument("reason", nargs="?")
    q = sp.add_parser("apply"); q.add_argument("file"); q.add_argument("--dry", action="store_true")
    q = sp.add_parser("predict"); q.add_argument("file")
    q = sp.add_parser("review"); q.add_argument("--site")
    sp.add_parser("transfer")
    q = sp.add_parser("site"); q.add_argument("site"); q.add_argument("--patch", type=int, default=1000); q.add_argument("--w", type=int, default=200); q.add_argument("--all", action="store_true")
    a = p.parse_args()
    globals()["cmd_" + a.cmd](a)

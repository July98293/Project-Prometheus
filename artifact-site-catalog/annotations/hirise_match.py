#!/usr/bin/env python
"""Locate a verified Mars object in the other HiRISE crops of its site, and place metric boxes there.

HiRISE crops are map-projected (north up), but each frame's projection is off by 100-400 m (pointing), so the
catalog pixel cannot be trusted. Anchor the object by eye in one crop, then:

  hirise_match.py match SITE ANCHOR_FILE X Y [--patch 500] [--kind grad] [--half 20] [--out sheet.png]
        NCC of a patch around (X, Y) of ANCHOR_FILE over every crop of SITE (scale from the resolutions, no
        rotation). Prints the matched point and peak ratio per crop (ratio > ~5 = real match, ~1 = the place is
        not in that crop) and writes a contact sheet of the matched points to check by eye. Keep the patch inside
        the anchor crop: a patch cut off by the crop edge matches badly.
  hirise_match.py place SPEC.json [--out sheet.png]
        SPEC: [{file, x, y, mode: bright|dark, search_m, half_x_m, up_m, down_m}]. Moves each point to the local
        brightest (deck) or darkest (shadow) pixel within search_m and puts a box of the given size in metres
        around it. Writes SPEC_out.json (with "box" and "centre") and a check sheet; turn it into a proposals file
        for `boxtool.py apply`.
"""
import argparse, csv, json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import register as RG

IMG = RG.IMG
META = RG.META
ORANGE, GREY, NAVY = (240, 136, 78), (200, 200, 200), (12, 22, 40)


def font(size=13):
    for p in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def gray(file):
    return np.asarray(Image.open(os.path.join(IMG, file)).convert("L"), np.float32)


def match(site, ref, X, Y, patch=500, kind="grad"):
    """[(file, x, y, ncc, peak ratio)] for every crop of `site`."""
    ra, rres = gray(ref), float(META[ref]["resolution_m_px"])
    out = []
    for f, m in META.items():
        if m["site_id"] != site:
            continue
        ta, tres = gray(f), float(m["resolution_m_px"])
        sc, h = rres / tres, patch // 2
        x0, y0 = max(0, X - h), max(0, Y - h)
        P = ra[y0:Y + h, x0:X + h]
        P = np.asarray(Image.fromarray(P).resize((round(P.shape[1] * sc), round(P.shape[0] * sc)), Image.LANCZOS), np.float32)
        ds = 2 if tres < 0.4 else 1
        sig = 2.0 if ds == 2 else 1.0
        I, T = RG.prep(ta, kind, sig, ds), RG.prep(P, kind, sig, ds)
        if T.shape[0] >= I.shape[0] or T.shape[1] >= I.shape[1]:
            continue
        c = RG.ncc_fft(I, T)
        py, px = np.unravel_index(np.argmax(c), c.shape)
        c2 = c.copy(); c2[max(0, py - 15):py + 15, max(0, px - 15):px + 15] = -1
        out.append((f, round(px * ds + (X - x0) * sc), round(py * ds + (Y - y0) * sc), float(c[py, px]), float(c[py, px] / max(c2.max(), 1e-6))))
    return out


def sheet(chips, path, cols=6, C=240):
    """chips: [(PIL RGB image, title, subtitle)]"""
    f = font()
    S = Image.new("RGB", (cols * (C + 6), ((len(chips) + cols - 1) // cols) * (C + 34)), NAVY)
    d = ImageDraw.Draw(S)
    for k, (im, t1, t2) in enumerate(chips):
        px, py = (k % cols) * (C + 6), (k // cols) * (C + 34)
        S.paste(im.resize((C, C), Image.NEAREST), (px, py + 32))
        d.text((px + 2, py + 2), t1, fill=ORANGE, font=f); d.text((px + 2, py + 16), t2, fill=GREY, font=f)
    S.save(path)


def cmd_match(a):
    res = match(a.site, a.anchor, a.x, a.y, a.patch, a.kind)
    chips = []
    for f, x, y, pk, ratio in res:
        m = META[f]; r = float(m["resolution_m_px"]); h = int(a.half / r)
        print(f"{f}  ({x},{y})  ncc={pk:.2f}  ratio={ratio:.1f}")
        im = Image.open(os.path.join(IMG, f)).convert("RGB").crop((x - h, y - h, x + h, y + h))
        chips.append((im, f"{m['frame_id']} {m['label'][:6]} r={ratio:.1f}", f"{r} m i={float(m['incidence_deg']):.0f} {m['utc_start'][:10]} ({x},{y})"))
    sheet(chips, a.out); print(a.out)


def cmd_place(a):
    spec, out, chips = json.load(open(a.spec)), [], []
    for s in spec:
        res = float(META[s["file"]]["resolution_m_px"])
        g = ndimage.gaussian_filter(gray(s["file"]), 0.8)
        r = max(2, int(s.get("search_m", 2.5) / res)); x, y = s["x"], s["y"]
        sub = g[y - r:y + r + 1, x - r:x + r + 1].copy()
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        far = xx ** 2 + yy ** 2 > r * r
        sub[far] = sub.min() if s["mode"] == "bright" else sub.max()
        iy, ix = np.unravel_index(np.argmax(sub) if s["mode"] == "bright" else np.argmin(sub), sub.shape)
        cx, cy = x - r + ix, y - r + iy
        hx, up, dn = (round(s.get(k, d) / res) for k, d in (("half_x_m", 3.5), ("up_m", 3.0), ("down_m", 5.0)))
        box = [max(0, int(cx - hx)), max(0, int(cy - up)), min(2047, int(cx + hx)), min(2047, int(cy + dn))]
        out.append(dict(s, box=box, centre=[int(cx), int(cy)]))
        h = int(15 / res); z = 240 / (2 * h)
        im = Image.open(os.path.join(IMG, s["file"])).convert("RGB").crop((cx - h, cy - h, cx + h, cy + h)).resize((240, 240), Image.NEAREST)
        ImageDraw.Draw(im).rectangle([(box[0] - cx + h) * z, (box[1] - cy + h) * z, (box[2] - cx + h + 1) * z, (box[3] - cy + h + 1) * z], outline=ORANGE, width=2)
        chips.append((im, f"{META[s['file']]['frame_id']} {s['mode']}", f"box {box}"))
    json.dump(out, open(a.spec.replace(".json", "_out.json"), "w"), indent=1)
    sheet(chips, a.out); print(a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(); sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("match"); q.add_argument("site"); q.add_argument("anchor"); q.add_argument("x", type=int); q.add_argument("y", type=int)
    q.add_argument("--patch", type=int, default=500); q.add_argument("--kind", default="grad"); q.add_argument("--half", type=float, default=20)
    q.add_argument("--out", default="hirise_match.png")
    q = sp.add_parser("place"); q.add_argument("spec"); q.add_argument("--out", default="hirise_place.png")
    a = p.parse_args()
    {"match": cmd_match, "place": cmd_place}[a.cmd](a)

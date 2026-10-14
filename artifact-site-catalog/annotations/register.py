"""Locate a verified object in another frame of the same site by terrain matching.

register.py REF_FILE X Y TARGET_FILE [--patch 500]
Prints the best (x, y) in TARGET for the point (X, Y) of REF, per orientation, with NCC peak and peak/second ratio.
Uses gradient magnitude (lighting-direction tolerant), scales by the resolution ratio, tries 8 orientations.
"""
import sys, os, argparse, csv
import numpy as np
from PIL import Image
from scipy import ndimage
from numpy.fft import rfft2, irfft2

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(os.path.dirname(os.path.dirname(HERE)), "data", "artifact-site-images")
META = {r["file_name"]: r for r in csv.DictReader(open(os.path.join(IMG, "metadata.csv")))}


def prep(a, kind="int", sigma=2.0, ds=2):
    g = ndimage.gaussian_filter(a, sigma)
    if kind == "int":                                       # band-passed intensity
        m = g - ndimage.uniform_filter(g, 101)
    elif kind == "grad":                                    # gradient magnitude, tolerant to sun direction
        m = np.hypot(ndimage.sobel(g, 1), ndimage.sobel(g, 0)); m = m / (ndimage.uniform_filter(m, 101) + 1e-3)
    elif kind == "lap":
        m = np.abs(ndimage.gaussian_laplace(a, sigma))
    return m[::ds, ::ds].astype(np.float32)


def ncc_fft(I, T):
    """Normalized cross-correlation of template T over image I (valid region), via FFT."""
    h, w = T.shape; H, W = I.shape
    T = (T - T.mean()) / (T.std() + 1e-6)
    FI = rfft2(I, (H + h, W + w)); FT = rfft2(T[::-1, ::-1], (H + h, W + w))
    corr = irfft2(FI * FT, (H + h, W + w))[h - 1:H, w - 1:W]
    ones = np.ones_like(T)
    FO = rfft2(ones, (H + h, W + w))
    s1 = irfft2(rfft2(I, (H + h, W + w)) * FO, (H + h, W + w))[h - 1:H, w - 1:W]
    s2 = irfft2(rfft2(I * I, (H + h, W + w)) * FO, (H + h, W + w))[h - 1:H, w - 1:W]
    n = h * w
    sd = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 1e-6))
    return corr / (n * sd)


ORIENTS = {"as-is": lambda t: t, "rot90": lambda t: np.rot90(t, 1), "rot180": lambda t: np.rot90(t, 2), "rot270": lambda t: np.rot90(t, 3),
           "flipx": lambda t: t[:, ::-1], "flipy": lambda t: t[::-1], "transpose": lambda t: t.T, "antitranspose": lambda t: np.rot90(t, 2).T}


def locate(ref_file, x, y, target_file, patch=1000, kind="int", sigma=3.0, ds=4, aspects=(0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4)):
    """Best (orientation, x, y, ncc, peak ratio, aspect) over 8 orientations x vertical aspect ratios."""
    ra = np.asarray(Image.open(os.path.join(IMG, ref_file)).convert("L"), np.float32)
    ta = np.asarray(Image.open(os.path.join(IMG, target_file)).convert("L"), np.float32)
    scale = float(META[ref_file]["resolution_m_px"]) / float(META[target_file]["resolution_m_px"])   # ref px -> target px
    half = patch // 2
    x0, y0 = max(0, x - half), max(0, y - half)
    P0 = ra[y0:y + half, x0:x + half]
    I = prep(ta, kind, sigma, ds)
    out = []
    for ry in aspects:
        P = np.asarray(Image.fromarray(P0).resize((round(P0.shape[1] * scale), round(P0.shape[0] * scale * ry)), Image.LANCZOS), np.float32)
        px, py = (x - x0) * scale / ds, (y - y0) * scale * ry / ds
        T0 = prep(P, kind, sigma, ds)
        idx = np.zeros(T0.shape, np.int64); idx.flat = np.arange(idx.size)
        iy, ix = min(int(round(py)), T0.shape[0] - 1), min(int(round(px)), T0.shape[1] - 1)
        for name, f in ORIENTS.items():
            T = np.ascontiguousarray(f(T0))
            pos = np.argwhere(f(idx) == idx[iy, ix])[0]
            c = ncc_fft(I, T)
            c[~np.isfinite(c)] = 0
            j = np.unravel_index(c.argmax(), c.shape)
            peak = c[j]
            c2 = c.copy(); c2[max(0, j[0] - 10):j[0] + 10, max(0, j[1] - 10):j[1] + 10] = 0
            out.append((name, int((j[1] + pos[1]) * ds), int((j[0] + pos[0]) * ds), float(peak), float(peak / max(c2.max(), 1e-6)), ry))
    out.sort(key=lambda t: -t[3])
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("ref"); p.add_argument("x", type=int); p.add_argument("y", type=int); p.add_argument("target"); p.add_argument("--patch", type=int, default=1000)
    p.add_argument("--kind", default="int"); p.add_argument("--sigma", type=float, default=3.0); p.add_argument("--ds", type=int, default=4)
    a = p.parse_args()
    for name, x, y, peak, ratio, ry in locate(a.ref, a.x, a.y, a.target, a.patch, a.kind, a.sigma, a.ds)[:6]:
        print(f"{name:14s} ry={ry:.1f} -> ({x:4d},{y:4d})  ncc={peak:.3f}  ratio={ratio:.2f}")


# ── point-set matching (lighting independent): craters/boulders as points, Hough vote over translation ──
def points(a, n=500, nms=15):
    b = ndimage.uniform_filter(a, 7)
    s = ndimage.maximum_filter(b, 15) - ndimage.minimum_filter(b, 15)
    s = s / (ndimage.uniform_filter(s, 201) + 1)                      # flatten
    peak = (s == ndimage.maximum_filter(s, 2 * nms + 1)) & (s > 0)
    ys, xs = np.nonzero(peak)
    order = np.argsort(-s[ys, xs])[:n]
    return np.stack([xs[order], ys[order]], 1).astype(np.float32)


def orient_pts(p, name, W, H):
    """Apply one of the 8 orientations to points in a W x H frame (same convention as ORIENTS on arrays)."""
    x, y = p[:, 0], p[:, 1]
    return {"as-is": (x, y), "rot90": (y, W - 1 - x), "rot180": (W - 1 - x, H - 1 - y), "rot270": (H - 1 - y, x),
            "flipx": (W - 1 - x, y), "flipy": (x, H - 1 - y), "transpose": (y, x), "antitranspose": (H - 1 - y, W - 1 - x)}[name]


SCALES = tuple(round(0.4 + 0.1 * i, 1) for i in range(17))   # ref px -> target px, 0.4 .. 2.0, x and y independent
OMAT = {"as-is": [[1, 0], [0, 1]], "rot90": [[0, -1], [1, 0]], "rot180": [[-1, 0], [0, -1]], "rot270": [[0, 1], [-1, 0]],
        "flipx": [[-1, 0], [0, 1]], "flipy": [[1, 0], [0, -1]], "transpose": [[0, 1], [1, 0]], "antitranspose": [[0, -1], [-1, 0]]}


def locate_points(ref_file, x, y, target_file, scales=SCALES, angles=(0,), bin_px=10, n=500):
    """Best (orientation, x, y, votes, peak ratio, rx, ry, angle) by Hough vote over translation for every
    orientation x (rx, ry) scale pair x rotation. NAC pixels are not square and the catalog resolution is
    only a mean, so both scales are searched. ratio > ~3 means a real match (error typically < 10 px)."""
    ra = np.asarray(Image.open(os.path.join(IMG, ref_file)).convert("L"), np.float32)
    ta = np.asarray(Image.open(os.path.join(IMG, target_file)).convert("L"), np.float32)
    P, Q = points(ra, n) - 1024, points(ta, n)
    A0 = np.array([[x, y]], np.float32) - 1024
    NB = 6000 // bin_px
    out = []
    for ang in angles:
        c, s_ = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        Rm = np.array([[c, s_], [-s_, c]], np.float32)
        for name, M in OMAT.items():
            M = np.array(M, np.float32)
            for rx in scales:
                for ry in scales:
                    S = np.array([[rx, 0], [0, ry]], np.float32)
                    T = Rm @ S @ M.T
                    Pt = P @ T; At = A0 @ T
                    dx = (Q[:, 0][:, None] - Pt[:, 0][None, :]).ravel(); dy = (Q[:, 1][:, None] - Pt[:, 1][None, :]).ravel()
                    hx = np.floor((dx + 3000) / bin_px).astype(np.int64); hy = np.floor((dy + 3000) / bin_px).astype(np.int64)
                    ok = (hx >= 0) & (hx < NB) & (hy >= 0) & (hy < NB)
                    h = np.bincount((hy * NB + hx)[ok], minlength=NB * NB).reshape(NB, NB).astype(np.float32)
                    h = ndimage.uniform_filter(h, 2) * 4
                    j = np.unravel_index(h.argmax(), h.shape); votes = h[j]
                    h2 = h.copy(); h2[max(0, j[0] - 3):j[0] + 4, max(0, j[1] - 3):j[1] + 4] = 0
                    tx, ty = (j[1] + 0.5) * bin_px - 3000, (j[0] + 0.5) * bin_px - 3000
                    out.append((name, int(At[0, 0] + tx), int(At[0, 1] + ty), float(votes), float(votes / max(h2.max(), 1)), rx, ry, ang))
    out.sort(key=lambda t: -t[3])
    return out

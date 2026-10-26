"""
Xenarch synthetic injections: natural-terrain images with a space-hardware object pasted in, plus the untouched
twin and a pixel mask, for injection-recovery tests (how often does the detector find an object of this size, at
this sun angle, on this terrain?). Test data only: Xenarch itself trains on natural terrain alone.

Backgrounds: 512x512 crops of the natural-terrain tiles (LROC NAC Moon, HiRISE Mars), never hardware sites.
Three kinds of injection, one object per image:

  rendered      past and new hardware built from simple 3D shapes at real dimensions (HARDWARE below), lit by the
                tile's own sun: Lambert shading, cast shadows from a height field, saturation of bright foil.
  real-sprite   real hardware cut out of the LROC / HiRISE crops (high-confidence boxes of boxes_coco.json): its
                contrast against its own ground, rotated so its shadow follows the tile's sun, rescaled to the tile's
                m/px, only into tiles whose sun incidence is within 15 deg of the source frame's.
  null-control  a patch of natural terrain from another tile pasted through the same pipeline: a detector that
                flags these is seeing paste artifacts, not objects.

Variability per image (all recorded in the table): object scale x0.85-1.2; target resolution (the tile's own,
and on the Moon a coarser camera by cutting 1-2x larger and resampling); sun incidence and azimuth (the tile's);
object brightness (albedo x0.6-1.4, sprite contrast x0.8-1.2); camera PSF 0.35-0.9 px, the tile's own noise
added to the object, gain / gamma on the whole image; terrain texture (12 landform folders, any spot of the
tile); context with the object: blast zone around landers, wheel tracks behind rovers, footpaths around crewed
landers, bright ejecta around impact craters.

Sun azimuth in the image is not in the labels: it is estimated from the tile's own shading (offset between dark
and bright blobs: on the Moon crater shadows sit on the sun side, so the offset points away from the sun; on Mars
relief dominates and it points toward it). Checked on the real hardware crops (shadow direction of the lander
itself): Moon 24 of 37 within 45 deg, the rest mostly weak-shading frames; Mars 14 of 14 clean rover / lander
frames. `sun_azimuth_confidence` < 0.1 means weak shading (high sun: short shadows, azimuth matters little).

Writes data/synthetic-injections/{injected,clean,masks}/<body>/<kind>/<id>.png, metadata.csv,
boxes_coco.json and contact-sheets/. Mask values: 1 object, 2 its cast shadow, 3 context (halo, tracks, ejecta).

  .venv/bin/python xenarch_inject.py [--per-tile 1] [--sprite-frac 0.6] [--null-frac 0.25] [--val-extra 2] [--limit N] [--seed 7]

Evaluate on `split` = validation (backgrounds never in Mk18's memory bank); the train-split images serve other models
and visual checks.
  .venv/bin/python xenarch_hf.py injections     publish to synthetic-injections_lroc-nac_hirise/ of the HF dataset
"""
import csv, hashlib, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = Path(__file__).resolve().parent
OUT = HERE / "data/synthetic-injections"
SIZE, FINE = 512, 3                        # output image side; render oversampling (fine px per image px)
TILES = {"Moon": HERE / "data/lroc-nac-set", "Mars": HERE / "data/hirise-set"}
CROPS = HERE / "data/artifact-site-images"
COCO = HERE / "artifact-site-catalog/annotations/boxes_coco.json"
SITES = HERE / "artifact-site-catalog/sites.csv"
CAMERA = {"Moon": "LROC NAC", "Mars": "HiRISE RED"}
SHADOW_AMBIENT = {"Moon": (0.02, 0.07), "Mars": (0.12, 0.28)}     # shadow radiance / flat lit ground
MAX_INCIDENCE = 86.0                       # renders at grazing sun get shadows longer than the image

# ── hardware catalogue: metres, height above ground, albedo relative to the regolith ────────────────────────────
# shapes: ("poly", n, r, h, alb) n-gon prism · ("cyl", r, h, alb) · ("box", w, l, h, alb[, dx, dy, rot])
#         ("dome", r, h, alb) · ("cone", r0, r1, z0, z1, alb) · ("hcyl", r, length, alb) · ("legs", n, r, h, alb)
#         ("disk", r, h, alb, dx, dy) flat panel on a mast · ("canopy", r, alb, dx, dy) parachute lying in folds
#         ("crater", diameter)
# the shapes of one object share its centre unless dx, dy are given; rot in degrees
HARDWARE = {
    # Moon, past
    "apollo-lm-descent-stage": dict(era="past", cls="lander", body="Moon", name="Apollo LM descent stage", context="footpaths",
        shapes=[("poly", 8, 2.1, 3.2, 2.6), ("legs", 4, 4.7, 2.0, 1.6), ("box", 1.2, 1.2, 3.5, 2.0, 0, 0, 0)], tracks=None),
    "surveyor": dict(era="past", cls="lander", body="Moon", name="Surveyor lander",
        shapes=[("poly", 3, 0.9, 1.4, 2.0), ("legs", 3, 2.1, 1.0, 1.4), ("cyl", 0.08, 3.0, 1.5), ("disk", 0.6, 2.6, 0.5, 0.4, 0.0)]),
    "luna-sample-return": dict(era="past", cls="lander", body="Moon", name="Luna 16-24 descent stage",
        shapes=[("cyl", 1.3, 1.9, 1.7), ("legs", 4, 2.1, 1.0, 1.4), ("dome", 0.6, 1.0, 1.8)]),
    "lunokhod": dict(era="past", cls="rover", body="Moon", name="Lunokhod rover",
        shapes=[("box", 1.6, 1.7, 1.35, 1.6), ("box", 1.6, 1.4, 1.5, 0.5, 0.0, 0.9, 0)], tracks=1.6),
    "apollo-lrv": dict(era="past", cls="rover", body="Moon", name="Apollo Lunar Roving Vehicle",
        shapes=[("box", 2.3, 3.1, 0.9, 1.5), ("disk", 0.5, 1.6, 2.5, 0.0, 1.0)], tracks=1.8),
    "change3-lander": dict(era="past", cls="lander", body="Moon", name="Chang'e 3/4 lander",
        shapes=[("box", 2.6, 2.6, 2.2, 2.2), ("legs", 4, 2.6, 1.2, 1.4), ("box", 1.0, 2.4, 2.4, 0.5, 2.0, 0, 0), ("box", 1.0, 2.4, 2.4, 0.5, -2.0, 0, 0)]),
    "yutu": dict(era="past", cls="rover", body="Moon", name="Yutu rover",
        shapes=[("box", 1.0, 1.5, 1.1, 1.8), ("box", 2.0, 0.8, 1.2, 0.5, 0.0, 0.1, 0)], tracks=1.0),
    "spacecraft-impact": dict(era="past", cls="impact-crater", body="Moon", name="spacecraft / rocket-stage impact crater",
        shapes=[("crater", 25.0)]),
    # Moon, new
    "nova-c": dict(era="new", cls="lander", body="Moon", name="Intuitive Machines Nova-C",
        shapes=[("poly", 6, 0.9, 4.3, 2.2), ("legs", 6, 2.3, 1.5, 1.4)]),
    "blue-ghost": dict(era="new", cls="lander", body="Moon", name="Firefly Blue Ghost",
        shapes=[("poly", 8, 1.6, 2.0, 2.0), ("legs", 4, 1.9, 1.0, 1.4), ("disk", 0.9, 2.3, 0.6, 0.0, 0.0)]),
    "blue-moon-mk1": dict(era="new", cls="lander", body="Moon", name="Blue Origin Blue Moon MK1",
        shapes=[("cyl", 1.5, 8.0, 2.4), ("cone", 1.5, 0.6, 8.0, 9.0, 2.4), ("legs", 4, 3.2, 1.8, 1.6)]),
    "starship-hls": dict(era="new", cls="lander", body="both", name="SpaceX Starship HLS",
        shapes=[("cyl", 4.5, 44.0, 2.8), ("cone", 4.5, 0.8, 44.0, 50.0, 2.8), ("legs", 6, 6.5, 3.0, 1.8)]),
    "surface-habitat": dict(era="new", cls="habitat", body="Moon", name="Artemis surface habitat (lying cylinder)",
        shapes=[("hcyl", 2.0, 10.0, 2.2), ("box", 2.0, 2.0, 2.2, 1.8, 0.0, 6.0, 0)], context="footpaths"),
    "lunar-terrain-vehicle": dict(era="new", cls="rover", body="Moon", name="Lunar Terrain Vehicle",
        shapes=[("box", 2.5, 4.0, 1.6, 1.8), ("disk", 0.6, 2.4, 0.5, 0.0, -1.2)], tracks=2.4),
    "vertical-solar-array": dict(era="new", cls="power", body="Moon", name="vertical solar array tower",
        shapes=[("box", 1.6, 0.3, 14.0, 0.6), ("poly", 4, 1.2, 1.0, 1.6)]),
    "fission-power-unit": dict(era="new", cls="power", body="Moon", name="fission surface power unit with radiator",
        shapes=[("cyl", 1.1, 4.0, 2.0), ("box", 6.0, 0.4, 3.0, 1.8, 0.0, 2.0, 0)]),
    # Mars, past
    "viking-lander": dict(era="past", cls="lander", body="Mars", name="Viking lander",
        shapes=[("poly", 6, 0.8, 1.6, 2.0), ("legs", 3, 1.5, 0.7, 1.5), ("disk", 0.4, 2.0, 2.4, 0.5, 0.3)]),
    "pathfinder": dict(era="past", cls="lander", body="Mars", name="Mars Pathfinder with open petals",
        shapes=[("poly", 3, 0.6, 0.9, 1.8), ("box", 1.1, 1.6, 0.25, 0.5, 1.1, 0.0, 0), ("box", 1.1, 1.6, 0.25, 0.5, -0.55, 0.95, 60), ("box", 1.1, 1.6, 0.25, 0.5, -0.55, -0.95, -60)]),
    "phoenix-insight": dict(era="past", cls="lander", body="Mars", name="Phoenix / InSight lander with round arrays",
        shapes=[("cyl", 0.8, 1.2, 2.0), ("disk", 1.1, 1.3, 0.6, 2.0, 0.0), ("disk", 1.1, 1.3, 0.6, -2.0, 0.0), ("legs", 3, 1.2, 0.8, 1.4)]),
    "mer-rover": dict(era="past", cls="rover", body="Mars", name="Spirit / Opportunity rover",
        shapes=[("box", 1.9, 1.6, 1.0, 0.7), ("cyl", 0.08, 1.5, 1.6)], tracks=1.2),
    "msl-rover": dict(era="past", cls="rover", body="Mars", name="Curiosity / Perseverance rover",
        shapes=[("box", 2.7, 3.0, 1.4, 1.8), ("cyl", 0.12, 2.2, 1.8, 0.5, 1.0)], tracks=2.8),
    "heat-shield": dict(era="past", cls="landing-hardware", body="Mars", name="heat shield (4.5 m)",
        shapes=[("cone", 2.25, 0.4, 0.0, 0.9, 2.4)]),
    "backshell-parachute": dict(era="past", cls="landing-hardware", body="Mars", name="backshell with parachute lying flat",
        shapes=[("cone", 2.25, 1.0, 0.0, 1.8, 1.8), ("canopy", 7.5, 2.4, 0.0, 10.0)]),
    # Mars, new
    "habitat-dome": dict(era="new", cls="habitat", body="Mars", name="habitat dome (10 m)",
        shapes=[("dome", 5.0, 4.5, 2.0)], context="footpaths"),
    "sample-retrieval-lander": dict(era="new", cls="lander", body="Mars", name="Mars Sample Retrieval Lander",
        shapes=[("poly", 6, 1.8, 2.5, 2.0), ("legs", 4, 3.2, 1.2, 1.4), ("disk", 1.6, 2.7, 0.6, 3.6, 0.0), ("disk", 1.6, 2.7, 0.6, -3.6, 0.0)]),
    "pressurised-rover": dict(era="new", cls="rover", body="Mars", name="pressurised crew rover",
        shapes=[("hcyl", 1.5, 6.0, 1.9), ("box", 3.4, 6.5, 0.8, 1.2)], tracks=3.0),
}
SPRITE_TYPES = {"crewed lander": "lander", "robotic lander": "lander", "rover": "rover", "landing hardware": "landing-hardware",
                "surface experiment": "surface-experiment"}
FIELDS = ["file_name", "clean_file", "mask_file", "split", "body", "instrument", "kind", "era", "object_class",
          "hardware_id", "hardware_name", "sprite_source", "x_center", "y_center", "box_x_min", "box_y_min", "box_x_max",
          "box_y_max", "object_px", "shadow_px", "context", "scale_factor", "width_m", "height_m", "resolution_m_px",
          "camera_downscale", "incidence_deg", "sun_azimuth_image_deg", "sun_azimuth_confidence", "albedo_factor",
          "contrast_factor", "psf_sigma_px", "noise_sigma", "gain", "gamma", "rotation_deg", "mirrored",
          "landform", "shadowed_fraction", "background_file", "background_x0", "background_y0", "background_side_px",
          "frame_id", "seed"]


# ── sun direction from shading ────────────────────────────────────────────────────────────────────────────────
def sun_azimuth(a, body, r0=2, r1=14):
    """(azimuth toward the sun in image degrees, 0 = +x, 90 = +y (down); strength) from the mean offset between
    dark and bright blobs of the high-passed image (see module doc for the sign per body)."""
    a = np.asarray(a, np.float32)
    g = ndimage.gaussian_filter(a, 1.0); hp = g - ndimage.gaussian_filter(g, 12)
    z = hp / (np.median(np.abs(hp - np.median(hp))) * 1.4826 + 1e-6)
    D = (z < -2.0).astype(np.float32); B = (z > 2.0).astype(np.float32)
    H, W = a.shape
    C = np.fft.fftshift(np.fft.irfft2(np.conj(np.fft.rfft2(D)) * np.fft.rfft2(B), (H, W)))
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[-r1:r1 + 1, -r1:r1 + 1]; rr = np.maximum(np.hypot(yy, xx), 1)
    win = C[cy - r1:cy + r1 + 1, cx - r1:cx + r1 + 1]
    anti, ring = win - win[::-1, ::-1], (rr >= r0) & (rr <= r1)
    vx, vy = (anti * xx / rr)[ring].sum(), (anti * yy / rr)[ring].sum()
    strength = math.hypot(vx, vy) / max((win + win[::-1, ::-1])[ring].sum() / 2, 1e-6)
    dark_to_bright = math.degrees(math.atan2(vy, vx))
    toward_sun = dark_to_bright + 180 if body == "Moon" else dark_to_bright
    return (toward_sun + 180) % 360 - 180, float(strength)


# ── rendering ─────────────────────────────────────────────────────────────────────────────────────────────────
def _poly_mask(X, Y, n, r, rot=0.0):
    phi = np.arctan2(Y, X) - rot
    seg = 2 * np.pi / n
    return np.hypot(X, Y) <= r * np.cos(np.pi / n) / np.cos((phi % seg) - seg / 2)


def heightfield(spec, X, Y, rot, scale, alb_k, rng):
    """Height (m), albedo (x regolith; 0 where no object), ground factor (craters / ejecta) on the grid X, Y (m)."""
    H = np.zeros_like(X); A = np.zeros_like(X); G = np.ones_like(X)
    c, s = math.cos(rot), math.sin(rot)
    for sh in spec["shapes"]:
        kind, p = sh[0], sh[1:]
        def local(dx=0.0, dy=0.0, r2=0.0):
            u0, v0 = X - (dx * c - dy * s) * scale, Y - (dx * s + dy * c) * scale
            cr, sr = math.cos(rot + r2), math.sin(rot + r2)
            return u0 * cr + v0 * sr, -u0 * sr + v0 * cr
        if kind == "crater":
            D = p[0] * scale; r = D / 2
            rho = np.hypot(X, Y)
            bowl = np.where(rho < r, -0.2 * D * (1 - (rho / r) ** 2), 0.0)
            rim = 0.04 * D * np.exp(-((rho - r) / (0.22 * r)) ** 2)
            H = H + bowl + rim
            th = np.arctan2(Y, X); rays = 1 + 0.5 * np.cos(rng.integers(5, 11) * th + rng.uniform(0, 6)) ** 8
            ej = np.clip(1 - (rho - r) / (2.2 * r), 0, 1) * (rho >= 0.8 * r)
            G = G * (1 + alb_k * 0.35 * ej * rays)
            continue
        if kind == "poly":
            n, r, h, alb = p; u, v = local(); m = _poly_mask(u, v, n, r * scale); z = np.where(m, h * scale, 0)
        elif kind == "cyl":
            r, h, alb, *d = p; u, v = local(*(d or [0, 0])); m = np.hypot(u, v) <= r * scale; z = np.where(m, h * scale, 0)
        elif kind == "box":
            w, l, h, alb, *d = p; dx, dy, r2 = (d + [0, 0, 0])[:3]
            u, v = local(dx, dy, math.radians(r2)); m = (np.abs(u) <= w * scale / 2) & (np.abs(v) <= l * scale / 2)
            z = np.where(m, h * scale, 0)
        elif kind == "dome":
            r, h, alb = p; rho = np.hypot(*local()); m = rho <= r * scale
            z = np.where(m, h * scale * np.sqrt(np.clip(1 - (rho / (r * scale)) ** 2, 0, 1)), 0)
        elif kind == "cone":
            r0, r1, z0, z1, alb = p; rho = np.hypot(*local()); m = rho <= r0 * scale
            t = np.clip((r0 * scale - rho) / max((r0 - r1) * scale, 1e-6), 0, 1)
            z = np.where(m, (z0 + (z1 - z0) * t) * scale, 0)
        elif kind == "hcyl":
            r, length, alb = p; u, v = local(); m = (np.abs(v) <= length * scale / 2) & (np.abs(u) <= r * scale)
            z = np.where(m, r * scale + np.sqrt(np.clip((r * scale) ** 2 - u ** 2, 0, None)), 0)
        elif kind == "legs":
            n, r, h, alb = p; z = np.zeros_like(X); m = np.zeros(X.shape, bool)
            for k in range(int(n)):
                a = rot + 2 * np.pi * k / n + np.pi / n
                fx, fy = r * scale * math.cos(a), r * scale * math.sin(a)
                ax, ay = 0.45 * fx, 0.45 * fy
                t = np.clip(((X - fx) * (ax - fx) + (Y - fy) * (ay - fy)) / ((ax - fx) ** 2 + (ay - fy) ** 2 + 1e-9), 0, 1)
                d = np.hypot(X - fx - t * (ax - fx), Y - fy - t * (ay - fy))
                mk = d <= 0.12 * scale + 0.25 * (np.hypot(X - fx, Y - fy) <= 0.3 * scale)    # strut + footpad
                z = np.maximum(z, np.where(mk, h * scale * t, 0)); m |= mk
        elif kind == "disk":
            r, h, alb, dx, dy = p; u, v = local(dx, dy); m = np.hypot(u, v) <= r * scale
            z = np.where(m, h * scale, 0)
        elif kind == "canopy":                                 # irregular outline, fold ridges, albedo streaks
            r, alb, dx, dy = p; u, v = local(dx, dy); th = np.arctan2(v, u); rho = np.hypot(u, v)
            edge = 1 + sum(rng.uniform(0.04, 0.14) * np.cos(k * th + rng.uniform(0, 6)) for k in range(2, 7))
            m = rho <= r * scale * edge * rng.uniform(0.6, 1.0)
            folds = np.cos(rng.integers(6, 14) * th + 0.15 * rho) ** 2
            z = np.where(m, 0.25 * scale * (0.4 + folds), 0)
            alb = np.where(m, alb * (0.75 + 0.5 * folds), 0)
        top = m & (z >= H)
        A = np.where(top, alb * alb_k, A); H = np.maximum(H, z)
    return H, A, G


def cast_shadow(H, cell, toward_sun_deg, incidence):
    """Boolean shadow of the height field H (cell = metres per cell) for the sun at `toward_sun_deg` (image deg),
    `incidence` from vertical: resampled along the sun direction, running horizon maximum, sampled back."""
    ny, nx = H.shape
    t = math.radians(toward_sun_deg); sx, sy = math.cos(t), math.sin(t)
    L = int(math.hypot(ny, nx)) + 4; u = np.arange(L) - L / 2
    U, V = np.meshgrid(u, u)                                   # U along the sun direction, V across
    cx, cy = (nx - 1) / 2, (ny - 1) / 2
    Hr = ndimage.map_coordinates(H, [cy + U * sy + V * sx, cx + U * sx - V * sy], order=1, cval=0.0)
    drop = cell * math.tan(math.radians(90 - min(incidence, MAX_INCIDENCE)))   # horizon fall per cell
    M = np.full(L, -np.inf, np.float32); sh = np.zeros_like(Hr, bool)
    for j in range(L - 2, -1, -1):                             # walk away from the sun; M = highest horizon so far
        M = np.maximum(M, Hr[:, j + 1]) - drop
        sh[:, j] = M > Hr[:, j] + 1e-3
    yy, xx = np.mgrid[0:ny, 0:nx]
    du, dv = (xx - cx) * sx + (yy - cy) * sy, -(xx - cx) * sy + (yy - cy) * sx
    return ndimage.map_coordinates(sh.astype(np.float32), [dv + L / 2, du + L / 2], order=1) > 0.5


def height_m(spec):
    """Tallest point of the object in metres (crater: its rim)."""
    z = {"poly": lambda p: p[2], "cyl": lambda p: p[1], "box": lambda p: p[2], "dome": lambda p: p[1], "cone": lambda p: p[3],
         "hcyl": lambda p: 2 * p[0], "legs": lambda p: p[2], "disk": lambda p: p[1], "canopy": lambda p: 0.35,
         "crater": lambda p: 0.05 * p[0]}
    return max(z[sh[0]](sh[1:]) for sh in spec["shapes"])


def render(spec, bg, res, cx, cy, sun_az, incidence, body, rng, scale, alb_k):
    """Paste a rendered object at (cx, cy) of the background `bg` (float, DN). Returns (image, mask, info).
    Rendered on a FINE x oversampled window around the object, padded by its longest possible shadow."""
    cell = res / FINE
    rot = rng.uniform(0, 2 * np.pi)
    reach = footprint_m(spec, scale) * (1.7 if spec["cls"] == "impact-crater" else 0.5) + 1.0
    shadow = height_m(spec) * scale * math.tan(math.radians(min(incidence, MAX_INCIDENCE)))
    pad = int(math.ceil((reach + shadow) / res)) + 2
    x0, x1 = max(0, int(cx) - pad), min(SIZE, int(cx) + pad + 1)
    y0, y1 = max(0, int(cy) - pad), min(SIZE, int(cy) + pad + 1)
    fx = (x0 - 0.5 + (np.arange((x1 - x0) * FINE) + 0.5) / FINE - cx) * res      # cell centres, metres from the object
    fy = (y0 - 0.5 + (np.arange((y1 - y0) * FINE) + 0.5) / FINE - cy) * res
    X, Y = np.meshgrid(fx, fy)
    H, A, G = heightfield(spec, X, Y, rot, scale, alb_k, rng)
    obj = A > 0
    # Lambert shading relative to flat ground, cast shadows on ground and object
    gy, gx = np.gradient(H, cell)
    i, t = math.radians(min(incidence, MAX_INCIDENCE)), math.radians(sun_az)
    sv = np.array([math.sin(i) * math.cos(t), math.sin(i) * math.sin(t), math.cos(i)])
    nz = 1 / np.sqrt(gx ** 2 + gy ** 2 + 1)
    lam = np.clip((-gx * sv[0] - gy * sv[1] + sv[2]) * nz, 0, None) / max(math.cos(i), 0.05)
    lam = np.where(obj & (gx ** 2 + gy ** 2 > 7.5), 1.0, lam)   # vertical walls are not seen from orbit: take the top
    sh = cast_shadow(H, cell, sun_az, incidence)
    amb = rng.uniform(*SHADOW_AMBIENT[body])
    lit = np.where(sh, amb, np.minimum(lam, 6.0))
    wy, wx = y1 - y0, x1 - x0
    down = lambda a: np.asarray(a, np.float32).reshape(wy, FINE, wx, FINE).mean(axis=(1, 3))
    win = (slice(y0, y1), slice(x0, x1))
    cov, gmean, orad = np.zeros((SIZE, SIZE), np.float32), np.ones((SIZE, SIZE), np.float32), np.zeros((SIZE, SIZE), np.float32)
    shc, bowl, ejecta = (np.zeros((SIZE, SIZE), np.float32) for _ in range(3))
    cov[win] = down(obj)
    gsum = down(np.where(obj, 0.0, lit * G))               # ground keeps its texture: crater bowls, ejecta, shadows
    gmean[win] = np.where(cov[win] < 0.999, gsum / np.maximum(1 - cov[win], 1e-6), 1.0)
    orad[win] = down(np.where(obj, A * lit, 0.0))
    shc[win] = down(sh & ~obj); bowl[win] = down((H < -0.02) & ~obj); ejecta[win] = down((np.abs(G - 1) > 0.02) & ~obj)
    dark = float(np.percentile(bg, 0.5)); level = ndimage.uniform_filter(bg, 41)
    ground = dark + (bg - dark) * gmean
    obj_dn = dark + (level - dark) * orad / np.maximum(cov, 1e-6)
    out = ground * (1 - cov) + obj_dn * cov
    mask = np.zeros((SIZE, SIZE), np.uint8)
    mask[ejecta > 0.3] = 3; mask[shc > 0.3] = 2; mask[(cov > 0.3) | (bowl > 0.3)] = 1
    hm = float(H.max()); wm = float(2 * np.hypot(X[obj], Y[obj]).max()) if obj.any() else float(2 * np.hypot(X, Y)[H < -1e-3].max(initial=0))
    return out, mask, dict(width_m=round(wm, 2), height_m=round(hm, 2), rotation_deg=round(math.degrees(rot), 1))


# ── context traces ────────────────────────────────────────────────────────────────────────────────────────────
def _path(rng, x, y, n, step, heading, wiggle):
    pts = [(x, y)]
    for _ in range(n):
        heading += rng.normal(0, wiggle); x += step * math.cos(heading); y += step * math.sin(heading); pts.append((x, y))
    return pts


def footprint_m(spec, scale=1.0):
    """Diameter of the object's footprint in metres (craters: the crater)."""
    r = 0.0
    for sh in spec["shapes"]:
        k, p = sh[0], sh[1:]
        if k == "crater":
            return p[0] * scale
        if k in ("poly", "cyl", "dome"):
            ext = p[1] if k == "poly" else p[0]; off = p[3:5] if k == "cyl" and len(p) > 3 else (0, 0)
        elif k == "cone":
            ext, off = p[0], (0, 0)
        elif k == "legs":
            ext, off = p[1], (0, 0)
        elif k == "disk":
            ext, off = p[0], p[3:5]
        elif k == "canopy":
            ext, off = p[0], p[2:4]
        elif k == "hcyl":
            ext, off = p[1] / 2, (0, 0)
        else:                                               # box
            ext, off = math.hypot(p[0], p[1]) / 2, (p[4:6] if len(p) > 5 else (0, 0))
        r = max(r, ext + math.hypot(*off))
    return 2 * r * scale


def context_layer(kind, spec, cx, cy, res, size_m, rng, body):
    """Multiplicative factor (SIZE x SIZE) for blast zone / tracks / footpaths, or None."""
    f = np.ones((SIZE, SIZE), np.float32)
    if kind == "blast-zone":
        R = size_m / 2 * rng.uniform(4, 9) / res
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]; rho = np.hypot(xx - cx, yy - cy)
        noise = ndimage.gaussian_filter(rng.normal(0, 1, (SIZE, SIZE)), max(R / 6, 1)); noise /= noise.std() + 1e-6
        amp = rng.uniform(0.04, 0.12) if body == "Moon" else rng.uniform(-0.15, 0.1)
        return f * (1 + amp * np.clip(1 - rho / (R * (1 + 0.25 * noise)), 0, 1) ** 0.7)
    up = 4; big = Image.new("F", (SIZE * up, SIZE * up), 0.0); d = ImageDraw.Draw(big)
    if kind == "tracks":
        gauge = spec["tracks"] / res * up; width = max(rng.uniform(0.45, 0.7) / res * up, 1)
        length = rng.uniform(15, 120) / res; step = 2.0
        pts = _path(rng, cx, cy, int(length / step), step, rng.uniform(0, 2 * np.pi), 0.06)
        for side in (-0.5, 0.5):
            line = []
            for k in range(len(pts) - 1):
                (x0, y0), (x1, y1) = pts[k], pts[k + 1]; hd = math.atan2(y1 - y0, x1 - x0)
                line.append((x0 * up - side * gauge * math.sin(hd), y0 * up + side * gauge * math.cos(hd)))
            d.line(line, fill=1.0, width=int(round(width)))
        depth = rng.uniform(0.12, 0.3) * (1 if body == "Moon" or rng.random() < 0.7 else -0.6)
    else:                                                   # footpaths: thin winding trails out of the lander
        width = max(0.6 / res * up, 1)
        for _ in range(rng.integers(1, 5)):
            pts = _path(rng, cx, cy, int(rng.uniform(20, 150) / res / 2), 2.0, rng.uniform(0, 2 * np.pi), 0.12)
            d.line([(x * up, y * up) for x, y in pts], fill=1.0, width=int(round(width)))
        depth = rng.uniform(0.08, 0.2)
    cov = np.asarray(big, np.float32).reshape(SIZE, up, SIZE, up).mean(axis=(1, 3))
    return f * (1 - depth * cov)


# ── real sprites ──────────────────────────────────────────────────────────────────────────────────────────────
def build_sprites():
    """Cut-outs of real hardware: contrast against the local ground (ratio - 1) and a soft alpha, with the source
    frame's m/px, incidence and sun azimuth."""
    sites = {r["site_id"]: r for r in csv.DictReader(SITES.open())}
    meta = {r["file_name"]: r for r in csv.DictReader((CROPS / "metadata.csv").open())}
    coco = json.load(COCO.open()); cat = {c["id"]: c["name"] for c in coco["categories"]}; ims = {i["id"]: i for i in coco["images"]}
    sprites, seen, sun_cache = [], set(), {}
    for a in sorted(coco["annotations"], key=lambda a: a["id"]):
        oid = cat[a["category_id"]]; f = ims[a["image_id"]]["file_name"]; m = meta[f]
        if a["confidence"] != "high" or oid not in sites or sites[oid]["object_type"] not in SPRITE_TYPES:
            continue
        x, y, w, h = a["bbox"]
        key = (m["frame_id"], oid, int(m["crop_x0"]) + x // 4, int(m["crop_y0"]) + y // 4)
        if key in seen or max(w, h) > 160:
            continue
        seen.add(key)
        img = np.asarray(Image.open(CROPS / f).convert("L"), np.float32)
        dark = float(np.percentile(img, 0.5))
        if f not in sun_cache:
            sun_cache[f] = sun_azimuth(img, m["body"])
        mg = max(4, round(0.35 * max(w, h)))
        x0, y0, x1, y1 = max(0, x - mg), max(0, y - mg), min(2048, x + w + mg), min(2048, y + h + mg)
        p = img[y0:y1, x0:x1]
        inside = np.zeros(p.shape, bool); inside[y - y0:y - y0 + h, x - x0:x - x0 + w] = True
        ring = ~ndimage.binary_dilation(inside, iterations=2)
        if ring.sum() < 20:
            continue
        yy, xx = np.nonzero(ring)                                # plane fit to the ring = ground under the object
        coef, *_ = np.linalg.lstsq(np.c_[np.ones(len(xx)), xx, yy], p[ring], rcond=None)
        gy_, gx_ = np.mgrid[0:p.shape[0], 0:p.shape[1]]
        ground = np.maximum(coef[0] + coef[1] * gx_ + coef[2] * gy_ - dark, 3.0)
        rr = (p - dark) / ground - 1
        nz = np.median(np.abs(rr[ring] - np.median(rr[ring]))) * 1.4826 + 1e-3
        core = (np.abs(rr) > 2.5 * nz) & ndimage.binary_dilation(inside, iterations=1)
        if core.sum() < 6 or np.abs(rr[core]).max() < 4 * nz:
            continue
        alpha = np.clip(ndimage.gaussian_filter(ndimage.binary_dilation(core).astype(np.float32), 0.7) * 1.3, 0, 1)
        sprites.append(dict(rr=rr.astype(np.float32), alpha=alpha.astype(np.float32), body=m["body"], oid=oid,
                            cls=SPRITE_TYPES[sites[oid]["object_type"]], res=float(m["resolution_m_px"]),
                            incidence=float(m["incidence_deg"]), sun=sun_cache[f][0], frame=m["frame_id"],
                            source=f"{f}#{x},{y},{w},{h}", name=sites[oid]["object"]))
    return sprites


def null_sprite(tiles, body, rng, w_px, res, incidence, sun_of):
    """A patch of natural terrain (another tile, similar sun incidence) with a blob-shaped alpha, as a stand-in
    object; it carries its tile's sun azimuth so paste() turns its shading the same way as the target's."""
    near = [t for t in tiles[body] if abs(t["incidence"] - incidence) <= 15] or tiles[body]
    t = near[rng.integers(len(near))]
    a = np.asarray(Image.open(t["path"]).convert("L"), np.float32); dark = float(np.percentile(a, 0.5))
    s = int(w_px * res / t["res"]) + 4                       # w_px object pixels in the target image
    y, x = rng.integers(16, a.shape[0] - s - 16), rng.integers(16, a.shape[1] - s - 16)
    p = a[y:y + s, x:x + s]; rr = (p - dark) / max(float(np.median(p)) - dark, 3.0) - 1
    yy, xx = np.mgrid[0:s, 0:s]; c = (s - 1) / 2
    blob = ((xx - c) / (s * rng.uniform(0.25, 0.45))) ** 2 + ((yy - c) / (s * rng.uniform(0.25, 0.45))) ** 2 <= 1
    alpha = np.clip(ndimage.gaussian_filter(blob.astype(np.float32), 0.8) * 1.3, 0, 1)
    return dict(rr=rr.astype(np.float32), alpha=alpha, res=t["res"], sun=sun_of(t)[0], oid="natural-terrain", cls="null-control",
                name="natural terrain patch", source=f"{t['rel']}#{x},{y},{s},{s}", incidence=t["incidence"])


def paste(bg, sp, cx, cy, res, sun_az, rng, contrast, mirror, extra_rot=0.0):
    """Paste sprite `sp` at (cx, cy): rotated so its sun azimuth becomes `sun_az`, scaled to `res`, optionally
    mirrored across the sun axis. Object pixels take the target ground level times (1 + contrast * rr)."""
    s = sp["res"] / res * rng.uniform(0.9, 1.1)
    d = math.radians(sun_az - sp["sun"] + extra_rot)
    a2 = math.radians(sun_az)
    h, w = sp["rr"].shape
    half = int(math.ceil(max(h, w) * s * 0.75)) + 2
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    if mirror:                                                 # reflect across the sun axis (keeps the sun direction)
        c2, s2 = math.cos(2 * a2), math.sin(2 * a2)
        xx, yy = c2 * xx + s2 * yy, s2 * xx - c2 * yy
    cd, sd = math.cos(-d), math.sin(-d)
    sx, sy = (cd * xx - sd * yy) / s + (w - 1) / 2, (sd * xx + cd * yy) / s + (h - 1) / 2
    rr = ndimage.map_coordinates(sp["rr"], [sy, sx], order=1, cval=0.0)
    al = ndimage.map_coordinates(sp["alpha"], [sy, sx], order=1, cval=0.0)
    out = bg.copy(); mask = np.zeros(bg.shape, np.uint8)
    y0, x0 = int(round(cy)) - half, int(round(cx)) - half
    ys, xs = slice(max(0, y0), min(SIZE, y0 + 2 * half + 1)), slice(max(0, x0), min(SIZE, x0 + 2 * half + 1))
    sub = (slice(ys.start - y0, ys.stop - y0), slice(xs.start - x0, xs.stop - x0))
    level = ndimage.uniform_filter(bg, 41)[ys, xs]; dark = float(np.percentile(bg, 0.5))
    rr, al = rr[sub], al[sub]
    obj = dark + (level - dark) * np.clip(1 + contrast * rr, 0.02, 8)
    out[ys, xs] = bg[ys, xs] * (1 - al) + obj * al
    m = np.zeros(rr.shape, np.uint8); m[(al > 0.3) & (rr > 0)] = 1; m[(al > 0.3) & (rr <= 0)] = 2
    mask[ys, xs] = m
    return out, mask, s, round(math.degrees(d), 1)


# ── one image ─────────────────────────────────────────────────────────────────────────────────────────────────
def load_tiles():
    tiles = {}
    for body, d in TILES.items():
        rows = list(csv.DictReader((d / "metadata.csv").open()))
        tiles[body] = [dict(rel=r["file_name"], path=d / r["file_name"], split=r["split"], landform=r["landform"], body=body,
                            frame=r["frame_id"], res=float(r["resolution_m_px"]), incidence=float(r["incidence_deg"] or 45))
                       for r in rows]
    return tiles


SHADOW_DN = 20                              # 8-bit stretched tiles: real shadows sit at 0-15


def background(t, rng, allow_coarser):
    """512x512 crop of tile t (on the Moon sometimes 1.5-2x larger and resampled: a coarser camera). Up to six
    draws to avoid crops that are mostly shadow (grazing sun), where no object could show."""
    a = np.asarray(Image.open(t["path"]).convert("L"), np.float32)
    H, W = a.shape
    f = 1.0
    if allow_coarser and min(H, W) >= 2 * SIZE:
        f = float(rng.choice([1.0, 1.0, 1.0, 1.5, 2.0]))
    side = int(SIZE * f)
    for _ in range(6):
        y0, x0 = int(rng.integers(0, H - side + 1)), int(rng.integers(0, W - side + 1))
        crop = a[y0:y0 + side, x0:x0 + side]
        if (crop < SHADOW_DN).mean() < 0.4:
            break
    if f != 1.0:
        crop = np.asarray(Image.fromarray(crop).resize((SIZE, SIZE), Image.BOX), np.float32)
    return crop, f, x0, y0, side


def noise_sigma(a):
    d = a[:, 1:] - a[:, :-1]
    return float(np.median(np.abs(d - np.median(d))) * 1.4826 / math.sqrt(2))


def make(item, tiles, sprites, sun_of, rng):
    t, body, kind = item["tile"], item["body"], item["kind"]
    bg, f, bx0, by0, side = background(t, rng, body == "Moon")
    res = t["res"] * f
    az, conf = sun_of(t)
    inc = t["incidence"]
    cx, cy = float(rng.uniform(96, SIZE - 96)), float(rng.uniform(96, SIZE - 96))
    info = dict(sprite_source="", rotation_deg="", mirrored="", contrast_factor="", albedo_factor="", context="none",
                scale_factor="", width_m="", height_m="")
    work = bg.copy()
    if kind == "rendered":
        hid = item["hardware"]; spec = HARDWARE[hid]
        scale, alb_k = rng.uniform(0.85, 1.2), rng.uniform(0.6, 1.4)
        size_guess = footprint_m(spec, scale)
        ctx = "none"
        if spec["cls"] == "lander" and rng.random() < 0.5:
            ctx = "blast-zone"
        if spec.get("tracks") and rng.random() < 0.7:
            ctx = "tracks"
        if spec.get("context") == "footpaths" and rng.random() < 0.6:
            ctx = "footpaths"
        if ctx != "none":
            work = work * context_layer(ctx, spec, cx, cy, res, size_guess, rng, body)
        out, mask, r = render(spec, work, res, cx, cy, az, inc, body, rng, scale, alb_k)
        if ctx != "none":
            mask[(mask == 0) & (np.abs(work - bg) > 2)] = 3
        info.update(r, scale_factor=round(scale, 3), albedo_factor=round(alb_k, 3), context=ctx)
        era, cls, name, hw = spec["era"], spec["cls"], spec["name"], hid
    else:
        if kind == "real-sprite":
            cands = [s for s in sprites if s["body"] == body and abs(s["incidence"] - inc) <= 15 and s["res"] <= res * 1.25]
            sp = cands[rng.integers(len(cands))]
            era, cls, name, hw = "past", sp["cls"], sp["name"], sp["oid"]
        else:
            sp = null_sprite(tiles, body, rng, rng.uniform(4, 30), res, inc, sun_of)     # sized like the real objects
            era, cls, name, hw = "", "null-control", sp["name"], "natural-terrain"
        contrast, mirror = rng.uniform(0.8, 1.2), bool(rng.random() < 0.5)
        out, mask, s, rot = paste(work, sp, cx, cy, res, az, rng, contrast, mirror)
        info.update(sprite_source=sp["source"], rotation_deg=rot, mirrored="yes" if mirror else "no",
                    contrast_factor=round(contrast, 3), scale_factor=round(s, 3))
    # camera: PSF on what was added, the tile's noise on the object, gain / gamma on both twins
    psf = rng.uniform(0.35, 0.9); diff = ndimage.gaussian_filter(out - bg, psf)
    sig = noise_sigma(bg); cover = ndimage.gaussian_filter((mask == 1).astype(np.float32), 0.5)
    out = bg + diff + rng.normal(0, sig, bg.shape) * cover
    gain, gamma = rng.uniform(0.9, 1.1), rng.uniform(0.85, 1.15)
    cam = lambda a: np.clip(255 * (np.clip(a, 0, 255) / 255) ** gamma * gain, 0, 255).astype(np.uint8)
    ys, xs = np.nonzero((mask == 1) | (mask == 2))
    if len(xs) == 0:
        return None
    oy, ox = np.nonzero(mask == 1)
    obj_px = int(max(np.ptp(ox), np.ptp(oy)) + 1) if len(ox) else 0
    if kind != "rendered":
        info["width_m"] = round(obj_px * res, 2)
    row = dict(split=t["split"], body=body, instrument=CAMERA[body], kind=kind, era=era, object_class=cls,
               hardware_id=hw, hardware_name=name, x_center=round(cx, 1), y_center=round(cy, 1),
               box_x_min=int(xs.min()), box_y_min=int(ys.min()), box_x_max=int(xs.max()), box_y_max=int(ys.max()),
               object_px=obj_px, shadow_px=int((mask == 2).sum()), resolution_m_px=round(res, 4),
               camera_downscale=f, incidence_deg=round(inc, 2), sun_azimuth_image_deg=round(az, 1),
               sun_azimuth_confidence=round(conf, 3), psf_sigma_px=round(psf, 2), noise_sigma=round(sig, 2),
               gain=round(gain, 3), gamma=round(gamma, 3), landform=t["landform"], background_file=t["rel"],
               background_x0=bx0, background_y0=by0, background_side_px=side, frame_id=t["frame"], seed=item["seed"],
               shadowed_fraction=round(float((bg < SHADOW_DN).mean()), 3), **info)
    return cam(out), cam(bg), mask, row


# ── plan, run, sheets ─────────────────────────────────────────────────────────────────────────────────────────
def plan(tiles, sprites, per_tile, sprite_frac, null_frac, seed, val_extra=2):
    """Per tile: `per_tile` renders (every design about equally often), a real sprite with probability
    `sprite_frac`, a null control with probability `null_frac`. Validation tiles (never in Mk18's memory) get
    `val_extra` more renders and always a sprite and a null control: the pool to evaluate on."""
    rng = np.random.default_rng(seed); items = []
    for body, ts in tiles.items():
        hw = [k for k, v in HARDWARE.items() if v["body"] in (body, "both")]
        bal = Counter()
        for t in ts:
            val = t["split"] == "validation"
            for _ in range(per_tile + (val_extra if val else 0)):
                h = min(hw, key=lambda k: (bal[k], rng.random())); bal[h] += 1      # every design about equally often
                items.append(dict(body=body, tile=t, kind="rendered", hardware=h))
            ok = [s for s in sprites if s["body"] == body and abs(s["incidence"] - t["incidence"]) <= 15 and s["res"] <= t["res"] * 1.25]
            if ok and (val or rng.random() < sprite_frac):
                items.append(dict(body=body, tile=t, kind="real-sprite"))
            if val or rng.random() < null_frac:
                items.append(dict(body=body, tile=t, kind="null-control"))
    for k, it in enumerate(items):
        it["seed"] = seed * 100003 + k
    return items


def run(per_tile=1, sprite_frac=0.6, null_frac=0.25, limit=None, seed=7, val_extra=2):
    tiles = load_tiles()
    print("cutting real sprites ...", flush=True)
    sprites = build_sprites()
    print(f"  {len(sprites)} sprites: {dict(Counter((s['body'], s['cls']) for s in sprites))}")
    sun_cache = {}
    def sun_of(t):
        if t["rel"] not in sun_cache:
            sun_cache[t["rel"]] = sun_azimuth(np.asarray(Image.open(t["path"]).convert("L"), np.float32), t["body"])
        return sun_cache[t["rel"]]
    items = plan(tiles, sprites, per_tile, sprite_frac, null_frac, seed, val_extra)
    if limit:
        items = items[:: max(1, len(items) // limit)][:limit]
    for sub in ("injected", "clean", "masks"):
        for old in (OUT / sub).glob("*/*/*.png"):
            old.unlink()
    rows, coco = [], {"info": {"description": "Xenarch synthetic injections: one space-hardware object pasted into natural terrain (LROC NAC, HiRISE)",
                               "version": "1.0"}, "licenses": [{"id": 1, "name": "CC0-1.0"}], "images": [], "annotations": [], "categories": []}
    cats = {}
    for k, it in enumerate(items):
        rng = np.random.default_rng(it["seed"])
        r = make(it, tiles, sprites, sun_of, rng)
        if r is None:
            continue
        inj, clean, mask, row = r
        b, kind = row["body"].lower(), row["kind"]
        stem = f"{b}_{kind}_{row['hardware_id']}_{k:05d}"
        for sub, arr in (("injected", inj), ("clean", clean), ("masks", mask)):
            p = OUT / sub / b / kind / f"{stem}.png"; p.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(p, optimize=True)
        row.update(file_name=f"injected/{b}/{kind}/{stem}.png", clean_file=f"clean/{b}/{kind}/{stem}.png", mask_file=f"masks/{b}/{kind}/{stem}.png")
        rows.append(row)
        c = cats.setdefault(row["object_class"], len(cats) + 1)
        coco["images"].append({"id": len(rows), "file_name": row["file_name"], "width": SIZE, "height": SIZE, "license": 1,
                               **{key: row[key] for key in ("body", "kind", "era", "split", "resolution_m_px", "incidence_deg")}})
        x0, y0, x1, y1 = row["box_x_min"], row["box_y_min"], row["box_x_max"], row["box_y_max"]
        coco["annotations"].append({"id": len(rows), "image_id": len(rows), "category_id": c, "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                                    "area": (x1 - x0 + 1) * (y1 - y0 + 1), "iscrowd": 0, "includes_shadow": True,
                                    "hardware_id": row["hardware_id"], "context": row["context"]})
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(items)}", flush=True)
    coco["categories"] = [{"id": v, "name": k, "supercategory": "synthetic"} for k, v in cats.items()]
    with (OUT / "metadata.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    json.dump(coco, (OUT / "boxes_coco.json").open("w"), indent=1)
    sheets(rows)
    n = Counter((r["body"], r["kind"]) for r in rows)
    print(f"{len(rows)} injected images in {OUT}: " + ", ".join(f"{b} {k} {v}" for (b, k), v in sorted(n.items())))


def _font(size):
    for f in ("/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def sheets(rows, t=200, cols=4, per=16):
    """contact-sheets/<body>_<kind>.jpg: clean | injected pairs with the box outlined, and overview.jpg."""
    d = OUT / "contact-sheets"; d.mkdir(parents=True, exist_ok=True)
    navy, orange, muted = (12, 22, 40), (240, 136, 78), (140, 150, 170)
    f12, f11 = _font(13), _font(11)
    by = defaultdict(list)
    for r in rows:
        by[(r["body"], r["kind"])].append(r)
    def pair(r):
        a = Image.open(OUT / r["clean_file"]).convert("RGB").resize((t, t), Image.LANCZOS)
        b = Image.open(OUT / r["file_name"]).convert("RGB").resize((t, t), Image.LANCZOS)
        k = t / SIZE; ImageDraw.Draw(b).rectangle([r["box_x_min"] * k - 3, r["box_y_min"] * k - 3, r["box_x_max"] * k + 3, r["box_y_max"] * k + 3], outline=orange, width=1)
        return a, b
    for (body, kind), rs in by.items():
        rs = sorted(rs, key=lambda r: r["hardware_id"])[:: max(1, len(rs) // per)][:per]
        nr = math.ceil(len(rs) / cols)
        im = Image.new("RGB", (cols * (2 * t + 18) + 8, 36 + nr * (t + 34)), navy); dr = ImageDraw.Draw(im)
        dr.text((8, 10), f"{body.upper()} · {kind.upper()} · CLEAN | INJECTED", fill=orange, font=f12)
        for i, r in enumerate(rs):
            x, y = 8 + (i % cols) * (2 * t + 18), 36 + (i // cols) * (t + 34)
            a, b = pair(r); im.paste(a, (x, y)); im.paste(b, (x + t + 2, y))
            dr.text((x, y + t + 4), f"{r['hardware_id'][:28]} · {r['resolution_m_px']} m/px · i={r['incidence_deg']:.0f}", fill=muted, font=f11)
            dr.text((x, y + t + 18), f"context {r['context']} · scale {r['scale_factor']}", fill=muted, font=f11)
        im.save(d / f"{body.lower()}_{kind}.jpg", quality=88)
    pick = []
    for (body, kind), rs in sorted(by.items()):
        pick += sorted(rs, key=lambda r: r["hardware_id"])[:: max(1, len(rs) // 4)][:4]
    nr = math.ceil(len(pick) / cols)
    im = Image.new("RGB", (cols * (2 * t + 18) + 8, 36 + nr * (t + 34)), navy); dr = ImageDraw.Draw(im)
    dr.text((8, 10), "XENARCH SYNTHETIC INJECTIONS · CLEAN | INJECTED", fill=orange, font=f12)
    for i, r in enumerate(pick):
        x, y = 8 + (i % cols) * (2 * t + 18), 36 + (i // cols) * (t + 34)
        a, b = pair(r); im.paste(a, (x, y)); im.paste(b, (x + t + 2, y))
        dr.text((x, y + t + 4), f"{r['body']} · {r['kind']} · {r['hardware_id'][:22]}", fill=muted, font=f11)
        dr.text((x, y + t + 18), f"{r['resolution_m_px']} m/px · i={r['incidence_deg']:.0f} · {r['context']}", fill=muted, font=f11)
    im.save(d / "overview.jpg", quality=88)


def _arg(name, default, cast=str):
    return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


if __name__ == "__main__":
    run(per_tile=_arg("--per-tile", 1, int), sprite_frac=_arg("--sprite-frac", 0.6, float),
        null_frac=_arg("--null-frac", 0.25, float), limit=_arg("--limit", None, int), seed=_arg("--seed", 7, int),
        val_extra=_arg("--val-extra", 2, int))

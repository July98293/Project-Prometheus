"""
Technosignature Detection: Xenarch Mk18 - PatchCore Edition
============================================================

What changed from Mk15:
✓ No VAE. Mk15's VAE rebuilt a blur of every chip, so its error was high on all natural
  texture and scored no better than a plain Gaussian blur (AUROC ≈ 0.51-0.56).
✓ PatchCore scoring: a pretrained ResNet18 describes every 8×8 patch (layer2 + layer3
  features); a memory bank keeps descriptions of natural terrain; a patch's anomaly score is
  its distance to the nearest remembered patch. Nothing is trained.
✓ Per-patch heatmap: anomalies are located inside the chip, not just flagged per chip.
✓ Held-out validation on whole images never seen by the memory bank, with two object
  strengths (subtle, as Mk15; obvious lander-sized blocks) + localisation check.
✓ Confidence = share of natural validation chips that score lower (a calibrated percentile).

Usage:
    python xenarch_mk18_script.py [--data DIR] [--test DIR] [--bank 30000] [--per-chip 24] [--hand 0]
                                  [--scene-z 6] [--reuse]

    --reuse   rescan with the memory bank saved in data/models (skips building it)
    --scene-z flag a hotspot when it is this many robust σ above the rest of its own image

    --hand sets how much the hand-crafted patch features (gradient, edge sharpness, contrast,
    edge straightness) count next to the CNN features; 0 turns them off.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger

from xenarch_live import LiveRun

Image.MAX_IMAGE_PIXELS = None
CHIP = 256
PATCH = 8                      # layer2 stride: one feature per 8×8 pixels -> 32×32 map per chip
EDGE = 2                       # patches at a chip's border see zero padding, not terrain: ignore them
EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SEED = 7


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ============================================
# 1. IMAGES -> CHIPS
# ============================================

def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def normalize(chip):
    """Robust [0, 1] stretch, same as Mk15's LunarDataset."""
    lo, hi = np.percentile(chip, [1, 99])
    return np.clip((chip - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)


def tile(arr, full=False, stride=CHIP):
    """Yield (y, x, chip). full=True also covers the right/bottom edge."""
    h, w = arr.shape
    if h < CHIP or w < CHIP:
        return
    ys, xs = list(range(0, h - CHIP + 1, stride)), list(range(0, w - CHIP + 1, stride))
    if full:
        ys += [h - CHIP] if ys[-1] != h - CHIP else []
        xs += [w - CHIP] if xs[-1] != w - CHIP else []
    for y in ys:
        for x in xs:
            yield y, x, arr[y:y + CHIP, x:x + CHIP]


def natural_chips(paths, on_image=None):
    """Normalized chips from natural-terrain images, skipping flat ones (as Mk15)."""
    for i, p in enumerate(paths, 1):
        if on_image:
            on_image(i, len(paths))
        try:
            arr = load_gray(p)
        except Exception as e:
            logger.warning(f"skip {p.name}: {e}")
            continue
        for _, _, c in tile(arr):
            if c.std() >= 5:
                yield normalize(c)


# ============================================
# 2. SYNTHETIC OBJECTS (validation only)
# ============================================

def inject_subtle(chip, rng):
    """Mk15's test object: faint rectangle / 3-px ring / 2-px line, 14-40 px, ±0.15-0.35."""
    h, w = chip.shape
    size = int(rng.integers(14, 40))
    cy, cx = int(rng.integers(size, h - size)), int(rng.integers(size, w - size))
    delta = float(rng.uniform(0.15, 0.35) * rng.choice([-1, 1]))
    yy, xx = np.ogrid[:h, :w]
    kind = int(rng.integers(3))
    if kind == 0:
        mask = (np.abs(yy - cy) <= size // 2) & (np.abs(xx - cx) <= size // 3)
    elif kind == 1:
        mask = np.abs(np.hypot(yy - cy, xx - cx) - size / 2) <= 1.5
    else:
        th = rng.uniform(0, np.pi)
        along = (xx - cx) * np.cos(th) + (yy - cy) * np.sin(th)
        across = -(xx - cx) * np.sin(th) + (yy - cy) * np.cos(th)
        mask = (np.abs(across) <= 1.0) & (np.abs(along) <= size)
    out = chip.copy()
    out[mask] = np.clip(out[mask] + delta, 0, 1)
    return out, mask


def inject_obvious(chip, rng):
    """Lander-sized solid block, 30-60 px, fully bright (sunlit) or fully dark (shadow)."""
    s = int(rng.integers(30, 60))
    cy, cx = (int(v) for v in rng.integers(s, CHIP - s, 2))
    mask = np.zeros_like(chip, dtype=bool)
    mask[cy - s // 2:cy + s // 2, cx - s // 3:cx + s // 3] = True
    out = chip.copy()
    out[mask] = float(rng.choice([0.0, 1.0]))
    return out, mask


# ============================================
# 3. HAND-CRAFTED PATCH FEATURES (Mk15's gradient / contrast / sharp-edge ideas)
# ============================================

HAND_NAMES = ("gradient", "edge_sharpness", "contrast", "edge_straightness")


def handcrafted(x):
    """(B, 1, 256, 256) -> (B, 4, 32, 32), one value per 8×8 patch:
    gradient = mean Sobel magnitude · edge_sharpness = max Sobel magnitude ·
    contrast = local std · edge_straightness = structure-tensor coherence (1 = one straight edge
    direction, as on a lander deck or a track; ~0 = texture pointing every way, as on regolith)."""
    k = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device) / 8
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx, gy = F.conv2d(xp, k.view(1, 1, 3, 3)), F.conv2d(xp, k.t().reshape(1, 1, 3, 3))
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
    avg = lambda t: F.avg_pool2d(t, PATCH)
    jxx, jyy, jxy = avg(gx * gx), avg(gy * gy), avg(gx * gy)
    return torch.cat([
        avg(mag),
        F.max_pool2d(mag, PATCH),
        torch.sqrt(torch.clamp(avg(x * x) - avg(x) ** 2, min=1e-8)),
        torch.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-8),
    ], 1)


# ============================================
# 4. PATCHCORE SCORER
# ============================================

class PatchCoreScorer:
    """Memory of natural-terrain patch features; anomaly = distance to the nearest known patch."""

    def __init__(self, device, bank_size=30000, per_chip=24, proj_dim=128, hand_weight=0.0):
        self.device, self.bank_size, self.per_chip, self.proj_dim = device, bank_size, per_chip, proj_dim
        self.hand_weight = hand_weight          # 0 = CNN features only
        self.hand_mu = self.hand_sd = None
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        self.net = torchvision.models.resnet18(weights=weights).to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.bank = None
        self.pool_size = 0
        self.live = None                        # optional LiveRun for coreset progress

    @torch.no_grad()
    def describe(self, x):
        """Patch descriptors used for the nearest-neighbour search: CNN + weighted hand-crafted z-scores."""
        x = x.to(self.device)
        f = self.cnn_features(x)
        return torch.cat([f, self.hand_weight * self.hand_z(x)], 1) if self.hand_weight else f

    @torch.no_grad()
    def hand_raw(self, x):
        return F.avg_pool2d(handcrafted(x.to(self.device)), 3, stride=1, padding=1)

    @torch.no_grad()
    def hand_z(self, x):
        """(B, 4, 32, 32) hand-crafted features as z-scores against natural terrain."""
        return (self.hand_raw(x) - self.hand_mu) / self.hand_sd

    @torch.no_grad()
    def cnn_features(self, x):
        """(B, 1, 256, 256) in [0, 1] -> (B, 384, 32, 32): layer2 + upsampled layer3, 3×3 local average."""
        x = (x.to(self.device).repeat(1, 3, 1, 1) - self.mean) / self.std
        n = self.net
        x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        l2 = n.layer2(n.layer1(x))
        l3 = n.layer3(l2)
        f = torch.cat([l2, F.interpolate(l3, size=l2.shape[-2:], mode="bilinear", align_corners=False)], 1)
        return F.avg_pool2d(f, 3, stride=1, padding=1)

    def fit(self, chips, batch=32):
        """Sample `per_chip` patches from every natural chip, then keep a greedy coreset of `bank_size`."""
        g = torch.Generator().manual_seed(SEED)
        pool, buf, n_chips = [], [], 0
        for c in chips:
            buf.append(c)
            if len(buf) == batch:
                pool.append(self._sample(buf, g)); n_chips += len(buf); buf = []
        if buf:
            pool.append(self._sample(buf, g)); n_chips += len(buf)
        if not pool:
            raise SystemExit("Mk18: no natural chips to build the memory bank from")
        pool = torch.cat(pool)
        cnn, hand = pool[:, :-len(HAND_NAMES)], pool[:, -len(HAND_NAMES):]
        self.hand_mu = hand.mean(0).view(1, -1, 1, 1).to(self.device)
        self.hand_sd = (hand.std(0) + 1e-6).view(1, -1, 1, 1).to(self.device)
        pool = cnn
        if self.hand_weight:
            z = (hand - self.hand_mu.view(1, -1).cpu()) / self.hand_sd.view(1, -1).cpu()
            pool = torch.cat([cnn, self.hand_weight * z], 1)
        self.pool_size = len(pool)
        logger.info(f"   {n_chips} chips -> {len(pool):,} candidate patches -> coreset of {min(self.bank_size, len(pool)):,}")
        if self.live:
            self.live.step("Building memory bank")
            self.live.log(f"{n_chips} chips -> {len(pool):,} candidate patches")
        self.bank = self._coreset(pool.to(self.device), g)
        return n_chips

    def _sample(self, buf, g):
        x = torch.from_numpy(np.stack(buf)[:, None]).to(self.device)
        f = torch.cat([self.cnn_features(x), self.hand_raw(x)], 1)
        f = f.flatten(2).transpose(1, 2).cpu()                         # (B, 1024, 384 + 4)
        idx = torch.randint(0, f.shape[1], (f.shape[0], self.per_chip), generator=g)
        return torch.gather(f, 1, idx[..., None].expand(-1, -1, f.shape[2])).reshape(-1, f.shape[2])

    def _coreset(self, pool, g):
        """Greedy k-center on a random projection: keeps rare terrain types, not just the common ones."""
        if len(pool) <= self.bank_size:
            return pool
        proj = torch.randn(pool.shape[1], self.proj_dim, generator=g).to(self.device) / self.proj_dim ** 0.5
        p = pool @ proj
        sel = [torch.tensor(0, device=self.device)]
        mind = torch.linalg.norm(p - p[0], dim=1)
        for k in range(self.bank_size - 1):
            if self.live and k % 500 == 0:
                self.live.progress(k, self.bank_size, "patches kept")
            i = torch.argmax(mind)
            sel.append(i)
            mind = torch.minimum(mind, torch.linalg.norm(p - p[i], dim=1))
        return pool[torch.stack(sel)]

    @torch.no_grad()
    def score_maps(self, chips, batch=16, bank=None):
        """List of (256, 256) chips -> (N, 32, 32) nearest-neighbour distance per patch.
        `bank` scores against a subset of the memory (the coreset's first k patches are itself a k-coreset)."""
        bank = self.bank if bank is None else bank
        out = []
        for i in range(0, len(chips), batch):
            f = self.describe(torch.from_numpy(np.stack(chips[i:i + batch])[:, None]))
            q = f.flatten(2).transpose(1, 2)
            d = torch.stack([torch.cdist(qi, bank).min(1).values for qi in q])
            out.append(d.view(-1, CHIP // PATCH, CHIP // PATCH).cpu())
        return torch.cat(out).numpy()


def interior(maps):
    """Drop the border patches of (N, 32, 32) maps."""
    return maps[..., EDGE:-EDGE, EDGE:-EDGE]


# ============================================
# 5. VALIDATION
# ============================================

KINDS = (("subtle", inject_subtle), ("obvious", inject_obvious))


def _score_sets(scorer, val_chips, bank=None):
    """Chip-level max scores for clean val chips and for each object strength (same objects every call)."""
    rng = np.random.default_rng(SEED)
    clean_maps = interior(scorer.score_maps(val_chips, bank=bank))
    sets = {"clean_maps": clean_maps, "clean": clean_maps.max(axis=(1, 2))}
    for kind, inject in KINDS:
        doctored, masks = zip(*(inject(c, rng) for c in val_chips))
        sets[kind] = (interior(scorer.score_maps(list(doctored), bank=bank)), masks)
    return sets


def _auroc(sets, kind):
    s = sets[kind][0].max(axis=(1, 2))
    return float(roc_auc_score(np.r_[np.zeros(len(sets["clean"])), np.ones(len(s))], np.r_[sets["clean"], s]))


def validate(scorer, val_chips, curve_ks=(1000, 2000, 5000, 10000, 20000), live=None):
    """AUROC / detection / localisation on held-out natural chips, clean vs with a synthetic object,
    plus the same AUROC as the memory grows (Mk18's counterpart of a loss curve)."""
    full = _score_sets(scorer, val_chips)
    clean = full["clean"]
    threshold = float(np.percentile(clean, 95))              # flags 5% of natural chips by design
    out = {"threshold": threshold, "val_chips": len(val_chips)}
    for kind, _ in KINDS:
        maps, masks = full[kind]
        s = maps.max(axis=(1, 2))
        hits = []
        for m, mask in zip(maps, masks):                      # is the hottest patch on the object?
            py, px = np.unravel_index(np.argmax(m), m.shape)
            ys, xs = np.where(mask)
            cy, cx = (py + EDGE) * PATCH + PATCH // 2, (px + EDGE) * PATCH + PATCH // 2
            hits.append(ys.min() - PATCH <= cy <= ys.max() + PATCH and xs.min() - PATCH <= cx <= xs.max() + PATCH)
        out[f"auroc_{kind}"] = _auroc(full, kind)
        out[f"detect_{kind}"] = float((s > threshold).mean())
        out[f"locate_{kind}"] = float(np.mean(hits))
        logger.info(f"   {kind:<8} AUROC {out[f'auroc_{kind}']:.3f} · detected {out[f'detect_{kind}']:.0%} "
                    f"at 5% false alarms · hottest patch on the object {out[f'locate_{kind}']:.0%}")

    n = len(scorer.bank)
    out["curve"] = []
    for k in [k for k in curve_ks if k < n] + [n]:
        sets = full if k == n else _score_sets(scorer, val_chips, scorer.bank[:k])
        row = {"memory": int(k), **{f"auroc_{kind}": _auroc(sets, kind) for kind, _ in KINDS}}
        out["curve"].append(row)
        logger.info(f"   memory {k:>6,} patches · AUROC subtle {row['auroc_subtle']:.3f} · obvious {row['auroc_obvious']:.3f}")
        if live:
            live.metric(**row)
    return out, np.sort(clean)


# ============================================
# 6. DETECTION + HEATMAP
# ============================================

def _why(scorer, arr, cy, cx):
    """Hand-crafted feature z-scores (vs natural terrain) at one pixel, from a chip centred on it."""
    y0 = int(np.clip(cy - CHIP // 2, 0, arr.shape[0] - CHIP))
    x0 = int(np.clip(cx - CHIP // 2, 0, arr.shape[1] - CHIP))
    z = scorer.hand_z(torch.from_numpy(normalize(arr[y0:y0 + CHIP, x0:x0 + CHIP]))[None, None]).cpu().numpy()[0]
    py, px = min((cy - y0) // PATCH, CHIP // PATCH - 1), min((cx - x0) // PATCH, CHIP // PATCH - 1)
    return {n: float(z[k, py, px]) for k, n in enumerate(HAND_NAMES)}


def detect(scorer, image_path, clean_sorted, out_png, scene_z=6.0, n_peaks=15, spacing=40):
    """Score half-overlapping chips of a test image and stitch a heatmap from each chip's interior
    (border patches never count: they see padding, not terrain). Hotspots are ranked against the
    rest of the same scene — robust z = (score - scene median) / MAD — one per 80-px neighbourhood,
    so a different camera or resolution doesn't make the whole image look anomalous."""
    arr = load_gray(image_path)
    h, w = arr.shape
    tiles = list(tile(arr, full=True, stride=CHIP // 2))
    maps = scorer.score_maps([normalize(c) for _, _, c in tiles])
    heat = np.full(arr.shape, -np.inf, dtype=np.float32)
    b = EDGE * PATCH
    for (y, x, _), m in zip(tiles, maps):
        region = (slice(y + b, y + CHIP - b), slice(x + b, x + CHIP - b))
        heat[region] = np.maximum(heat[region], np.kron(interior(m), np.ones((PATCH, PATCH), np.float32)))
    valid = np.isfinite(heat)
    med = float(np.median(heat[valid]))
    mad = float(1.4826 * np.median(np.abs(heat[valid] - med))) + 1e-8
    z = np.where(valid, (heat - med) / mad, -np.inf)

    rows, zz = [], z.copy()
    for _ in range(n_peaks):
        cy, cx = (int(v) for v in np.unravel_index(np.argmax(zz), zz.shape))
        if not np.isfinite(zz[cy, cx]):
            break
        zz[max(0, cy - spacing):cy + spacing, max(0, cx - spacing):cx + spacing] = -np.inf
        score = float(heat[cy, cx])
        rows.append({"chip": f"{Path(image_path).stem} · y{cy} x{cx}", "score": score, "scene_z": float(z[cy, cx]),
                     "confidence": float(np.searchsorted(clean_sorted, score) / len(clean_sorted)),
                     "anomaly": bool(z[cy, cx] >= scene_z), "hotspot": [cy, cx],
                     "why": _why(scorer, arr, cy, cx)})

    smooth = F.avg_pool2d(torch.from_numpy(np.where(valid, z, 0).astype(np.float32))[None, None], 9, stride=1,
                          padding=4, count_include_pad=False)[0, 0].numpy()
    alpha = np.clip((smooth - 2) / (scene_z - 2), 0, 1)          # 2σ: starts to glow · scene_z σ: full
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(arr, cmap="gray")
    ax.imshow(alpha, cmap="inferno", alpha=alpha * 0.85, vmin=0, vmax=1)
    for k, r in enumerate(rows[:10], 1):                          # numbered like the dashboard table
        cy, cx = r["hotspot"]
        col = "#f0884e" if r["anomaly"] else "#b2bdd2"
        ax.add_patch(plt.Circle((cx, cy), 18, fill=False, ec=col, lw=1.5))
        ax.text(cx + 21, cy - 12, str(k), color=col, fontsize=11, fontweight="bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return rows, {"image": Path(image_path).name, "median": med, "mad": mad, "tiles": len(tiles)}


# ============================================
# 7. RUN LOG (read by dashboard/index.html)
# ============================================

def save_run_record(record, path=Path("dashboard") / "runs.json"):
    runs = json.loads(path.read_text()) if path.exists() else []
    runs.append(record)
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(runs, indent=1))
    logger.info(f"Run record saved: {path}")


# ============================================
# 8. MAIN PIPELINE
# ============================================

def _arg(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def main(live=None):
    started = datetime.now()
    config = {
        "chip_size": CHIP,
        "backbone": "resnet18 · layer2+layer3",
        "bank_size": _arg("--bank", 30000, int),
        "per_chip": _arg("--per-chip", 24, int),
        "scene_z": _arg("--scene-z", 6.0, float),       # flag hotspots this many σ above their own scene
        "hand_weight": _arg("--hand", 0.0, float),       # 0: validation showed they hurt (0.922 -> 0.869 -> 0.793)
        "val_frac": 0.10,
        "val_chips": 150,
        "device": pick_device(),
    }
    train_dir, test_dir = _arg("--data", "training data"), _arg("--test", "Test data")
    logger.info("=" * 70)
    logger.info("XENARCH Mk18: PATCHCORE ANOMALY DETECTION")
    logger.info("=" * 70)
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    # 1. Split by image, so validation chips come from images the memory never saw
    imgs = sorted(p for p in Path(train_dir).rglob("*") if p.suffix.lower() in EXTS)
    if not imgs:
        raise SystemExit(f"Mk18: no images in '{train_dir}'")
    rng = np.random.default_rng(SEED)
    rng.shuffle(imgs)
    n_val = max(1, int(len(imgs) * config["val_frac"]))
    val_imgs, train_imgs = imgs[:n_val], imgs[n_val:]
    val_chips = list(natural_chips(val_imgs))
    rng.shuffle(val_chips)
    val_chips = val_chips[:config["val_chips"]]
    if live:
        live.samples(val_chips[:12])
        live.log(f"{len(train_imgs)} memory images · {len(val_imgs)} validation images · {len(val_chips)} validation chips")
    logger.info(f"\n[STEP 1/4] {len(train_imgs)} memory images · {len(val_imgs)} validation images "
                f"({len(val_chips)} chips)")

    # 2. Memory bank of natural terrain
    logger.info(f"\n[STEP 2/4] Building the memory bank on {config['device']}...")
    scorer = PatchCoreScorer(config["device"], config["bank_size"], config["per_chip"],
                             hand_weight=config["hand_weight"])
    scorer.live = live
    bank_path = Path("data") / "models" / "xenarch_mk18_bank.pt"
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    if "--reuse" in sys.argv and bank_path.exists():             # rescan with the saved memory, no rebuild
        if live:
            live.step("Loading memory bank")
        ck = torch.load(bank_path, weights_only=False)
        scorer.bank = ck["bank"].to(scorer.device)
        scorer.hand_mu, scorer.hand_sd = ck["hand_mu"].to(scorer.device), ck["hand_sd"].to(scorer.device)
        n_train, scorer.pool_size = ck.get("train_chips", 0), ck.get("pool_patches", 0)
        logger.info(f"   reusing {bank_path} ({len(scorer.bank):,} patches)")
    else:
        if live:
            live.step("Reading memory images")
        n_train = scorer.fit(natural_chips(train_imgs, on_image=live and (lambda i, n: live.progress(i, n, "images"))))

    # 3. Validation
    logger.info("\n[STEP 3/4] Validating on held-out natural chips (clean vs synthetic objects)...")
    if live:
        live.step("Validation")
    val, clean_sorted = validate(scorer, val_chips, live=live)
    torch.save({"train_chips": n_train, "pool_patches": scorer.pool_size,
                "bank": scorer.bank.cpu(), "hand_mu": scorer.hand_mu.cpu(), "hand_sd": scorer.hand_sd.cpu(),
                "config": config, "validation": val,
                "clean_scores": clean_sorted}, bank_path)
    logger.info(f"   Memory bank saved: {bank_path}")

    # 4. Detection on the test images
    logger.info("\n[STEP 4/4] Scanning test images...")
    run_id = started.strftime("%Y-%m-%d %H:%M")
    rows, heatmaps, scenes = [], [], []
    tests = sorted(q for q in Path(test_dir).glob("*") if q.suffix.lower() in EXTS)
    if live:
        live.step("Scanning test images")
    for k, p in enumerate(tests, 1):
        if live:
            live.progress(k, len(tests), "images")
        name = f"mk18_{started:%Y%m%d_%H%M%S}_{p.stem.replace(' ', '_')}.png"
        found, scene = detect(scorer, p, clean_sorted, Path("dashboard") / "heatmaps" / name, config["scene_z"])
        rows += found
        scenes.append(scene)
        heatmaps.append(f"heatmaps/{name}")
        Path("results/mk18").mkdir(parents=True, exist_ok=True)
        (Path("results/mk18") / f"heatmap_{p.stem.replace(' ', '_')}.png").write_bytes(
            (Path("dashboard") / "heatmaps" / name).read_bytes())
    rows.sort(key=lambda r: -r["scene_z"])

    for rank, r in enumerate(rows[:10], 1):
        logger.info(f"   Rank {rank}: {r['chip']}  {r['scene_z']:.1f}σ above scene  score {r['score']:.3f}"
                    f"{'  FLAGGED' if r['anomaly'] else ''}")

    save_run_record({
        "kind": "patchcore",
        "id": run_id,
        "script": Path(__file__).name,
        "data": train_dir,
        "train_chips": n_train,
        "test_chips": sum(sc["tiles"] for sc in scenes),
        "minutes": round((datetime.now() - started).total_seconds() / 60, 1),
        "config": config,
        "epochs": [],
        "patchcore": {**val, "bank_size": int(len(scorer.bank)), "pool_patches": scorer.pool_size,
                      "scene_z": config["scene_z"], "scenes": scenes, "reused_bank": "--reuse" in sys.argv},
        "heatmaps": heatmaps,
        "performance": {
            "anomalies": int(sum(r["anomaly"] for r in rows)),
            "high_confidence": int(sum(r["confidence"] > 0.8 for r in rows)),
            "mean_confidence": float(np.mean([r["confidence"] for r in rows])) if rows else 0.0,
            "top": rows[:10],
        },
    })
    logger.info("\n" + "=" * 70)
    logger.info(f"XENARCH Mk18 COMPLETE · heatmaps in dashboard/heatmaps and results/mk18")
    logger.info("=" * 70)


if __name__ == "__main__":
    live = LiveRun(__file__, _arg("--data", "training data"),
                   (["Loading memory bank"] if "--reuse" in sys.argv else ["Reading memory images", "Building memory bank"])
                   + ["Validation", "Scanning test images"])
    try:
        main(live)
        live.done()
    except BaseException as e:
        live.fail(e)
        raise

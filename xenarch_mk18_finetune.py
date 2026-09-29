"""
Xenarch Mk18 backbone fine-tuning — adapt ResNet18 from everyday photos to natural planetary terrain.

PatchCore (xenarch_mk18_script.py) describes patches with an ImageNet ResNet18. This script fine-tunes that
network's layer2 + layer3 (the layers PatchCore reads) on natural chips with the Mean-Shifted Contrastive
loss (Reiss & Hoshen 2021, "Mean-Shifted Contrastive Loss for Anomaly Detection"): two random views of a
chip (crop, flip, 90° rotation, brightness / contrast / gamma, blur) must match after subtracting the
mean natural feature, so the features stop spending capacity on lighting and orientation and keep it for
what the terrain looks like. Normal features are kept around a fixed centre, which avoids the collapse
plain contrastive fine-tuning suffers.

Only memory images are used — the same split by source image as Mk18, so validation chips, the synthetic
objects and the test images are never seen. BatchNorm statistics and the stem + layer1 stay frozen.

  .venv/bin/python xenarch_mk18_finetune.py --data data/xenarch-imagery,data/lroc-nac-set,data/hirise-set
        [--steps 800 --batch 48 --lr 1e-5 --chips 6000 --out data/models/mk18_backbone_ft.pt]
  then: .venv/bin/python xenarch_mk18_script.py --data <same> --backbone data/models/mk18_backbone_ft.pt
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from loguru import logger

from xenarch_live import LiveRun
from xenarch_mk18_script import EXTS, SEED, natural_chips, pick_device, split_by_source

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _arg(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def memory_chips(folders, frac, limit, rng):
    """Natural chips from the memory side of Mk18's split only (validation sources are left out)."""
    paths = []
    for d in folders:
        imgs = sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in EXTS)
        _, train = split_by_source(imgs, frac, np.random.default_rng(SEED))
        paths += train
    chips = list(natural_chips(paths))
    idx = rng.permutation(len(chips))[:limit]
    return np.stack([chips[i] for i in idx]), len(paths)


def augment(x, g):
    """Two-view augmentation on the GPU. x: (B, 1, 256, 256) in [0, 1]."""
    b, _, h, w = x.shape
    out = []
    for i in range(b):
        s = float(torch.empty(1).uniform_(0.5, 1.0, generator=g)) ** 0.5
        ch, cw = int(h * s), int(w * s)
        y0 = int(torch.randint(0, h - ch + 1, (1,), generator=g)); x0 = int(torch.randint(0, w - cw + 1, (1,), generator=g))
        v = F.interpolate(x[i:i + 1, :, y0:y0 + ch, x0:x0 + cw], size=(h, w), mode="bilinear", align_corners=False)
        v = torch.rot90(v, int(torch.randint(0, 4, (1,), generator=g)), dims=(2, 3))
        if torch.rand(1, generator=g) < 0.5:
            v = torch.flip(v, dims=(3,))
        gamma = float(torch.empty(1).uniform_(0.7, 1.4, generator=g))
        contrast = float(torch.empty(1).uniform_(0.7, 1.3, generator=g))
        bright = float(torch.empty(1).uniform_(-0.1, 0.1, generator=g))
        v = v.clamp(1e-4, 1) ** gamma
        v = ((v - v.mean()) * contrast + v.mean() + bright).clamp(0, 1)
        if torch.rand(1, generator=g) < 0.3:
            v = F.avg_pool2d(F.pad(v, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        out.append(v)
    return torch.cat(out)


class Embed(torch.nn.Module):
    """ResNet18 up to layer3; embedding = pooled layer2 + layer3 (the features PatchCore reads), L2-normed."""

    def __init__(self, device):
        super().__init__()
        self.net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).to(device)
        for p in self.net.parameters():
            p.requires_grad = False
        for layer in (self.net.layer2, self.net.layer3):
            for name, p in layer.named_parameters():
                p.requires_grad = "bn" not in name and "downsample.1" not in name
        self.mean, self.std = MEAN.to(device), STD.to(device)

    def train(self, mode=True):
        super().train(mode)
        self.net.eval()                     # BatchNorm running statistics stay as pretrained
        return self

    def forward(self, x):
        n = self.net
        x = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        l2 = n.layer2(n.layer1(n.maxpool(n.relu(n.bn1(n.conv1(x))))))
        l3 = n.layer3(l2)
        f = torch.cat([F.adaptive_avg_pool2d(l2, 1), F.adaptive_avg_pool2d(l3, 1)], 1).flatten(1)
        return F.normalize(f, dim=1)


def msc_loss(z1, z2, center, tau=0.25):
    """Mean-shifted NT-Xent: views are compared after removing the centre of natural features."""
    m1, m2 = F.normalize(z1 - center, dim=1), F.normalize(z2 - center, dim=1)
    z = torch.cat([m1, m2])
    sim = z @ z.t() / tau
    sim.fill_diagonal_(-1e9)
    n = len(m1)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


def main(live):
    device = pick_device()
    folders = [d for d in _arg("--data", "data/xenarch-imagery").split(",") if d]
    steps, batch, lr = _arg("--steps", 800, int), _arg("--batch", 48, int), _arg("--lr", 1e-5, float)
    out = Path(_arg("--out", "data/models/mk18_backbone_ft.pt"))
    rng = np.random.default_rng(SEED)
    g = torch.Generator().manual_seed(SEED)

    live.step("Reading memory chips")
    chips, n_imgs = memory_chips(folders, 0.10, _arg("--chips", 6000, int), rng)
    logger.info(f"{len(chips)} natural chips from {n_imgs} memory images ({', '.join(folders)})")
    live.log(f"{len(chips)} chips from {n_imgs} memory images")
    live.samples(list(chips[:12]))

    model = Embed(device)
    with torch.no_grad():                                   # fixed centre of natural features
        center = torch.cat([model(torch.from_numpy(chips[i:i + 64][:, None]).to(device))
                            for i in range(0, min(len(chips), 2048), 64)]).mean(0, keepdim=True)
        center = F.normalize(center, dim=1)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-5)
    logger.info(f"training {sum(p.numel() for p in params):,} weights for {steps} steps · batch {batch} · lr {lr} · {device}")

    live.step("Fine-tuning")
    model.train()
    t0, hist = time.time(), []
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(chips), (batch,), generator=g).numpy()
        x = torch.from_numpy(chips[idx][:, None]).to(device)
        z1, z2 = model(augment(x, g)), model(augment(x, g))
        loss = msc_loss(z1, z2, center)
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(loss.item())
        if step % 25 == 0 or step == steps:
            avg = float(np.mean(hist[-25:]))
            live.progress(step, steps, "steps")
            live.metric(step=step, loss=round(avg, 4))
            logger.info(f"   step {step:>4}/{steps} · loss {avg:.4f} · {time.time() - t0:.0f}s")

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"resnet18": model.net.state_dict(), "method": "mean-shifted contrastive (layer2+layer3)",
                "data": folders, "steps": steps, "batch": batch, "lr": lr, "chips": int(len(chips)),
                "loss_first": float(np.mean(hist[:25])), "loss_last": float(np.mean(hist[-25:]))}, out)
    logger.info(f"saved {out} · loss {np.mean(hist[:25]):.3f} -> {np.mean(hist[-25:]):.3f}")
    live.log(f"saved {out}")


if __name__ == "__main__":
    live = LiveRun(__file__, _arg("--data", "data/xenarch-imagery"), ["Reading memory chips", "Fine-tuning"])
    try:
        main(live)
        live.done()
    except BaseException as e:
        live.fail(e)
        raise

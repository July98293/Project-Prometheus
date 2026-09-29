# Project Prometheus (Xenarch)

Unsupervised anomaly detection on planetary imagery: the model only sees natural terrain, and anything unlike it is flagged. Mk15 uses a VAE (reconstruction error); Mk18 uses PatchCore (distance to the nearest remembered natural patch).

## Workflow

- Python env: `.venv/` (torch, rasterio, flask…). Always run scripts with `.venv/bin/python`.
- Train + record a run: `./train.sh [image folder]` (default `data/xenarch-imagery`). Runs `xenarch_mk15_script.py --data <dir>`, logs to `logs/`, appends the run to `dashboard/runs.json`.
- Mk18 (PatchCore, current best): `./train.sh --mk18 [folder]` or `.venv/bin/python xenarch_mk18_script.py --data <dir> [--bank 30000] [--hand 0]`. No training: builds a memory bank of ResNet18 patch features (saved to `data/models/xenarch_mk18_bank.pt`), validates on held-out images with synthetic objects, writes heatmaps to `dashboard/heatmaps/` and `results/mk18/`. Judge changes by its validation AUROC (subtle / obvious) — the Mk15 VAE scored no better than a Gaussian blur.
- Hyperparameter search: `.venv/bin/python xenarch_mk17_with_finetuning.py --tune [--data dir --configs N --rungs 1,3,8 --confirm 10 --max-chips 600]`. Writes `results/mk17_finetune.csv/.png`, `dashboard/finetune.json`, `data/models/mk17_finetuned.pt`.
- Live progress: Mk15, Mk17 `--tune` and Mk18 write `dashboard/live.json` while running via `xenarch_live.LiveRun` (step, progress, metrics, log, sample chips); the page polls it every 2 s. New scripts should use it too.
- Dashboard: `./train.sh --view` → http://localhost:8050 (static files in `dashboard/`, served by `python -m http.server`).
- Data: HF dataset `Giulia928982/xenarch-imagery` is gated. Download with `.venv/bin/hf download … --include "*_png/*" --local-dir data/xenarch-imagery` after `hf auth login`. Never put tokens in files or chat. `data/` and `*.pth` are gitignored.
- Moon high-res set: `.venv/bin/python xenarch_lroc_nac_dataset.py [--out data/lroc-nac-set --per-category 12 --suns 2]` downloads 1024² LROC NAC tiles of natural terrain (6 landform folders, several sun angles, `metadata.csv`) from NASA ODE / PDS with HTTP range requests; skips targets within 10 km of landing/impact sites. The NASA store rate-limits (429): keep requests sequential. `--upload <user>/<repo>` pushes the folder as a private HF dataset (needs `hf auth login`).
- Mars set: `.venv/bin/python xenarch_hirise_dataset.py [--out data/hirise-set --per-category 16]` picks HiRISE observations by the team's RATIONALE_DESC (index cached in `data/cache/hirise/`), 6 landform folders, and cuts 1024×512 tiles from bin-2 EDR channels (RED5→RED4→RED3→RED6; RED4 is missing in many recent observations). Bin-1 EDRs were tried and rejected: raw column pattern too strong. Same `--upload`.
- Runs overwrite tracked files in `results/`; say so when it happens.
- Several Claude sessions may work on this repo at once: before a long run (GPU) or a big edit to a shared file, check `ps` / `ListAgents` and coordinate.

## Dashboard style (reuse `dashboard/xenarch.css` for any new page)

- Dark navy background `#0c1628` with a 32px grid of thin lines ("Latent Lane" look), soft blue glow behind the hero.
- Fonts: **Poppins** for headings and text; **JetBrains Mono** for labels, numbers and tables. Labels are uppercase, letter-spaced, muted.
- Title small and quiet (~38px), subtitle ~15px light.
- Accent: orange `#f0884e` (links, section numbers, flagged items). **No green anywhere.**
- **No icons, logos, emoji or check marks** — use words (e.g. `FLAGGED`, `—`).
- Space-mission vocabulary: Mission control, telemetry, flight log, ground station, target coordinates, UTC clock.
- Thin hairline borders, semi-transparent navy cards, no heavy shadows.
- Chart series colors (validated on the navy surface, incl. color-blind checks): `#3987e5` blue, `#c98500` amber, `#d55181` magenta; single-series lines use the ink color. Validate any new color set with the dataviz skill's `validate_palette.js` before using it.
- Charts: one y-axis only, legend for 2+ series, hover tooltip, exact values also in a table.
- Plain HTML + CSS + a little vanilla JS. No frameworks; fonts from Google Fonts only.
- Must work at 390px width with no horizontal scroll.

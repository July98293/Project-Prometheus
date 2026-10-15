# Xenarch artifact site bounding boxes

Published as `human-hardware_ground-truth/boxes_lroc-nac_hirise.csv`, `boxes-coco_lroc-nac_hirise.json`, `unboxed_lroc-nac_hirise.csv`
in the private HF dataset `jules1235813/xenarch` (`.venv/bin/python xenarch_hf.py tables`; `file_name` becomes e.g.
`moon_lroc-nac/apollo11_lm_lroc-nac/M117338434RC.png`).

Bounding boxes around human-made hardware in the 2048×2048 orbital crops of
`human-hardware_ground-truth/crops_lroc-nac_hirise/` of the HF dataset `jules1235813/xenarch`. Ground truth for Xenarch, an unsupervised anomaly detector trained only on natural terrain.

| File | What |
|---|---|
| `boxes.csv` | one row per object per crop: pixel box, confidence, how it was found (source of truth) |
| `boxes_coco.json` | the same boxes in COCO layout, with every crop (positive or negative) as an image record |
| `unboxed.csv` | positive crops where the hardware could **not** be identified, with the reason |
| `review/*.png` | one review sheet per annotated crop: tight zoom with boxes + 600 px context with the catalog nominal position as a red cross |
| `review.html` | all review sheets and box tables on one page |
| `boxtool.py`, `register.py` | the annotation helper (candidate hunt, pixel grid, frame-to-frame terrain matching, review/COCO export) |

## Status (2026-10-14)

| | crops | boxed | not identified | negative (before arrival) |
|---|---|---|---|---|
| Moon (LROC NAC) | 131 | 63 | 64 | 4 |
| Mars (HiRISE) | 126 | 13 | 105 | 8 |

102 boxes: 48 high, 41 medium, 13 low confidence. All Apollo crops are boxed except five
near-terminator or oblique frames. Surveyor, Luna, Lunokhod and Chang'e landers (2–4 m, 3–8 px) are
boxed only where a shadow or a terrain match made the identification defensible. Mars rovers (12 px)
and landers were boxed only where an object sits at the waypoint or catalog position with a shadow.

## Box conventions

- Pixel coordinates in the 2048² crop, origin top-left, `x_max`/`y_max` inclusive. COCO `bbox` is `[x, y, w, h]`.
- The box covers the object **and its shadow** (`includes_shadow`), because for the detector the shadow is part of the anomaly.
  S-IVB impact craters: the crater, not the rays.
- `confidence`: `high` = unambiguous (bright body + hard shadow, tracks, or matches a published labeled image);
  `medium` = right size and shadow at the predicted position but no independent confirmation;
  `low` = plausible but could be a boulder / rock.
- `object_id` matches `sites.csv` in the catalog. `apollo11_lrrr` (Laser Ranging Retroreflector) is an extra id not in `sites.csv`.
- `note` says how the box was found and the offset from the catalog's nominal pixel.

## How the boxes were found

The catalog's nominal pixel (`obj_x`, `obj_y` in the images' `metadata.csv`) is only a starting point:
for LROC NAC crops it comes from linear interpolation between the frame footprint corners and is off by
up to ~1300 px (the Apollo 11 LM is 217 px from the nominal centre). The crop uncertainty note "about ±60 px"
in the catalog is wrong; it should read "up to ±1300 px".

Per site: one frame was anchored by eye (Apollo 11 against NASA's labeled release of M175124932R; others
by the bright body + eastward shadow + astronaut tracks). Other frames of the same site were located by
`register.py locate_points`: crater positions as a point set, Hough vote over translation for all eight
orientations × independent x/y scales (NAC pixels are not square). A vote ratio above ~3 means a real match
(error below 10 px). Other objects in the same frame were predicted from the anchor plus the catalog's
relative offset (`boxtool.py predict`), then confirmed by eye on a pixel grid.

## Caveats

- Most Mars crops are unboxed: a rover is 12 px at 0.25 m/px and indistinguishable from rocks without
  a published reference. The Phoenix, InSight and Perseverance boxes rely on the HiRISE map projection
  being accurate to a few pixels.
- `low` boxes should be treated as candidates, not truth.

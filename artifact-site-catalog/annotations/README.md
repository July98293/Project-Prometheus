# Xenarch artifact site bounding boxes

Published as `human-hardware_ground-truth/boxes-coco_lroc-nac_hirise.json`
in the private HF dataset `jules1235813/xenarch` (`.venv/bin/python xenarch_hf.py tables`; `file_name` becomes e.g.
`moon_lroc-nac/apollo11_lm_lroc-nac/M117338434RC.png`).

Bounding boxes around human-made hardware in the 2048×2048 orbital crops of
`human-hardware_ground-truth/crops_lroc-nac_hirise/` of the HF dataset `jules1235813/xenarch`. Ground truth for Xenarch, an unsupervised anomaly detector trained only on natural terrain.

| File | What |
|---|---|
| `boxes_coco.json` | **the only annotation file** (COCO). `images`: every crop, positive or negative; a positive crop where the hardware could not be identified has `unboxed_reason`. `annotations`: one box per object per crop with `includes_shadow`, `confidence`, `annotator`, `note`. `categories`: one per `object_id`. Edit it through `boxtool.py` (`box`, `rect`, `drop`, `unbox`, `transfer`) |
| `review/*.png` | one review sheet per annotated crop: tight zoom with boxes + 600 px context with the catalog nominal position as a red cross |
| `review.html` | all review sheets and box tables on one page |
| `boxtool.py`, `register.py` | the annotation helper (candidate hunt, pixel grid, frame-to-frame terrain matching, review/COCO export) |

## Status (2026-10-24)

| | crops | boxed | not identified | negative (before arrival) |
|---|---|---|---|---|
| Moon (LROC NAC) | 254 | 124 | 108 | 22 |
| Mars (HiRISE) | 273 | 164 | 83 | 26 |

338 boxes: 324 over 58 objects plus 14 track segments over 8 track categories; 232 high, 83 medium, 23 low confidence
(Moon 92 / 54 / 22, Mars 140 / 29 / 1).
Every positive crop is boxed or has a frame-specific `unboxed_reason`. Mars boxes rest on HiRISE release
images (uahirise.org) or a before/after change, then on terrain matching between frames
(`hirise_match.py`); the earlier Mars boxes placed at the catalog position were wrong and are gone.
Moon boxes for the objects added on 2026-10-15 rest on LROC featured/labelled images, catalog offsets
from an anchored object in the same frame, or before/after pairs. Known gaps: Schiaparelli, Mars 3,
Zhurong, Ingenuity, GRAIL, LADEE, LCROSS and most rover frames are not identifiable in these crops.
`mars/spirit/ESP_012787_1650.png` is labelled `before_arrival` but already shows Spirit (it entered Troy
in late April 2009): do not use it as a negative.

## Mars crops are centred by terrain match

HiRISE map projections are off by 100–400 m from frame to frame, so a crop centred on the catalog
coordinate can miss the object (InSight was 390 m off). `download_images.py` therefore matches a ~1 m/px
overview of each new frame against the high-confidence boxes of the site and centres the crop on the
match (`match_ratio`, `position_note`); `--recut` redoes unboxed crops once a site has a reference.

## Box conventions

- Pixel coordinates in the 2048² crop, origin top-left. COCO `bbox` is `[x, y, w, h]`: the box covers pixels `x … x+w-1`.
- The box covers the object **and its shadow** (`includes_shadow`), because for the detector the shadow is part of the anomaly.
  S-IVB impact craters: the crater, not the rays.
- `confidence`: `high` = unambiguous (bright body + hard shadow, tracks, or matches a published labeled image);
  `medium` = right size and shadow at the predicted position but no independent confirmation;
  `low` = plausible but could be a boulder / rock.
- The category `name` (`object_id`) matches `sites.csv` in the catalog. `apollo11_lrrr` (Laser Ranging Retroreflector) is an extra id not in `sites.csv`.
- Tracks are categories of their own, not in `sites.csv`: `<mission>_footpaths` (astronaut trails) and `<rover>_tracks`
  (`apollo15_lrv_tracks`, `curiosity_rover_tracks`, …). One box per clearly visible segment (`boxtool.py rect … --add`
  keeps the other segments), `includes_shadow` false; `medium` where LRV tracks and footpaths may be mixed.
  Found on Apollo 12, 14, 15, 16, 17 and Curiosity (sol 33); not identifiable without a reference on Lunokhod 1/2,
  Yutu-2, Perseverance, Spirit, Opportunity.
- `../build_labels.py` turns every box into a 256² classification chip (`human-traces_labels/` on HF).
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

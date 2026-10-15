---
license: cc0-1.0
pretty_name: Xenarch
tags: [moon, mars, lroc, hirise, planetary-science, anomaly-detection, technosignatures, object-detection]
size_categories: [1K<n<10K]
configs:
  - config_name: terrain_moon
    data_dir: terrain/moon
  - config_name: terrain_mars
    data_dir: terrain/mars
  - config_name: terrain_legacy
    data_dir: terrain/legacy
  - config_name: artifact_images
    data_dir: hardware/images
    default: true
  - config_name: sites
    data_files: hardware/sites.csv
  - config_name: frames
    data_files: hardware/frames.csv
  - config_name: rover_frames
    data_files: hardware/mars_mission_frames.csv
  - config_name: boxes
    data_files: hardware/boxes.csv
  - config_name: unboxed
    data_files: hardware/unboxed.csv
---
# Xenarch

Data for Xenarch, an unsupervised anomaly detector for planetary imagery. The model sees only natural
terrain (`terrain/`); the places where it *should* raise a flag are the known human-made hardware on the
Moon and Mars: landers, rovers, experiment packages, rocket-stage impact craters (`hardware/`).
Keep the two apart: `terrain/` is training data, `hardware/` is the test.

```
terrain/                        natural terrain, no hardware (training)
  moon/<landform>/*.png         445 LROC NAC tiles, 1024×1024, 0.4–1.8 m/px
  moon/metadata.csv
  mars/<landform>/*.png         384 HiRISE tiles, 1024×512, ~0.5 m/px
  mars/metadata.csv
  legacy/                       the original mixed set (Clementine, CTX, HiRISE, MOC, LROC, TMC2 + 1,250 "squares")
hardware/                       known human-made objects (ground truth)
  sites.csv                     44 objects: coordinates, arrival date, best frame
  frames.csv                    2,298 object × frame rows: every orbital frame covering each object
  mars_mission_frames.csv       57 HiRISE frames that caught the Curiosity / Perseverance rovers
  images/<body>/<site>/*.png    257 crops of 2048×2048 centred on each object (moon/, mars/)
  images/metadata.csv
  boxes.csv                     102 bounding boxes around the hardware in 76 crops
  boxes_coco.json               the same boxes in COCO layout
  unboxed.csv                   171 positive crops where the hardware could not be identified, with the reason
```

All images are 8-bit grayscale PNG at native resolution (no resampling), each stretched from its
0.5–99.5 percentile. Scripts that build every file live in the Project Prometheus repository
(`xenarch_lroc_nac_dataset.py`, `xenarch_hirise_dataset.py`, `artifact-site-catalog/`).

## terrain/

**moon** (LROC NAC CDR, calibrated): six landform folders, `fresh_craters`, `mass_wasting`, `tectonic`,
`volcanic`, `chains_terrain`, `polar`. Targets come from the IAU gazetteer; each was imaged under several
sun angles; tile 0 is centred on the target and tiles 1–3 are its neighbours across the strip. Anything
within 10 km of a landing or impact site was skipped. `metadata.csv`: target, coordinates, product ID,
incidence / emission / phase, `resolution_m`, UTC, tile origin, source URL.

**mars** (HiRISE EDR, raw bin-2 RED channels, odd columns repaired): six landform folders, `aeolian`,
`slopes`, `periglacial_ice`, `polar`, `impacts`, `layers_volcanic`, chosen from the HiRISE team's own
rationale per observation and spread over sun angles. Observations mentioning hardware or within 15 km
of a lander, rover or crash site were skipped. `metadata.csv`: landform, observation ID, rationale,
channel, coordinates, incidence / emission, solar longitude, UTC, source URL.

Neither set has been checked tile by tile for hardware.

**legacy** is the first-phase training set, formerly `xenarch-imagery`: per instrument a raw folder
(`clementine/`, `ctx/`…), a `*_png/` folder of 8-bit PNGs, a `*_meta/` folder of labels, plus
`squares/` (1,250 fixed-size chips) and `manifest.csv`. Mixed bodies, instruments and resolutions, no
landform labels, no hardware exclusion. Mk15 trains on its `*_png/` folders; prefer `moon/` and `mars/`
for new work.

## hardware/

**sites.csv**: one row per object. Apollo 11–17 (LM, ALSEP/PSE, LRV, S-IVB impact craters), Surveyor 1/3/5/6/7,
Luna 16/17/20/21/23/24, Lunokhod 1/2, Chang'e 3/4/5/6, Yutu, Phoenix, Curiosity, InSight, Perseverance.
Lunar coordinates from Wagner et al. 2016 (LROC), accurate to ~0.5 m.

**frames.csv**: one row per object per frame whose footprint contains it, with `usable`, `after_arrival`,
sun and viewing angles, resolution and archive URLs. `obj_sample_frac` / `obj_line_frac` give the object's
position in a raw NAC frame **only approximately** (interpolated from footprint corners, off by up to ~1,300 px).

**images/**: 2048² crops around each object, `label` = `artifact` or `before_arrival` (same ground before the
object arrived). Moon crops are in raw NAC geometry, Mars crops are map-projected. `obj_x`, `obj_y` is the
catalog's nominal pixel, not the measured one: use `boxes.csv` for that.

**boxes.csv**: pixel box per object per crop, inclusive edges, origin top-left. Boxes include the object's
shadow. `confidence` is `high` (unambiguous: bright body with hard shadow, tracks, or matches a published
labeled image), `medium` (right size and shadow at the predicted position) or `low` (could be a rock).
`note` says how the box was found. All six Apollo sites are boxed; most small Soviet and Chinese landers and
most Mars rover frames are not (`unboxed.csv` says why).

## Credit

Moon: NASA / GSFC / Arizona State University (LROC). Mars: NASA / JPL / University of Arizona (HiRISE).
Source data are public domain; this compilation is CC0.

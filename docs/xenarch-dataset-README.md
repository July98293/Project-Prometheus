---
license: cc0-1.0
pretty_name: Xenarch
tags: [moon, mars, lroc, hirise, planetary-science, anomaly-detection, technosignatures, object-detection]
size_categories: [1K<n<10K]
configs:
  - config_name: terrain_moon
    data_dir: natural-terrain_training/moon_lroc-nac
  - config_name: terrain_mars
    data_dir: natural-terrain_training/mars_hirise
  - config_name: terrain_legacy
    data_dir: natural-terrain_training/legacy_mixed-cameras
  - config_name: artifact_images
    data_dir: human-hardware_ground-truth/crops_lroc-nac_hirise
    default: true
  - config_name: catalog
    data_files: human-hardware_ground-truth/catalog_moon-mars_lroc-nac_hirise.csv
  - config_name: sites
    data_files: human-hardware_ground-truth/derived/sites_moon-mars.csv
  - config_name: frames
    data_files: human-hardware_ground-truth/derived/frames_lroc-nac_hirise.csv
  - config_name: rover_frames
    data_files: human-hardware_ground-truth/derived/rover-frames_mars_hirise.csv
---
# Xenarch

Data for Xenarch, an unsupervised anomaly detector for planetary imagery. The model sees only natural
terrain (`natural-terrain_training/`); the places where it *should* raise a flag are the known human-made hardware on the
Moon and Mars: landers, rovers, experiment packages, crash and rocket-stage impact craters (`human-hardware_ground-truth/`).
Keep the two apart: `natural-terrain_training/` is training data, `human-hardware_ground-truth/` is the test.

Every folder and file name says what it holds and which camera took it (`lroc-nac` = LRO Narrow Angle
Camera, Moon; `hirise` = MRO HiRISE, Mars), so a folder copied out of the dataset still explains itself.

```
natural-terrain_training/                      natural terrain, no hardware (training + validation)
  moon_lroc-nac/
    metadata.csv
    moon_<landform>_lroc-nac/*.png             445 LROC NAC tiles, 1024×1024, 0.4–1.8 m/px
  mars_hirise/
    metadata.csv
    mars_<landform>_hirise/*.png               384 HiRISE tiles, 1024×512, ~0.5 m/px
  legacy_mixed-cameras/                        the original mixed set (formerly xenarch-imagery)
    metadata.csv
    <body>_<camera>/*.png                      984 full images: moon_clementine, moon_lroc-nac-wac,
                                               moon_chandrayaan2-tmc2, mars_ctx, mars_hirise
human-hardware_ground-truth/                   known human-made objects (ground truth)
  catalog_moon-mars_lroc-nac_hirise.csv        THE table: 4,633 rows, one per object × orbital frame (100 objects)
  boxes-coco_lroc-nac_hirise.json              bounding boxes around the hardware in the crops (COCO); crops where
                                               the hardware could not be identified carry `unboxed_reason`
  crops_lroc-nac_hirise/                       257 crops of 2048×2048 centred on the first 44 objects
    metadata.csv
    moon_lroc-nac/<site>_lroc-nac/<frame>_<site>.png
    mars_hirise/<site>_hirise/<frame>_<site>.png
  derived/                                     views of the catalog, rebuilt with it
    sites_moon-mars.csv                        100 objects: coordinates and their source, arrival, best frame
    frames_lroc-nac_hirise.csv                 every orbital frame covering each object
    rover-frames_mars_hirise.csv               HiRISE frames that caught Curiosity / Perseverance where they were that sol
raw/legacy_source-files/                       what the legacy set was made from (not needed for training)
  <body>_<camera>_originals/                   Clementine JPEGs, HiRISE .IMG EDRs as downloaded
  <body>_<camera>_pds-labels/                  one PDS label (JSON / XML) per image, summarised in the legacy metadata.csv
  squares_mixed-cameras/                       1,240 fixed-size chips cut by the first pipeline
  manifest.csv                                 download log of the first pipeline
```

All images are 8-bit grayscale PNG at native resolution (no resampling), each stretched from its
0.5–99.5 percentile. Scripts that build every file live in the Project Prometheus repository
(`xenarch_lroc_nac_dataset.py`, `xenarch_hirise_dataset.py`, `artifact-site-catalog/`; `xenarch_hf.py` publishes
them under these names).

## Columns

The same name means the same thing in every table:

| Column | Meaning |
|---|---|
| `file_name` | image path relative to the folder of the table (crops: relative to `crops_lroc-nac_hirise/`) |
| `split` | `train` or `validation`; fixed per source frame (a hash of `frame_id`, ~10% validation), so tiles of one frame never straddle the split |
| `body`, `instrument` | `Moon` / `Mars`; `LROC NAC`, `HiRISE RED`, … |
| `landform` | terrain class, = the folder name's middle part |
| `frame_id` | archive product (LROC `M…LC`) or observation (HiRISE `ESP_…`) the image was cut from |
| `lat`, `lon_east` | degrees, planetocentric, east-positive 0–360 |
| `incidence_deg` | sun angle from vertical (0° = sun overhead; 90° = sun at the horizon) |
| `emission_deg`, `phase_deg` | viewing angle from vertical; sun–surface–camera angle |
| `resolution_m_px` | metres per pixel |
| `utc_start`, `solar_longitude_ls` | acquisition time; Mars season (Ls) |

## natural-terrain_training/

**moon_lroc-nac** (LROC NAC CDR, calibrated): six landform folders, `moon_fresh_craters_lroc-nac`,
`moon_mass_wasting_lroc-nac`, `moon_tectonic_lroc-nac`, `moon_volcanic_lroc-nac`, `moon_chains_terrain_lroc-nac`,
`moon_polar_lroc-nac`. Targets come from the IAU gazetteer; each was imaged under several
sun angles; tile 0 is centred on the target and tiles 1–3 are its neighbours across the strip, so `lat`, `lon_east`
are the target's coordinates (`target` is its name). Anything within 10 km of a landing or impact site was skipped.
Also `line0`, `sample0` (tile origin in the source image) and `source_url`.

**mars_hirise** (HiRISE EDR, raw bin-2 RED channels, odd columns repaired): six landform folders,
`mars_aeolian_hirise`, `mars_slopes_hirise`, `mars_periglacial_ice_hirise`, `mars_polar_hirise`, `mars_impacts_hirise`,
`mars_layers_volcanic_hirise`, chosen from the HiRISE team's own rationale per observation (`rationale`) and spread
over sun angles. File names end with the `sub_landform` (dunes, gullies, …). `lat`, `lon_east` are the observation
centre; `channel` is the CCD channel. Observations mentioning hardware or within 15 km of a lander, rover or crash
site were skipped.

Neither set has been checked tile by tile for hardware.

**legacy_mixed-cameras** is the first-phase training set, formerly `xenarch-imagery`, one folder per body and camera.
Mixed bodies, instruments and resolutions (Clementine ~100–200 m/px, CTX ~6 m/px, HiRISE bin 1–4), no landform labels,
no hardware exclusion. `metadata.csv` is summarised from the PDS labels in `raw/`: empty where the archive marks a value
missing (most Clementine angles); HiRISE position, angles and time come from the HiRISE RDR index, its resolution is
nominal (0.25 m × binning). Mk15 trains on it; prefer `moon_lroc-nac/` and `mars_hirise/` for new work.

## human-hardware_ground-truth/

**catalog_moon-mars_lroc-nac_hirise.csv** is the primary table: one row per object per orbital frame that contains it,
with the object (id, name, type, mission, coordinates and their uncertainty, arrival date), the frame (sun and viewing
angles, resolution, `usable`, `after_arrival`, archive URLs), the crop cut from it (`crop_file` relative to
`crops_lroc-nac_hirise/`, `crop_x`, `crop_y`) and its box if any. Frames taken before the object arrived are kept
(`after_arrival` = False): the "before" half of before/after pairs. `usable` = after arrival, sunlit (incidence < 90°)
and resolution ≤ 2 m/px.

Objects (78 Moon, 22 Mars; positions from the LROC team's anthropogenic-objects table, 2026-08-14, unless noted):

- Moon, landed: Apollo 11–17 LMs, ALSEPs, seismometers, retroreflectors, LRVs, Apollo 17 SEP; Surveyor 1/3/5/6/7;
  Luna 16/17/20/21/23/24, Lunokhod 1/2; Chang'e 3/4/5/6, Yutu, Yutu-2; Chandrayaan-3 Vikram and Pragyan; SLIM;
  IM-1 Odysseus, IM-2 Athena; Blue Ghost 1.
- Moon, impacts: Apollo 13–17 S-IVB stages, Apollo 12 LM ascent stage, Surveyor 3/5/6 retrorockets, Ranger 6–9,
  LCROSS Centaur (permanent shadow: no NAC frame shows it), GRAIL Ebb and Flow, LADEE, Beresheet, Longjiang-2,
  Chandrayaan-2 Vikram, Hakuto-R 1 and 2, Luna 25, the 2022 rocket-body double crater, the 2026 Falcon 9 upper stage
  (no archived frame after the impact yet).
- Moon, candidate positions only (`object` says so): Apollo 14 LM ascent stage, SMART-1, Chang'e 1, two rival
  Luna 9 sites ~28 km apart. Yutu-2's position is in the Chinese mission frame (±450 m).
- Mars: Phoenix; Curiosity and Perseverance (landing site and rover per sol); Perseverance heat shield, backshell,
  descent stage and Ingenuity; InSight with heat shield and backshell; Viking 1/2, Pathfinder (radio tracking,
  ±300 m); Spirit and Opportunity final positions; Beagle 2; Tianwen-1 lander. Schiaparelli, Zhurong and the Mars 3
  candidate have only the centre of a HiRISE image aimed at them (±1–3 km).

Not in the catalog (no published position): Luna 13; Apollo 15/17 LM ascent stages; Kaguya, Chandrayaan-1 MIP,
Lunar Prospector; Chang'e 5/6 ascenders; Curiosity's and Phoenix's heat shields and backshells; Schiaparelli's and
Tianwen-1's parachutes and shields; Spirit's and Opportunity's landing hardware.

**crops_lroc-nac_hirise/**: 2048² crops around each object, `label` = `artifact` or `before_arrival` (same ground before the
object arrived). Moon crops are in raw NAC geometry, Mars crops are map-projected. `obj_x`, `obj_y` is the
catalog's nominal pixel, not the measured one: use the boxes file for that. Objects added to the catalog later may have
no crop yet (empty `crop_file`).

**boxes-coco_lroc-nac_hirise.json**: COCO file. `images` lists every crop (`file_name` relative to `crops_lroc-nac_hirise/`,
plus frame metadata); `annotations` holds one `bbox` `[x, y, w, h]` per object per crop, pixels in the 2048² crop, origin
top-left; `categories` has one entry per object (`name` = `site_id` of the catalog). Boxes include the object's shadow. `confidence` is `high` (unambiguous: bright body with hard
shadow, tracks, or matches a published labeled image), `medium` (right size and shadow at the predicted position) or
`low` (could be a rock). `note` says how the box was found. All six Apollo sites are boxed; most small Soviet and
Chinese landers and most Mars rover frames are not (their image record's `unboxed_reason` says why).

**derived/** holds views of the same data, rebuilt with the catalog; edit nothing there by hand.
`sites_moon-mars.csv` (one row per object, with the source of its coordinates), `frames_lroc-nac_hirise.csv` (every
frame per object; `obj_sample_frac` / `obj_line_frac` give the object's position in a raw NAC frame **only
approximately**, off by up to ~1,300 px), `rover-frames_mars_hirise.csv` (`object_lat`, `object_lon_east`: where the
rover was on that sol, from NASA/JPL waypoints).

## Credit

Moon: NASA / GSFC / Arizona State University (LROC). Mars: NASA / JPL / University of Arizona (HiRISE).
Legacy set: NASA PDS (Clementine, MRO CTX / HiRISE, MGS MOC), ISRO (Chandrayaan-2 TMC-2).
Source data are public domain; this compilation is CC0.

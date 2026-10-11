---
license: cc0-1.0
tags: [moon, mars, lroc, hirise, planetary-science, anomaly-detection, technosignatures, remote-sensing]
pretty_name: Xenarch artifact site catalog
size_categories: [1K<n<10K]
task_categories: [image-classification, object-detection]
configs:
  - config_name: images
    data_dir: images
    default: true
  - config_name: sites
    data_files: sites.csv
  - config_name: frames
    data_files: frames.csv
  - config_name: mars_mission_frames
    data_files: mars_mission_frames.csv
---
# Xenarch artifact site catalog

Known human-made objects on the Moon and Mars, plus every orbital frame that covers them, with sun angle and
resolution. This is ground truth for Xenarch, an unsupervised anomaly detector trained only on natural terrain:
these are the places where it *should* raise a flag.

`images/` holds 2048×2048 crops centred on each object, cut from the source frames. Every frame row in the
catalog also gives the source frame ID and archive URL.

## Contents

| File | Rows | One row per |
|---|---|---|
| `sites.csv` | 44 | object: coordinates, uncertainty, arrival date, frame counts, best frame |
| `frames.csv` | 2,298 | object × frame whose footprint contains the object |
| `mars_mission_frames.csv` | 57 | HiRISE frame containing the Curiosity or Perseverance rover where it was that sol, plus ballast / balance-mass impact zones |
| `images/` | 257 | 2048×2048 PNG crop centred on an object (`images/metadata.csv`) |
| `build_catalog.py` | — | rebuilds the three CSV files (runs inside the Project Prometheus repo) |
| `download_images.py` | — | cuts the crops (needs `opj_decompress` from OpenJPEG) |

### Objects

**Moon: LROC NAC frames, 0.4–2 m/pixel**

- Apollo 11, 12, 14, 15, 16, 17: LM descent stage, ALSEP / PSE central station, LRV (15–17), S-IVB impact craters (13–17)
- Surveyor 1, 3, 5, 6, 7
- Luna 16, 17, 20, 21, 23, 24, plus Lunokhod 1 and 2 at their final positions
- Chang'e 3 (and Yutu), 4, 5, 6

**Mars: HiRISE RED frames, 0.25–1 m/pixel**

- Phoenix, Curiosity (Bradbury Landing), InSight, Perseverance (Octavia E. Butler Landing)
- Curiosity and Perseverance rovers along their traverses (position on each frame's sol from NASA/JPL waypoints)

## Columns

`sites.csv`: `site_id`, `body`, `mission`, `object`, `object_type`, `lat`, `lon_east` (planetocentric, 0–360° E),
`elevation_m`, `coord_uncertainty_m`, `arrival_utc_date`, `frames_total`, `frames_after_arrival`,
`frames_usable`, `best_resolution_m_px`, `best_frame` (finest usable frame, lowest emission breaks ties),
`incidence_min_deg` / `incidence_max_deg` (over usable frames), `coord_source`.

`frames.csv` and `mars_mission_frames.csv`:

| Column | Meaning |
|---|---|
| `frame_id` | LROC NAC product ID (e.g. `M175124932RC`) or HiRISE observation ID (e.g. `ESP_073068_1985`) |
| `usable` | `True` if after arrival, sunlit (incidence < 90°) and ≤ 2 m/pixel |
| `after_arrival`, `days_since_arrival` | `False` / negative = taken **before** the object was there: a before/after pair |
| `incidence_deg` | sun angle from vertical (0° = sun overhead, 90° = sun on the horizon) |
| `sun_elevation_deg` | 90 − incidence |
| `sun_azimuth_image_deg` | HiRISE only: sub-solar azimuth, clockwise from the image's right edge |
| `emission_deg`, `phase_deg` | viewing geometry |
| `resolution_m_px` | NAC: pixel scale from ODE; HiRISE: RDR map scale (0.25 / 0.5 / 1 m) |
| `local_solar_time_h`, `solar_longitude_ls` | HiRISE only (Mars season) |
| `obj_sample_frac`, `obj_line_frac` | NAC only: object position in the raw frame, 0–1 across samples / down lines (from the ODE footprint corners, which are rounded to 0.01°: off by up to ~150–300 m) |
| `team_rationale` | why the instrument team took the frame |
| `image_url`, `label_url`, `browse_url` | NAC CDR `.IMG` / PDS4 label / LROC viewer; HiRISE RDR `.JP2` / `.LBL` / uahirise.org page |

`mars_mission_frames.csv` has five more columns:

| Column | Meaning |
|---|---|
| `object_lat`, `object_lon` | where the rover was when the frame was taken (last waypoint up to that sol) |
| `rover_sol` | mission sol of the frame |
| `position_uncertainty_m` | 0 if the rover did not drive that sol; otherwise the distance between its start and end of sol (it may be at either) |
| `found_by` | `rover waypoint`, or `team rationale` for ballast / balance-mass impact zones (position in frame unknown; `object_lat/lon` empty) |

## Summary

| | Moon (NAC) | Mars (HiRISE) |
|---|---|---|
| object × frame rows | 2,149 | 149 |
| usable | 1,976 | 132 |
| median resolution (usable) | 0.92 m/px | 0.25 m/px |
| incidence 0–45° / 45–65° / 65–80° / 80–90° | 859 / 545 / 369 / 203 | 16 / 94 / 12 / 10 |

Every object has at least 13 usable frames. Chang'e 3–6 and all four Mars sites also have frames taken
before arrival. Rovers: 40 frames contain Curiosity and 13 contain Perseverance where they were that sol;
21 of these were taken on a drive sol (`position_uncertainty_m` > 0, at most 119 m).

## Images

`images/{moon,mars}/<site_id>/<frame_id>.png` (720 MB): grayscale, 8-bit, each crop stretched from its 0.5–99.5 percentile.
Native resolution, no resampling: 1 pixel = `resolution_m_px` metres. Images are in the source frame's own
orientation: raw NAC geometry for the Moon, map-projected RDR for Mars (north up, except polar-stereographic
Phoenix frames, where up points to the pole). Polar RDRs are projected with the polar radius (3376.2 km)
although their labels say 3396.19 km; positions use the radius implied by MAP_RESOLUTION × MAP_SCALE.

| | Moon (LROC NAC) | Mars (HiRISE) |
|---|---|---|
| crops | 131 | 126 |
| `artifact` / `before_arrival` | 127 / 4 | 118 / 8 |
| objects (own crop or shared) | 38 | 6 (incl. Curiosity rover 38 crops, Perseverance rover 12) |

Sun incidence of the 245 `artifact` crops: 0–45° 53, 45–65° 112, 65–80° 42, 80–90° 38. All 44 catalogued
objects appear in at least one crop (`objects_in_crop`). Not cut: 6 HiRISE frames where the object sits at the
strip edge (32–67% of the crop empty) and, for Phoenix / InSight / the landing sites, frames beyond 12 per site
(the HiRISE server gives ~57 KB/s); `frames.csv` lists them all.

`images/metadata.csv` (Hugging Face imagefolder):

| Column | Meaning |
|---|---|
| `file_name`, `site_id`, `object`, `mission`, `body` | crop and the object it is centred on |
| `label` | `artifact`, or `before_arrival`: the same ground before the object arrived |
| `obj_x`, `obj_y` | catalogue position of the object in the crop (pixels) |
| `objects_in_crop` | every catalogued object in the crop, `site_id@x,y` (Apollo LM, ALSEP and LRV share crops) |
| `frame_id`, `instrument`, `utc_start`, `after_arrival`, `days_since_arrival` | source frame |
| `incidence_deg`, `sun_elevation_deg`, `emission_deg`, `phase_deg`, `resolution_m_px` | sun, viewing geometry, scale |
| `crop_x0`, `crop_y0` | crop origin in the source frame (sample, line) |
| `position_note`, `source_url` | how the position was found; source file |

**The object is in the crop, but not always at `obj_x, obj_y`.** NAC positions come from ODE footprints rounded
to 0.01° and HiRISE from each RDR's map projection, so the object can sit 100–300 m from that pixel (Apollo 11
in M175124932RC: ~220 px; Curiosity in ESP_061961_1750: ~360 px). The 2048 px crop leaves room for that error.
Exact per-frame positions (image registration) are not done yet.

Which frames were cut:
- **Moon:** per object, the sharpest usable frame in each sun-incidence bin (0–45°, 45–65°, 65–80°, 80–90°),
  plus one sunlit frame from before arrival when there is one (Chang'e 3–6). Only the 2048 lines through the
  object are downloaded (HTTP range, ~20 MB per crop).
- **Mars:** every usable rover frame; for Phoenix, InSight and the two landing sites up to 12 sunlit frames
  spread over sun angle and date (Phoenix has 40, from a first pass), plus 2 from before arrival. The HiRISE RDRs are
  ~500 MB JPEG 2000 files ordered by position (PCRL) with packet lengths in PLT markers. Only the packets of
  the precincts around the object are downloaded (~5 MB), rebuilt into a valid codestream with empty packets
  elsewhere, and decoded with `opj_decompress -d`.
- Crops more than 25% missing or saturated were dropped.

## How it was built

- **Coordinates.** Moon: the LROC team's *Coordinates of Lunar Anthropogenic Targets* (28 July 2016,
  Wagner et al. 2016, Icarus 283), mean ME-frame positions measured on many NAC images, mostly ±0.3–20 m.
  Chang'e 4, 5 and 6 come from the LROC posts listed in `coord_source`. Mars: published landing coordinates
  (Golombek et al. 2020 for InSight). Curiosity and Perseverance appear twice: the landing site (descent
  stage blast zone, fixed) and the rover, which has driven ~38 km and ~45 km. Rover positions come from the
  NASA/JPL MMGIS waypoint files (position after every drive; Curiosity to sol 5021, Perseverance to sol 1980).
  For Curiosity and Perseverance, `after_arrival` uses the touchdown time, so the HiRISE images of the descent
  under parachute count as before arrival.
- **Moon frames.** NASA PDS Orbital Data Explorer (ODE) search for LROC NAC CDR products around each object.
  A frame is kept only if its footprint quadrilateral contains the object.
- **Mars frames.** HiRISE RDR cumulative index (RED products, through 2026-07-30). A frame is kept if its
  four corners contain the site. Rover frames: for each HiRISE frame, the sol is computed from its UTC time and
  the rover's position on that sol is looked up; the frame is kept if its corners contain that position. Frames
  whose rationale names the rover but that do not contain it ("Curiosity Rover field site", 2–38 km away) are
  left out. The team rationale is used only for the ballast / balance-mass impact zones, which have no
  published coordinates.

Not included yet: Ranger, GRAIL and LADEE impact sites; Luna 9 and 13; Chandrayaan-3, SLIM, IM-1, Blue Ghost
and Yutu-2; Mars heat shields, backshells and parachutes; Spirit, Opportunity, Viking, Pathfinder, Zhurong.

## Credit

LROC images: NASA / GSFC / Arizona State University. HiRISE images: NASA / JPL / University of Arizona.
Search: NASA PDS Geosciences Node ODE (Washington University in St. Louis). The catalog itself is CC0.

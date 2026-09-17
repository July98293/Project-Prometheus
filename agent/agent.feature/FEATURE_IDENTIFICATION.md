# Feature identification guide — single-feature classes

Companion to `SOURCES.md` and `MARS_BENCH.md`. Where those two are about *where the
exemplars and priors come from*, this file is about *how the feature agent tells one
class from another on a single flagged chip*, class by class, using its own tools
(`measure`, `profile`, `shadow_height`, `crater_morphometry`, `nearest_exemplars`) and
citable morphology criteria — independent of Mars-Bench and not Mars-only. Use it to
write the per-class sections of the feature agent's system prompt and as a checklist
when hand-labeling exemplars.

For each class: what it looks like, how big it is, what measurement actually
discriminates it, what else it gets confused with, and the source the criterion comes
from.

## How to read the "discriminator" column

Every class here should be closeable with one or two tool calls, not eyeballing. If a
class has no discriminator beyond "looks like X", it is not ready for the taxonomy —
go back to the reference and find the quantitative criterion.

## 1. Impact craters

| Subtype | Visual signature | Size range | Discriminator | Confused with | Source |
|---|---|---|---|---|---|
| Simple crater, fresh | Bowl-shaped depression, sharp raised rim, continuous ejecta blanket, high depth/diameter | mm–15 km (simple/complex transition ~15–20 km Moon, ~5–7 km Mars) | `crater_morphometry(D, d)`: fresh simple craters sit at d/D ≈ 0.2 (Pike 1977); rim height/D ≈ 0.04. Sharp rim in `profile()` (steep rim-floor-rim, minimal rounding) | Pit (no ejecta, no rim), volcanic cone (rim without floor depression at the same depth ratio), CCD dropout (no shadow-consistent relief) | Pike 1977; Melosh, *Impact Cratering* |
| Simple crater, degraded | Shallower bowl, subdued/discontinuous rim, no visible ejecta | same as fresh | d/D below the fresh trend; rim height indistinct in `profile()`; degradation class from Fassett & Thomson stages (crisp → rounded → subdued → ghost) | Secondary crater (irregular outline instead of degraded-but-round), dune trough | Fassett & Thomson 2014; Stopar et al. 2017 (small-crater morphometry from NAC DTMs) |
| Complex crater | Central peak or peak ring, terraced walls, flat floor, d/D lower than simple trend | > simple/complex transition | `profile()` across the floor shows a central rise; `crater_morphometry` flags D above transition diameter for the body | Volcanic caldera (no ejecta blanket, no bowl at rim) | Melosh, *Impact Cratering* |
| Secondary crater / crater chain / crater field | Irregular, elongated or overlapping bowls, shallow relative to primaries of the same size, often radial to a larger primary, no continuous individual ejecta | typically m–few km | `measure()`: elongation/irregular outline vs. simple-crater symmetry score; `relations()` (spatial agent) for radial alignment to a primary — feature agent flags the shape, spatial confirms the radial pattern | Boulder field, dust-devil track cluster | Melosh, *Impact Cratering*; DoMars16k "cra/sfx" classes |

## 2. Boulders and boulder tracks

| Subtype | Visual signature | Size range | Discriminator | Confused with | Source |
|---|---|---|---|---|---|
| Boulder | Small bright blob with a sharp dark shadow on the anti-sun side; no ejecta, no rim | ~0.5–20 m | `shadow_height(bbox)`: length × tan(sun elevation) gives real height; boulders are near-equant (major/minor axis ratio close to 1) and height roughly comparable to width, unlike a shallow bright patch with no matching shadow | Bright albedo spot / small fresh crater seen at low sun (crater has a rim profile in `profile()`, a boulder does not) | Wagner et al. 2017 (shadow heights); Krishna & Kumar 2016 (size–frequency) |
| Boulder trail / rockfall track | Linear or curvilinear dark (or bright) streak running downslope from a source boulder or scarp, tapering or ending at the boulder | tens–hundreds of m long | `profile()` along the track: monotonic width/brightness trend from source to terminus; track originates at a `shadow_height`-confirmed boulder or at a scarp top, and requires slope (spatial agent's `terrain()`) | Dust-devil track (does not originate at a boulder; crosses terrain irrespective of slope), secondary crater chain | Bickel et al. 2020 (lunar rockfall map); Bickel et al. 2020, IEEE JSTARS (Mars rockfalls) |

## 3. Aeolian and slope features (mostly Mars)

| Subtype | Visual signature | Size range | Discriminator | Confused with | Source |
|---|---|---|---|---|---|
| Dust-devil track | Dark (sometimes bright) sinuous or branching streak on the surface, no topographic relief, crosses craters/ridges without deflection, fades over time | tens of m – several km long, ~m wide | No `shadow_height` signal (zero relief) — this is the key negative test; `profile()` across the track shows a brightness/albedo change only, no rim or depth; path often meandering/branching, not confined to slope-driven paths | Boulder trail (has relief and a source boulder), scarp (has shadow/relief), gully channel (confined to a slope, has depth) | Reiss et al. 2016 |
| Dune / aeolian bedform | Regular, periodic ridges, often barchan (crescent) or transverse, asymmetric profile (gentle windward, steep lee slope) | ridge spacing ~10–100s of m | `profile()` perpendicular to the ridge crest: asymmetric slope profile (lee slope near the angle of repose, steeper than the windward slope); repeats periodically in `scene_objects()` (spatial agent) | Wrinkle ridge (symmetric profile, much larger, tectonic not aeolian), ripples (same asymmetry, much smaller period) | Mars Global Digital Dune Database; DoMars16k "ael/aec" |
| Gully | Alcove-channel-apron system on a slope: a bowl-shaped source alcove, one or more incised channels running downslope, a fan-shaped debris apron at the base | tens–hundreds of m long | `profile()` down the channel shows monotonic descent with a concave source and convex apron; requires a slope (spatial `terrain()`); channel has measurable depth (`measure`), unlike a dust-devil track | Dust-devil track (no channel depth, not confined by slope), RSL (no persistent channel/apron, seasonal streak instead) | Malin & Edgett 2000 |
| RSL (recurring slope lineae) | Narrow dark streaks on steep, sun-facing slopes, incrementally lengthen downslope over a season, no coarse apron | ~0.5–5 m wide, tens–hundreds of m long | Restricted to slopes above ~25–30° (spatial `terrain()`); no measurable channel depth in `profile()` (albedo change only, like a dust-devil track but confined to slope and oriented strictly downslope) | Dust-devil track (not slope-confined), gully channel (has depth and an apron) | McEwen et al. 2011 |
| Slope streak | Dark (or occasionally bright) fan- or tongue-shaped streak on a dust-mantled slope, widens downslope from a point source, no channel incision | tens–hundreds of m long | `measure()`: fan widens monotonically from an apex; `profile()` shows no depth (surface albedo change only), distinguishing it from a gully channel | Gully (has an incised channel + apron), RSL (RSL is narrow and roughly constant-width, not fan-shaped) | DoMars16k "fse" class |
| Mass wasting deposit | Lobate debris apron at the base of a slope, hummocky surface texture, no single channel | slope-base scale, 10s–100s of m | `profile()` shows a lobate, convex-outward toe; texture is rough/hummocky rather than smooth apron of a gully | Gully apron (part of a channel-alcove system), landslide (larger, may have longitudinal ridges) | DoMars16k "fss" class |

## 4. Tectonic and volcanic landforms

| Subtype | Visual signature | Size range | Discriminator | Confused with | Source |
|---|---|---|---|---|---|
| Scarp / ridge (lobate scarp, wrinkle ridge) | Long, linear-to-sinuous topographic step or ridge, roughly constant width and relief along strike | 100s of m – 10s of km long | `profile()` perpendicular to strike: asymmetric, roughly triangular cross-section for a wrinkle ridge; `shadow_height` gives relief consistent along the length | Dune (periodic and much shorter-wavelength), CCD seam artifact (perfectly straight, fixed image-column position, no consistent relief) | Encyclopedia of Planetary Landforms |
| Mound / volcanic cone | Roughly circular to elliptical positive-relief bump, may have a summit pit | 10s–100s of m diameter | `shadow_height` confirms positive relief all around, not one-sided like a crater rim; `profile()` radial from centre is symmetric-convex, opposite sign from a crater's concave bowl | Simple crater (concave, not convex, in `profile()`), boulder (much smaller, equant, no summit pit) | DoMars16k "sfe" class; Mars-Bench conequest sets |
| Pit / skylight | Circular to sub-circular opening with little to no raised rim and no ejecta; floor often not visible (shadow-filled) at any sun angle | m–100s of m diameter | `profile()`: floor is dark/indeterminate rather than showing a bowl reflectance gradient; no ejecta blanket and near-zero rim height, unlike a crater at the same diameter | Fresh simple crater (has rim + ejecta + visible bowl floor) | LROC pit atlas (Wagner & Robinson) |

## 5. Rare-but-natural (checked against the ContextBrief's `rare_natural_nearby`)

| Subtype | Visual signature | Size range | Discriminator | Confused with | Source |
|---|---|---|---|---|---|
| Irregular mare patch (IMP) | Small, morphologically fresh-looking mound-and-hollow terrain, sharp albedo contrast, no comparably fresh craters nearby of the same crater-retention age | 100s of m – a few km | Only flagged as this class where the brief lists a known IMP nearby (e.g. Ina) — otherwise treat the anomaly as unexplained rather than guessing IMP; `measure()` records the mound/hollow relief pattern for comparison to the known catalog examples | Ordinary mound field, degraded crater cluster | Braden et al. 2014 |
| Lunar swirl | High-albedo sinuous or whorl-shaped bright marking with no topographic expression | 100s of m – 10s of km | `shadow_height` returns ~0 (no relief) despite strong albedo contrast — the diagnostic negative test, same logic as dust-devil tracks but stationary, not track-shaped | Fresh crater ray / bright ejecta (has a source crater with relief), dust-devil track (network/branching shape, associated with wind streaks not whorls) | Denevi et al. 2016 |

## 6. Instrument artifacts (must be ruled out before any natural class is assigned)

| Subtype | Visual signature | Discriminator | Source |
|---|---|---|---|
| CCD boundary / seam offset | Perfectly straight brightness or geometric step, fixed pixel column/row, spans the full image or a large fraction of it | `measure()` edge orientation is exactly aligned to the image axes and its position matches the known CCD boundary column for the instrument (from `instrument_sheet`, Spatial/Literature agent); no shadow, no relief | McEwen et al. 2007 (HiRISE); Humm et al. 2016 (LROC NAC) |
| Saturation bleed | Bright smear/streak radiating along the readout direction from an overexposed source | Streak direction matches the sensor's readout axis, not the sun azimuth; brightness clips at the sensor's max DN | Delamere et al. 2010 |
| Dropout / jitter | Dark row/column gaps, or a locally shifted/repeated strip of image content | Geometric discontinuity aligned to scan direction, not to any real edge in the scene; no consistent shadow relief across the discontinuity | Delamere et al. 2010; Chowdhury et al. 2020 (TMC-2) |

## 7. Hardware (human-made objects)

| Visual signature | Size range | Discriminator | Source |
|---|---|---|---|
| Small, high-albedo-contrast object(s) with unnaturally regular geometry (rectilinear, radially symmetric, or a cluster of a lander + legs/blast zone + rover tracks) | ~1–10 m | `shadow_height` gives a height inconsistent with any natural class at that footprint width; `measure()` finds right angles / perfect circularity that natural weathered material does not retain at this scale; check against `known_human_objects` in the ContextBrief (distance to a known landing site) | LROC Apollo site images; HiRISE lander/rover catalog images |

## Using this table

1. Rule out instrument artifacts first (§6) — they have no shadow-relief signature
   consistent with the sun geometry, and their geometry is tied to the sensor, not the
   scene.
2. Test relief with `shadow_height`: zero relief with strong albedo contrast routes to
   dust-devil track, RSL, slope streak, or swirl (§3, §5); positive relief routes to
   boulder, mound/cone, hardware; negative relief (a depression) routes to crater or
   pit.
3. Within a relief bucket, `profile()` shape (bowl vs. fan vs. periodic vs. lobate)
   and `measure()` (aspect ratio, symmetry, width trend) separate the remaining
   classes.
4. Cross-check size and expected class against the ContextBrief's
   `expected_landforms` before calling something "unexplained".
5. When two classes remain tied, call `nearest_exemplars(k)` and let the labeled
   exemplar chips break the tie.

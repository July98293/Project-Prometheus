"""
Xenarch artifact site catalog — known human-made objects on the Moon and Mars, and every orbital frame that
covers them, with sun angle and resolution. The positive (anomaly) counterpart of the natural-terrain sets.

  Moon  Apollo LMs, ALSEPs, LRVs and S-IVB impact craters; Surveyor 1-7; Luna 16-24 and Lunokhods; Chang'e 3-6.
        Frames: LROC NAC calibrated images (CDRNAC4) whose footprint contains the object, found with NASA ODE.
  Mars  Phoenix, Curiosity, InSight and Perseverance landing sites.
        Frames: HiRISE RED observations whose footprint contains the site, from the HiRISE RDR cumulative index
        (cached in data/cache/hirise/ by xenarch_hirise_dataset.py).
        mars_mission_frames.csv: frames containing the Curiosity / Perseverance rover where it was on that sol
        (NASA/JPL waypoints, cached in data/cache/rovers/), plus frames whose team rationale names the ballast /
        balance-mass impact zones (position in frame unknown).

Writes sites.csv (one row per object, rovers at their latest position), frames.csv (one row per object × frame)
and mars_mission_frames.csv next to this script. Frames taken before the object arrived are kept and flagged
(after_arrival = False): they are the "before" half of before/after pairs. Nothing is downloaded except ODE
search results, the HiRISE index and the rover waypoint files.

  .venv/bin/python artifact-site-catalog/build_catalog.py
"""
import csv, json, math, re, sys, time, urllib.parse
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from xenarch_live import LiveRun
from xenarch_lroc_nac_dataset import ODE, http, locate, unit

import numpy as np

HIRISE_INDEX = HERE.parent / "data/cache/hirise"
ODE_CACHE = HERE.parent / "data/cache/ode_nac"
HIRISE_PDS = "https://hirise-pds.lpl.arizona.edu/PDS"

LROC_2016 = "LROC 2016 Coordinates of Lunar Anthropogenic Targets (Wagner et al. 2016, Icarus)"
# site_id, body, mission, object, object_type, lat, east lon, elevation m, uncertainty m, arrival (UTC date), source
SITES = [
    ("apollo11_lm", "Moon", "Apollo 11", "Apollo 11 LM descent stage", "crewed lander", 0.67416, 23.47314, -1926, 0.3, "1969-07-20", LROC_2016),
    ("apollo11_pse", "Moon", "Apollo 11", "Apollo 11 Passive Seismic Experiment", "surface experiment", 0.67322, 23.47315, -1928, 0.4, "1969-07-20", LROC_2016),
    ("apollo12_lm", "Moon", "Apollo 12", "Apollo 12 LM descent stage", "crewed lander", -3.0128, 336.5781, -1422, 2.2, "1969-11-19", LROC_2016),
    ("apollo12_alsep", "Moon", "Apollo 12", "Apollo 12 ALSEP central station", "surface experiment", -3.0098, 336.5751, -1422, 2.2, "1969-11-19", LROC_2016),
    ("apollo14_lm", "Moon", "Apollo 14", "Apollo 14 LM descent stage", "crewed lander", -3.64589, 342.52806, -1062, 0.4, "1971-02-05", LROC_2016),
    ("apollo14_alsep", "Moon", "Apollo 14", "Apollo 14 ALSEP central station", "surface experiment", -3.64419, 342.52232, -1064, 0.4, "1971-02-05", LROC_2016),
    ("apollo15_lm", "Moon", "Apollo 15", "Apollo 15 LM descent stage", "crewed lander", 26.13239, 3.63330, -1931, 1.0, "1971-07-30", LROC_2016),
    ("apollo15_alsep", "Moon", "Apollo 15", "Apollo 15 ALSEP central station", "surface experiment", 26.13406, 3.62991, -1924, 0.5, "1971-07-30", LROC_2016),
    ("apollo15_lrv", "Moon", "Apollo 15", "Apollo 15 Lunar Roving Vehicle", "rover", 26.13174, 3.63803, -1928, 0.5, "1971-07-30", LROC_2016),
    ("apollo16_lm", "Moon", "Apollo 16", "Apollo 16 LM descent stage", "crewed lander", -8.9734, 15.5011, 8, 3.0, "1972-04-21", LROC_2016),
    ("apollo16_alsep", "Moon", "Apollo 16", "Apollo 16 ALSEP central station", "surface experiment", -8.9759, 15.4986, 12, 3.0, "1972-04-21", LROC_2016),
    ("apollo16_lrv", "Moon", "Apollo 16", "Apollo 16 Lunar Roving Vehicle", "rover", -8.9729, 15.5037, 10, 3.7, "1972-04-21", LROC_2016),
    ("apollo17_lm", "Moon", "Apollo 17", "Apollo 17 LM descent stage", "crewed lander", 20.1911, 30.7723, -2626, 3.5, "1972-12-11", LROC_2016),
    ("apollo17_alsep", "Moon", "Apollo 17", "Apollo 17 ALSEP central station", "surface experiment", 20.1923, 30.7655, -2622, 3.5, "1972-12-11", LROC_2016),
    ("apollo17_lrv", "Moon", "Apollo 17", "Apollo 17 Lunar Roving Vehicle", "rover", 20.1896, 30.7769, -2628, 3.2, "1972-12-11", LROC_2016),
    ("apollo13_sivb", "Moon", "Apollo 13", "Apollo 13 S-IVB impact crater", "impact crater", -2.5550, 332.1125, -1166, 9.8, "1970-04-15", LROC_2016),
    ("apollo14_sivb", "Moon", "Apollo 14", "Apollo 14 S-IVB impact crater", "impact crater", -8.1810, 333.9695, -1785, 4.5, "1971-02-04", LROC_2016),
    ("apollo15_sivb", "Moon", "Apollo 15", "Apollo 15 S-IVB impact crater", "impact crater", -1.2897, 348.1755, -1109, 3.8, "1971-07-29", LROC_2016),
    ("apollo16_sivb", "Moon", "Apollo 16", "Apollo 16 S-IVB impact crater", "impact crater", 1.9210, 335.3770, -1104, 19.2, "1972-04-19", LROC_2016),
    ("apollo17_sivb", "Moon", "Apollo 17", "Apollo 17 S-IVB impact crater", "impact crater", -4.1681, 347.6693, -1206, 5.6, "1972-12-10", LROC_2016),
    ("surveyor1", "Moon", "Surveyor 1", "Surveyor 1 lander", "robotic lander", -2.4745, 316.6602, -1889, 5.6, "1966-06-02", LROC_2016),
    ("surveyor3", "Moon", "Surveyor 3", "Surveyor 3 lander", "robotic lander", -3.0162, 336.5820, -1433, 2.6, "1967-04-20", LROC_2016),
    ("surveyor5", "Moon", "Surveyor 5", "Surveyor 5 lander", "robotic lander", 1.4551, 23.1943, -2052, 11.9, "1967-09-11", LROC_2016),
    ("surveyor6", "Moon", "Surveyor 6", "Surveyor 6 lander", "robotic lander", 0.4742, 358.5725, -757, 6.7, "1967-11-10", LROC_2016),
    ("surveyor7", "Moon", "Surveyor 7", "Surveyor 7 lander", "robotic lander", -40.9812, 348.4873, 81, 3.5, "1968-01-10", LROC_2016),
    ("luna16", "Moon", "Luna 16", "Luna 16 lander", "robotic lander", -0.5137, 56.3638, -2452, 3.7, "1970-09-20", LROC_2016),
    ("luna17", "Moon", "Luna 17", "Luna 17 lander", "robotic lander", 38.23764, 324.99837, -2471, 0.5, "1970-11-17", LROC_2016),
    ("lunokhod1", "Moon", "Luna 17", "Lunokhod 1 rover (final position)", "rover", 38.3150, 324.9919, -2471, 3.8, "1970-11-17", LROC_2016),
    ("luna20", "Moon", "Luna 20", "Luna 20 lander", "robotic lander", 3.7863, 56.6242, -1780, 4.3, "1972-02-21", LROC_2016),
    ("luna21", "Moon", "Luna 21", "Luna 21 lander", "robotic lander", 25.9994, 30.4076, -2680, 13.4, "1973-01-15", LROC_2016),
    ("lunokhod2", "Moon", "Luna 21", "Lunokhod 2 rover (final position)", "rover", 25.8323, 30.9222, -2761, 7.5, "1973-01-15", LROC_2016),
    ("luna23", "Moon", "Luna 23", "Luna 23 lander", "robotic lander", 12.6667, 62.1511, -3668, 3.0, "1974-11-06", LROC_2016),
    ("luna24", "Moon", "Luna 24", "Luna 24 lander", "robotic lander", 12.7142, 62.2129, -3670, 3.7, "1976-08-18", LROC_2016),
    ("change3", "Moon", "Chang'e 3", "Chang'e 3 lander", "robotic lander", 44.1214, 340.4883, -2630, 9.1, "2013-12-14", LROC_2016),
    ("yutu", "Moon", "Chang'e 3", "Yutu rover (final position)", "rover", 44.1208, 340.4878, -2630, 12.9, "2013-12-14", LROC_2016),
    ("change4", "Moon", "Chang'e 4", "Chang'e 4 lander", "robotic lander", -45.4561, 177.5885, -5927, 3.0, "2019-01-03",
     "LROC, Chang'e 4 Lander Coordinates (lroc.asu.edu/posts/1087, NAC DTM)"),
    ("change5", "Moon", "Chang'e 5", "Chang'e 5 lander", "robotic lander", 43.0576, 308.0839, -2570, 20, "2020-12-01",
     "LROC, First Look: Chang'e 5 (lroc.asu.edu/posts/1172)"),
    ("change6", "Moon", "Chang'e 6", "Chang'e 6 lander", "robotic lander", -41.6385, 206.0148, -5256, 30, "2024-06-01",
     "LROC, First Look: Chang'e 6 (lroc.asu.edu/images/1374)"),
    ("phoenix", "Mars", "Phoenix", "Phoenix lander", "robotic lander", 68.218830, 234.250778, None, 50, "2008-05-25",
     "NASA Phoenix landing site (areocentric, from HiRISE)"),
    ("curiosity_landing", "Mars", "Curiosity", "Curiosity landing site (Bradbury Landing)", "rover landing site", -4.5895, 137.4417, None, 50, "2012-08-06T05:17:57",
     "MSL / Malin Space Science Systems (MARDI + HiRISE)"),
    ("insight", "Mars", "InSight", "InSight lander", "robotic lander", 4.502384, 135.623447, -2613, 5, "2018-11-26",
     "Golombek et al. 2020, Earth and Space Science"),
    ("perseverance_landing", "Mars", "Perseverance", "Perseverance landing site (Octavia E. Butler Landing)", "rover landing site", 18.4447, 77.4508, None, 50, "2021-02-18T20:43:49",
     "NASA / JPL Mars 2020 landing site"),
]

# HiRISE rationale text that names a mission (for rover-tracking observations away from the landing site).
# Kept only when it names the hardware itself, not the area around it or a planned route.
MISSION_WORDS = {"Phoenix": r"phoenix", "Curiosity": r"curiosity|\bmsl\b|mars science lab",
                 "InSight": r"insight", "Perseverance": r"perseverance|mars 2020|\bm2020\b|ingenuity"}
HARDWARE_NAMED = re.compile(r"rover|lander|helicopter|ingenuity|ballast|balance mass|heat ?shield|parachute|backshell|"
                            r"descent stage|field site|track", re.I)
NOT_THE_HARDWARE = re.compile(r"\bnear\b|surrounding|\b(west|east|north|south)\b of|downrange|future|possible (traverse|route|msl|rover)|"
                              r"traverse route|candidate|proposed|potential|investigated by|landing site ellipse", re.I)
# landers and rovers are placed by coordinates; the rationale is only used for hardware with no published position
NOT_THE_ROVER = re.compile(r"ballast|balance mass|heat ?shield|parachute|backshell|descent stage", re.I)

# Rover positions after every drive, published by NASA/JPL (MMGIS). Sol 0 starts at local midnight before landing:
# landing UTC minus the landing local mean solar time (one Mars sol = 88775.244 s).
ROVERS = {"Curiosity": dict(url="https://mars.nasa.gov/mmgis-maps/MSL/Layers/json/MSL_waypoints.json",
                            landing="2012-08-06T05:17:57", landing_lmst_h=15.00, site="curiosity_landing"),
          "Perseverance": dict(url="https://mars.nasa.gov/mmgis-maps/M20/Layers/json/M20_waypoints.json",
                               landing="2021-02-18T20:43:49", landing_lmst_h=15.88, site="perseverance_landing")}
SOL_S = 88775.244
ROVER_CACHE = HERE.parent / "data/cache/rovers"

MAX_RES_M = 2.0    # coarser frames (off-nadir, high orbit) cannot show a lander; kept but not `usable`

FRAME_FIELDS = ["site_id", "object", "mission", "body", "frame_id", "instrument", "usable", "utc_start", "after_arrival",
                "days_since_arrival", "incidence_deg", "sun_elevation_deg", "sun_azimuth_image_deg", "emission_deg",
                "phase_deg", "resolution_m_px", "local_solar_time_h", "solar_longitude_ls", "obj_sample_frac",
                "obj_line_frac", "frame_center_lat", "frame_center_lon", "team_rationale", "image_url", "label_url",
                "browse_url"]
MISSION_FIELDS = FRAME_FIELDS + ["object_lat", "object_lon", "rover_sol", "position_uncertainty_m", "found_by"]

live = None


def _log(msg):
    print(msg, flush=True)
    if live:
        live.log(msg)


def inside(lat, lon, corners):
    """True if (lat, east lon) lies in the quadrilateral `corners` [(lat, lon) × 4] — tested on the tangent plane."""
    t = unit(lat, lon)
    east = np.cross([0, 0, 1], t)
    east = east / np.linalg.norm(east) if np.linalg.norm(east) > 1e-9 else np.array([1., 0, 0])
    north = np.cross(t, east)
    P = [np.array([unit(a, b) @ east, unit(a, b) @ north]) / (unit(a, b) @ t) for a, b in corners]
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
    sign = [np.sign(cross(P[(k + 1) % 4] - P[k], -P[k])) for k in range(4)]
    return all(s >= 0 for s in sign) or all(s <= 0 for s in sign)


def days_since(utc, arrival):
    return (datetime.fromisoformat(utc[:10]) - datetime.fromisoformat(arrival[:10])).days


def after(utc, arrival):
    """Arrival is a date, or a touchdown time where a descent image exists (the object was still in the air)."""
    return utc[:len(arrival)] >= arrival


# ── Moon: LROC NAC frames from ODE ─────────────────────────────────────────
def nac_frames(site):
    sid, body, mission, obj, _, lat, lon, _, _, arrival, _ = site
    cache = ODE_CACHE / f"{sid}.json"      # ODE search results; delete the folder to search again
    d, prods, offset = 0.002, [], 0
    dl = d / max(math.cos(math.radians(lat)), 0.02)
    while not cache.exists():
        q = dict(query="product", target="moon", ihid="LRO", iid="LROC", pt="CDRNAC4", results="mf", output="JSON",
                 limit=500, offset=offset, minlat=lat - d, maxlat=lat + d, westlon=lon - dl, eastlon=lon + dl)
        res = json.loads(http(ODE + "?" + urllib.parse.urlencode(q)))["ODEResults"]
        page = (res.get("Products") or {}).get("Product") or []
        page = page if isinstance(page, list) else [page]
        prods += page
        if len(page) < 500:
            break
        offset += 500
    if cache.exists():
        prods = json.loads(cache.read_text())
    else:
        ODE_CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(prods))
    rows = []
    for p in prods:
        try:
            poly = [tuple(map(float, xy.split()))[::-1] for xy in
                    re.search(r"\(\(([^()]+)\)\)", p["Footprint_geometry"]).group(1).split(",")]
            files = p["Product_files"]["Product_file"]
            files = files if isinstance(files, list) else [files]
            img = next(f["URL"] for f in files if f["FileName"].upper().endswith(".IMG"))
            inc = float(p["Incidence_angle"])
        except (KeyError, ValueError, AttributeError, StopIteration):
            continue
        uv = locate(poly, lat, lon) if len(poly) == 5 else None
        if not uv or not (0 <= uv[0] <= 1 and 0 <= uv[1] <= 1):
            continue
        utc = p.get("UTC_start_time", "")
        rows.append(dict(site_id=sid, object=obj, mission=mission, body=body, frame_id=p["Product_name"][:-4],
                         instrument="LROC NAC", utc_start=utc[:19], after_arrival=after(utc, arrival),
                         days_since_arrival=days_since(utc, arrival), incidence_deg=round(inc, 2),
                         sun_elevation_deg=round(90 - inc, 2), sun_azimuth_image_deg="",
                         emission_deg=p.get("Emission_angle", ""), phase_deg=p.get("Phase_angle", ""),
                         resolution_m_px=p.get("Map_resolution", ""), local_solar_time_h="", solar_longitude_ls="",
                         obj_sample_frac=round(uv[0], 4), obj_line_frac=round(uv[1], 4),
                         frame_center_lat=p.get("Center_latitude", ""), frame_center_lon=p.get("Center_longitude", ""),
                         team_rationale=p.get("Comment", ""), image_url=img, label_url=p.get("LabelURL", ""),
                         browse_url=p.get("External_url", "")))
    return rows


# ── Mars: HiRISE observations from the RDR cumulative index ────────────────
def hirise_index():
    lbl = (HIRISE_INDEX / "RDRCUMINDEX.LBL").read_text()
    cols = {m[0]: (int(m[1]) - 1, int(m[1]) - 1 + int(m[2])) for m in
            re.findall(r"NAME\s*=\s*(\w+).*?START_BYTE\s*=\s*(\d+).*?BYTES\s*=\s*(\d+)", lbl, re.S)}
    rows = []
    for line in (HIRISE_INDEX / "RDRCUMINDEX.TAB").open("rb"):
        s = line.decode("latin1")
        g = lambda k: s[cols[k][0]:cols[k][1]].strip().strip('"').strip()
        if not g("PRODUCT_ID").endswith("_RED"):
            continue
        try:
            f = lambda k: float(g(k))
            rows.append(dict(obs=g("OBSERVATION_ID"), path=g("FILE_NAME_SPECIFICATION"), why=g("RATIONALE_DESC"),
                             time=g("START_TIME"), inc=f("INCIDENCE_ANGLE"), emi=f("EMISSION_ANGLE"),
                             phase=f("PHASE_ANGLE"), sun_az=f("SUB_SOLAR_AZIMUTH"), ls=f("SOLAR_LONGITUDE"),
                             lst=f("LOCAL_TIME"), scale=f("MAP_SCALE"),
                             lat=(f("MINIMUM_LATITUDE") + f("MAXIMUM_LATITUDE")) / 2,
                             lon=(f("MINIMUM_LONGITUDE") + f("MAXIMUM_LONGITUDE")) / 2,
                             corners=[(f(f"CORNER{k}_LATITUDE"), f(f"CORNER{k}_LONGITUDE")) for k in range(1, 5)]))
        except ValueError:
            continue
    return rows


def hirise_row(site, o):
    sid, body, mission, obj, _, lat, lon, _, _, arrival, _ = site
    url = f"{HIRISE_PDS}/{o['path']}"
    return dict(site_id=sid, object=obj, mission=mission, body=body, frame_id=o["obs"], instrument="HiRISE RED",
                utc_start=o["time"][:19], after_arrival=after(o["time"], arrival),
                days_since_arrival=days_since(o["time"], arrival), incidence_deg=round(o["inc"], 2),
                sun_elevation_deg=round(90 - o["inc"], 2), sun_azimuth_image_deg=round(o["sun_az"], 2),
                emission_deg=round(o["emi"], 2), phase_deg=round(o["phase"], 2), resolution_m_px=o["scale"],
                local_solar_time_h=round(o["lst"], 2), solar_longitude_ls=round(o["ls"], 2), obj_sample_frac="",
                obj_line_frac="", frame_center_lat=round(o["lat"], 4), frame_center_lon=round(o["lon"], 4),
                team_rationale=o["why"], image_url=url, label_url=url.rsplit(".", 1)[0] + ".LBL",
                browse_url=f"https://www.uahirise.org/{o['obs']}")


# ── Mars rovers: where the rover was when each HiRISE frame was taken ─────
def waypoints(mission):
    """[(sol, lat, east lon)] in drive order, from the cached MMGIS waypoint file."""
    path = ROVER_CACHE / f"{mission}_waypoints.json"
    if not path.exists():
        ROVER_CACHE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(http(ROVERS[mission]["url"]))
    pts = [(int(f["properties"]["sol"]), float(f["properties"]["lat"]), float(f["properties"]["lon"]) % 360)
           for f in json.loads(path.read_text())["features"]]
    return sorted(pts, key=lambda p: p[0])      # stable: drives within a sol keep their order


def mission_sol(utc, mission):
    r = ROVERS[mission]
    epoch = datetime.fromisoformat(r["landing"]).timestamp() - r["landing_lmst_h"] / 24 * SOL_S
    return math.floor((datetime.fromisoformat(utc[:19]).timestamp() - epoch) / SOL_S)


def km_mars(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    c = math.sin(la1) * math.sin(la2) + math.cos(la1) * math.cos(la2) * math.cos(lo1 - lo2)
    return 3389.5 * math.acos(max(-1.0, min(1.0, c)))


def rover_frames(mission, obs, site):
    """HiRISE frames that contain the rover. Its position on the frame's sol is the last waypoint up to that sol;
    on a sol with a drive the rover may have been at the start or the end, so the gap is the uncertainty."""
    wp = waypoints(mission)
    last_sol = wp[-1][0]
    lat0 = sum(p[1] for p in wp) / len(wp)
    rows, late = [], 0
    for o in obs:
        if abs(o["lat"] - lat0) > 0.6 or not after(o["time"], site[9]):
            continue
        sol = mission_sol(o["time"], mission)
        if sol > last_sol:
            late += inside(*wp[-1][1:], o["corners"]); continue
        end = [p for p in wp if p[0] <= sol] or wp[:1]
        start = [p for p in wp if p[0] < sol] or wp[:1]
        a, b = start[-1][1:], end[-1][1:]
        if not (inside(*b, o["corners"]) or inside(*a, o["corners"])):
            continue
        rows.append(hirise_row(site, o) | dict(site_id=f"{mission.lower()}_rover", object=f"{mission} rover",
                                               object_lat=round(b[0], 6), object_lon=round(b[1], 6), rover_sol=sol,
                                               position_uncertainty_m=round(km_mars(a, b) * 1000, 1),
                                               found_by="rover waypoint"))
    return rows, wp, late


def write_csv(path, fields, rows):
    for r in rows if "usable" in fields else []:      # sunlit (incidence < 90°), sharp enough, taken after the object arrived
        r["usable"] = r["after_arrival"] and float(r["incidence_deg"]) < 90 and float(r["resolution_m_px"]) <= MAX_RES_M
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fields); w.writeheader(); w.writerows(rows)


def summarise(site, frames):
    sid, body, mission, obj, kind, lat, lon, elev, unc, arrival, src = site
    after = [f for f in frames if f["after_arrival"]]
    usable = [f for f in after if f["incidence_deg"] < 90 and float(f["resolution_m_px"]) <= MAX_RES_M]
    res = [float(f["resolution_m_px"]) for f in usable]
    inc = [f["incidence_deg"] for f in usable]
    best = min(usable, key=lambda f: (float(f["resolution_m_px"] or 99), float(f["emission_deg"] or 99)), default=None)
    return dict(site_id=sid, body=body, mission=mission, object=obj, object_type=kind, lat=lat, lon_east=lon,
                elevation_m="" if elev is None else elev, coord_uncertainty_m=unc, arrival_utc_date=arrival,
                frames_total=len(frames), frames_after_arrival=len(after), frames_usable=len(usable),
                best_resolution_m_px=min(res) if res else "", best_frame=best["frame_id"] if best else "",
                incidence_min_deg=min(inc) if inc else "", incidence_max_deg=max(inc) if inc else "",
                coord_source=src)


def build():
    sites_out, frames_out = [], []
    moon = [s for s in SITES if s[1] == "Moon"]
    mars = [s for s in SITES if s[1] == "Mars"]

    live.step("Moon (LROC NAC)")
    for k, site in enumerate(moon):
        live.progress(k, len(moon), "objects")
        frames = nac_frames(site)
        frames_out += frames; sites_out.append(summarise(site, frames))
        s = sites_out[-1]
        _log(f"   {site[3]:<40} {s['frames_total']:4d} frames ({s['frames_after_arrival']} after arrival), "
             f"best {s['best_resolution_m_px']} m/px")
        time.sleep(0.5)

    live.step("Mars (HiRISE)")
    obs = hirise_index()
    _log(f"   HiRISE index: {len(obs)} RED observations, latest {max(o['time'] for o in obs)[:10]}")
    extra, covered = [], set()
    for k, site in enumerate(mars):
        live.progress(k, len(mars), "sites")
        _, _, mission, obj, _, lat, lon, *_ = site
        near = [o for o in obs if abs(o["lat"] - lat) < 0.5 and inside(lat, lon, o["corners"])]
        frames = sorted((hirise_row(site, o) for o in near), key=lambda f: f["utc_start"])
        covered |= {o["obs"] for o in near}
        frames_out += frames; sites_out.append(summarise(site, frames))
        s = sites_out[-1]
        _log(f"   {obj:<40} {s['frames_total']:4d} frames ({s['frames_after_arrival']} after arrival), "
             f"best {s['best_resolution_m_px']} m/px")
        found = set()
        if mission in ROVERS:
            rows, wp, late = rover_frames(mission, obs, site)
            extra += rows; found = {r["frame_id"] for r in rows}
            rover = (f"{mission.lower()}_rover", "Mars", mission, f"{mission} rover (latest published position)", "rover",
                     wp[-1][1], wp[-1][2], None, "", site[9], f"NASA/JPL MMGIS waypoints, sol {wp[-1][0]}")
            sites_out.append(summarise(rover, rows))
            _log(f"      + {len(rows)} frames containing the {mission} rover (waypoints to sol {wp[-1][0]}; "
                 f"{late} later frames skipped, position not yet published)")
        rx = re.compile(MISSION_WORDS[mission], re.I)
        named = [hirise_row(site, o) | dict(site_id=f"{mission.lower()}_mission", found_by="team rationale")
                 for o in obs if o["obs"] not in covered | found and rx.search(o["why"])
                 and after(o["time"], site[9])
                 and HARDWARE_NAMED.search(o["why"]) and not NOT_THE_HARDWARE.search(o["why"])
                 and NOT_THE_ROVER.search(o["why"])]
        extra += named
        _log(f"      + {len(named)} more {mission} frames whose rationale names the hardware")

    write_csv(HERE / "frames.csv", FRAME_FIELDS, frames_out)
    write_csv(HERE / "mars_mission_frames.csv", MISSION_FIELDS, sorted(extra, key=lambda f: (f["mission"], f["utc_start"])))
    write_csv(HERE / "sites.csv", list(sites_out[0]), sites_out)
    live.metric(objects=len(sites_out), frames=len(frames_out), mars_mission_frames=len(extra))
    _log(f"Done: {len(sites_out)} objects, {len(frames_out)} object-frame rows, {len(extra)} Mars mission frames")


if __name__ == "__main__":
    live = LiveRun(__file__, str(HERE), ["Moon (LROC NAC)", "Mars (HiRISE)"])
    try:
        build()
        live.done()
    except BaseException as e:
        live.fail(e); raise

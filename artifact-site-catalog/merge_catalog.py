"""catalog.csv: ONE table with everything. One row per known object x orbital frame that contains it.

Merges sites.csv (object, coordinates), frames.csv (every frame per object), mars_mission_frames.csv (rover
positions per sol), the crops' metadata.csv (which frames were cut) and annotations/boxes_coco.json (which crops are
boxed). Run after build_catalog.py; build_catalog.py calls it at the end.
"""
import csv, json, os
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMG_META = HERE.parent / "data" / "artifact-site-images" / "metadata.csv"

FIELDS = ["site_id", "object", "object_type", "mission", "body",
          "lat", "lon_east", "coord_uncertainty_m", "arrival_utc_date",
          "frame_id", "instrument", "utc_start", "after_arrival", "days_since_arrival", "rover_sol",
          "incidence_deg", "sun_elevation_deg", "sun_azimuth_image_deg", "emission_deg", "phase_deg",
          "resolution_m_px", "local_solar_time_h", "solar_longitude_ls", "usable",
          "crop_file", "crop_x", "crop_y", "boxed", "box_confidence", "box_x_min", "box_y_min", "box_x_max", "box_y_max",
          "position_note", "team_rationale", "image_url", "label_url", "browse_url", "coord_source"]


def read(p):
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def merge():
    sites = {r["site_id"]: r for r in read(HERE / "sites.csv")}
    first = {}
    for r in sites.values():
        first.setdefault(r["mission"], r["site_id"])
    crops = {}
    for r in read(IMG_META):                       # every object listed in a crop, not only the one it is centred on
        for tok in r["objects_in_crop"].split(";"):
            oid, xy = tok.split("@")
            crops.setdefault((oid, r["frame_id"]), (r["file_name"], *xy.split(","), r["position_note"]))
    boxes = {}
    coco = HERE / "annotations" / "boxes_coco.json"
    if coco.exists():
        d = json.load(open(coco))
        ims = {i["id"]: i["file_name"] for i in d["images"]}
        cats = {c["id"]: c["name"] for c in d["categories"]}
        for a in d["annotations"]:
            x, y, w, h = a["bbox"]
            boxes.setdefault((cats[a["category_id"]], ims[a["image_id"]]),
                             dict(confidence=a["confidence"], x_min=x, y_min=y, x_max=x + w - 1, y_max=y + h - 1))
    out = []
    for src, rover in ((HERE / "frames.csv", False), (HERE / "mars_mission_frames.csv", True)):
        for f in read(src):
            if f["site_id"] not in sites:                 # ballast / balance-mass impact zones: no coordinates, listed under the mission's first site
                f = dict(f, site_id=first[f["mission"]])
            s = sites[f["site_id"]]
            row = {k: f.get(k, "") for k in FIELDS}
            row.update(site_id=f["site_id"], object=s["object"], object_type=s["object_type"], mission=s["mission"], body=s["body"],
                       coord_uncertainty_m=s["coord_uncertainty_m"], arrival_utc_date=s["arrival_utc_date"], coord_source=s["coord_source"])
            if rover:
                row.update(lat=f["object_lat"], lon_east=f["object_lon_east"], rover_sol=f["rover_sol"],
                           coord_uncertainty_m=f["position_uncertainty_m"] or s["coord_uncertainty_m"],
                           coord_source="NASA/JPL rover waypoints" if f["found_by"] == "rover waypoint" else "team rationale (impact zone, position in frame unknown)")
                if f["found_by"] != "rover waypoint":
                    row.update(object=f["object"], lat=s["lat"], lon_east=s["lon_east"], coord_uncertainty_m="")
            else:
                row.update(lat=s["lat"], lon_east=s["lon_east"])
            c = crops.get((f["site_id"], f["frame_id"]))
            if c:
                row.update(crop_file=c[0], crop_x=c[1], crop_y=c[2], position_note=c[3])
                b = boxes.get((f["site_id"], c[0]))
                row["boxed"] = "yes" if b else "no"
                if b:
                    row.update(box_confidence=b["confidence"], box_x_min=b["x_min"], box_y_min=b["y_min"], box_x_max=b["x_max"], box_y_max=b["y_max"])
            else:
                row["boxed"] = ""
            out.append(row)
    out.sort(key=lambda r: (r["body"] != "Moon", r["mission"], r["site_id"], r["utc_start"]))
    with open(HERE / "catalog.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, FIELDS); w.writeheader(); w.writerows(out)
    n_crop = sum(1 for r in out if r["crop_file"]); n_box = sum(1 for r in out if r["boxed"] == "yes")
    print(f"catalog.csv: {len(out)} rows ({len(sites)} objects), {n_crop} with a crop, {n_box} boxed")
    return out


if __name__ == "__main__":
    merge()

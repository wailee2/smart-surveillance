"""
download_weights.py
-------------------
Helper script to download / copy your trained model weights into the
expected locations before starting the app.

Usage options
─────────────
1. You have .pt files on disk:
      python download_weights.py --traffic path/to/traffic.pt \
                                  --vehicle path/to/vehicle.pt \
                                  --plate   path/to/plate.pt

2. You have a Roboflow project with trained weights:
      python download_weights.py --roboflow-api-key YOUR_KEY \
                                  --traffic-project  your/traffic-project/1 \
                                  --vehicle-project  your/vehicle-project/1 \
                                  --plate-project    your/plate-project/1

3. Manual placement (no script):
      Just copy the .pt files to:
        weights/traffic_light_best.pt
        weights/vehicle_best.pt
        weights/license_plate_best.pt
"""

import argparse
import os
import shutil

WEIGHT_TARGETS = {
    "traffic": "weights/traffic_light_best.pt",
    "vehicle": "weights/vehicle_best.pt",
    "plate":   "weights/license_plate_best.pt",
}


def copy_weights(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  ✓ Copied {src} → {dst}")


def roboflow_download(api_key, project_slug, dest_path):
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Install roboflow: pip install roboflow")
        return
    rf = Roboflow(api_key=api_key)
    parts = project_slug.split("/")
    workspace, project, version = parts[0], parts[1], int(parts[2])
    model = rf.workspace(workspace).project(project).version(version).model
    # The YOLOv11 export location varies by Roboflow version
    weights_path = model.path + "/weights/best.pt"
    if os.path.exists(weights_path):
        copy_weights(weights_path, dest_path)
    else:
        print(f"  ⚠️  Could not find weights at {weights_path}. "
              "Download manually from the Roboflow dashboard.")


def main():
    parser = argparse.ArgumentParser(description="Download / copy model weights")
    parser.add_argument("--traffic",  help="Path to traffic light weights .pt")
    parser.add_argument("--vehicle",  help="Path to vehicle weights .pt")
    parser.add_argument("--plate",    help="Path to license plate weights .pt")
    parser.add_argument("--roboflow-api-key",    dest="rf_key")
    parser.add_argument("--traffic-project",     dest="rf_traffic")
    parser.add_argument("--vehicle-project",     dest="rf_vehicle")
    parser.add_argument("--plate-project",       dest="rf_plate")
    args = parser.parse_args()

    os.makedirs("weights", exist_ok=True)

    if args.traffic:
        copy_weights(args.traffic, WEIGHT_TARGETS["traffic"])
    if args.vehicle:
        copy_weights(args.vehicle, WEIGHT_TARGETS["vehicle"])
    if args.plate:
        copy_weights(args.plate, WEIGHT_TARGETS["plate"])

    if args.rf_key:
        if args.rf_traffic:
            roboflow_download(args.rf_key, args.rf_traffic, WEIGHT_TARGETS["traffic"])
        if args.rf_vehicle:
            roboflow_download(args.rf_key, args.rf_vehicle, WEIGHT_TARGETS["vehicle"])
        if args.rf_plate:
            roboflow_download(args.rf_key, args.rf_plate, WEIGHT_TARGETS["plate"])

    # Summary
    print("\nWeight file status:")
    for label, path in WEIGHT_TARGETS.items():
        exists = "✓" if os.path.exists(path) else "✗  (MISSING)"
        print(f"  {label:8s}: {path}  {exists}")


if __name__ == "__main__":
    main()

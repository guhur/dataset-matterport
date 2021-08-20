"""
This compiles a JSON file with:
    scan, viewpoint_id, region_label level, list obj ids
"""
from pathlib import Path
from typing import Union, DefaultDict, List, Dict
from collections import defaultdict
import csv
import json
from tqdm import tqdm
import argtyped


PathOrStr = Union[Path, str]
REGION_LABELS = {
    "a": "bathroom",
    "b": "bedroom",
    "c": "closet",
    "d": "dining room",
    "e": "entryway/foyer/lobby",
    "f": "familyroom",
    "g": "garage",
    "h": "hallway",
    "i": "library",
    "j": "laundryroom/mudroom",
    "k": "kitchen",
    "l": "living room",
    "m": "meetingroom/conferenceroom",
    "n": "lounge ",
    "o": "office",
    "p": "porch/terrace/deck/driveway",
    "r": "rec/game",
    "s": "stairs",
    "t": "toilet",
    "u": "utilityroom/toolroom",
    "v": "tv",
    "w": "workout/gym/exercise",
    "x": "outdoor",
    "y": "balcony",
    "z": "other room",
    "B": "bar",
    "C": "classroom",
    "D": "dining booth",
    "S": "spa/sauna",
    "Z": "junk",
    "-": "no label",
}


def save_json(data, filename: PathOrStr):
    with open(filename, "w") as fid:
        json.dump(data, fid, indent=2)


def load_json(filename: PathOrStr):
    with open(filename) as fid:
        return json.load(fid)


def load_house_segmentation(filename: PathOrStr):
    region_labels = {}
    viewpoints = {}
    scan = Path(filename).stem

    with open(filename, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=" ", skipinitialspace=True)
        for row in reader:
            if row[0] == "R":
                region_labels[int(row[1])] = {
                    "level": int(row[2]),
                    "label": REGION_LABELS[row[5]],
                    "scan": scan,
                }

            elif row[0] == "P":
                region_id = int(row[3])
                viewpoints[row[1]] = region_id

    return [
        {"viewpoint": vp, **region_labels[region_id]}
        if region_id != -1
        else {"viewpoint": vp, "error": "This viewpoint has no annotated region"}
        for vp, region_id in viewpoints.items()
    ]


class Arguments(argtyped.Arguments):
    matterport: Path = Path("data") / "v1"
    bbox: Path = Path("data") / "bbox"
    output: Path = Path("viewpoints_details.json")


def load_bbox(bbox_dir: Path) -> DefaultDict[str, Dict]:
    boxes = defaultdict(dict)
    for filename in bbox_dir.glob("*.json"):
        data = load_json(filename)
        boxes.update(data)
    return boxes


if __name__ == "__main__":

    args = Arguments()
    print(args)

    bbox = load_bbox(args.bbox)
    results: List[Dict] = []
    houses = list(args.matterport.rglob("*.house"))

    for house_segmentation in tqdm(houses):
        viewpoints = load_house_segmentation(house_segmentation)
        for vp in viewpoints:
            vp["object_ids"] = list(bbox[vp["viewpoint"]].keys())
            results.append(vp)

    save_json(results, args.output)

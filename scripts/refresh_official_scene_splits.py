from __future__ import annotations

import csv
import io
import re
import urllib.request
from pathlib import Path

SPLITS_URL = (
    "https://api.pcloud.com/getpubtextfile"
    "?code=kZ46bk0Z5JKM8r2bzfyjYl3dW85U60XaBmPV&fileid=57823192235"
)
OUTPUT_PATH = Path("src/cr_train/resources/official_scene_splits.csv")
SPLIT_LABELS = {"1": "train", "2": "val", "3": "test"}
PATTERN = re.compile(r"ROIs\d+_(spring|summer|fall|winter)_s1_(\d+)_p\d+\.tif")
SEASON_ORDER = {"spring": 0, "summer": 1, "fall": 2, "winter": 3}


def main() -> None:
    text = urllib.request.urlopen(SPLITS_URL).read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    scene_to_stage: dict[tuple[str, str], str] = {}
    for row in reader:
        match = PATTERN.fullmatch(row["sample"])
        if match is None:
            continue
        season, scene = match.groups()
        stage = SPLIT_LABELS[row["split"]]
        key = (season, scene)
        previous = scene_to_stage.get(key)
        if previous is not None and previous != stage:
            raise RuntimeError(f"conflicting stage for {key}: {previous} vs {stage}")
        scene_to_stage[key] = stage

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stage", "season", "scene"))
        for (season, scene), stage in sorted(
            scene_to_stage.items(),
            key=lambda item: (SEASON_ORDER[item[0][0]], int(item[0][1])),
        ):
            writer.writerow((stage, season, scene))


if __name__ == "__main__":
    main()

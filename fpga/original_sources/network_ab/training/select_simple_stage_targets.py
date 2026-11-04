"""Record manually reviewed, vehicle-light target scenes for the starter trial."""
from __future__ import annotations

import json
from pathlib import Path

from .runtime import configure_workspace, file_hash, write_json


# Reviewed against daylight_candidates_00/01.jpg. Favor foliage, paths,
# stairs and buildings. Avoid the vehicle-heavy road and emergency scenes.
FILES = (
    "C_3_01095.png", "C_3_01895.png", "G_3_02260.png", "G_3_02275.png",
    "N_18-54-23_00160.png", "N_18-59-02_00035.png", "N_18-59-02_00055.png",
    "N_18-59-02_00130.png", "N_18-59-02_00175.png", "N_18-59-02_00205.png",
    "N_18-59-02_00605.png", "N_18-59-02_00625.png", "N_18-59-02_00660.png",
    "N_18-59-02_00690.png", "N_18-59-02_00840.png", "N_18-59-02_00845.png",
    "N_18-59-02_00855.png", "N_18-59-02_01045.png", "N_18-59-02_02080.png",
    "N_latest_035.png",
)


def main() -> None:
    root = configure_workspace()
    manifest = root / "data/processed/style-target-20261001-v14-expanded-reviewed/dataset-manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("status") != "approved" or data.get("image_count") != len(data["records"]):
        raise ValueError("Approved target manifest changed")
    records = {row["file"]: row for row in data["records"]}
    if len(FILES) != len(set(FILES)) or len(FILES) != 20:
        raise ValueError("Vehicle-light manual selection must contain 20 unique frames")
    selected = []
    for name in FILES:
        row = records[name]
        if not (row.get("lighting") == "daylight" or
                row.get("scene", "").startswith("daylight")):
            raise ValueError(f"Non-daylight selection: {name}")
        path = manifest.parent / name
        if not path.is_file() or file_hash(path) != row["sha256"]:
            raise ValueError(f"Modified target image: {name}")
        selected.append({"file": name, "sha256": row["sha256"],
                         "source_video": row["source_video"], "second": row["second"]})
    index_path = root / "runs/content-quality-eval-20261001/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    holdout = Path(index["selections"]["validation_scene_coverage"]["paths"][0])
    out = root / "runs/simple-scene-stage-20261002/target-selection.json"
    write_json(out, {
        "schema": "simple-stage-target-selection-v1",
        "status": "reviewed_for_simple_stage",
        "parent_manifest": str(manifest), "parent_manifest_sha256": file_hash(manifest),
        "selection_policy": "Manually inspected both daylight contact sheets; choose 20 clear daylight game views dominated by paths, stairs, foliage and buildings, with few or no visible vehicles. This starter subset does not claim complete UI/weapon elimination.",
        "candidate_contact_sheets": [
            str(out.parent / "daylight_candidates_00.jpg"),
            str(out.parent / "daylight_candidates_01.jpg")],
        "records": selected, "image_count": len(selected),
        "fixed_unseen_content": str(holdout),
        "fixed_unseen_content_sha256": file_hash(holdout),
        "content_index": str(index_path),
        "content_index_sha256": file_hash(index_path),
    })
    print(out, flush=True)


if __name__ == "__main__":
    main()

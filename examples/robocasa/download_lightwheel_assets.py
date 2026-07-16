"""Download RoboCasa's lightwheel kitchen assets (objects + fixtures).

Workaround for a broken upstream step: RoboCasa 365's `download_kitchen_assets.py` fetches a single
`objects_lightwheel.zip` / `fixtures_lightwheel.zip` from `nvidia/PhysicalAI-Kitchen-Assets`, but
NVIDIA renamed that repo to `nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF` and
restructured it into per-object zips under `objects_lightwheel/` and `fixtures_lightwheel/`, so the
official script 404s. This fetches those per-object zips and extracts them to the same folders the
official script targets, producing the identical on-disk layout.

    uv run examples/robocasa/download_lightwheel_assets.py

(The other asset packs — textures, objaverse, aigen — download fine via the official
`download_kitchen_assets.py`.)
"""

import io
import zipfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF"
_ROBOCASA_ASSETS = Path(__file__).resolve().parents[2] / "third_party/robocasa/robocasa/models/assets"
# HF subdir -> local extraction target (matches download_kitchen_assets.py's objs_lw / fixtures_lw).
TARGETS = {
    "objects_lightwheel": _ROBOCASA_ASSETS / "objects/lightwheel",
    "fixtures_lightwheel": _ROBOCASA_ASSETS / "fixtures",
}


def main() -> None:
    api = HfApi()
    files = api.list_repo_files(repo_id=REPO, repo_type="dataset")
    for subdir, target in TARGETS.items():
        zips = sorted(f for f in files if f.startswith(f"{subdir}/") and f.endswith(".zip"))
        target.mkdir(parents=True, exist_ok=True)
        print(f"{subdir}: {len(zips)} object zips -> {target}")
        for i, f in enumerate(zips, 1):
            local = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=f)
            with zipfile.ZipFile(local) as z:
                z.extractall(target)
            print(f"  [{i}/{len(zips)}] {Path(f).name}")
    print("Done.")


if __name__ == "__main__":
    main()

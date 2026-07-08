"""Generate the project dashboard webpage (self-contained HTML).

A living overview of this repo's openpi × RoboCasa 365 work: implementation status, roadmap, and
a browsable gallery of the 50 RoboCasa 365 target tasks (thumbnail + category + horizon +
description + on-disk status) to help decide what to work on.

Re-run any time to refresh (it re-scans the dataset dir and embeds any new thumbnails):

    uv run examples/robocasa/gen_dashboard.py \
        --output-dir /data5/jellyho/robocasa365 \
        --out examples/robocasa/dashboard.html

Thumbnails are read from examples/robocasa/assets/<Task>.jpg (extract them from the converted
datasets with, per task, e.g.:
    ffmpeg -ss 1.2 -i <Task>/videos/observation.images.robot0_agentview_left/chunk-000/file-000.mp4 \
        -frames:v 1 -vf scale=400:-1 examples/robocasa/assets/<Task>.jpg
)
"""

import argparse
import base64
import html
import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ASSETS = _HERE / "assets"

# --- Project status: edit these as the work evolves; the page regenerates from them. ---------

PROJECT = {
    "title": "openpi × RoboCasa 365",
    "tagline": "Fine-tuning π-models on RoboCasa 365 kitchen demos, on the LeRobot v3.0 data stack.",
    "stats": [
        {"label": "LeRobot", "value": "0.4.4", "note": "dataset format v3.0"},
        {"label": "Target tasks", "value": "50", "note": "human demos, 500 ea."},
        {"label": "Base model", "value": "π0 / π0.5", "note": "transformers 4.53.2"},
        {"label": "Sim source", "value": "RoboCasa", "note": "robocasa365_release"},
    ],
    "implemented": [
        ("RoboCasa vendored", "Added `robocasa` (branch `robocasa365_release`) as a git submodule under `third_party/`; removed the unused aloha / libero submodules."),
        ("LeRobot v3.0 migration", "Moved openpi from a pinned lerobot v2.1 rev to `lerobot==0.4.4` (CODEBASE_VERSION v3.0). Updated the dataset module path and normalized the v3.0 `meta.tasks` DataFrame back into the `{task_index: task}` prompt map."),
        ("Dependency stack", "Bumped numpy to 2.x (required by lerobot→rerun-sdk) across openpi + openpi-client; kept transformers 4.53.2 (pi0/pi0.5 vendored patches). Regenerated uv.lock. Retired the DROID/RLDS group (TF 2.15 pins numpy<2)."),
        ("Dataset pipeline", "`prepare_robocasa365.py`: download the 50 target/human LeRobot tars from the public NVIDIA mirror, extract, and upgrade v2.1 → v3.0. Resumable — finished tasks are skipped."),
        ("Hub upload", "`--push-to-hub`: publish each converted dataset as `<user>/robocasa365-<Task>` (tagged v3.0) and gather them into a HF collection."),
        ("Training config", "Added `robocasa_policy.py` (3 cameras → base/left-wrist/right-wrist, 16-d state, 12-d action) plus `LeRobotRoboCasaDataConfig` and the `pi05_robocasa` / `pi05_robocasa_low_mem_finetune` train configs."),
        ("Sim eval client", "`main.py`: server/client rollout in the RoboCasa sim — builds the model input from live obs, reorders the 12-d action into the env's order, steps to success, reports success rate + optional videos."),
    ],
    "roadmap": [
        ("Norm stats + training", "Compute norm stats over a chosen task and run a first π0.5 fine-tune from the base checkpoint."),
        ("Task selection", "Pick the target task(s) to train/evaluate on from the gallery below."),
        ("Run evaluation", "Serve the trained checkpoint and roll out with `main.py`; confirm camera orientation against a dataset frame."),
        ("Multi-task training", "Merge tasks into one LeRobot dataset (or add multi-repo loading) to train across the target set at once."),
    ],
}

# --- Dataset schema (metadata), from meta/info.json + the PandaOmron modality spec. ----------

DATASET = {
    "facts": [
        ("Format", "LeRobot v3.0"),
        ("Robot", "PandaOmron"),
        ("Frame rate", "20 fps"),
        ("Demos / task", "~500 (target, human)"),
        ("Cameras", "3 × 256²  RGB"),
    ],
    "cameras": [
        ("observation.images.robot0_agentview_left", "exterior · third-person"),
        ("observation.images.robot0_agentview_right", "exterior · second view"),
        ("observation.images.robot0_eye_in_hand", "wrist"),
    ],
    "state": [
        ("0:3", "base position"),
        ("3:7", "base rotation (quat)"),
        ("7:10", "EE position (relative)"),
        ("10:14", "EE rotation (relative)"),
        ("14:16", "gripper qpos"),
    ],
    "action": [
        ("0:4", "base motion"),
        ("4:5", "control mode (discrete)"),
        ("5:8", "EE position"),
        ("8:11", "EE rotation"),
        ("11:12", "gripper (close)"),
    ],
}

# --- Task metadata: (name, category, horizon, description). ---------------------------------
# Descriptions marked from RoboCasa's canonical task language where available, else concise.

TASKS = [
    # atomic
    ("CloseBlenderLid", "atomic", 600, "Close the lid of the blender."),
    ("CloseFridge", "atomic", 600, "Close the refrigerator door."),
    ("CloseToasterOvenDoor", "atomic", 300, "Close the toaster oven door."),
    ("CoffeeSetupMug", "atomic", 400, "Place a mug on the coffee machine's tray, ready to brew."),
    ("NavigateKitchen", "atomic", 300, "Drive the mobile base to a target location in the kitchen."),
    ("OpenCabinet", "atomic", 700, "Open the cabinet door."),
    ("OpenDrawer", "atomic", 500, "Open the drawer."),
    ("OpenStandMixerHead", "atomic", 300, "Lift the head of the stand mixer."),
    ("PickPlaceCounterToCabinet", "atomic", 500, "Pick an object from the counter and place it in the cabinet."),
    ("PickPlaceCounterToStove", "atomic", 400, "Pick an object from the counter and place it on the stove."),
    ("PickPlaceDrawerToCounter", "atomic", 500, "Pick an object from the drawer and place it on the counter."),
    ("PickPlaceSinkToCounter", "atomic", 600, "Pick an object from the sink and place it on the counter."),
    ("PickPlaceToasterToCounter", "atomic", 400, "Pick an item from the toaster and place it on the counter."),
    ("SlideDishwasherRack", "atomic", 300, "Slide the dishwasher rack back in."),
    ("TurnOffStove", "atomic", 500, "Turn off the stove burner."),
    ("TurnOnElectricKettle", "atomic", 300, "Switch on the electric kettle."),
    ("TurnOnMicrowave", "atomic", 300, "Start the microwave."),
    ("TurnOnSinkFaucet", "atomic", 400, "Turn on the sink faucet."),
    # composite
    ("ArrangeBreadBasket", "composite", 2900, "Arrange assorted bread into the serving basket."),
    ("ArrangeTea", "composite", 1500, "Move the kettle to the tray, add the mug from the cabinet, then close the cabinet doors."),
    ("BreadSelection", "composite", 1300, "Select a croissant onto the cutting board, then add a jar of jam from the cabinet."),
    ("CategorizeCondiments", "composite", 1100, "Sort the condiment bottles into groups by type."),
    ("CuttingToolSelection", "composite", 800, "Choose the right cutting tool and set it on the cutting board."),
    ("DeliverStraw", "composite", 1700, "Retrieve a straw and deliver it with the drink."),
    ("GarnishPancake", "composite", 1800, "Garnish the pancake with its toppings."),
    ("GatherTableware", "composite", 1500, "Gather plates and utensils and set them together."),
    ("GetToastedBread", "composite", 2000, "Toast bread and serve the toasted slice."),
    ("HeatKebabSandwich", "composite", 1800, "Heat a kebab sandwich."),
    ("KettleBoiling", "composite", 1000, "Fill the kettle and set it to boil."),
    ("LoadDishwasher", "composite", 1200, "Load the dishes into the dishwasher."),
    ("MakeIceLemonade", "composite", 2000, "Prepare iced lemonade with ice and lemon."),
    ("PackIdenticalLunches", "composite", 2600, "Pack two identical lunches into containers."),
    ("PanTransfer", "composite", 1200, "Transfer the food from the pan onto a plate."),
    ("PortionHotDogs", "composite", 1500, "Portion the hot dogs onto plates."),
    ("PreSoakPan", "composite", 1600, "Fill a dirty pan with water at the sink to pre-soak it."),
    ("PrepareCoffee", "composite", 1200, "Set up the mug and brew a cup of coffee."),
    ("RecycleBottlesByType", "composite", 1900, "Sort bottles into recycling by material type."),
    ("RinseSinkBasin", "composite", 900, "Rinse down the sink basin with the faucet."),
    ("ScrubCuttingBoard", "composite", 800, "Scrub the cutting board clean."),
    ("SearingMeat", "composite", 2900, "Sear the meat in a pan on the stove."),
    ("SeparateFreezerRack", "composite", 1600, "Separate the items on the freezer rack."),
    ("SetUpCuttingStation", "composite", 1600, "Set up a cutting station with board and tools."),
    ("StackBowlsCabinet", "composite", 1400, "Stack the bowls and store them in the cabinet."),
    ("SteamInMicrowave", "composite", 1400, "Steam the food in the microwave."),
    ("StirVegetables", "composite", 1600, "Stir the vegetables in the pan."),
    ("StoreLeftoversInBowl", "composite", 1700, "Transfer the leftovers into a bowl for storage."),
    ("WaffleReheat", "composite", 2700, "Reheat a waffle."),
    ("WashFruitColander", "composite", 2100, "Wash fruit in a colander at the sink."),
    ("WashLettuce", "composite", 1100, "Rinse the lettuce at the sink."),
    ("WeighIngredients", "composite", 2000, "Weigh the ingredients on the kitchen scale."),
]

_MAX_HORIZON = max(h for _, _, h, _ in TASKS)

# Number of distinct natural-language instructions per task (from
# annotation.human.task_description across all episodes). 1 = a single fixed instruction; >1 =
# episode-specific instructions that describe the randomized objects/config. Recompute with:
#   uv run examples/robocasa/make_previews.py --instruction-counts (writes assets/instruction_counts.json)
INSTRUCTION_COUNTS = {
    "SeparateFreezerRack": 409, "StirVegetables": 238, "WashFruitColander": 157,
    "PickPlaceCounterToCabinet": 105, "PackIdenticalLunches": 51, "PickPlaceSinkToCounter": 34,
    "SearingMeat": 32, "PickPlaceCounterToStove": 28, "StoreLeftoversInBowl": 17,
    "SteamInMicrowave": 17, "NavigateKitchen": 13, "PickPlaceDrawerToCounter": 9,
    "TurnOffStove": 8, "WeighIngredients": 7, "CuttingToolSelection": 7,
    "SlideDishwasherRack": 2, "OpenDrawer": 2, "OpenCabinet": 2, "CloseFridge": 2,
}


def _instruction_count(task: str) -> int:
    """Distinct instructions for a task. Prefers a cached scan, falls back to the baked-in map."""
    cache = _ASSETS / "instruction_counts.json"
    if cache.exists():
        try:
            return int(json.loads(cache.read_text()).get(task, INSTRUCTION_COUNTS.get(task, 1)))
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return INSTRUCTION_COUNTS.get(task, 1)


def _thumb_data_uri(task: str) -> str | None:
    p = _ASSETS / f"{task}.jpg"
    if not p.exists():
        return None
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _status(task: str, output_dir: Path) -> str:
    info = output_dir / task / "meta" / "info.json"
    if not info.exists():
        return "pending"
    try:
        return "ready" if json.loads(info.read_text()).get("codebase_version") == "v3.0" else "downloaded"
    except (json.JSONDecodeError, OSError):
        return "pending"


_STATUS_LABEL = {"ready": "v3.0 ready", "downloaded": "downloaded", "pending": "not yet"}


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def render(output_dir: Path, *, mode: str = "artifact") -> str:
    ready = sum(1 for t, *_ in TASKS if _status(t, output_dir) == "ready")
    site_videos = _HERE / "site" / "videos"

    stat_tiles = "\n".join(
        f'<div class="tile"><span class="tile-v">{_esc(s["value"])}</span>'
        f'<span class="tile-l">{_esc(s["label"])}</span>'
        f'<span class="tile-n">{_esc(s["note"])}</span></div>'
        for s in PROJECT["stats"]
    )

    implemented = "\n".join(
        f'<li><span class="done-mark" aria-hidden="true">✓</span>'
        f'<div><h3>{_esc(t)}</h3><p>{_esc(d)}</p></div></li>'
        for t, d in PROJECT["implemented"]
    )
    roadmap = "\n".join(
        f'<li><span class="next-mark" aria-hidden="true"></span>'
        f'<div><h3>{_esc(t)}</h3><p>{_esc(d)}</p></div></li>'
        for t, d in PROJECT["roadmap"]
    )

    ds_facts = "\n".join(
        f'<div class="fact"><span class="fact-v mono">{_esc(v)}</span><span class="fact-l">{_esc(k)}</span></div>'
        for k, v in DATASET["facts"]
    )
    ds_cams = "\n".join(
        f'<li><code>{_esc(k)}</code><span>{_esc(v)}</span></li>' for k, v in DATASET["cameras"]
    )
    ds_state = "\n".join(
        f'<tr><td class="mono">{_esc(i)}</td><td>{_esc(v)}</td></tr>' for i, v in DATASET["state"]
    )
    ds_action = "\n".join(
        f'<tr><td class="mono">{_esc(i)}</td><td>{_esc(v)}</td></tr>' for i, v in DATASET["action"]
    )

    cards = []
    for name, cat, horizon, desc in TASKS:
        st = _status(name, output_dir)
        uri = _thumb_data_uri(name)
        bar = round(100 * horizon / _MAX_HORIZON, 1)
        # In "site" mode, reference the full-trajectory mp4 (played on click) with the poster
        # embedded inline; in "artifact" mode, embed only the static poster (self-contained).
        has_video = mode == "site" and (site_videos / f"{name}.mp4").exists()
        if has_video:
            poster = f' poster="{uri}"' if uri else ""
            media = (
                f'<video class="thumb" preload="none" controls playsinline{poster}>'
                f'<source src="videos/{_esc(name)}.mp4" type="video/mp4"></video>'
            )
        elif uri:
            media = f'<img class="thumb" loading="lazy" src="{uri}" alt="{_esc(name)} scene">'
        else:
            media = '<div class="thumb thumb-empty"><span>no preview yet</span></div>'
        n_instr = _instruction_count(name)
        instr_kind = "single" if n_instr <= 1 else "multi"
        instr_label = "1 instruction" if n_instr <= 1 else f"{n_instr} instructions"
        instr_title = (
            "Single fixed language instruction for every episode"
            if n_instr <= 1
            else f"{n_instr} distinct, episode-specific language instructions"
        )
        cards.append(
            f'<article class="card" data-cat="{cat}" data-status="{st}" data-instr="{instr_kind}" '
            f'data-name="{_esc(name.lower())}">'
            f"{media}"
            f'<div class="card-body">'
            f'<div class="card-top"><span class="chip chip-{cat}">{cat}</span>'
            f'<span class="chip chip-instr chip-{instr_kind}" title="{instr_title}">{instr_label}</span>'
            f'<span class="status status-{st}"><i></i>{_STATUS_LABEL[st]}</span></div>'
            f'<h3 class="task-name">{_esc(name)}</h3>'
            f'<p class="task-desc">{_esc(desc)}</p>'
            f'<div class="horizon"><span class="horizon-l">horizon</span>'
            f'<span class="horizon-bar"><i style="width:{bar}%"></i></span>'
            f'<span class="horizon-v">{horizon}</span></div>'
            f"</div></article>"
        )
    cards_html = "\n".join(cards)

    n_atomic = sum(1 for _, c, *_ in TASKS if c == "atomic")
    n_comp = sum(1 for _, c, *_ in TASKS if c == "composite")
    n_single = sum(1 for name, *_ in TASKS if _instruction_count(name) <= 1)
    n_multi = len(TASKS) - n_single

    return _TEMPLATE.format(
        title=_esc(PROJECT["title"]),
        tagline=_esc(PROJECT["tagline"]),
        stat_tiles=stat_tiles,
        implemented=implemented,
        roadmap=roadmap,
        ds_facts=ds_facts,
        ds_cams=ds_cams,
        ds_state=ds_state,
        ds_action=ds_action,
        cards=cards_html,
        n_total=len(TASKS),
        n_atomic=n_atomic,
        n_comp=n_comp,
        n_single=n_single,
        n_multi=n_multi,
        n_ready=ready,
        css=_CSS,
        js=_JS,
    )


_CSS = """
:root{
  --bg:#f4f6f8; --surface:#ffffff; --surface2:#eef1f5; --ink:#141a22; --muted:#5c6672;
  --hairline:#dde3ea; --amber:#c06914; --amber-bg:#f6ead9; --teal:#0f7d86; --teal-bg:#dff0f1;
  --green:#1f8f4d; --shadow:0 1px 2px rgba(20,26,34,.06),0 8px 24px rgba(20,26,34,.05);
  --radius:14px;
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0d1014; --surface:#161b22; --surface2:#1d232c; --ink:#e7ebf1; --muted:#8b95a3;
  --hairline:#262d38; --amber:#e79a4d; --amber-bg:#2c2114; --teal:#4bc2cf; --teal-bg:#10262a;
  --green:#57c878; --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 32px rgba(0,0,0,.35);
}}
:root[data-theme="light"]{
  --bg:#f4f6f8; --surface:#ffffff; --surface2:#eef1f5; --ink:#141a22; --muted:#5c6672;
  --hairline:#dde3ea; --amber:#c06914; --amber-bg:#f6ead9; --teal:#0f7d86; --teal-bg:#dff0f1;
  --green:#1f8f4d; --shadow:0 1px 2px rgba(20,26,34,.06),0 8px 24px rgba(20,26,34,.05);
}
:root[data-theme="dark"]{
  --bg:#0d1014; --surface:#161b22; --surface2:#1d232c; --ink:#e7ebf1; --muted:#8b95a3;
  --hairline:#262d38; --amber:#e79a4d; --amber-bg:#2c2114; --teal:#4bc2cf; --teal-bg:#10262a;
  --green:#57c878; --shadow:0 1px 2px rgba(0,0,0,.4),0 12px 32px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  line-height:1.55;-webkit-font-smoothing:antialiased;}
.mono{font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace;}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px;}
a{color:var(--teal);text-decoration:none;}
a:hover{text-decoration:underline;}

/* header */
header{border-bottom:1px solid var(--hairline);background:
  linear-gradient(180deg,color-mix(in srgb,var(--amber) 6%,var(--bg)),var(--bg));}
.head{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;
  padding:40px 0 34px;}
.eyebrow{font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--amber);
  font-weight:600;margin:0 0 10px;}
h1{font-size:clamp(28px,4.4vw,44px);line-height:1.05;margin:0;letter-spacing:-.02em;
  text-wrap:balance;font-weight:700;}
.tagline{color:var(--muted);font-size:17px;margin:12px 0 0;max-width:56ch;}
.theme-btn{flex:none;border:1px solid var(--hairline);background:var(--surface);color:var(--muted);
  border-radius:9px;padding:8px 12px;cursor:pointer;font-size:13px;display:inline-flex;gap:7px;
  align-items:center;}
.theme-btn:hover{color:var(--ink);border-color:var(--muted);}

/* stat tiles */
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0 44px;}
.tile{background:var(--surface);border:1px solid var(--hairline);border-radius:var(--radius);
  padding:18px 18px 16px;box-shadow:var(--shadow);display:flex;flex-direction:column;gap:2px;}
.tile-v{font-size:26px;font-weight:700;letter-spacing:-.01em;}
.tile-l{font-size:12px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);
  font-weight:600;margin-top:4px;}
.tile-n{font-size:12.5px;color:var(--muted);}

section{padding:8px 0 12px;}
.section-head{display:flex;align-items:baseline;gap:14px;margin:34px 0 18px;}
.section-head h2{font-size:20px;margin:0;letter-spacing:-.01em;}
.section-head .count{font-size:13px;color:var(--muted);}
.rule{height:1px;background:var(--hairline);margin:0 0 4px;}

/* two-column status/roadmap */
.cols{display:grid;grid-template-columns:1fr 1fr;gap:28px;margin-bottom:20px;}
.cols h2{font-size:16px;text-transform:uppercase;letter-spacing:.09em;color:var(--muted);
  margin:0 0 14px;}
.list{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:14px;}
.list li{display:flex;gap:12px;}
.list h3{font-size:15px;margin:0 0 3px;}
.list p{font-size:14px;color:var(--muted);margin:0;}
.done-mark{flex:none;width:22px;height:22px;border-radius:6px;background:color-mix(in srgb,var(--green) 16%,transparent);
  color:var(--green);display:grid;place-items:center;font-size:13px;font-weight:700;margin-top:1px;}
.next-mark{flex:none;width:22px;height:22px;border-radius:6px;border:1.5px dashed var(--amber);
  margin-top:1px;}

/* controls */
.controls{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:6px 0 22px;
  position:sticky;top:0;background:var(--bg);padding:12px 0;z-index:5;border-bottom:1px solid var(--hairline);}
.filters{display:flex;gap:6px;flex-wrap:wrap;}
.filter{border:1px solid var(--hairline);background:var(--surface);color:var(--muted);
  border-radius:999px;padding:7px 14px;font-size:13px;cursor:pointer;font-weight:500;}
.filter[aria-pressed="true"]{background:var(--ink);color:var(--bg);border-color:var(--ink);}
.filter:hover:not([aria-pressed="true"]){color:var(--ink);border-color:var(--muted);}
.search{margin-left:auto;border:1px solid var(--hairline);background:var(--surface);color:var(--ink);
  border-radius:9px;padding:8px 12px;font-size:14px;min-width:200px;}
.search::placeholder{color:var(--muted);}
.search:focus{outline:2px solid var(--teal);outline-offset:1px;}

/* card grid */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(258px,1fr));gap:18px;
  padding-bottom:60px;}
.card{background:var(--surface);border:1px solid var(--hairline);border-radius:var(--radius);
  overflow:hidden;box-shadow:var(--shadow);display:flex;flex-direction:column;
  transition:transform .14s ease,border-color .14s ease;}
.card:hover{transform:translateY(-3px);border-color:var(--muted);}
.thumb{width:100%;aspect-ratio:1/1;object-fit:cover;display:block;background:var(--surface2);
  border-bottom:1px solid var(--hairline);}
.thumb-empty{display:grid;place-items:center;color:var(--muted);font-size:12.5px;
  background:repeating-linear-gradient(45deg,var(--surface2),var(--surface2) 10px,transparent 10px,transparent 20px);}
.card-body{padding:14px 15px 15px;display:flex;flex-direction:column;gap:8px;flex:1;}
.card-top{display:flex;justify-content:flex-start;align-items:center;gap:6px;}
.chip{font-size:11px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;
  padding:3px 9px;border-radius:999px;font-family:ui-monospace,monospace;}
.chip-atomic{background:var(--teal-bg);color:var(--teal);}
.chip-composite{background:var(--amber-bg);color:var(--amber);}
.chip-instr{background:var(--surface2);color:var(--muted);border:1px solid var(--hairline);}
.chip-multi{background:transparent;color:var(--amber);border:1px solid color-mix(in srgb,var(--amber) 45%,transparent);}
.card-top{flex-wrap:wrap;}
.status{margin-left:auto;font-size:11.5px;color:var(--muted);display:inline-flex;align-items:center;gap:5px;
  font-family:ui-monospace,monospace;}
.status i{width:7px;height:7px;border-radius:50%;background:var(--muted);flex:none;}
.status-ready{color:var(--green);}
.status-ready i{background:var(--green);box-shadow:0 0 0 3px color-mix(in srgb,var(--green) 20%,transparent);}
.status-downloaded i{background:var(--amber);}
.task-name{font-size:15.5px;margin:0;letter-spacing:-.01em;font-family:ui-monospace,"SF Mono",Menlo,monospace;
  font-weight:600;word-break:break-word;}
.task-desc{font-size:13.5px;color:var(--muted);margin:0;flex:1;}
.horizon{display:flex;align-items:center;gap:8px;margin-top:4px;}
.horizon-l{font-size:10.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);}
.horizon-bar{flex:1;height:4px;border-radius:2px;background:var(--surface2);overflow:hidden;}
.horizon-bar i{display:block;height:100%;border-radius:2px;
  background:linear-gradient(90deg,var(--teal),var(--amber));}
.horizon-v{font-size:12px;font-variant-numeric:tabular-nums;color:var(--ink);font-family:ui-monospace,monospace;}
.empty-note{color:var(--muted);font-size:14px;padding:30px 0;text-align:center;display:none;}

/* dataset schema */
.ds{margin:16px 0 8px;}
.ds-facts{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:18px;}
.fact{background:var(--surface);border:1px solid var(--hairline);border-radius:10px;
  padding:10px 14px;display:flex;flex-direction:column;gap:1px;min-width:120px;}
.fact-v{font-size:15px;font-weight:600;}
.fact-l{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);}
.ds-grid{display:grid;grid-template-columns:1.1fr 1fr 1fr;gap:16px;}
.ds-block{background:var(--surface);border:1px solid var(--hairline);border-radius:var(--radius);
  padding:16px 18px;box-shadow:var(--shadow);}
.ds-block h3{font-size:14px;margin:0 0 12px;display:flex;justify-content:space-between;
  align-items:baseline;gap:8px;flex-wrap:wrap;}
.dim{font-size:11px;color:var(--muted);font-weight:400;}
.cams{list-style:none;margin:0;padding:0;display:flex;flex-direction:column;gap:10px;}
.cams li{display:flex;flex-direction:column;gap:2px;}
.cams code{font-size:11.5px;color:var(--teal);word-break:break-all;
  font-family:ui-monospace,monospace;}
.cams span{font-size:12.5px;color:var(--muted);}
.schema{width:100%;border-collapse:collapse;font-size:13px;}
.schema td{padding:5px 0;border-bottom:1px solid var(--hairline);vertical-align:top;}
.schema tr:last-child td{border-bottom:0;}
.schema td:first-child{width:56px;color:var(--amber);font-variant-numeric:tabular-nums;}
.schema td:last-child{color:var(--muted);}

footer{border-top:1px solid var(--hairline);color:var(--muted);font-size:13px;
  padding:24px 0 48px;}
@media (max-width:820px){.tiles{grid-template-columns:repeat(2,1fr)}.cols{grid-template-columns:1fr}
  .ds-grid{grid-template-columns:1fr}.head{flex-direction:column}.search{margin-left:0;width:100%}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

_JS = """
(function(){
  var root=document.documentElement;
  var btn=document.getElementById('themeBtn');
  if(btn){btn.addEventListener('click',function(){
    var cur=root.getAttribute('data-theme');
    if(!cur){cur=matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';}
    root.setAttribute('data-theme',cur==='dark'?'light':'dark');
  });}
  var cards=[].slice.call(document.querySelectorAll('.card'));
  var filters=[].slice.call(document.querySelectorAll('.filter'));
  var search=document.getElementById('search');
  var empty=document.getElementById('empty');
  var mode='all';
  function apply(){
    var q=(search&&search.value||'').trim().toLowerCase();
    var shown=0;
    cards.forEach(function(c){
      var okF = mode==='all' || c.dataset.cat===mode || c.dataset.status===mode || c.dataset.instr===mode;
      var okQ = !q || c.dataset.name.indexOf(q)>-1;
      var vis=okF&&okQ; c.style.display=vis?'':'none'; if(vis)shown++;
    });
    if(empty)empty.style.display=shown?'none':'block';
  }
  filters.forEach(function(f){f.addEventListener('click',function(){
    filters.forEach(function(x){x.setAttribute('aria-pressed','false');});
    f.setAttribute('aria-pressed','true'); mode=f.dataset.filter; apply();
  });});
  if(search)search.addEventListener('input',apply);
})();
"""

_TEMPLATE = """<style>{css}</style>
<header><div class="wrap head">
  <div>
    <p class="eyebrow">Project dashboard</p>
    <h1>{title}</h1>
    <p class="tagline">{tagline}</p>
  </div>
  <button class="theme-btn" id="themeBtn" type="button">◐ theme</button>
</div></header>

<main class="wrap">
  <div class="tiles">{stat_tiles}</div>

  <div class="cols">
    <div>
      <h2>Implemented</h2>
      <ul class="list">{implemented}</ul>
    </div>
    <div>
      <h2>Roadmap</h2>
      <ul class="list">{roadmap}</ul>
    </div>
  </div>

  <section>
    <div class="section-head"><h2>Dataset schema</h2>
      <span class="count">what each converted LeRobot v3.0 task contains</span></div>
    <div class="rule"></div>
    <div class="ds">
      <div class="ds-facts">{ds_facts}</div>
      <div class="ds-grid">
        <div class="ds-block">
          <h3>Cameras</h3>
          <ul class="cams">{ds_cams}</ul>
        </div>
        <div class="ds-block">
          <h3>State <span class="dim mono">observation.state · 16-d</span></h3>
          <table class="schema"><tbody>{ds_state}</tbody></table>
        </div>
        <div class="ds-block">
          <h3>Action <span class="dim mono">action · 12-d</span></h3>
          <table class="schema"><tbody>{ds_action}</tbody></table>
        </div>
      </div>
    </div>
  </section>

  <section>
    <div class="section-head">
      <h2>RoboCasa 365 · target tasks</h2>
      <span class="count">{n_ready}/{n_total} v3.0 ready · {n_atomic} atomic · {n_comp} composite ·
        {n_single} single-instruction · {n_multi} multi-instruction</span>
    </div>
    <div class="controls">
      <div class="filters">
        <button class="filter" data-filter="all" aria-pressed="true">All</button>
        <button class="filter" data-filter="atomic" aria-pressed="false">Atomic</button>
        <button class="filter" data-filter="composite" aria-pressed="false">Composite</button>
        <button class="filter" data-filter="single" aria-pressed="false">Single instr.</button>
        <button class="filter" data-filter="multi" aria-pressed="false">Multi instr.</button>
        <button class="filter" data-filter="ready" aria-pressed="false">v3.0 ready</button>
        <button class="filter" data-filter="pending" aria-pressed="false">Not yet</button>
      </div>
      <input class="search" id="search" type="search" placeholder="Search tasks…" aria-label="Search tasks">
    </div>
    <div class="grid">{cards}</div>
    <p class="empty-note" id="empty">No tasks match.</p>
  </section>
</main>

<footer><div class="wrap">
  Generated by <span class="mono">examples/robocasa/gen_dashboard.py</span> ·
  thumbnails sampled from the converted agent-view videos · re-run to refresh.
</div></footer>
<script>{js}</script>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=Path("/data5/jellyho/robocasa365"),
                    help="Converted-dataset dir, scanned for per-task status.")
    ap.add_argument("--mode", choices=["site", "artifact"], default="site",
                    help="'site' (default): the GitHub Pages dashboard; references site/videos/<Task>.mp4 "
                         "for full-trajectory playback (serve locally or via Pages). "
                         "'artifact': a self-contained single file with static thumbnails and no videos.")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output HTML path. Defaults to dashboard.html (artifact) or site/index.html (site).")
    args = ap.parse_args()

    out = args.out or (_HERE / ("site/index.html" if args.mode == "site" else "dashboard.html"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(args.output_dir, mode=args.mode))

    n_thumbs = len(list(_ASSETS.glob("*.jpg"))) if _ASSETS.exists() else 0
    if args.mode == "site":
        n_vid = len(list((_HERE / "site" / "videos").glob("*.mp4"))) if (_HERE / "site" / "videos").exists() else 0
        print(f"Wrote {out} ({out.stat().st_size // 1024} KB). Referencing {n_vid} full-trajectory videos.")
        print(f"Serve:  cd {out.parent}  &&  python -m http.server 8000   →   http://localhost:8000/")
    else:
        print(f"Wrote {out} ({out.stat().st_size // 1024} KB, {n_thumbs} thumbnails embedded).")


if __name__ == "__main__":
    main()

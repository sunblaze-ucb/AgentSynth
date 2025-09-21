# make_mm_chat_from_traces.py
import os, io, json
from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm
from huggingface_hub import list_repo_files, hf_hub_download
from PIL import Image
import base64

REPO_ID = "sunblaze-ucb/AgentSynth"
OUT_DIR = Path("mm_chat")
IMG_DIR = OUT_DIR / "images"
OUT_JSON = OUT_DIR / "dataset.json"

RESIZE_LONGEST = 768  # keep training cost down; change to 512/768/1024

from prompts import SYS_TASK_ACTION

def save_png(b64, out_path, max_side=RESIZE_LONGEST):
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if max_side:
        w, h = img.size
        s = max(w, h)
        if s > max_side:
            scale = max_side / s
            img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
    img.save(out_path, "PNG", compress_level=6)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    files = list_repo_files(REPO_ID, repo_type="dataset")
    zips = sorted([p for p in files if p.startswith("trajectory/") and p.endswith(".zip")])

    records = []
    ex_id = 0

    for zp in tqdm(zips, desc="Zip archives"):
        local = hf_hub_download(REPO_ID, filename=zp, repo_type="dataset")
        with ZipFile(local) as zf:
            inner = sorted(n for n in zf.namelist() if n.endswith(".json"))
            for name in tqdm(inner, leave=False, desc=f"Files in {os.path.basename(zp)}"):
                with zf.open(name) as f:
                    traj = json.load(io.TextIOWrapper(f, encoding="utf-8"))

                # task-level arrays
                thoughts_all  = traj.get("thoughts") or []
                actions_all   = traj.get("actions") or []
                screenshots_all = traj.get("screenshots") or []
                done_all      = traj.get("done") or []
                info_history  = traj.get("info_history") or []
                task_hist     = traj.get("task_history") or []
                task_hist_orig= traj.get("task_history_original") or []

                n_slots = min(len(thoughts_all), len(actions_all), len(screenshots_all), len(done_all))
                if n_slots == 0: 
                    continue

                for t in range(n_slots):
                    # you can filter to successes only by uncommenting:
                    # if not (isinstance(done_all[t], bool) and done_all[t]): 
                    #     continue

                    thoughts = thoughts_all[t] or []
                    actions  = actions_all[t]  or []
                    shots    = screenshots_all[t] or []

                    steps = min(len(thoughts), len(actions), len(shots))
                    if steps == 0:
                        continue

                    # subtask text (match inference)
                    if t < len(task_hist_orig) and isinstance(task_hist_orig[t], str) and task_hist_orig[t].strip():
                        subtask = task_hist_orig[t]
                    elif t < len(task_hist) and isinstance(task_hist[t], str):
                        subtask = task_hist[t]
                    else:
                        subtask = "Follow the subtask implied by the UI."

                    # build per-step samples
                    for i in range(steps):
                        # histories *before* step i
                        prev_thoughts = [str(x) for x in thoughts[:i]]
                        prev_actions  = [str(x) for x in actions[:i]]
                        info_slice    = info_history[:t]

                        user_text = (
                            f"Given the task: {subtask}. "
                            f"You have gathered some information {info_slice}. "
                            f"Here is your previous thinking process to complete the task {prev_thoughts}. "
                            f"Here is your previous actions tried {prev_actions}. "
                            f"Here is the current screenshot, what would be the next action?"
                        )

                        label = {"action": str(actions[i]), "thoughts": str(thoughts[i])}

                        # save image
                        img_name = f"ex_{ex_id:09d}.png"
                        save_png(shots[i], IMG_DIR / img_name)

                        records.append({
                            "id": f"ex_{ex_id:09d}",
                            "images": [f"images/{img_name}"],
                            "conversations": [
                                # include your exact system prompt if you want to train it too
                                {"from":"system","value": SYS_TASK_ACTION},
                                {"from":"human","value": "<image>\n" + user_text},
                                {"from":"gpt","value": json.dumps(label, ensure_ascii=False)}
                            ]
                        })
                        ex_id += 1

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"Wrote {len(records)} samples → {OUT_JSON} ; images under {IMG_DIR}")

if __name__ == "__main__":
    main()

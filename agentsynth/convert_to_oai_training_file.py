"""
Script for converting the AgentSynth dataset to a training file.
WARNING: The training files are large, make sure you have >40GB of free space.
"""

import os, io, json, itertools
from typing import List, Dict, Any
from zipfile import ZipFile
from huggingface_hub import list_repo_files, hf_hub_download
from tqdm import tqdm

# ============== CONFIG ==============
REPO_ID = "sunblaze-ucb/AgentSynth"
OUT_JSONL = "openai_finetune_per_action.jsonl"

# Use OpenAI Responses-style messages (input_text/input_image + output_text)
EMIT_MODE = "responses"   # or "chat" for Chat Completions-style

# Keep only successful subtasks (done[t] == True). Set to False to keep all.
FILTER_SUCCESS_ONLY = True

# Dev-time limits
MAX_ZIPS = None
MAX_FILES_PER_ZIP = None
MAX_ROWS = None

# Try to import your exact system prompt
from prompts import SYS_TASK_ACTION as SYSTEM_TEXT
# ====================================


def list_zip_paths(repo_id: str) -> List[str]:
    files = list_repo_files(repo_id, repo_type="dataset")
    zips = [p for p in files if p.startswith("trajectory/") and p.endswith(".zip")]
    zips.sort()
    return zips[:MAX_ZIPS] if MAX_ZIPS else zips


def normalize_str(x) -> str:
    if isinstance(x, str): return x
    if isinstance(x, (int, float, bool)): return str(x)
    return json.dumps(x, ensure_ascii=False)


def gen_user_prompt(task: str, info_history_slice, thoughts_hist_step, actions_hist_step) -> str:
    # EXACT shape from utils.generate_action(...)
    # Given the task: {task}. You have gathered some information {info_history}. Here is your previous thinking process...
    return (
        f"Given the task: {task}. "
        f"You have gathered some information {info_history_slice}. "
        f"Here is your previous thinking process to complete the task {thoughts_hist_step}. "
        f"Here is your previous actions tried {actions_hist_step}. "
        f"Here is the current screenshot, what would be the next action?"
    )


def emit_openai_row(system_text: str, user_text: str, image_b64: str, action: str, thought: str, mode="responses") -> Dict[str, Any]:
    label = {"thoughts": thought, "action": action}
    img_url = f"data:image/png;base64,{image_b64}"

    if mode == "responses":
        return {
            "messages": [
                {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
                {"role": "user", "content": [
                    {"type": "input_text", "text": user_text},
                    {"type": "input_image", "image_url": img_url},
                ]},
                {"role": "assistant", "content": [{"type": "output_text", "text": json.dumps(label, ensure_ascii=False)}]},
            ]
        }
    else:
        return {
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": system_text}]},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": img_url},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": json.dumps(label, ensure_ascii=False)}]},
            ]
        }


def build_jsonl():
    zips = list_zip_paths(REPO_ID)
    total = 0

    with open(OUT_JSONL, "w", encoding="utf-8") as fout:
        for zp in tqdm(zips, desc="Zip archives"):
            local_zip = hf_hub_download(REPO_ID, filename=zp, repo_type="dataset")
            with ZipFile(local_zip) as zf:
                inner = sorted([n for n in zf.namelist() if n.endswith(".json")])
                if MAX_FILES_PER_ZIP:
                    inner = inner[:MAX_FILES_PER_ZIP]

                for name in tqdm(inner, desc=f"Files in {os.path.basename(zp)}", leave=False):
                    with zf.open(name) as f:
                        traj = json.load(io.TextIOWrapper(f, encoding="utf-8"))

                    # Task-level arrays (one list per subtask)
                    thoughts_all = traj.get("thoughts") or []
                    actions_all = traj.get("actions") or []
                    screenshots_all = traj.get("screenshots") or []
                    done_all = traj.get("done") or []
                    info_history = traj.get("info_history") or []
                    task_hist = traj.get("task_history") or []
                    task_hist_orig = traj.get("task_history_original") or []

                    n_slots = min(len(thoughts_all), len(actions_all), len(screenshots_all), len(done_all))
                    if n_slots == 0:
                        continue

                    # Iterate subtasks (t)
                    t_iter = range(n_slots)
                    if FILTER_SUCCESS_ONLY:
                        t_iter = [t for t in t_iter if isinstance(done_all[t], bool) and done_all[t] is True]

                    for t in tqdm(t_iter, desc="Subtasks", leave=False):
                        thoughts = thoughts_all[t] or []
                        actions = actions_all[t] or []
                        screenshots = screenshots_all[t] or []

                        # Align lengths per step; no screenshot → skip that step (often the terminal DONE)
                        n_steps = min(len(thoughts), len(actions), len(screenshots))
                        if n_steps == 0:
                            continue

                        # Task string used during execution of this subtask
                        task_str = (task_hist_orig[t] if t < len(task_hist_orig) and isinstance(task_hist_orig[t], str) and task_hist_orig[t].strip()
                                    else task_hist[t] if t < len(task_hist) else "Follow the subtask implied by the UI.")

                        # Info available to the agent before starting this subtask
                        info_hist_for_subtask = info_history[:t]

                        # Per-step examples
                        for i in range(n_steps):
                            # histories seen by generate_action(...) BEFORE deciding step i
                            thoughts_hist_step = [normalize_str(x) for x in thoughts[:i]]
                            actions_hist_step = [normalize_str(x) for x in actions[:i]]

                            # target is exactly that step's action+thought
                            action_i = normalize_str(actions[i])
                            thought_i = normalize_str(thoughts[i])
                            image_i = screenshots[i]

                            user_text = gen_user_prompt(task_str, info_hist_for_subtask, thoughts_hist_step, actions_hist_step)
                            row = emit_openai_row(SYSTEM_TEXT, user_text, image_i, action_i, thought_i, mode=EMIT_MODE)
                            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                            total += 1

                            if MAX_ROWS and total >= MAX_ROWS:
                                print(f"Reached MAX_ROWS={MAX_ROWS}")
                                print(f"Wrote {total} rows to {OUT_JSONL}")
                                return

    print(f"Wrote {total} rows to {OUT_JSONL}")


if __name__ == "__main__":
    build_jsonl()

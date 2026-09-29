"""Does a constant pick (e.g. `drive.x = backward` on every call) come from the adapter or the checkpoint?

Re-scores logged WorldState requests (`{"state", "questions"}` bodies from the agent's raw traces, e.g. 2 per
case) under other option renderings, option-order averaging, and a state without the task prose (so nothing
is truncated), and counts agreement of `drive.yaw` with the target's bearing when the way is clear.

    modal run benchmarks/dimos_nav/probe_options.py --sample sample.json --out probe.json
"""
import collections
import json
import sys

import modal

app = modal.App("laya-dimos-nav-probe")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", "torchvision==0.29.0", "transformers==5.17.0", "safetensors", "huggingface_hub",
                 "numpy", "pillow", "num2words")
    .env({"HF_HOME": "/cache/hf"})
    .add_local_dir("laya", "/root/laya")
    .add_local_dir("benchmarks/dimos_nav", "/root/dimos_nav")
)
hf_vol = modal.Volume.from_name("laya-hf-cache")


def yaw_oracle(state):
    """Target bearing word when the way is clear (the questions' own rule); None otherwise."""
    if state.get("way_to_target", {}).get("state") != "clear" or not state.get("objects"):
        return None
    b = state["objects"][0].get("bearing", "")
    if b == "ahead":
        return "none"
    if b.endswith("left") or b == "left":
        return "turn_left"
    if b.endswith("right") or b == "right":
        return "turn_right"
    return None


@app.function(image=image, gpu="L4", volumes={"/cache/hf": hf_vol}, timeout=3600)
def probe(sample, overrides=None):
    sys.path.insert(0, "/root/dimos_nav")
    import laya
    from systemone_server import to_laya

    agent = laya.load_vlm("thaitea/laya-vision", revision="f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc", **(overrides or {}))

    def variants(body):
        qs = {k: v for k, v in body["questions"].items() if k in ("drive.x", "drive.yaw", "task")}
        as_run = to_laya(qs)
        full = {k: {**v, "criteria": {c: (None if d is None else json.dumps(d)) for c, d in qs[k]["criteria"].items()}}
                for k, v in as_run.items()}
        labels = {k: {**v, "criteria": {c: None for c in v["criteria"]}} for k, v in as_run.items()}
        short_state = {k: v for k, v in body["state"].items() if k != "task"}
        return {
            "as_run": (body["state"], as_run, 1),
            "as_run_perm6": (body["state"], as_run, 6),
            "full_json_criteria": (body["state"], full, 1),
            "labels_only": (body["state"], labels, 1),
            "no_task_prose": (short_state, as_run, 1),
        }

    picks = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    yaw_agree = collections.Counter()
    yaw_n = 0
    cut = collections.Counter()
    for body in sample:
        oracle = yaw_oracle(body["state"])
        yaw_n += oracle is not None
        for name, (state, qs, perm) in variants(body).items():
            ans = agent.predict(state, qs, n_permutations=perm)["answers"]
            cut[name] += any("truncated" in a for a in ans.values())
            for q, a in ans.items():
                picks[name][q][a["choice"]] += 1
            if oracle is not None:
                yaw_agree[name] += ans["drive.yaw"]["choice"] == oracle
    return {"picks": {n: {q: dict(c) for q, c in d.items()} for n, d in picks.items()},
            "overrides": overrides or {}, "calls_with_truncation": dict(cut), "yaw_oracle_n": yaw_n, "yaw_agree": dict(yaw_agree),
            "yaw_oracle_dist": dict(collections.Counter(filter(None, (yaw_oracle(b["state"]) for b in sample))))}


@app.local_entrypoint()
def main(sample: str, out: str, max_len: int = 0, head_max_len: int = 0):
    import os
    if os.path.exists(out):
        raise SystemExit(f"{out} exists (create-only)")
    sample = json.load(open(sample))
    res = probe.remote(sample, {k: v for k, v in (("max_len", max_len), ("head_max_len", head_max_len)) if v})
    print(json.dumps(res, indent=1))
    json.dump(res, open(out, "w"), indent=1)

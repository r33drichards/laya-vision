"""Modal job for site-docs/concepts/game-caching.md: one game decision's latency breakdown on a GPU, and duplicate-frame rates.

    modal run modal_game_cache.py::main            # L4: examples/bench_game_step.py in bf16 and fp32, causal and block
    modal run modal_game_cache.py::main --mode graphs    # only the CUDA-graph comparison and the ViZDoom duplicates
    modal run modal_game_cache.py::verify           # StaticStep parity tests on CUDA + its ms/decision
    modal run modal_game_cache.py::tests            # tests/test_vlm.py in full on a CPU container
    modal run modal_game_cache.py::main --gpu A10G

No training and no checkpoints: a fresh agent on the SmolVLM-256M backbone (timing does not depend on the weights).
Volumes (created out of band; never ``modal deploy`` this app): laya-hf-cache -> /cache/hf (HF_HOME).
"""
import json

import modal

app = modal.App("laya-game-cache")
hf_vol = modal.Volume.from_name("laya-hf-cache")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", "torchvision==0.29.0", "transformers==5.17.0", "safetensors", "huggingface_hub",
                 "numpy", "pillow", "num2words", "ale-py", "gymnasium", "vizdoom")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("laya")
    .add_local_file("examples/bench_game_step.py", "/root/bench_game_step.py")
    .add_local_dir("tests", "/root/tests")
)


def _duplicate_frames(games=("Breakout", "Pong", "Boxing", "Freeway", "SpaceInvaders"), steps=2000, seed=0):
    """How often a random policy sees exactly the frame it saw last step (an answer that could be reused as is),
    and how many pixels change between consecutive decisions otherwise."""
    import ale_py
    import gymnasium as gym
    import numpy as np

    gym.register_envs(ale_py)
    out = {}
    for game in games:
        env = gym.make("ALE/%s-v5" % game)
        rng = np.random.default_rng(seed)
        obs, _ = env.reset(seed=seed)
        same, changed = 0, []
        for _ in range(steps):
            nxt, _, term, trunc, _ = env.step(int(rng.integers(env.action_space.n)))
            diff = np.any(nxt != obs, -1)
            same += int(not diff.any())
            changed.append(float(diff.mean()))
            obs = nxt if not (term or trunc) else env.reset()[0]
        env.close()
        out[game] = {"steps": steps, "identical_to_previous": round(same / steps, 4),
                     "median_frac_pixels_changed": round(float(np.median(changed)), 5)}
        print(game, out[game], flush=True)
    return out


def _doom_duplicates(scenarios=("basic", "defend_the_center", "deadly_corridor"), steps=1000, tics=4, seed=0):
    """The same for ViZDoom at the live viewer's settings (320x240 RGB, each action held for 4 tics)."""
    import os

    import numpy as np
    import vizdoom as vzd

    out = {}
    for sc in scenarios:
        game = vzd.DoomGame()
        game.load_config(os.path.join(vzd.scenarios_path, sc + ".cfg"))
        game.set_window_visible(False)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        game.set_seed(seed)
        game.init()
        n = game.get_available_buttons_size()
        rng = np.random.default_rng(seed)
        game.new_episode()
        obs = game.get_state().screen_buffer.copy()
        same, changed = 0, []
        for _ in range(steps):
            game.make_action([bool(i == rng.integers(n)) for i in range(n)], tics)
            if game.is_episode_finished():
                game.new_episode()
            nxt = game.get_state().screen_buffer.copy()
            diff = np.any(nxt != obs, -1)
            same += int(not diff.any())
            changed.append(float(diff.mean()))
            obs = nxt
        game.close()
        out[sc] = {"steps": steps, "identical_to_previous": round(same / steps, 4),
                   "median_frac_pixels_changed": round(float(np.median(changed)), 5)}
        print(sc, out[sc], flush=True)
    return out


def _run(argv):
    import sys

    sys.path.insert(0, "/root")
    import bench_game_step

    return bench_game_step.main(argv)


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=30 * 60, volumes={"/cache/hf": hf_vol})
def bench_gpu(reps: int = 30, mode: str = "all"):
    if mode == "graphs":
        res = {"doom_duplicates": _doom_duplicates()}
        for dtype in ("bf16", "fp32"):
            res["graphs_" + dtype] = _run(["--device", "cuda", "--dtype", dtype, "--reps", str(reps), "--only-graphs"])
        res["graphs_block_bf16"] = _run(["--device", "cuda", "--dtype", "bf16", "--reps", str(reps), "--only-graphs",
                                         "--option-attention", "block"])
        return json.dumps(res)
    res = {"duplicates": _duplicate_frames(), "doom_duplicates": _doom_duplicates()}
    for dtype in ("bf16", "fp32"):
        res["causal_" + dtype] = _run(["--device", "cuda", "--dtype", dtype, "--reps", str(reps)])
    res["block_bf16"] = _run(["--device", "cuda", "--dtype", "bf16", "--reps", str(reps), "--option-attention",
                              "block", "--no-batching", "--workloads", "1,2"])
    res["cpu4_fp32"] = _run(["--device", "cpu", "--reps", "5", "--threads", "4", "--no-batching", "--workloads",
                             "1,3"])
    return json.dumps(res)


@app.function(image=image, gpu="L4", cpu=4, memory=16384, timeout=30 * 60, volumes={"/cache/hf": hf_vol})
def verify_static(reps: int = 30):
    """The StaticStep parity tests on CUDA, then its ms/decision against ``predict`` / ``action_probs``."""
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)
    rc = subprocess.run([sys.executable, "-m", "pytest", "-q", "/root/tests/test_vlm.py", "-k",
                         "static_step or cuda_graph"], cwd="/root").returncode
    import numpy as np
    import torch

    sys.path.insert(0, "/root")
    from bench_game_step import WORKLOADS, timed
    from laya.atari_train import action_probs
    from laya.static_step import StaticStep
    from laya.vlm import VLMAgent

    res = {"pytest_returncode": rc, "rows": []}
    rng = np.random.default_rng(0)
    for dtype in ("bf16", "fp32"):
        agent = VLMAgent(backbone="HuggingFaceTB/SmolVLM-256M-Instruct", device="cuda", dtype=dtype,
                         option_attention="block")
        for i in (1, 3):
            label, shape, question = WORKLOADS[i]
            q = question["action"]
            for frames, batch in ((1, 1), (2, 1), (1, 4), (1, 16)):
                cur = [rng.integers(0, 256, shape, dtype=np.uint8) for _ in range(batch)]
                prev = [rng.integers(0, 256, shape, dtype=np.uint8) for _ in range(batch)] if frames == 2 else None
                step = StaticStep(agent, q, frames=frames, batch=batch)
                eager = timed(lambda: action_probs(agent, cur, q, prev), agent.device, reps)
                graph = timed(lambda: step.probs(cur, prev), agent.device, reps)
                d = float(np.abs(step.probs(cur, prev) - action_probs(agent, cur, q, prev)).max())
                row = {"dtype": dtype, "workload": label, "frames": frames, "batch": batch,
                       "action_probs_ms": eager, "static_step_ms": graph, "speedup": round(eager / graph, 2),
                       "ms_per_frame_static": round(graph / batch, 3), "max_abs_prob_diff": d}
                if batch == 1 and frames == 1:
                    row["predict_ms"] = timed(lambda: agent.predict({"image": cur[0]}, question), agent.device, reps)
                    row["answer_ms"] = timed(lambda: step.answer(cur[0]), agent.device, reps)
                print(json.dumps(row), flush=True)
                res["rows"].append(row)
        del agent
        torch.cuda.empty_cache()
    return json.dumps(res)


@app.function(image=image, cpu=8, memory=16384, timeout=40 * 60, volumes={"/cache/hf": hf_vol})
def test_suite_cpu():
    """``tests/test_vlm.py`` in full on a CPU container (the StaticStep tests run their eager path there)."""
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "/root/tests/test_vlm.py"],
                          cwd="/root", capture_output=True, text=True).stdout[-3000:]


@app.local_entrypoint()
def tests():
    print(test_suite_cpu.remote())


@app.local_entrypoint()
def verify(out: str = ""):
    text = json.dumps(json.loads(verify_static.remote()), indent=1)
    print(text)
    if out:
        with open(out, "w") as f:
            f.write(text)


@app.local_entrypoint()
def main(gpu: str = "L4", reps: int = 30, mode: str = "all", out: str = ""):
    fn = bench_gpu if gpu == "L4" else bench_gpu.with_options(gpu=gpu)
    text = json.dumps(json.loads(fn.remote(reps, mode)), indent=1)
    print(text)
    if out:
        with open(out, "w") as f:
            f.write(text)

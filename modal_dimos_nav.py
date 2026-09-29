"""Dimensional's System One navigation eval (Habitat-Sim, HSSD) with a Laya Vision checkpoint as the policy.

The eval (https://research.dimensional.org/system-one-navigation/) drives a robot to a named object in a
furnished house from a text-only WorldState; each step the policy answers six typed questions (drive.x,
drive.y, drive.yaw, stop, task, target). dimos's own `TypeSafeAgent` builds the WorldState, asks the questions
and turns the answers into velocity commands; it calls TypeSafe's `POST /v1/systemone`, which
`benchmarks/dimos_nav/systemone_server.py` serves from a Laya checkpoint in the same container
(`TYPESAFE_BASE_URL=http://127.0.0.1:8765`). Nothing in dimos is changed.

    modal run modal_dimos_nav.py::prepare_hssd                 # once: HSSD scenes onto the dimos-habitat volume
    modal run modal_dimos_nav.py::list_cases
    modal run modal_dimos_nav.py::main --out results/dimos-nav/<new-name> [--cases a,b] [--limit N]

One L4 per case (Habitat renders through EGL, the checkpoint runs on the same GPU). Each case's dimos run dir
(results.jsonl, nav_metrics.json, the agent's request/response traces) and the server's call log come back
as `<out>/<case>.tar.gz`; `benchmarks/dimos_nav/summarize.py <out>` scores them.
"""

import io
import json
import os
import subprocess
import tarfile
import time

import modal

app = modal.App("laya-dimos-nav")

# dimos feat/typesafe-world-state, the branch the eval page runs; the HSSD ground truth ships in
# dimensionalOS/dimos#4211.
DIMOS_COMMIT = "3cf006dd607631988c2d1680b9c28c59c37909f6"
GROUND_TRUTH_COMMIT = "5448392aab080e9a0c1fb2588d9bdcbff657b0bf"
HSSD_REPO = "hssd/hssd-hab"
HSSD_REVISION = "4369cb9876214c7fbebcf552eb532380e4d287e4"
LAYA_MODEL = "thaitea/laya-vision"
LAYA_REVISION = "f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc"
SCENES_DIR = "/app/dimos/evals/suites/scenes/habitat"
SUITE = "dimos.evals.suites.habitat_nav"

habitat_vol = modal.Volume.from_name("dimos-habitat", create_if_missing=True)
hf_vol = modal.Volume.from_name("laya-hf-cache")

UV = "/root/.local/bin/uv"
nav_image = (
    modal.Image.from_registry("ubuntu:22.04", add_python="3.12")
    .env({"DEBIAN_FRONTEND": "noninteractive"})
    .apt_install("curl", "wget", "ca-certificates", "git", "git-lfs", "build-essential", "pkg-config", "bzip2",
                 "libegl1", "libgl1", "libglib2.0-0", "libgomp1", "portaudio19-dev", "libturbojpeg0-dev",
                 "libssl-dev", "clang", "cmake")
    .run_commands(
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local bin/micromamba",
        "curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal",
    )
    .run_commands(
        "git clone https://github.com/dimensionalOS/dimos.git /app",
        f"cd /app && git checkout {DIMOS_COMMIT} && git fetch origin {GROUND_TRUTH_COMMIT} "
        f"&& git checkout {GROUND_TRUTH_COMMIT} -- misc/habitat/ground_truth",
    )
    .run_commands(f"cd /app && {UV} sync --frozen --no-dev")
    .run_commands("cd /app/dimos/simulation/habitat/nix && ./install.sh")
    .run_commands("cd /app/dimos/navigation/nav_3d/mls_planner/rust && . /root/.cargo/env && cargo build --release")
    .run_commands(
        f"{UV} venv /opt/laya --python 3.12",
        f"{UV} pip install --python /opt/laya/bin/python torch==2.14.0 torchvision==0.29.0 transformers==5.17.0 "
        "safetensors huggingface_hub numpy pillow num2words",
    )
    # The NVIDIA EGL vendor file; Modal mounts the driver's libraries but not glvnd's ICD entry.
    .run_commands(
        "mkdir -p /usr/share/glvnd/egl_vendor.d && printf '{\"file_format_version\":\"1.0.0\",\"ICD\":"
        "{\"library_path\":\"libEGL_nvidia.so.0\"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json"
    )
    .env({"HF_HOME": "/cache/hf", "PATH": "/app/.venv/bin:/root/.cargo/bin:/root/.local/bin:/usr/local/bin:/usr/bin:/bin",
          "DIMOS_TRANSPORT": "zenoh", "CI": "1", "TYPESAFE_API_KEY": "laya-local",
          "TYPESAFE_BASE_URL": "http://127.0.0.1:8765", "NVIDIA_DRIVER_CAPABILITIES": "all"})
    .add_local_dir("laya", "/opt/laya-src/laya")
    .add_local_dir("benchmarks/dimos_nav", "/opt/laya-src/dimos_nav")
)


@app.function(image=nav_image, volumes={"/data": habitat_vol}, timeout=6 * 3600, cpu=8,
              secrets=[modal.Secret.from_name("huggingface-thaitea")])  # anonymous listing hits 429
def prepare_hssd(rev: str = HSSD_REVISION) -> dict:
    """What the suite's HSSD scenes load: the dataset config, semantics, each scene's instance and stage, and
    every object template (config, mesh, decomposed parts) those scenes place. A full snapshot is ~100k
    files, far over the Hub's 2,500-requests-per-5-minutes API quota, so this lists only the directories it
    needs and fetches files through `resolve` URLs, backing off on 429."""
    import concurrent.futures
    import requests

    out = "/data/hssd-hab"
    if os.path.exists(f"{out}/manifest.json"):
        return json.load(open(f"{out}/manifest.json"))
    hdr = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    base = f"https://huggingface.co/datasets/{HSSD_REPO}"

    def http(url, **kw):
        for _ in range(40):
            r = requests.get(url, headers=hdr, timeout=120, **kw)
            if r.status_code != 429:
                r.raise_for_status()
                return r
            print("429, waiting 60 s", flush=True)
            time.sleep(60)
        raise RuntimeError(f"still rate limited: {url}")

    def ls(d):
        url, got = f"https://huggingface.co/api/datasets/{HSSD_REPO}/tree/{rev}/{d}?limit=1000", []
        while url:
            r = http(url)
            got += [x["path"] for x in r.json() if x["type"] == "file"]
            url = r.links.get("next", {}).get("url")
        return got

    def get(path):
        dst = f"{out}/{path}"
        if os.path.exists(dst):
            return
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with http(f"{base}/resolve/{rev}/{path}", stream=True) as r, open(dst + ".part", "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        os.replace(dst + ".part", dst)

    def get_all(paths):
        with concurrent.futures.ThreadPoolExecutor(8) as ex:
            list(ex.map(get, paths))

    scenes = [c["scene_id"] for c in _cases() if "hssd" in (c.get("scene_dataset_config") or "")]
    first = ["hssd-hab.scene_dataset_config.json"] + ls("semantics")
    first += [f"scenes/{s}.scene_instance.json" for s in scenes]
    first += [f"stages/{s}{ext}" for s in scenes for ext in (".glb", ".stage_config.json")]
    get_all(first)
    templates = set()
    for s in scenes:
        inst = json.load(open(f"{out}/scenes/{s}.scene_instance.json"))
        templates |= {o["template_name"] for o in inst.get("object_instances", [])}
    bases = {t.split("_part_", 1)[0] for t in templates}
    listing = [p for c in sorted({t[0] for t in bases}) for p in ls(f"objects/{c}")] + ls("objects/openings")
    decomposed = {d.split("/")[-1] for d in _dirs(http, rev, "objects/decomposed")}
    for t in sorted(bases & decomposed):
        listing += ls(f"objects/decomposed/{t}")
    want = [p for p in listing if os.path.basename(p).split(".", 1)[0].split("_part_", 1)[0] in bases]
    print(f"{len(scenes)} scenes, {len(bases)} templates, {len(want)} object files", flush=True)
    get_all(want)
    missing = [t for t in bases if not any(os.path.basename(p).startswith(t) for p in want)]
    manifest = {"repo": HSSD_REPO, "revision": rev, "scenes": scenes, "templates": len(bases),
                "files": len(first) + len(want), "templates_not_found": sorted(missing), "time": time.time()}
    json.dump(manifest, open(f"{out}/manifest.json", "w"))
    habitat_vol.commit()
    return manifest


def _dirs(http, rev, d):
    url, got = f"https://huggingface.co/api/datasets/{HSSD_REPO}/tree/{rev}/{d}?limit=1000", []
    while url:
        r = http(url)
        got += [x["path"] for x in r.json() if x["type"] == "directory"]
        url = r.links.get("next", {}).get("url")
    return got


def _cases() -> list[dict]:
    out = []
    for f in sorted(os.listdir(SCENES_DIR)):
        if f.endswith(".json"):
            out.append({"file": f, **{k: v for k, v in json.load(open(f"{SCENES_DIR}/{f}")).items()
                                      if k in ("scene_id", "scene_dataset_config", "ground_truth", "cases")}})
    return out


@app.function(image=nav_image, volumes={"/data": habitat_vol}, timeout=600)
def case_ids() -> list[dict]:
    _link_data()
    r = subprocess.run(["python", "-c", f"import json; from {SUITE} import SUITE; "
                        "print(json.dumps([{'id': c.id, 'tags': sorted(c.tags)} for c in SUITE]))"],
                       cwd="/app", capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-3000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


def _link_data() -> None:
    dst = "/app/target/habitat/data/hssd-hab"
    if os.path.isdir("/data/hssd-hab") and not os.path.exists(dst):
        os.symlink("/data/hssd-hab", dst)


@app.function(image=nav_image, gpu="L4", volumes={"/data": habitat_vol, "/cache/hf": hf_vol}, timeout=3 * 3600,
              cpu=8, memory=32768)
def run_case(case_id: str, model: str = LAYA_MODEL, revision: str = LAYA_REVISION, timeout_s: int = 1800) -> bytes:
    _link_data()
    work = f"/work/{case_id}"
    os.makedirs(work, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": "/opt/laya-src", "XDG_STATE_HOME": f"{work}/state",
           "DIMOS_EVAL_TIMEOUT_S": str(timeout_s)}
    server = subprocess.Popen(
        ["/opt/laya/bin/python", "/opt/laya-src/dimos_nav/systemone_server.py", "--model", model, "--revision",
         revision, "--port", "8765", "--log", f"{work}/systemone.jsonl"],
        env=env, stdout=open(f"{work}/server.log", "w"), stderr=subprocess.STDOUT)
    import urllib.request
    for _ in range(600):
        try:
            urllib.request.urlopen("http://127.0.0.1:8765/", timeout=2)
            break
        except Exception:
            if server.poll() is not None:
                raise RuntimeError(open(f"{work}/server.log").read()[-3000:])
            time.sleep(1)
    t0 = time.time()
    r = subprocess.run(
        ["dimos", "evals", "run", SUITE, "--agent", "dimos.evals.agents.topic",
         "--set", 'modules=["type-safe-agent"]', "--set", "trace=TypeSafeAgent", "--case", case_id],
        cwd="/app", env=env, stdout=open(f"{work}/dimos.log", "w"), stderr=subprocess.STDOUT)
    server.terminate()
    json.dump({"case": case_id, "model": model, "revision": revision, "dimos_commit": DIMOS_COMMIT,
               "ground_truth_commit": GROUND_TRUTH_COMMIT, "returncode": r.returncode,
               "wall_s": time.time() - t0, "timeout_s": timeout_s,
               "hssd": json.load(open("/data/hssd-hab/manifest.json")) if os.path.exists("/data/hssd-hab/manifest.json") else None},
              open(f"{work}/case.json", "w"))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # recordings (memory.db) are big and only needed to re-grade; keep them out
        tar.add(work, arcname=case_id, filter=lambda ti: None if ti.name.endswith((".db", ".rrd", ".mp4")) else ti)
    return buf.getvalue()


@app.local_entrypoint()
def list_cases():
    for c in case_ids.remote():
        print(c["id"], " ".join(c["tags"]))


@app.local_entrypoint()
def main(out: str, cases: str = "", limit: int = 0, timeout_s: int = 1800):
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; results are create-only, pick a new --out")
    ids = [c for c in cases.split(",") if c] or [c["id"] for c in case_ids.remote()]
    if limit:
        ids = ids[:limit]
    os.makedirs(out)
    print(f"{len(ids)} cases -> {out}")
    for cid, blob in zip(ids, run_case.map(ids, kwargs={"timeout_s": timeout_s}, return_exceptions=True)):
        if isinstance(blob, Exception):
            print(f"{cid}: ERROR {blob!r}")
            continue
        open(f"{out}/{cid}.tar.gz", "wb").write(blob)
        print(f"{cid}: {len(blob) // 1024} KB")

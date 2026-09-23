"""Cheapest end-to-end Modal check for this repo (1 CPU, ~15 s, no GPU, no image build beyond debian_slim).

    modal run .claude/skills/modal/smoke.py        # run from the repo root

Proves: the client connects through the session proxy, local code uploads (add_local_dir), the shared volumes
mount, and which prepared datasets (``_READY``) and checkpoint roots exist. Run it before spending GPU time.
"""
import modal

app = modal.App("laya-smoke")
data_vol = modal.Volume.from_name("laya-datasets")
ckpt_vol = modal.Volume.from_name("laya-checkpoints")
image = modal.Image.debian_slim(python_version="3.12").add_local_dir("laya", "/root/laya_src")


@app.function(image=image, cpu=1, memory=1024, timeout=300,
              volumes={"/data": data_vol.read_only(), "/ckpt": ckpt_vol.read_only()})
def probe():
    import os
    ready = sorted(d for d in os.listdir("/data/vqa") if os.path.exists("/data/vqa/%s/_READY" % d))
    return {"code_files": len(os.listdir("/root/laya_src")), "ready_sets": ready, "ckpt_roots": sorted(os.listdir("/ckpt"))}


@app.local_entrypoint()
def main():
    r = probe.remote()
    print("uploaded laya/ files:", r["code_files"])
    print("checkpoint roots:", ", ".join(r["ckpt_roots"]))
    print("prepared datasets (%d):" % len(r["ready_sets"]), ", ".join(r["ready_sets"]))

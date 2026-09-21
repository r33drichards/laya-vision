"""Serve a saved run from the ``laya-checkpoints`` volume as an HTTP API on Modal (``laya.serve``).

    modal serve modal_serve.py                       # dev URL, hot reload
    modal deploy modal_serve.py                      # stable URL
    LAYA_RUN=modernvbert/mvb-3ep/best modal deploy modal_serve.py   # another checkpoint (see modal_app._ckpt_path)
    LAYA_GPU=T4 modal deploy modal_serve.py          # cheaper GPU; LAYA_GPU= (empty) serves on CPU

Kept apart from ``modal_app.py`` on purpose: that one is ``modal run`` only, this one is meant to be deployed.
It shares the image, the volumes and the checkpoint layout. The checkpoint loads once per container
(``@modal.enter``); the container stays warm ``SCALEDOWN_S`` after its last request.

    POST <url>/predict     multipart: images=@a.jpg -F images=@b.jpg -F questions='{...}'  (see laya.serve)
    GET  <url>/health
    GET  <url>/docs, <url>/openapi.json

Set ``LAYA_PROXY_AUTH=1`` to require Modal proxy auth tokens on the endpoint.
"""
import os

import modal

from modal_app import _ckpt_path, _with_local_code, base_image, ckpt_vol, hf_vol

RUN_NAME = os.environ.get("LAYA_RUN", "all3-3ep/best")
GPU = os.environ.get("LAYA_GPU", "L4") or None
SCALEDOWN_S = int(os.environ.get("LAYA_SCALEDOWN_S", "300"))

app = modal.App("laya-vision-serve")

image = _with_local_code(base_image.pip_install("fastapi[standard]").env({"LAYA_RUN": RUN_NAME}))


@app.cls(
    image=image,
    gpu=GPU,
    volumes={"/cache/hf": hf_vol, "/ckpt": ckpt_vol.read_only()},
    scaledown_window=SCALEDOWN_S,
    timeout=10 * 60,
)
class Server:
    @modal.enter()
    def load(self):
        from laya.serve import warm_up
        from laya.vlm import VLMAgent

        path = _ckpt_path(RUN_NAME)
        if not os.path.exists(os.path.join(path, "vlm_agent_config.json")):
            raise RuntimeError("no checkpoint at %s (LAYA_RUN=%s)" % (path, RUN_NAME))
        self.agent = VLMAgent(path, device="cuda" if GPU else "cpu")
        warm_up(self.agent)

    @modal.asgi_app(requires_proxy_auth=bool(os.environ.get("LAYA_PROXY_AUTH")))
    def api(self):
        from laya.serve import make_app

        return make_app(self.agent, run_name=RUN_NAME)

# Serve a checkpoint over HTTP

`laya-serve` puts [`predict`](../reference/predict.md) behind a small HTTP API, and `laya.client.LayaClient` calls it
from any Python program without torch. The server is
[`laya/server.py`](https://github.com/r33drichards/laya-vision/blob/main/laya/server.py), the client
[`laya/client.py`](https://github.com/r33drichards/laya-vision/blob/main/laya/client.py).

## Start the server

```bash
pip install "laya[serve]"
laya-serve                                   # thaitea/laya-vision at the commit the Space pins, as "laya-vision"
laya-serve --model prod=./runs/a --model next=thaitea/laya-vision@8b318c99d7ad3ce19c24369263463882eada9d1e
laya-serve --mock                            # a fake model (laya.mock): no weights, no torch, meaningless numbers
```

It listens on `127.0.0.1:8710` (`--host`, `--port`). Each `--model NAME=PATH` serves a local checkpoint directory or
a Hub id (pin it with `@revision`) under `NAME`; the first is the default. `--device` and `--dtype` go to
`load_vlm`.

Set `LAYA_API_KEY` to require `Authorization: Bearer <key>` on every route except `/health`. Browsers from other
origins are refused unless you pass `--cors`.

## Endpoints

| Route | Body | Returns |
|---|---|---|
| `POST /v1/systemone` | `{"state", "questions", "model"?, "temperature"?, "n_permutations"?, "strict"?}` | `predict`'s result, with `model` set to the served name and the model's `generation` |
| `POST /v1/rank` | `{"candidates": [...], "instructions"?, "state"?, "model"?, "temperature"?, "chunk_size"?}` | `{"ranked": [{"rank", "candidate", "prob", "round"}], "tournament", "usage"}` |
| `GET /v1/models` | | every served name, its source, `generation`, load time and last reload error |
| `GET /health` | | `{"ok", "mock", "default_model", "models": {name: generation}}` |

`state` and `questions` are `predict`'s, except that an image in `state["image"]` or `state["images"]` is a string:
base64, a `data:image/...;base64,` URL, or an `http(s)://` URL the server fetches (`--no-image-urls` turns fetching
off). The server never opens a local path named in a request.

Errors carry a `detail` message: 401 for a missing or wrong key, 422 for a malformed request (an unknown model, a
question with one option, bad base64, more than `--max-options` options), 502 when an image URL cannot be fetched,
503 when a model is not loaded.

## Rank many candidates

`/v1/rank` asks one `choice` question whose options are the candidates, word for word. Past `chunk_size` candidates
(default and maximum `--max-options`, 255) it runs a tournament: near-equal chunks are asked in one `predict` call,
the best `chunk_size // chunks` of each go on, and rounds repeat until one question holds the rest. Each candidate's
`prob` comes from the last round it played (`round`, the final being the highest), so compare probabilities only
within a round; the order puts the finalists first.

The limit of 255 is the API's, not a recommendation. A question's options share the checkpoint's `head_max_len`
token budget, so a long list is cut to fit (`tournament.truncated` names every question that was cut), and on the
causal backbones an early option cannot see the later ones. Pass a smaller `chunk_size`, such as 8 to 16, for long
lists, and check the ranking on your own data.

## Reload a checkpoint in place

A local checkpoint directory is watched: when `vlm_agent_config.json`, `model.safetensors` or `head.safetensors`
changes and then stays unchanged for `--reload-settle` seconds (2 by default), the next request loads it again,
swaps it in and bumps that model's `generation`. Requests keep using the old model while the new one loads, and a
load that fails keeps the old model and shows the error in `/v1/models`. `--no-reload` turns this off. Hub ids are
not watched.

## Call it from Python

```python
from laya.client import LayaClient, LayaError

client = LayaClient()                          # LAYA_BASE_URL (default http://127.0.0.1:8710), LAYA_API_KEY
r = client.system_one(
    {"image": "photo.jpg", "note": "front door camera"},        # a path, bytes, PIL image, URL or base64
    {"person": {"type": "noul", "instructions": "Is there a person at the door?"},
     "weather": {"type": "choice", "instructions": "What is the weather?", "criteria": ["sun", "rain", "snow"]}},
)
r.answers["person"].noul, r.answers["weather"].choice, r.answers["weather"].probabilities
client.rank(["a red car", "a blue bus"], "Which caption fits the image?", state={"image": "photo.jpg"}).best
```

Answers are `NoulAnswer`, `ChoiceAnswer` and `ScoreAnswer` dataclasses; a non-200 answer raises `LayaError` with its
`status` and `message`. `import laya.client` loads no torch.

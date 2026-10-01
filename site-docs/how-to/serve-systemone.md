# Serve a checkpoint as an Ollama-style decision API

Ollama 0.35 added decision models (`ollama pull nimble`) behind a local `POST /v1/systemone` endpoint, which
follows TypeSafe's Jev API: a `state`, up to 64 named `choice`, `noul` and `score` questions, and typed answers with
probabilities. `predict` already takes the same questions and returns the same answers, so `laya.serve` puts a Laya
Vision checkpoint behind the same endpoint. To move a client written for Ollama over, change the base URL and the
model name. Unlike Ollama's endpoint, this one also takes images.

The server is in [`laya/serve.py`](https://github.com/r33drichards/laya-vision/blob/main/laya/serve.py). It needs
only the standard library on top of the package.

## Start the server

```bash
pip install -e . torchvision
laya-serve --model thaitea/laya-vision --revision f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc
# serving thaitea/laya-vision (revision f2fe3c1…) as laya-vision, thaitea/laya-vision on http://127.0.0.1:11435/v1/systemone
```

`python -m laya.serve` does the same. It listens on port 11435, next to Ollama's 11434, so both can run on one
machine. Clients send `"model": "laya-vision"` or the checkpoint id. Any other name gets a 404, as an unpulled
model does in Ollama.

| Option | Meaning |
|---|---|
| `--model`, `--revision`, `--backbone-revision` | The checkpoint and the Hub commits to pin, as in [`load_vlm`](../reference/predict.md). |
| `--name` | A model name clients send. Repeat it to accept several names, for example `--name nimble` so an Ollama client works with no change. |
| `--host`, `--port` | Where to listen. The default host, `127.0.0.1`, accepts local connections only. |
| `--device` | A torch device; defaults to the best available. |
| `--n-permutations` | `predict`'s `n_permutations`: option orders averaged per question. |
| `--allow-truncation` | Cut inputs that exceed the token budgets and report it in each answer's `truncated`, instead of returning a 400. |
| `--max-body-bytes` | The request size limit, 16 MiB by default. Ollama's limit is 64 KiB; images need more. |

## Call it

This is the request from Ollama's announcement, with only the model name changed:

```bash
curl http://localhost:11435/v1/systemone -d '{
  "model": "laya-vision",
  "state": {"ticket": "I was charged twice. Please refund the extra payment."},
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
             "criteria": {"billing": "Payments and refunds", "technical": "Bugs and integrations",
                          "other": "None of the above"}},
    "refund": {"type": "noul", "instructions": "Does the customer explicitly ask for a refund?"},
    "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                "criteria": ["Routine", "Soon", "Urgent"]}
  }
}'
```

```json
{"model": "laya-vision",
 "answers": {"team": {"type": "choice", "choice": "billing",
                      "probabilities": {"billing": 0.6622, "technical": 0.1553, "other": 0.1824}, "confidence": 0.2058},
             "refund": {"type": "noul", "noul": 0.3116},
             "urgency": {"type": "score", "score": 1.2656, "legend": {"0": "Routine", "1": "Soon", "2": "Urgent"},
                         "probabilities": {"0": 0.2539, "1": 0.2266, "2": 0.5195}, "confidence": 0.0673}},
 "usage": {"input_tokens": 187, "output_tokens": 0}}
```

That response came from the checkpoint on a CPU. Laya Vision was trained on images; on text-only questions like this
one it is far less accurate than a 9B text decision model such as `nimble` (here it doubts the refund request).
Use it where the decision depends on a picture.

TypeSafe's Python SDK works unchanged:

```bash
pip install typesafe-sdk
export TYPESAFE_BASE_URL=http://localhost:11435 TYPESAFE_API_KEY=unused TYPESAFE_DEFAULT_MODEL=laya-vision
```

```python
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

with TypeSafeClient(timeout=120) as client:
    result = client.system_one(state={"ticket": "..."}, questions={"refund": Noul(instructions="...")})
print(result.nouls["refund"].noul)
```

## Send images

Images go in as base64 strings, plain or as `data:image/...;base64,` URLs, in any of three places:

- `state["image"]`, one image, as in `predict`'s state;
- `state["images"]`, a list of them;
- a top-level `images` list, the field Ollama's `/api/generate` uses. With a string or array state, the text is
  passed to the model as `{"text": <state>}` beside the images.

```python
import base64, json, urllib.request

photo = base64.b64encode(open("photo.jpg", "rb").read()).decode()
req = {"model": "laya-vision",
       "state": {"image": photo, "note": "customer says it arrived broken"},
       "questions": {"damage": {"type": "score", "instructions": "How much damage does the item show?",
                                "criteria": ["none", "cosmetic", "functional", "destroyed"]},
                     "outdoors": {"type": "noul", "instructions": "Was the photo taken outdoors?"}}}
with urllib.request.urlopen(urllib.request.Request("http://localhost:11435/v1/systemone",
                                                   json.dumps(req).encode())) as r:
    print(json.load(r)["answers"])
```

The server decodes image strings as base64 and never opens them as file paths, so a client cannot make it read a file
on the server's disk. A request takes at most 16 images.

## Differences from Ollama

- **The response** has the fields Ollama documents. It drops what `predict` adds: `confidence` on `noul` answers,
  `action` and `provenance`. `usage.output_tokens` is always 0, because nothing is generated, and
  `usage.input_tokens` counts every scored row, so the image and state count once per question.
- **Requests** must use the same fields and limits: 1 to 64 questions, and 2 to 26 options or levels. Two Laya
  extensions are accepted on top: a list of option names as a `choice` question's `criteria`, and images. An
  unknown field, such as `stream`, is a 400 rather than being ignored.
- **`keep_alive`** is accepted and ignored. The model stays loaded until the server stops.
- **Over-budget questions.** A question that does not fit the checkpoint's token budgets (see
  [Truncation](../reference/predict.md#truncation)) is a 400, as a prompt longer than the loaded context is in
  Ollama, unless the server runs with `--allow-truncation`.
- **Concurrency.** Requests are answered one at a time, because the model is not thread-safe.
- **Other endpoints.** `GET /api/tags` lists the served names and `GET /` answers `Laya is running`. There are no
  other Ollama endpoints (`/api/pull`, `/api/generate` and so on).

To run a checkpoint inside Ollama itself, see [Serve with Ollama](serve-with-ollama.md).

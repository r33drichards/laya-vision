# Fine-tune on your own dataset

The published checkpoint answers general questions about photos, charts and documents. If your images or questions
look different (a product catalogue, inspection photos, your own rubric), fine-tune it on a few hundred to a few
thousand labelled questions of your own. This guide writes the data, trains locally in Python or on a GPU on Modal,
and loads the result with `load_vlm`.

If the model already gets your answers right and only its probabilities are off, you do not need to fine-tune:
[Calibrate on your own data](calibrate.md) instead.

## 1. Write your data as JSONL

A dataset is a directory with one JSONL file per split and the images next to it:

```text
my_dataset/
  train.jsonl
  val.jsonl
  images/
    0001.jpg
    ...
```

Each line is one question about one image, written with the same question types as
[`predict`](../reference/predict.md#question-types), plus the index of the right answer:

```json
{"id": "0001-color", "image": "images/0001.jpg", "state_text": "listing photo, seller note: barely used",
 "question": {"type": "choice", "instructions": "What colour is the item?", "criteria": ["red", "blue", "green"]},
 "label": 1}
{"id": "0001-damaged", "image": "images/0001.jpg",
 "question": {"type": "noul", "instructions": "Is the item visibly damaged?"},
 "label": 0}
{"id": "0001-condition", "image": "images/0001.jpg",
 "question": {"type": "score", "instructions": "What condition is the item in?",
              "criteria": ["poor: broken or missing parts", "fair: heavy wear", "good: light wear", "like new"]},
 "label": 2}
```

| Field | Meaning |
|---|---|
| `image` | Path relative to the dataset directory. `images` (a list of paths) instead, for a question about several images. Leave both out for a text-only question. |
| `state_text` | Optional text the model reads next to the image, like the extra keys of a `predict` state. |
| `question` | `type`, `instructions` and `criteria`, exactly as you will pass them to `predict`. |
| `label` | The index of the right option: the position in `criteria` for `choice`, the level for `score`, `0` (false) or `1` (true) for `noul`. |
| `target` | Optional: a probability per option, in the same order, replacing the one-hot target from `label`. Use it when your labels are votes, e.g. 3 of 5 raters said "good" (`[0, 0.2, 0.6, 0.2]`). `label` is still what accuracy is scored against. |
| `id` | Optional, carried through to the evaluation records. |

Records with a `label` out of range, or a `choice` with duplicate option names, are skipped when loading.

Some things that make the fine-tuned model better:

- **Phrase questions the way you will ask them.** The model learns your instructions and option wording; ask the
  same at inference. Several phrasings of one question in training make it less brittle.
- **Put every image's questions in the same split.** A question about an image the model has trained on is not a
  fair validation question.
- **Keep a few hundred validation questions** per question you care about; accuracy on 30 questions moves by
  ±15 points by chance.

Check that the files load before you train:

```python
from laya.vlm_train import load_jsonl_examples

train_ex = load_jsonl_examples(".", "my_dataset", "train")   # <root>/<name>/<split>.jsonl
print(len(train_ex), train_ex[0]["q"], train_ex[0]["target"])
```

## 2a. Fine-tune locally

For a small dataset, or to try the data before paying for a GPU, train in Python. This starts from the published
checkpoint, pinned to a Hub commit, so you keep what it already knows:

```python
import laya
from laya.vlm_train import (collect_logits, fit_temperatures_from, format_metrics, load_jsonl_examples,
                            metrics_from, train)

agent = laya.load_vlm("thaitea/laya-vision", revision="8b318c99d7ad3ce19c24369263463882eada9d1e")

train_ex = load_jsonl_examples(".", "my_dataset", "train")
val_ex = load_jsonl_examples(".", "my_dataset", "val")
calib_ex, train_ex = train_ex[-300:], train_ex[:-300]          # held out of training, to fit temperatures

print("before:", format_metrics(metrics_from(collect_logits(agent.model, agent.processor, val_ex))))

train(agent.model, agent.processor, train_ex, steps=500, batch_size=8, freeze="head",
      lr_head=1e-4, lr_backbone=2e-5, warmup=25, device=str(agent.device), log_every=50)

agent.temperature = fit_temperatures_from(collect_logits(agent.model, agent.processor, calib_ex))
val = collect_logits(agent.model, agent.processor, val_ex)
print("after:", format_metrics(metrics_from(val, agent.temperature)))

agent.save("my-laya")
```

`freeze` picks what trains:

| `freeze` | Trains | Use it when |
|---|---|---|
| `"head"` | The answer head only; the backbone is frozen. | You have a few hundred questions, or no GPU. Fast, and cannot forget what the backbone knows. |
| `"last_n"` | The head and the last `n_last` language-model layers. | A few thousand questions on images the head alone does not separate. |
| `"full"` | Everything except the vision tower. | Many thousands of questions, on a GPU. This is how the published checkpoints were trained. |

- `steps × batch_size` is the number of questions seen; aim for 2 to 3 passes over your training set and watch the
  validation accuracy, not the loss. `max_minutes=` caps wall-clock time instead.
- Options are shuffled for every training question, so the model cannot learn that the answer is usually first.
- Fitting the temperatures on held-out questions is what keeps `confidence` meaningful after training. With fewer
  than about 300 training questions, hold out fewer (and expect noisier temperatures), or keep the checkpoint's by
  skipping that line.
- Save the whole model (`agent.save(path)`, the default): `include_backbone=False` writes only the head and
  reloads the backbone from the original pretrained weights, which is wrong for a checkpoint whose backbone was
  itself fine-tuned, as the published one's was.

On a CPU, each step at batch 4 takes a few seconds with `freeze="head"`; anything beyond the head wants a GPU.

## 2b. Fine-tune on Modal

For a larger dataset, `finetune_long` in [`modal_app.py`](https://github.com/r33drichards/laya-vision/blob/main/modal_app.py)
trains on an A100 with per-epoch evaluation, keeps the best checkpoint and fits the temperatures for you. The
volumes and setup it expects are in [Run jobs on Modal](run-on-modal.md).

Upload the dataset to the `laya-datasets` volume under `vqa/<name>`, with an empty `_READY` file that marks it
complete (jobs skip a dataset without one). Use a name that is not on the volume yet: prepared datasets are never
overwritten.

```bash
touch my_dataset/_READY
modal volume put laya-datasets my_dataset /vqa/my_dataset
```

To start from the published checkpoint rather than from the bare backbone, put it on the `laya-checkpoints` volume
once, then pass it to `--init-from`:

```bash
hf download thaitea/laya-vision --revision 8b318c99d7ad3ce19c24369263463882eada9d1e --local-dir laya-vision-8b318c9
modal volume put laya-checkpoints laya-vision-8b318c9 /smolvlm/hub/laya-vision-8b318c9

modal run --detach modal_app.py::finetune_long --init-from hub/laya-vision-8b318c9 \
    --datasets my_dataset --val-datasets my_dataset,vqa --run-name my-dataset-3ep --epochs 3 --n-calib 300
```

- `--run-name` must be new: the run saves to `/ckpt/smolvlm/<run-name>/`, with `best/`, `last/` and a
  `metrics.json` of every evaluation.
- `--n-calib` questions from the end of each training file are held out to fit the temperatures; keep the
  training file well above that, or lower it.
- `--val-datasets my_dataset,vqa` also scores A-OKVQA, ScienceQA and VQAv2 yes/no at every evaluation (when they
  are prepared on your volume), so you see whether the model is forgetting its general skills. The best checkpoint is chosen by the mean accuracy over all
  validation sets.
- To keep those skills, train on your data mixed with The Cauldron (prepared with `prepare_cauldron`):
  `--datasets my_dataset,cauldron` with `--mix my_dataset=5` to draw your set five times as often as each Cauldron subset, and `--max-passes 4` to stop
  a small set from repeating too often.
- Without `--init-from`, the run starts from the pretrained backbone (`--backbone`) with an untrained head; that
  needs far more data than continuing from the checkpoint.

Follow the run with `modal app logs laya-smolvlm`, then download the result:

```bash
modal volume get laya-checkpoints /smolvlm/my-dataset-3ep/best my-laya
```

To score it on your validation set and the VQA sets, run `modal run modal_app.py::evaluate --run-name my-dataset-3ep/best --datasets
my_dataset,vqa`; the full suite is in [Evaluate a checkpoint](evaluate.md).

## 3. Use the fine-tuned model

A saved directory loads like a Hub checkpoint:

```python
import laya
from PIL import Image

agent = laya.load_vlm("my-laya")
result = agent.predict(
    {"image": Image.open("photo.jpg"), "note": "listing photo, seller note: barely used"},
    {"condition": {"type": "score", "instructions": "What condition is the item in?",
                   "criteria": ["poor: broken or missing parts", "fair: heavy wear", "good: light wear", "like new"]}},
)
```

The checkpoint records the Hub commit it started from (`loaded_from` in `vlm_agent_config.json`). To share it,
upload the directory to the Hub (`hf upload user/name my-laya`), and load it pinned to the commit that upload made,
as for the published checkpoint.

## See also

- [Training data](../concepts/data.md): the datasets the published checkpoints were trained on, in the same format.
- [How it works](../concepts/how-it-works.md): the answer head and the training objective.
- [Calibration](../concepts/calibration.md): what the fitted temperatures change.
- [`laya/vlm_train.py`](https://github.com/r33drichards/laya-vision/blob/main/laya/vlm_train.py): `train`,
  `load_jsonl_examples` and every training option.

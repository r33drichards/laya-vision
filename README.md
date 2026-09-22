# Laya Vision

Image inputs for [Laya](https://github.com/NandhaKishorM/laya): typed, calibrated decisions (`choice`, `score`, `noul`) over an **image plus optional text**, in one forward pass with no text generation.

Laya Vision swaps Laya's ModernBERT text encoder for [SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct), which already understands images. It keeps Laya's `predict(state, questions)` API, output schema, proper-scoring-rule training and temperature calibration.

- **Models:** [thaitea/laya-vision-smolvlm-256m](https://huggingface.co/thaitea/laya-vision-smolvlm-256m) (the results below) and [thaitea/laya-vision-modernvbert-250m](https://huggingface.co/thaitea/laya-vision-modernvbert-250m) (bidirectional backbone, trained on The Cauldron; see the ModernVBERT section)
- **Try it in the browser:** [thaitea/laya-vision-demo](https://huggingface.co/spaces/thaitea/laya-vision-demo), a Hugging Face Space on free CPU at about 3 s per image. Its source is in `space/`.
- **Status:** experimental research fork. It is not affiliated with Convai Innovations, the authors of Laya.

## Results

This is the fine-tuned checkpoint `all3-3ep/best`: 3 passes over 72k training examples, about 33 minutes on one A100. Scores are on the full validation splits.

| Dataset | Type | Chance | Accuracy | ECE (raw → calibrated) |
|---|---|---|---|---|
| A-OKVQA | 4-way `choice` | 25% | 61.8% | 0.295 → 0.123 |
| ScienceQA (image subset) | 2–5-way `choice` | ~36% | 86.6% | 0.090 → 0.034 |
| VQAv2 yes/no (re-split of official val) | `noul` | 50% | 73.4% | 0.102 → 0.041 |
| **All** | | | **75.2%** | 0.124 → **0.034** |

- **Latency:** about 71 ms for one image question on an NVIDIA L4 (bf16). The image is encoded once and reused for every question in the call.
- **Option-order sensitivity:** across 4 rotations of the A-OKVQA option order, accuracy varies by 0.7 points.
- **`score` questions are not trained yet** in this checkpoint: there was no ordinal image data, so treat its `score` outputs as meaningless. Four rubric-scored sets (response grading, aesthetics votes, generated-image ratings, damage severity) are now prepared for it; see [docs/score-data.md](docs/score-data.md).

## Usage

```python
import laya
from PIL import Image

agent = laya.load_vlm("thaitea/laya-vision-smolvlm-256m")   # downloads from the Hub
result = agent.predict(
    {"image": Image.open("photo.jpg"), "note": "customer says it arrived broken"},
    {
        "damaged":  {"type": "noul",   "instructions": "Does the item in the photo look damaged?"},
        "category": {"type": "choice", "instructions": "What kind of item is this?",
                     "criteria": ["electronics", "clothing", "furniture", "food", "other"]},
    },
)
print(result["answers"]["damaged"]["noul"], result["answers"]["category"]["choice"])
```

Install with `pip install -e .`, plus `torchvision`, which the SmolVLM image processor needs.

### Run on Modal

`modal_app.py` expects the Modal volumes `laya-hf-cache`, `laya-datasets` and `laya-checkpoints`.

```bash
modal run modal_app.py::try_model --image photo.jpg [--questions q.json] [--text "..."]   # ask a checkpoint about an image
modal run modal_app.py::test                                                          # GPU tests + latency
modal run --detach modal_app.py::finetune_long                                        # ~3-epoch A100 fine-tune
modal run modal_app.py::evaluate --run-name all3-3ep/best                             # re-score a checkpoint
```

The training data is written to `laya-datasets:/data/vqa/<name>/{train,val}.jsonl` by the data-prep job on the [`siglip-projector-experiment`](https://github.com/r33drichards/laya-vision/tree/siglip-projector-experiment) branch.

## Playing games

Laya Vision can also act as a game policy: the screen is the image and the options are the game's buttons. `examples/atari_live.py` and `examples/vizdoom_live.py` let you watch it play in a local window.

The released checkpoint doesn't know any games; in ViZDoom `basic` it only ever shoots. Trained for 7 minutes on 20,000 frames auto-labelled by a scripted expert, it plays `basic` at expert level: mean reward +75.4 and 100% kills over 50 unseen episodes, against the expert's +75.8. See [docs/game-training.md](docs/game-training.md) for the pipeline, results and training-data ideas.

## ModernVBERT experiment

Laya's text model reads each option at a bidirectional `[MASK]` marker. SmolVLM is a causal decoder, so `laya/vlm.py` has to put the options last, read each one at its line terminator, and work around the option-order bias that leaves (random orders in training, permutation averaging at inference). [ModernVBERT](https://huggingface.co/ModernVBERT/modernvbert) (ModernBERT-150M plus a SigLIP2 vision tower, 250M parameters, MIT licence) is a bidirectional encoder pretrained with masked language modelling behind the same Idefics3 image processor SmolVLM uses, so the text model's readout carries over unchanged:

```
[CLS]User:<image tokens> choice question: What kind of item is this?[SEP][MASK] electronics[MASK] clothing ...[SEP]{"note": "..."}[SEP]
```

Architecture diagrams of every branch (the text model, causal SmolVLM, SmolVLM with bidirectional options, ModernVBERT) and of the shared head are in [docs/architecture.md](docs/architecture.md).

Both backbones run through the same `VLMAgent`, training loop, evaluation and checkpoint format. The checkpoint records which it is (`"readout": "mask"` or `"terminator"` in `vlm_agent_config.json`), and `laya.load_vlm` picks the right sequence builder from it.

```python
agent = laya.load_vlm(backbone="ModernVBERT/modernvbert")   # fresh, untrained head
```

```bash
python -m laya.vlm_train --synthetic --steps 3 --backbone ModernVBERT/modernvbert           # local smoke run
modal run modal_app.py::test                                                                # tests + latency, both backbones
modal run --detach modal_app.py::finetune_long --backbone ModernVBERT/modernvbert --run-name mvb-3ep   # same recipe as all3-3ep
modal run modal_app.py::evaluate --run-name modernvbert/mvb-3ep/best
```

`finetune_long` keeps everything else identical to the SmolVLM run above (data, objective, schedule, evaluation, calibration), which is what makes the two comparable. The ModernVBERT run saves under `/ckpt/modernvbert/`, so it can reuse a run name.

## Post-training on The Cauldron

[The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) is the 50-subset instruction mixture both SmolVLM and ModernVBERT were aligned on. Laya answers typed questions, so `laya/cauldron.py` keeps only the turns whose answer is a closed choice and maps them onto Laya's types:

| Cauldron format | Subsets | Laya type |
|---|---|---|
| `Question: ... Choices: A. ... B. ...` → `Answer: B` | ai2d, iconqa, intergps, scienceqa, tqa, visual7w | `choice` (a ScienceQA lecture becomes the state's context) |
| `Options: a, b, c, d.` → `Cab.` | aokvqa | `choice` |
| `... Answer yes or no.` → `Yes.` | figureqa, hateful_memes, nlvr2 (two images), vsr, vqarad, plus the yes/no share of clevr, dvqa, mapqa, ocrvqa, vqav2, chartqa | `noul` |
| `Which figure should complete the logical sequence?` → `B` | raven | `choice` over the letters A–H |

Numbers, free-form answers, captions and code are skipped. The prep job streams each subset, keeps up to 10,000 usable rows with at most 4 questions each, holds out 5% of rows as `val`, and saves the images as JPEG at 1024 pixels on the long side:

```bash
modal run modal_app.py::prepare_cauldron                    # all 19 subsets, one container each -> /data/vqa/cauldron_<subset>
modal run modal_app.py::prepare_cauldron --subsets ai2d,vsr --max-rows 2000
```

`finetune` and `finetune_long` default to these sets. The sampler draws subsets equally, so cap the passes over the small ones, and score the official A-OKVQA / ScienceQA / VQAv2 val splits alongside the Cauldron holdouts to stay comparable with the table above:

```bash
modal run --detach modal_app.py::finetune_long --run-name cauldron-3ep --max-passes 4 \
    --val-datasets aokvqa,scienceqa,vqav2_yesno,cauldron_ai2d,cauldron_aokvqa,cauldron_nlvr2,cauldron_vsr
modal run --detach modal_app.py::finetune_long --run-name cauldron-3ep --max-passes 4 --backbone ModernVBERT/modernvbert \
    --val-datasets aokvqa,scienceqa,vqav2_yesno,cauldron_ai2d,cauldron_aokvqa,cauldron_nlvr2,cauldron_vsr
modal run modal_app.py::evaluate --run-name cauldron-3ep/best    # every prepared val set
```

The Cauldron is train-only upstream, so its `aokvqa`, `scienceqa` and `vqav2` rows are the official train splits and do not overlap those val splits. The original three sets remain available with `--datasets aokvqa,scienceqa,vqav2_yesno`.

### Result: ModernVBERT on The Cauldron

Run `modernvbert/cauldron-2ep`: 2 epochs over the 19 Cauldron subsets, 4 passes at most over any one of them, 68 minutes on one A100. Full per-subset numbers and the training log are in [docs/modernvbert-cauldron.md](docs/modernvbert-cauldron.md).

| Official val set | ModernVBERT + Cauldron | SmolVLM release (3 epochs on these sets) |
|---|---|---|
| A-OKVQA | **65.2%**, ECE 0.064 | 61.8%, ECE 0.123 |
| ScienceQA | 79.0%, ECE 0.058 | **86.6%**, ECE 0.034 |
| VQAv2 yes/no | 71.8%, ECE 0.037 | **73.4%**, ECE 0.041 |
| All 22 val sets (22,886 questions) | 71.8%, ECE 0.022 | |

A second run, 3 epochs with the cross-entropy weight annealed to zero (`cauldron-3ep-anneal`), tied it on accuracy (72.1% over all sets) and calibrated ECE (0.030) while making the raw model more overconfident, so the constant-weight recipe stays the default; details in the same doc.

ModernVBERT was pretrained for document retrieval and its paper reports no VQA numbers; on Laya's typed questions it trains as readily as SmolVLM, beats the released model on A-OKVQA, and trails it on ScienceQA, where the SmolVLM run made 12 passes over the train split against 4 here. It is also the faster of the two at inference (32 ms vs 41 ms per image question in bf16 on an L4). Published as [thaitea/laya-vision-modernvbert-250m](https://huggingface.co/thaitea/laya-vision-modernvbert-250m) (`laya.load_vlm("thaitea/laya-vision-modernvbert-250m")`), model card in `hf_model_card_modernvbert.md`. Needs `transformers >= 5.3` (the Modal jobs and the Space pin 5.17).

## What didn't work

The branch [`siglip-projector-experiment`](https://github.com/r33drichards/laya-vision/tree/siglip-projector-experiment) tried to keep Laya's ModernBERT encoder and feed it SigLIP2 image patches through a learned projector. It kept text-only answers bit-identical, but in 5 training runs it never learned to use the image. Every run collapsed to uniform predictions, and accuracy with shuffled images matched accuracy with the real ones. Details are in that branch's `laya/vision_train.py` and commit history.

## License

- **Code:** Apache 2.0, inherited from Laya. See `LICENSE`.
- **Weights:** the published weights are trained partly on ScienceQA, which is CC BY-NC-SA 4.0, so the model card marks them as non-commercial.

---

## Credits

Laya Vision is an independent fork of [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations, Apache 2.0, and is not affiliated with them. The text decision model, its typed-question API (`choice` / `score` / `noul`), the RLCD training objective and the temperature calibration are theirs; this fork adds image input, the SmolVLM and ModernVBERT backbones and the game work. The design is described in the author's [write-up](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me). The original text model and its Colab/Kaggle notebooks live in the upstream repo.

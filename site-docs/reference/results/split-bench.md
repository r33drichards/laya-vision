# Image splitting on SmolVLM2: accuracy against latency

SmolVLM2's processor can split an image into 512 px tiles plus one downscaled global view, instead of shrinking the whole image to a single 512 px tile. This benchmark trains the same Laya decision model three times on SmolVLM2-256M with splitting off, at 1024 and at 2048, and measures what each setting gains in accuracy and costs in latency.

**Result:** splitting at 1024 gained about 1 point of accuracy (+1.1 ± 0.9 over all questions, +1.4 on the mean of six sets) for 35% more latency. Splitting at 2048 gained nothing over no splitting (−0.3 ± 1.0) and was 2.5× slower. Keep splitting off by default. Turn on 1024 only where a point of accuracy is worth about 27 ms per call on an L4.

## Setup

`modal run --detach modal_app.py::split_bench` runs one `finetune_long` per setting. The runs are identical except for `split_edge`:

- **Model:** SmolVLM2-256M-Video-Instruct with a fresh decision head and bidirectional option attention. The vision tower is frozen; the language model and head are trained.
- **Data:** 4,000 training rows from each of six Cauldron subsets (24,000 in total). Only closed-form turns are kept: AI2D, A-OKVQA and TQA give `choice` questions, and OCR-VQA, MapQA and VQAv2 give only their yes/no questions (`noul`). Validation is a seeded 5% of each subset's rows (4,711 questions); calibration uses 300 held-out training records per set.
- **Schedule:** 2 epochs, 1,500 steps at batch 32, one A100 80GB per run. The best checkpoint by mean validation accuracy is kept, then per-type temperatures are fitted. Every run's best was its final step, and every run finished all 1,500 steps.
- **Latency:** `bench_latency` times `predict` with one question on 300 validation images, bf16 on an NVIDIA L4. The clock includes the CPU preprocessing (resize and tiling) and the forward pass.

| Setting | Longest edge before tiling | Views per image (max / mean) | Sequence cap |
|---|---|---|---|
| nosplit | none, one 512 px view | 1 / 1.0 | 1024 |
| split1024 | 1024 px | 5 / 4.6 | 1536 |
| split2048 | 2048 px | 17 / 13.6 | 2304 |

## Results

| Setting | Mean acc (6 sets) | All questions | ECE (calibrated) | Input tokens | L4 median (p90) | vs nosplit | Train steps/s |
|---|---|---|---|---|---|---|---|
| nosplit | 68.7% | 69.2% | 0.020 | 111 | 77.0 ms (81.6) | – | 4.20 |
| split1024 | **70.2%** | **70.3%** | **0.019** | 368 | 103.6 ms (107.1) | **+35%** | 1.71 |
| split2048 | 68.4% | 68.9% | 0.028 | 950 | 196.4 ms (240.3) | **+155%** | 0.56 |

### Accuracy per set

Calibrated validation accuracy. The Δ columns are against nosplit, with ± one standard error of the difference (unpaired, so slightly conservative).

| Set | Type | n | nosplit | split1024 | split2048 | Δ 1024 | Δ 2048 |
|---|---|---|---|---|---|---|---|
| AI2D (science diagrams) | choice | 282 | 72.3% | 74.5% | 74.1% | +2.1 (±3.7) | +1.8 (±3.7) |
| A-OKVQA (photos) | choice | 552 | 65.0% | 67.6% | 67.8% | +2.5 (±2.8) | +2.7 (±2.8) |
| TQA (textbook figures) | choice | 264 | 58.3% | 59.8% | 52.7% | +1.5 (±4.3) | −5.7 (±4.3) |
| OCR-VQA (book covers) | yes/no | 996 | 84.1% | 84.0% | 82.9% | −0.1 (±1.6) | −1.2 (±1.7) |
| MapQA (maps) | yes/no | 1,598 | 61.0% | 61.6% | 59.1% | +0.7 (±1.7) | −1.8 (±1.7) |
| VQAv2 (photos) | yes/no | 1,019 | 71.6% | 73.4% | 73.9% | +1.8 (±2.0) | +2.3 (±2.0) |
| **All** | | 4,711 | 69.2% | 70.3% | 68.9% | **+1.1 (±0.9)** | **−0.3 (±1.0)** |

Calibrated ECE per set is in the metrics files. Pooled, it is 0.020, 0.019 and 0.028.

### Which sets benefit

I expected the text- and detail-heavy sets, OCR-VQA and MapQA, to gain most. They did not: OCR-VQA was flat at both settings and MapQA gained 0.7 at 1024 and lost 1.8 at 2048. Two things in the data explain most of this:

- **Only their yes/no questions are in the benchmark.** OCR-VQA's are about genre ("Is this book related to Law?", "Is this a sci-fi book?"), which the cover's layout and imagery can answer without reading small print. MapQA's ask whether a state has the highest or lowest value in a region of a colour-shaded US map ("Does Nevada have the lowest value in the West?"). That turns on comparing colour shades and knowing where each state is, not on reading text.
- **Every image was stored at 1024 px at most** (`prepare_cauldron_subset(max_side=1024)`). At 2048 the processor upscales each image 2× before tiling, so the extra views carry no new detail, only more tokens to attend over in the same 1,500 steps.

The consistent gains were on photos and diagrams: A-OKVQA +2.5 / +2.7, VQAv2 +1.8 / +2.3 and AI2D +2.1 / +1.8. Each is about 1 standard error on its own, but they agree in sign across both settings. Where they help, the extra views seem to help as more pixels per object rather than as readable text. TQA's −5.7 at 2048 is the largest single change, but TQA has only 264 questions and it is 1.3 standard errors.

### Latency cost

- **split1024:** +26.6 ms median (+35%) and +25.5 ms at p90 on an L4. The processor makes 4.6 views per image on average, and the language model reads 368 tokens instead of 111.
- **split2048:** +119.4 ms median (+155%) and +158.7 ms at p90 (+195%). The processor makes 13.6 views per image and the model reads 950 tokens. The p90 spreads further because the tile count depends on aspect ratio.
- **Training:** 2.5× (1024) and 7.5× (2048) slower per step than nosplit. Data wait stayed at 1–3% for every run, so the GPU was the bottleneck, not the tiling on the CPU.

## Recommendation

- **Default: no splitting.** It matches the released checkpoints, is the fastest, and splitting's gain is small.
- **split1024 where accuracy on photos or diagrams matters more than about 27 ms per call.** The gain is about 1 point overall and 2–2.5 on photos and diagrams. It is only suggestive at this sample size and with one run per setting.
- **Not split2048 on this data.** It cost 2.5× the latency and 7.5× the training time for no gain.
- **To test splitting properly:** prepare the data at full resolution (`max_side` 2048 or no cap), include sets whose questions need fine detail in a closed form (for example chart or document questions turned into `choice`), and train two or three runs per setting so that run-to-run variation can be measured.

## Caveats

- **One training run per setting.** One rough indication of run-to-run noise: a preemption made Modal rerun the whole benchmark, and the rerun's nosplit scored 68.87% mean accuracy against 68.74% the first time. Differences of a point or two between single runs should be read with that in mind.
- **Validation sets are drawn from the Cauldron's training rows** (the Cauldron has no other split). The numbers are not comparable to published results on the official benchmarks.
- **GPUs:** all training ran on A100 80GB, but Modal mixed SXM4 and PCIe cards across runs, so the steps/s column is only roughly comparable. Accuracy and the L4 latency numbers are unaffected.
- **Data loader:** the split runs used fewer loader workers so their batches fit in shared memory (`_loader_fit`). That only changes speed.
- **How the table was assembled:** `split_bench` is supposed to write `results.md` itself. Here the scheduler container was preempted mid-run, so the table above comes from each run's `metrics.json` (copied here as `smolvlm2-split-bench-{nosplit,split1024,split2048}-metrics.json`) and from `bench_latency` run on each best checkpoint separately. `finetune_long` now saves resumable state (`state.pt`) so a preemption no longer restarts a run from step 0.

## Checkpoint

The split1024 run is published as [thaitea/laya-vision-smolvlm2-256m-split1024](https://huggingface.co/thaitea/laya-vision-smolvlm2-256m-split1024).

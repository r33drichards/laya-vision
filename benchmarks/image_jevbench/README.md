# Image JevBench, reproducible subset

[Image JevBench](https://benchmarkheaven.com/image-jev-bench) scores vision-language systems on typed multiple-choice
decisions about an image. Its official score cannot be reproduced here: the 228-item public split and the 456-item
sealed split are not released, and the page publishes aggregates only. The benchmark's maintainers run the sealed
split themselves for systems submitted through its "request evaluation" page.

This directory scores laya-vision on what the benchmark's public repository
([`fstandhartinger/model-market-comparison@33b337e`](https://github.com/fstandhartinger/model-market-comparison/tree/33b337e3bb592abdaf1b39143e8f7dcd8030bf67))
does publish:

- **80 rebuilt image-reasoning items** from its multimodal preview (`items-extended-real.json`): CLEVR-HOPE,
  Geometry3K, ArxivQA and FinQA, 20 each. The preview ships questions, options and gold answers but no images, so
  `build.py` fetches each upstream row at a pinned Hub commit and checks its question and answer against the item.
  FinQA is a table, rendered the way the benchmark's own published FinQA example is (identical size, same layout).
  The preview's 48 ScreenSpot and Mind2Web items are left out: their click markers were drawn at unpublished positions.
- **8 public examples**, with the benchmark's own images.

These are not the v0.1 public split (which is mostly everyday photos and Mind2Web), so the numbers are not comparable
to the leaderboard's. The same 80 items do carry published per-item results for six other systems, so they can be
compared with those.

## Run

CPU is enough (about 1 s per item). The images and question text are third-party records: they go into the work
directory, not the repository.

```bash
python benchmarks/image_jevbench/build.py --work /tmp/image-jevbench          # needs node for the examples module
python benchmarks/image_jevbench/run.py --work /tmp/image-jevbench --out eval-results/image-jevbench-<name>
cd benchmarks/image_jevbench && python score.py ../../eval-results/image-jevbench-<name>
```

`run.py` defaults to `thaitea/laya-vision` at `8b318c9` and refuses to overwrite an existing result.

## Result: `thaitea/laya-vision@8b318c9`, CPU fp32

Rows: [`eval-results/image-jevbench-laya-vision-8b318c9.predictions.jsonl.gz`](https://github.com/r33drichards/laya-vision/blob/main/eval-results/image-jevbench-laya-vision-8b318c9.predictions.jsonl.gz),
meta and provenance: [`.meta.json`](https://github.com/r33drichards/laya-vision/blob/main/eval-results/image-jevbench-laya-vision-8b318c9.meta.json).
Two builds on 2026-09-25 produced identical image hashes and identical predictions.

| Set | n | Accuracy | Chance | Chance-corrected | ECE (10 bins) |
|---|---:|---:|---:|---:|---:|
| CLEVR-HOPE | 20 | 65.0% | 50.0% | +30.0% | 0.028 |
| Geometry3K | 20 | 0.0% | 25.0% | -33.3% | 0.585 |
| ArxivQA | 20 | 55.0% | 25.0% | +40.0% | 0.238 |
| FinQA | 20 | 15.0% | 27.5% | -20.0% | 0.461 |
| **Rebuilt, all** | 80 | 33.8% | 31.9% | +4.2% | 0.256 |
| Public examples | 8 | 37.5% | 28.7% | +19.8% | 0.323 |

Same 80 items, the benchmark's published per-item results (its `RUN-RESULT.md`, 22 Sep 2026):

| System | CLEVR-HOPE | Geometry3K | ArxivQA | FinQA | All 80 |
|---|---:|---:|---:|---:|---:|
| **thaitea/laya-vision** (201M) | 65% | 0% | 55% | 15% | 33.8% |
| GPT-5.6 Luna | 80% | 100% | 65% | 65% | 77.5% |
| Gemini 3.1 Flash-Lite | 75% | 70% | 55% | 60% | 65.0% |
| AlexWortega/openjev 4B v2 | 75% | 95% | 45% | 35% | 62.5% |
| Mapika/decider-2b-vision | 90% | 40% | 55% | 40% | 56.2% |
| kshetrajna12/reflex 4B | 75% | 35% | 70% | 50% | 57.5% |

Reading it:

- Overall the checkpoint is at chance (33.8% against 31.9%). With 20 items per set, one item is 5 points and a 95%
  interval on a set is about ±20 points.
- **CLEVR-HOPE** (yes/no spatial questions) and **ArxivQA** (figure questions) are above chance, and CLEVR-HOPE is
  well calibrated (ECE 0.028).
- **Geometry3K and FinQA need arithmetic.** The wrong options are near-miss numbers (48 against 52.8, 7.85 against 9.85),
  and a one-pass classifier with no generation cannot compute them: 0/20 and 3/20, confidently wrong (ECE 0.59 and
  0.46). Two Geometry3K items also put the answer next to "All listed values" / "None of the listed values"; the
  model picked those.
- Of the public examples, it gets the two everyday photos and one of four click-marker screenshots.
- Item `mm-065` (FinQA) has an empty string as its correct option, a defect in the benchmark's item.

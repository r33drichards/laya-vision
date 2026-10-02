# Install

Laya Vision is a Python package (`laya`) built on PyTorch and Hugging Face `transformers`. It installs from the
repository; checkpoints download from the Hugging Face Hub the first time you load them.

## From the repository

```bash
git clone https://github.com/r33drichards/laya-vision
cd laya-vision
pip install -e .
```

- Python 3.9 or newer, `transformers >= 4.56`. `torchvision` (the SmolVLM image processor) and `pillow` are
  declared dependencies.
- ModernVBERT checkpoints need `transformers >= 5.3` (`pip install -e ".[modernvbert]"`); SmolVLM2 backbones need
  `num2words` (`".[smolvlm2]"`).

Then load the recommended checkpoint:

```python
import laya

agent = laya.load_vlm("thaitea/laya-vision")   # downloads the weights from the Hub on first use
```

`load_vlm` takes `revision=` (and `backbone_revision=`) to pin the Hub commit, which is what you want for anything
you publish or compare later; see [predict()](../reference/predict.md#loading-a-checkpoint).

## No install: the demo

The published checkpoint runs in a Hugging Face Space, with nothing to install:
[Try the demo](../tutorials/space-demo.md).

## Tests

From the repository root:

```bash
python -m pytest -q
```

`tests/test_vlm.py` downloads SmolVLM-256M, about 2 minutes on a CPU.

## GPU jobs

Training, evaluation and the benchmarks run on [Modal](https://modal.com); see
[Run jobs on Modal](../how-to/run-on-modal.md).

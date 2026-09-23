# Install

Laya Vision is a Python package (`laya`) built on PyTorch and Hugging Face `transformers`. It installs from the
repository; checkpoints download from the Hugging Face Hub the first time you load them.

## From the repository

```bash
git clone https://github.com/r33drichards/laya-vision
cd laya-vision
pip install -e . torchvision
```

- `torchvision` is needed by the SmolVLM image processor, and is not a declared dependency of the package.
- ModernVBERT checkpoints need `transformers >= 5.3`.

Then load the recommended checkpoint:

```python
import laya

agent = laya.load_vlm("thaitea/laya-vision")   # downloads the weights from the Hub on first use
```

`load_vlm` takes `revision=` (and `backbone_revision=`) to pin the Hub commit, which is what you want for anything
you publish or compare later; see [predict()](../reference/predict.md#loading-a-checkpoint).

## No install: the browser

The same checkpoint runs in a web page with ONNX Runtime Web, with nothing to install:
[Run it in your browser](../tutorials/browser-demo.md).

## Tests

From the repository root:

```bash
python -m pytest -q
```

`tests/test_vlm.py` downloads SmolVLM-256M, about 2 minutes on a CPU.

## GPU jobs

Training, evaluation and the benchmarks run on [Modal](https://modal.com); see
[Run jobs on Modal](../how-to/run-on-modal.md).

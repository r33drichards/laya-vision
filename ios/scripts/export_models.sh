#!/usr/bin/env bash
# Export the checkpoint to ONNX for the iOS app, into ios/Models (git-ignored), and check it against PyTorch.
#
#   ios/scripts/export_models.sh                 # fp32 + fp16 + q8, about 1.5 GB
#   VARIANTS=q8 ios/scripts/export_models.sh     # keep fp32 + q8 only (fp32 is always written)
#
# Makes a virtualenv in ios/.venv with the Space's pinned libraries plus the ONNX tools. Takes about ten minutes on
# an Apple silicon Mac, most of it the download and the fp16/q8 conversions.
set -euo pipefail
cd "$(dirname "$0")/../.."

CHECKPOINT=${CHECKPOINT:-thaitea/laya-vision}
REVISION=${REVISION:-8b318c99d7ad3ce19c24369263463882eada9d1e}  # the 201M checkpoint the demo Space runs
VARIANTS=${VARIANTS:-fp16,q8}
PY=${PYTHON:-python3}

if [ ! -x ios/.venv/bin/python ]; then
  "$PY" -m venv ios/.venv
  ios/.venv/bin/pip install -q --upgrade pip
  ios/.venv/bin/pip install -q torch==2.14.0 torchvision==0.29.0 transformers==5.17.0 safetensors huggingface_hub \
    numpy pillow num2words onnx==1.23.0 onnxruntime==1.30.0 onnxscript==0.7.2
  ios/.venv/bin/pip install -q --no-deps -e .
fi

ios/.venv/bin/python scripts/export_onnx.py "$CHECKPOINT" --revision "$REVISION" --out ios/Models \
  --quantize "$VARIANTS" --validate
echo "Wrote ios/Models. Now: cd ios && xcodegen && open LayaVision.xcodeproj"

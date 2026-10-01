"""Turn a ``finetune_ollama`` run into files Ollama loads: the GGUF text model, its mmproj, and a Modelfile.

    python scripts/export_ollama.py --run laya-vision-ollama-6k --llama-cpp ~/llama.cpp --out dist/
    ollama create laya-vision -f dist/Modelfile

``--run`` downloads ``/ckpt/smolvlm-ollama/<run>/hf`` from the ``laya-checkpoints`` volume (needs the Modal CLI);
``--hf-dir`` takes a directory already on disk instead. ``--llama-cpp`` is a llama.cpp checkout at the tag the target
Ollama pins (``LLAMA_CPP_VERSION`` in ollama/ollama, b11232 when this was written), whose ``convert_hf_to_gguf.py``
writes the files. The text model is written as Q8_0 by default and the projector as F16.
"""
import argparse
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from laya.ollama import modelfile  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="a finetune_ollama run name on the laya-checkpoints volume")
    src.add_argument("--hf-dir", help="a saved run directory (the run's hf/) on disk")
    p.add_argument("--llama-cpp", required=True, help="llama.cpp checkout at the tag Ollama pins")
    p.add_argument("--out", default="dist/ollama")
    p.add_argument("--name", default="laya-vision", help="file name stem")
    p.add_argument("--outtype", default="q8_0", help="text model type for convert_hf_to_gguf.py (f32, f16, q8_0)")
    p.add_argument("--mmproj-outtype", default="f16")
    p.add_argument("--num-ctx", type=int, default=4096)
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    hf_dir = args.hf_dir
    if args.run:
        hf_dir = os.path.join(args.out, "hf")
        if os.path.exists(hf_dir):
            shutil.rmtree(hf_dir)
        subprocess.run(["modal", "volume", "get", "laya-checkpoints", "smolvlm-ollama/%s/hf" % args.run, args.out],
                       check=True)
    convert = os.path.join(args.llama_cpp, "convert_hf_to_gguf.py")
    model = os.path.join(args.out, "%s-%s.gguf" % (args.name, args.outtype))
    mmproj = os.path.join(args.out, "%s-mmproj-%s.gguf" % (args.name, args.mmproj_outtype))
    subprocess.run([sys.executable, convert, hf_dir, "--outfile", model, "--outtype", args.outtype], check=True)
    subprocess.run([sys.executable, convert, hf_dir, "--mmproj", "--outfile", mmproj, "--outtype", args.mmproj_outtype],
                   check=True)
    with open(os.path.join(args.out, "Modelfile"), "w") as f:
        f.write(modelfile("./" + os.path.basename(model), "./" + os.path.basename(mmproj), args.num_ctx))
    print("wrote %s, %s and %s" % (model, mmproj, os.path.join(args.out, "Modelfile")))


if __name__ == "__main__":
    main()

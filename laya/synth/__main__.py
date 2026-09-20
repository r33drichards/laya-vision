"""python -m laya.synth --root DIR --sources websight,screenqa --n-train 20000 --n-val 1000"""
import argparse
import json

from . import SOURCES, prepare


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="prepared-dataset root; each source becomes <root>/synth_<source>/")
    ap.add_argument("--sources", default=",".join(SOURCES), help="comma-separated: " + ", ".join(SOURCES))
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1024, help="saved images are resized to this longest side")
    ap.add_argument("--val-frac", type=float, default=0.05, help="val share for sources without an official val split")
    a = ap.parse_args()
    for name in [s for s in a.sources.split(",") if s]:
        meta = prepare(a.root, name, a.n_train, a.n_val, a.seed, a.max_side, a.val_frac)
        print(json.dumps({k: meta[k] for k in ("source", "train", "val", "dropped", "seconds")}, indent=1))


if __name__ == "__main__":
    main()

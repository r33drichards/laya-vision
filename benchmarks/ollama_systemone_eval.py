"""Score validation rows end to end through an Ollama ``/v1/systemone`` endpoint (with the image patch).

    python benchmarks/ollama_systemone_eval.py --url http://127.0.0.1:11434 --model laya-vision \
        --data-root /path/to/vqa --datasets vqav2_yesno,cauldron_aokvqa,score_ava --n 100 --out ollama-eval.jsonl

``--data-root`` holds the prepared sets as on the ``laya-datasets`` volume (``<name>/val.jsonl`` and its images).
Each row is sent as the request ``finetune_ollama`` validates on (one question, options in the dataset's order),
and the answer's probabilities are scored like ``laya.vlm_train.metrics_from``. ``--hf-dir`` also scores the same
requests with the exported model in PyTorch and reports the largest probability difference, which checks the GGUF
conversion and Ollama's prompt against training.
"""
import argparse
import base64
import json
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def answer_probs(a):
    if a["type"] == "noul":
        return [1 - a["noul"], a["noul"]]
    return list(a["probabilities"].values())


def main(argv=None):
    import torch

    from laya.ollama_train import requests_from
    from laya.vlm_train import load_jsonl_examples, metrics_from

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", default="http://127.0.0.1:11434")
    p.add_argument("--model", default="laya-vision")
    p.add_argument("--data-root", required=True)
    p.add_argument("--datasets", required=True)
    p.add_argument("--n", type=int, default=100, help="first rows of each val.jsonl")
    p.add_argument("--hf-dir", default="")
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    examples = []
    for name in args.datasets.split(","):
        examples += load_jsonl_examples(args.data_root, name, "val", limit=args.n)
    rows = requests_from(examples, random.Random(2), p_single=1.0, shuffle=False)
    torch_model = tok = None
    if args.hf_dir:
        from transformers import AutoProcessor, Idefics3ForConditionalGeneration

        from laya.ollama_train import collate, encode_row, letter_logits

        torch_model = Idefics3ForConditionalGeneration.from_pretrained(args.hf_dir, torch_dtype=torch.float32).eval()
        tok = AutoProcessor.from_pretrained(args.hf_dir).tokenizer
    records, out, worst, t0 = [], [], 0.0, time.time()
    for r in rows:
        body = {"model": args.model, "state": r["state"], "questions": r["questions"],
                "images": [base64.b64encode(open(path, "rb").read()).decode() for path in r["images"]]}
        req = urllib.request.Request(args.url.rstrip("/") + "/v1/systemone", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        res = json.load(urllib.request.urlopen(req))
        probs = answer_probs(res["answers"][r["name"]])
        rec = {"logits": torch.log(torch.tensor(probs).clamp_min(1e-12)), "label": r["label"], "qtype": r["qtype"],
               "target": torch.tensor(r["target"]), "dataset": r["dataset"]}
        records.append(rec)
        row = {"id": r["id"], "dataset": r["dataset"], "label": r["label"], "ollama": probs,
               "input_tokens": res["usage"]["input_tokens"]}
        if torch_model is not None:
            with torch.no_grad():
                z = letter_logits(torch_model, collate([encode_row(r, tok)], tok.pad_token_id), "cpu")[0]
            row["torch"] = z.softmax(-1).tolist()
            worst = max(worst, max(abs(a - b) for a, b in zip(probs, row["torch"])))
        out.append(row)
    m = metrics_from(records)
    for name in sorted(m, key=lambda k: (k != "all", k)):
        print("%-22s n=%4d acc=%.3f ece=%.3f nll=%.3f" % (name, m[name]["n"], m[name]["acc"], m[name]["ece"],
                                                         m[name]["nll"]))
    print("%d requests in %.1f s" % (len(rows), time.time() - t0))
    if torch_model is not None:
        print("largest |P(ollama) - P(torch)|: %.4f" % worst)
    if args.out:
        with open(args.out, "w") as f:
            for row in out:
                f.write(json.dumps(row) + "\n")
        print("wrote %s" % args.out)
    return m


if __name__ == "__main__":
    main()

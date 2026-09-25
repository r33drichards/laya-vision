"""The evaluation-set converters (``laya.evalsets``): no downloads, rows are shaped like the sources show them."""
import io
import random
import zipfile

import pytest

from laya.evalsets import (CIFAR10_CLASSES, FERPLUS_CRITERIA, MultiPartZip, cifar10h_record, evalmuse_record,
                           ferplus_agreement, ferplus_record, histogram, koniq_record, mode_level, pope_record,
                           rf100vl_records, stable_split, vizwiz_record)
from laya.rubric import CRITERIA
from laya.vlm_train import jsonl_example


def test_histogram_and_mode():
    assert histogram([4, 3, 3], 5) == [0.0, 0.0, 0.6667, 0.3333, 0.0]
    assert histogram([0, 9], 5) is None
    assert mode_level([0.1, 0.4, 0.4, 0.1]) == 1  # ties go to the lower level


def test_stable_split_is_deterministic_and_sized():
    keys = ["p%05d" % i for i in range(20000)]
    vals = [k for k in keys if stable_split(k, 10.0) == "val"]
    assert 0.09 < len(vals) / len(keys) < 0.11
    assert vals == [k for k in keys if stable_split(k, 10.0) == "val"]
    assert vals != [k for k in keys if stable_split(k, 10.0, seed=1) == "val"]


def test_koniq_record():
    row = {"image_name": "10004473376.jpg", "c1": "0.0", "c2": "0.0", "c3": "0.238095238095", "c4": "0.695238095238",
           "c5": "0.0666666666667", "c_total": "105", "MOS": "77.38", "SD": "0.53", "set": "training"}
    split, rec = koniq_record(row)
    assert split == "train" and rec["id"] == "koniq-10004473376"
    assert rec["question"]["type"] == "score" and rec["question"]["criteria"] == CRITERIA["quality"]
    assert rec["label"] == 3 and rec["target"] == [0.0, 0.0, 0.2381, 0.6952, 0.0667]
    assert koniq_record(dict(row, set="validation"))[0] == "val" and koniq_record(dict(row, set="test"))[0] == "test"
    assert koniq_record(dict(row, set="other")) is None


def test_evalmuse_record():
    row = {"prompt_id": "00110", "prompt": "A puffin sitting in booth while eating a pastry at a diner. Etching",
           "img_path": "SDXL-Turbo/00110.png", "total_score": [4, 3, 3], "promt_meaningless": [0, 0, 0]}
    rec = evalmuse_record(row)
    assert rec["id"] == "evalmuse-SDXL-Turbo-00110" and rec["source"] == "EvalMuse/SDXL-Turbo"
    assert rec["state_text"].startswith("Prompt: A puffin") and rec["question"]["criteria"] == CRITERIA["alignment"]
    assert rec["label"] == 2 and rec["target"] == [0.0, 0.0, 0.6667, 0.3333, 0.0]
    assert evalmuse_record(dict(row, promt_meaningless=[1, 1, 0])) is None  # most raters: no subject to check
    assert evalmuse_record(dict(row, total_score=[])) is None


def test_multipart_zip_reads_members_across_part_boundaries():
    buf = io.BytesIO()
    payload = {"dataset/images/A/1.png": bytes(range(256)) * 40, "dataset/images/B/2.png": b"x" * 5000}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for name, data in payload.items():
            z.writestr(name, data)
    blob = buf.getvalue()
    cut = [0, 3000, 7777, len(blob)]  # three parts, members straddle the cuts
    parts = [blob[a:b] for a, b in zip(cut, cut[1:])]
    calls = []

    def fetch(i, start, end):
        calls.append((i, start, end))
        return parts[i][start:end]

    zf = MultiPartZip([len(p) for p in parts], fetch).open()
    for name, data in payload.items():
        assert zf.read(name) == data
    assert {c[0] for c in calls} == {0, 1, 2}


def test_cifar10h_record():
    counts = [0, 0, 0, 45, 0, 5, 0, 0, 0, 0]
    rec = cifar10h_record({"label": 3, "expert_counts": counts}, 7)
    assert rec["id"] == "cifar10h-00007" and rec["question"]["criteria"] == list(CIFAR10_CLASSES)
    assert rec["label"] == 3 and rec["target"][3] == 0.9 and rec["target"][5] == 0.1
    assert cifar10h_record({"label": 3, "expert_counts": [0] * 10}, 0) is None


def _fer_row(usage="Training", name="fer0000000.png", **votes):
    row = {"Usage": usage, "Image name": name, "neutral": 0, "happiness": 0, "surprise": 0, "sadness": 0, "anger": 0,
           "disgust": 0, "fear": 0, "contempt": 0, "unknown": 0, "NF": 0}
    row.update({k: str(v) for k, v in votes.items()})
    return {k: str(v) for k, v in row.items()}


def test_ferplus_record():
    split, rec = ferplus_record(_fer_row(neutral=4, sadness=1, anger=3, disgust=2), 0)
    assert split == "train" and rec["question"]["criteria"] == list(FERPLUS_CRITERIA)
    assert rec["label"] == 0 and rec["target"] == [0.4, 0.0, 0.0, 0.1, 0.3, 0.2, 0.0, 0.0]
    assert ferplus_record(_fer_row("PublicTest", happiness=10), 1)[0] == "val"
    assert ferplus_record(_fer_row("PrivateTest", happiness=10), 2)[0] == "test"
    assert ferplus_record(_fer_row(name="", happiness=10), 3) is None  # removed from FER+
    assert ferplus_record(_fer_row(happiness=4, NF=6), 4) is None  # mostly not a face
    assert ferplus_record(_fer_row(happiness=6, unknown=4), 5) is not None


def test_ferplus_agreement_detects_a_shifted_join():
    fer = {"anger": 0, "disgust": 1, "fear": 2, "happiness": 3, "sadness": 4, "surprise": 5, "neutral": 6}
    rng = random.Random(0)
    emotions = [rng.choice(list(fer)) for _ in range(200)]
    votes = [_fer_row(**{e: 8, "neutral": 2}) if e != "neutral" else _fer_row(neutral=10) for e in emotions]
    labels = [fer[e] for e in emotions]
    assert ferplus_agreement(zip(votes, labels)) == 1.0
    assert ferplus_agreement(zip(votes[1:], labels)) < 0.3


def test_vizwiz_record():
    row = {"question_id": "VizWiz_val_00000001", "question": "Can you tell me what this medicine is please?",
           "answers": ["no", "unanswerable", "night time", "unanswerable"] + ["night time"] * 6, "category": "other"}
    rec = vizwiz_record(row)
    assert rec["question"]["type"] == "noul" and rec["state_text"].startswith("Question: Can you tell")
    assert rec["label"] == 1 and rec["target"] == [0.2, 0.8]
    rec = vizwiz_record(dict(row, answers=["unanswerable"] * 9 + ["blank"], category="unanswerable"))
    assert rec["label"] == 0 and rec["target"] == [0.9, 0.1]


def test_pope_record():
    row = {"question_id": "2", "question": "Is there a backpack in the image?", "answer": "no", "category": "adversarial"}
    rec = pope_record(row)
    assert rec["id"] == "pope-adversarial-2" and rec["label"] == 0 and rec["question"]["type"] == "noul"
    assert "target" not in rec and pope_record(dict(row, answer="maybe")) is None


def test_rf100vl_records():
    # shaped like a probicheaux/rf100-vl test row: annotations are a struct of lists, category ids per dataset
    row = {"image_id": "78_344", "dataset_id": "78", "dataset_name": "x-ray-id",
           "annotations": {"id": ["78_1", "78_2", "78_3"], "category_id": [0, 0, 2], "bbox": [[1, 2, 3, 4]] * 3}}
    recs = rf100vl_records(row, ["DIP", "MCP", "PIP"], "x-ray-id")
    assert [r["id"] for r in recs] == ["rf100vl-x-ray-id-78_344-c0", "rf100vl-x-ray-id-78_344-c1",
                                       "rf100vl-x-ray-id-78_344-c2"]
    assert [r["label"] for r in recs] == [1, 0, 1] and {r["question"]["type"] for r in recs} == {"noul"}
    assert recs[1]["question"]["instructions"] == 'Is there at least one "MCP" in the image?'
    assert recs[0]["state_text"] == "An image from the x-ray-id dataset, labelled for: DIP, MCP, PIP."
    assert [r["label"] for r in rf100vl_records(dict(row, annotations={"category_id": []}), ["a", "b"], "d")] == [0, 0]
    assert rf100vl_records(dict(row, annotations={"category_id": [5]}), ["a", "b"], "d") == []  # id out of range
    assert rf100vl_records(row, [], "d") == []


@pytest.mark.parametrize("rec", [
    koniq_record({"image_name": "1.jpg", "c1": 0.1, "c2": 0.2, "c3": 0.4, "c4": 0.2, "c5": 0.1, "set": "test"})[1],
    evalmuse_record({"prompt_id": "1", "prompt": "a cat", "img_path": "M/1.png", "total_score": [5, 5, 4]}),
    cifar10h_record({"label": 0, "expert_counts": [50] + [0] * 9}, 0),
    ferplus_record(_fer_row(happiness=10), 0)[1],
    vizwiz_record({"question_id": "q", "question": "what?", "answers": ["a"] * 10, "category": "other"}),
    pope_record({"question_id": "1", "question": "Is there a dog in the image?", "answer": "yes", "category": "random"}),
    rf100vl_records({"image_id": "1_2", "annotations": {"category_id": [1]}}, ["cat", "dog"], "pets")[1],
])
def test_records_load_as_examples_with_their_targets(rec):
    rec = dict(rec, image="images/x.jpg")
    ex = jsonl_example(rec, "/data/vqa/eval_x", "eval_x")
    assert ex is not None and ex["label"] == rec["label"] and ex["state"]["image"] == "/data/vqa/eval_x/images/x.jpg"
    if "target" in rec:
        assert ex["target"] == pytest.approx([t / sum(rec["target"]) for t in rec["target"]])

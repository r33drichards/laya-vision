"""Tests for laya.synth: question construction from fake rows and the prepared-dataset writer. No downloads."""
import json
import os
import random

import pytest
from PIL import Image

from laya.common import render_options
from laya.synth import DatasetWriter, Example, Pool, run_source, split_of
from laya.synth.sources import SOURCES, AVA, NLVR2, AOKVQA, ScreenQA, Screen2Words, VizWiz, VQAv2, WebSight
from laya.vlm_train import jsonl_example, load_jsonl_examples


def img(w=64, h=48, color=(200, 30, 30)):
    return Image.new("RGB", (w, h), color)


def check_examples(exs, rng=None):
    """Every example must be a valid public question with a label inside its rendered options."""
    from laya.vlm import VLMAgent

    assert exs
    for ex in exs:
        assert ex.valid(), ex.question
        q = VLMAgent._to_internal(ex.question)
        k = len(render_options(q))
        assert 0 <= ex.label < k
        if ex.target is not None:
            assert len(ex.target) == k and abs(sum(ex.target) - 1) < 1e-6
            assert ex.label == max(range(k), key=lambda i: ex.target[i])


# ---------------------------------------------------------------------------------------------------------
# core
# ---------------------------------------------------------------------------------------------------------


def test_split_is_deterministic_and_roughly_proportional():
    keys = ["img-%d" % i for i in range(4000)]
    a = [split_of(k, 0.1, seed=1) for k in keys]
    assert a == [split_of(k, 0.1, seed=1) for k in keys]
    frac = a.count("val") / len(a)
    assert 0.07 < frac < 0.13
    assert a != [split_of(k, 0.1, seed=2) for k in keys]


def test_pool_excludes_answers_and_shared_words():
    pool = Pool(seed=0)
    for s in ["Law Firm", "Fashion Brand", "Fashion Retailer", "Restaurant Chain", "Art Gallery", "Florist", "law firm"]:
        pool.add("site", s)
    assert pool.size("site") == 6  # "law firm" is a duplicate of "Law Firm"
    rng = random.Random(0)
    got = pool.sample("site", 10, rng, exclude=["Fashion Brand"], disjoint=True)
    assert "Fashion Brand" not in got and "Fashion Retailer" not in got
    assert set(got) == {"Law Firm", "Restaurant Chain", "Art Gallery", "Florist"}
    assert pool.sample("nothing", 3, rng) == []
    assert len(pool.sample("nothing", 3, rng, fallback="site")) == 3


def test_pool_cap_keeps_bounded():
    pool = Pool(cap=10, seed=0)
    for i in range(500):
        pool.add("k", "answer %d" % i)
    assert pool.size("k") == 10


def test_example_validation():
    from laya.synth import choice_q, noul_q, score_q

    assert Example("f", choice_q("q", ["a", "b"]), 1).valid()
    assert not Example("f", choice_q("q", ["a", "a"]), 0).valid()  # duplicates collapse
    assert not Example("f", choice_q("q", ["a"]), 0).valid()
    assert not Example("f", noul_q(""), 0).valid()
    assert not Example("f", noul_q("q"), 2).valid()
    assert not Example("f", score_q("q", ["lo", "hi"]), 0, target=[0.5]).valid()
    assert Example("f", score_q("q", ["lo", "hi"]), 0, target=[0.7, 0.3]).valid()


# ---------------------------------------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------------------------------------

WS_ROWS = [
    {"image": img(), "llm_generated_idea": "%s: some idea" % name,
     "text": '<html><body class="%s"><nav><a href="#">x</a><a>y</a></nav>%s<footer></footer></body></html>' % (cls, extra)}
    for name, cls, extra in [("Law Firm", "bg-gray-100", "<img src=a>"), ("Fashion Brand", "bg-gray-900 text-white", "<form><input></form>"),
                             ("Restaurant Chain", "bg-white", "<table></table><img><img><img><img>"), ("Art Gallery", "bg-black", ""),
                             ("Florist", "bg-green-100", "<button>go</button>"), ("Medical Clinic", "", "")]
]


def test_websight_labels_come_from_html():
    src = WebSight()
    for r in WS_ROWS:
        src.observe(r)
    src.per_row = 99
    rng = random.Random(0)
    exs = src.examples(WS_ROWS[1], rng)
    check_examples(exs)
    by = {e.family: e for e in exs}
    assert by["ws_has_nav"].label == 1 and by["ws_has_footer"].label == 1 and by["ws_has_form"].label == 1
    assert by["ws_has_table"].label == 0 and by["ws_has_img"].label == 0 and by["ws_has_button"].label == 0
    assert by["ws_dark"].label == 1
    assert by["ws_links"].label == 1 and by["ws_images"].label == 0
    site = by["ws_site_type"]
    assert site.question["criteria"][site.label] == "Fashion Brand" and len(site.question["criteria"]) == 5
    exs = src.examples(WS_ROWS[2], rng)
    by = {e.family: e for e in exs}
    assert by["ws_dark"].label == 0 and by["ws_has_table"].label == 1 and by["ws_images"].label == 3
    # the default samples a few questions per page
    src.per_row = 3
    assert len(src.examples(WS_ROWS[0], rng)) == 3


def test_websight_unknown_theme_is_skipped():
    src = WebSight()
    src.per_row = 99
    r = dict(WS_ROWS[0], text='<html><body class="bg-teal-500"></body></html>')
    assert "ws_dark" not in {e.family for e in src.examples(r, random.Random(0))}


def test_screen2words_distractors_share_no_words():
    src = Screen2Words()
    rows = [{"screenId": i, "image": img(), "category": c, "captions": caps} for i, (c, caps) in enumerate([
        ("Weather", ["weather forecast page", "page showing the forecast"]),
        ("Music & Audio", ["music player screen", "screen with a playing song"]),
        ("Finance", ["bank account balance", "page with account totals"]),
        ("Shopping", ["shopping cart contents", "cart with items"]),
        ("Education", ["quiz question page", "multiple choice quiz"]),
        ("Social", ["chat conversation", "messages with a friend"]),
        ("Travel & Local", ["hotel booking form", "form to book a hotel"]),
    ])]
    for r in rows:
        src.observe(r)
    rng = random.Random(0)
    exs = src.examples(rows[0], rng)
    check_examples(exs)
    by = {e.family: e for e in exs}
    cap = by["s2w_caption"]
    assert cap.question["criteria"][cap.label] in rows[0]["captions"]
    for i, opt in enumerate(cap.question["criteria"]):
        if i != cap.label:
            assert "forecast" not in opt and "weather" not in opt
    cat = by["s2w_category"]
    assert cat.question["criteria"][cat.label] == "Weather" and len(set(cat.question["criteria"])) == 5


def test_screenqa_choice_and_verify():
    src = ScreenQA()
    rows = [{"screen_id": i, "image": img(), "question": "What is the default %s?" % w, "ground_truth": [a]}
            for i, (w, a) in enumerate([("period length", "five days"), ("cycle length", "30 days"), ("alarm time", "7:00 am"),
                                        ("volume", "50%"), ("language", "English"), ("theme", "dark")])]
    for r in rows:
        src.observe(r)
    rng = random.Random(1)
    row = {"screen_id": 9, "image": img(), "question": "What is the default snooze time?", "ground_truth": ["10 minutes", "10 min"]}
    exs = src.examples(row, rng)
    check_examples(exs)
    ch = [e for e in exs if e.family == "sqa_answer"][0]
    assert ch.question["criteria"][ch.label] == "10 minutes" and len(ch.question["criteria"]) == 4
    assert "10 min" not in ch.question["criteria"]
    ver = [e for e in exs if e.family == "sqa_verify"][0]
    assert ("'10 minutes'" in ver.question["instructions"] or "10 minutes." in ver.question["instructions"]) == bool(ver.label)


def test_vqav2_choice_soft_targets_and_counts():
    src = VQAv2()
    votes = lambda *a: [{"answer": x} for x in a]
    for i, a in enumerate(["red", "blue", "green", "white", "black", "yellow"]):
        src.observe({"question_type": "what color is the", "answer_type": "other", "multiple_choice_answer": a, "image_id": i,
                     "question": "What color is the bus?", "answers": votes(a)})
    rng = random.Random(0)
    row = {"question_type": "what color is the", "answer_type": "other", "multiple_choice_answer": "red", "image_id": 7,
           "question": "What color is the car?", "answers": votes(*(["red"] * 7 + ["blue"] * 2 + ["maroon"]))}
    exs = src.examples(row, rng)
    check_examples(exs)
    ex = exs[0]
    opts = ex.question["criteria"]
    assert opts[ex.label] == "red" and "blue" not in opts and "maroon" not in opts and len(opts) == 4
    assert ex.target[ex.label] == 1.0  # only the answer's own votes land on the options
    row = {"question_type": "how many", "answer_type": "number", "multiple_choice_answer": "2", "image_id": 8,
           "question": "How many dogs are there?", "answers": votes(*(["2"] * 6 + ["two"] * 2 + ["3"] + ["lots"]))}
    exs = src.examples(row, rng)
    check_examples(exs)
    assert exs[0].question["type"] == "score" and exs[0].label == 2
    assert abs(exs[0].target[2] - 8 / 9) < 1e-9 and abs(exs[0].target[3] - 1 / 9) < 1e-9
    yn = dict(row, answer_type="yes/no", answers=votes("yes"))
    assert src.examples(yn, rng) == []


def test_aokvqa_verify_uses_own_choices():
    src = AOKVQA()
    row = {"question_id": "q1", "image": img(), "question": "What is he waiting for?", "choices": ["train", "cab", "bus"],
           "correct_choice_idx": 1}
    seen = set()
    for seed in range(20):
        exs = src.examples(row, random.Random(seed))
        check_examples(exs)
        ins = exs[0].question["instructions"]
        assert ("'cab'" in ins) == bool(exs[0].label)
        seen.add(exs[0].label)
    assert seen == {0, 1}


def test_vizwiz_answerability_is_soft():
    src = VizWiz()
    for i, a in enumerate(["night time", "aspirin", "cereal", "soup", "rice", "coffee"]):
        src.observe({"question_id": i, "image": img(), "question": "What is this?", "category": "other", "answers": [a] * 10})
    rng = random.Random(0)
    row = {"question_id": 99, "image": img(), "question": "What is this medicine?", "category": "other",
           "answers": ["unanswerable"] * 2 + ["night time"] * 7 + ["cold medicine"]}
    exs = src.examples(row, rng)
    check_examples(exs)
    by = {e.family: e for e in exs}
    assert by["vw_answerable"].label == 1 and abs(by["vw_answerable"].target[1] - 0.8) < 1e-9
    assert by["vw_choice"].question["criteria"][by["vw_choice"].label] == "night time"
    row = {"question_id": 100, "image": img(), "question": "Is this blue?", "category": "yes/no",
           "answers": ["no"] * 6 + ["yes"] * 3 + ["unanswerable"]}
    by = {e.family: e for e in src.examples(row, rng)}
    assert by["vw_yesno"].label == 0 and abs(by["vw_yesno"].target[1] - 1 / 3) < 1e-9
    row = {"question_id": 101, "image": img(), "question": "What is this?", "category": "unanswerable",
           "answers": ["unanswerable"] * 9 + ["blank"]}
    by = {e.family: e for e in src.examples(row, rng)}
    assert by["vw_answerable"].label == 0 and list(by) == ["vw_answerable"]


def test_ava_collapses_ten_bins_to_five_levels():
    src = AVA()
    row = {"image_id": "1", "image": img(), "rating_counts": [1, 2, 7, 34, 53, 50, 28, 19, 7, 7]}
    exs = src.examples(row, random.Random(0))
    check_examples(exs)
    ex = exs[0]
    assert ex.question["type"] == "score" and len(ex.question["criteria"]) == 5
    assert ex.label == 2 and abs(ex.target[2] - 103 / 208) < 1e-9
    assert src.examples({"image_id": "2", "image": img(), "rating_counts": [0] * 10}, random.Random(0)) == []


def test_nlvr2_two_images():
    src = NLVR2()
    row = {"identifier": "dev-1-0-1", "question": "There are six bottles in the right image.", "answer": "False",
           "left_image": img(), "right_image": img(32, 32)}
    exs = src.examples(row, random.Random(0))
    check_examples(exs)
    assert exs[0].label == 0 and len(src.images(row)) == 2
    assert "right image" in exs[0].question["instructions"]


# ---------------------------------------------------------------------------------------------------------
# writer + driver, end to end into the trainer's loader
# ---------------------------------------------------------------------------------------------------------


def test_run_source_writes_loadable_dataset(tmp_path):
    src = ScreenQA()
    rows = [{"screen_id": i // 2, "image": img(300, 500), "question": "What is the default %s %d?" % (w, i), "ground_truth": ["value %d" % i]}
            for i, w in enumerate(["period"] * 40)]
    writer = DatasetWriter(str(tmp_path), "synth_screenqa", max_side=128)
    meta = run_source(src, writer, n_train=12, n_val=4, seed=0, rows={"train": rows[:30], "test": rows[30:]}, min_pool=5, log=lambda *a: None)
    root = tmp_path / "synth_screenqa"
    assert (root / "_READY").exists() and not (tmp_path / "synth_screenqa.tmp").exists()
    assert meta["train"]["records"] >= 12 and meta["val"]["records"] >= 4
    assert meta["hf_splits"] == {"train": "train", "val": "test"} and meta["dropped"]["warmup"] > 0
    recs = [json.loads(l) for l in open(root / "train.jsonl")]
    assert all(os.path.exists(root / r["image"]) for r in recs)
    # one image file per screen, shared by its questions
    assert len({r["image"] for r in recs}) == meta["train"]["images"] < len(recs)
    with Image.open(root / recs[0]["image"]) as im:
        assert max(im.size) == 128
    exs = load_jsonl_examples(str(tmp_path), "synth_screenqa", "train")
    assert len(exs) == len(recs) and exs[0]["dataset"] == "synth_screenqa"
    assert {"sqa_answer", "sqa_verify"} <= {r["family"] for r in recs}


def test_run_source_hash_split_and_multi_image(tmp_path):
    src = NLVR2()
    rows = [{"identifier": "dev-%d-0-1" % i, "question": "Statement %d." % i, "answer": "True" if i % 2 else "False",
             "left_image": img(), "right_image": img(40, 40, (0, 0, 200))} for i in range(200)]
    writer = DatasetWriter(str(tmp_path), "synth_nlvr2", max_side=64)
    meta = run_source(src, writer, n_train=150, n_val=10, seed=0, val_frac=0.1, rows={"unbalanced_dev": rows}, min_pool=0, log=lambda *a: None)
    assert meta["val"]["records"] == 10 and meta["train"]["records"] == 150
    recs = [json.loads(l) for l in open(tmp_path / "synth_nlvr2" / "val.jsonl")]
    assert all(len(r["images"]) == 2 and "image" not in r for r in recs)
    ex = jsonl_example(recs[0], str(tmp_path / "synth_nlvr2"), "synth_nlvr2")
    assert len(ex["state"]["images"]) == 2 and all(os.path.exists(p) for p in ex["state"]["images"])
    # a record's split follows its key, not its position in the stream
    for r in recs:
        assert split_of("nlvr2-" + r["id"].split("-nlvr2_pair")[0][len("nlvr2-"):], 0.1, 0) == "val"


def test_all_sources_registered():
    assert set(SOURCES) == {"websight", "screen2words", "screenqa", "vqav2", "aokvqa", "vizwiz", "ava", "nlvr2"}
    for cls in SOURCES.values():
        assert cls.repo and cls.license and cls.train_split

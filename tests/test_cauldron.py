"""The Cauldron turn parsers (``laya.cauldron``): no downloads, the texts are verbatim from the dataset viewer."""
import random

import pytest

from laya.cauldron import RAVEN_OPTIONS, cauldron_records, parse_turn


def test_lettered_choices_with_context():
    user = ("Lecture: Maps have four cardinal directions, or main directions.\nA compass rose is a set of arrows.\n"
            "Question: Which of these states is farthest north?\nChoices:\nA. Maine\nB. Texas\nC. Florida\nAnswer with the letter.")
    p = parse_turn(user, "Answer: A")
    assert p == {"question": {"type": "choice", "instructions": "Which of these states is farthest north?",
                              "criteria": ["Maine", "Texas", "Florida"]}, "label": 0,
                 "state_text": "Lecture: Maps have four cardinal directions, or main directions.\nA compass rose is a set of arrows."}
    p = parse_turn("Question: How many actions are depicted in the diagram?\nChoices:\nA. 6.\nB. 4.\nC. 8.\nD. 7.\nAnswer with the letter.", "Answer: D")
    assert p["question"]["criteria"] == ["6.", "4.", "8.", "7."] and p["label"] == 3 and p["state_text"] is None
    assert parse_turn("Question: q\nChoices:\nA. same\nB. same\nAnswer with the letter.", "Answer: A") is None  # duplicates
    assert parse_turn("Question: q\nChoices:\nA. x\nB. y\nAnswer with the letter.", "Answer: E") is None  # out of range
    assert parse_turn("Question: q\nChoices:\nA. x\nB. y\nAnswer with the letter.", "The answer is B") is None


def test_options_list():
    user = "What is the man by the bags awaiting?\nMake your selection from the four choices given to correctly answer the question.\nOptions: Skateboarder, train, delivery, cab."
    p = parse_turn(user, "Cab.")
    assert p == {"question": {"type": "choice", "instructions": "What is the man by the bags awaiting?",
                              "criteria": ["skateboarder", "train", "delivery", "cab"]}, "label": 3, "state_text": None}
    user = "Where does this man eat pizza?\nPick the right solution, then justify: 'Answer: answer\nRationale: rationale.'\nOptions: Office, cafe, motel, outside."
    p = parse_turn(user, "Answer: office.\nRationale: The man is eating pizza at a work desk in an office setting.")
    assert p["label"] == 0 and p["question"]["instructions"] == "Where does this man eat pizza?"
    assert parse_turn(user, "Answer: kitchen.") is None  # answer not among the options


def test_yes_no():
    p = parse_turn('Is the statement "The cake is next to the person." accurate regarding the image?\nAnswer yes or no.', "Yes.")
    assert p == {"question": {"type": "noul", "instructions": 'Is the statement "The cake is next to the person." accurate regarding the image?',
                              "criteria": None}, "label": 1, "state_text": None}
    p = parse_turn("Is this a cat?\nKeep it brief.", "No.")  # VQAv2-style: brevity instruction dropped
    assert p["question"]["instructions"] == "Is this a cat?" and p["label"] == 0
    nl = 'The first image is the image on the left, the second image is the image on the right. Assess this claim about the two images: "There are exactly two flutes.". Correct or not? Answer yes or no.'
    p = parse_turn(nl, "Yes.", n_images=2)
    assert p["question"]["type"] == "noul" and p["question"]["instructions"] == nl[: -len(" Answer yes or no.")]
    assert parse_turn("What color is the snow?\nQuick response, please.", "White.") is None  # open answer


def test_raven():
    p = parse_turn("Which figure should complete the logical sequence?", "B", n_images=1)
    assert p["question"]["criteria"] == RAVEN_OPTIONS and p["label"] == 1
    assert parse_turn("Which figure should complete the logical sequence?", "B", n_images=2) is None
    assert parse_turn("Which figure should complete the logical sequence?", "Yes.", n_images=2)["question"]["type"] == "noul"


def test_records_sample_and_point_at_images():
    texts = [{"user": "Is it red?\nAnswer yes or no.", "assistant": "Yes.", "source": "VSR"},
             {"user": "Describe the image.", "assistant": "A long caption.", "source": "VSR"},
             {"user": "Is it blue?\nAnswer yes or no.", "assistant": "No.", "source": "VSR"},
             {"user": "Is it big?\nAnswer yes or no.", "assistant": "No.", "source": "VSR"}]
    recs = cauldron_records(texts, ["images/vsr-7-0.jpg"], "vsr-7")
    assert [r["id"] for r in recs] == ["vsr-7-0", "vsr-7-2", "vsr-7-3"] and all(r["image"] == "images/vsr-7-0.jpg" for r in recs)
    assert recs[0]["label"] == 1 and recs[1]["label"] == 0 and recs[0]["source"] == "VSR"
    two = cauldron_records(texts, ["a.jpg", "b.jpg"], "nlvr2-1", max_texts=2, rng=random.Random(0))
    assert len(two) == 2 and all(r["images"] == ["a.jpg", "b.jpg"] and "image" not in r for r in two)
    assert cauldron_records(texts[1:2], ["a.jpg"], "x") == []

"""Sources: one class per existing dataset, each turning a labelled row into typed questions.

Each source declares the Hub repo and splits, a stable ``key`` per image, the ``images`` of a row, an ``observe``
hook that feeds its distractor pools, and ``examples`` that returns ``Example`` objects. Prepared datasets are
named ``synth_<source>``; ``SOURCES`` maps the short name to the class.

Screens: ``websight`` (web pages, labels from the HTML), ``screen2words`` and ``screenqa`` (RICO mobile screens).
Photos: ``vqav2``, ``aokvqa``, ``vizwiz`` (answerability), ``ava`` (ordinal aesthetics), ``nlvr2`` (two images).
"""
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional

from .core import Example, Pool, choice_q, content_words, norm, noul_q, pick, score_q, soft_target

NUMBER_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}


def as_int(s: str) -> Optional[int]:
    n = norm(s)
    if n in NUMBER_WORDS:
        return NUMBER_WORDS[n]
    return int(n) if re.fullmatch(r"\d+", n) else None


class Source:
    name = ""
    repo = ""
    config: Optional[str] = None
    license = ""
    origin = ""
    train_split = "train"
    val_split: Optional[str] = None  # None: hash-split the train stream
    columns: Optional[List[str]] = None  # keep only these columns while streaming (rows can carry extra images)
    data_files: Optional[Dict[str, str]] = None  # split -> parquet glob, when the repo card omits the split

    def __init__(self, seed: int = 0):
        self.pool = Pool(seed=seed)

    def key(self, row: Dict, idx: int) -> str:
        return str(idx)

    def images(self, row: Dict) -> List[Any]:
        return [row["image"]]

    def observe(self, row: Dict) -> None:
        pass

    def ready(self, min_pool: int) -> bool:
        return True

    def examples(self, row: Dict, rng: random.Random) -> List[Example]:
        raise NotImplementedError


# ---------------------------------------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------------------------------------


class WebSight(Source):
    """Synthetic web pages with their HTML (2M, CC-BY-4.0). Labels are read from the HTML, so they are exact.

    The screenshots are full-page, so an element in the HTML is on the screenshot. Questions: presence of a
    navigation bar, footer, form, table, images and buttons (``noul``), dark colour scheme (``noul``), the kind
    of business the site is for (``choice``, from the generation prompt) and bucketed link and image counts
    (``score``). ``per_row`` questions are sampled per page so one page does not dominate a batch.
    """

    name = "websight"
    columns = ["image", "text", "llm_generated_idea"]
    repo = "HuggingFaceM4/WebSight"
    config = "v0.2"
    license = "CC-BY-4.0"
    origin = "https://huggingface.co/datasets/HuggingFaceM4/WebSight (v0.2)"
    per_row = 3

    TAGS = {
        "nav": (r"<nav[\s>]", ["Does this web page have a navigation bar or menu?",
                                "Is there a site navigation menu on this page?",
                                "Can you see a navigation bar with links on this page?"]),
        "footer": (r"<footer[\s>]", ["Does the page have a footer section at the bottom?",
                                      "Is there a footer on this web page?",
                                      "Does this page end with a footer area?"]),
        "form": (r"<form[\s>]|<input[\s>]|<textarea[\s>]", ["Is there a form with input fields on this page?",
                                                            "Does this page contain any text input fields?",
                                                            "Can the user type something into a field on this page?"]),
        "table": (r"<table[\s>]", ["Does the page contain a table?",
                                    "Is there a data table on this web page?",
                                    "Is any of the content on this page laid out as a table with rows and columns?"]),
        "img": (r"<img[\s>]", ["Does the page show any images or photos?",
                                "Are there pictures on this web page?",
                                "Does this page include at least one image?"]),
        "button": (r"<button[\s>]|<input[^>]*type=\"?submit", ["Are there any buttons on the page?",
                                                                "Does this page have a clickable button?",
                                                                "Is there a button the user can press on this page?"]),
    }
    DARK = re.compile(r"\bbg-(?:black|(?:gray|slate|zinc|neutral|stone|blue|indigo|purple)-(?:800|900|950))\b")
    LIGHT = re.compile(r"\bbg-(?:white|(?:gray|slate|zinc|neutral|stone|blue|green|red|pink|purple|yellow|indigo)-(?:50|100|200))\b")
    SITE_PHRASINGS = ["What kind of business or organisation is this website for?",
                      "Who is this web page most likely for?",
                      "Which of these best describes the site shown?"]
    LINK_LEVELS = ["none", "one to three", "four to eight", "more than eight"]
    IMAGE_LEVELS = ["none", "one", "two or three", "four or more"]

    def observe(self, row: Dict) -> None:
        site = self.site_type(row)
        if site:
            self.pool.add("site", site)

    def ready(self, min_pool: int) -> bool:
        return self.pool.size("site") >= min(min_pool, 50)

    @staticmethod
    def site_type(row: Dict) -> Optional[str]:
        idea = row.get("llm_generated_idea") or ""
        head = idea.split(":")[0].strip() if ":" in idea else ""
        return head if 2 < len(head) <= 40 and len(head.split()) <= 4 else None

    @staticmethod
    def body_class(html: str) -> str:
        m = re.search(r'<body[^>]*class="([^"]*)"', html)
        return m.group(1) if m else ""

    def examples(self, row: Dict, rng: random.Random) -> List[Example]:
        html = row["text"].lower()
        cands: List[Example] = []
        for tag, (pattern, phr) in self.TAGS.items():
            present = re.search(pattern, html) is not None
            cands.append(Example("ws_has_" + tag, noul_q(pick(rng, phr)), int(present)))
        body = self.body_class(html)
        if self.DARK.search(body):
            cands.append(Example("ws_dark", noul_q(pick(rng, ["Does this page use a dark colour scheme?",
                                                              "Is the page background dark with light text?"])), 1))
        elif self.LIGHT.search(body) or not body:
            cands.append(Example("ws_dark", noul_q(pick(rng, ["Does this page use a dark colour scheme?",
                                                              "Is the page background dark with light text?"])), 0))
        site = self.site_type(row)
        if site:
            distractors = self.pool.sample("site", 4, rng, exclude=[site], disjoint=True)
            if len(distractors) == 4:
                opts = [site] + distractors
                rng.shuffle(opts)
                cands.append(Example("ws_site_type", choice_q(pick(rng, self.SITE_PHRASINGS), opts), opts.index(site)))
        n_links = len(re.findall(r"<a[\s>]", html))
        lvl = 0 if n_links == 0 else 1 if n_links <= 3 else 2 if n_links <= 8 else 3
        cands.append(Example("ws_links", score_q(pick(rng, ["How many links does this page have?",
                                                            "Roughly how many clickable links are on the page?"]),
                                                 self.LINK_LEVELS), lvl))
        n_imgs = len(re.findall(r"<img[\s>]", html))
        lvl = 0 if n_imgs == 0 else 1 if n_imgs == 1 else 2 if n_imgs <= 3 else 3
        cands.append(Example("ws_images", score_q(pick(rng, ["How many images are on this page?",
                                                             "How many pictures does the page show?"]),
                                                  self.IMAGE_LEVELS), lvl))
        return rng.sample(cands, min(self.per_row, len(cands)))


class Screen2Words(Source):
    """RICO mobile screens with five human summaries each and the app's Play Store category (CC-BY-4.0).

    ``choice``: which summary describes the screen (distractors are other screens' summaries sharing no content
    word with any of this screen's five); which app category the screen is from (5-way over the 21 categories).
    """

    name = "screen2words"
    columns = ["screenId", "image", "captions", "category"]
    repo = "bevaya/RICO-Screen2Words"
    license = "CC-BY-4.0"
    origin = "https://huggingface.co/datasets/bevaya/RICO-Screen2Words (Screen2Words, Wang et al. 2021)"
    train_split = "train"
    val_split = "test"
    CAPTION_PHRASINGS = ["Which description best matches this screen?",
                         "Which of these summaries describes what is on the screen?",
                         "Pick the sentence that describes this app screen."]
    CATEGORY_PHRASINGS = ["What category of app is this screen most likely from?",
                          "Which kind of app does this screen belong to?"]

    def key(self, row: Dict, idx: int) -> str:
        return str(row["screenId"])

    def observe(self, row: Dict) -> None:
        for c in row.get("captions") or []:
            self.pool.add("caption", c)
        if row.get("category"):
            self.pool.add("category", row["category"])

    def ready(self, min_pool: int) -> bool:
        return self.pool.size("caption") >= min_pool and self.pool.size("category") >= 5

    def examples(self, row: Dict, rng: random.Random) -> List[Example]:
        out = []
        caps = [c for c in (row.get("captions") or []) if c.strip()]
        if caps:
            answer = rng.choice(caps)
            distractors = self.pool.sample("caption", 4, rng, exclude=caps, disjoint=True)
            if len(distractors) == 4:
                opts = [answer] + distractors
                rng.shuffle(opts)
                out.append(Example("s2w_caption", choice_q(pick(rng, self.CAPTION_PHRASINGS), opts), opts.index(answer)))
        cat = row.get("category")
        if cat:
            distractors = self.pool.sample("category", 4, rng, exclude=[cat])
            if len(distractors) == 4:
                opts = [cat] + distractors
                rng.shuffle(opts)
                out.append(Example("s2w_category", choice_q(pick(rng, self.CATEGORY_PHRASINGS), opts), opts.index(cat)))
        return out


class ScreenQA(Source):
    """RICO screens with short-answer questions (ScreenQA-Short, CC-BY-4.0).

    ``choice``: the question with the answer and three distractors drawn from other rows whose question starts
    with the same two words (so "What is the default cycle length?" gets other lengths and dates, not app
    names). ``noul``: "<question> Is the answer '<x>'?" with a true or a distractor answer, half and half.
    """

    name = "screenqa"
    columns = ["screen_id", "image", "question", "ground_truth"]
    repo = "bevaya/RICO-ScreenQA-Short"
    license = "CC-BY-4.0"
    origin = "https://huggingface.co/datasets/bevaya/RICO-ScreenQA-Short (ScreenQA, Hsiao et al. 2022)"
    train_split = "train"
    val_split = "test"

    @staticmethod
    def prefix(question: str, n: int = 3) -> str:
        return " ".join(norm(question).split()[:n])

    def key(self, row: Dict, idx: int) -> str:
        return str(row["screen_id"])

    def observe(self, row: Dict) -> None:
        truths = row.get("ground_truth") or []
        if truths:
            self.pool.add(self.prefix(row["question"], 3), truths[0])
            self.pool.add(self.prefix(row["question"], 2), truths[0])
            self.pool.add("all", truths[0])

    def ready(self, min_pool: int) -> bool:
        return self.pool.size("all") >= min_pool

    def examples(self, row: Dict, rng: random.Random) -> List[Example]:
        q, truths = row["question"].strip(), [t for t in (row.get("ground_truth") or []) if t.strip()]
        if not q or not truths or len(truths[0]) > 60:
            return []
        answer = truths[0]
        distractors = self.pool.sample(self.prefix(q, 3), 3, rng, exclude=truths, fallback=[self.prefix(q, 2), "all"])
        if len(distractors) < 3:
            return []
        opts = [answer] + distractors
        rng.shuffle(opts)
        out = [Example("sqa_answer", choice_q(q, opts), opts.index(answer))]
        if rng.random() < 0.5:
            out.append(Example("sqa_verify", noul_q(pick(rng, ["{q} Is the answer '{a}'?", "{q} Answer: {a}. Is that right?"],
                                                         q=q, a=answer)), 1))
        else:
            wrong = rng.choice(distractors)
            out.append(Example("sqa_verify", noul_q(pick(rng, ["{q} Is the answer '{a}'?", "{q} Answer: {a}. Is that right?"],
                                                         q=q, a=wrong)), 0))
        return out


# ---------------------------------------------------------------------------------------------------------
# Photos
# ---------------------------------------------------------------------------------------------------------


def votes_of(answers: List[Any]) -> Counter:
    return Counter(norm(a["answer"] if isinstance(a, dict) else a) for a in answers or [])


class VQAv2(Source):
    """VQAv2 (CC-BY-4.0) minus its yes/no questions, which the existing ``vqav2_yesno`` set already covers.

    ``choice``: the question, the majority answer and three distractors from other rows of the same
    ``question_type`` ("what color is the" ...), the classic VQA multiple-choice construction; the target is the
    ten annotators' votes spread over the options. ``score``: "how many" questions as an ordinal count
    (none / one / two / three / four or more), soft from the votes.
    """

    name = "vqav2"
    columns = ["image_id", "image", "question", "question_type", "answer_type", "multiple_choice_answer", "answers"]
    data_files = {"train": "data/train-*.parquet", "validation": "data/validation-*.parquet"}  # card lists no train split
    repo = "lmms-lab/VQAv2"
    license = "CC-BY-4.0"
    origin = "https://huggingface.co/datasets/lmms-lab/VQAv2 (VQAv2, Goyal et al. 2017; COCO images)"
    train_split = "train"
    val_split = "validation"
    COUNT_LEVELS = ["none", "one", "two", "three", "four or more"]

    def key(self, row: Dict, idx: int) -> str:
        return str(row["image_id"])

    def observe(self, row: Dict) -> None:
        if row.get("answer_type") == "other" and row.get("multiple_choice_answer"):
            self.pool.add(row.get("question_type") or "all", row["multiple_choice_answer"])
            self.pool.add("all", row["multiple_choice_answer"])

    def ready(self, min_pool: int) -> bool:
        return self.pool.size("all") >= min_pool

    def examples(self, row: Dict, rng: random.Random) -> List[Example]:
        q, votes = row["question"].strip(), votes_of(row.get("answers"))
        if not q or not votes:
            return []
        if row.get("answer_type") == "number":
            nums = [as_int(a) for a in votes.elements()]
            nums = [n for n in nums if n is not None]
            if len(nums) < 6 or not norm(q).startswith("how many"):
                return []
            counts = Counter(min(n, 4) for n in nums)
            target = [counts.get(i, 0) / len(nums) for i in range(5)]
            return [Example("vqa_count", score_q(q, self.COUNT_LEVELS), max(range(5), key=lambda i: target[i]), target)]
        if row.get("answer_type") != "other":
            return []
        answer = row.get("multiple_choice_answer") or votes.most_common(1)[0][0]
        if len(answer) > 40:
            return []
        distractors = self.pool.sample(row.get("question_type") or "all", 3, rng, exclude=list(votes), fallback="all")
        if len(distractors) < 3:
            return []
        opts = [answer] + distractors
        rng.shuffle(opts)
        return [Example("vqa_choice", choice_q(q, opts), opts.index(answer), soft_target(votes, opts))]


class AOKVQA(Source):
    """A-OKVQA (CC-BY-4.0). The native 4-way ``choice`` is already a training set; this adds ``noul`` verification:
    "<question> Is the answer '<x>'?" where x is the correct choice or one of the row's own wrong choices, so the
    negatives are the dataset's hard distractors.
    """

    name = "aokvqa"
    columns = ["question_id", "image", "question", "choices", "correct_choice_idx"]
    repo = "HuggingFaceM4/A-OKVQA"
    license = "CC-BY-4.0"
    origin = "https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA (Schwenk et al. 2022; COCO images)"
    train_split = "train"
    val_split = "validation"

    def key(self, row: Dict, idx: int) -> str:
        return str(row["question_id"])

    def examples(self, row: Dict, rng: random.Random) -> List[Example]:
        q, choices, ci = row["question"].strip(), list(row["choices"]), int(row["correct_choice_idx"])
        if not q or not 0 <= ci < len(choices) or len(choices) < 2:
            return []
        truth = rng.random() < 0.5
        a = choices[ci] if truth else rng.choice([c for i, c in enumerate(choices) if i != ci])
        ins = pick(rng, ["{q} Is the answer '{a}'?", "{q} Someone answered '{a}'. Is that correct?",
                         "Is '{a}' the right answer to: {q}"], q=q, a=a)
        return [Example("aok_verify", noul_q(ins), int(truth))]


class VizWiz(Source):
    """VizWiz-VQA val (photos by blind users; CC-BY-4.0), ten answers each, many "unanswerable".

    ``noul`` answerability, soft from the share of "unanswerable" votes; ``noul`` for yes/no questions, soft;
    ``choice`` for other answerable questions with distractors from rows sharing the question's first two words.
    Only the val split has answers, so it is hash-split.
    """

    name = "vizwiz"
    columns = ["question_id", "image", "question", "answers", "category"]
    repo = "lmms-lab/VizWiz-VQA"
    license = "CC-BY-4.0"
    origin = "https://huggingface.co/datasets/lmms-lab/VizWiz-VQA (VizWiz-VQA, Gurari et al. 2018)"
    train_split = "val"
    val_split = None
    ANSWERABLE = ["Can the question \"{q}\" be answered from this photo?",
                  "Is this photo clear and complete enough to answer: {q}",
                  "Someone asked: {q} Does the photo contain what is needed to answer that?"]

    def key(self, row: Dict, idx: int) -> str:
        return str(row["question_id"])

    def observe(self, row: Dict) -> None:
        votes = votes_of(row.get("answers"))
        votes.pop("unanswerable", None)
        if votes and row.get("category") == "other":
            best = votes.most_common(1)[0][0]
            self.pool.add(ScreenQA.prefix(row["question"], 2), best)
            self.pool.add("all", best)

    def ready(self, min_pool: int) -> bool:
        return self.pool.size("all") >= min(min_pool, 100)

    def examples(self, row: Dict, rng: random.Random) -> List[Example]:
        q, votes = row["question"].strip(), votes_of(row.get("answers"))
        n = sum(votes.values())
        if not q or n == 0:
            return []
        p_un = votes.get("unanswerable", 0) / n
        out = [Example("vw_answerable", noul_q(pick(rng, self.ANSWERABLE, q=q)), int(p_un < 0.5), [p_un, 1 - p_un])]
        yn = votes.get("yes", 0) + votes.get("no", 0)
        if row.get("category") == "yes/no" and yn >= 5:
            p_yes = votes.get("yes", 0) / yn
            out.append(Example("vw_yesno", noul_q(q), int(p_yes >= 0.5), [1 - p_yes, p_yes]))
        elif row.get("category") == "other" and p_un < 0.3:
            votes.pop("unanswerable", None)
            answer = votes.most_common(1)[0][0]
            if len(answer) <= 40:
                distractors = self.pool.sample(ScreenQA.prefix(q, 2), 3, rng, exclude=list(votes), fallback="all")
                if len(distractors) == 3:
                    opts = [answer] + distractors
                    rng.shuffle(opts)
                    out.append(Example("vw_choice", choice_q(q, opts), opts.index(answer), soft_target(votes, opts)))
        return out


class AVA(Source):
    """AVA aesthetic ratings (a 10% mirror with at least 50 votes per photo). Each photo has a histogram of 1-10
    ratings; collapsed to five ``score`` levels with the histogram as the soft target. The photos are from
    DPChallenge and the dataset is research-use, so keep it out of a commercial checkpoint.
    """

    name = "ava"
    columns = ["image_id", "image", "rating_counts"]
    repo = "trojblue/AVA-aesthetics-10pct-min50-10bins"
    license = "research use only (AVA, Murray et al. 2012; DPChallenge photographs)"
    origin = "https://huggingface.co/datasets/trojblue/AVA-aesthetics-10pct-min50-10bins"
    train_split = "train"
    val_split = None
    LEVELS = ["very poor", "below average", "average", "good", "excellent"]
    PHRASINGS = ["How aesthetically pleasing is this photograph?",
                 "Rate the photographic quality of this image.",
                 "How would a photography judge rate this photo?"]

    def key(self, row: Dict, idx: int) -> str:
        return str(row["image_id"])

    def examples(self, row: Dict, rng: random.Random) -> List[Example]:
        counts = list(row.get("rating_counts") or [])
        if len(counts) != 10 or sum(counts) <= 0:
            return []
        five = [counts[2 * i] + counts[2 * i + 1] for i in range(5)]
        target = [c / sum(five) for c in five]
        return [Example("ava_aesthetic", score_q(pick(rng, self.PHRASINGS), self.LEVELS),
                        max(range(5), key=lambda i: target[i]), target)]


class NLVR2(Source):
    """NLVR2 dev (two photos and a statement that is true or false of the pair). The record carries both images
    in order, so the ``noul`` asks about "the left image" and "the right image" as the statements do. Only the
    public dev/test splits have images on the Hub; the unbalanced dev split is hash-split by sentence group so
    the same statement never appears on both sides.
    """

    name = "nlvr2"
    columns = ["identifier", "question", "answer", "left_image", "right_image"]
    repo = "lmms-lab/NLVR2"
    license = "CC-BY-4.0 (NLVR2 annotations, Suhr et al. 2019; web images)"
    origin = "https://huggingface.co/datasets/lmms-lab/NLVR2 (unbalanced_dev)"
    train_split = "unbalanced_dev"
    val_split = None
    PHRASINGS = ["Two photos are shown, the left one first and then the right one. Is this statement true? {s}",
                 "Consider the first image as the left image and the second as the right image. True or false: {s}"]

    def key(self, row: Dict, idx: int) -> str:
        return str(row["identifier"])

    def images(self, row: Dict) -> List[Any]:
        return [row["left_image"], row["right_image"]]

    def examples(self, row: Dict, rng: random.Random) -> List[Example]:
        s = row["question"].strip()
        if not s or row.get("answer") not in ("True", "False"):
            return []
        return [Example("nlvr2_pair", noul_q(pick(rng, self.PHRASINGS, s=s)), int(row["answer"] == "True"))]


SOURCES = {cls.name: cls for cls in (WebSight, Screen2Words, ScreenQA, VQAv2, AOKVQA, VizWiz, AVA, NLVR2)}
SCREEN_SOURCES = ("websight", "screen2words", "screenqa")
PHOTO_SOURCES = ("vqav2", "aokvqa", "vizwiz", "ava", "nlvr2")

"""HTTP API for a loaded ``VLMAgent``: one ``POST /predict`` that takes one or more images.

The FastAPI app is built by ``make_app(agent)`` so it can run anywhere the agent can (``modal_serve.py`` puts
it behind a GPU on Modal; tests run it against a stub). The request is ``multipart/form-data``:

    images        one or more image files (JPEG, PNG, ...); at most ``MAX_IMAGES``
    questions     JSON object ``{qid: {"type", "instructions", "criteria"?}}``, the ``predict`` schema
    text          optional text state, sent alongside the image(s)
    joint         "false" (default): every image is answered on its own, one result per image, in order
                  "true": all images form one state (``{"images": [...]}``) and there is one result
    n_permutations  option orders to average for the causal backbone (see ``VLMAgent.predict``)

Multipart rather than base64 JSON keeps the upload a third smaller and lets clients stream the files
without buffering. Every image is decoded one at a time, and JPEGs are decoded at reduced scale
(``Image.draft``): the model only ever sees a ``ImagePrep.image_size`` tile, so a 12-megapixel photo never
has to exist as 36 MB of pixels.

The OpenAPI schema (``/openapi.json``) types the questions and every answer variant, so clients can be
generated from it: ``python -m laya.serve --openapi > openapi.json`` writes it without loading a model.
"""
import io
import json
import time
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

MAX_IMAGES = 16
#: JPEGs are decoded at the smallest DCT scale that still leaves at least this many pixels on each side,
#: twice the 512 tile the vision tower sees, so the final LANCZOS hop still has headroom.
DRAFT_SIZE = 1024


# ---------------------------------------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------------------------------------


class Question(BaseModel):
    """One typed question, exactly what ``VLMAgent.predict`` takes per question id."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["choice", "score", "noul"]
    instructions: Union[str, Dict[str, Any], List[Any]] = Field(
        description="The question. A non-string is serialized to JSON, as the agent does."
    )
    criteria: Optional[Union[List[str], Dict[str, Optional[str]]]] = Field(
        default=None,
        description='"choice": the options, a list of labels or {label: description}. '
        '"score": the ordered scale, lowest first. "noul": unused.',
    )


Questions = Dict[str, Question]
QUESTIONS_ADAPTER: TypeAdapter = TypeAdapter(Questions)


class ActionInfo(BaseModel):
    act_probability: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    probabilities: Dict[str, float]
    confidence: float
    action: ActionInfo


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float
    legend: Dict[str, str]
    probabilities: Dict[str, float]
    confidence: float
    action: ActionInfo


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float = Field(description="P(true)")
    confidence: float
    action: ActionInfo


Answer = Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    images: int


class Result(BaseModel):
    """``VLMAgent.predict`` output for one state."""

    model: str
    answers: Dict[str, Answer] = Field(description="Keyed by question id; discriminated on `type`.")
    usage: Usage


class PredictResponse(BaseModel):
    run: str = Field(description="The checkpoint that answered.")
    results: List[Result] = Field(description="One per image in upload order, or a single result when `joint`.")
    timing_ms: float = Field(description="Wall time spent decoding and predicting, in milliseconds.")


class Health(BaseModel):
    status: Literal["ok"]
    run: str


# ---------------------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------------------


def decode_image(data: bytes, draft_size: int = DRAFT_SIZE):
    """Bytes -> RGB PIL image. JPEGs are decoded at reduced scale when they are much larger than ``draft_size``;
    other formats decode as-is. Raises ``ValueError`` for anything Pillow can't open."""
    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(io.BytesIO(data))
        img.draft("RGB", (draft_size, draft_size))  # a no-op for anything but JPEG
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ValueError("not a decodable image: %s" % e) from e


def parse_questions(raw: str) -> Dict[str, Dict[str, Any]]:
    """Validate the JSON form field against ``Question`` and return plain dicts for the agent."""
    try:
        parsed = QUESTIONS_ADAPTER.validate_json(raw)
    except ValidationError as e:
        raise ValueError(e.errors(include_url=False)) from e
    if not parsed:
        raise ValueError("questions must contain at least one question")
    return {qid: q.model_dump(exclude_none=True) for qid, q in parsed.items()}


def warm_up(agent) -> None:
    """One throwaway prediction so the first real request doesn't pay for CUDA kernels and allocator growth."""
    from PIL import Image

    agent.predict({"image": Image.new("RGB", (64, 64), (128, 128, 128))},
                  {"warm": {"type": "noul", "instructions": "Is this a warm-up?"}})


# ---------------------------------------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------------------------------------


def make_app(agent: Any, run_name: str = "", version: str = "0.2.0", predict: Optional[Callable] = None):
    """Build the FastAPI app around ``agent`` (anything with ``.predict(state, questions, n_permutations=)``).

    ``predict`` overrides ``agent.predict``; ``agent`` may then be ``None`` (used for schema generation).
    """
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile

    predict_fn = predict or (agent.predict if agent is not None else None)

    app = FastAPI(
        title="laya-vision",
        version=version,
        description="Calibrated typed decisions (choice / score / yes-no) about one or more images.",
    )

    @app.get("/health", response_model=Health, operation_id="health")
    def health() -> Health:
        return Health(status="ok", run=run_name)

    @app.post("/predict", response_model=PredictResponse, operation_id="predict",
              summary="Answer typed questions about one or more images")
    def predict_route(  # a plain ``def``: FastAPI runs it in a worker thread, so the GPU call never blocks the loop
        images: List[UploadFile] = File(..., description="One or more image files (at most %d)." % MAX_IMAGES),
        questions: str = Form(..., description="JSON object {question_id: Question}. See the `Question` schema.",
                              json_schema_extra={"contentMediaType": "application/json",
                                                 "contentSchema": {"$ref": "#/components/schemas/Questions"}}),
        text: str = Form("", description="Optional text state sent with the image(s)."),
        joint: bool = Form(False, description="Answer once over all images as a single state instead of per image."),
        n_permutations: int = Form(1, ge=1, le=24, description="Option orders averaged per question (causal backbone only)."),
    ) -> PredictResponse:
        if predict_fn is None:
            raise HTTPException(503, "no model loaded")
        if not images:
            raise HTTPException(422, "at least one image is required")
        if len(images) > MAX_IMAGES:
            raise HTTPException(413, "at most %d images per request, got %d" % (MAX_IMAGES, len(images)))
        try:
            qs = parse_questions(questions)
        except ValueError as e:
            raise HTTPException(422, {"questions": e.args[0]})

        t0 = time.time()

        def load(i: int):
            f = images[i]
            try:
                return decode_image(f.file.read())
            except ValueError as e:
                raise HTTPException(422, "images[%d] (%s): %s" % (i, f.filename or "?", e))
            finally:
                f.file.close()

        def state_for(imgs):
            st = {"images": imgs} if len(imgs) > 1 else {"image": imgs[0]}
            if text:
                st["text"] = text
            return st

        results = []
        if joint:
            imgs = [load(i) for i in range(len(images))]  # one state needs all of them at once
            results.append(predict_fn(state_for(imgs), qs, n_permutations=n_permutations))
        else:
            for i in range(len(images)):  # one decoded image alive at a time
                results.append(predict_fn(state_for([load(i)]), qs, n_permutations=n_permutations))
        return PredictResponse(run=run_name, results=results, timing_ms=round((time.time() - t0) * 1000, 1))

    # ``Questions`` is only referenced from the form field's contentSchema, so register it explicitly
    base_openapi = app.openapi

    def openapi():
        if app.openapi_schema:
            return app.openapi_schema
        schema = base_openapi()
        comps = schema.setdefault("components", {}).setdefault("schemas", {})
        qschema = QUESTIONS_ADAPTER.json_schema(ref_template="#/components/schemas/{model}")
        comps.update(qschema.pop("$defs", {}))
        comps["Questions"] = qschema
        app.openapi_schema = schema
        return schema

    app.openapi = openapi
    return app


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="laya-vision HTTP API")
    ap.add_argument("--openapi", action="store_true", help="print the OpenAPI schema (no model needed) and exit")
    ap.add_argument("--model", default=None, help="checkpoint dir or Hub id to serve locally with uvicorn")
    ap.add_argument("--device", default=None)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    if args.openapi:
        json.dump(make_app(None).openapi(), sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    import uvicorn

    from .vlm import load_vlm

    agent = load_vlm(args.model, device=args.device)
    warm_up(agent)
    uvicorn.run(make_app(agent, run_name=args.model or "default"), host="0.0.0.0", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

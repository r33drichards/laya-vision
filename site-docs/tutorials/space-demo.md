# Try the demo

The demo is a Hugging Face Space that runs the published checkpoint,
[thaitea/laya-vision](https://huggingface.co/thaitea/laya-vision), with PyTorch on a free CPU:

**Open it: <https://huggingface.co/spaces/thaitea/laya-vision-demo>**

It runs the same `predict` call as the Python package, so its answers are the model's own. It takes a few seconds per
image; a Space that has been idle can take a minute to wake up.

## 1. Pick an image and context

Upload an image under **Image**, or click the photo of a cat under **Example** to load it.

**Optional text context** is sent with the image as part of the state, for example "customer says it arrived
broken". It changes the answers, so leave it empty unless it is part of what you are asking about.

## 2. Ask questions

**Question format** has two modes:

- **Quick**: yes/no questions (one per line), one multiple-choice question with comma-separated options, and one
  rubric-score question with its levels one per line, lowest first. Leave a field empty to skip it.
- **JSON**: any number of questions in the same shape as the second argument of
  [`predict`](../reference/predict.md): each with a `type` (`choice`, `score` or `noul`), `instructions`, and
  `criteria` for `choice` and `score`.

Press **Ask**.

## 3. Read the answers

The table shows, for each question:

- a multiple-choice answer with the probability of every option;
- a yes/no answer with P(yes);
- a rubric score as the expected level, with the probability of each level;
- the confidence (one minus the normalised entropy for choices and scores, the larger of P(yes) and P(no) for yes/no
  questions).

**Raw output** is the full `predict` result, the same schema as in Python.

The probabilities are over the options you listed: a question whose right answer is not among them still gets a
confident-looking distribution. List the answer you expect to be right, or add "other" or "none of these"; see
[Calibration](../concepts/calibration.md).

## Next

- The same questions from Python: [Your first prediction](first-prediction.md).
- The Space's source is [`space/`](https://github.com/r33drichards/laya-vision/tree/main/space); it is pushed with
  `modal run modal_app.py::publish_space`.

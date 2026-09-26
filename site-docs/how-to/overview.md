# How-to guides

These are **task-oriented** recipes for getting specific things done. They assume you have the package
[installed](../install/overview.md) or, for the GPU jobs, a Modal account. For end-to-end walkthroughs, see the
[Tutorials](../tutorials/overview.md) instead.

## Guides

- [Calibrate on your own data](calibrate.md): fit per-type temperatures on your labelled questions and check that
  they help.
- [Run jobs on Modal](run-on-modal.md): the volumes and secret `modal_app.py` expects, and the commands for data
  preparation, training, evaluation and publishing.
- [Evaluate a checkpoint](evaluate.md): run the whole evaluation suite on one checkpoint and turn the results into
  a scorecard.
- [Play games with a checkpoint](play-games.md): watch a checkpoint play in a local window, and score it on the games
  suite.
- [Serve a checkpoint over HTTP](serve.md): `laya-serve`, its endpoints, and the Python client.

## See also

- [Concepts](../concepts/overview.md): the reasoning behind these tasks.
- [Reference](../reference/overview.md): every argument, field and file format.

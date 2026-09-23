# Atari training data format

All Atari training data, whatever its source, is written to the Modal volume `laya-datasets` (workspace `r33drichards`) in one layout, so a single loader reads everything.

```
/data/atari/<source>/<Game>/
    train.jsonl
    val.jsonl
    images/<id>.png        (or .jpg)
    meta.json
    _READY                 (written last, after vol.commit(); readers only use games that have it)
```

- **`<source>`** is one of:
  - `jat`: `jat-project/jat-dataset`
  - `atari_head`: Atari-HEAD human play
  - `expert`: frames we generate with pretrained agents
- **`<Game>`** is the ALE v5 game name exactly as in `gym.make("ALE/<Game>-v5")`: `Breakout`, `MsPacman`, `SpaceInvaders`, `MontezumaRevenge`, `PrivateEye` and so on. This matches `--game` in `examples/atari_live.py`.

## Records (one JSON object per line)

```json
{
  "id": "jat-MsPacman-e000123-s000456",
  "image": "images/jat-MsPacman-e000123-s000456.png",
  "game": "MsPacman",
  "actions": ["NOOP", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"],
  "label": 3,
  "target": [0.01, 0.02, 0.05, 0.85, 0.03, 0.01, 0.01, 0.01, 0.01],
  "question": {"type": "choice", "instructions": "...", "criteria": {"NOOP": "do nothing", "...": "..."}},
  "source": "jat",
  "episode": 123,
  "step": 456
}
```

- **`actions`**: the game's minimal action set, in ALE v5 order, i.e. `gym.make("ALE/<Game>-v5").unwrapped.get_action_meanings()`. The model's options are always exactly this list.
- **`label`**: index into `actions` of the action taken. If a source records actions in the full 18-action set, map them to names with `ale_py.Action` or the full `get_action_meanings()`. Drop frames whose action isn't in the minimal set, and count them in `meta.json`.
- **`target`** (optional): a probability distribution over `actions`, the same length, summing to 1. Use it when the expert has a policy distribution; leave it out for human or one-hot data.
- **`question`**: exactly `laya.games.atari_question(game, actions)["action"]`, so training and play ask the same question.
- **`image`**: one frame, the one the action was taken on. Keep whatever the source provides and record it in `meta.json` under `frame_format`: `"rgb_210x160"` for full-colour ALE screens, or `"gray_84x84"` for preprocessed frames. Don't upscale or recolour.
- **`source`, `episode`, `step`**: provenance. Val must use **episodes disjoint from train**.

## Caps and split

- Up to **20,000 train frames and 1,000 val frames per game per source**. If a source has more, subsample evenly across episodes rather than taking the first N.
- Split train and val by episode, roughly 95/5.

## `meta.json` per game

```json
{"source": "jat", "game": "MsPacman", "frame_format": "gray_84x84", "actions": ["..."],
 "train": {"records": 20000, "episodes": 57, "labels": {"LEFT": 5123}}, "val": {"records": 1000, "episodes": 3},
 "dropped": {"not_in_minimal_set": 0}, "origin": "dataset id / URL / agent repo", "license": "..."}
```

For `expert`, also record the agent's own mean episode score and a random-policy mean score for the game, measured over at least 5 episodes each (`"expert_score"`, `"random_score"`). These are the baselines for evaluation.

## Two-frame records (source `expert2f`)

`/data/atari/expert2f/<Game>/` uses the same layout and fields, plus one:

- **`prev_image`**: `images/<id>_prev.png`, the raw RGB frame from the **previous decision step** of the same episode, 4 emulator frames earlier. That is exactly what the model will have at play time, where it keeps the last observation. On an episode's first step, and after auto-FIRE on reset or life loss, `prev_image` is a copy of `image`.

A two-frame model gets the state `{"images": [prev, current]}`, oldest first. A one-frame model ignores `prev_image`, so one dataset serves both.

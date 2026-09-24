# Play games with a checkpoint

The screen is the image and the options are the game's buttons. How the game checkpoints were trained, and how
they did: [Game training](../reference/results/game-training.md).

## Watch a checkpoint play

[`examples/atari_live.py`](https://github.com/r33drichards/laya-vision/blob/main/examples/atari_live.py) and
[`examples/vizdoom_live.py`](https://github.com/r33drichards/laya-vision/blob/main/examples/vizdoom_live.py) play in
a local window:

```bash
python examples/atari_live.py --game <Name> --model <checkpoint dir>      # any of the 104 Atari games in ale-py
python examples/vizdoom_live.py --scenario <name> --model <checkpoint dir>
```

`--model` reads the checkpoint's game frame mode (`game_frames`, such as `stack-2` or `trail-4`, else the older
`atari_frames`), so a two-frame model gets `{"images": [previous, current]}` with no extra flag (`atari_live.py
--frames 1|2` overrides); `--sample` draws from the probabilities instead of taking the top action.
Trained for 7 minutes on 20,000 auto-labelled frames, a checkpoint plays ViZDoom `basic` at expert level (mean
reward +75.4 against the expert's +75.8 over 50 unseen episodes).

## Score a checkpoint on the games suite

```bash
modal run modal_app.py::games_eval --model <run>/best
```

It plays, in one go:

- Atari Freeway, Breakout and Galaxian, at `atari_eval`'s settings, against random play and, where expert data
  exists, the expert. Galaxian has no expert data, so it is compared with random only.
- ViZDoom `basic`, against the scripted expert, random and always-attack.
- Maze, at 4×4, 6×6 and 8×8 cells: solve rate, and path efficiency against the BFS shortest path.
- Snake, on a 10×10 board: food eaten and steps survived, against a greedy BFS expert and random.
- Classic control from Gymnasium (CartPole, Acrobot, MountainCar, LunarLander), 10 episodes each: episode return
  and share solved, normalized between random play (0) and a scripted controller (1). A single frame hides
  velocity, so the screen ghosts the previous frame under the current one.

Every game is played in the checkpoint's game frame mode (`laya.frames.mode_for`: `game_frames` in its config, or
`stack-2` for Atari from an old `atari_frames: 2`, else `single`), built from each episode's own screens, and
each model result records it as `frames`. The random and expert baselines do not depend on it.

Maze and Snake are small seeded games in `laya/gridgames.py`, and the classic-control wrappers are in
`laya/controlgames.py`, so every checkpoint plays the same levels. `maze_eval`, `snake_eval` and `control_eval`
compare several checkpoints on one game.

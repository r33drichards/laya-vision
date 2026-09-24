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

`--model` reads `atari_frames` from the checkpoint, so a two-frame model gets `{"images": [previous, current]}` with
no extra flag (`--frames 1|2` overrides); `--sample` draws from the probabilities instead of taking the top action.
Trained for 7 minutes on 20,000 auto-labelled frames, a checkpoint plays ViZDoom `basic` at expert level (mean
reward +75.4 against the expert's +75.8 over 50 unseen episodes).

## Record a MuJoCo pendulum

[`laya/mujocogames.py`](https://github.com/r33drichards/laya-vision/blob/main/laya/mujocogames.py) wraps Gymnasium's
MuJoCo `InvertedPendulum-v5` and `InvertedDoublePendulum-v5` the way `laya/controlgames.py` wraps classic control:
the image is the ghosted screen, from a fixed side view, and the question is `laya.games.control_question`. MuJoCo
actions are continuous, so the model picks among named pushes: `LEFT`, `NONE`, `RIGHT` for the single pole, and
gentle and hard pushes each way for the double pole. A scripted expert (a linear controller, and an LQR rounded to the
nearest push) keeps each pole up for the full 1000 steps; random play falls within about 5.
[`examples/mujoco_video.py`](https://github.com/r33drichards/laya-vision/blob/main/examples/mujoco_video.py) records
an episode, with the chosen push and the model's probabilities beside the screen:

```bash
pip install -e . torchvision "gymnasium[mujoco]" "imageio[ffmpeg]"
python examples/mujoco_video.py --policy expert --out expert.webm
python examples/mujoco_video.py --game InvertedDoublePendulum --policy model \
    --model thaitea/laya-vision --revision 8b318c99d7ad3ce19c24369263463882eada9d1e --out model.webm
```

On a machine with no display it renders through EGL (`apt-get install libegl1`), or set `MUJOCO_GL=osmesa`
(`libosmesa6`). The pendulums are not in the games suite yet, and no checkpoint has been trained on them.

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

Maze and Snake are small seeded games in `laya/gridgames.py`, and the classic-control wrappers are in
`laya/controlgames.py`, so every checkpoint plays the same levels. `maze_eval`, `snake_eval` and `control_eval`
compare several checkpoints on one game.

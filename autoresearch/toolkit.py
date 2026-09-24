"""Fixed helpers for experiments: game training data made from the games the eval plays (``import toolkit``).

The harness puts ``autoresearch/`` on ``sys.path``, so an experiment writes ``import toolkit`` and calls these in
``build(ctx)`` (not timed). Every example has the ``laya.vlm_train.jsonl_example`` shape, with the image as PNG bytes
in the state, the question exactly ``laya.games``' (converted with ``VLMAgent._to_internal``), a ``"value"`` and,
except at an episode's last recorded step, a ``"next_target"``:

    {"state": {"image": <png bytes>}, "q": {"t", "ins", "crit"}, "target": [p per option], "label": int,
     "value": float in [0, 1], "next_target": [p per option], "dataset": "game_<name>", "id": str}

``next_target`` is the auxiliary next-move target (KataGo's opponent-move head, arXiv 1902.10565 section 3.4, for a
single agent): the target this generator gives the state at step t + 1 of the same trajectory (the state the noisy
walker actually reached), built the same way as ``target`` (soft or one-hot, smoothing) and mapped through this
example's own symmetry or mirror. It is omitted when step t + 1 is not part of the recorded trajectory (the maze
goal was reached, the snake died, the episode ended or hit its cap). A model with ``"next_head": true`` learns it
(``train(..., w_next=0.15)``); others ignore it. ``next_target=False`` leaves it out.

Lessons carried over from autogo: train on soft targets where the expert has ties, augment with the game's exact
symmetries, draw many diverse short slices of episodes rather than a few long ones, include off-expert states
(epsilon-noisy experts; the target is always the expert's answer in the state actually reached), and give a value.

Seeds. The eval plays Maze / Snake / control games on seeds ``200_000 + i`` (``GRID_SEED`` in modal_app.py) and
Atari from ``100_000``; every generator here uses seeds ``seed + i < 100_000`` and raises otherwise.

What an experiment calls::

    import toolkit
    games = (toolkit.maze_examples(20_000) + toolkit.snake_examples(20_000)
             + toolkit.control_examples("CartPole", 5_000) + ...)
    ctx.data, MIX = toolkit.game_mix(ctx.train_examples(), games, frac=0.3, base_weights=MIX)
    # then train(..., mix_weights=MIX)

Maze (``maze_examples``)
    Episodes on seeded mazes of the given sizes; half start at the usual top-left cell, half at a random floor cell
    (the model wanders off the path at eval time). The walker is the BFS expert, but with probability ``eps`` a
    uniformly random non-wall move. ``target``: uniform over every move that lies on a shortest path to the goal
    (``soft``; in these perfect mazes that is one move except at the goal, so it equals the one-hot) or one-hot on
    the BFS expert's move; ``smooth`` mixes in a uniform distribution. ``value`` = ``gamma ** d`` where ``d`` is the
    remaining shortest-path distance (in moves), or 0 when ``d`` exceeds the steps left under the eval cap
    (``4 x optimal`` from the start, counting steps already taken): the discounted probability-like "the expert
    solves it from here within the cap", shaped by how far the goal is.
Snake (``snake_examples``)
    Episodes of the BFS expert where, with probability ``eps``, it takes a uniformly random *non-fatal* move instead
    (off-expert but not suicidal, so long snakes still appear). ``target``: uniform over non-reversing moves that
    are safe for the next step and keep the BFS distance to the food optimal (the expert's own passability: body
    minus the tail); if the food is unreachable, uniform over the safe moves; with none safe, the expert's move.
    ``value`` = 1 if the (noise-free) expert, continuing from this exact state including the food RNG, is still
    alive ``horizon`` steps later (or has won), else 0.
Control (``control_examples``)
    Seeded ``laya.controlgames.ControlGame`` episodes of the epsilon-noisy scripted expert (random action with prob.
    ``eps``, per game in ``CONTROL_EPS``). Images are ``ControlGame.render()`` exactly: the frame at step t with the
    frame at t-1 ghosted in (step 0 has no ghost, like the eval). Only a ``keep`` fraction of steps
    (``CONTROL_KEEP``) is rendered and kept. ``target``: one-hot on the expert's action in that state,
    label-smoothed by ``smooth``. ``value`` with ``gamma = 1 - 1/VALUE_HORIZON`` and ``k`` steps from this frame to
    the episode's end: CartPole (survive to the time limit): 1 if the pole never falls, else ``1 - gamma ** k``;
    Acrobot / MountainCar / LunarLander (reach the goal / land): ``gamma ** k`` if the episode ends in success
    (terminated; for LunarLander terminated by landing, reward +100), else 0. A Monte Carlo sample under the noisy
    expert.

Symmetries (``flip``): Maze and Snake use all 8 symmetries of the square (mirror left-right, up-down, transpose),
the actions mapped by moving each move's vector; CartPole, Acrobot and LunarLander a left-right mirror of the image
with LEFT/RIGHT, CLOCKWISE/COUNTERCLOCKWISE and LEFT_ENGINE/RIGHT_ENGINE swapped. MountainCar has none (the flag
is on the right).

Generation runs in ``workers`` forked processes (children only use numpy/PIL/gymnasium, never CUDA); the output
depends only on the arguments, not on ``workers``.
"""
import io
import math
import os
import random
import sys
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.isdir(os.path.join(_REPO, "laya")) and _REPO not in sys.path:
    sys.path.append(_REPO)  # ``import toolkit`` from autoresearch/ still finds the laya package

EVAL_SEED_FLOOR = 100_000   # every eval seed range starts at or above this (Atari 100_000, grid/control 200_000)
PNG_LEVEL = 1               # zlib level: flat game screens compress well even at the fastest setting

GRID_ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
_VEC = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
_OPP = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

CONTROL_FLIPS = {"CartPole": {"LEFT": "RIGHT"}, "Acrobot": {"CLOCKWISE": "COUNTERCLOCKWISE"},
                 "LunarLander": {"LEFT_ENGINE": "RIGHT_ENGINE"}}
VALUE_HORIZON = {"CartPole": 100, "Acrobot": 100, "MountainCar": 100, "LunarLander": 200}
_SURVIVAL = {"CartPole"}
# noise and the fraction of steps kept, per game: CartPole survives eps=0.3 and runs 500 steps, so keep few frames
# of many episodes; LunarLander mostly crashes at eps=0.2, so less noise there
CONTROL_EPS = {"CartPole": 0.3, "Acrobot": 0.2, "MountainCar": 0.2, "LunarLander": 0.1}
CONTROL_KEEP = {"CartPole": 0.03, "Acrobot": 0.1, "MountainCar": 0.08, "LunarLander": 0.05}


# -- shared ---------------------------------------------------------------------------------------------------------


def _png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=PNG_LEVEL)
    return buf.getvalue()


def _check_seeds(seed: int, episodes: int) -> None:
    if seed < 0 or seed + episodes > EVAL_SEED_FLOOR:
        raise ValueError("training seeds %d..%d overlap the eval range (keep seed + episodes <= %d)"
                         % (seed, seed + episodes - 1, EVAL_SEED_FLOOR))


def _internal(qdef: Dict) -> Dict:
    from laya.vlm import VLMAgent

    return VLMAgent._to_internal(qdef["action"])


def _smooth(target: Sequence[float], smooth: float) -> List[float]:
    k = len(target)
    return [float((1 - smooth) * p + smooth / k) for p in target]


def sym_action(action: str, sym: int) -> str:
    """Where a grid move goes under symmetry ``sym`` (0..7; bit 0 transpose, bit 1 flip up-down, bit 2 flip
    left-right, applied in that order)."""
    dr, dc = _VEC[action]
    if sym & 1:
        dr, dc = dc, dr
    if sym & 2:
        dr = -dr
    if sym & 4:
        dc = -dc
    return next(a for a, v in _VEC.items() if v == (dr, dc))


def sym_grid(grid: np.ndarray, sym: int) -> np.ndarray:
    """Apply symmetry ``sym`` (see ``sym_action``) to an ``(n, n, ...)`` grid."""
    if sym & 1:
        grid = grid.swapaxes(0, 1)
    if sym & 2:
        grid = grid[::-1]
    if sym & 4:
        grid = grid[:, ::-1]
    return np.ascontiguousarray(grid)


def sym_pos(pos: Tuple[int, int], n: int, sym: int) -> Tuple[int, int]:
    r, c = pos
    if sym & 1:
        r, c = c, r
    if sym & 2:
        r = n - 1 - r
    if sym & 4:
        c = n - 1 - c
    return r, c


def sym_target(target: Sequence[float], sym: int) -> List[float]:
    out = [0.0] * 4
    for i, a in enumerate(GRID_ACTIONS):
        out[GRID_ACTIONS.index(sym_action(a, sym))] = float(target[i])
    return out


def _grid_png(grid: np.ndarray, sym: int) -> bytes:
    """``laya.gridgames._render(grid)`` after symmetry ``sym`` as a palette PNG: the same RGB pixels once loaded
    (``_load_image`` converts to RGB), about 5x faster to encode and 3x smaller than an RGB PNG."""
    from PIL import Image

    from laya.gridgames import _cell_px

    g = sym_grid(grid, sym)
    px = _cell_px(g.shape[0])
    colors, idx = np.unique(g.reshape(-1, 3), axis=0, return_inverse=True)
    small = idx.reshape(g.shape[:2]).astype(np.uint8)
    im = Image.fromarray(np.ascontiguousarray(small.repeat(px, 0).repeat(px, 1)), "P")
    im.putpalette(colors.astype(np.uint8).tobytes())
    return _png(im)


def _bfs_dist(passable, src: Tuple[int, int], shape: Tuple[int, int]) -> Dict[Tuple[int, int], int]:
    dist = {src: 0}
    q = deque([src])
    while q:
        r, c = q.popleft()
        d = dist[(r, c)] + 1
        for dr, dc in _VEC.values():
            nxt = (r + dr, c + dc)
            if 0 <= nxt[0] < shape[0] and 0 <= nxt[1] < shape[1] and nxt not in dist and passable(*nxt):
                dist[nxt] = d
                q.append(nxt)
    return dist


def _run(fn, args_list: List[tuple], workers: int) -> List:
    """``[fn(*a) for a in args_list]``, in order, over ``workers`` forked processes when that is worth it."""
    workers = min(workers or (os.cpu_count() or 1), len(args_list))
    if workers <= 1:
        return [fn(*a) for a in args_list]
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    try:
        ctx = mp.get_context("fork")
    except ValueError:  # no fork (Windows): serial
        return [fn(*a) for a in args_list]
    with ProcessPoolExecutor(workers, mp_context=ctx) as ex:
        return list(ex.map(_star, [(fn, a) for a in args_list]))


def _star(fa):
    fn, a = fa
    return fn(*a)


def _generate(episode_fn, n: int, seed: int, cfg: Dict, workers: int, per_ep: float) -> List[Dict]:
    """Run episodes ``0, 1, ...`` in rounds until ``n`` examples; keep the first ``n`` in episode order. The first
    round uses the ``per_ep`` guess for half the need, later rounds the yield measured so far (little overshoot)."""
    out: List[Dict] = []
    ep, chunk = 0, 8
    while len(out) < n:
        need = n - len(out)
        rate = len(out) / ep if ep else per_ep
        n_eps = max(1, math.ceil(need / max(0.5, rate) * (0.5 if not ep else 1.02)))
        _check_seeds(seed, ep + n_eps)
        size = max(1, min(chunk, math.ceil(n_eps / max(1, workers or (os.cpu_count() or 1)))))
        jobs = [(list(range(s, min(s + size, ep + n_eps))), seed, cfg) for s in range(ep, ep + n_eps, size)]
        for part in _run(episode_fn, jobs, workers):
            out.extend(part)
        ep += n_eps
    return out[:n]


# -- Maze -----------------------------------------------------------------------------------------------------------


def maze_optimal_moves(wall: np.ndarray, pos: Tuple[int, int], goal: Tuple[int, int]) -> List[str]:
    """Every move from ``pos`` that starts a shortest path to ``goal`` (empty at the goal)."""
    dist = _bfs_dist(lambda r, c: not wall[r, c], goal, wall.shape)
    return _maze_moves(wall, dist, pos)


def _maze_moves(wall, dist, pos) -> List[str]:
    if pos not in dist or dist[pos] == 0:
        return []
    return [a for a in GRID_ACTIONS
            if dist.get((pos[0] + _VEC[a][0], pos[1] + _VEC[a][1]), -9) == dist[pos] - 1]


def _maze_grid(wall, pos, goal) -> np.ndarray:
    from laya.gridgames import AGENT, FLOOR, GOAL, WALL

    grid = np.where(wall[..., None], np.array(WALL, np.uint8), np.array(FLOOR, np.uint8)).astype(np.uint8)
    grid[goal] = GOAL
    grid[pos] = AGENT
    return grid


def _maze_episodes(eps_ids: List[int], seed: int, cfg: Dict) -> List[Dict]:
    from laya.gridgames import Maze

    out = []
    for i in eps_ids:
        rng = random.Random("maze-%d-%d" % (seed, i))
        maze = Maze(rng.choice(cfg["sizes"]), seed + i)
        wall, goal = maze.wall, maze.goal
        dist = _bfs_dist(lambda r, c: not wall[r, c], goal, wall.shape)
        pos = maze.pos
        if rng.random() < cfg["random_start"]:
            floor = sorted(p for p in dist if p != goal)
            pos = rng.choice(floor)
        cap = maze.max_steps  # 4 x optimal from the usual start, as the eval
        visited, steps = [], 0
        while pos != goal and steps < cap:
            if not visited or visited[-1][0] != pos:
                visited.append((pos, steps))
            opt = _maze_moves(wall, dist, pos)
            if rng.random() < cfg["eps"]:
                legal = [a for a in GRID_ACTIONS if not wall[pos[0] + _VEC[a][0], pos[1] + _VEC[a][1]]]
                a = rng.choice(legal)
            else:
                a = opt[0]
            pos = (pos[0] + _VEC[a][0], pos[1] + _VEC[a][1])
            steps += 1
        at = {t: p for p, t in visited}  # step -> position along the walk actually taken
        keep = visited if len(visited) <= cfg["per_episode"] else rng.sample(visited, cfg["per_episode"])
        for pos, t in keep:
            target, expert = _maze_target(wall, dist, pos, cfg["soft"])
            sym = rng.randrange(8) if cfg["flip"] else 0
            d = dist[pos]
            ex = {"state": {"image": _grid_png(_maze_grid(wall, pos, goal), sym)},
                  "target": _smooth(sym_target(target, sym), cfg["smooth"]),
                  "label": GRID_ACTIONS.index(sym_action(expert, sym)),
                  "id": "maze-%d-n%d-r%d-c%d-t%d-g%d" % (seed + i, maze.size, pos[0], pos[1], t, sym)}
            if cfg["value"]:
                ex["value"] = float(cfg["gamma"] ** d) if d <= cap - t else 0.0
            if cfg["next_target"] and t + 1 in at:  # the walk's next state (none at the goal or past the cap)
                ex["next_target"] = _smooth(sym_target(_maze_target(wall, dist, at[t + 1], cfg["soft"])[0], sym),
                                            cfg["smooth"])
            out.append(ex)
    return out


def _maze_target(wall, dist, pos, soft: bool) -> Tuple[List[float], str]:
    """(target over ``GRID_ACTIONS``, the BFS expert's move) at ``pos``, before any symmetry or smoothing."""
    opt = _maze_moves(wall, dist, pos)
    expert = opt[0]  # BFS in ACTIONS order: the same move Maze.expert() makes
    if soft:
        return [1.0 / len(opt) if a in opt else 0.0 for a in GRID_ACTIONS], expert
    return [float(a == expert) for a in GRID_ACTIONS], expert


def maze_examples(n: int, sizes: Sequence[int] = (4, 6), seed: int = 0, soft: bool = True, flip: bool = True,
                  value: bool = True, eps: float = 0.3, per_episode: int = 8, random_start: float = 0.5,
                  gamma: float = 0.97, smooth: float = 0.0, workers: int = 0,
                  dataset: str = "game_maze", next_target: bool = True) -> List[Dict]:
    """``n`` Maze examples from seeds ``seed, seed + 1, ...`` (must stay below 100_000); see the module docstring."""
    from laya.games import maze_question

    cfg = dict(sizes=tuple(sizes), soft=soft, flip=flip, value=value, eps=eps, per_episode=per_episode,
               random_start=random_start, gamma=gamma, smooth=smooth, next_target=next_target)
    out = _generate(_maze_episodes, n, seed, cfg, workers, per_episode * 0.9)
    q = _internal(maze_question())
    for ex in out:
        ex["q"], ex["dataset"] = q, dataset
    return out


# -- Snake ----------------------------------------------------------------------------------------------------------


def _snake_clone(s):
    from laya.gridgames import Snake

    c = Snake.__new__(Snake)
    c.__dict__.update(s.__dict__)
    c.body = deque(s.body)
    c.rng = random.Random()
    c.rng.setstate(s.rng.getstate())
    return c


def snake_moves(s) -> Tuple[List[str], List[str]]:
    """(optimal, safe): non-reversing moves that survive the next step, and those of them that keep the BFS
    distance to the food shortest through the expert's passable cells (the body minus its tail)."""
    safe = []
    for a in GRID_ACTIONS:
        if a == _OPP[s.heading]:
            continue
        _, pos = s._next(a)
        if not s._fatal(pos, pos == s.food):
            safe.append(a)
    if s.food is None or not safe:
        return [], safe
    body = set(list(s.body)[:-1])
    dist = _bfs_dist(lambda r, c: (r, c) not in body, s.food, (s.size, s.size))
    head = s.body[0]
    ds = {a: dist.get((head[0] + _VEC[a][0], head[1] + _VEC[a][1])) for a in safe}
    ds = {a: d for a, d in ds.items() if d is not None}
    if not ds:
        return [], safe
    best = min(ds.values())
    return [a for a in safe if ds.get(a) == best], safe


def _snake_value(s, horizon: int) -> float:
    c = _snake_clone(s)
    for _ in range(horizon):
        if c.dead is not None:
            break
        c.step(c.expert())
    return 1.0 if c.dead in (None, "won") else 0.0


def _snake_grid(s) -> np.ndarray:
    from laya.gridgames import BODY, FOOD, HEAD, WALL, FLOOR

    n = s.size + 2
    grid = np.empty((n, n, 3), dtype=np.uint8)
    grid[:] = WALL
    grid[1:-1, 1:-1] = FLOOR
    if s.food is not None:
        grid[s.food[0] + 1, s.food[1] + 1] = FOOD
    for k, (r, c) in enumerate(s.body):
        if 0 <= r < s.size and 0 <= c < s.size:
            grid[r + 1, c + 1] = HEAD if k == 0 else BODY
    return grid


def _snake_episodes(eps_ids: List[int], seed: int, cfg: Dict) -> List[Dict]:
    from laya.gridgames import Snake

    out = []
    for i in eps_ids:
        rng = random.Random("snake-%d-%d" % (seed, i))
        s = Snake(rng.choice(cfg["sizes"]), seed + i)
        snaps = []
        while s.dead is None and s.steps < cfg["episode_cap"]:
            snaps.append(_snake_clone(s))
            a = s.expert()
            if rng.random() < cfg["eps"]:
                _, safe = snake_moves(s)
                a = rng.choice(safe) if safe else rng.choice(GRID_ACTIONS)
            s.step(a)
        step_of = {id(st): j for j, st in enumerate(snaps)}
        keep = snaps if len(snaps) <= cfg["per_episode"] else rng.sample(snaps, cfg["per_episode"])
        for st in keep:
            target, expert = _snake_target(st, cfg["soft"])
            sym = rng.randrange(8) if cfg["flip"] else 0
            ex = {"state": {"image": _grid_png(_snake_grid(st), sym)},
                  "target": _smooth(sym_target(target, sym), cfg["smooth"]),
                  "label": GRID_ACTIONS.index(sym_action(expert, sym)),
                  "id": "snake-%d-t%d-g%d" % (seed + i, st.steps, sym)}
            if cfg["value"]:
                ex["value"] = _snake_value(st, cfg["horizon"])
            j = step_of[id(st)] + 1
            if cfg["next_target"] and j < len(snaps):  # the next recorded state (none after death or the cap)
                ex["next_target"] = _smooth(sym_target(_snake_target(snaps[j], cfg["soft"])[0], sym), cfg["smooth"])
            out.append(ex)
    return out


def _snake_target(s, soft: bool) -> Tuple[List[float], str]:
    """(target over ``GRID_ACTIONS``, the expert's move) in snake state ``s``, before any symmetry or smoothing."""
    expert = s.expert()
    if soft:
        opt, safe = snake_moves(s)
        support = opt or safe or [expert]
        return [1.0 / len(support) if a in support else 0.0 for a in GRID_ACTIONS], expert
    return [float(a == expert) for a in GRID_ACTIONS], expert


def snake_examples(n: int, sizes: Sequence[int] = (8, 10), seed: int = 0, soft: bool = True, flip: bool = True,
                   value: bool = True, eps: float = 0.2, per_episode: int = 12, horizon: int = 30,
                   episode_cap: int = 600, smooth: float = 0.0, workers: int = 0,
                   dataset: str = "game_snake", next_target: bool = True) -> List[Dict]:
    """``n`` Snake examples from seeds ``seed, seed + 1, ...`` (below 100_000); see the module docstring."""
    from laya.games import snake_question

    cfg = dict(sizes=tuple(sizes), soft=soft, flip=flip, value=value, eps=eps, per_episode=per_episode,
               horizon=horizon, episode_cap=episode_cap, smooth=smooth, next_target=next_target)
    out = _generate(_snake_episodes, n, seed, cfg, workers, per_episode * 0.9)
    q = _internal(snake_question())
    for ex in out:
        ex["q"], ex["dataset"] = q, dataset
    return out


# -- classic control ------------------------------------------------------------------------------------------------


def _control_success(game: str, env, last_reward: float) -> bool:
    if game in _SURVIVAL:
        return not env.terminated
    if game == "LunarLander":
        return env.terminated and last_reward == 100.0  # landed at rest (crash / out of bounds give -100)
    return env.terminated


def _control_episodes(eps_ids: List[int], seed: int, cfg: Dict) -> List[Dict]:
    from PIL import Image

    from laya.controlgames import ControlGame

    game, out = cfg["game"], []
    actions = None
    flips = CONTROL_FLIPS.get(game, {})
    flips = {**flips, **{v: k for k, v in flips.items()}}
    gamma = 1.0 - 1.0 / VALUE_HORIZON[game]
    for i in eps_ids:
        rng = random.Random("control-%s-%d-%d" % (game, seed, i))
        env = ControlGame(game, seed + i)
        actions = env.actions
        recs, experts = [], []  # experts[t]: the expert's action at every step t, kept or not
        keep_now = rng.random() < cfg["keep"]
        reward = 0.0
        while not env.done:
            keep_next = rng.random() < cfg["keep"]
            expert = env.expert()
            experts.append(expert)
            if keep_now:
                img = env.render()  # ghosts the previous frame if it was rendered (it was, see keep_next)
                recs.append((env.steps, expert, img))
            elif keep_next:
                env.frame()  # the next kept frame ghosts this one, as in the every-step eval loop
            a = rng.choice(actions) if rng.random() < cfg["eps"] else expert
            reward = env.step(a)
            keep_now = keep_next
        success, end = _control_success(game, env, reward), env.steps
        # no env.close(): pygame.quit() costs ~30 ms an episode, and the renderer is only a Surface here
        for t, expert, img in recs:
            k = end - t
            if game in _SURVIVAL:
                v = 1.0 if success else 1.0 - gamma ** k
            else:
                v = gamma ** k if success else 0.0
            flip = bool(flips) and cfg["flip"] and rng.random() < 0.5
            label = expert
            if flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                label = flips.get(expert, expert)
            target = _smooth([float(a == label) for a in actions], cfg["smooth"])
            ex = {"state": {"image": _png(img)}, "target": target, "label": actions.index(label),
                  "id": "%s-%d-t%d%s" % (game.lower(), seed + i, t, "-f" if flip else "")}
            if cfg["value"]:
                ex["value"] = float(v)
            if cfg["next_target"] and t + 1 < end:  # the expert's action at step t + 1 (none at the last step)
                nxt = flips.get(experts[t + 1], experts[t + 1]) if flip else experts[t + 1]
                ex["next_target"] = _smooth([float(a == nxt) for a in actions], cfg["smooth"])
            out.append(ex)
    return out


def control_examples(game: str, n: int, seed: int = 0, eps: Optional[float] = None, keep: Optional[float] = None,
                     smooth: float = 0.1,
                     flip: bool = True, value: bool = True, workers: int = 0,
                     dataset: Optional[str] = None, next_target: bool = True) -> List[Dict]:
    """``n`` examples of a ``laya.controlgames`` game from seeds ``seed, seed + 1, ...`` (below 100_000); see the
    module docstring. ``eps`` / ``keep`` default per game (``CONTROL_EPS`` / ``CONTROL_KEEP``). Needs
    ``gymnasium[classic-control,box2d]`` (imported lazily)."""
    from laya.controlgames import GAMES
    from laya.games import control_question

    if game not in GAMES:
        raise ValueError("unknown control game %r (%s)" % (game, ", ".join(GAMES)))
    eps = CONTROL_EPS[game] if eps is None else eps
    keep = CONTROL_KEEP[game] if keep is None else keep
    mean_len = {"CartPole": 500, "Acrobot": 100, "MountainCar": 140, "LunarLander": 220}[game]
    per_ep = mean_len * keep
    cfg = dict(game=game, eps=eps, keep=keep, smooth=smooth, flip=flip, value=value, next_target=next_target)
    out = _generate(_control_episodes, n, seed, cfg, workers, per_ep)
    q = _internal(control_question(game))
    for ex in out:
        ex["q"], ex["dataset"] = q, dataset or "game_" + game.lower()
    return out


# -- Atari / ViZDoom records ------------------------------------------------------------------------------------------


def examples_from_records(records: List[Dict], root: str, dataset: str) -> List[Dict]:
    """Records in the ``docs/atari-data-format.md`` layout (also ``/data/vqa/doom_basic``) -> examples, through
    ``laya.vlm_train.jsonl_example`` (soft ``target`` kept). ``root`` is the directory the ``image`` paths are
    relative to; pool the files into memory yourself (the volume is slow per file) and pass records whose
    ``image`` you then replace, or read lazily from ``root``."""
    from laya.vlm_train import jsonl_example

    out = [jsonl_example(r, root, dataset) for r in records]
    return [ex for ex in out if ex is not None]


# -- mixing with the VQA pool ---------------------------------------------------------------------------------------


def game_mix(ctx_examples: List[Dict], game_examples: List[Dict], frac: float,
             base_weights: Optional[Dict[str, float]] = None,
             alpha: float = 0.0) -> Tuple[List[Dict], Dict[str, float]]:
    """Combine the VQA pool with game examples; returns ``(examples, mix_weights)`` for ``train(...,
    mix_weights=..., mix_alpha=alpha)`` so the game datasets together are drawn ``frac`` of the time (split
    equally between them) and the VQA datasets share the rest in the proportions ``base_weights`` / ``alpha`` gave
    them. Game examples without a ``"dataset"`` get ``"game"``; names not starting with ``"game_"`` are prefixed."""
    from laya.vlm_train import group_weights

    if not 0.0 <= frac <= 1.0:
        raise ValueError("frac must be in [0, 1]")
    games = []
    for ex in game_examples:
        name = ex.get("dataset") or "game"
        if not name.startswith("game"):
            name = "game_" + name
        games.append(dict(ex, dataset=name) if name != ex.get("dataset") else ex)
    groups_v: Dict[str, list] = {}
    for ex in ctx_examples:
        groups_v.setdefault(ex.get("dataset", "_"), []).append(ex)
    groups_g: Dict[str, list] = {}
    for ex in games:
        groups_g.setdefault(ex["dataset"], []).append(ex)
    clash = set(groups_v) & set(groups_g)
    if clash:
        raise ValueError("game dataset names clash with the VQA pool: %s" % sorted(clash))
    wv = group_weights(groups_v, base_weights, alpha)
    sv = sum(wv.values())
    if not groups_g or frac == 0.0:
        return list(ctx_examples), {k: v for k, v in (base_weights or {}).items() if k in groups_v}
    if not groups_v or sv == 0 or frac == 1.0:  # games only (a zero weight would still leave a group in the mix)
        return games, {k: 1.0 / len(groups_g) / float(len(g)) ** alpha for k, g in groups_g.items()}
    weights: Dict[str, float] = {}
    # group_weights multiplies by n ** alpha, so divide it out to land on the requested shares
    for k, v in wv.items():
        weights[k] = (1.0 - frac) * v / sv / float(len(groups_v[k])) ** alpha
    for k, g in groups_g.items():
        weights[k] = frac / len(groups_g) / float(len(g)) ** alpha
    return list(ctx_examples) + games, weights


__all__ = ["maze_examples", "snake_examples", "control_examples", "examples_from_records", "game_mix",
           "maze_optimal_moves", "snake_moves", "sym_action", "sym_grid", "sym_pos", "sym_target",
           "EVAL_SEED_FLOOR", "GRID_ACTIONS", "CONTROL_FLIPS", "CONTROL_EPS", "CONTROL_KEEP", "VALUE_HORIZON"]

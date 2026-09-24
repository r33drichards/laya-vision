"""Maze and Snake: small seeded grid games for closed-loop evaluation of the image model as a policy.

Both are plain Python (numpy and PIL only), so they run anywhere the model does, and every checkpoint plays the
same levels from the same seeds. Each step the screen is rendered to an RGB image and the model answers one
``choice`` over ``ACTIONS`` (``laya.games.maze_question`` / ``snake_question``); a scripted expert and a
random policy on the same seeds are the reference points.

* **Maze**: a perfect maze (one path between any two cells) of ``size`` x ``size`` cells, carved by a seeded
  depth-first search. The agent starts top-left, the goal is bottom-right; bumping a wall wastes the step. An
  episode ends at the goal or after ``max_steps`` (default 4x the shortest path). The expert follows the BFS
  shortest path. Reported: solve rate, and path efficiency (shortest / taken) on solved mazes. ``size`` is the
  difficulty ladder.
* **Snake**: a ``size`` x ``size`` board with walls around it. The snake starts 3 long heading right; eating
  food grows it by one; hitting a wall or its own body ends the episode, as does going ``size**2`` steps without
  food (so a looping policy does not run forever). Turning straight back into the neck is ignored (the snake keeps
  going), as in most Snake games. The head is drawn darker than the body, so one frame shows the heading. The
  expert takes the BFS shortest path to the food through free cells, else any move that survives the next step.
  Reported: food eaten per episode and steps survived.
"""
import random
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import frames as F

ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
MOVES = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

WALL, FLOOR = (30, 30, 30), (245, 245, 245)
AGENT, GOAL = (40, 90, 220), (40, 170, 70)
HEAD, BODY, FOOD = (20, 100, 40), (90, 190, 90), (220, 50, 50)

Pos = Tuple[int, int]


def _render(grid: np.ndarray, cell_px: int) -> "Image.Image":
    """An ``(h, w, 3)`` uint8 colour grid -> a PIL image with each grid cell ``cell_px`` pixels square."""
    from PIL import Image

    return Image.fromarray(np.kron(grid, np.ones((cell_px, cell_px, 1), dtype=np.uint8)))


def _cell_px(cells: int, target: int = 384) -> int:
    return max(4, target // cells)


def bfs_path(passable, start: Pos, goal: Pos, shape: Tuple[int, int]) -> Optional[List[str]]:
    """The shortest list of moves from ``start`` to ``goal`` through cells where ``passable(r, c)``; ``None`` when
    unreachable. Neighbours are tried in ``ACTIONS`` order, so the path is deterministic."""
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            path = []
            while prev[cur] is not None:
                cur, a = prev[cur]
                path.append(a)
            return path[::-1]
        for a in ACTIONS:
            dr, dc = MOVES[a]
            nxt = (cur[0] + dr, cur[1] + dc)
            if 0 <= nxt[0] < shape[0] and 0 <= nxt[1] < shape[1] and nxt not in prev and passable(*nxt):
                prev[nxt] = (cur, a)
                q.append(nxt)
    return None


# -- Maze ---------------------------------------------------------------------------------------------------------


def carve_maze(size: int, rng: random.Random) -> np.ndarray:
    """A perfect maze of ``size`` x ``size`` cells as a ``(2*size+1)`` square boolean grid, True = wall.
    Cell ``(i, j)`` is grid ``(2i+1, 2j+1)``; iterative depth-first search, so any size is safe."""
    n = 2 * size + 1
    wall = np.ones((n, n), dtype=bool)
    seen = {(0, 0)}
    stack = [(0, 0)]
    wall[1, 1] = False
    while stack:
        i, j = stack[-1]
        nbrs = [(i + di, j + dj) for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if 0 <= i + di < size and 0 <= j + dj < size and (i + di, j + dj) not in seen]
        if not nbrs:
            stack.pop()
            continue
        ni, nj = rng.choice(nbrs)
        wall[2 * ni + 1, 2 * nj + 1] = False
        wall[i + ni + 1, j + nj + 1] = False  # the wall between the two cells
        seen.add((ni, nj))
        stack.append((ni, nj))
    return wall


class Maze:
    """One seeded maze episode. ``step(action)`` returns True once the goal is reached."""

    def __init__(self, size: int = 6, seed: int = 0, max_steps: int = 0):
        self.size = size
        self.wall = carve_maze(size, random.Random(seed))
        n = self.wall.shape[0]
        self.pos, self.goal = (1, 1), (n - 2, n - 2)
        self.optimal = len(self.expert_path())
        self.max_steps = max_steps or 4 * self.optimal
        self.steps, self.bumps, self.solved = 0, 0, False

    @property
    def done(self) -> bool:
        return self.solved or self.steps >= self.max_steps

    def expert_path(self) -> List[str]:
        return bfs_path(lambda r, c: not self.wall[r, c], self.pos, self.goal, self.wall.shape)

    def expert(self) -> str:
        return self.expert_path()[0]

    def step(self, action: str) -> bool:
        dr, dc = MOVES[action]
        nxt = (self.pos[0] + dr, self.pos[1] + dc)
        if self.wall[nxt]:
            self.bumps += 1
        else:
            self.pos = nxt
        self.steps += 1
        self.solved = self.pos == self.goal
        return self.solved

    def render(self, cell_px: int = 0):
        grid = np.where(self.wall[..., None], np.array(WALL, np.uint8), np.array(FLOOR, np.uint8)).astype(np.uint8)
        grid[self.goal] = GOAL
        grid[self.pos] = AGENT
        return _render(grid, cell_px or _cell_px(self.wall.shape[0]))


# -- Snake --------------------------------------------------------------------------------------------------------


class Snake:
    """One seeded Snake episode on a ``size`` x ``size`` board (walls drawn around it)."""

    def __init__(self, size: int = 10, seed: int = 0, max_steps: int = 0):
        self.size, self.rng = size, random.Random(seed)
        mid = size // 2
        self.body = deque([(mid, mid), (mid, mid - 1), (mid, mid - 2)])  # head first
        self.heading = "RIGHT"
        self.food = self._place_food()
        self.max_steps = max_steps or 50 * size * size
        self.steps, self.eaten, self.since_food = 0, 0, 0
        self.dead: Optional[str] = None

    @property
    def done(self) -> bool:
        return self.dead is not None or self.steps >= self.max_steps

    def _place_food(self) -> Optional[Pos]:
        free = [(r, c) for r in range(self.size) for c in range(self.size) if (r, c) not in self.body]
        return self.rng.choice(free) if free else None

    def _next(self, action: str) -> Tuple[str, Pos]:
        if action == OPPOSITE[self.heading]:
            action = self.heading
        dr, dc = MOVES[action]
        head = self.body[0]
        return action, (head[0] + dr, head[1] + dc)

    def _fatal(self, pos: Pos, grows: bool) -> Optional[str]:
        if not (0 <= pos[0] < self.size and 0 <= pos[1] < self.size):
            return "wall"
        tail_moves_away = not grows
        body = list(self.body)[:-1] if tail_moves_away else list(self.body)
        return "self" if pos in body else None

    def step(self, action: str) -> None:
        action, head = self._next(action)
        self.heading = action
        grows = head == self.food
        self.steps += 1
        self.dead = self._fatal(head, grows)
        if self.dead:
            return
        self.body.appendleft(head)
        if grows:
            self.eaten += 1
            self.since_food = 0
            self.food = self._place_food()
            if self.food is None:
                self.dead = "won"
        else:
            self.body.pop()
            self.since_food += 1
            if self.since_food >= self.size * self.size:
                self.dead = "starved"

    def expert(self) -> str:
        """BFS to the food through cells the body will have left by then (the tail vacates as the snake moves),
        approximated as: free cells plus the tail; else any move that survives the next step."""
        body = set(list(self.body)[:-1])
        passable = lambda r, c: (r, c) not in body  # noqa: E731
        if self.food is not None:
            path = bfs_path(passable, self.body[0], self.food, (self.size, self.size))
            if path and path[0] != OPPOSITE[self.heading]:
                return path[0]
        for a in (self.heading,) + ACTIONS:
            if a == OPPOSITE[self.heading]:
                continue
            _, pos = self._next(a)
            if not self._fatal(pos, pos == self.food):
                return a
        return self.heading

    def render(self, cell_px: int = 0):
        n = self.size + 2
        grid = np.empty((n, n, 3), dtype=np.uint8)
        grid[:] = WALL
        grid[1:-1, 1:-1] = FLOOR
        if self.food is not None:
            grid[self.food[0] + 1, self.food[1] + 1] = FOOD
        for k, (r, c) in enumerate(self.body):
            if 0 <= r < self.size and 0 <= c < self.size:
                grid[r + 1, c + 1] = HEAD if k == 0 else BODY
        return _render(grid, cell_px or _cell_px(n))


# -- episodes -----------------------------------------------------------------------------------------------------


def make_game(game: str, size: int, seed: int, max_steps: int = 0):
    if game == "maze":
        return Maze(size or 6, seed, max_steps)
    if game == "snake":
        return Snake(size or 10, seed, max_steps)
    raise ValueError("unknown grid game %r (maze or snake)" % game)


def play_episodes(game: str, policy, episodes: int, size: int = 0, seed: int = 0, max_steps: int = 0) -> Dict:
    """Play ``episodes`` seeded episodes (seed ``seed + i``); ``policy(env) -> action`` sees the live game (a model
    policy only renders it). Returns per-episode results and the game's summary metrics."""
    from collections import Counter

    counts, eps = Counter(), []
    for i in range(episodes):
        env = make_game(game, size, seed + i, max_steps)
        while not env.done:
            a = policy(env)
            counts[a] += 1
            env.step(a)
        if game == "maze":
            eps.append({"solved": env.solved, "steps": env.steps, "optimal": env.optimal, "bumps": env.bumps})
        else:
            eps.append({"eaten": env.eaten, "steps": env.steps, "end": env.dead or "capped"})
    out = {"game": game, "size": size or (6 if game == "maze" else 10), "episodes": episodes, "seed": seed,
           "actions": dict(counts), "results": eps}
    if game == "maze":
        solved = [e for e in eps if e["solved"]]
        out.update(solve_rate=len(solved) / episodes,
                   efficiency=float(np.mean([e["optimal"] / e["steps"] for e in solved])) if solved else 0.0,
                   mean_steps=float(np.mean([e["steps"] for e in eps])))
    else:
        out.update(mean_eaten=float(np.mean([e["eaten"] for e in eps])), max_eaten=max(e["eaten"] for e in eps),
                   mean_steps=float(np.mean([e["steps"] for e in eps])),
                   ends=dict(Counter(e["end"] for e in eps)))
    return out


def expert_policy(env) -> str:
    return env.expert()


def random_policy(seed: int = 0):
    rng = random.Random(seed)
    return lambda env: rng.choice(ACTIONS)


def model_policy(agent, game: str, mode: str = "single"):
    """The model's most likely action from the rendered screen, via ``predict`` like the live viewers.

    ``mode`` is the checkpoint's game frame mode (``laya.frames.mode_for(agent.cfg, "grid")``): ``single`` sends
    the rendered screen alone, exactly as before modes existed; any other mode sends ``laya.frames.state`` of the
    episode's screens at its decision points (``episode_policy``)."""
    from laya.games import maze_question, snake_question

    q = maze_question() if game == "maze" else snake_question()
    if mode == "single":
        return lambda env: agent.predict({"image": env.render()}, q)["answers"]["action"]["choice"]
    return F.episode_policy(lambda env, st: agent.predict(st, q)["answers"]["action"]["choice"],
                            lambda env: env.render(), mode, "grid")


__all__ = ["ACTIONS", "Maze", "Snake", "carve_maze", "bfs_path", "make_game", "play_episodes", "expert_policy",
           "random_policy", "model_policy"]

"""Maze and Snake (``laya.gridgames``): mechanics, experts and rendering, no model."""
import random

import numpy as np
import pytest

from laya.games import maze_question, snake_question
from laya.gridgames import (ACTIONS, AGENT, BODY, FOOD, GOAL, HEAD, Maze, Snake, bfs_path, carve_maze, expert_policy,
                            play_episodes, random_policy)
from laya.vlm import VLMAgent


@pytest.mark.parametrize("size", [2, 5, 12])
def test_maze_is_perfect(size):
    wall = carve_maze(size, random.Random(size))
    n = 2 * size + 1
    assert wall.shape == (n, n) and wall[0].all() and wall[-1].all() and wall[:, 0].all() and wall[:, -1].all()
    open_cells = int((~wall).sum())
    # a spanning tree over size^2 cells: size^2 cells plus size^2 - 1 opened walls, and every cell reachable
    assert open_cells == 2 * size * size - 1
    for i in range(size):
        for j in range(size):
            assert bfs_path(lambda r, c: not wall[r, c], (1, 1), (2 * i + 1, 2 * j + 1), wall.shape) is not None


def test_maze_is_seeded_and_the_expert_solves_it_optimally():
    assert (Maze(6, seed=3).wall == Maze(6, seed=3).wall).all() and (Maze(6, seed=3).wall != Maze(6, seed=4).wall).any()
    res = play_episodes("maze", expert_policy, episodes=10, size=6, seed=0)
    assert res["solve_rate"] == 1.0 and res["efficiency"] == 1.0
    assert all(e["steps"] == e["optimal"] and e["bumps"] == 0 for e in res["results"])


def test_maze_walls_block_and_the_step_cap_ends_it():
    m = Maze(4, seed=0)
    blocked = next(a for a in ACTIONS if m.wall[m.pos[0] + {"UP": -1, "DOWN": 1}.get(a, 0),
                                                  m.pos[1] + {"LEFT": -1, "RIGHT": 1}.get(a, 0)])
    start = m.pos
    m.step(blocked)
    assert m.pos == start and m.bumps == 1 and m.steps == 1
    while not m.done:
        m.step(blocked)
    assert not m.solved and m.steps == m.max_steps == 4 * m.optimal
    res = play_episodes("maze", random_policy(0), episodes=20, size=8, seed=0)
    assert res["solve_rate"] < 0.5  # an 8x8 maze in 4x the shortest path is out of a random walk's reach


def test_maze_render_colours():
    m = Maze(3, seed=0)
    im = np.asarray(m.render(cell_px=4))
    assert im.shape == (7 * 4, 7 * 4, 3)
    assert tuple(im[1 * 4 + 1, 1 * 4 + 1]) == AGENT and tuple(im[5 * 4 + 1, 5 * 4 + 1]) == GOAL


def test_snake_moves_eats_and_grows():
    s = Snake(10, seed=0)
    head = s.body[0]
    s.food = (head[0], head[1] + 1)
    s.step("RIGHT")
    assert s.eaten == 1 and len(s.body) == 4 and s.body[0] == (head[0], head[1] + 1) and s.dead is None
    s.step("LEFT")  # straight back into the neck: ignored, the snake keeps going right
    assert s.body[0] == (head[0], head[1] + 2) and s.heading == "RIGHT"


def test_snake_dies_on_walls_and_itself():
    s = Snake(6, seed=0)
    while not s.done:
        s.step("RIGHT")
    assert s.dead == "wall"
    s = Snake(6, seed=0)
    s.food = None
    # head (3, 3) heading right, the body curls under it: moving down hits (4, 3), which is not the tail
    s.body = type(s.body)([(3, 3), (3, 2), (4, 2), (4, 3), (4, 4)])
    s.heading = "RIGHT"
    s.step("DOWN")
    assert s.dead == "self"
    # the tail itself is safe to move into: it leaves as the head arrives
    s = Snake(6, seed=0)
    s.food = None
    s.body = type(s.body)([(3, 3), (3, 2), (4, 2), (4, 3)])
    s.heading = "RIGHT"
    s.step("DOWN")
    assert s.dead is None and s.body[0] == (4, 3)


def test_snake_starves_when_it_loops():
    s = Snake(6, seed=0)
    s.food = (0, 0)
    loop = ["DOWN", "LEFT", "UP", "RIGHT"]
    k = 0
    while not s.done:
        s.step(loop[k % 4])
        k += 1
    assert s.dead == "starved" and s.steps == 36


def test_snake_expert_beats_random():
    exp = play_episodes("snake", expert_policy, episodes=5, size=8, seed=0)
    rnd = play_episodes("snake", random_policy(0), episodes=5, size=8, seed=0)
    assert exp["mean_eaten"] >= 10 and rnd["mean_eaten"] < 3


def test_snake_render_colours():
    s = Snake(4, seed=0)
    im = np.asarray(s.render(cell_px=2))
    assert im.shape == (6 * 2, 6 * 2, 3)
    (hr, hc), (br, bc) = s.body[0], s.body[1]
    assert tuple(im[(hr + 1) * 2, (hc + 1) * 2]) == HEAD and tuple(im[(br + 1) * 2, (bc + 1) * 2]) == BODY
    fr, fc = s.food
    assert tuple(im[(fr + 1) * 2, (fc + 1) * 2]) == FOOD


def test_questions_offer_the_four_moves():
    for q in (maze_question(), snake_question()):
        internal = VLMAgent._to_internal(q["action"])
        assert internal["t"] == "choice" and list(internal["crit"]) == list(ACTIONS)

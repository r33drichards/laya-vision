"""Watch Laya Vision play a ViZDoom scenario live, on your own machine.

Each step the current screen goes into the model as the image, and a `choice` question over the scenario's
buttons picks the action (held for 4 tics). The window shows the game next to the model's probabilities.

    pip install -e . torchvision vizdoom pygame
    python examples/vizdoom_live.py                        # "basic": shoot the monster in front of you
    python examples/vizdoom_live.py --scenario defend_the_center --device cpu
    python examples/vizdoom_live.py --device cuda --cuda-graph --dtype bf16   # docs/game-caching.md

Scenarios: basic, defend_the_center, defend_the_line, health_gathering, take_cover, predict_position,
deadly_corridor, my_way_home. Keys: SPACE pause/resume, R new episode, ESC or close the window to quit.
Zero-shot: the model was trained on photo/diagram questions, not games.
"""
import argparse
import os
import time
from collections import Counter

import pygame
import torch
import vizdoom as vzd
from PIL import Image

import laya
from laya.games import doom_buttons, doom_question
from laya.static_step import StaticStep
from atari_live import PANEL_W, draw

SCALE = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="basic")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--model", default="thaitea/laya-vision-smolvlm-256m")
    ap.add_argument("--tics", type=int, default=4, help="game tics each chosen action is held for")
    ap.add_argument("--steps", type=int, default=0, help="quit after this many steps in total (0 = run until closed)")
    ap.add_argument("--dtype", choices=("fp32", "bf16"), default=None,
                    help="weights dtype (default: the checkpoint's); bf16 makes the vision tower ~4.6x faster on a GPU")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="run each decision as one captured CUDA graph (same answer, 3-5x faster at batch 1 in bf16 "
                         "on an L4; see docs/game-caching.md). Runs eagerly, with no speedup, off CUDA")
    args = ap.parse_args()

    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, args.scenario + ".cfg"))
    game.set_window_visible(False)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_seed(0)
    game.init()
    buttons = doom_buttons(game)
    one_hot = {b: [i == j for j in range(len(buttons))] for i, b in enumerate(buttons)}

    print("loading %s on %s ..." % (args.model, args.device))
    agent = laya.load_vlm(args.model, device=args.device, dtype=args.dtype)
    qs = doom_question(args.scenario, buttons)
    static = StaticStep(agent, qs["action"]) if args.cuda_graph else None
    print("buttons:", buttons)

    pygame.init()
    game.new_episode()
    obs = game.get_state().screen_buffer
    screen = pygame.display.set_mode((obs.shape[1] * SCALE + PANEL_W, max(obs.shape[0] * SCALE, 560)))
    pygame.display.set_caption("Laya Vision plays Doom: %s" % args.scenario)
    fonts = [pygame.font.SysFont("menlo,monospace", s) for s in (26, 18, 15)]

    ep, step, total, best, ms, ans, paused = 1, 0, 0, float("-inf"), 0.0, None, False
    counts = Counter()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                paused = not paused
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                game.new_episode(); step = 0
                obs = game.get_state().screen_buffer
        score = game.get_total_reward()
        if paused:
            draw(screen, fonts, obs, "Doom: " + args.scenario, ep, step, score, max(best, 0), ans, ms, counts, True, SCALE)
            time.sleep(0.05)
            continue
        t0 = time.perf_counter()
        if static is not None:
            ans = static.answer(obs)
        else:
            ans = agent.predict({"image": Image.fromarray(obs)}, qs)["answers"]["action"]
        ms = 0.8 * ms + 0.2 * (time.perf_counter() - t0) * 1000 if ms else (time.perf_counter() - t0) * 1000
        counts[ans["choice"]] += 1
        game.make_action(one_hot[ans["choice"]], args.tics)
        step, total = step + 1, total + 1
        if game.is_episode_finished():
            score = game.get_total_reward()
            best = max(best, score)
            print("episode %d: reward %.1f in %d steps" % (ep, score, step))
            game.new_episode(); ep, step = ep + 1, 0
        obs = game.get_state().screen_buffer
        draw(screen, fonts, obs, "Doom: " + args.scenario, ep, step, game.get_total_reward(),
             best if best > float("-inf") else 0, ans, ms, counts, False, SCALE)
        if args.steps and total >= args.steps:
            running = False
    print("steps %d, %.0f ms/step, actions %s" % (total, ms, dict(counts)))
    game.close()
    pygame.quit()


if __name__ == "__main__":
    main()

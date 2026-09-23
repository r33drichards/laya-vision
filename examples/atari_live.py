"""Watch Laya Vision play an Atari game live, on your own machine.

Each step the current screen goes into the model as the image, and a `choice` question over the game's
actions picks the move. The window shows the game next to the model's action probabilities.

    pip install -e . torchvision ale-py gymnasium pygame
    python examples/atari_live.py                         # Breakout on Apple GPU (mps) if available, else CPU
    python examples/atari_live.py --device cpu --game Pong
    python examples/atari_live.py --model checkpoints/atari-dag2f-rlcd            # a trained two-frame model
    python examples/atari_live.py --model checkpoints/atari-8g-2f-512gpu --game Boxing   # device-side path
    python examples/atari_live.py --sample                # draw the action from the probabilities
    python examples/atari_live.py --device cuda --cuda-graph --dtype bf16   # site-docs/concepts/game-caching.md

Keys: SPACE pause/resume, R restart episode, ESC or close the window to quit.
By default FIRE is pressed automatically at the start of each game and after each lost life (the standard
Atari `FireResetEnv` trick), because the model doesn't know Breakout waits for FIRE. Use --no-auto-fire to
leave every move to the model.
With `--frames 2` the model also sees the screen at the previous decision, the way the game-trained
two-frame checkpoints were trained (`expert2f`): the previous frame is a copy of the current one on an
episode's first step and on the first step after an auto-FIRE, exactly as in `play_atari`. `--frames 0` (the
default) uses whatever the checkpoint itself was trained with (`atari_frames` in its config, 1 if it does not
say), and likewise its own `image_size` / `preprocess`, so a device-side checkpoint plays on that path here too.
The action is the most likely one, or with `--sample` drawn from the model's calibrated probabilities.
The default model is zero-shot: it was trained on photo/diagram questions, not games.
"""
import argparse
import time
from collections import Counter

import ale_py
import gymnasium as gym
import numpy as np
import pygame
import torch

import laya
from laya.games import atari_question
from laya.static_step import StaticStep

SCALE, PANEL_W = 3, 420
BG, FG, DIM, ACCENT, BAR = (18, 18, 24), (235, 235, 240), (140, 140, 155), (255, 196, 64), (80, 120, 220)


def draw(screen, fonts, obs, game, ep, step, score, best, ans, ms, counts, paused, scale=SCALE):
    big, mid, small = fonts
    screen.fill(BG)
    frame = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
    screen.blit(pygame.transform.scale(frame, (obs.shape[1] * scale, obs.shape[0] * scale)), (0, 0))
    x, y = obs.shape[1] * scale + 20, 16
    screen.blit(big.render(game, True, FG), (x, y)); y += 40
    for line in ("episode %d   step %d" % (ep, step), "score %.0f   best %.0f" % (score, best),
                 "%.0f ms / step  (%.1f steps/s)" % (ms, 1000 / max(ms, 1e-3))):
        screen.blit(mid.render(line, True, DIM), (x, y)); y += 26
    y += 14
    screen.blit(mid.render("PAUSED (space)" if paused else "model's action probabilities", True, ACCENT if paused else FG), (x, y)); y += 32
    if ans:
        col = max(mid.size(a)[0] for a in ans["probabilities"]) + 12
        for a, p in sorted(ans["probabilities"].items(), key=lambda kv: -kv[1]):
            chosen = a == ans["choice"]
            screen.blit(mid.render(a, True, ACCENT if chosen else FG), (x, y))
            w = int((PANEL_W - col - 70) * p)
            pygame.draw.rect(screen, ACCENT if chosen else BAR, (x + col, y + 4, max(w, 1), 16))
            screen.blit(small.render("%.0f%%" % (100 * p), True, DIM), (x + col + 6 + w, y + 4))
            y += 28
        y += 8
        screen.blit(small.render("confidence %.2f" % ans["confidence"], True, DIM), (x, y)); y += 34
    screen.blit(mid.render("actions taken", True, FG), (x, y)); y += 28
    total = max(1, sum(counts.values()))
    for a, c in counts.most_common():
        screen.blit(small.render("%-10s %5d  (%.0f%%)" % (a, c, 100 * c / total), True, DIM), (x, y)); y += 20
    pygame.display.flip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="Breakout")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--model", default="thaitea/laya-vision-smolvlm-256m")
    ap.add_argument("--max-steps", type=int, default=5000, help="per episode (the model may never press FIRE)")
    ap.add_argument("--steps", type=int, default=0, help="quit after this many steps in total (0 = run until closed)")
    ap.add_argument("--no-auto-fire", action="store_true", help="don't press FIRE on reset / after a lost life")
    ap.add_argument("--frames", type=int, choices=(0, 1, 2), default=0,
                    help="frames the model sees: 1 (current), 2 (previous + current, the expert2f rule), "
                         "or 0 to use what the checkpoint was trained with (default)")
    ap.add_argument("--sample", action="store_true",
                    help="draw the action from the model's probabilities instead of taking the most likely one")
    ap.add_argument("--seed", type=int, default=0, help="episode and sampling seed")
    ap.add_argument("--dtype", choices=("fp32", "bf16"), default=None,
                    help="weights dtype (default: the checkpoint's); bf16 makes the vision tower ~4.6x faster on a GPU")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="run each decision as one captured CUDA graph (same answer, 3-5x faster at batch 1 in bf16 "
                         "on an L4; see site-docs/concepts/game-caching.md). Runs eagerly, with no speedup, off CUDA")
    args = ap.parse_args()

    gym.register_envs(ale_py)
    env = gym.make("ALE/%s-v5" % args.game)
    actions = env.unwrapped.get_action_meanings()
    print("loading %s on %s ..." % (args.model, args.device))
    agent = laya.load_vlm(args.model, device=args.device, dtype=args.dtype)
    n_frames = args.frames or int(agent.cfg.get("atari_frames", 1))
    print("%d frame(s) per decision, %s action" % (n_frames, "sampled" if args.sample else "top"))
    qs = atari_question(args.game, actions)
    static = StaticStep(agent, qs["action"], frames=n_frames) if args.cuda_graph else None
    rng = np.random.default_rng(args.seed)

    pygame.init()
    auto_fire = "FIRE" in actions and not args.no_auto_fire

    def reset(**kw):
        o, info = env.reset(**kw)
        return (fire(o, info) if auto_fire else (o, info.get("lives", 0)))

    def fire(o, info):
        o, _, _, _, info = env.step(actions.index("FIRE"))
        return o, info.get("lives", 0)

    obs, lives = reset(seed=args.seed)
    prev = obs  # the previous decision's screen; a copy of the current one on the first step (the expert2f rule)
    screen = pygame.display.set_mode((obs.shape[1] * SCALE + PANEL_W, obs.shape[0] * SCALE))
    pygame.display.set_caption("Laya Vision plays %s" % args.game)
    fonts = [pygame.font.SysFont("menlo,monospace", s) for s in (26, 18, 15)]

    ep, step, total, score, best, ms, ans, paused = 1, 0, 0, 0.0, 0.0, 0.0, None, False
    counts = Counter()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                running = False
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE:
                paused = not paused
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                obs, lives = reset(); prev = obs; step, score = 0, 0.0
        if paused:
            draw(screen, fonts, obs, args.game, ep, step, score, best, ans, ms, counts, True)
            time.sleep(0.05)
            continue
        t0 = time.perf_counter()
        # the raw uint8 observation goes in as-is: both preprocessing paths take it, and on the GPU path
        # this avoids a PIL round-trip that as_uint8_chw would only undo
        if static is not None:
            ans = static.answer(obs, prev if n_frames == 2 else None)
        else:
            state = {"images": [prev, obs]} if n_frames == 2 else {"image": obs}
            ans = agent.predict(state, qs)["answers"]["action"]
        if args.sample:  # the panel highlights the action actually taken
            names, p = zip(*ans["probabilities"].items())
            p = np.asarray(p, dtype=float)
            ans = dict(ans, choice=str(rng.choice(names, p=p / p.sum())))
        ms = 0.8 * ms + 0.2 * (time.perf_counter() - t0) * 1000 if ms else (time.perf_counter() - t0) * 1000
        counts[ans["choice"]] += 1
        prev = obs  # the screen this decision was made on
        obs, r, term, trunc, info = env.step(actions.index(ans["choice"]))
        if auto_fire and not (term or trunc) and info.get("lives", lives) < lives:
            obs, _ = fire(obs, info)
            prev = obs  # after an auto-FIRE the previous frame is a copy of the current one
        lives = info.get("lives", lives)
        score, step, total = score + r, step + 1, total + 1
        draw(screen, fonts, obs, args.game, ep, step, score, best, ans, ms, counts, False)
        if term or trunc or step >= args.max_steps:
            best = max(best, score)
            print("episode %d: score %.0f in %d steps (%s)" % (ep, score, step, "game over" if term or trunc else "step cap"))
            obs, lives = reset(); prev = obs; ep, step, score = ep + 1, 0, 0.0
        if args.steps and total >= args.steps:
            running = False
    print("steps %d, %.0f ms/step, actions %s" % (total, ms, dict(counts)))
    pygame.quit()


if __name__ == "__main__":
    main()

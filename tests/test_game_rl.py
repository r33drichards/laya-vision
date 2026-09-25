"""RL from game rewards (``laya.game_rl``): advantages, lockstep rollout bookkeeping, the policy-gradient direction on
a toy bandit, seeds below 100,000, and RL-off leaving ``train()`` and the experiment unchanged. The stub tests run on
the CPU with no downloads; the last ones load SmolVLM-256M like tests/test_vlm.py."""
import math
import os
import sys

import numpy as np
import pytest
import torch

from laya import frames as F
from laya import game_rl as R

HERE = os.path.dirname(os.path.abspath(__file__))
AUTORESEARCH = os.path.join(os.path.dirname(HERE), "autoresearch")


# -- advantages ---------------------------------------------------------------------------------------------------


def test_group_advantages_normalise_within_the_group():
    a = R.group_advantages([1.0, 2.0, 3.0])
    assert np.allclose(a, np.array([-1.0, 0.0, 1.0]) / math.sqrt(2 / 3))
    assert np.allclose(R.group_advantages([5.0, 5.0, 5.0]), 0.0)  # no spread, no signal
    b = R.group_advantages([10.0, 20.0, 30.0])  # the scale of a game's score cancels
    assert np.allclose(a, b) and np.allclose(R.group_advantages([-201.0, -200.0, -199.0]), a)


def test_returns_to_go():
    assert np.allclose(R.returns_to_go([1.0, 1.0, 1.0], 0.5), [1.75, 1.5, 1.0])
    assert np.allclose(R.returns_to_go([0.0, 0.0, 2.0], 1.0), [2.0, 2.0, 2.0])
    assert R.returns_to_go([], 0.9).shape == (0,)


def test_togo_advantages_compare_the_group_at_the_same_step():
    # episode 0 lasts 3 steps (+1 each), episode 1 crashes after 1: after its end its return-to-go is 0
    adv = R.togo_advantages([[1.0, 1.0, 1.0], [1.0]], gamma=1.0)
    G = np.array([[3.0, 2.0, 1.0], [1.0, 0.0, 0.0]])
    C = G - G.mean(0)
    s = math.sqrt(np.mean(np.concatenate([C[0], C[1, :1]]) ** 2))
    assert np.allclose(adv[0], C[0] / s) and np.allclose(adv[1], C[1, :1] / s)
    assert (adv[0] > 0).all() and adv[1][0] < 0
    # costs per step (Acrobot, MountainCar): the episode that reached the goal first has the better return-to-go
    adv = R.togo_advantages([[-1.0, -1.0, -1.0], [-1.0]], gamma=1.0)
    assert adv[1][0] > 0 and (adv[0] < 0).all()
    assert all(np.allclose(a, 0) for a in R.togo_advantages([[1.0, 2.0], [1.0, 2.0]], 0.9))


def test_advantages_are_per_group_and_per_step():
    E = R.Episode
    eps = [E("A", 1, 0, rewards=[1.0, 1.0]), E("A", 1, 0, rewards=[0.0]), E("A", 2, 1, rewards=[5.0]),
           E("A", 2, 1, rewards=[5.0]), E("B", 1, 0, rewards=[0.0, 0.0, 3.0]), E("B", 1, 0, rewards=[1.0])]
    adv = R.advantages(eps, "episode")
    assert [len(a) for a in adv] == [2, 1, 1, 1, 3, 1]
    assert np.allclose(adv[0], 1.0) and np.allclose(adv[1], -1.0)   # group (A, 0)
    assert np.allclose(adv[2], 0.0) and np.allclose(adv[3], 0.0)    # group (A, 1): a tie
    assert np.allclose(adv[4], 1.0) and np.allclose(adv[5], -1.0)   # (B, 0) is its own group, not (A, 0)
    togo = R.advantages(eps, "togo", 1.0)
    assert [len(a) for a in togo] == [2, 1, 1, 1, 3, 1]
    with pytest.raises(ValueError):
        R.advantages(eps, "sum")


# -- rollouts -----------------------------------------------------------------------------------------------------


class RecordingPolicy:
    """Deterministic stub: action ``(call + position) % n`` per live env; records every state it is shown."""

    def __init__(self, n_actions):
        self.n, self.calls = n_actions, []

    def act(self, states):
        self.calls.append([(g, [np.array(x) for x in imgs]) for g, imgs in states])
        c = len(self.calls)
        acts = [(c + j) % self.n[g] for j, (g, _) in enumerate(states)]
        return acts, [math.log(0.5)] * len(states), [(g, len(imgs)) for g, imgs in states]


@pytest.mark.parametrize("mode", ["single", "stack-2", "trail-3"])
def test_lockstep_rollout_bookkeeping(mode):
    caps = {"Maze4": 5, "CartPole": 7}
    make = lambda name, seed: R.make_env(name, seed, caps[name])  # noqa: E731
    policy = RecordingPolicy({"Maze4": 4, "CartPole": 2})
    groups = [("Maze4", 11), ("CartPole", 12), ("Maze4", 13)]
    modes = {"Maze4": mode, "CartPole": mode}
    eps = R.rollout(policy, groups, 3, modes, make)
    assert len(eps) == 9 and [ep.group for ep in eps] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert [ep.seed for ep in eps] == [11] * 3 + [12] * 3 + [13] * 3
    for ep in eps:  # one action, log-prob, input and reward per step, and the game's own step count
        assert len(ep.actions) == len(ep.logps) == len(ep.inputs) == len(ep.rewards) == ep.steps > 0
        assert ep.steps <= caps[ep.game] and not ep.cut
    # done handling: each round the policy saw exactly the envs still running
    live = [sum(ep.steps > r for ep in eps) for r in range(max(ep.steps for ep in eps))]
    assert [len(c) for c in policy.calls] == live
    # every recorded state is this episode's own history in the mode: replay each episode alone
    pos = {i: 0 for i in range(len(eps))}
    per_ep = {i: [] for i in range(len(eps))}
    for r, call in enumerate(policy.calls):
        running = [i for i, ep in enumerate(eps) if ep.steps > r]
        for i, st in zip(running, call):
            per_ep[i].append(st)
            pos[i] += 1
    for i, ep in enumerate(eps):
        env = make(ep.game, ep.seed)
        hist = F.History(F.frames_needed(mode, env.family))
        for t, a in enumerate(ep.actions):
            hist.push(env.frame())
            want = [np.asarray(x) for x in F.state_images(F.state(hist.frames, mode, env.family))]
            got = per_ep[i][t][1]
            assert len(got) == len(want) and all(np.array_equal(x, y) for x, y in zip(got, want))
            env.step(env.actions[a])
        assert env.score == ep.score and np.isclose(sum(ep.rewards), ep.score)
        env.close()
    # the same seed gives every episode of a group the same start; different seeds differ
    first = {i: per_ep[i][0][1] for i in per_ep}
    for g in range(3):
        a, b, c = first[3 * g], first[3 * g + 1], first[3 * g + 2]
        assert all(np.array_equal(x, y) for x, y in zip(a, b)) and all(np.array_equal(x, y) for x, y in zip(a, c))
    assert not np.array_equal(first[0][-1], first[6][-1])
    if mode == "stack-2":  # the first step repeats the first frame
        assert np.array_equal(first[0][0], first[0][1])


def test_rollout_stops_at_the_deadline():
    make = lambda name, seed: R.make_env(name, seed, 50)  # noqa: E731
    eps = R.rollout(RecordingPolicy({"CartPole": 2}), [("CartPole", 3)], 2, {"CartPole": "single"}, make,
                    deadline=0.0)
    assert all(ep.cut and ep.steps == 0 for ep in eps)


def test_atari_env_matches_play_semantics():
    pytest.importorskip("ale_py")
    env = R.make_env("Breakout", 5, cap=30)
    assert env.actions[1] == "FIRE" and env.frame().shape == (210, 160, 3)
    n = 0
    while not env.done:
        env.step("NOOP")
        n += 1
    assert n == env.steps == 30
    env.close()


# -- seeds --------------------------------------------------------------------------------------------------------


def test_training_seeds_stay_below_the_benchmark():
    rl = R.GameRL(None, R.RLConfig(games=("Maze4", "CartPole"), group=4, episodes_per_phase=32),
                  policy=RecordingPolicy({}), params=[], modes={"Maze4": "single", "CartPole": "single"})
    seeds = [s for _ in range(200) for _, s in rl.groups()]
    assert len(seeds) == 200 * 16 and all(10_000 <= s < R.TRAIN_SEED_MAX == 100_000 for s in seeds)
    for bad in (100_000, 700_000, -1):
        with pytest.raises(ValueError):
            R.check_seed(bad)
    with pytest.raises(ValueError):
        R.make_env("Maze4", 730_000)
    with pytest.raises(ValueError):
        R.rollout(RecordingPolicy({"Maze4": 4}), [("Maze4", 700_000)], 1, {"Maze4": "single"})
    with pytest.raises(ValueError):
        R.GameRL(None, R.RLConfig(games=("Maze4",), seed_lo=100_000), policy=RecordingPolicy({}), params=[],
                 modes={"Maze4": "single"})


def test_games_and_caps_match_the_benchmark():
    sys.path.insert(0, AUTORESEARCH)
    import games_eval as ge

    for name, g in R.GAMES.items():
        spec = ge.SUITE[name]
        assert (g.family, g.cap, g.params) == (spec.family, spec.cap, spec.params)
        if g.family in ("grid", "control"):
            assert R.question_for(name, "stack-2") == ge.question_for(spec, "stack-2")
    assert min(s.seed for s in ge.SUITE.values()) >= R.TRAIN_SEED_MAX


# -- the loss on a toy bandit -------------------------------------------------------------------------------------


class Bandit:
    """One step, two actions; B pays 1, A pays 0."""
    family = "grid"
    actions = ("A", "B")

    def __init__(self):
        self.steps, self.score = 0, 0.0

    @property
    def done(self):
        return self.steps >= 1

    def frame(self):
        return np.zeros((4, 4, 3), np.uint8)

    def step(self, a):
        self.steps += 1
        r = float(a == "B")
        self.score += r
        return r


class TinyPolicy(torch.nn.Module):
    """A tiny model: logits = W x + b on a constant input, sampled like ModelPolicy."""

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.lin = torch.nn.Linear(3, 2)
        self.gen = torch.Generator().manual_seed(seed)

    def scaled_logits(self, inputs, model=None):
        return (model or self).lin(torch.ones(len(inputs), 3))

    @torch.no_grad()
    def act(self, states):
        z = self.scaled_logits(states)
        logp = torch.log_softmax(z, -1)
        a = torch.multinomial(logp.exp(), 1, generator=self.gen).squeeze(1)
        return a.tolist(), logp.gather(1, a[:, None]).squeeze(1).tolist(), [g for g, _ in states]

    def p_b(self):
        with torch.no_grad():
            return float(torch.softmax(self.scaled_logits([0]), -1)[0, 1])


def bandit_rl(policy, **kw):
    cfg = R.RLConfig(games=("bandit",), group=8, episodes_per_phase=32, lr=0.05, minibatch=16, **kw)
    return R.GameRL(None, cfg, policy=policy, params=list(policy.parameters()), modes={"bandit": "single"},
                    make=lambda name, seed: Bandit(), log=lambda s: None)


def test_pg_loss_gradient_raises_the_better_action():
    z = torch.zeros(4, 2, requires_grad=True)
    acts, adv = torch.tensor([1, 0, 1, 0]), torch.tensor([1.0, -1.0, 1.0, -1.0])
    loss, st = R.pg_loss(z, acts, adv, torch.log(torch.full((4,), 0.5)))
    loss.backward()
    assert (z.grad[:, 1] < 0).all() and (z.grad[:, 0] > 0).all()  # descent raises logit B
    assert st["clip_frac"] == 0 and math.isclose(st["entropy"], math.log(2), rel_tol=1e-6)
    # the PPO clip: a ratio already past 1 + clip gets no more push from a positive advantage
    z2 = torch.tensor([[0.0, 2.0]], requires_grad=True)
    loss, st = R.pg_loss(z2, torch.tensor([1]), torch.tensor([1.0]), torch.log(torch.tensor([0.5])), clip=0.2)
    loss.backward()
    assert st["clip_frac"] == 1 and torch.all(z2.grad == 0)
    # the KL term alone pulls toward the reference
    z3 = torch.tensor([[2.0, 0.0]], requires_grad=True)
    loss, st = R.pg_loss(z3, torch.tensor([0]), torch.tensor([0.0]), torch.tensor([0.0]),
                         ref_logits=torch.zeros(1, 2), kl=1.0)
    loss.backward()
    assert st["kl_start"] > 0 and z3.grad[0, 0] > 0 > z3.grad[0, 1]


@pytest.mark.parametrize("returns", ["episode", "togo"])
def test_grpo_learns_the_bandit(returns):
    policy = TinyPolicy()
    before = policy.p_b()
    rl = bandit_rl(policy, returns=returns)
    rl.run(minutes=1, max_phases=6)
    assert len(rl.history) == 6 and all(h["update_steps"] > 0 for h in rl.history[:1])
    assert policy.p_b() > max(0.9, before + 0.3)
    assert rl.history[-1]["games"]["bandit"]["mean_score"] > rl.history[0]["games"]["bandit"]["mean_score"]
    s = rl.summary()
    assert s["episodes"] == 6 * 32 and 0 < s["rollout_frac"] < 1 and s["episodes_per_s"] > 0


def test_kl_holds_the_policy_near_the_start():
    import copy

    free, held = TinyPolicy(), TinyPolicy()
    ref = copy.deepcopy(held)
    bandit_rl(free).run(minutes=1, max_phases=4)
    rl = bandit_rl(held, kl=5.0)
    rl.ref = ref
    rl.run(minutes=1, max_phases=4)
    assert "kl_start" in rl.history[-1]
    assert 0.5 < held.p_b() < free.p_b()


def test_hook_runs_phases_on_schedule_and_within_the_deadline():
    import time

    rl = bandit_rl(TinyPolicy(), every=3)
    ran = [s for s in range(1, 10) if rl.hook(s, {"lr_scale": 0.5, "elapsed_s": 1.0, "deadline": time.time() + 60})]
    assert ran == [3, 6, 9] and all(h["lr"] == 0.05 * 0.5 for h in rl.history)
    assert not rl.hook(12, {"deadline": time.time() - 1})  # no time left: skipped
    share = bandit_rl(TinyPolicy(), every=0.25)
    assert share.hook(1, {"elapsed_s": 10.0}) and not share.hook(2, {"elapsed_s": 1e-6})


# -- RL off leaves the experiment unchanged -----------------------------------------------------------------------


def test_rl_off_calls_train_exactly_as_before(monkeypatch):
    sys.path.insert(0, AUTORESEARCH)
    import experiment as exp

    import laya.vlm_train as vt

    got = {}
    monkeypatch.setattr(vt, "train", lambda *a, **kw: got.update(args=a, kw=kw))

    class Agent:
        model, processor = object(), object()

    class Ctx:
        data, mix, time_budget_s, device = [1], {"x": 1.0}, 900, "cpu"

    assert exp.RL_GAMES == ()
    exp.train(Agent(), Ctx())
    assert set(got["kw"]) == {"steps", "batch_size", "freeze", "lr_head", "lr_backbone", "warmup", "mix_weights",
                              "w_next", "max_minutes", "num_workers", "log_every", "device"}
    monkeypatch.setattr(exp, "RL_GAMES", ("CartPole",))
    monkeypatch.setattr(exp, "rl_trainer", lambda agent: type("T", (), {"hook": "the hook"})())
    exp.train(Agent(), Ctx())
    assert got["kw"]["step_hook"] == "the hook"


# -- with the real model (downloads SmolVLM-256M) -------------------------------------------------------------------

BACKBONE = "HuggingFaceTB/SmolVLM-256M-Instruct"


@pytest.fixture(scope="module")
def agent():
    from laya.vlm import VLMAgent

    torch.manual_seed(0)
    a = VLMAgent(backbone=BACKBONE, device="cpu", game_frames="stack-2")
    a.temperature = [2.0, 2.0, 2.0]
    return a


def test_train_with_an_idle_hook_is_unchanged():
    from laya.vlm import VLMAgent
    from laya.vlm_train import synthetic_examples, train

    exs, calls = synthetic_examples(2), []
    runs = []
    for hook in (None, lambda step, ctx: calls.append((step, sorted(ctx))) and False):
        torch.manual_seed(1)
        a = VLMAgent(backbone=BACKBONE, device="cpu")
        runs.append(train(a.model, a.processor, exs, steps=2, batch_size=2, seed=3, log_every=0, step_hook=hook))
    assert runs[1] == pytest.approx(runs[0], rel=1e-5)  # CPU kernels are not bit-deterministic across runs
    assert [c[0] for c in calls] == [1, 2] and calls[0][1] == ["deadline", "device", "elapsed_s", "lr_scale",
                                                               "progress"]


def test_model_policy_matches_the_benchmark_forward_and_updates(agent):
    sys.path.insert(0, AUTORESEARCH)
    import games_eval as ge

    from laya.vlm import set_trainable

    set_trainable(agent.model, "head")
    cfg = R.RLConfig(games=("Maze4", "CartPole"), group=2, episodes_per_phase=2, lr=1e-3, minibatch=4,
                     entropy=0.01, caps={"Maze4": 3, "CartPole": 3})
    rl = R.GameRL(agent, cfg, log=lambda s: None)
    assert rl.modes == {"Maze4": "stack-2", "CartPole": "stack-2"}
    pol = rl.policy
    # the policy's distribution at temperature 1 is games_eval's calibrated one, on the same stack-2 states
    env = R.make_env("CartPole", 7, 3)
    hist = [env.frame()]
    env.step("LEFT")
    hist.append(env.frame())
    st = ge.state_input(hist, "stack-2", "control")
    pol.begin()
    with torch.no_grad():
        _, _, inputs = pol.act([("CartPole", st)])
        p = torch.softmax(pol.scaled_logits(inputs), -1)[0, :2].numpy()
    want = ge.batched_probs(agent, [st], R.question_for("CartPole", "stack-2")["action"], cache=ge.frame_cache(
        "stack-2", "control"))[0]
    assert np.allclose(p, want, atol=1e-4)
    env.close()
    # a phase: the rollout's log-probs are what the update recomputes (ratio 1 before the first step)
    before = {n: v.detach().clone() for n, v in agent.model.named_parameters() if v.requires_grad}
    agent.model.eval()
    pol.begin()
    eps = R.rollout(pol, [("Maze4", 21), ("CartPole", 22)], 2, rl.modes, rl._make)
    x = [i for ep in eps for i in ep.inputs]
    with torch.no_grad():
        z = pol.scaled_logits(x)
    lp = torch.log_softmax(z, -1).gather(1, torch.tensor([a for ep in eps for a in ep.actions])[:, None]).squeeze(1)
    assert torch.allclose(lp, torch.tensor([v for ep in eps for v in ep.logps]), atol=1e-4)
    assert all(len(i[1]) == 2 for i in x)  # stack-2: two images' features per state
    agent.model.train()  # as train() leaves it; the phase plays and updates in eval mode (no dropout), then restores it
    rec = rl.phase()
    assert agent.model.training  # the mode train() left it in comes back
    assert set(rec["games"]) == {"Maze4", "CartPole"} and rec["decisions"] > 0
    assert rec["update_steps"] > 0  # the entropy term keeps every step, tied groups included
    assert any(not torch.equal(v, dict(agent.model.named_parameters())[n]) for n, v in before.items())


def test_lr_vision_gets_its_own_group(monkeypatch):
    """``train(lr_vision=...)`` puts the vision tower in its own optimizer group; without it the groups are as before."""
    import torch
    import laya.vlm_train as vt

    class Enc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_model = torch.nn.Linear(2, 2)
            self.text_model = torch.nn.Linear(2, 2)

    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, self.head = Enc(), torch.nn.Linear(2, 2)

    seen = []

    class Stop(Exception):
        pass

    def fake_adamw(groups, **kw):
        seen.append([(len(g["params"]), g["lr"]) for g in groups])
        raise Stop

    monkeypatch.setattr(vt, "set_trainable", lambda model, mode, n_last=4: 0)
    monkeypatch.setattr(torch.optim, "AdamW", fake_adamw)
    for kw in ({}, {"lr_vision": 1e-6}):
        try:
            vt.train(M(), None, [], lr_head=1e-4, lr_backbone=1e-5, device="cpu", **kw)
        except Stop:
            pass
    assert seen[0] == [(2, 1e-4), (4, 1e-5)]
    assert seen[1] == [(2, 1e-4), (2, 1e-5), (2, 1e-6)]

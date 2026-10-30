"""CPU checks for window coverage, conditional probabilities and the GRPO update."""
import math
import re
from types import SimpleNamespace

import pytest
import torch

# Reuse the repository's tiny, locally generated Qwen3.5 fixture (no Hub weights).
from test_unit import tiny_base

from kev.model import ContextOverflow
from kev.suite import read_json, read_jsonl, write_json
from kev.webqsp_data import Action, Example, Graph, NO_OP, Query
from kev.webqsp_grpo import WindowSelector, grpo, window_decision
from kev.webqsp_policy import CandidateSampler, Decision, Trajectory, beam_search, rollout


def star(degree):
    return Graph({"entities": list(range(degree + 1)), "tuples": [[0, 0, i] for i in range(1, degree + 1)]},
                 [f"e{i}" for i in range(degree + 1)], ["move"], inverse=False)


QUERY = Query("q", "Which entity answers the question?", "e0", "Topic")


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.linspace(-0.2, 0.2, 65))

    def trainable_parameters(self):
        return [self.weights]


class TinyPolicy:
    """A differentiable actor with option-dependent logits, no external weights."""
    def __init__(self, max_options=100):
        self.model, self.records, self.calls = TinyActor(), [], []
        self.max_options, self.row_tokens, self.device = max_options, 100, "cpu"
        self.checkpoint = SimpleNamespace(meta=SimpleNamespace(extra={}))

    def encode(self, record):
        options = record["questions"][0]["options"]
        if len(options) > self.max_options:
            raise ContextOverflow("option budget")
        self.records.append(record)
        indices = [0 if "NO_OP" in x else int(re.search(r"-> e(\d+)", x).group(1)) for x in options]
        return {"ids": [1] * (5 + 2 * len(options)), "indices": indices, "record": record}

    def distribution(self, encoded):
        self.calls.append((encoded, self.model.weights.detach().clone(), torch.is_grad_enabled()))
        return torch.distributions.Categorical(logits=self.model.weights[encoded["indices"]])

    def provenance(self):
        return {"test_actor": "cpu"}

    def save(self, directory, provenance):
        directory.mkdir(parents=True)
        self.checkpoint.meta.extra = {"webqsp": provenance}
        write_json(directory / "test-metadata.json", provenance)


def test_windows_cover_all_edges_with_actual_variable_token_lengths():
    actions = (Action(NO_OP, "start"), *(Action("r", "x" * i) for i in range(1, 10)))

    def encode(xs):
        tokens = 3 + sum(len(a.target) + 2 for a in xs)
        if tokens > 30:
            raise ContextOverflow("token budget")
        return {"ids": [0] * tokens, "actions": xs}

    selector = WindowSelector(4, 2)
    windows = list(selector.windows(actions[1:], actions[0], encode))
    assert tuple(a for xs, _ in windows for a in xs) == actions[1:]
    assert all(1 <= len(xs) <= 4 and enc["actions"] == xs for xs, enc in windows)
    assert all(len(encode((actions[0], *xs))["ids"]) <= 30 for xs, _ in windows)
    assert selector.counts()["token_limited_windows"] > 0


def test_two_nominations_store_conditional_masks_and_all_windows_share_instructions():
    torch.manual_seed(17)
    policy, selector = TinyPolicy(), WindowSelector(4, 2)
    trajectory = rollout(policy, QUERY, star(8), {}, 1, selector, prepare=window_decision)
    assert len(trajectory.decisions) == 5  # Two draws from each of two windows, then a graph move.
    assert [s.stage for s in trajectory.decisions] == ["nomination"] * 4 + ["move"]
    for first, second in (trajectory.decisions[:2], trajectory.decisions[2:4]):
        assert first.encoded is second.encoded
        assert first.excluded == () and second.excluded == (first.action_index,)
        assert first.action_index != second.action_index
        p = policy.distribution(first.encoded).probs.detach()
        assert first.old_log_probability == pytest.approx(math.log(float(p[first.action_index])), abs=1e-6)
        assert second.old_log_probability == pytest.approx(
            math.log(float(p[second.action_index] / (1 - p[first.action_index]))), abs=1e-6)
        assert 0 not in first.encoded["indices"]  # NO_OP never competes for a nomination.
    final = trajectory.decisions[-1]
    assert final.encoded["indices"].count(0) == 1
    assert len(final.encoded["indices"]) == 5
    assert len({r["questions"][0]["instr"] for r in policy.records}) == 1
    assert len({r["state"] for r in policy.records}) == 1
    assert selector.counts()["inspected_edges"] == 8
    assert trajectory.log_probability == pytest.approx(sum(s.old_log_probability for s in trajectory.decisions))


def test_small_windows_are_passed_through_and_noop_remains_legal():
    policy = TinyPolicy()
    path = rollout(policy, QUERY, star(2), {}, 1, WindowSelector(4, 2), prepare=window_decision)
    assert len(path.decisions) == 1 and path.decisions[0].stage == "move"
    assert path.decisions[0].encoded["indices"] == [0, 1, 2]
    forced = rollout(policy, QUERY, star(0), {}, 3, WindowSelector(), prepare=window_decision)
    assert forced.endpoint == "e0" and len(forced.path) == 3 and not forced.decisions
    assert all(a.relation == NO_OP for a in forced.path)


def test_oversized_shortlists_reduce_recursively_and_eval_is_deterministic():
    policy, selector = TinyPolicy(max_options=5), WindowSelector(4, 2)
    rng_before = torch.get_rng_state().clone()
    paths, failures = beam_search(policy, QUERY, star(25), {}, 1, 3, selector, prepare=window_decision)
    assert not failures and paths
    assert torch.equal(rng_before, torch.get_rng_state())  # Top-k nominations never consume action RNG.
    assert selector.counts()["inspected_edges"] == 25
    assert selector.counts()["reduction_rounds"] > 1
    again, _ = beam_search(policy, QUERY, star(25), {}, 1, 3, WindowSelector(4, 2), prepare=window_decision)
    assert paths == again
    assert paths[0].endpoint == "e25"
    assert all(len(enc["indices"]) <= 5 for enc, _, _ in policy.calls)


def test_unrepresentable_edges_fail_without_silent_pruning_and_keep_earlier_choices():
    class LongLastEdge(TinyPolicy):
        def encode(self, record):
            if any("-> e9" in x for x in record["questions"][0]["options"]):
                raise ContextOverflow("one whole edge cannot fit")
            return super().encode(record)

    trajectory = rollout(LongLastEdge(), QUERY, star(9), {}, 1, WindowSelector(4, 2), prepare=window_decision)
    assert trajectory.endpoint is None and trajectory.failure.startswith("context_overflow:")
    assert len(trajectory.decisions) == 4  # Preserve both earlier windows, including their second draws.
    assert all(s.stage == "nomination" for s in trajectory.decisions)


def test_context_that_cannot_reduce_fails_instead_of_looping():
    trajectory = rollout(TinyPolicy(max_options=3), QUERY, star(8), {}, 1,
                         WindowSelector(4, 2), prepare=window_decision)
    assert "cannot fit or shrink" in trajectory.failure
    assert not trajectory.decisions  # Every two-edge window was a deterministic pass-through.


def test_missing_graph_and_overlong_state_produce_no_decisions():
    class LongState(TinyPolicy):
        def encode(self, record):
            raise ContextOverflow("state too long", state_tokens=100, max_state=10)

    for policy, graph in ((TinyPolicy(), None), (LongState(), star(8))):
        trajectory = rollout(policy, QUERY, graph, {}, 1, WindowSelector(4, 2), prepare=window_decision)
        assert trajectory.failure and not trajectory.decisions


def test_grpo_matches_joint_score_gradient_and_replays_every_nomination():
    torch.manual_seed(19)
    policy = TinyPolicy()
    trajectories = [rollout(policy, QUERY, star(8), {}, 1, WindowSelector(4, 2), prepare=window_decision) for _ in range(4)]
    # Choose a fixed answer set that splits this sampled group; the actor only saw QUERY.
    answers = {trajectories[0].endpoint}
    rewards = torch.tensor([float(t.endpoint in answers) for t in trajectories])
    assert 0 < rewards.mean() < 1
    advantages = (rewards - rewards.mean()) / (rewards.std(correction=0) + 1e-8)
    # Independent reference: sum the exact log joint of all internal draws and moves.
    reference = policy.model.weights.new_zeros(())
    for trajectory, advantage in zip(trajectories, advantages):
        for step in trajectory.decisions:
            logits = policy.model.weights[step.encoded["indices"]]
            allowed = [i for i in range(len(logits)) if i not in step.excluded]
            logp = logits[step.action_index] - torch.logsumexp(logits[allowed], dim=0)
            reference = reference - advantage * logp / len(trajectories)
    expected_grad, = torch.autograd.grad(reference, policy.model.weights)
    expected_grad *= min(1.0, 1.0 / (float(expected_grad.norm()) + 1e-6))
    before = policy.model.weights.detach().clone()
    policy.calls.clear()
    optimizer = torch.optim.SGD(policy.model.trainable_parameters(), lr=0.1)
    stats = grpo(policy, trajectories, answers, optimizer)
    assert torch.allclose(policy.model.weights, before - 0.1 * expected_grad, atol=1e-7)
    assert stats["max_initial_logp_error"] < 1e-6
    assert stats["nomination_decisions"] == 16 and stats["move_decisions"] == 4
    assert len(policy.calls) == 12  # Two window forwards and one final forward per trajectory.
    assert all(torch.equal(weights, before) and grad for _, weights, grad in policy.calls)
    assert stats["optimizer_steps"] == 1 and stats["clip_fraction"] == 0


@pytest.mark.parametrize("reward", [0, 1])
def test_zero_variance_groups_skip_even_entropy_and_adam_moments(reward):
    policy = TinyPolicy()
    trajectories = [Trajectory("same", decisions=(Decision({"indices": [0, 1]}, 0,
                                                          old_log_probability=-math.log(2)),))] * 4
    optimizer = torch.optim.AdamW(policy.model.trainable_parameters(), lr=0.1)
    before = policy.model.weights.detach().clone()
    stats = grpo(policy, trajectories, {"same"} if reward else set(), optimizer, entropy_weight=1.0)
    assert stats["zero_variance_group"] and stats["advantages"] == [0.0] * 4
    assert stats["optimizer_steps"] == 0 and not optimizer.state and not policy.calls
    assert torch.equal(before, policy.model.weights)


@pytest.mark.parametrize("success", [True, False])
def test_clipping_stops_improvement_outside_both_sides_of_the_ratio_bound(success):
    parameter = torch.nn.Parameter(torch.tensor(math.log(4) if success else -math.log(4)))
    policy = SimpleNamespace(model=SimpleNamespace(trainable_parameters=lambda: [parameter]),
                             distribution=lambda encoded: torch.distributions.Categorical(logits=torch.stack([parameter * 0, parameter])))
    # Both recorded behavior probabilities were 0.5; current p(chosen) is 0.8 or 0.2.
    choice = Decision({}, 1, old_log_probability=math.log(0.5))
    trajectories = [Trajectory("answer" if success else "wrong", decisions=(choice,)),
                    Trajectory("wrong" if success else "answer")]
    before = parameter.detach().clone()
    stats = grpo(policy, trajectories, {"answer"}, torch.optim.SGD([parameter], lr=1.0))
    assert torch.equal(parameter, before) and stats["clip_fraction"] == 1.0
    assert stats["loss"] == pytest.approx(-0.6 if success else 0.4)


def test_multiple_updates_keep_behavior_probabilities_and_respect_no_decision_failures():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    seen = []

    def distribution(encoded):
        seen.append(float(parameter.detach()))
        return torch.distributions.Categorical(logits=torch.stack([parameter * 0, parameter]))

    policy = SimpleNamespace(model=SimpleNamespace(trainable_parameters=lambda: [parameter]), distribution=distribution)
    choice = Decision({}, 1, old_log_probability=-math.log(2))
    trajectories = [Trajectory("answer", decisions=(choice,)), Trajectory(None, failure="missing_graph")]
    stats = grpo(policy, trajectories, {"answer"}, torch.optim.SGD([parameter], lr=0.1), iterations=2)
    assert seen[0] == 0 and seen[1] > 0 and len(seen) == 2
    assert choice.old_log_probability == -math.log(2)
    assert stats["optimizer_steps"] == 2 and stats["failed_rollouts"] == 1
    assert stats["reward"] == 0.5 and stats["reward_std"] == 0.5
    assert stats["approx_old_policy_kl"] > 0


def test_rejects_missing_behavior_logps_and_a_masked_selection():
    policy = TinyPolicy()
    optimizer = torch.optim.SGD(policy.model.trainable_parameters(), lr=0.1)
    with pytest.raises(ValueError, match="at least two"):
        grpo(policy, [Trajectory("answer")], {"answer"}, optimizer)
    for choice, message in ((Decision({}, 0), "old log probabilities"),
                            (Decision({}, 0, old_log_probability=0, excluded=(0,)), "masked out")):
        with pytest.raises(ValueError, match=message):
            grpo(policy, [Trajectory("answer", decisions=(choice,)), Trajectory("wrong")], {"answer"}, optimizer)


def test_nonfinite_ratios_fail_before_changing_parameters():
    policy = TinyPolicy()
    before = policy.model.weights.detach().clone()
    optimizer = torch.optim.AdamW(policy.model.trainable_parameters(), lr=0.1)
    choice = Decision({"indices": [0, 1]}, 0, old_log_probability=-1000.0)
    with pytest.raises(FloatingPointError, match="probability ratio"):
        grpo(policy, [Trajectory("answer", decisions=(choice,)), Trajectory("wrong")], {"answer"}, optimizer)
    assert torch.equal(before, policy.model.weights) and not optimizer.state


def cli_args(tmp_path, command="train", extra=()):
    from kev.webqsp import parse_args
    return parse_args([command, "--questions", "questions.json", "--nsm-dir", "nsm", "--names", "names.json",
                       "--out", str(tmp_path / "run"), "--device", "cpu", "--precision", "fp32", *extra])


def test_cli_defaults_and_checkpoint_selection_settings(tmp_path):
    from kev.webqsp import resolve_candidates
    baseline = cli_args(tmp_path)
    assert (baseline.algorithm, baseline.candidate_mode, baseline.baseline, baseline.entropy) == ("reinforce", "sampled", "value", 0.01)
    args = cli_args(tmp_path, extra=("--algorithm", "grpo"))
    assert (args.candidate_mode, args.candidate_edges, args.nominees_per_window, args.rollouts) == ("windowed", 128, 2, 4)
    assert args.baseline == "group" and args.entropy == 0 and args.grpo_iterations == 1
    policy = TinyPolicy()
    policy.checkpoint.meta.extra = {"webqsp": {"args": {**vars(args), "candidate_edges": 16}}}
    evaluation = cli_args(tmp_path, "evaluate")
    resolve_candidates(evaluation, policy)
    assert evaluation.candidate_mode == "windowed" and evaluation.candidate_edges == 16
    override = cli_args(tmp_path, "evaluate", ("--candidate-mode", "sampled", "--candidate-edges", "0"))
    resolve_candidates(override, policy)
    assert override.candidate_mode == "sampled" and override.candidate_edges == 0


@pytest.mark.parametrize("extra", [("--rollouts", "1"), ("--baseline", "value"), ("--candidate-edges", "0"),
                                  ("--candidate-edges", "2"), ("--clip-epsilon", "nan"),
                                  ("--grpo-iterations", "0"), ("--max-groups", "-1")])
def test_cli_rejects_invalid_grpo_settings_before_loading_weights(tmp_path, extra):
    with pytest.raises(SystemExit):
        cli_args(tmp_path, extra=("--algorithm", "grpo", *extra))


@pytest.mark.parametrize("all_fail", [False, True])
def test_training_evaluation_and_metadata_end_to_end_on_cpu(tmp_path, monkeypatch, all_fail):
    from kev import webqsp
    monkeypatch.setattr(webqsp, "kernel_environment", lambda *args: {"test": "cpu"})
    args = cli_args(tmp_path, extra=("--algorithm", "grpo", "--candidate-edges", "4", "--hops", "1",
                                     "--max-groups", "1", "--grpo-iterations", "3", "--max-updates", "2"))
    example = Example(QUERY, (("absent",) if all_fail else tuple(f"e{i}" for i in range(1, 5)),))
    dataset = SimpleNamespace(examples=[example], graph=lambda ex: star(8), names={}, split="dev", inverse=False, provenance={})
    policy = TinyPolicy()
    before = policy.model.weights.detach().clone()
    out = tmp_path / "run"
    out.mkdir()
    torch.manual_seed(19)
    webqsp.train(policy, dataset, dataset, args, out, {"args": vars(args)})
    result = read_json(out / "result.json")
    row, = read_jsonl(out / "train.jsonl")
    saved = read_json(out / "epoch-1/checkpoint/test-metadata.json")
    report = read_json(out / "epoch-1/dev/report.json")
    assert result["groups"] == 1 and saved["objective"] == "terminal-membership GRPO"
    assert saved["baseline_mode"] == "group" and saved["candidate_selection"]["method"] == "exhaustive_windows"
    assert report["candidate_sampling"]["candidate_edges"] == 4
    assert report["path_probability"] == "final moves conditional on deterministic nominations"
    assert row["nomination_decisions"] == 16 and row["move_decisions"] == 4
    assert row["sampling"]["inspected_edges"] == 32
    assert result["updates"] == (0 if all_fail else 2)
    assert torch.equal(before, policy.model.weights) == all_fail
    assert result["zero_variance_groups"] == int(all_fail)
    assert not (out / "epoch-2").exists()


def test_grpo_also_accepts_the_original_sampler_for_a_control_arm():
    torch.manual_seed(7)
    policy = TinyPolicy()
    trajectories = [rollout(policy, QUERY, star(8), {}, 1, CandidateSampler(4, seed=i)) for i in range(4)]
    stats = grpo(policy, trajectories, {trajectories[0].endpoint}, torch.optim.SGD(policy.model.trainable_parameters(), lr=0.1))
    assert stats["nomination_decisions"] == 0 and stats["move_decisions"] == 4
    assert stats["max_initial_logp_error"] < 1e-6


def test_real_qwen_policy_updates_only_lora_and_pointer_and_round_trips(tiny_base, tmp_path):
    import shutil
    from kev.checkpoint import Meta, read_meta, write_meta
    from kev.model import DecisionModel, load_tokenizer
    from kev.webqsp_policy import Policy

    base = tmp_path / "Qwen3.5-tiny"
    shutil.copytree(tiny_base / "base", base)
    tok = load_tokenizer(base)
    model = DecisionModel(str(base), tok, "cpu", lora=2, head_dim=8)
    initial = tmp_path / "initial"
    model.lm.save_pretrained(initial)
    tok.save_pretrained(initial)
    write_meta(initial, Meta(base=str(base), lora=2, head_dim=8, head=model.head.state_dict()))
    policy = Policy(initial, "cpu", "fp32", row_tokens=512, temperature=1.0, training=True)
    before = {name: p.detach().clone() for name, p in policy.model.named_parameters()}
    torch.manual_seed(19)
    trajectories = [rollout(policy, QUERY, star(8), {}, 1, WindowSelector(4, 2), prepare=window_decision) for _ in range(4)]
    assert len({t.endpoint for t in trajectories}) > 1
    stats = grpo(policy, trajectories, {trajectories[0].endpoint},
                 torch.optim.AdamW(policy.model.trainable_parameters(), lr=1e-5, weight_decay=0))
    assert stats["max_initial_logp_error"] < 1e-6 and stats["optimizer_steps"] == 1
    changed = [name for name, p in policy.model.named_parameters() if not torch.equal(before[name], p)]
    assert any(name.startswith("head.") for name in changed) and any(".lora_" in name for name in changed)
    assert all(name.startswith("head.") or ".lora_" in name for name in changed)
    assert policy.value_head is None
    saved = tmp_path / "trained"
    policy.save(saved, {"args": {"algorithm": "grpo", "candidate_mode": "windowed", "candidate_edges": 4,
                                "nominees_per_window": 2}, "objective": "terminal-membership GRPO"})
    metadata = read_meta(saved)
    assert "webqsp_value_head" not in metadata.extra and not metadata.holdout
    assert metadata.extra["webqsp"]["args"]["algorithm"] == "grpo"
    restored = Policy(saved, "cpu", "fp32", row_tokens=512)
    policy.model.eval()
    with torch.no_grad():
        for step in trajectories[0].decisions:
            assert torch.equal(policy.distribution(step.encoded).logits, restored.distribution(step.encoded).logits)

"""Weight-free checks for the WebQSP protocol. No checkpoint downloads."""
import json
import math
from types import SimpleNamespace

import pytest
import torch

from kev.checkpoint import Meta, read_meta
from kev.model import ContextOverflow, DecisionModel, PointerHead
from kev.suite import write_json, write_jsonl
from kev.webqsp_data import Action, Dataset, Example, Graph, NO_OP, Query, observation, score
from kev.webqsp_policy import CandidateSampler, Decision, Policy, Trajectory, ValueHead, beam_search, reinforce, rollout


def make_graph(inverse=True):
    return Graph({"entities": [0, 1, 2], "tuples": [[0, 0, 1], [1, 1, 2], [0, 0, 1]]},
                 ["m.topic", "m.cvt", "m.answer"], ["first", "second"], inverse)


def test_edges_keep_mediator_ids_inverse_and_noop():
    graph = make_graph()
    assert graph.actions("m.topic") == (Action(NO_OP, "m.topic"), Action("first", "m.cvt"))
    assert Action("first", "m.topic", reverse=True) in graph.actions("m.cvt")
    assert "m.answer" not in graph.reachable("m.topic", 1)
    assert graph.reachable("m.topic", 2) == {"m.topic", "m.cvt", "m.answer"}
    assert make_graph(inverse=False).actions("m.answer") == (Action(NO_OP, "m.answer"),)
    with pytest.raises(ValueError, match="vocabulary index"):
        Graph({"entities": [0], "tuples": [[-1, 0, 0]]}, ["m.a"], ["r"])


def test_observation_cannot_see_answer_or_parse_annotations():
    query = Query("q", "Who is related to Topic?", "m.topic", "Topic")
    a = Example(query, (("m.secret-one",),))
    b = Example(query, (("m.secret-two",),))
    actions = (Action(NO_OP, "m.topic"), Action("relation", "m.other"), Action("relation", "m.third"))
    names = {"m.other": "Same name", "m.third": "Same name"}
    record = observation(a.query, "m.topic", (), actions, names, 3)
    assert record == observation(b.query, "m.topic", (), actions, names, 3)
    text = json.dumps(record)
    assert "secret" not in text
    assert "Same name [m.other]" in text and "Same name [m.third]" in text
    assert record["questions"][0]["label"] == 0


def test_best_parse_metrics_do_not_union_answer_denominators():
    query = Query("q", "Question", "m.topic", "Topic")
    example = Example(query, (("m.a", "m.b"), ("m.c",)))
    assert score(example, ["m.c"]) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert score(example, ["m.a", "m.a"])["f1"] == pytest.approx(2 / 3)
    assert score(example, []) == {"precision": 1.0, "recall": 0.0, "f1": 0.0}
    assert score(Example(query, ((),)), []) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert score(Example(query, ((),)), ["m.a"]) == {"precision": 0.0, "recall": 1.0, "f1": 0.0}


def microsoft_question(qid, good=True):
    return {"QuestionId": qid, "RawQuestion": "Question", "Parses": [{
        "TopicEntityMid": "m.topic", "TopicEntityName": "Topic",
        "AnnotatorComment": {"QuestionQuality": "Good" if good else "Bad", "ParseQuality": "Complete"},
        "Answers": [{"AnswerArgument": "m.answer", "EntityName": "GOLD LABEL MUST NOT BECOME A NAME"}],
        "InferentialChain": ["GOLD CHAIN"], "Sparql": "GOLD QUERY",
    }]}


@pytest.fixture
def data_files(tmp_path):
    nsm = tmp_path / "nsm"
    nsm.mkdir()
    (nsm / "entities.txt").write_text("m.topic\nm.answer\n", encoding="utf-8")
    (nsm / "relations.txt").write_text("relation\n", encoding="utf-8")
    for split, qid in (("train", "WebQTrn-0"), ("dev", "WebQTrn-1"), ("test", "WebQTest-0")):
        write_jsonl(nsm / f"{split}_simple.json", [{"id": qid, "answers": [{"kb_id": "m.poison"}],
                                                   "subgraph": {"entities": [0, 1], "tuples": [[0, 0, 1]]}}])
    train_file, test_file, names = (tmp_path / name for name in ("train.json", "test.json", "names.json"))
    write_json(train_file, {"Questions": [microsoft_question("WebQTrn-0"), microsoft_question("WebQTrn-1")]})
    write_json(test_file, {"Questions": [microsoft_question("WebQTest-0"), microsoft_question("WebQTest-1"),
                                         microsoft_question("WebQTest-2", good=False)]})
    write_json(names, {"m.topic": "Topic", "m.answer": "Independent name"})
    return train_file, test_file, nsm, names


def test_split_isolation_original_labels_and_missing_test_graphs(data_files):
    train_file, test_file, nsm, names = data_files
    train = Dataset(train_file, nsm, names, "train")
    dev = Dataset(train_file, nsm, names, "dev")
    test = Dataset(test_file, nsm, names, "test")
    assert [e.query.id for e in train.examples] == ["WebQTrn-0"]
    assert [e.query.id for e in dev.examples] == ["WebQTrn-1"]
    assert len(test.examples) == 2
    assert test.excluded == ["WebQTest-2"]
    assert test.graph(test.examples[1]) is None
    assert train.examples[0].answers == {"m.answer"}
    assert "GOLD LABEL" not in json.dumps(train.names)
    with pytest.raises(ValueError, match="original Microsoft train"):
        Dataset(test_file, nsm, names, "train")
    write_jsonl(nsm / "dev_simple.json", [{"id": "WebQTrn-0", "subgraph": {"entities": [], "tuples": []}}])
    with pytest.raises(ValueError, match="overlap"):
        Dataset(train_file, nsm, names, "train")


class FixedPolicy:
    def encode(self, record):
        return record

    def distribution(self, record):
        # Always move on a forward edge when possible; NO_OP remains legal.
        logits = [0.0 if "NO_OP" in option else 10.0 for option in record["questions"][0]["options"]]
        return torch.distributions.Categorical(logits=torch.tensor(logits))


def test_candidate_sampling_keeps_noop_and_reuses_a_seeded_stream():
    actions = (Action(NO_OP, "start"), *(Action(f"r{i}", f"e{i}") for i in range(30)))
    encode = lambda selected: {"ids": list(range(len(selected))), "selected": selected}
    first, replay = CandidateSampler(4, 17), CandidateSampler(4, 17)
    draws = [first.select(actions, encode)[0] for _ in range(3)]
    assert draws == [replay.select(actions, encode)[0] for _ in range(3)]
    assert draws[0] != draws[1]
    assert all(len(draw) == 5 and draw[0] == actions[0] and len(set(draw)) == 5 for draw in draws)
    assert first.counts()["legal_actions"] == 3 * len(actions)
    assert first.counts()["offered_actions"] == 15
    assert first.counts()["capped_decisions"] == 3


def test_candidate_budget_keeps_whole_options_and_strict_mode_propagates_overflow():
    actions = (Action(NO_OP, "start"), *(Action(f"r{i}", f"e{i}") for i in range(12)))

    def encode(selected):
        tokens = 7 + 3 * len(selected)
        if tokens > 19:
            raise ContextOverflow("branch too long")
        return {"ids": list(range(tokens)), "selected": selected}

    sampler = CandidateSampler(8, 42)
    selected, encoded = sampler.select(actions, encode)
    assert len(selected) == 4 and selected[0] == actions[0]
    assert encoded["selected"] == selected and len(encoded["ids"]) == 19
    assert sampler.counts()["token_limited_decisions"] == 1
    with pytest.raises(ContextOverflow):
        CandidateSampler(0).select(actions, encode)


def test_candidate_sampler_does_not_truncate_an_overlong_state_or_instruction():
    actions = (Action(NO_OP, "start"), Action("edge", "end"))
    for error in (ContextOverflow("state", state_tokens=20, max_state=10), ContextOverflow("instruction")):
        def encode(selected):
            raise error
        with pytest.raises(ContextOverflow) as raised:
            CandidateSampler(4).select(actions, encode)
        assert raised.value is error


def test_noop_only_after_budgeting_is_forced_and_small_lists_keep_their_order():
    actions = (Action(NO_OP, "start"), Action("first", "a"), Action("second", "b"))

    def only_noop_fits(selected):
        if len(selected) > 1:
            raise ContextOverflow("branch too long")
        return {"ids": [1, 2, 3]}

    sampler = CandidateSampler(4)
    assert sampler.select(actions, only_noop_fits) == ((actions[0],), None)
    assert sampler.counts()["noop_only_decisions"] == 1
    assert sampler.counts()["encoded_tokens"] == 3
    selected, encoded = sampler.select(actions, lambda xs: {"ids": [0] * len(xs)})
    assert selected == actions and len(encoded["ids"]) == len(actions)


def test_rollout_and_beam_share_candidate_construction_and_store_the_offered_input():
    class RecordingPolicy(FixedPolicy):
        def __init__(self):
            self.records = []

        def encode(self, record):
            self.records.append(record)
            return {**record, "ids": list(range(len(record["questions"][0]["options"])))}

    graph = Graph({"entities": list(range(10)), "tuples": [[0, 0, i] for i in range(1, 10)]},
                  [f"e{i}" for i in range(10)], ["move"], inverse=False)
    query = Query("q", "Question", "e0", "Topic")
    train_policy, eval_policy = RecordingPolicy(), RecordingPolicy()
    trajectory = rollout(train_policy, query, graph, {}, 1, CandidateSampler(3, 12))
    beam_search(eval_policy, query, graph, {}, 1, 2, CandidateSampler(3, 12))
    assert train_policy.records == eval_policy.records
    assert len(train_policy.records[0]["questions"][0]["options"]) == 4
    assert trajectory.decisions[0].encoded["questions"] == train_policy.records[0]["questions"]
    assert 0 <= trajectory.decisions[0].action_index < 4


def test_greedy_consumes_raw_edges_and_noop_uses_remaining_hop():
    query = Query("q", "Question", "m.topic", "Topic")
    paths, failures = beam_search(FixedPolicy(), query, make_graph(inverse=False), {}, hops=3, width=1)
    assert not failures
    assert paths[0].endpoint == "m.answer"
    assert [a.relation for a in paths[0].path] == ["first", "second", NO_OP]


def test_context_overflow_is_failure_even_when_start_is_an_answer():
    class Overlong(FixedPolicy):
        def encode(self, record):
            raise ContextOverflow("too many options")
    query = Query("q", "Question", "m.topic", "Topic")
    trajectory = rollout(Overlong(), query, make_graph(), {}, hops=3)
    assert trajectory.endpoint is None
    assert trajectory.failure.startswith("context_overflow:")
    paths, failures = beam_search(Overlong(), query, make_graph(), {}, hops=3, width=8)
    assert not paths and len(failures) == 1


def test_reinforce_accumulates_before_one_step_and_uses_terminal_reward():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    seen_parameters = []

    class TinyPolicy:
        model = SimpleNamespace(trainable_parameters=lambda: [parameter])

        def distribution(self, encoded):
            seen_parameters.append(float(parameter.detach()))
            return torch.distributions.Categorical(logits=torch.stack([parameter * 0, parameter]))

    class CountingSGD(torch.optim.SGD):
        steps = 0

        def step(self, closure=None):
            self.steps += 1
            return super().step(closure)

    optimizer = CountingSGD([parameter], lr=0.1)
    trajectories = [Trajectory("answer", decisions=(Decision({}, 1), Decision({}, 1))),
                    Trajectory("wrong", decisions=(Decision({}, 0), Decision({}, 0)))]
    stats = reinforce(TinyPolicy(), trajectories, {"answer"}, optimizer, baseline=0.0, entropy_weight=0.0)
    assert seen_parameters == [0.0] * 4  # No update while scoring this group.
    assert optimizer.steps == 1
    assert parameter.item() == pytest.approx(0.05)  # -(1/2) * 2 * (1 - p), with p=1/2.
    assert stats["reward"] == 0.5
    assert stats["entropy_per_decision"] == pytest.approx(math.log(2))


@pytest.mark.parametrize("mode", ["packed", "rows", "shared_prefix"])
def test_decide_features_share_the_logit_pass_and_preserve_default_scoring(monkeypatch, mode):
    model = DecisionModel.__new__(DecisionModel)
    torch.nn.Module.__init__(model)
    model.device, model.pad_id, model.hybrid = "cpu", 0, mode != "packed"
    model.head = PointerHead(2, dp=2)
    encs = [{"ids": list(range(offset, offset + 8)), "seg": [0, 0, 1, 1, 1, 2, 2, 2],
             "pos": [0, 1, 2, 3, 4, 2, 3, 4], "decide_idx": [4, 7], "opt_idx": [[2, 3], [5, 6]]}
            for offset in (0, 10)]
    calls = []

    def hidden(ids):
        ids = torch.tensor(ids, dtype=torch.float32)
        return torch.stack([ids, ids + 1], dim=-1)

    def packed(batch):
        calls.append("packed")
        return torch.stack([hidden(e["ids"]) for e in batch])

    def rows(batch):
        calls.append("rows")
        return [hidden(ids) for ids, _ in batch]

    def shared(lm, splits, pad_id, device):
        calls.append("shared_prefix")
        return [[hidden(r["ids"]) for r in branches] for _, _, branches in splits]

    monkeypatch.setattr(model, "hidden_batch", packed)
    monkeypatch.setattr(model, "_rows_hidden", rows)
    if mode == "shared_prefix":
        model.lm = torch.nn.Identity()
        monkeypatch.setattr("kev.shared_prefix.branch_hidden", shared)
    plain = model.forward_batch(encs, shared_prefix=mode == "shared_prefix")
    paired = model.forward_batch(encs, shared_prefix=mode == "shared_prefix", return_features=True)
    assert calls == [mode, mode]  # One backbone call per invocation, not a second pass for features.
    for e, logits, pairs in zip(encs, plain, paired):
        expected = hidden(e["ids"])
        for index, (z, features) in enumerate(pairs):
            assert torch.equal(z, logits[index])
            assert torch.equal(features, expected[e["decide_idx"][index]])


def test_value_head_preserves_rng_and_detaches_actor_features():
    rng = torch.get_rng_state().clone()
    head = ValueHead(3)
    assert torch.equal(torch.get_rng_state(), rng)
    features = torch.ones(2, 3, requires_grad=True)
    assert torch.equal(head(features), torch.full((2,), 0.5))
    head(features).sum().backward()
    assert features.grad is None
    assert head.projection.weight.grad.abs().sum() > 0


def test_policy_value_prediction_uses_one_pass_one_temperature_and_copies_only_features():
    source = torch.arange(12, dtype=torch.float32).reshape(6, 2).requires_grad_()
    calls = []

    class TinyModel(torch.nn.Module):
        def forward(self, encoded, return_features=False):
            calls.append(return_features)
            logits = torch.tensor([0.0, 2.0])
            return [(logits, source[3])] if return_features else [logits]

    policy = Policy.__new__(Policy)
    policy.model, policy.value_head = TinyModel(), ValueHead(2)
    policy.precision, policy.temperature = "fp32", 2.0
    distribution, baseline, features = policy.distribution_and_value({})
    assert calls == [True] and baseline == 0.5
    assert torch.allclose(distribution.probs, torch.tensor([0.0, 1.0]).softmax(-1))
    assert not features.requires_grad and features.device.type == "cpu"
    assert features.untyped_storage().nbytes() == features.numel() * features.element_size()
    with torch.no_grad():
        source.add_(100)
    assert torch.equal(features, torch.tensor([6.0, 7.0]))


def test_rollout_records_pre_action_values_and_beam_does_not_request_them():
    class RecordingPolicy(FixedPolicy):
        value_calls = 0

        def distribution_and_value(self, encoded):
            self.value_calls += 1
            return self.distribution(encoded), 0.25, torch.tensor([1.0, 2.0])

    policy = RecordingPolicy()
    query = Query("q", "Question", "m.topic", "Topic")
    trajectory = rollout(policy, query, make_graph(inverse=False), {}, hops=1, with_value=True)
    step, = trajectory.decisions
    assert step.baseline == 0.25 and torch.equal(step.value_features, torch.tensor([1.0, 2.0]))
    beam_search(policy, query, make_graph(inverse=False), {}, hops=1, width=1)
    assert policy.value_calls == 1


def test_value_update_uses_saved_baselines_and_does_not_backpropagate_into_actor():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    features = [torch.tensor([1.0, 0.0], requires_grad=True), torch.tensor([0.0, 1.0], requires_grad=True)]
    head, events = ValueHead(2), []

    class TinyPolicy:
        model = SimpleNamespace(trainable_parameters=lambda: [parameter])
        value_head = head

        def distribution(self, encoded):
            assert parameter.item() == 0.0  # Entire actor group precedes both optimizer steps.
            assert head.projection.weight.count_nonzero() == 0
            return torch.distributions.Categorical(logits=torch.stack([parameter * 0, parameter]))

    class RecordingSGD(torch.optim.SGD):
        def step(self, closure=None):
            events.append("actor" if self.param_groups[0]["params"][0] is parameter else "critic")
            return super().step(closure)

    actor = RecordingSGD([parameter], lr=0.1)
    critic = RecordingSGD(head.parameters(), lr=1.0)
    # Deliberately differ from the head's CURRENT 0.5 prediction to detect a
    # mistaken recomputation of the baseline when applying the actor gradient.
    trajectories = [Trajectory("answer", decisions=(Decision({}, 1, 0.25, features[0]),) * 2),
                    Trajectory("wrong", decisions=(Decision({}, 0, 0.75, features[1]),) * 2)]
    stats = reinforce(TinyPolicy(), trajectories, {"answer"}, actor, baseline=0.0,
                      entropy_weight=0.0, value_optimizer=critic)
    assert events == ["actor", "critic"]
    assert parameter.item() == pytest.approx(0.075)
    assert parameter.grad.item() == pytest.approx(-0.75)
    assert all(f.grad is None for f in features)
    assert torch.allclose(head.projection.weight, torch.tensor([[0.125, -0.125]]))
    assert stats["value_loss"] == pytest.approx(0.25)
    assert stats["value_mse_before_update"] == pytest.approx(0.75 ** 2)
    assert stats["baseline_mode"] == "value" and stats["value_optimizer_step"]


def test_value_checkpoint_round_trip_and_ema_save_drops_stale_critic(tmp_path, monkeypatch):
    from kev import webqsp_policy

    class LocalCheckpoint:
        def __init__(self, directory):
            self.meta = read_meta(directory)

    monkeypatch.setattr(webqsp_policy, "Checkpoint", LocalCheckpoint)
    policy = Policy.__new__(Policy)
    policy.temperature, policy.value_head = 2.0, ValueHead(2)
    with torch.no_grad():
        policy.value_head.projection.weight.copy_(torch.tensor([[0.2, -0.3]]))
        policy.value_head.projection.bias.fill_(0.1)
    policy.checkpoint = SimpleNamespace(meta=Meta(base="Qwen/Qwen3.5-4B", holdout=["old-calibration"]))
    policy.model = SimpleNamespace(head=torch.nn.Linear(2, 2), lm=SimpleNamespace(save_pretrained=lambda path: None))
    policy.tok = SimpleNamespace(save_pretrained=lambda path: None)
    directory = tmp_path / "value"
    policy.save(directory, {"baseline_mode": "value"})
    metadata = read_meta(directory)
    restored = ValueHead(2)
    restored.load_checkpoint_state(metadata.extra["webqsp_value_head"])
    features = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    assert torch.equal(restored(features), policy.value_head(features))
    assert metadata.temperature == 2.0 and metadata.holdout == []
    with pytest.raises(ValueError, match="incompatible"):
        ValueHead(3).load_checkpoint_state(metadata.extra["webqsp_value_head"])
    policy.value_head = None
    policy.save(tmp_path / "ema", {"baseline_mode": "ema"})
    assert "webqsp_value_head" not in read_meta(tmp_path / "ema").extra


def test_value_mode_handles_no_decisions_and_rejects_missing_rollout_values():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    head = ValueHead(2)
    policy = SimpleNamespace(model=SimpleNamespace(trainable_parameters=lambda: [parameter]), value_head=head)
    actor, critic = torch.optim.SGD([parameter], lr=0.1), torch.optim.SGD(head.parameters(), lr=0.1)
    stats = reinforce(policy, [Trajectory(None, failure="missing_graph")], {"answer"}, actor, 0.0, 0.0, critic)
    assert not stats["optimizer_step"] and not stats["value_optimizer_step"]
    assert stats["value_loss"] is None and stats["baseline_mean"] is None
    with pytest.raises(ValueError, match="saved during rollout"):
        reinforce(policy, [Trajectory("answer", decisions=(Decision({}, 1),))], {"answer"}, actor, 0.0, 0.0, critic)

"""Weight-free checks for the WebQSP protocol. No checkpoint downloads."""
import json
import math
from types import SimpleNamespace

import pytest
import torch

from kev.model import ContextOverflow
from kev.suite import write_json, write_jsonl
from kev.webqsp_data import Action, Dataset, Example, Graph, NO_OP, Query, observation, score
from kev.webqsp_policy import Decision, Trajectory, beam_search, reinforce, rollout


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
